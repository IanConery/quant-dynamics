"""Target generation, Triple Barrier Method, meta-label filtering, and cost adjustments."""

from collections import Counter
from typing import Dict, List
import lightgbm as lgb
import numpy as np
import pandas as pd

from config.settings import (
    CLASSIFICATION_THRESHOLD,
    META_LABEL,
    PREDICTION_WINDOWS,
    TERNARY_CLASSIFICATION,
    TRIPLE_BARRIER,
    VOL_NORMALIZED_TARGETS,
)
from utils.logger import setup_logger

logger = setup_logger(__name__)


def _interval_to_hours(interval: str) -> float:
    """Convert an interval string (e.g. '15m', '1h', '4h', '1d') to hours."""
    interval = interval.strip()
    mapping = {
        "1m": 1 / 60, "5m": 5 / 60, "15m": 0.25, "30m": 0.5,
        "1h": 1.0, "2h": 2.0, "4h": 4.0, "1d": 24.0, "1w": 168.0,
    }
    if interval in mapping:
        return mapping[interval]
    parts = interval.replace("d", "").replace("h", "").replace("m", "").strip()
    if parts.isdigit():
        num = int(parts)
        if "d" in interval:
            return num * 24.0
        elif "h" in interval:
            return float(num)
        elif "m" in interval:
            return num / 60.0
    return 1.0


def create_pbr_targets(
    df: pd.DataFrame,
    interval: str,
    windows: Dict[str, List[int]] = None,
    tp_pct: float = 0.06,
    sl_pct: float = 0.03,
) -> pd.DataFrame:
    """PBR-inspired cumulative position-based return targets."""
    if windows is None:
        windows = PREDICTION_WINDOWS
    df = df.copy()
    interval_hours = _interval_to_hours(interval)
    target_windows_h = windows.get(interval, [1, 4, 24])
    close = df["close"].values

    for w in target_windows_h:
        max_steps = max(1, round(w / interval_hours))
        col_pbr = f"pbr_target_{w}h"
        targets = np.zeros(len(df), dtype=np.float64)

        for i in range(len(df) - max_steps):
            entry_price = close[i]
            cumulative = 0.0
            for j in range(1, max_steps + 1):
                if i + j >= len(close):
                    break
                ret = (close[i + j] - entry_price) / entry_price
                if ret >= tp_pct:
                    cumulative = ret
                    break
                elif ret <= -sl_pct:
                    cumulative = ret
                    break
                elif j == max_steps:
                    cumulative = ret
            targets[i] = cumulative

        df[col_pbr] = targets
        logger.info(f"PBR targets created for {w}h window (TP={tp_pct:.1%}, SL={sl_pct:.1%})")

    return df


def adjust_targets_for_cost(
    y_reg: np.ndarray,
    round_trip_cost: float = 0.0025,
) -> np.ndarray:
    """Commission-adjusted training targets."""
    abs_ret = np.abs(y_reg)
    adjusted = np.where(
        abs_ret > round_trip_cost,
        y_reg,
        np.zeros_like(y_reg),
    )
    n_zeroed = int(np.sum(abs_ret <= round_trip_cost))
    logger.info(f"Cost-adjusted targets: {n_zeroed}/{len(y_reg)} samples zeroed (|r| <= {round_trip_cost:.4f})")
    return adjusted


def create_targets_triple_barrier(
    df: pd.DataFrame,
    interval: str,
    windows: Dict[str, List[int]] = None,
    config: Dict = None,
) -> pd.DataFrame:
    """Triple Barrier Method targets (Lopez de Prado)."""
    if windows is None:
        windows = PREDICTION_WINDOWS
    if config is None:
        config = TRIPLE_BARRIER

    df = df.copy()
    interval_hours = _interval_to_hours(interval)
    target_windows_h = windows.get(interval, [1, 4, 24])
    close = df["close"].values

    upper = config.get("upper_barrier", 0.06)
    lower = config.get("lower_barrier", -0.03)
    vol_adjusted = config.get("volatility_adjusted", False)

    if vol_adjusted:
        ret = df["close"].pct_change()
        rolling_vol = ret.rolling(window=20, min_periods=20).std().values
        rolling_vol = np.nan_to_num(rolling_vol, nan=rolling_vol[~np.isnan(rolling_vol)].mean())
    else:
        rolling_vol = None

    for w in target_windows_h:
        max_steps = max(1, round(w / interval_hours))
        tb_col = f"tb_target_{w}h"
        tb_reg_col = f"tb_reg_target_{w}h"

        labels = np.full(len(df), fill_value=1, dtype=np.int8)
        reg_targets = np.zeros(len(df), dtype=np.float64)

        for i in range(len(df) - max_steps):
            entry_price = close[i]
            if entry_price <= 0:
                continue

            u = upper
            l = lower
            if vol_adjusted and rolling_vol is not None:
                median_vol = np.median(rolling_vol) if rolling_vol.size > 0 else 0.02
                if median_vol > 0 and rolling_vol[i] > 0:
                    scale = rolling_vol[i] / median_vol
                    u = upper * scale
                    l = lower * scale

            hit = False
            for j in range(1, max_steps + 1):
                if i + j >= len(close):
                    break
                ret = (close[i + j] - entry_price) / entry_price
                if ret >= u:
                    labels[i] = 2
                    reg_targets[i] = ret
                    hit = True
                    break
                elif ret <= l:
                    labels[i] = 0
                    reg_targets[i] = ret
                    hit = True
                    break
                elif j == max_steps:
                    reg_targets[i] = ret
            if not hit:
                reg_targets[i] = (close[min(i + max_steps, len(close) - 1)] - entry_price) / entry_price

        df[tb_col] = labels
        df[tb_reg_col] = reg_targets

        dist = Counter(labels)
        logger.info(f"Triple barrier {w}h: UP={dist[2]}, SIDEWAYS={dist[1]}, DOWN={dist[0]} "
                    f"(TP={upper:.1%}, SL={abs(lower):.1%}, horizon={max_steps} bars)")

    return df


def meta_label_filter(
    df: pd.DataFrame,
    target_col: str,
    stage1_features: List[str] = None,
    threshold: float = None,
) -> pd.DataFrame:
    """Meta-Label filter (Stage 1: TRADE vs NO_TRADE)."""
    if stage1_features is None:
        stage1_features = META_LABEL.get("stage1_features",
                                         ["rsi_14", "volume_ratio", "is_bull", "taker_buy_ratio"])
    if threshold is None:
        threshold = META_LABEL.get("stage1_threshold", 0.60)

    available = [f for f in stage1_features if f in df.columns]
    if len(available) < 2:
        logger.warning(f"Meta-label: insufficient features (need 2+, have {len(available)}). Using fallback.")
        available = [c for c in df.columns if c not in ["timestamp", "open", "high", "low", "close", "volume", target_col]
                     and df[c].dtype in (np.float64, np.float32, np.int64, np.float64)][:10]

    if len(available) < 2:
        df["trade_filter"] = 1
        df["trade_filter_proba"] = 0.5
        logger.warning("Meta-label: could not determine features, defaulting to all TRADE")
        return df

    stage1_target = (df[target_col].values != 1).astype(int)

    X = df[available].values.astype(np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = stage1_target

    valid = ~np.isnan(y.astype(float))
    X = X[valid]
    y = y[valid].astype(int)
    mask = np.zeros(len(df), dtype=bool)
    mask[valid] = True

    if len(y) < 100:
        logger.warning("Meta-label: insufficient data (<100), defaulting to all TRADE")
        df["trade_filter"] = 1
        df["trade_filter_proba"] = 0.5
        return df

    split = int(len(y) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    class_counts = np.bincount(y_train, minlength=2)
    logger.info(f"Meta-label Stage 1 distribution (train): NO_TRADE={class_counts[0]}, TRADE={class_counts[1]}")

    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    scale_pos = max(n_neg / max(n_pos, 1), 1)

    model = lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        objective="binary",
        n_jobs=-1,
        verbose=-1,
        random_state=42,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        sample_weight=np.where(y_train == 1, scale_pos, 1.0),
    )

    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)

    test_proba = model.predict_proba(X_test)[:, 1]
    test_acc = float(np.mean((test_proba >= threshold).astype(int) == y_test))
    logger.info(f"Meta-label Stage 1: test_acc={test_acc:.4f}, threshold={threshold:.2f}, "
                f"features={available}")

    df["trade_filter_proba"] = np.nan
    df.loc[mask, "trade_filter_proba"] = proba
    df["trade_filter_proba"] = df["trade_filter_proba"].fillna(0.5)

    df["trade_filter"] = 0
    df.loc[mask, "trade_filter"] = pred

    n_trade = int(df["trade_filter"].sum())
    logger.info(f"Meta-label: {n_trade}/{len(df)} samples filtered as TRADE ({n_trade/len(df)*100:.1f}%)")
    return df


def create_targets(df: pd.DataFrame, interval: str,
                    windows: Dict[str, List[int]] = None) -> pd.DataFrame:
    """Create continuous regression targets, binary direction, and optional ternary/vol targets."""
    if windows is None:
        windows = PREDICTION_WINDOWS
    df = df.copy()
    interval_hours = _interval_to_hours(interval)
    target_windows_h = windows.get(interval, [1, 4, 24])

    all_steps = []
    for w in target_windows_h:
        steps = max(1, round(w / interval_hours))
        all_steps.append(steps)
        col_reg = f"reg_target_{w}h"
        col_clf = f"clf_target_{w}h"
        df[col_reg] = df["close"].shift(-steps) / df["close"] - 1
        df[col_clf] = (df[col_reg] > CLASSIFICATION_THRESHOLD).astype(int)

        if TERNARY_CLASSIFICATION.get("enabled", False):
            band = TERNARY_CLASSIFICATION["bands"].get(interval, 0.005)
            col_ternary = f"ternary_target_{w}h"
            df[col_ternary] = np.select(
                [df[col_reg] < -band, df[col_reg] > band],
                [0, 2],
                default=1,
            )

        if VOL_NORMALIZED_TARGETS.get("enabled", False):
            vol_w = VOL_NORMALIZED_TARGETS.get("vol_window", 20)
            col_vol_reg = f"vol_reg_target_{w}h"
            ret = df["close"].pct_change()
            vol = ret.rolling(window=vol_w, min_periods=vol_w).std()
            df[col_vol_reg] = df[col_reg] / vol.shift(-steps).replace(0, np.nan)

    max_lookahead = max(all_steps)
    logger.info(f"Created targets for {len(target_windows_h)} windows, dropping last {max_lookahead} rows")
    return df
