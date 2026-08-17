import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.settings import (
    BACKTEST_CONFIG,
    MODEL_DIR,
    BACKTEST_DIR,
    REGIME_PARAMS,
    BAYESIAN_UPDATING,
)
from data.loader import load_processed, chronological_split, get_datasets
from models.classical import ClassicalModel
from models.deep import DeepModel
from models.ensemble import StackingEnsemble
from backtest.metrics import compute_backtest_metrics, compute_buy_and_hold_metrics, walk_forward_significance
from utils.logger import setup_logger

logger = setup_logger(__name__)


# ─── Strategy Functions ──────────────────────────────────────────────────────


def _get_regime_params(regime: str, config: Dict = None) -> Dict:
    """Look up trading parameters for a given regime."""
    if config is None:
        config = REGIME_PARAMS
    return config.get(regime, config.get("sideways", {
        "stop_loss_pct": 0.03,
        "take_profit_pct": 0.06,
        "risk_per_trade": 0.02,
        "trailing_stop_pct": 0.015,
        "max_concurrent_trades": 1,
    }))


def _regime_aware_strategy(
    predictions: np.ndarray,
    regime_labels: np.ndarray,
    strat_config: Dict,
    config: Dict = None,
) -> np.ndarray:
    """Strategy that respects regime-specific max_concurrent_trades.

    Each regime has its own trading limits and risk parameters.
    During regime transitions (first `min_regime_duration` bars), position is held.
    """
    if config is None:
        config = REGIME_PARAMS
    buy_threshold = strat_config["buy_threshold"]
    sell_threshold = strat_config["sell_threshold"]
    positions = np.zeros(len(predictions), dtype=int)
    current_pos = 0
    last_regime = "sideways"
    regime_duration = 0
    min_regime_duration = 5

    for i in range(len(predictions)):
        prob = predictions[i]
        regime = regime_labels[i] if i < len(regime_labels) else "sideways"

        if regime != last_regime:
            regime_duration = 0
            last_regime = regime
        regime_duration += 1

        params = _get_regime_params(regime, config)
        max_trades = params.get("max_concurrent_trades", 1)

        if max_trades <= 0:
            if current_pos == 1:
                current_pos = 0
            positions[i] = current_pos
            continue

        if regime_duration < min_regime_duration:
            positions[i] = current_pos
            continue

        if prob >= buy_threshold and current_pos == 0:
            current_pos = 1
        elif prob <= sell_threshold and current_pos == 1:
            current_pos = 0
        positions[i] = current_pos

    return positions


def signal_based_strategy(
    predictions: np.ndarray,
    config: Dict = None,
) -> np.ndarray:
    if config is None:
        config = BACKTEST_CONFIG["strategies"]["signal_based"]
    buy_threshold = config["buy_threshold"]
    sell_threshold = config["sell_threshold"]
    positions = np.zeros(len(predictions), dtype=int)
    current_pos = 0
    for i in range(len(predictions)):
        prob = predictions[i]
        if prob >= buy_threshold:
            current_pos = 1
        elif prob <= sell_threshold and current_pos == 1:
            current_pos = 0
        positions[i] = current_pos
    return positions


def confidence_based_strategy(
    predictions: np.ndarray,
    config: Dict = None,
) -> np.ndarray:
    if config is None:
        config = BACKTEST_CONFIG["strategies"]["confidence_based"]
    min_confidence = config["min_confidence"]
    buy_threshold = config["buy_threshold"]
    sell_threshold = config["sell_threshold"]
    positions = np.zeros(len(predictions), dtype=int)
    current_pos = 0
    for i in range(len(predictions)):
        prob = predictions[i]
        confidence = max(prob, 1 - prob)
        if confidence >= min_confidence:
            if prob >= buy_threshold:
                current_pos = 1
            elif prob <= sell_threshold:
                current_pos = 0
        positions[i] = current_pos
    return positions


# ─── Prediction Extraction ───────────────────────────────────────────────────


def _confidence_cooldown_strategy(
    predictions: np.ndarray,
    min_confidence: float,
    strat_config: Dict,
    min_holding_bars: int,
) -> np.ndarray:
    buy_threshold = strat_config["buy_threshold"]
    sell_threshold = strat_config["sell_threshold"]
    positions = np.zeros(len(predictions), dtype=int)
    current_pos = 0
    last_trade_bar = -min_holding_bars - 1
    for i in range(len(predictions)):
        prob = predictions[i]
        confidence = max(prob, 1 - prob)
        in_cooldown = (i - last_trade_bar) < min_holding_bars
        if confidence >= min_confidence and not in_cooldown:
            if prob >= buy_threshold and current_pos == 0:
                current_pos = 1
                last_trade_bar = i
            elif prob <= sell_threshold and current_pos == 1:
                current_pos = 0
                last_trade_bar = i
        positions[i] = current_pos
    return positions


def _apply_cooldown(
    positions: np.ndarray,
    min_holding_bars: int,
) -> np.ndarray:
    result = np.zeros_like(positions)
    current_pos = 0
    last_trade_bar = -min_holding_bars - 1
    for i in range(len(positions)):
        in_cooldown = (i - last_trade_bar) < min_holding_bars
        if positions[i] != current_pos and not in_cooldown:
            current_pos = positions[i]
            last_trade_bar = i
        result[i] = current_pos
    return result


def _get_predictions(model, model_key: str, X_test: np.ndarray, X_test_seq: np.ndarray,
                     task: str = "classification") -> np.ndarray:
    """Extract probability predictions from a loaded model."""
    if isinstance(model, tuple):
        ensemble, base_models = model
        n_overlap = min(len(X_test), len(X_test_seq))
        pred = ensemble.predict(base_models, X_test[:n_overlap], X_test_seq[:n_overlap])
        return pred
    elif isinstance(model, ClassicalModel):
        if task == "classification":
            return model.predict_proba(X_test)
        return model.predict(X_test)
    elif isinstance(model, DeepModel):
        return model.predict(X_test_seq)
    else:
        if task == "classification":
            return model.predict_proba(X_test)
        return model.predict(X_test)


# ─── Single Backtest ─────────────────────────────────────────────────────────


def run_backtest(
    model,
    model_key: str,
    test_data: pd.DataFrame,
    X_test: np.ndarray,
    X_test_seq: np.ndarray,
    strategy: str = "signal_based",
    bt_config: Dict = None,
    interval: str = "1h",
    target_window: str = "1h",
    zero_commission: bool = False,
    min_confidence: float = None,
    min_holding_bars: int = 0,
    regime_labels: np.ndarray = None,
    regime_config: Dict = None,
    bayesian: bool = False,
) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, float]]:
    if bt_config is None:
        bt_config = BACKTEST_CONFIG

    risk = bt_config["risk_management"]
    strat_config = bt_config["strategies"].get(strategy, bt_config["strategies"]["signal_based"])

    initial_capital = bt_config["initial_capital"]
    commission_rate = bt_config["commission_rate"]
    slippage = bt_config["slippage"]
    if zero_commission:
        commission_rate = 0.0
        slippage = 0.0
    stop_loss_pct = risk["stop_loss_pct"]
    take_profit_pct = risk["take_profit_pct"]
    trailing_stop_pct = risk["trailing_stop_pct"]
    max_dd_cb = risk["max_drawdown_circuit_breaker"]

    is_regime = regime_labels is not None and len(regime_labels) > 0
    if regime_config is None:
        regime_config = REGIME_PARAMS

    predictions = _get_predictions(model, model_key, X_test, X_test_seq, task="classification")

    # Bayesian Updating — refine raw predictions with sequential belief updates
    if bayesian:
        from models.bayes_updater import BayesUpdater
        bayes_cfg = BAYESIAN_UPDATING if BAYESIAN_UPDATING.get("enabled") else None
        updater = BayesUpdater(config=bayes_cfg)

        # Set prior: regime-based if available, otherwise base rate
        if is_regime:
            updater.set_prior_from_regime_label(regime_labels[0] if len(regime_labels) > 0 else "sideways")
        else:
            # Default uninformative prior
            updater.set_prior(0.5)

        predictions = updater.update_batch(predictions)
        s = updater.summary()
        logger.info(f"Bayesian update: {s.get('n_applied', 0)} updates applied, "
                     f"{s.get('n_skipped', 0)} skipped, "
                     f"mean_prior={s.get('mean_prior', 0):.3f}, "
                     f"mean_posterior={s.get('mean_posterior', 0):.3f}")

    if is_regime:
        positions = _regime_aware_strategy(predictions, regime_labels, strat_config, regime_config)
    elif min_confidence is not None:
        positions = _confidence_cooldown_strategy(predictions, min_confidence, strat_config, min_holding_bars)
    else:
        positions = signal_based_strategy(predictions, strat_config) if strategy == "signal_based" else confidence_based_strategy(predictions, strat_config)

    if min_holding_bars > 0 and min_confidence is None and not is_regime:
        positions = _apply_cooldown(positions, min_holding_bars)

    close_col = "close"
    if close_col not in test_data.columns:
        raise ValueError(f"'{close_col}' not in test_data columns")

    prices = test_data[close_col].values.astype(np.float64)
    n = min(len(prices), len(predictions))
    prices = prices[:n]
    predictions = predictions[:n]
    positions = positions[:n]

    timestamps = test_data.index[:n] if hasattr(test_data.index, 'isoformat') else range(n)
    if hasattr(test_data, "timestamp"):
        ts_col = test_data.get("timestamp", None)
        if ts_col is not None:
            timestamps = ts_col[:n].values

    # Regime breakdown tracking
    regime_trade_counts = {"bull": 0, "sideways": 0, "bear": 0}
    regime_pnls = {"bull": 0.0, "sideways": 0.0, "bear": 0.0}
    regime_wins = {"bull": 0, "sideways": 0, "bear": 0}

    trades = []
    equity_curve = [initial_capital]
    capital = initial_capital
    in_position = False
    entry_price = 0.0
    entry_time = None
    entry_index = 0
    peak_price = 0.0
    position_size = 0.0
    circuit_breaker = False
    peak_equity = initial_capital
    entry_regime = "sideways"

    def _get_current_params(i):
        """Get current risk parameters, regime-aware if enabled."""
        if is_regime and i < len(regime_labels):
            return _get_regime_params(regime_labels[i], regime_config)
        return risk

    def _close_trade(i, exit_reason):
        nonlocal capital, in_position, peak_price
        price = prices[i]
        long_return = (price - entry_price) / entry_price
        position_value = position_size * (1 + long_return)
        pnl = position_value - position_size
        comm = position_size * commission_rate

        cur_params = _get_current_params(entry_index)
        sl_pct = cur_params.get("stop_loss_pct", stop_loss_pct)
        tp_pct = cur_params.get("take_profit_pct", take_profit_pct)
        sl_price = entry_price * (1 - sl_pct)
        tp_price = entry_price * (1 + tp_pct)

        # Track regime breakdown
        reg = entry_regime if is_regime else "sideways"
        regime_trade_counts[reg] = regime_trade_counts.get(reg, 0) + 1
        regime_pnls[reg] = regime_pnls.get(reg, 0.0) + (pnl - comm)
        if pnl - comm > 0:
            regime_wins[reg] = regime_wins.get(reg, 0) + 1

        trades.append({
            "entry_time": entry_time,
            "exit_time": timestamps[i] if i < len(timestamps) else i,
            "entry_price": entry_price,
            "exit_price": price,
            "pnl": pnl - comm,
            "return_pct": long_return,
            "duration": i - entry_index,
            "exit_reason": exit_reason,
            "stop_loss": sl_price,
            "take_profit": tp_price,
            "position_size": position_size,
            "commission": comm,
        })
        capital = equity - comm
        in_position = False
        peak_price = 0.0

    for i in range(n):
        price = prices[i]
        equity = capital

        if in_position:
            long_return = (price - entry_price) / entry_price
            position_value = position_size * (1 + long_return)
            equity = capital - position_size + position_value
            peak_equity = max(peak_equity, equity)

            if (peak_equity - equity) / peak_equity > max_dd_cb:
                circuit_breaker = True
                _close_trade(i, "circuit_breaker")
                continue

            cur_params = _get_current_params(entry_index)
            sl_price = entry_price * (1 - cur_params.get("stop_loss_pct", stop_loss_pct))
            tp_price = entry_price * (1 + cur_params.get("take_profit_pct", take_profit_pct))
            ts_price = peak_price * (1 - cur_params.get("trailing_stop_pct", trailing_stop_pct))

            if price <= sl_price:
                _close_trade(i, "stop_loss")
            elif price >= tp_price:
                _close_trade(i, "take_profit")
            elif price <= ts_price and peak_price > entry_price:
                _close_trade(i, "trailing_stop")
            elif positions[i] == 0 and i > 0:
                _close_trade(i, "signal")
            else:
                peak_price = max(peak_price, price)
        else:
            if positions[i] == 1 and not circuit_breaker:
                cur_params = _get_current_params(i)
                rpt = cur_params.get("risk_per_trade", risk["risk_per_trade"])
                sl_pct = cur_params.get("stop_loss_pct", stop_loss_pct)
                risk_amt = rpt * equity
                position_size = risk_amt / sl_pct if sl_pct > 0 else equity * 0.02
                position_size = min(position_size, equity * 0.1)
                entry_price = price * (1 + slippage)
                entry_time = timestamps[i] if i < len(timestamps) else i
                entry_index = i
                peak_price = entry_price
                entry_regime = regime_labels[i] if is_regime and i < len(regime_labels) else "sideways"
                in_position = True

        equity_curve.append(equity)

    if in_position:
        _close_trade(n - 1, "end_of_data")
        equity_curve[-1] = capital

    trades_df = pd.DataFrame(trades)
    equity_arr = np.array(equity_curve)

    metrics = compute_backtest_metrics(trades_df, equity_arr)
    metrics["circuit_breaker_triggered"] = circuit_breaker
    metrics["zero_commission"] = zero_commission
    metrics["min_confidence"] = min_confidence if min_confidence is not None else 0.0
    metrics["min_holding_bars"] = min_holding_bars
    metrics["regime_aware"] = is_regime
    metrics["bayesian"] = bayesian

    if is_regime:
        for reg in ["bull", "sideways", "bear"]:
            cnt = regime_trade_counts.get(reg, 0)
            pnl = regime_pnls.get(reg, 0.0)
            wins = regime_wins.get(reg, 0)
            metrics[f"regime_{reg}_trades"] = cnt
            metrics[f"regime_{reg}_pnl"] = round(pnl, 2)
            metrics[f"regime_{reg}_win_rate"] = round(wins / max(cnt, 1), 4)

    # Buy-and-hold comparison
    bh_metrics = compute_buy_and_hold_metrics(
        prices,
        initial_capital=initial_capital,
        commission_rate=commission_rate,
    )
    metrics["bh_total_return"] = bh_metrics["total_return"]
    metrics["bh_sharpe"] = bh_metrics["sharpe_ratio"]
    metrics["bh_drawdown"] = bh_metrics["max_drawdown"]
    metrics["bh_beaten"] = metrics["total_return"] > bh_metrics["total_return"]

    return trades_df, equity_arr, metrics


# ─── Walk-Forward Backtest ──────────────────────────────────────────────────


def walk_forward_backtest(
    interval: str,
    target_window: str,
    model_type: str = "ensemble",
    strategy: str = "signal_based",
    model_dir: str = None,
    output_dir: str = None,
    walk_forward_config: Dict = None,
    zero_commission: bool = False,
    wf_method: str = "expanding",
    min_confidence: float = None,
    min_holding_bars: int = 0,
    regime_aware: bool = False,
    bayesian: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    tw = target_window if "h" in str(target_window) else f"{target_window}h"
    if model_dir is None:
        model_dir = os.path.join(MODEL_DIR, interval, tw)
    if output_dir is None:
        output_dir = os.path.join(BACKTEST_DIR, interval, tw, model_type)
    if walk_forward_config is None:
        walk_forward_config = BACKTEST_CONFIG["walk_forward"]

    os.makedirs(output_dir, exist_ok=True)

    df = load_processed(interval)
    logger.info(f"Walk-forward backtest: {interval}/{tw} | model={model_type} | strategy={strategy}")
    logger.info(f"  Total data: {len(df)} rows")

    step_size = walk_forward_config.get("step_size", "1M")
    min_train = walk_forward_config.get("min_train_samples", 1260)
    min_test = walk_forward_config.get("min_test_samples", 720)
    max_folds = walk_forward_config.get("max_folds", 20)

    hours_per_step = {"15m": 1, "1h": 1, "1d": 24}.get(interval, 1)
    if step_size == "1M":
        step_samples = 720 // max(1, hours_per_step)
    elif step_size == "1W":
        step_samples = 168 // max(1, hours_per_step)
    else:
        step_samples = 720 // max(1, hours_per_step)

    all_trades = []
    all_metrics = []
    n = len(df)
    train_start = 0
    fold = 0

    while fold < max_folds:
        if wf_method == "rolling":
            train_end = train_start + min_train
            test_end = train_end + min_test
            if test_end > n:
                break
        else:
            test_end = min(train_start + min_train + min_test, n)
            train_end = test_end - min_test
            if train_end - train_start < min_train:
                train_start = 0
                train_end = min_train + min_test
                test_end = train_end + min_test
                if test_end > n:
                    break

        train_df = df.iloc[train_start:train_end]
        test_df = df.iloc[train_end:test_end]

        logger.info(f"  Fold {fold+1}: train={len(train_df)}, test={len(test_df)} "
                     f"({train_df['timestamp'].iloc[0].isoformat() if 'timestamp' in train_df.columns else '?'} -> "
                     f"{test_df['timestamp'].iloc[-1].isoformat() if 'timestamp' in test_df.columns else '?'})")

        try:
            from models.trainer import train_classical, train_deep, train_ensemble
            fold_data = _build_fold_data(train_df, test_df, tw, interval)

            if model_type == "classical":
                trained, _ = train_classical(interval, tw, fold_data,
                                             output_dir=os.path.join(output_dir, f"fold_{fold}"))
                model = trained.get("lightgbm_clf", trained.get("xgboost_clf"))
            elif model_type == "deep":
                trained, _ = train_deep(interval, tw, fold_data,
                                        output_dir=os.path.join(output_dir, f"fold_{fold}"))
                model = trained.get("itransformer_clf", trained.get("lstm_clf"))
            else:
                trained, _ = train_classical(interval, tw, fold_data,
                                             output_dir=os.path.join(output_dir, f"fold_{fold}"))
                trained_d, _ = train_deep(interval, tw, fold_data,
                                          output_dir=os.path.join(output_dir, f"fold_{fold}"))
                trained_e, _ = train_ensemble(interval, tw, fold_data,
                                              output_dir=os.path.join(output_dir, f"fold_{fold}"))
                model = trained_e.get("ensemble_clf")

            if model is None:
                logger.warning(f"  Fold {fold+1}: no model trained, skipping")
                fold += 1
                continue

            trades, equity, metrics = run_backtest(
                model, model_type, test_df,
                fold_data["X_test"], fold_data["X_test_seq"],
                strategy=strategy, interval=interval, target_window=tw,
                zero_commission=zero_commission,
                min_confidence=min_confidence,
                min_holding_bars=min_holding_bars,
                regime_labels=fold_data.get("regime_labels") if regime_aware else None,
                bayesian=bayesian,
            )
            metrics["fold"] = fold + 1
            all_trades.append(trades)
            all_metrics.append(metrics)

        except Exception as e:
            logger.error(f"  Fold {fold+1} error: {e}")

        fold += 1
        train_start = train_end + step_samples
        if train_start >= n - min_test:
            break

    combined_trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    aggregate = {k: float(np.mean([m[k] for m in all_metrics if isinstance(m.get(k), (int, float))]))
                 for k in all_metrics[0].keys() if isinstance(all_metrics[0].get(k), (int, float))} if all_metrics else {}

    if all_metrics:
        aggregate["total_folds"] = len(all_metrics)

    if all_metrics and len(all_metrics) >= 2:
        sig = walk_forward_significance(all_metrics)
        aggregate.update(sig)

    aggregate["wf_method"] = wf_method
    aggregate["zero_commission"] = zero_commission

    if combined_trades is not None and len(combined_trades) > 0:
        combined_trades.to_parquet(os.path.join(output_dir, "trades.parquet"), index=False)
    import json
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(aggregate, f, indent=2, default=str)

    logger.info(f"Walk-forward complete: {len(all_metrics)} folds, {len(combined_trades)} trades")
    logger.info(f"Results saved to {output_dir}")

    return combined_trades, aggregate


def _predict_fold_regimes(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    """Predict regime labels for test period using a regime classifier trained on train period."""
    try:
        from models.regime_classifier import RegimeClassifier

        rc = RegimeClassifier()
        n = len(train_df)
        train_end = int(n * 0.7)
        val_end = int(n * 0.85)
        rc.fit(train_df, train_end, val_end)

        labels = rc.predict(test_df)
        regime_names = np.array(["bear", "sideways", "bull"])
        return regime_names[labels]
    except Exception as e:
        logger.warning(f"Regime prediction failed: {e}. Falling back to all sideways.")
        return np.full(len(test_df), "sideways")


def _build_fold_data(train_df: pd.DataFrame, test_df: pd.DataFrame, tw: str, interval: str) -> Dict:
    """Build dataset dict for a single walk-forward fold."""
    from data.processor import build_feature_matrix
    from sklearn.preprocessing import StandardScaler
    from data.loader import create_sequences
    from config.settings import DEEP_PARAMS

    full_df = pd.concat([train_df, test_df], ignore_index=True)
    X, y_reg, y_clf, feature_names = build_feature_matrix(full_df, tw)

    train_end = len(train_df)
    X_train_raw = X[:train_end]
    X_test_raw = X[train_end:]
    y_reg_train = y_reg[:train_end]
    y_reg_test = y_reg[train_end:]
    y_clf_train = y_clf[:train_end]
    y_clf_test = y_clf[train_end:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_raw)
    X_test_s = scaler.transform(X_test_raw)

    seq_len = DEEP_PARAMS["lstm"]["seq_length"]
    X_train_seq, y_reg_train_seq = create_sequences(X_train_s, y_reg_train, seq_len)
    X_test_seq, y_reg_test_seq = create_sequences(X_test_s, y_reg_test, seq_len)
    _, y_clf_train_seq = create_sequences(X_train_s, y_clf_train, seq_len)
    _, y_clf_test_seq = create_sequences(X_test_s, y_clf_test, seq_len)

    return {
        "X_train": X_train_raw,
        "y_reg_train": y_reg_train,
        "y_clf_train": y_clf_train,
        "X_val": X_test_raw[:len(X_test_raw)//2],
        "y_reg_val": y_reg_test[:len(y_reg_test)//2],
        "y_clf_val": y_clf_test[:len(y_clf_test)//2],
        "X_test": X_test_raw[len(X_test_raw)//2:],
        "y_reg_test": y_reg_test[len(y_reg_test)//2:],
        "y_clf_test": y_clf_test[len(y_clf_test)//2:],
        "X_train_seq": X_train_seq,
        "y_reg_train_seq": y_reg_train_seq,
        "y_clf_train_seq": y_clf_train_seq,
        "X_val_seq": create_sequences(X_test_s[:len(X_test_s)//2], y_reg_test[:len(y_reg_test)//2], seq_len)[0],
        "y_reg_val_seq": create_sequences(X_test_s[:len(X_test_s)//2], y_reg_test[:len(y_reg_test)//2], seq_len)[1],
        "y_clf_val_seq": create_sequences(X_test_s[:len(X_test_s)//2], y_clf_test[:len(y_clf_test)//2], seq_len)[1],
        "X_test_seq": X_test_seq[len(X_test_seq)//2:] if len(X_test_seq) > 1 else X_test_seq,
        "y_reg_test_seq": y_reg_test_seq[len(y_reg_test_seq)//2:] if len(y_reg_test_seq) > 1 else y_reg_test_seq,
        "y_clf_test_seq": y_clf_test_seq[len(y_clf_test_seq)//2:] if len(y_clf_test_seq) > 1 else y_clf_test_seq,
        "feature_names": feature_names,
        "seq_length": seq_len,
        "scaler": scaler,
        "target_window": tw,
        "interval": interval,
        "regime_labels": _predict_fold_regimes(train_df, test_df),
    }


# ─── Strategy Comparison ───────────────────────────────────────────────────


def compare_strategies(
    interval: str,
    target_window: str,
    model_type: str = "ensemble",
    model_dir: str = None,
    output_dir: str = None,
) -> pd.DataFrame:
    tw = target_window if "h" in str(target_window) else f"{target_window}h"
    if model_dir is None:
        model_dir = os.path.join(MODEL_DIR, interval, tw)
    if output_dir is None:
        output_dir = os.path.join(BACKTEST_DIR, interval, tw)
    os.makedirs(output_dir, exist_ok=True)

    data = get_datasets(interval, tw)
    df = load_processed(interval)
    _, _, test_df = chronological_split(df)

    strategies = list(BACKTEST_CONFIG["strategies"].keys())
    results = []

    for strategy in strategies:
        logger.info(f"  Running {strategy} strategy...")

        model_path = os.path.join(model_dir, f"ensemble_clf.pkl")
        if os.path.exists(model_path):
            ensemble = StackingEnsemble.load(model_path)
            base_models = {}
            for fname in sorted(os.listdir(model_dir)):
                if fname.endswith(".pkl") or fname.endswith(".pth"):
                    key = fname.replace(".pkl", "").replace(".pth", "")
                    if "clf" in key and key != "ensemble_clf":
                        fpath = os.path.join(model_dir, fname)
                        try:
                            if fname.endswith(".pkl"):
                                base_models[fname] = ClassicalModel.load(fpath)
                            else:
                                base_models[fname] = DeepModel.load(fpath)
                        except Exception as e:
                            logger.warning(f"Could not load {fname}: {e}")
            model = (ensemble, base_models)
        else:
            model_path = os.path.join(model_dir, f"lightgbm_clf.pkl")
            if os.path.exists(model_path):
                model = ClassicalModel.load(model_path)
            else:
                logger.warning(f"No model found for {interval}/{tw}, skipping")
                continue

        trades, equity, metrics = run_backtest(
            model, "ensemble_clf",
            test_df, data["X_test"], data["X_test_seq"],
            strategy=strategy, interval=interval, target_window=tw,
        )
        metrics["strategy"] = strategy
        results.append(metrics)

    df = pd.DataFrame(results)
    if len(df) > 0:
        strat_cols = ["strategy"]
        metric_cols = [c for c in df.columns if c != "strategy"]
        df = df[[c for c in strat_cols + metric_cols if c in df.columns]]
        df.to_csv(os.path.join(output_dir, "strategy_comparison.csv"), index=False)
        logger.info(f"\nStrategy comparison ({interval}/{tw}):")
        logger.info(df.to_string(index=False))

    return df
