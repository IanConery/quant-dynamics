import os
import pickle
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config.settings import (
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    DEEP_PARAMS,
    PROCESSED_DIR,
    FEATURE_SELECTION,
    TERNARY_CLASSIFICATION,
)
from data.processor import (
    build_feature_matrix,
    build_feature_matrix_triple_barrier,
    select_features,
)
from utils.logger import setup_logger

logger = setup_logger(__name__)

# Cache directory for dataset dicts
_DATASET_CACHE_DIR = "artifacts/dataset_cache"


def _cache_path(interval: str, tw: str) -> str:
    """Return the cache file path for a given interval/window combination."""
    safe_interval = interval.replace("/", "_")
    safe_tw = tw.replace("/", "_")
    return os.path.join(_DATASET_CACHE_DIR, f"{safe_interval}_{safe_tw}.pkl")


def _load_cached_datasets(interval: str, tw: str) -> Optional[Dict]:
    """Try to load a cached dataset dict from disk."""
    path = _cache_path(interval, tw)
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            logger.info(f"Loaded cached datasets from {path}")
            return data
        except Exception as e:
            logger.warning(f"Failed to load cached datasets from {path}: {e}")
    return None


def _save_cached_datasets(interval: str, tw: str, data: Dict) -> None:
    """Save a dataset dict to disk for future reuse."""
    os.makedirs(_DATASET_CACHE_DIR, exist_ok=True)
    path = _cache_path(interval, tw)
    try:
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Cached datasets to {path}")
    except Exception as e:
        logger.warning(f"Failed to cache datasets to {path}: {e}")


def load_processed(interval: str) -> pd.DataFrame:
    import os
    path = f"{PROCESSED_DIR}/XRP-USDT_{interval}_processed.parquet"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Processed data not found: {path}. Run 'data process' first.")
    df = pd.read_parquet(path)
    logger.info(f"Loaded processed data: {len(df)} rows, {len(df.columns)} columns")
    return df


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    logger.info(
        f"Chronological split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
    )
    return train_df, val_df, test_df


def create_sequences(
    X: np.ndarray,
    y: np.ndarray,
    seq_length: int = 60,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create sliding-window sequences using vectorized stride tricks."""
    n_windows = len(X) - seq_length
    if n_windows <= 0:
        return np.empty((0, seq_length, X.shape[1])), np.empty(0)
    # sliding_window_view on (N, F) with window W, axis=0 → (N-W+1, F, W)
    # need to transpose to (N-W+1, W, F) to match (batch, seq_len, features)
    X_seq = np.lib.stride_tricks.sliding_window_view(X, seq_length, axis=0).copy()
    X_seq = X_seq.transpose(0, 2, 1)
    y_seq = y[seq_length - 1:].copy()
    logger.info(f"Created sequences: X={X_seq.shape}, y={y_seq.shape}")
    return X_seq, y_seq


def scale_features(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def get_datasets(
    interval: str,
    target_window: str,
    seq_length: int = None,
    feature_selection_enabled: bool = True,
) -> Dict:
    if seq_length is None:
        seq_length = DEEP_PARAMS["lstm"]["seq_length"]

    tw = target_window if any(s in str(target_window) for s in ["h", "d"]) else f"{target_window}h"

    cached = _load_cached_datasets(interval, tw)
    if cached is not None:
        return cached

    logger.info(f"Loading datasets for {interval} / {tw} target")
    df = load_processed(interval)

    X, y_reg, y_clf, feature_names = build_feature_matrix(df, tw)
    all_features = list(feature_names)

    if feature_selection_enabled:
        X, feature_names = select_features(
            X, y_reg, feature_names,
            method=FEATURE_SELECTION["method"],
            n_top=FEATURE_SELECTION["n_top_features"],
        )

    logger.info(f"Features: {len(feature_names)} ({len(all_features)} total before selection)")

    df_feat = df.copy()
    df_feat[feature_names] = X

    train_df, val_df, test_df = chronological_split(df_feat)

    y_reg_train = train_df[f"reg_target_{tw}"].values.astype(np.float64)
    y_clf_train = train_df[f"clf_target_{tw}"].values.astype(np.int8)

    X_val = val_df[feature_names].values.astype(np.float64)
    y_reg_val = val_df[f"reg_target_{tw}"].values.astype(np.float64)
    y_clf_val = val_df[f"clf_target_{tw}"].values.astype(np.int8)

    X_test = test_df[feature_names].values.astype(np.float64)
    y_reg_test = test_df[f"reg_target_{tw}"].values.astype(np.float64)
    y_clf_test = test_df[f"clf_target_{tw}"].values.astype(np.int8)

    # Ternary targets
    ternary_col = f"ternary_target_{tw}"
    has_ternary = ternary_col in train_df.columns
    if has_ternary:
        y_ternary_train = train_df[ternary_col].values.astype(np.int8)
        y_ternary_val = val_df[ternary_col].values.astype(np.int8)
        y_ternary_test = test_df[ternary_col].values.astype(np.int8)
    else:
        y_ternary_train = y_ternary_val = y_ternary_test = None

    # Volatility-normalized targets
    vol_reg_col = f"vol_reg_target_{tw}"
    has_vol_reg = vol_reg_col in train_df.columns
    if has_vol_reg:
        y_vol_reg_train = train_df[vol_reg_col].values.astype(np.float64)
        y_vol_reg_val = val_df[vol_reg_col].values.astype(np.float64)
        y_vol_reg_test = test_df[vol_reg_col].values.astype(np.float64)
    else:
        y_vol_reg_train = y_vol_reg_val = y_vol_reg_test = None

    X_train = train_df[feature_names].values.astype(np.float64)

    X_train_s, X_val_s, X_test_s, scaler = scale_features(X_train, X_val, X_test)

    X_train_seq, y_reg_train_seq = create_sequences(X_train_s, y_reg_train, seq_length)
    X_val_seq, y_reg_val_seq = create_sequences(X_val_s, y_reg_val, seq_length)
    X_test_seq, y_reg_test_seq = create_sequences(X_test_s, y_reg_test, seq_length)

    _, y_clf_train_seq = create_sequences(X_train_s, y_clf_train, seq_length)
    _, y_clf_val_seq = create_sequences(X_val_s, y_clf_val, seq_length)
    _, y_clf_test_seq = create_sequences(X_test_s, y_clf_test, seq_length)

    # Ternary sequences
    if has_ternary:
        _, y_ternary_train_seq = create_sequences(X_train_s, y_ternary_train, seq_length)
        _, y_ternary_val_seq = create_sequences(X_val_s, y_ternary_val, seq_length)
        _, y_ternary_test_seq = create_sequences(X_test_s, y_ternary_test, seq_length)
    else:
        y_ternary_train_seq = y_ternary_val_seq = y_ternary_test_seq = None

    # Vol-reg sequences
    if has_vol_reg:
        _, y_vol_reg_train_seq = create_sequences(X_train_s, y_vol_reg_train, seq_length)
        _, y_vol_reg_val_seq = create_sequences(X_val_s, y_vol_reg_val, seq_length)
        _, y_vol_reg_test_seq = create_sequences(X_test_s, y_vol_reg_test, seq_length)
    else:
        y_vol_reg_train_seq = y_vol_reg_val_seq = y_vol_reg_test_seq = None

    # Triple barrier targets
    tb_col = f"tb_target_{tw}"
    tb_reg_col = f"tb_reg_target_{tw}"
    has_tb = tb_col in train_df.columns
    if has_tb:
        y_tb_train = train_df[tb_col].values.astype(np.int8)
        y_tb_val = val_df[tb_col].values.astype(np.int8)
        y_tb_test = test_df[tb_col].values.astype(np.int8)
        _, y_tb_train_seq = create_sequences(X_train_s, y_tb_train, seq_length)
        _, y_tb_val_seq = create_sequences(X_val_s, y_tb_val, seq_length)
        _, y_tb_test_seq = create_sequences(X_test_s, y_tb_test, seq_length)
    else:
        y_tb_train = y_tb_val = y_tb_test = None
        y_tb_train_seq = y_tb_val_seq = y_tb_test_seq = None

    has_tb_reg = tb_reg_col in train_df.columns
    if has_tb_reg:
        y_tb_reg_train = train_df[tb_reg_col].values.astype(np.float64)
        y_tb_reg_val = val_df[tb_reg_col].values.astype(np.float64)
        y_tb_reg_test = test_df[tb_reg_col].values.astype(np.float64)
        _, y_tb_reg_train_seq = create_sequences(X_train_s, y_tb_reg_train, seq_length)
        _, y_tb_reg_val_seq = create_sequences(X_val_s, y_tb_reg_val, seq_length)
        _, y_tb_reg_test_seq = create_sequences(X_test_s, y_tb_reg_test, seq_length)
    else:
        y_tb_reg_train = y_tb_reg_val = y_tb_reg_test = None
        y_tb_reg_train_seq = y_tb_reg_val_seq = y_tb_reg_test_seq = None

    # Trade filter
    has_filter = "trade_filter" in train_df.columns
    if has_filter:
        y_filter_train = train_df["trade_filter"].values.astype(np.int8)
        y_filter_val = val_df["trade_filter"].values.astype(np.int8)
        y_filter_test = test_df["trade_filter"].values.astype(np.int8)
    else:
        y_filter_train = y_filter_val = y_filter_test = None

    result = {
        "X_train": X_train,
        "y_reg_train": y_reg_train,
        "y_clf_train": y_clf_train,
        "X_val": X_val,
        "y_reg_val": y_reg_val,
        "y_clf_val": y_clf_val,
        "X_test": X_test,
        "y_reg_test": y_reg_test,
        "y_clf_test": y_clf_test,
        "X_train_seq": X_train_seq,
        "y_reg_train_seq": y_reg_train_seq,
        "y_clf_train_seq": y_clf_train_seq,
        "X_val_seq": X_val_seq,
        "y_reg_val_seq": y_reg_val_seq,
        "y_clf_val_seq": y_clf_val_seq,
        "X_test_seq": X_test_seq,
        "y_reg_test_seq": y_reg_test_seq,
        "y_clf_test_seq": y_clf_test_seq,
        "y_ternary_train": y_ternary_train,
        "y_ternary_val": y_ternary_val,
        "y_ternary_test": y_ternary_test,
        "y_ternary_train_seq": y_ternary_train_seq,
        "y_ternary_val_seq": y_ternary_val_seq,
        "y_ternary_test_seq": y_ternary_test_seq,
        "y_vol_reg_train": y_vol_reg_train,
        "y_vol_reg_val": y_vol_reg_val,
        "y_vol_reg_test": y_vol_reg_test,
        "y_vol_reg_train_seq": y_vol_reg_train_seq,
        "y_vol_reg_val_seq": y_vol_reg_val_seq,
        "y_vol_reg_test_seq": y_vol_reg_test_seq,
        "feature_names": feature_names,
        "selected_features": feature_names,
        "all_features": all_features,
        "seq_length": seq_length,
        "scaler": scaler,
        "target_window": tw,
        "interval": interval,
        "class_balance": {
            "train": float(y_clf_train.mean()),
            "val": float(y_clf_val.mean()),
            "test": float(y_clf_test.mean()),
        },
        "has_ternary": has_ternary,
        "has_vol_reg": has_vol_reg,
        # Triple barrier targets
        "y_tb_train": y_tb_train,
        "y_tb_val": y_tb_val,
        "y_tb_test": y_tb_test,
        "y_tb_train_seq": y_tb_train_seq,
        "y_tb_val_seq": y_tb_val_seq,
        "y_tb_test_seq": y_tb_test_seq,
        "y_tb_reg_train": y_tb_reg_train,
        "y_tb_reg_val": y_tb_reg_val,
        "y_tb_reg_test": y_tb_reg_test,
        "y_tb_reg_train_seq": y_tb_reg_train_seq,
        "y_tb_reg_val_seq": y_tb_reg_val_seq,
        "y_tb_reg_test_seq": y_tb_reg_test_seq,
        "y_filter_train": y_filter_train,
        "y_filter_val": y_filter_val,
        "y_filter_test": y_filter_test,
        "has_tb": has_tb,
        "has_tb_reg": has_tb_reg,
        "has_filter": has_filter,
    }
    _save_cached_datasets(interval, tw, result)
    return result


def get_tb_datasets(
    interval: str,
    target_window: str,
    seq_length: int = None,
    feature_selection_enabled: bool = True,
    apply_meta_filter: bool = False,
) -> Dict:
    """Get datasets with triple barrier targets remapped to standard keys.

    Uses TB ternary labels (0=DOWN, 1=SIDEWAYS, 2=UP) as classification targets
    and TB exit returns as regression targets. Optionally applies meta-label filter
    to only keep TRADE samples.

    Args:
        interval: Data interval (15m, 1h, 1d)
        target_window: Prediction window (e.g., 72h)
        seq_length: Sequence length for deep models
        feature_selection_enabled: Whether to run feature selection
        apply_meta_filter: Whether to filter to only TRADE samples

    Returns:
        Dataset dict with TB targets remapped to standard y_clf_* and y_reg_* keys.
    """
    tw = target_window if any(s in str(target_window) for s in ["h", "d"]) else f"{target_window}h"

    data = get_datasets(interval, tw, seq_length, feature_selection_enabled)

    if not data.get("has_tb"):
        raise ValueError(
            f"No triple barrier targets found for {interval}/{tw}. "
            "Run 'python main.py label-triple-barrier --interval {interval} --window {tw}' first."
        )

    if apply_meta_filter and not data.get("has_filter"):
        logger.warning("Meta-label filter requested but 'trade_filter' column not found. Proceeding without filter.")
        apply_meta_filter = False

    result = dict(data)
    result["n_classes"] = 3

    # Remap TB targets to standard keys
    result["y_clf_train"] = data["y_tb_train"].copy()
    result["y_clf_val"] = data["y_tb_val"].copy()
    result["y_clf_test"] = data["y_tb_test"].copy()
    result["y_reg_train"] = data["y_tb_reg_train"].copy()
    result["y_reg_val"] = data["y_tb_reg_val"].copy()
    result["y_reg_test"] = data["y_tb_reg_test"].copy()

    # Remap TB sequences
    result["y_clf_train_seq"] = data["y_tb_train_seq"].copy()
    result["y_clf_val_seq"] = data["y_tb_val_seq"].copy()
    result["y_clf_test_seq"] = data["y_tb_test_seq"].copy()
    result["y_reg_train_seq"] = data["y_tb_reg_train_seq"].copy()
    result["y_reg_val_seq"] = data["y_tb_reg_val_seq"].copy()
    result["y_reg_test_seq"] = data["y_tb_reg_test_seq"].copy()

    tb_dist = {
        "train": dict(zip(*np.unique(result["y_clf_train"], return_counts=True))),
        "val": dict(zip(*np.unique(result["y_clf_val"], return_counts=True))),
        "test": dict(zip(*np.unique(result["y_clf_test"], return_counts=True))),
    }

    if apply_meta_filter:
        # Apply meta-label filter to train/val sets
        train_mask = data["y_filter_train"] == 1
        val_mask = data["y_filter_val"] == 1
        test_mask = data["y_filter_test"] == 1

        n_train_before = len(result["X_train"])
        n_val_before = len(result["X_val"])
        n_test_before = len(result["X_test"])

        # Filter flat data
        result["X_train"] = result["X_train"][train_mask]
        result["y_clf_train"] = result["y_clf_train"][train_mask]
        result["y_reg_train"] = result["y_reg_train"][train_mask]
        result["X_val"] = result["X_val"][val_mask]
        result["y_clf_val"] = result["y_clf_val"][val_mask]
        result["y_reg_val"] = result["y_reg_val"][val_mask]
        result["X_test"] = result["X_test"][test_mask]
        result["y_clf_test"] = result["y_clf_test"][test_mask]
        result["y_reg_test"] = result["y_reg_test"][test_mask]

        # Rebuild sequences from filtered scaled data
        s = result["scaler"]
        X_train_s = s.transform(result["X_train"])
        X_val_s = s.transform(result["X_val"])
        X_test_s = s.transform(result["X_test"])

        result["X_train_seq"], result["y_reg_train_seq"] = create_sequences(
            X_train_s, result["y_reg_train"], data["seq_length"])
        result["X_val_seq"], result["y_reg_val_seq"] = create_sequences(
            X_val_s, result["y_reg_val"], data["seq_length"])
        result["X_test_seq"], result["y_reg_test_seq"] = create_sequences(
            X_test_s, result["y_reg_test"], data["seq_length"])

        _, result["y_clf_train_seq"] = create_sequences(
            X_train_s, result["y_clf_train"], data["seq_length"])
        _, result["y_clf_val_seq"] = create_sequences(
            X_val_s, result["y_clf_val"], data["seq_length"])
        _, result["y_clf_test_seq"] = create_sequences(
            X_test_s, result["y_clf_test"], data["seq_length"])

        n_train_after = len(result["X_train"])
        n_val_after = len(result["X_val"])
        n_test_after = len(result["X_test"])
        logger.info(f"Meta-filter applied: train {n_train_before}->{n_train_after}, "
                     f"val {n_val_before}->{n_val_after}, test {n_test_before}->{n_test_after}")

    tb_dist = {
        "train": dict(zip(*np.unique(result["y_clf_train"], return_counts=True))),
        "val": dict(zip(*np.unique(result["y_clf_val"], return_counts=True))),
        "test": dict(zip(*np.unique(result["y_clf_test"], return_counts=True))),
    }
    logger.info(f"TB dataset ({interval}/{tw}): n_classes=3, "
                 f"train_dist={tb_dist['train']}, val_dist={tb_dist['val']}, test_dist={tb_dist['test']}")

    result["tb_used"] = True
    result["meta_filter_applied"] = apply_meta_filter
    result["has_tb"] = True
    result["has_tb_reg"] = True
    result["has_filter"] = apply_meta_filter

    return result


def temporal_cross_validation(
    df: pd.DataFrame,
    target_window: str,
    n_folds: int = 5,
    min_train_samples: int = 500,
    min_val_samples: int = 100,
    gap: int = 0,
    seq_length: int = None,
    feature_selection_enabled: bool = True,
) -> List[Dict]:
    """Perform temporal (walk-forward) cross-validation.

    Creates expanding-window splits: each fold uses all data up to a point for training
    and the next segment for validation. Includes a configurable gap between train and val
    to prevent immediate temporal leakage.

    Args:
        df: Processed dataframe with features and targets
        target_window: Prediction window (e.g., '24h')
        n_folds: Number of folds
        min_train_samples: Minimum training samples per fold
        min_val_samples: Minimum validation samples per fold
        gap: Number of samples to skip between train and val (prevents leakage)
        seq_length: Sequence length for deep models (None = use default)
        feature_selection_enabled: Whether to run feature selection

    Returns:
        List of dataset dicts, one per fold. Each dict has same structure as get_datasets().
    """
    if seq_length is None:
        seq_length = DEEP_PARAMS["lstm"]["seq_length"]

    tw = target_window if any(s in str(target_window) for s in ["h", "d"]) else f"{target_window}h"

    X, y_reg, y_clf, feature_names = build_feature_matrix(df, tw)
    all_features = list(feature_names)

    if feature_selection_enabled:
        X, feature_names = select_features(
            X, y_reg, feature_names,
            method=FEATURE_SELECTION["method"],
            n_top=FEATURE_SELECTION["n_top_features"],
        )

    n = len(X)
    fold_size = max((n - min_train_samples) // n_folds, min_val_samples)

    folds = []
    train_start = 0
    train_end = min_train_samples

    for fold_i in range(n_folds):
        val_end = train_end + gap + fold_size
        if val_end > n:
            break

        X_train = X[train_start:train_end]
        y_reg_train = y_reg[train_start:train_end]
        y_clf_train = y_clf[train_start:train_end]

        X_val = X[train_end + gap:val_end]
        y_reg_val = y_reg[train_end + gap:val_end]
        y_clf_val = y_clf[train_end + gap:val_end]

        X_train_s, X_val_s, _, _ = scale_features(X_train, X_val, X_val)

        X_train_seq, y_reg_train_seq = create_sequences(X_train_s, y_reg_train, seq_length)
        X_val_seq, y_reg_val_seq = create_sequences(X_val_s, y_reg_val, seq_length)
        _, y_clf_train_seq = create_sequences(X_train_s, y_clf_train, seq_length)
        _, y_clf_val_seq = create_sequences(X_val_s, y_clf_val, seq_length)

        fold_data = {
            "X_train": X_train,
            "y_reg_train": y_reg_train,
            "y_clf_train": y_clf_train,
            "X_val": X_val,
            "y_reg_val": y_reg_val,
            "y_clf_val": y_clf_val,
            "X_test": X_val,
            "y_reg_test": y_reg_val,
            "y_clf_test": y_clf_val,
            "X_train_seq": X_train_seq,
            "y_reg_train_seq": y_reg_train_seq,
            "y_clf_train_seq": y_clf_train_seq,
            "X_val_seq": X_val_seq,
            "y_reg_val_seq": y_reg_val_seq,
            "y_clf_val_seq": y_clf_val_seq,
            "X_test_seq": X_val_seq,
            "y_reg_test_seq": y_reg_val_seq,
            "y_clf_test_seq": y_clf_val_seq,
            "y_ternary_train": None,
            "y_ternary_val": None,
            "y_ternary_test": None,
            "y_ternary_train_seq": None,
            "y_ternary_val_seq": None,
            "y_ternary_test_seq": None,
            "y_vol_reg_train": None,
            "y_vol_reg_val": None,
            "y_vol_reg_test": None,
            "y_vol_reg_train_seq": None,
            "y_vol_reg_val_seq": None,
            "y_vol_reg_test_seq": None,
            "feature_names": feature_names,
            "selected_features": feature_names,
            "all_features": all_features,
            "seq_length": seq_length,
            "scaler": None,
            "target_window": tw,
            "interval": "",
            "class_balance": {
                "train": float(y_clf_train.mean()),
                "val": float(y_clf_val.mean()),
                "test": float(y_clf_val.mean()),
            },
            "has_ternary": False,
            "has_vol_reg": False,
            "fold": fold_i + 1,
            "train_size": len(X_train),
            "val_size": len(X_val),
        }
        folds.append(fold_data)

        train_end = val_end

    logger.info(f"Temporal CV: {len(folds)} folds created (gap={gap}, min_train={min_train_samples})")
    return folds
