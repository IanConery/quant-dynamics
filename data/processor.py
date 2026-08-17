"""Data processing and feature matrix orchestration.

Refactored to delegate feature engineering to `data.features` subpackage
while maintaining complete backward compatibility.
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.settings import PROCESSED_DIR
from data.features import (
    _compute_adx,
    _interval_to_hours,
    add_adx,
    add_atr,
    add_bollinger,
    add_btc_lead_lag_features,
    add_cross_asset_features,
    add_derivatives_features,
    add_ema,
    add_external_features,
    add_lag_features,
    add_macd,
    add_macro_features,
    add_momentum_features,
    add_obv,
    add_onchain_features,
    add_orderbook_features,
    add_price_patterns,
    add_regime_features,
    add_rsi,
    add_sma,
    add_stochastic,
    add_taker_flow_features,
    add_time_features,
    add_vol_adj_momentum_features,
    add_volume_sma,
    add_vwap,
    adjust_targets_for_cost,
    audit_scale_invariance,
    create_pbr_targets,
    create_targets,
    create_targets_triple_barrier,
    meta_label_filter,
    select_features,
    select_features_multi_threshold,
)
from utils.logger import setup_logger

logger = setup_logger(__name__)


# ─── Master Processing Pipeline ──────────────────────────────────────────────


def process_raw_data(
    df: pd.DataFrame,
    btc_df: pd.DataFrame = None,
    eth_df: pd.DataFrame = None,
    fng_df: pd.DataFrame = None,
    funding_df: pd.DataFrame = None,
    oi_df: pd.DataFrame = None,
    liq_df: pd.DataFrame = None,
    active_addr_df: pd.DataFrame = None,
    exchange_flow_df: pd.DataFrame = None,
    vix_df: pd.DataFrame = None,
    dxy_df: pd.DataFrame = None,
    us10y_df: pd.DataFrame = None,
    ob_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """Run full feature engineering pipeline on raw OHLCV and external data."""
    logger.info("Computing technical indicators...")
    df = add_sma(df)
    df = add_ema(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger(df)
    df = add_atr(df)
    df = add_stochastic(df)
    df = add_adx(df)
    df = add_obv(df)
    df = add_vwap(df)
    df = add_volume_sma(df)
    df = add_lag_features(df)
    df = add_price_patterns(df)

    logger.info("Computing time features...")
    df = add_time_features(df)

    logger.info("Computing regime features...")
    df = add_regime_features(df)

    logger.info("Computing cross-asset features...")
    if btc_df is not None and eth_df is not None:
        df = add_cross_asset_features(df, btc_df, eth_df)

    logger.info("Computing external features...")
    if fng_df is not None:
        df = add_external_features(df, fng_df)

    logger.info("Computing derivatives features...")
    df = add_derivatives_features(df, funding_df, oi_df, liq_df)

    logger.info("Computing on-chain features...")
    df = add_onchain_features(df, active_addr_df, exchange_flow_df)

    logger.info("Computing macro features...")
    df = add_macro_features(df, vix_df, dxy_df, us10y_df)

    logger.info("Computing order book features...")
    df = add_orderbook_features(df, ob_df)

    logger.info("Computing taker flow features...")
    df = add_taker_flow_features(df)

    logger.info("Computing momentum features...")
    df = add_momentum_features(df)

    logger.info("Computing volatility-adjusted momentum features...")
    df = add_vol_adj_momentum_features(df)

    return df


# ─── Feature Matrix Construction ─────────────────────────────────────────────


def _get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Extract numeric feature column names, excluding targets, timestamp, and metadata."""
    exclude_prefixes = [
        "timestamp", "open", "high", "low", "close", "volume",
        "btc_close", "eth_close",
        "reg_target_", "clf_target_", "ternary_target_", "vol_reg_target_",
        "tb_target_", "tb_reg_target_", "trade_filter", "pbr_target_",
    ]
    exclude_exact = {"trade_filter", "trade_filter_proba"}
    return [
        col for col in df.columns
        if col not in exclude_exact
        and df[col].dtype in (np.float64, np.float32, np.int64, np.int32, np.int8, float, int, bool)
        and not any(col.startswith(p) for p in exclude_prefixes)
    ]


def build_feature_matrix(
    df: pd.DataFrame,
    target_window: str,
    feature_cols: List[str] = None,
    exclude_cols: List[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Build feature matrix X, continuous regression targets y_reg, and binary targets y_clf."""
    if feature_cols is None:
        feature_cols = _get_feature_columns(df)
    if exclude_cols:
        feature_cols = [c for c in feature_cols if c not in exclude_cols]

    reg_col = f"reg_target_{target_window}" if "h" not in str(target_window) else f"reg_target_{target_window}"
    clf_col = f"clf_target_{target_window}" if "h" not in str(target_window) else f"clf_target_{target_window}"
    df_clean = df.dropna(subset=feature_cols + [reg_col, clf_col])

    X = df_clean[feature_cols].values.astype(np.float64)
    y_reg = df_clean[reg_col].values.astype(np.float64)
    y_clf = df_clean[clf_col].values.astype(np.int8)

    if np.isnan(X).any():
        X = np.nan_to_num(X, nan=0.0)
    if np.isinf(X).any():
        X = np.where(np.isinf(X), np.sign(X) * np.finfo(float).max, X)

    return X, y_reg, y_clf, feature_cols


def build_feature_matrix_triple_barrier(
    df: pd.DataFrame,
    target_window: str,
    feature_cols: List[str] = None,
    exclude_cols: List[str] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], List[str]]:
    """Build feature matrix for triple barrier targets.

    Returns:
        X: Feature matrix
        y_tb: Ternary triple barrier labels (0=DOWN, 1=SIDEWAYS, 2=UP)
        y_tb_reg: Exit returns
        y_filter: Binary meta-label trade filter
        feature_cols: List of used feature names
    """
    if feature_cols is None:
        feature_cols = _get_feature_columns(df)
    if exclude_cols:
        feature_cols = [c for c in feature_cols if c not in exclude_cols]

    tb_col = f"tb_target_{target_window}"
    tb_reg_col = f"tb_reg_target_{target_window}"
    filter_col = "trade_filter"

    target_cols = [c for c in feature_cols if c in df.columns]
    tb_exists = tb_col in df.columns
    tb_reg_exists = tb_reg_col in df.columns
    filter_exists = filter_col in df.columns

    drop_subset = target_cols
    if tb_exists:
        drop_subset = drop_subset + [tb_col]
    if tb_reg_exists:
        drop_subset = drop_subset + [tb_reg_col]
    if filter_exists:
        drop_subset = drop_subset + [filter_col]

    df_clean = df.dropna(subset=drop_subset)

    X = df_clean[target_cols].values.astype(np.float64)
    if np.isnan(X).any():
        X = np.nan_to_num(X, nan=0.0)
    if np.isinf(X).any():
        X = np.where(np.isinf(X), np.sign(X) * np.finfo(float).max, X)

    y_tb = df_clean[tb_col].values.astype(np.int8) if tb_exists else None
    y_tb_reg = df_clean[tb_reg_col].values.astype(np.float64) if tb_reg_exists else None
    y_filter = df_clean[filter_col].values.astype(np.int8) if filter_exists else None

    return X, y_tb, y_tb_reg, y_filter, list(target_cols)


# ─── Data Storage ────────────────────────────────────────────────────────────


def save_processed_data(df: pd.DataFrame, interval: str) -> None:
    """Save processed dataframe to parquet format in artifacts directory."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    path = f"{PROCESSED_DIR}/XRP-USDT_{interval}_processed.parquet"
    df.to_parquet(path, index=False)
    logger.info(f"Saved processed data: {path} ({len(df)} rows, {len(df.columns)} columns)")


__all__ = [
    # Pipeline & matrix builders
    "process_raw_data",
    "_get_feature_columns",
    "build_feature_matrix",
    "build_feature_matrix_triple_barrier",
    "save_processed_data",
    # Technical features
    "add_sma",
    "add_ema",
    "add_rsi",
    "add_macd",
    "add_bollinger",
    "add_atr",
    "add_stochastic",
    "add_adx",
    "add_obv",
    "add_vwap",
    "add_volume_sma",
    "add_lag_features",
    "add_price_patterns",
    "_compute_adx",
    # Cross-asset features
    "add_cross_asset_features",
    "add_btc_lead_lag_features",
    "add_momentum_features",
    "add_vol_adj_momentum_features",
    # External features
    "add_regime_features",
    "add_time_features",
    "add_external_features",
    "add_derivatives_features",
    "add_onchain_features",
    "add_macro_features",
    "add_orderbook_features",
    "add_taker_flow_features",
    # Targets
    "_interval_to_hours",
    "create_pbr_targets",
    "adjust_targets_for_cost",
    "create_targets_triple_barrier",
    "meta_label_filter",
    "create_targets",
    # Selection
    "select_features",
    "select_features_multi_threshold",
    "audit_scale_invariance",
]
