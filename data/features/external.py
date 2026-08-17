"""External data, macro, derivatives, on-chain, order book, regime, and time features."""

from typing import Dict
import numpy as np
import pandas as pd

from config.settings import (
    DERIVATIVES_CONFIG,
    MACRO_CONFIG,
    ON_CHAIN_CONFIG,
    ORDER_BOOK_CONFIG,
    REGIME_FEATURES,
    TAKER_FLOW_FEATURES,
    TIME_FEATURES,
)
from utils.logger import setup_logger

logger = setup_logger(__name__)


def add_regime_features(df: pd.DataFrame, config: Dict = None) -> pd.DataFrame:
    """Add regime features (rolling Sharpe, ADX trend flag, vol clustering, bull/bear flag)."""
    if config is None:
        config = REGIME_FEATURES
    df = df.copy()
    close = df["close"]

    # Rolling Sharpe ratio (return / volatility)
    if config.get("rolling_sharpe"):
        w = config.get("sharpe_window", 20)
        ret = close.pct_change()
        rolling_mean = ret.rolling(window=w, min_periods=w).mean()
        rolling_std = ret.rolling(window=w, min_periods=w).std()
        df["rolling_sharpe"] = rolling_mean / rolling_std.replace(0, np.nan)

    # ADX trend flag (trending vs ranging)
    if config.get("adx_trend_flag"):
        threshold = config.get("adx_threshold", 25)
        df["is_trending"] = (df.get("adx", pd.Series(dtype=float)) > threshold).astype(float)

    # Volatility clustering (current vol / mean vol)
    if config.get("vol_clustering"):
        w = config.get("vol_window", 20)
        ret = close.pct_change()
        rolling_std = ret.rolling(window=w, min_periods=w).std()
        mean_std = rolling_std.rolling(window=w * 4, min_periods=w).mean()
        df["vol_ratio"] = rolling_std / mean_std.replace(0, np.nan)

    # Bull/bear regime (price vs long SMA)
    if config.get("bull_bear_flag"):
        sma_period = config.get("bull_bear_sma", 200)
        sma_col = f"sma_{sma_period}"
        if sma_col in df.columns:
            df["is_bull"] = (close > df[sma_col]).astype(float)
        else:
            sma = close.rolling(window=sma_period, min_periods=sma_period).mean()
            df["is_bull"] = (close > sma).astype(float)

    return df


def add_time_features(df: pd.DataFrame, config: Dict = None) -> pd.DataFrame:
    """Add cyclical and calendar time features."""
    if config is None:
        config = TIME_FEATURES
    ts = df["timestamp"]
    if config.get("hour_of_day"):
        df["hour_sin"] = np.sin(2 * np.pi * ts.dt.hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * ts.dt.hour / 24)
    if config.get("day_of_week"):
        df["dow_sin"] = np.sin(2 * np.pi * ts.dt.dayofweek / 7)
        df["dow_cos"] = np.cos(2 * np.pi * ts.dt.dayofweek / 7)
    if config.get("month"):
        df["month_sin"] = np.sin(2 * np.pi * ts.dt.month / 12)
        df["month_cos"] = np.cos(2 * np.pi * ts.dt.month / 12)
    if config.get("is_month_end"):
        df["is_month_end"] = (ts.dt.day >= 28).astype(int)
    if config.get("is_week_end"):
        df["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)
    return df


def add_external_features(df: pd.DataFrame, fng_df: pd.DataFrame) -> pd.DataFrame:
    """Add sentiment / Fear & Greed Index features."""
    if fng_df is None or fng_df.empty:
        logger.warning("FNG data empty — skipping external features")
        return df

    df = df.copy()
    freq = pd.infer_freq(df["timestamp"])
    if freq is None:
        freq = "1h"

    df = df.set_index("timestamp")
    fng_indexed = fng_df.set_index("timestamp")["value"]
    df["fng_value"] = df.index.to_series().map(fng_indexed.asof)
    df = df.reset_index()
    df["fng_value"] = df["fng_value"].fillna(df["fng_value"].median())

    steps_per_day = {"1min": 1440, "5min": 288, "15min": 96, "30min": 48,
                     "1h": 24, "2h": 12, "4h": 6, "1d": 1}.get(freq, 24)
    steps_7d = steps_per_day * 7

    df["fng_7d_ma"] = df["fng_value"].rolling(window=steps_7d, min_periods=1).mean()
    df["fng_change_7d"] = df["fng_value"] - df["fng_value"].shift(steps_7d)
    df["fng_extreme_fear"] = (df["fng_value"] < 25).astype(int)
    df["fng_extreme_greed"] = (df["fng_value"] > 75).astype(int)
    return df


def add_derivatives_features(
    df: pd.DataFrame,
    funding_df: pd.DataFrame = None,
    oi_df: pd.DataFrame = None,
    liq_df: pd.DataFrame = None,
    config: Dict = None,
) -> pd.DataFrame:
    """Add derivatives-based features: funding rates, OI, liquidations."""
    if config is None:
        config = DERIVATIVES_CONFIG

    df = df.copy()
    if df.empty:
        return df

    # Funding rate features
    if funding_df is not None and not funding_df.empty and config.get("funding_rate", {}).get("enabled"):
        df = _add_funding_features(df, funding_df)

    # Open interest features
    if oi_df is not None and not oi_df.empty and config.get("open_interest", {}).get("enabled"):
        df = _add_oi_features(df, oi_df)

    # Liquidation features
    if liq_df is not None and not liq_df.empty and config.get("liquidations", {}).get("enabled"):
        df = _add_liq_features(df, liq_df)

    # Fill NaN for optional features with forward/backward fill
    for col in ["funding_rate", "funding_rate_zscore", "funding_rate_abs", "funding_rate_change",
                "funding_extreme_pos", "funding_extreme_neg",
                "open_interest", "oi_change_pct", "oi_change_24h_pct", "oi_volume_ratio",
                "total_liq", "liq_ratio", "liq_rolling_max"]:
        if col in df.columns:
            df[col] = df[col].ffill().bfill().fillna(0)

    return df


def _add_funding_features(df: pd.DataFrame, funding_df: pd.DataFrame) -> pd.DataFrame:
    """Merge funding rate data and compute derived features."""
    df = df.set_index("timestamp")
    fr = funding_df.set_index("timestamp")["funding_rate"]

    df["funding_rate"] = df.index.to_series().map(fr.asof)
    df = df.reset_index()
    df["funding_rate"] = df["funding_rate"].ffill().bfill()

    fr_mean = df["funding_rate"].rolling(window=20, min_periods=1).mean()
    fr_std = df["funding_rate"].rolling(window=20, min_periods=1).std()
    df["funding_rate_zscore"] = (df["funding_rate"] - fr_mean) / fr_std.replace(0, np.nan)
    df["funding_rate_abs"] = df["funding_rate"].abs()
    df["funding_rate_change"] = df["funding_rate"].diff()
    df["funding_extreme_pos"] = (df["funding_rate"] > 0.0001).astype(int)
    df["funding_extreme_neg"] = (df["funding_rate"] < -0.0001).astype(int)

    logger.info(f"Funding rate features added ({len(funding_df)} source records)")
    return df


def _add_oi_features(df: pd.DataFrame, oi_df: pd.DataFrame) -> pd.DataFrame:
    """Merge OI data and compute derived features."""
    df = df.set_index("timestamp")

    if "open_interest" in oi_df.columns:
        oi = oi_df.set_index("timestamp")["open_interest"]
        df["open_interest"] = df.index.to_series().map(oi.asof)
        df = df.reset_index()
        df["open_interest"] = df["open_interest"].ffill().bfill()

        df["oi_change_pct"] = df["open_interest"].pct_change()
        df["oi_change_24h_pct"] = df["open_interest"].pct_change(periods=24)

        if "volume" in df.columns:
            df["oi_volume_ratio"] = df["open_interest"] / df["volume"].replace(0, np.nan)
    elif "volume" in oi_df.columns:
        oi = oi_df.set_index("timestamp")["volume"]
        df["open_interest"] = df.index.to_series().map(oi.asof)
        df = df.reset_index()
        df["open_interest"] = df["open_interest"].ffill().bfill()
        df["oi_change_pct"] = df["open_interest"].pct_change()

    logger.info(f"OI features added ({len(oi_df)} source records)")
    return df


def _add_liq_features(df: pd.DataFrame, liq_df: pd.DataFrame) -> pd.DataFrame:
    """Merge liquidation data and compute derived features."""
    df = df.set_index("timestamp")

    if "total_liq" in liq_df.columns:
        tl = liq_df.set_index("timestamp")["total_liq"]
        df["total_liq"] = df.index.to_series().map(tl.asof)
        df = df.reset_index()
        df["total_liq"] = df["total_liq"].ffill().bfill()

        if "volume" in df.columns:
            df["liq_ratio"] = df["total_liq"] / df["volume"].replace(0, np.nan)

        df["liq_rolling_max"] = df["total_liq"].rolling(window=7, min_periods=1).max()
    elif "liq_count" in liq_df.columns:
        lc = liq_df.set_index("timestamp")["liq_count"]
        df["total_liq"] = df.index.to_series().map(lc.asof)
        df = df.reset_index()
        df["total_liq"] = df["total_liq"].ffill().bfill()
        df["liq_ratio"] = 0.0

    logger.info(f"Liquidation features added ({len(liq_df)} source records)")
    return df


def add_onchain_features(
    df: pd.DataFrame,
    active_addr_df: pd.DataFrame = None,
    exchange_flow_df: pd.DataFrame = None,
    config: Dict = None,
) -> pd.DataFrame:
    """Add on-chain data features."""
    if config is None:
        config = ON_CHAIN_CONFIG

    df = df.copy()
    if df.empty:
        return df

    if active_addr_df is not None and not active_addr_df.empty:
        df = _add_active_address_features(df, active_addr_df)

    if exchange_flow_df is not None and not exchange_flow_df.empty:
        df = _add_exchange_flow_features(df, exchange_flow_df)

    for col in ["n_requests", "hashrate", "n_requests_change", "hashrate_change",
                "total_volume_24h", "market_cap", "price_change_24h"]:
        if col in df.columns:
            df[col] = df[col].ffill().bfill().fillna(0)

    return df


def _add_active_address_features(df: pd.DataFrame, addr_df: pd.DataFrame) -> pd.DataFrame:
    """Add active address and hashrate features."""
    df = df.set_index("timestamp")

    for col in ["n_requests", "hashrate"]:
        if col in addr_df.columns:
            series = addr_df.set_index("timestamp")[col]
            df[col] = df.index.to_series().map(series.asof)

    df = df.reset_index()
    for col in ["n_requests", "hashrate"]:
        if col in df.columns:
            df[col] = df[col].ffill().bfill()
            df[f"{col}_change"] = df[col].pct_change()

    logger.info("Active address features added")
    return df


def _add_exchange_flow_features(df: pd.DataFrame, flow_df: pd.DataFrame) -> pd.DataFrame:
    """Add exchange flow proxy features."""
    df = df.set_index("timestamp")

    for col in ["total_volume_24h", "market_cap", "price_change_24h"]:
        if col in flow_df.columns:
            series = flow_df.set_index("timestamp")[col]
            df[col] = df.index.to_series().map(series.asof)

    df = df.reset_index()
    for col in ["total_volume_24h", "market_cap", "price_change_24h"]:
        if col in df.columns:
            df[col] = df[col].ffill().bfill()

    logger.info("Exchange flow features added")
    return df


def add_macro_features(
    df: pd.DataFrame,
    vix_df: pd.DataFrame = None,
    dxy_df: pd.DataFrame = None,
    us10y_df: pd.DataFrame = None,
    config: Dict = None,
) -> pd.DataFrame:
    """Add macroeconomic features: VIX, DXY, US10Y."""
    if config is None:
        config = MACRO_CONFIG

    df = df.copy()
    if df.empty:
        return df

    for name, macro_df in [("vix", vix_df), ("dxy", dxy_df), ("us10y", us10y_df)]:
        if macro_df is None or macro_df.empty:
            continue

        value_col = f"{name}_value"
        df = df.set_index("timestamp")

        if "close" in macro_df.columns:
            series = macro_df.set_index("timestamp")["close"]
        elif value_col in macro_df.columns:
            series = macro_df.set_index("timestamp")[value_col]
        else:
            value_cols = [c for c in macro_df.columns if c != "timestamp"]
            if value_cols:
                series = macro_df.set_index("timestamp")[value_cols[0]]
            else:
                continue

        df[f"{name}_value"] = df.index.to_series().map(series.asof)
        df = df.reset_index()
        df[f"{name}_value"] = df[f"{name}_value"].ffill().bfill()

        df[f"{name}_change"] = df[f"{name}_value"].pct_change()
        df[f"{name}_change_5d"] = df[f"{name}_value"].pct_change(periods=5)

    for col in ["vix_value", "vix_change", "vix_change_5d",
                "dxy_value", "dxy_change", "dxy_change_5d",
                "us10y_value", "us10y_change", "us10y_change_5d"]:
        if col in df.columns:
            df[col] = df[col].ffill().bfill().fillna(0)

    logger.info("Macro features added")
    return df


def add_orderbook_features(
    df: pd.DataFrame,
    ob_df: pd.DataFrame = None,
    config: Dict = None,
) -> pd.DataFrame:
    """Add order book imbalance features."""
    if config is None:
        config = ORDER_BOOK_CONFIG

    df = df.copy()
    if ob_df is None or ob_df.empty:
        logger.info("No order book data available — skipping")
        return df

    df = df.set_index("timestamp")

    for col in ["order_book_imbalance", "depth_ratio", "spread_pct"]:
        if col in ob_df.columns:
            series = ob_df.set_index("timestamp")[col]
            df[col] = df.index.to_series().map(series.asof)

    df = df.reset_index()
    for col in ["order_book_imbalance", "depth_ratio", "spread_pct"]:
        if col in df.columns:
            df[col] = df[col].ffill().bfill()

    logger.info("Order book features added")
    return df


def add_taker_flow_features(df: pd.DataFrame, config: Dict = None) -> pd.DataFrame:
    """Add taker buy/sell ratio features from kline data."""
    if config is None:
        config = TAKER_FLOW_FEATURES

    if not config.get("enabled", True):
        return df

    df = df.copy()

    taker_cols = []
    for col in ["taker_buy_base_volume", "taker_buy_quote_volume", "trade_count"]:
        if col in df.columns:
            taker_cols.append(col)

    if not taker_cols or df.empty:
        logger.info("Taker flow: insufficient data, skipping")
        return df

    if "taker_buy_base_volume" in df.columns:
        df["taker_buy_ratio"] = (
            df["taker_buy_base_volume"] / df["volume"].replace(0, np.nan)
        )

    if "taker_buy_quote_volume" in df.columns and "quote_volume" in df.columns:
        df["taker_buy_quote_ratio"] = (
            df["taker_buy_quote_volume"] / df["quote_volume"].replace(0, np.nan)
        )

    if "taker_buy_ratio" in df.columns:
        ma_w = config.get("ma_window", 20)
        df["taker_buy_ratio_ma"] = (
            df["taker_buy_ratio"].rolling(window=ma_w, min_periods=1).mean()
        )

    for col in ["taker_buy_ratio", "taker_buy_quote_ratio", "trade_count", "taker_buy_ratio_ma"]:
        if col in df.columns:
            df[col] = df[col].ffill().bfill().fillna(0.5 if "ratio" in col else 0)

    logger.info("Taker flow features added")
    return df
