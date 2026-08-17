"""Cross-asset, lead/lag, and momentum features."""

from typing import Dict
import numpy as np
import pandas as pd

from config.settings import (
    CROSS_ASSET_FEATURES,
    BTC_LEAD_LAG,
    MOMENTUM_TARGETS,
    VOL_ADJ_MOMENTUM,
)
from data.features.technical import _compute_adx
from utils.logger import setup_logger

logger = setup_logger(__name__)


def add_cross_asset_features(df: pd.DataFrame, btc_df: pd.DataFrame,
                              eth_df: pd.DataFrame, config: Dict = None) -> pd.DataFrame:
    """Add cross-asset ratios, rolling correlations, beta, and lead/lag features."""
    if config is None:
        config = CROSS_ASSET_FEATURES
    df = df.copy()

    if btc_df.empty or eth_df.empty:
        logger.warning("BTC or ETH data empty — skipping cross-asset features")
        return df

    if len(df) < 2:
        return df

    freq = pd.infer_freq(df["timestamp"])
    if freq is None:
        freq = "1h"

    df = df.set_index("timestamp")

    btc_indexed = btc_df.set_index("timestamp")["close"]
    eth_indexed = eth_df.set_index("timestamp")["close"]

    df["btc_close"] = df.index.to_series().map(btc_indexed.asof)
    df["eth_close"] = df.index.to_series().map(eth_indexed.asof)

    df = df.reset_index()

    if config.get("ratio"):
        df["xrp_btc_ratio"] = df["close"] / df["btc_close"].replace(0, np.nan)
        df["xrp_eth_ratio"] = df["close"] / df["eth_close"].replace(0, np.nan)

    btc_ret = df["btc_close"].pct_change()
    eth_ret = df["eth_close"].pct_change()
    xrp_ret = df["close"].pct_change()

    if config.get("rolling_correlation"):
        for w in config.get("correlation_windows", [20, 60]):
            df[f"corr_xrp_btc_{w}"] = xrp_ret.rolling(window=w, min_periods=w).corr(btc_ret)
            df[f"corr_xrp_eth_{w}"] = xrp_ret.rolling(window=w, min_periods=w).corr(eth_ret)

    if config.get("market_beta"):
        bw = config.get("beta_window", 60)
        cov = xrp_ret.rolling(window=bw, min_periods=bw).cov(btc_ret)
        var_btc = btc_ret.rolling(window=bw, min_periods=bw).var()
        df[f"beta_xrp_btc_{bw}"] = cov / var_btc.replace(0, np.nan)

    df["btc_dominance"] = df["btc_close"] / (df["btc_close"] + df["eth_close"]).replace(0, np.nan)

    # BTC lead/lag features
    df = add_btc_lead_lag_features(df, config.get("btc_lead_lag", BTC_LEAD_LAG))
    return df


def add_btc_lead_lag_features(df: pd.DataFrame, config: Dict = None) -> pd.DataFrame:
    """Add BTC lead/lag features to capture cross-asset lead effects."""
    if config is None:
        config = BTC_LEAD_LAG
    if not config.get("enabled", True):
        return df

    if "btc_close" not in df.columns or df["btc_close"].dropna().empty:
        logger.warning("No BTC close data — skipping BTC lead/lag features")
        return df

    df = df.copy()
    btc_close = df["btc_close"]
    xrp_close = df["close"]

    # BTC returns
    btc_ret = btc_close.pct_change()
    xrp_ret = xrp_close.pct_change()

    # 1. BTC return leads
    for n in config.get("lead_bars", [1, 2, 4, 8]):
        df[f"btc_ret_lead_{n}"] = btc_ret.shift(-n)

    # 2. BTC momentum spillover
    for w in config.get("momentum_windows", [4, 8, 16, 24]):
        df[f"btc_momentum_{w}"] = btc_ret.rolling(window=w, min_periods=1).sum()

    # 3. BTC-XRP momentum ratio
    for w in config.get("ratio_windows", [4, 8, 16]):
        xrp_mom = xrp_ret.rolling(window=w, min_periods=1).sum()
        btc_mom = btc_ret.rolling(window=w, min_periods=1).sum()
        df[f"btc_xrp_mom_ratio_{w}"] = xrp_mom / btc_mom.replace(0, np.nan)

    # 4. BTC volatility regimes
    if config.get("vol_regimes", True):
        for w in config.get("vol_windows", [20, 60]):
            btc_vol = btc_ret.rolling(window=w, min_periods=10).std()
            df[f"btc_vol_{w}"] = btc_vol
            q25 = btc_vol.quantile(0.25)
            q75 = btc_vol.quantile(0.75)
            df[f"btc_vol_regime_{w}"] = np.select(
                [btc_vol < q25, btc_vol > q75],
                [0, 2],
                default=1
            )

    # 5. BTC trend strength
    if config.get("trend_strength", True):
        adx_period = config.get("adx_period", 14)
        df["btc_adx"] = _compute_adx(btc_close, adx_period)

    return df


def add_momentum_features(df: pd.DataFrame, config: Dict = None) -> pd.DataFrame:
    """Add momentum-based features for continuation and acceleration."""
    if config is None:
        config = MOMENTUM_TARGETS
    if not config.get("enabled", True):
        return df

    df = df.copy()
    ret = df["close"].pct_change()
    windows = config.get("momentum_windows", [4, 8, 16, 24])

    for w in windows:
        mom = ret.rolling(window=w, min_periods=1).sum()
        df[f"momentum_{w}"] = mom

        if config.get("acceleration", True):
            df[f"momentum_accel_{w}"] = mom.diff()

        if config.get("continuation", True):
            mom_mean = mom.rolling(window=config.get("continuation_window", 20),
                                    min_periods=1).mean()
            df[f"momentum_ratio_{w}"] = mom / mom_mean.replace(0, np.nan)

    return df


def add_vol_adj_momentum_features(df: pd.DataFrame, config: Dict = None) -> pd.DataFrame:
    """Add volatility-normalized momentum features."""
    if config is None:
        config = VOL_ADJ_MOMENTUM
    if not config.get("enabled", True):
        return df

    df = df.copy()
    ret = df["close"].pct_change()
    vol_window = config.get("vol_window", 20)
    windows = config.get("momentum_windows", [4, 8, 16, 24])

    rolling_vol = ret.rolling(window=vol_window, min_periods=10).std()

    for w in windows:
        mom = ret.rolling(window=w, min_periods=1).sum()
        df[f"vol_adj_momentum_{w}"] = mom / rolling_vol.replace(0, np.nan)

        if config.get("acceleration", True):
            df[f"vol_adj_accel_{w}"] = mom.diff() / rolling_vol.replace(0, np.nan)

    if config.get("regime_flags", True):
        q25 = rolling_vol.quantile(0.25)
        q75 = rolling_vol.quantile(0.75)
        df["vol_regime"] = np.select(
            [rolling_vol < q25, rolling_vol > q75],
            [0, 2],
            default=1
        )

    return df
