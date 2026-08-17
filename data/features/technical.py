"""Technical indicator feature engineering."""

from typing import Dict, List
import numpy as np
import pandas as pd

from config.settings import INDICATORS


def add_sma(df: pd.DataFrame, close_col: str = "close", windows: List[int] = None) -> pd.DataFrame:
    """Add Simple Moving Averages."""
    if windows is None:
        windows = INDICATORS["sma"]
    for w in windows:
        df[f"sma_{w}"] = df[close_col].rolling(window=w, min_periods=w).mean()
    return df


def add_ema(df: pd.DataFrame, close_col: str = "close", windows: List[int] = None) -> pd.DataFrame:
    """Add Exponential Moving Averages."""
    if windows is None:
        windows = INDICATORS["ema"]
    for w in windows:
        df[f"ema_{w}"] = df[close_col].ewm(span=w, adjust=False).mean()
    return df


def add_rsi(df: pd.DataFrame, close_col: str = "close", periods: List[int] = None) -> pd.DataFrame:
    """Add Relative Strength Index."""
    if periods is None:
        periods = INDICATORS["rsi"]
    delta = df[close_col].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    for p in periods:
        avg_gain = gain.ewm(alpha=1 / p, min_periods=p, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / p, min_periods=p, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df[f"rsi_{p}"] = 100 - (100 / (1 + rs))
    return df


def add_macd(df: pd.DataFrame, close_col: str = "close", **kwargs: Dict) -> pd.DataFrame:
    """Add Moving Average Convergence Divergence."""
    cfg = kwargs or INDICATORS["macd"]
    fast = cfg.get("fast", 12)
    slow = cfg.get("slow", 26)
    signal = cfg.get("signal", 9)
    ema_fast = df[close_col].ewm(span=fast, adjust=False).mean()
    ema_slow = df[close_col].ewm(span=slow, adjust=False).mean()
    df["macd"] = ema_fast - ema_slow
    df["macd_signal"] = df["macd"].ewm(span=signal, adjust=False).mean()
    df["macd_histogram"] = df["macd"] - df["macd_signal"]
    return df


def add_bollinger(df: pd.DataFrame, close_col: str = "close", **kwargs: Dict) -> pd.DataFrame:
    """Add Bollinger Bands."""
    cfg = kwargs or INDICATORS["bollinger"]
    period = cfg.get("period", 20)
    std = cfg.get("std", 2)
    sma = df[close_col].rolling(window=period, min_periods=period).mean()
    std_dev = df[close_col].rolling(window=period, min_periods=period).std()
    df["bb_upper"] = sma + std * std_dev
    df["bb_lower"] = sma - std * std_dev
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / sma
    df["bb_percent"] = (df[close_col] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)
    return df


def add_atr(df: pd.DataFrame, high_col: str = "high", low_col: str = "low",
            close_col: str = "close", periods: List[int] = None) -> pd.DataFrame:
    """Add Average True Range."""
    if periods is None:
        periods = INDICATORS["atr"]
    prev_close = df[close_col].shift(1)
    tr1 = df[high_col] - df[low_col]
    tr2 = (df[high_col] - prev_close).abs()
    tr3 = (df[low_col] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    for p in periods:
        df[f"atr_{p}"] = tr.ewm(alpha=1 / p, min_periods=p, adjust=False).mean()
    return df


def add_stochastic(df: pd.DataFrame, **kwargs: Dict) -> pd.DataFrame:
    """Add Stochastic Oscillator."""
    cfg = kwargs or INDICATORS["stochastic"]
    k_period = cfg.get("k", 14)
    d_period = cfg.get("d", 3)
    lowest_low = df["low"].rolling(window=k_period, min_periods=k_period).min()
    highest_high = df["high"].rolling(window=k_period, min_periods=k_period).max()
    df["stoch_k"] = 100 * (df["close"] - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
    df["stoch_d"] = df["stoch_k"].rolling(window=d_period, min_periods=d_period).mean()
    return df


def add_adx(df: pd.DataFrame, **kwargs: Dict) -> pd.DataFrame:
    """Add Average Directional Index."""
    period = (kwargs or {}).get("period", 14)
    high_diff = df["high"].diff()
    low_diff = -df["low"].diff()
    plus_dm = ((high_diff > low_diff) & (high_diff > 0)).astype(float) * high_diff
    minus_dm = ((low_diff > high_diff) & (low_diff > 0)).astype(float) * low_diff
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df["adx"] = dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    df["plus_di"] = plus_di
    df["minus_di"] = minus_di
    return df


def add_obv(df: pd.DataFrame, close_col: str = "close", volume_col: str = "volume") -> pd.DataFrame:
    """Add On-Balance Volume."""
    close_diff = df[close_col].diff()
    direction = close_diff.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    df["obv"] = (direction * df[volume_col]).cumsum()
    return df


def add_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """Add Volume Weighted Average Price."""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    df["vwap"] = (tp * df["volume"]).cumsum() / df["volume"].cumsum().replace(0, np.nan)
    return df


def add_volume_sma(df: pd.DataFrame, volume_col: str = "volume",
                    windows: List[int] = None) -> pd.DataFrame:
    """Add Volume SMAs and volume ratio."""
    if windows is None:
        windows = INDICATORS["volume_sma"]
    for w in windows:
        df[f"volume_sma_{w}"] = df[volume_col].rolling(window=w, min_periods=w).mean()
    df["volume_ratio"] = df[volume_col] / df[f"volume_sma_{windows[-1]}"].replace(0, np.nan)
    return df


def add_lag_features(df: pd.DataFrame, close_col: str = "close",
                      lags: List[int] = None) -> pd.DataFrame:
    """Add price lags and returns."""
    if lags is None:
        lags = [1, 2, 3, 5, 10, 20]
    for lag in lags:
        df[f"close_lag_{lag}"] = df[close_col].shift(lag)
        df[f"return_{lag}"] = df[close_col].pct_change(lag)
    return df


def add_price_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Add candlestick body/wick geometry features."""
    df["body_size"] = (df["close"] - df["open"]).abs() / df["open"]
    df["upper_wick"] = (df["high"] - df[["open", "close"]].max(axis=1)) / df["open"]
    df["lower_wick"] = (df[["open", "close"]].min(axis=1) - df["low"]) / df["open"]
    df["is_bullish"] = (df["close"] > df["open"]).astype(int)
    df["body_ratio"] = df["body_size"] / (df["high"] - df["low"]).replace(0, np.nan)
    return df


def _compute_adx(close: pd.Series, period: int = 14) -> pd.Series:
    """Compute ADX (Average Directional Index) approximation for a single price series."""
    tr = close.diff().abs().rolling(window=period, min_periods=period).mean().replace(0, np.nan)
    plus_dm = close.diff().clip(lower=0)
    minus_dm = (-close.diff()).clip(lower=0)

    plus_di = 100 * plus_dm.rolling(window=period, min_periods=period).mean() / tr
    minus_di = 100 * minus_dm.rolling(window=period, min_periods=period).mean() / tr

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.rolling(window=period, min_periods=period).mean()
    return adx
