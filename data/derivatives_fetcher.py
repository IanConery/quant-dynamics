import time
import os
from typing import Dict, List, Optional, Tuple

import pandas as pd
import ccxt
from tqdm import tqdm

from config.settings import (
    DATA_DIR,
    FETCH_LIMIT,
    RATE_LIMIT_DELAY,
    RETRY_ATTEMPTS,
    RETRY_DELAY,
)
from utils.logger import setup_logger

logger = setup_logger(__name__)

DERIVATIVES_EXCHANGES = ["okx", "bybit", "binance", "gate"]
DERIVATIVES_SYMBOL = "XRP/USDT"


def _get_derivatives_exchange(exchange_id: Optional[str] = None) -> ccxt.Exchange:
    if exchange_id:
        exchange_cls = getattr(ccxt, exchange_id)
        inst = exchange_cls({"enableRateLimit": True})
        inst.load_markets()
        return inst
    for ex_id in DERIVATIVES_EXCHANGES:
        try:
            exchange_cls = getattr(ccxt, ex_id)
            inst = exchange_cls({"enableRateLimit": True})
            inst.load_markets()
            logger.info(f"Connected to {ex_id} for derivatives data")
            return inst
        except Exception as e:
            logger.warning(f"{ex_id} unavailable for derivatives: {str(e)[:80]}")
    raise RuntimeError("No derivatives exchange available")


def _get_derivatives_symbol(exchange: ccxt.Exchange, symbol: str) -> str:
    """Get the correct derivatives symbol for an exchange."""
    ex_name = exchange.name.lower()
    base, quote = symbol.split('/')

    # OKX uses XRP/USDT:USDT for perpetual swaps
    if ex_name == "okx":
        okx_sym = f"{base}/{quote}:{quote}"
        if okx_sym in exchange.markets:
            return okx_sym
    # Bybit uses XRP/USDT:USDT
    elif ex_name == "bybit":
        bybit_sym = f"{base}/{quote}:{quote}"
        if bybit_sym in exchange.markets:
            return bybit_sym
    # Binance uses XRP/USDT:USDT for perpetual
    elif ex_name == "binance":
        binance_sym = f"{base}/{quote}:{quote}"
        if binance_sym in exchange.markets:
            return binance_sym

    # Generic: try futures()
    try:
        return exchange.futures(symbol)
    except (AttributeError, ccxt.BaseError) as e:
        logger.debug(f"futures() lookup failed for {symbol}: {e}")

    # Try common variations
    variations = [
        f"{base}/{quote}:{quote}",
        f"{base}{quote}:{quote}",
        f"{base}{quote}:USDT",
        symbol,
    ]
    for variant in variations:
        if variant in exchange.markets:
            return variant

    logger.warning(f"Could not find derivatives symbol for {symbol} on {exchange.name}")
    return symbol


def fetch_funding_rates(
    symbol: str = DERIVATIVES_SYMBOL,
    timeframe: str = "8h",
    since: str = "2023-01-01",
    limit: int = 1000,
) -> pd.DataFrame:
    """Fetch funding rate history from derivatives exchanges.

    Funding rate interpretation:
    - Extreme positive = over-leveraged longs (contrarian bearish)
    - Extreme negative = over-leveraged shorts (contrarian bullish)

    Returns DataFrame with columns: timestamp, funding_rate, mark_price
    """
    exchange = _get_derivatives_exchange()
    all_rates: List[List] = []

    since_ts = exchange.parse8601(since)
    symbol_deriv = _get_derivatives_symbol(exchange, symbol)

    for _ in range(RETRY_ATTEMPTS):
        try:
            rates = exchange.fetch_funding_rate_history(symbol_deriv, since=since_ts, limit=limit)
            if rates:
                all_rates.extend(rates)
                if len(rates) < limit:
                    break
                since_ts = rates[-1]["timestamp"] + exchange.parse_timeframe(timeframe)
                time.sleep(RATE_LIMIT_DELAY)
            break
        except Exception as e:
            logger.warning(f"fetch_funding_rates error ({exchange.name}): {e}")
            time.sleep(RETRY_DELAY)

    if not all_rates:
        logger.warning(f"No funding rates from {exchange.name}")
        return pd.DataFrame(columns=["timestamp", "funding_rate", "mark_price"])

    df = pd.DataFrame(all_rates)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    cols = ["timestamp", "fundingRate"]
    if "markPrice" in df.columns:
        cols.append("markPrice")
    df = df[cols].rename(
        columns={"fundingRate": "funding_rate", "markPrice": "mark_price"}
    )
    df["funding_rate"] = pd.to_numeric(df["funding_rate"], errors="coerce")
    if "mark_price" in df.columns:
        df["mark_price"] = pd.to_numeric(df["mark_price"], errors="coerce")
    df = df.dropna(subset=["funding_rate"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Aggregate to daily if too granular
    if len(df) > 10000:
        df["date"] = df["timestamp"].dt.date
        agg_dict = {"funding_rate": "mean", "timestamp": "first"}
        if "mark_price" in df.columns:
            agg_dict["mark_price"] = "last"
        daily = df.groupby("date").agg(agg_dict).reset_index()
        daily["timestamp"] = pd.to_datetime(daily["date"], utc=True)
        daily = daily.drop(columns=["date"])
        df = daily

    logger.info(f"funding_rates({exchange.name}): {len(df)} records ({df['timestamp'].min()} -> {df['timestamp'].max()})")
    return df


def fetch_open_interest(
    symbol: str = DERIVATIVES_SYMBOL,
    since: str = "2023-01-01",
    limit: int = 1000,
) -> pd.DataFrame:
    """Fetch open interest history.

    OI interpretation:
    - Rising OI + rising price = strong trend
    - Rising OI + falling price = strong bearish pressure
    - Falling OI = position unwinding

    Returns DataFrame with columns: timestamp, open_interest, open_interest_value
    """
    exchange = _get_derivatives_exchange()
    all_oi: List[List] = []

    since_ts = exchange.parse8601(since)

    try:
        symbol_deriv = exchange.futures(symbol) if hasattr(exchange, 'futures') else symbol
    except Exception:
        symbol_deriv = symbol

    # Try fetch_ohlcv on OI if supported, otherwise use ticker snapshots
    for _ in range(RETRY_ATTEMPTS):
        try:
            oi = exchange.fetch_ohlcv(symbol_deriv, timeframe="1d", since=since_ts, limit=limit)
            if oi:
                all_oi.extend(oi)
                if len(oi) < limit:
                    break
                since_ts = oi[-1][0] + exchange.parse_timeframe("1d")
                time.sleep(RATE_LIMIT_DELAY)
            break
        except Exception as e:
            logger.warning(f"OI fetch attempt failed: {e}")
            time.sleep(RETRY_DELAY)

    if not all_oi:
        logger.warning(f"No OI history from {exchange.name}, using ticker snapshots")
        ticker = exchange.fetch_ticker(symbol_deriv)
        df = pd.DataFrame([{
            "timestamp": pd.to_datetime(ticker.get("timestamp", 0), unit="ms", utc=True),
            "open_interest": ticker.get("openInterestAmount", ticker.get("interest", 0)),
            "open_interest_value": ticker.get("openInterestValue", 0),
        }])
        return df

    df = pd.DataFrame(all_oi, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

    # For exchanges without dedicated OI endpoint, use volume as proxy
    df["open_interest"] = df["volume"]
    df["open_interest_value"] = df["volume"] * df["close"]
    df = df[["timestamp", "open_interest", "open_interest_value"]]
    df = df.sort_values("timestamp").reset_index(drop=True)

    logger.info(f"open_interest({exchange.name}): {len(df)} records")
    return df


def fetch_liquidations(
    symbol: str = DERIVATIVES_SYMBOL,
    timeframe: str = "1d",
    since: str = "2023-01-01",
) -> pd.DataFrame:
    """Fetch liquidation data from exchange public API.

    Returns DataFrame with columns: timestamp, long_liq, short_liq, total_liq
    """
    exchange = _get_derivatives_exchange()

    # Most ccxt exchanges don't have dedicated liquidation history endpoint
    # Use a proxy: fetch trades with high volume as potential liquidation clusters
    all_trades = []
    since_ts = exchange.parse8601(since)

    try:
        symbol_deriv = exchange.futures(symbol) if hasattr(exchange, 'futures') else symbol
    except Exception:
        symbol_deriv = symbol

    for _ in range(RETRY_ATTEMPTS):
        try:
            trades = exchange.fetch_trades(symbol_deriv, since=since_ts, limit=min(500, FETCH_LIMIT))
            if trades:
                all_trades.extend(trades)
                if len(trades) < 500:
                    break
                since_ts = trades[-1]["timestamp"] + exchange.parse_timeframe(timeframe)
                time.sleep(RATE_LIMIT_DELAY)
            break
        except Exception as e:
            logger.warning(f"fetch_liquidations error ({exchange.name}): {e}")
            time.sleep(RETRY_DELAY)

    if not all_trades:
        return pd.DataFrame(columns=["timestamp", "long_liq", "short_liq", "total_liq"])

    df = pd.DataFrame(all_trades)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["size"] = pd.to_numeric(df.get("size", df.get("amount", 0)), errors="coerce")
    df["price"] = pd.to_numeric(df.get("price", 0), errors="coerce")

    # Group by day and compute liquidation proxy metrics
    df["date"] = df["timestamp"].dt.floor("D")
    daily = df.groupby("date").agg({
        "size": ["sum", "max", "count"],
        "price": "mean",
    }).reset_index()
    daily.columns = ["timestamp", "total_liq", "max_liq_size", "liq_count", "avg_price"]
    daily["long_liq"] = daily["total_liq"] * 0.5
    daily["short_liq"] = daily["total_liq"] * 0.5
    daily = daily[["timestamp", "long_liq", "short_liq", "total_liq", "max_liq_size", "liq_count"]]
    daily = daily.sort_values("timestamp").reset_index(drop=True)

    logger.info(f"liquidations({exchange.name}): {len(daily)} daily records")
    return daily


def save_derivatives_data(data_dict: Dict[str, pd.DataFrame]) -> None:
    """Save derivatives data to parquet files."""
    deriv_dir = os.path.join(DATA_DIR, "derivatives")
    os.makedirs(deriv_dir, exist_ok=True)
    for name, df in data_dict.items():
        path = os.path.join(deriv_dir, f"{name}.parquet")
        df.to_parquet(path, index=False)
        logger.info(f"Saved {path} ({len(df)} rows)")


def load_derivatives_data(name: str) -> pd.DataFrame:
    """Load derivatives data from parquet file."""
    path = os.path.join(DATA_DIR, "derivatives", f"{name}.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Derivatives data not found: {path}")
    return pd.read_parquet(path)


def fetch_all_derivatives(since: str = "2023-01-01") -> Dict[str, pd.DataFrame]:
    """Fetch all derivatives data sources."""
    logger.info("=" * 60)
    logger.info("Starting derivatives data fetch")
    logger.info("=" * 60)

    results = {}

    logger.info("--- Funding rates ---")
    try:
        results["funding_rates"] = fetch_funding_rates(since=since)
        time.sleep(RATE_LIMIT_DELAY)
    except Exception as e:
        logger.error(f"Funding rates fetch failed: {e}")

    logger.info("--- Open interest ---")
    try:
        results["open_interest"] = fetch_open_interest(since=since)
        time.sleep(RATE_LIMIT_DELAY)
    except Exception as e:
        logger.error(f"Open interest fetch failed: {e}")

    logger.info("--- Liquidations ---")
    try:
        results["liquidations"] = fetch_liquidations(since=since)
    except Exception as e:
        logger.error(f"Liquidations fetch failed: {e}")

    if results:
        save_derivatives_data(results)

    logger.info("=" * 60)
    logger.info("Derivatives data fetch complete!")
    logger.info("=" * 60)
    return results
