import time
import os
import io
import zipfile
from typing import Dict, List, Optional, Tuple

import pandas as pd
import ccxt
from tqdm import tqdm

from config.settings import (
    EXCHANGE,
    SYMBOL,
    CROSS_ASSET_SYMBOLS,
    EXTERNAL_DATA,
    INTERVALS,
    START_DATE,
    DATA_DIR,
    FETCH_LIMIT,
    RATE_LIMIT_DELAY,
    RETRY_ATTEMPTS,
    RETRY_DELAY,
    BINANCE_VISION,
)
from utils.logger import setup_logger

logger = setup_logger(__name__)

EXCHANGE_FALLBACKS = ["kraken", "coinbase", "bybit", "okx"]


def _get_exchange(exchange_id: Optional[str] = None) -> ccxt.Exchange:
    if exchange_id:
        exchange_cls = getattr(ccxt, exchange_id)
        return exchange_cls({"enableRateLimit": True})
    for ex_id in EXCHANGE_FALLBACKS:
        try:
            exchange_cls = getattr(ccxt, ex_id)
            inst = exchange_cls({"enableRateLimit": True})
            inst.load_markets()
            logger.info(f"Connected to {ex_id}")
            return inst
        except Exception as e:
            logger.warning(f"{ex_id} unavailable: {str(e)[:80]}")
    raise RuntimeError("No exchange available from fallback list")


def _fetch_yfinance(symbol: str, interval: str, start: str = START_DATE) -> pd.DataFrame:
    import yfinance as yf
    yf_symbol = symbol.replace("/", "-").replace("USDT", "USD")
    ticker = yf.Ticker(yf_symbol)
    hist = ticker.history(period="max", interval=interval, auto_adjust=False)
    if hist.empty:
        logger.warning(f"yfinance returned empty for {yf_symbol} {interval}")
        return pd.DataFrame()
    df = hist.reset_index()
    ts_col = "Datetime" if "Datetime" in df.columns else "Date"
    df = df.rename(columns={
        ts_col: "timestamp", "Open": "open", "High": "high",
        "Low": "low", "Close": "close", "Volume": "volume",
    })
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    df = df.sort_values("timestamp").reset_index(drop=True)
    logger.info(f"yfinance {symbol} {interval}: {len(df)} candles")
    return df


def _fetch_ccxt(symbol: str, timeframe: str, start: str = START_DATE, limit: int = 720, since: Optional[int] = None) -> pd.DataFrame:
    exchange_instance = _get_exchange()
    if since is not None:
        since_ts = since
    else:
        since_ts = exchange_instance.parse8601(start)
    all_candles: List[List] = []
    since = since_ts
    while True:
        candles = None
        attempt = 0
        while attempt < RETRY_ATTEMPTS:
            try:
                candles = exchange_instance.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
                break
            except Exception as e:
                attempt += 1
                if attempt < RETRY_ATTEMPTS:
                    time.sleep(RETRY_DELAY * (2 ** attempt))
                else:
                    logger.error(f"Failed {symbol} {timeframe}: {e}")
                    raise
        if not candles:
            break
        all_candles.extend(candles)
        since = candles[-1][0] + exchange_instance.parse_timeframe(timeframe)
        if len(candles) < limit:
            break
        time.sleep(RATE_LIMIT_DELAY)
        if len(all_candles) >= 200000:
            break
    df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates(subset=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    logger.info(f"ccxt({exchange_instance.name}) {symbol} {timeframe}: {len(df)} candles")
    return df


def _merge_dfs(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    if not dfs:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    merged = pd.concat(dfs, ignore_index=True)
    merged = merged.drop_duplicates(subset=["timestamp"])
    merged = merged.sort_values("timestamp").reset_index(drop=True)
    return merged


def fetch_ohlcv(symbol: str = SYMBOL, timeframe: str = "1h", since: str = START_DATE) -> pd.DataFrame:
    """Fetch OHLCV from multiple sources, merge for maximum coverage."""
    source_dfs: List[pd.DataFrame] = []
    logger.info(f"Fetching {symbol} {timeframe} from multiple sources...")
    try:
        ccxt_df = _fetch_ccxt(symbol, timeframe, since)
        if not ccxt_df.empty:
            source_dfs.append(ccxt_df)
    except Exception as e:
        logger.warning(f"ccxt failed for {symbol} {timeframe}: {e}")
    try:
        yf_df = _fetch_yfinance(symbol, timeframe, since)
        if not yf_df.empty:
            source_dfs.append(yf_df)
    except Exception as e:
        logger.warning(f"yfinance failed for {symbol} {timeframe}: {e}")
    if not source_dfs:
        raise RuntimeError(f"No source returned data for {symbol} {timeframe}")
    merged = _merge_dfs(source_dfs)
    logger.info(f"Merged {symbol} {timeframe}: {len(merged)} candles ({merged['timestamp'].min()} -> {merged['timestamp'].max()})")
    return merged


def fetch_cross_assets(intervals: Optional[List[str]] = None) -> Dict[str, Dict[str, pd.DataFrame]]:
    if intervals is None:
        intervals = INTERVALS
    results: Dict[str, Dict[str, pd.DataFrame]] = {}
    for symbol in CROSS_ASSET_SYMBOLS:
        sym_key = symbol.replace("/", "-")
        results[sym_key] = {}
        for interval in tqdm(intervals, desc=f"Fetching {sym_key}"):
            df = fetch_ohlcv(symbol=symbol, timeframe=interval)
            results[sym_key][interval] = df
            save_raw_data({interval: df}, symbol=sym_key)
            time.sleep(RATE_LIMIT_DELAY)
    return results


def fetch_all_intervals(intervals: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
    if intervals is None:
        intervals = INTERVALS
    results: Dict[str, pd.DataFrame] = {}
    for interval in tqdm(intervals, desc="Fetching primary symbol"):
        results[interval] = fetch_ohlcv(timeframe=interval)
        time.sleep(RATE_LIMIT_DELAY)
    save_raw_data(results, symbol="XRP-USDT")
    return results


def fetch_external_data() -> Dict[str, pd.DataFrame]:
    results: Dict[str, pd.DataFrame] = {}
    ext_dir = os.path.join(DATA_DIR, "external")
    os.makedirs(ext_dir, exist_ok=True)
    for source_name, source_cfg in EXTERNAL_DATA.items():
        import requests as req
        attempt = 0
        resp = None
        while attempt < RETRY_ATTEMPTS:
            try:
                resp = req.get(source_cfg["api_url"], timeout=30)
                resp.raise_for_status()
                break
            except Exception as e:
                attempt += 1
                if attempt < RETRY_ATTEMPTS:
                    time.sleep(RETRY_DELAY * (2 ** attempt))
                else:
                    logger.error(f"Failed to fetch {source_name}: {e}")
                    raise
        data = resp.json().get("data", [])
        df = pd.DataFrame(data)
        ts_col = source_cfg["timestamp_column"]
        val_col = source_cfg["value_column"]
        ts_vals = df[ts_col].astype(int)
        df["timestamp"] = pd.to_datetime(ts_vals, unit="s", utc=True)
        df["value"] = df[val_col].astype(float)
        df = df[["timestamp", "value"]].sort_values("timestamp").reset_index(drop=True)
        results[source_name] = df
        df.to_parquet(os.path.join(ext_dir, f"{source_name}.parquet"), index=False)
        logger.info(f"Fetched {source_name}: {len(df)} records")
    return results


def run_full_fetch(intervals: Optional[List[str]] = None, fetch_binance: bool = False) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    logger.info("=" * 60)
    logger.info("Starting full data fetch")
    if fetch_binance:
        logger.info("  Binance Vision backfill ENABLED")
    logger.info("=" * 60)

    # Step 1: Binance Vision historical backfill (if requested)
    if fetch_binance:
        logger.info("--- Step 1: Binance Vision historical download ---")
        fetch_all_binance_historical(intervals)
        logger.info("--- Step 2: Merge Binance with Kraken ---")
        merge_all_binance(intervals)
    else:
        # Step 1: Primary symbol
        logger.info("--- Primary symbol (XRP/USDT) ---")
        fetch_all_intervals(intervals)
        # Step 2: Cross-asset symbols
        logger.info("--- Cross-asset symbols (BTC/ETH) ---")
        fetch_cross_assets(intervals)

    # Step 3: External data
    logger.info("--- External data (Fear & Greed Index) ---")
    fetch_external_data()

    logger.info("=" * 60)
    logger.info("Data fetch complete!")
    logger.info("=" * 60)


def save_raw_data(data_dict: Dict[str, pd.DataFrame], symbol: str = "XRP-USDT") -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    for interval, df in data_dict.items():
        path = os.path.join(DATA_DIR, f"{symbol}_{interval}.parquet")
        df.to_parquet(path, index=False)
        logger.info(f"Saved {path} ({len(df)} rows)")


def load_raw_data(interval: str, symbol: str = "XRP-USDT") -> pd.DataFrame:
    path = os.path.join(DATA_DIR, f"{symbol}_{interval}.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Raw data not found: {path}. Run 'data fetch' first.")
    return pd.read_parquet(path)


def load_cross_asset_data(symbol: str, interval: str) -> pd.DataFrame:
    sym_key = symbol.replace("/", "-")
    return load_raw_data(interval, symbol=sym_key)


def load_external_data(source: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "external", f"{source}.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"External data not found: {path}. Run 'data fetch' first.")
    return pd.read_parquet(path)


def refresh_data(
    interval: str,
    symbol: str = "XRP-USDT",
    since_offset: int = 0,
) -> pd.DataFrame:
    """
    Fetch only new candles since the last saved timestamp.

    Args:
        interval: Data interval (15m, 1h, 1d)
        symbol: Symbol to refresh
        since_offset: Extra seconds to look back (avoids missed candles at boundary)

    Returns:
        Only the newly fetched candles (not merged yet).
    """
    path = os.path.join(DATA_DIR, f"{symbol}_{interval}.parquet")
    existing_df = None
    if os.path.exists(path):
        existing_df = pd.read_parquet(path)
        last_ts = existing_df["timestamp"].max()
        since_ts = (last_ts - pd.Timedelta(seconds=since_offset)).timestamp() * 1000
        logger.info(f"Refreshing {symbol}/{interval}: last={last_ts.isoformat()}, fetching from {since_ts}")
    else:
        logger.warning(f"No existing data for {symbol}/{interval}, doing full fetch")
        return fetch_ohlcv(symbol.replace("-", "/"), timeframe=interval)

    try:
        new_df = _fetch_ccxt(
            symbol.replace("-", "/"),
            timeframe=interval,
            since=int(since_ts),
            limit=FETCH_LIMIT,
        )
        if new_df.empty:
            logger.info(f"No new data for {symbol}/{interval}")
            return pd.DataFrame()

        new_df = new_df[new_df["timestamp"] > existing_df["timestamp"].max()]
        if new_df.empty:
            logger.info(f"No genuinely new candles for {symbol}/{interval}")
            return new_df

        merged = pd.concat([existing_df, new_df], ignore_index=True)
        merged = merged.drop_duplicates(subset=["timestamp"])
        merged = merged.sort_values("timestamp").reset_index(drop=True)
        save_raw_data({interval: merged}, symbol=symbol)
        logger.info(f"Refreshed {symbol}/{interval}: {len(existing_df)} -> {len(merged)} rows ({len(new_df)} new)")
        return new_df
    except Exception as e:
        logger.error(f"Failed to refresh {symbol}/{interval}: {e}")
        raise


# ─── Binance Vision Historical Data ──────────────────────────────────────────


def _get_binance_vision_url(symbol: str, interval: str, year: int, month: int) -> str:
    base = BINANCE_VISION["base_url"]
    return f"{base}/{symbol}/{interval}/{symbol}-{interval}-{year}-{month:02d}.zip"


def _download_zip(url: str, max_retries: int = 3, retry_delay: int = 2) -> Optional[bytes]:
    import requests as req
    for attempt in range(max_retries):
        try:
            resp = req.get(url, timeout=60, stream=True)
            if resp.status_code == 200:
                return resp.content
            elif resp.status_code == 404:
                return None
            else:
                resp.raise_for_status()
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (2 ** attempt))
            else:
                logger.warning(f"Failed to download {url}: {e}")
                return None
    return None


def _parse_binance_klines_csv(zip_data: bytes) -> pd.DataFrame:
    """Parse Binance klines CSV from zip data.

    Binance klines CSV columns (from ccxt /api/v3/klines):
    0: Open time
    1: Open
    2: High
    3: Low
    4: Close
    5: Volume
    6: Close time
    7: Quote asset volume
    8: Number of trades
    9: Taker buy base asset volume
    10: Taker buy quote asset volume
    11: Ignore

    Timestamp format: milliseconds pre-2025, microseconds from 2025-01-01 onwards.
    """
    with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
        csv_name = zf.namelist()[0]
        with zf.open(csv_name) as f:
            df = pd.read_csv(f, header=None)

    df = df.iloc[:, :11]
    df.columns = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trade_count",
        "taker_buy_base_volume", "taker_buy_quote_volume",
    ]

    for c in ["open", "high", "low", "close", "volume", "quote_volume",
               "taker_buy_base_volume", "taker_buy_quote_volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["trade_count"] = pd.to_numeric(df["trade_count"], errors="coerce").astype(int)

    # Handle timestamp: microseconds from 2025-01-01, milliseconds before
    timestamps = pd.to_numeric(df["open_time"], errors="coerce")
    if timestamps.max() > 1e15:
        # Microseconds
        df["timestamp"] = pd.to_datetime(timestamps, unit="us", utc=True)
    else:
        # Milliseconds
        df["timestamp"] = pd.to_datetime(timestamps, unit="ms", utc=True)

    df = df.drop(columns=["open_time", "close_time"])
    df = df.dropna(subset=["timestamp", "open", "high", "low", "close", "volume"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def fetch_binance_historical(
    symbol: str,
    interval: str,
    start_year: int = 2019,
    end_year: int = 2026,
    end_month: int = 7,
    download_dir: str = None,
) -> pd.DataFrame:
    """Download monthly kline files from Binance Vision.

    Args:
        symbol: e.g. XRPUSDT
        interval: e.g. 15m, 1h, 4h, 1d
        start_year: First year to download (default 2019)
        end_year: Last year to download (default 2026)
        end_month: Last month to download (default 7 = July)
        download_dir: Output directory for parquet files

    Returns:
        Combined DataFrame of all downloaded months.
    """
    if download_dir is None:
        download_dir = BINANCE_VISION.get("download_dir", "artifacts/raw_data/binance")
    os.makedirs(download_dir, exist_ok=True)

    max_retries = BINANCE_VISION.get("max_retries", 3)
    retry_delay = BINANCE_VISION.get("retry_delay", 2)
    rate_delay = BINANCE_VISION.get("rate_limit_delay", 0.5)

    all_months: List[pd.DataFrame] = []
    total_months = (end_year - start_year + 1) * 12
    # Cap end_month for the final year
    end_month = min(end_month, 12)

    logger.info(f"Binance Vision: downloading {symbol} {interval} ({start_year} → {end_year}-{end_month:02d})")

    for year in range(start_year, end_year + 1):
        month_range = range(1, 13) if year < end_year else range(1, end_month + 1)
        for month in month_range:
            url = _get_binance_vision_url(symbol, interval, year, month)
            zip_data = _download_zip(url, max_retries, retry_delay)
            if zip_data is None:
                continue
            try:
                df = _parse_binance_klines_csv(zip_data)
                if not df.empty:
                    all_months.append(df)
                    logger.info(f"  {year}-{month:02d}: {len(df)} candles")
            except Exception as e:
                logger.warning(f"  Failed to parse {year}-{month:02d}: {e}")
            time.sleep(rate_delay)

    if not all_months:
        logger.warning(f"No data downloaded for {symbol} {interval}")
        return pd.DataFrame()

    combined = pd.concat(all_months, ignore_index=True)
    combined = combined.drop_duplicates(subset=["timestamp"])
    combined = combined.sort_values("timestamp").reset_index(drop=True)

    out_path = os.path.join(download_dir, f"{symbol}_{interval}.parquet")
    combined.to_parquet(out_path, index=False)
    logger.info(
        f"Binance Vision {symbol} {interval}: {len(combined)} candles "
        f"({combined['timestamp'].min().isoformat()} → {combined['timestamp'].max().isoformat()})"
    )
    return combined


def fetch_all_binance_historical(intervals: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
    """Download Binance Vision klines for all configured symbols and intervals.

    Returns dict of {symbol: combined_df} for each symbol.
    """
    if intervals is None:
        intervals = BINANCE_VISION.get("intervals", ["15m", "1h", "4h", "1d"])

    all_results: Dict[str, Dict[str, pd.DataFrame]] = {}
    symbols = BINANCE_VISION.get("symbols", {})

    for symbol, cfg in tqdm(symbols.items(), desc="Binance Vision symbols"):
        start_year = cfg.get("start_year", 2019)
        all_results[symbol] = {}
        for interval in tqdm(intervals, desc=f"  {symbol} intervals", leave=False):
            df = fetch_binance_historical(symbol, interval, start_year=start_year)
            if not df.empty:
                all_results[symbol][interval] = df
            time.sleep(BINANCE_VISION.get("rate_limit_delay", 0.5))

    return all_results


def _merge_binance_kraken(kraken_df: pd.DataFrame, binance_df: pd.DataFrame) -> pd.DataFrame:
    """Merge Kraken and Binance data. For overlapping periods, prefer Binance.

    Binance data has additional columns (taker_buy_base_volume, etc.) that Kraken lacks.
    """
    if binance_df.empty:
        return kraken_df
    if kraken_df.empty:
        return binance_df

    # Standardize column names
    standard_cols = ["timestamp", "open", "high", "low", "close", "volume"]
    extra_cols = ["quote_volume", "trade_count", "taker_buy_base_volume", "taker_buy_quote_volume"]

    for df in [kraken_df, binance_df]:
        for col in standard_cols:
            if col not in df.columns:
                df[col] = 0.0

    # Find overlap range
    kraken_end = kraken_df["timestamp"].max()
    binance_start = binance_df["timestamp"].min()
    binance_end = binance_df["timestamp"].max()

    # Non-overlapping Kraken data (before Binance starts)
    pre_binance = kraken_df[kraken_df["timestamp"] < binance_start][standard_cols].copy()
    # Binance data (full — preferred in overlap)
    binance_standard = binance_df[standard_cols].copy()
    # Kraken data after Binance ends (rare, but handle it)
    post_binance = kraken_df[kraken_df["timestamp"] > binance_end][standard_cols].copy()

    # Combine
    merged = pd.concat([pre_binance, binance_standard, post_binance], ignore_index=True)
    merged = merged.drop_duplicates(subset=["timestamp"])
    merged = merged.sort_values("timestamp").reset_index(drop=True)

    # Add extra columns from Binance (NaN where no Binance data)
    for col in extra_cols:
        if col in binance_df.columns:
            if col not in merged.columns:
                merged[col] = float("nan")
            # Fill from Binance where timestamps match
            bseries = binance_df.set_index("timestamp")[col]
            merged[col] = merged["timestamp"].map(bseries).fillna(merged.get(col, float("nan")))
        else:
            if col not in merged.columns:
                merged[col] = float("nan")

    return merged


def merge_binance_with_kraken(binance_symbol: str, interval: str, internal_symbol: str) -> pd.DataFrame:
    """Merge Binance Vision data with existing Kraken data.

    Args:
        binance_symbol: Binance-format symbol (e.g., XRPUSDT)
        interval: Data interval (e.g., 15m, 1h)
        internal_symbol: Internal-format symbol (e.g., XRP-USDT)

    Saves merged result to artifacts/raw_data/{internal_symbol}_{interval}.parquet.
    """
    binance_dir = BINANCE_VISION.get("download_dir", "artifacts/raw_data/binance")
    binance_path = os.path.join(binance_dir, f"{binance_symbol}_{interval}.parquet")
    kraken_path = os.path.join(DATA_DIR, f"{internal_symbol}_{interval}.parquet")

    # Load Kraken data
    kraken_df = pd.DataFrame()
    if os.path.exists(kraken_path):
        kraken_df = pd.read_parquet(kraken_path)
        logger.info(f"  Kraken {internal_symbol}/{interval}: {len(kraken_df)} rows")
    else:
        logger.info(f"  No existing Kraken data for {internal_symbol}/{interval}")

    # Load Binance data
    if not os.path.exists(binance_path):
        logger.warning(f"  No Binance data found: {binance_path}")
        return kraken_df

    binance_df = pd.read_parquet(binance_path)
    logger.info(f"  Binance {binance_symbol}/{interval}: {len(binance_df)} rows")

    # Merge
    merged = _merge_binance_kraken(kraken_df, binance_df)

    # Save to main DATA_DIR with internal symbol naming
    os.makedirs(DATA_DIR, exist_ok=True)
    merged.to_parquet(kraken_path, index=False)
    logger.info(
        f"  Merged {internal_symbol}/{interval}: {len(kraken_df)} + {len(binance_df)} → {len(merged)} rows "
        f"({merged['timestamp'].min().isoformat()} → {merged['timestamp'].max().isoformat()})"
    )
    return merged


def merge_all_binance(intervals: Optional[List[str]] = None) -> None:
    """Merge Binance Vision data with Kraken for all symbols and intervals."""
    if intervals is None:
        intervals = BINANCE_VISION.get("intervals", ["15m", "1h", "4h", "1d"])

    # Map Binance symbols to internal naming
    symbol_map = {
        "XRPUSDT": "XRP-USDT",
        "BTCUSDT": "BTC-USDT",
        "ETHUSDT": "ETH-USDT",
    }

    for binance_symbol, internal_symbol in symbol_map.items():
        for interval in intervals:
            try:
                merge_binance_with_kraken(binance_symbol, interval, internal_symbol)
            except Exception as e:
                logger.error(f"  Failed to merge {binance_symbol}/{interval}: {e}")


def refresh_all(intervals: Optional[List[str]] = None) -> None:
    """Incrementally refresh all symbols and intervals with new data."""
    if intervals is None:
        intervals = INTERVALS
    os.makedirs(DATA_DIR, exist_ok=True)
    logger.info("=" * 60)
    logger.info("Starting incremental data refresh")
    logger.info("=" * 60)

    # Primary symbol
    logger.info("--- Primary symbol (XRP/USDT) ---")
    for interval in tqdm(intervals, desc="Refreshing XRP-USDT"):
        refresh_data(interval, symbol="XRP-USDT", since_offset=60)
        time.sleep(RATE_LIMIT_DELAY)

    # Cross-asset symbols
    logger.info("--- Cross-asset symbols (BTC/ETH) ---")
    for sym_key in [s.replace("/", "-") for s in CROSS_ASSET_SYMBOLS]:
        for interval in tqdm(intervals, desc=f"Refreshing {sym_key}"):
            refresh_data(interval, symbol=sym_key, since_offset=60)
            time.sleep(RATE_LIMIT_DELAY)

    # External data (always re-fetch, it's small)
    logger.info("--- External data (Fear & Greed Index) ---")
    fetch_external_data()

    logger.info("=" * 60)
    logger.info("Data refresh complete!")
    logger.info("=" * 60)
