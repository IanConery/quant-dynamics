import time
import os
from typing import Dict, Optional

import pandas as pd
import numpy as np

from config.settings import DATA_DIR, RETRY_ATTEMPTS, RETRY_DELAY
from utils.logger import setup_logger

logger = setup_logger(__name__)


def fetch_on_chain_data() -> Dict[str, pd.DataFrame]:
    """Fetch on-chain data from free APIs.

    Sources:
    - Blockchain.com API: exchange net flow, active addresses
    - CoinMarketCap/alternative.me: crypto fear & greed (already handled)

    Returns dict of DataFrames by source name.
    """
    results = {}
    ext_dir = os.path.join(DATA_DIR, "external")
    os.makedirs(ext_dir, exist_ok=True)

    # Active addresses from Blockchain.com
    logger.info("--- Active addresses (Blockchain.com) ---")
    try:
        df = _fetch_active_addresses()
        if not df.empty:
            results["active_addresses"] = df
    except Exception as e:
        logger.error(f"Active addresses fetch failed: {e}")

    # Exchange net flow proxy (from volume patterns)
    logger.info("--- Exchange net flow proxy ---")
    try:
        df = _fetch_exchange_flow_proxy()
        if not df.empty:
            results["exchange_net_flow"] = df
    except Exception as e:
        logger.error(f"Exchange net flow fetch failed: {e}")

    if results:
        for name, df in results.items():
            path = os.path.join(ext_dir, f"onchain_{name}.parquet")
            df.to_parquet(path, index=False)
            logger.info(f"Saved {path} ({len(df)} rows)")

    return results


def _fetch_active_addresses() -> pd.DataFrame:
    """Fetch active addresses from Blockchain.com API (no key required)."""
    import requests as req

    url = "https://api.blockchain.info/stats"
    for attempt in range(RETRY_ATTEMPTS):
        try:
            resp = req.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            n_requests = data.get("n_requests", 0)
            hashes_per_sec = data.get("hashrate", 0)
            timestamp = data.get("time", int(time.time() * 1000))

            return pd.DataFrame([{
                "timestamp": pd.to_datetime(timestamp, unit="ms", utc=True),
                "n_requests": n_requests,
                "hashrate": hashes_per_sec,
            }])
        except Exception as e:
            logger.warning(f"_fetch_active_addresses attempt {attempt+1}: {e}")
            time.sleep(RETRY_DELAY)

    return pd.DataFrame(columns=["timestamp", "n_requests", "hashrate"])


def _fetch_exchange_flow_proxy() -> pd.DataFrame:
    """Fetch exchange flow data from CoinGecko (free, no key).

    Uses 24h volume changes as proxy for exchange inflow/outflow.
    """
    import requests as req

    url = "https://api.coingecko.com/api/v3/coins/xrp?localization=false&community_data=false&developer_data=false&sparkline=false"
    for attempt in range(RETRY_ATTEMPTS):
        try:
            resp = req.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            md = data.get("market_data", {})
            return pd.DataFrame([{
                "timestamp": pd.Timestamp.utcnow(),
                "total_volume_24h": md.get("total_volume", {}).get("usdt", 0),
                "market_cap": md.get("market_cap", {}).get("usdt", 0),
                "price_change_24h": md.get("price_change_percentage_24h", 0),
            }])
        except Exception as e:
            logger.warning(f"_fetch_exchange_flow_proxy attempt {attempt+1}: {e}")
            time.sleep(RETRY_DELAY)

    return pd.DataFrame(columns=["timestamp", "total_volume_24h", "market_cap", "price_change_24h"])


def fetch_macro_data() -> Dict[str, pd.DataFrame]:
    """Fetch macroeconomic data.

    Sources:
    - yfinance: VIX (^VIX), DXY (DX-Y.NYB), US10Y (^TNX)
    - FRED API (optional, requires key)

    Returns dict of DataFrames.
    """
    import yfinance as yf

    results = {}
    ext_dir = os.path.join(DATA_DIR, "external")
    os.makedirs(ext_dir, exist_ok=True)

    macro_symbols = {
        "vix": "^VIX",
        "dxy": "DX-Y.NYB",
        "us10y": "^TNX",
    }

    for name, ticker in macro_symbols.items():
        logger.info(f"--- Macro: {name} ({ticker}) ---")
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period="max", interval="1d")
            if hist.empty:
                logger.warning(f"yfinance returned empty for {ticker}")
                continue
            df = hist.reset_index()
            df = df.rename(columns={
                "Datetime": "timestamp", "Date": "timestamp",
                "Open": "open", "High": "high", "Low": "low",
                "Close": "close", "Volume": "volume",
            })
            # Ensure UTC timezone for proper alignment
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df[["timestamp", "close"]].rename(columns={"close": f"{name}_value"})
            df = df.dropna()
            df = df.sort_values("timestamp").reset_index(drop=True)
            results[name] = df

            path = os.path.join(ext_dir, f"macro_{name}.parquet")
            df.to_parquet(path, index=False)
            logger.info(f"Saved {path} ({len(df)} rows)")
        except Exception as e:
            logger.error(f"Macro fetch failed for {ticker}: {e}")

    return results


def fetch_order_book_data(
    symbol: str = "XRP/USDT",
    exchange: str = "binance",
    n_levels: int = 20,
) -> Optional[pd.DataFrame]:
    """Fetch current order book snapshot from exchange.

    Note: This is a point-in-time snapshot. For historical analysis,
    you'd need an order book replay system.

    Returns DataFrame with current order book imbalance metrics.
    """
    import ccxt

    exchange_cls = getattr(ccxt, exchange)
    ex = exchange_cls({"enableRateLimit": True})
    ex.load_markets()

    try:
        ob = ex.fetch_order_book(symbol, limit=n_levels)
        bids = pd.DataFrame(ob["bids"][:n_levels], columns=["price", "size"])
        asks = pd.DataFrame(ob["asks"][:n_levels], columns=["price", "size"])

        bid_total = (bids["price"] * bids["size"]).sum()
        ask_total = (asks["price"] * asks["size"]).sum()
        total = bid_total + ask_total

        imbalance = (bid_total - ask_total) / total if total > 0 else 0.0
        depth_ratio = bid_total / ask_total if ask_total > 0 else 1.0

        spread = asks["price"].iloc[0] - bids["price"].iloc[0]
        mid = (bids["price"].iloc[0] + asks["price"].iloc[0]) / 2
        spread_pct = spread / mid if mid > 0 else 0.0

        return pd.DataFrame([{
            "timestamp": pd.Timestamp.utcnow(),
            "order_book_imbalance": imbalance,
            "depth_ratio": depth_ratio,
            "spread_pct": spread_pct,
            "bid_depth": bid_total,
            "ask_depth": ask_total,
        }])
    except Exception as e:
        logger.warning(f"Order book fetch failed ({exchange}): {e}")
        return None


def fetch_all_external(
    fetch_onchain: bool = True,
    fetch_macro: bool = True,
    fetch_orderbook: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Fetch all external data sources."""
    logger.info("=" * 60)
    logger.info("Starting external data fetch")
    logger.info("=" * 60)

    results = {}

    if fetch_onchain:
        try:
            oc = fetch_on_chain_data()
            results.update(oc)
        except Exception as e:
            logger.error(f"On-chain fetch failed: {e}")

    if fetch_macro:
        try:
            macro = fetch_macro_data()
            results.update(macro)
        except Exception as e:
            logger.error(f"Macro fetch failed: {e}")

    if fetch_orderbook:
        try:
            ob = fetch_order_book_data()
            if ob is not None:
                results["order_book_snapshot"] = ob
                ext_dir = os.path.join(DATA_DIR, "external")
                os.makedirs(ext_dir, exist_ok=True)
                ob.to_parquet(os.path.join(ext_dir, "order_book_snapshot.parquet"), index=False)
        except Exception as e:
            logger.error(f"Order book fetch failed: {e}")

    logger.info("=" * 60)
    logger.info("External data fetch complete!")
    logger.info("=" * 60)
    return results


def load_external_data(source: str) -> pd.DataFrame:
    """Load external data from parquet file."""
    ext_dir = os.path.join(DATA_DIR, "external")
    for suffix in ["", "onchain_", "macro_"]:
        path = os.path.join(ext_dir, f"{suffix}{source}.parquet")
        if os.path.exists(path):
            return pd.read_parquet(path)
    raise FileNotFoundError(f"External data not found: {source} (tried {suffix} prefixes)")
