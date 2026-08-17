"""Message handler and normalizer for WebSocket data feeds.

Parses exchange-specific messages into normalized OHLCV format.
"""

import json
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.logger import setup_logger

logger = setup_logger(__name__)


class MessageNormalizer:
    """Normalizes exchange-specific messages to standard OHLCV format.

    Each normalized record contains:
        timestamp (datetime), open, high, low, close, volume
    """

    def __init__(self):
        self.known_fields = {"timestamp", "open", "high", "low", "close", "volume"}

    def normalize_kraken(self, message: str) -> Optional[Dict]:
        """Parse Kraken WebSocket OHLC message.

        Kraken OHLC format: [time, open, high, low, close, vw, tr, vol_w, count]
        """
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            return None

        if isinstance(data, dict) and "event" in data:
            return None

        if isinstance(data, list) and len(data) >= 5:
            channel = data[0]
            payload = data[1]

            if isinstance(payload, list) and len(payload) >= 4:
                candle = payload[-1] if isinstance(payload[-1], list) else payload

                if len(candle) >= 7:
                    ts = float(candle[0])
                    return {
                        "timestamp": pd.Timestamp.fromtimestamp(ts, tz="UTC"),
                        "open": float(candle[1]),
                        "high": float(candle[2]),
                        "low": float(candle[3]),
                        "close": float(candle[4]),
                        "volume": float(candle[6]) if len(candle) > 6 else 0.0,
                        "source": "kraken",
                    }
        return None

    def normalize_coinbase(self, message: str) -> Optional[Dict]:
        """Parse Coinbase WebSocket match/ticker message."""
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            return None

        if not isinstance(data, dict):
            return None

        if data.get("type") == "match" or data.get("type") == "last_match":
            return {
                "timestamp": pd.Timestamp(data.get("time", "")),
                "close": float(data.get("price", 0)),
                "volume": float(data.get("size", 0)),
                "source": "coinbase",
                "is_tick": True,
            }
        elif data.get("type") == "ticker":
            return {
                "timestamp": pd.Timestamp(data.get("time", "")),
                "close": float(data.get("price", 0)),
                "volume": float(data.get("volume_24h", 0)),
                "source": "coinbase",
                "is_tick": True,
            }
        return None

    def normalize_okx(self, message: str) -> Optional[Dict]:
        """Parse OKX WebSocket trade/funding rate message."""
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            return None

        if not isinstance(data, dict) or "arg" not in data:
            return None

        arg = data.get("arg", {})
        channel = arg.get("channel", "")
        data_list = data.get("data", [])

        if not data_list:
            return None

        record = data_list[0]

        if channel == "trades":
            trade = record
            ts_ms = int(trade.get("ts", 0))
            return {
                "timestamp": pd.Timestamp(ts_ms, unit="ms", tz="UTC"),
                "close": float(trade.get("px", 0)),
                "volume": float(trade.get("sz", 0)),
                "source": "okx",
                "is_tick": True,
            }
        elif channel == "funding-rate":
            ts_ms = int(record.get("ts", 0))
            return {
                "timestamp": pd.Timestamp(ts_ms, unit="ms", tz="UTC"),
                "funding_rate": float(record.get("fundingRate", 0)),
                "source": "okx",
                "is_funding": True,
            }
        return None

    def normalize(self, source: str, message: str) -> Optional[Dict]:
        """Normalize message based on source."""
        if source == "kraken":
            return self.normalize_kraken(message)
        elif source == "coinbase":
            return self.normalize_coinbase(message)
        elif source == "okx":
            return self.normalize_okx(message)
        return None


class CandleAggregator:
    """Aggregates tick data into candles (OHLCV bars).

    Maintains a rolling buffer of ticks and flushes completed candles.
    """

    def __init__(self, interval_seconds: int = 3600, max_history: int = 1000):
        self.interval = interval_seconds
        self.max_history = max_history
        self.current_candle = None
        self.candle_history: List[Dict] = []
        self.tick_buffer: List[Dict] = []

    def add_tick(self, tick: Dict) -> Optional[Dict]:
        """Add a tick and return completed candle if interval elapsed.

        Args:
            tick: Normalized tick record with 'timestamp' and 'close'

        Returns:
            Completed candle dict or None
        """
        if tick.get("is_tick") and not tick.get("is_funding"):
            ts = tick["timestamp"]
            price = tick.get("close", 0)
            vol = tick.get("volume", 0)

            ts_floor = int(ts.timestamp()) // self.interval * self.interval

            if self.current_candle is None:
                self.current_candle = {
                    "timestamp": pd.Timestamp(ts_floor, tz="UTC"),
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "volume": vol,
                }
                self.current_tick_ts = ts_floor
            else:
                if ts_floor != self.current_tick_ts:
                    completed = dict(self.current_candle)
                    self.candle_history.append(completed)
                    if len(self.candle_history) > self.max_history:
                        self.candle_history = self.candle_history[-self.max_history:]

                    self.current_candle = {
                        "timestamp": pd.Timestamp(ts_floor, tz="UTC"),
                        "open": price,
                        "high": price,
                        "low": price,
                        "close": price,
                        "volume": vol,
                    }
                    self.current_tick_ts = ts_floor
                else:
                    self.current_candle["high"] = max(self.current_candle["high"], price)
                    self.current_candle["low"] = min(self.current_candle["low"], price)
                    self.current_candle["close"] = price
                    self.current_candle["volume"] += vol

        return None

    def add_ohlcv(self, candle: Dict) -> Optional[Dict]:
        """Add a complete OHLCV candle from source.

        Args:
            candle: Complete candle record from Kraken/other sources

        Returns:
            The candle record for downstream processing
        """
        if not candle.get("is_tick"):
            self.candle_history.append(candle)
            if len(self.candle_history) > self.max_history:
                self.candle_history = self.candle_history[-self.max_history:]
            return candle
        return self.add_tick(candle)

    def get_latest_candle(self) -> Optional[Dict]:
        """Get the most recent completed candle."""
        if self.candle_history:
            return self.candle_history[-1]
        if self.current_candle:
            return self.current_candle
        return None

    def get_history_df(self) -> pd.DataFrame:
        """Get candle history as DataFrame."""
        if not self.candle_history:
            return pd.DataFrame()
        return pd.DataFrame(self.candle_history)


class MessageHandler:
    """High-level message handler: parse → normalize → aggregate → dispatch."""

    def __init__(self, on_candle: Optional[Callable] = None, on_tick: Optional[Callable] = None):
        self.normalizer = MessageNormalizer()
        self.aggregators: Dict[str, CandleAggregator] = {}
        self.on_candle = on_candle
        self.on_tick = on_tick
        self.funding_rates: List[Dict] = []

    def get_aggregator(self, source: str, interval_seconds: int = 3600) -> CandleAggregator:
        """Get or create aggregator for a source."""
        key = f"{source}_{interval_seconds}"
        if key not in self.aggregators:
            self.aggregators[key] = CandleAggregator(interval_seconds)
        return self.aggregators[key]

    async def handle(self, source: str, message: str, interval_seconds: int = 3600) -> None:
        """Handle an incoming WebSocket message.

        Args:
            source: Exchange name (kraken, coinbase, okx)
            message: Raw message string
            interval_seconds: Candle aggregation interval
        """
        normalized = self.normalizer.normalize(source, message)
        if normalized is None:
            return

        if normalized.get("is_funding"):
            self.funding_rates.append(normalized)
            return

        agg = self.get_aggregator(source, interval_seconds)
        completed = agg.add_ohlcv(normalized)

        if completed and self.on_candle:
            await self.on_candle(completed)
        elif normalized.get("is_tick") and self.on_tick:
            await self.on_tick(normalized)
