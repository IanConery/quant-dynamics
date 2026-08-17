"""WebSocket connection manager for real-time data feeds.

Provides persistent WebSocket connections to exchanges with auto-reconnect,
heartbeat, and graceful degradation.
"""

import asyncio
import json
import time
import os
from typing import Any, Callable, Dict, List, Optional

import websockets
from websockets.exceptions import ConnectionClosed, ConnectionClosedError, ConnectionClosedOK

from config.settings import STREAMING_CONFIG
from utils.logger import setup_logger

logger = setup_logger(__name__)


class WebSocketConnection:
    """Manages a single persistent WebSocket connection with auto-reconnect."""

    def __init__(
        self,
        name: str,
        url: str,
        on_message: Optional[Callable] = None,
        on_connect: Optional[Callable] = None,
        on_disconnect: Optional[Callable] = None,
    ):
        self.name = name
        self.url = url
        self.on_message = on_message
        self.on_connect = on_connect
        self.on_disconnect = on_disconnect
        self.ws = None
        self.connected = False
        self.running = False
        self._reconnect_config = STREAMING_CONFIG.get("reconnect", {})
        self.max_retries = self._reconnect_config.get("max_retries", 10)
        self.base_delay = self._reconnect_config.get("base_delay", 1.0)
        self.max_delay = self._reconnect_config.get("max_delay", 60.0)
        self.backoff_factor = self._reconnect_config.get("backoff_factor", 2.0)
        self._retry_count = 0
        self._task = None

    async def connect(self) -> None:
        """Establish WebSocket connection."""
        try:
            self.ws = await websockets.connect(self.url, ping_interval=20, ping_timeout=10)
            self.connected = True
            self._retry_count = 0
            logger.info(f"[{self.name}] Connected to {self.url}")
            if self.on_connect:
                await self.on_connect(self.name)
        except Exception as e:
            logger.error(f"[{self.name}] Connection failed: {e}")
            self.connected = False
            if self.on_disconnect:
                await self.on_disconnect(self.name, str(e))

    async def disconnect(self) -> None:
        """Close WebSocket connection."""
        self.running = False
        if self.ws:
            await self.ws.close()
            self.connected = False
            logger.info(f"[{self.name}] Disconnected")

    async def send(self, message: str) -> None:
        """Send a message to the WebSocket."""
        if self.ws and self.connected:
            await self.ws.send(message)

    async def receive(self) -> Optional[str]:
        """Receive a message from the WebSocket."""
        if self.ws and self.connected:
            try:
                msg = await asyncio.wait_for(self.ws.recv(), timeout=30)
                return msg
            except asyncio.TimeoutError:
                pass
            except ConnectionClosed:
                self.connected = False
        return None

    async def _reconnect(self) -> None:
        """Reconnect with exponential backoff."""
        for attempt in range(1, self.max_retries + 1):
            delay = min(self.base_delay * (self.backoff_factor ** (attempt - 1)), self.max_delay)
            logger.warning(f"[{self.name}] Reconnecting in {delay:.1f}s (attempt {attempt}/{self.max_retries})")
            await asyncio.sleep(delay)
            await self.connect()
            if self.connected:
                self._retry_count = 0
                return
            self._retry_count += 1

        logger.error(f"[{self.name}] Max reconnection attempts reached")
        self.running = False

    async def _run(self) -> None:
        """Main event loop for the WebSocket connection."""
        await self.connect()

        while self.running and self.connected:
            try:
                msg = await self.receive()
                if msg is None:
                    if self.running:
                        await self._reconnect()
                    continue

                if self.on_message:
                    await self.on_message(self.name, msg)

            except Exception as e:
                logger.error(f"[{self.name}] Error in message loop: {e}")
                self.connected = False
                if self.running:
                    await self._reconnect()

    async def start(self) -> None:
        """Start the WebSocket connection loop."""
        self.running = True
        self._task = asyncio.current_task()
        await self._run()

    def stop(self) -> None:
        """Signal the connection to stop."""
        self.running = False


class WebSocketManager:
    """Manages multiple WebSocket connections with subscription handling."""

    def __init__(self, config: Dict = None):
        self.config = config or STREAMING_CONFIG
        self.connections: Dict[str, WebSocketConnection] = {}
        self._tasks: List[asyncio.Task] = []
        self._running = False

    def add_connection(
        self,
        name: str,
        url: str,
        on_message: Optional[Callable] = None,
    ) -> WebSocketConnection:
        """Add a WebSocket connection."""
        conn = WebSocketConnection(name, url, on_message=on_message)
        self.connections[name] = conn
        return conn

    def get_subscription_message(self, source: str, symbol: str) -> str:
        """Generate exchange-specific subscription message."""
        if source == "kraken":
            pair = symbol.replace("/", "")
            return json.dumps({
                "event": "subscribe",
                "pair": [pair],
                "subscription": {"name": "ohlc", "interval": 15},
            })
        elif source == "coinbase":
            return json.dumps({
                "type": "subscribe",
                "product_ids": [symbol.replace("-", "/")],
                "channels": ["matches"],
            })
        elif source == "okx":
            return json.dumps({
                "op": "subscribe",
                "args": [{"channel": "trades", "instId": symbol}],
            })
        return ""

    async def subscribe_all(self) -> None:
        """Subscribe all connections to configured channels."""
        sources = self.config.get("sources", {})
        for name, cfg in sources.items():
            if not cfg.get("enabled", False):
                continue
            if name in self.connections:
                sub_msg = self.get_subscription_message(name, cfg.get("symbol", ""))
                if sub_msg:
                    await self.connections[name].send(sub_msg)
                    logger.info(f"[{name}] Subscribed to {cfg.get('symbol', '?')}")

    async def start(self) -> None:
        """Start all WebSocket connections."""
        self._running = True
        sources = self.config.get("sources", {})

        for name, cfg in sources.items():
            if not cfg.get("enabled", False):
                continue
            if name not in self.connections:
                self.add_connection(name, cfg["url"])

        for conn in self.connections.values():
            task = asyncio.create_task(conn.start())
            self._tasks.append(task)

        await self.subscribe_all()

    async def stop(self) -> None:
        """Stop all WebSocket connections."""
        self._running = False
        for conn in self.connections.values():
            conn.stop()
            await conn.disconnect()

        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        self._tasks.clear()
        logger.info("All WebSocket connections stopped")


async def run_streaming_loop(
    handler: Callable,
    duration: int = 0,
) -> None:
    """Run WebSocket streaming loop.

    Args:
        handler: Async function(name, message) to process each message
        duration: Seconds to run (0 = unlimited)
    """
    manager = WebSocketManager()

    start_time = time.time()

    def _on_msg(name: str, msg: str):
        asyncio.create_task(handler(name, msg))

    for name, cfg in manager.config.get("sources", {}).items():
        if cfg.get("enabled", False):
            manager.add_connection(name, cfg["url"], on_message=_on_msg)

    await manager.start()

    try:
        while manager._running:
            if duration > 0 and time.time() - start_time > duration:
                logger.info(f"Duration limit reached ({duration}s), stopping")
                break
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Streaming interrupted by user")
    finally:
        await manager.stop()
