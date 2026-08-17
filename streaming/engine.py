"""Main streaming engine — orchestrates WebSocket, prediction, alerting, and drift.

Connects real-time data feeds to trained prediction models with full
monitoring pipeline.
"""

import asyncio
import json
import os
import signal
import sys
import time
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd

from config.settings import STREAMING_CONFIG, ALERT_CONFIG, DRIFT_CONFIG
from data.websocket import WebSocketManager
from streaming.handler import MessageHandler, CandleAggregator
from streaming.predictor import StreamPredictor
from streaming.alert import AlertGenerator
from streaming.drift_monitor import DriftMonitor
from utils.logger import setup_logger

logger = setup_logger(__name__)


class StreamingEngine:
    """Real-time prediction pipeline engine.

    Pipeline flow:
    1. WebSocket Manager receives messages
    2. MessageHandler normalizes and aggregates candles
    3. StreamPredictor computes features and runs inference
    4. AlertGenerator triggers alerts on high-confidence predictions
    5. DriftMonitor tracks feature distribution stability
    """

    def __init__(
        self,
        interval: str = "1h",
        target_window: str = "24h",
        model_type: str = "ensemble",
    ):
        self.interval = interval
        self.target_window = target_window if "h" in str(target_window) else f"{target_window}h"
        self.model_type = model_type
        self._running = False
        self._start_time = 0

        self.ws_manager: Optional[WebSocketManager] = None
        self.message_handler: Optional[MessageHandler] = None
        self.predictor: Optional[StreamPredictor] = None
        self.alert_gen: Optional[AlertGenerator] = None
        self.drift_monitor: Optional[DriftMonitor] = None

    async def initialize(self) -> bool:
        """Initialize all streaming components."""
        logger.info("=" * 60)
        logger.info(f"Initializing streaming engine: {self.interval}/{self.target_window}")
        logger.info("=" * 60)

        # Initialize predictor
        self.predictor = StreamPredictor(
            interval=self.interval,
            target_window=self.target_window,
            model_type=self.model_type,
        )
        if not self.predictor.load_models():
            logger.error("Failed to load prediction models")
            return False

        self.predictor.load_state()

        # Initialize message handler
        interval_seconds = self._interval_to_seconds(self.interval)
        self.message_handler = MessageHandler(
            on_candle=self._on_candle,
            on_tick=self._on_tick,
        )

        # Initialize alert generator
        self.alert_gen = AlertGenerator(config=ALERT_CONFIG)

        # Initialize drift monitor
        self.drift_monitor = DriftMonitor(config=DRIFT_CONFIG)
        if self.predictor.feature_names:
            try:
                from data.loader import get_datasets
                data = get_datasets(self.interval, self.target_window)
                self.drift_monitor.set_reference(data["X_train"], self.predictor.feature_names)
            except Exception as e:
                logger.warning(f"Could not set drift reference: {e}")

        # Initialize WebSocket manager
        self.ws_manager = WebSocketManager(config=STREAMING_CONFIG)

        logger.info("Streaming engine initialized successfully")
        return True

    async def start(self, duration: int = 0) -> None:
        """Start streaming pipeline.

        Args:
            duration: Run duration in seconds (0 = unlimited)
        """
        self._running = True
        self._start_time = time.time()

        logger.info(f"Starting streaming pipeline (duration={duration}s)")

        def _on_msg(name: str, msg: str):
            asyncio.create_task(
                self.message_handler.handle(name, msg,
                self._interval_to_seconds(self.interval))
            )

        for name, cfg in self.ws_manager.config.get("sources", {}).items():
            if cfg.get("enabled", False):
                self.ws_manager.add_connection(name, cfg["url"], on_message=_on_msg)

        await self.ws_manager.start()

        try:
            while self._running:
                if duration > 0 and time.time() - self._start_time > duration:
                    logger.info(f"Duration reached ({duration}s), stopping")
                    break
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            await self.stop()

    async def stop(self) -> None:
        """Stop streaming pipeline."""
        logger.info("Stopping streaming pipeline...")
        self._running = False

        if self.predictor:
            self.predictor.save_state()

        if self.ws_manager:
            await self.ws_manager.stop()

        logger.info("Streaming pipeline stopped")

    async def _on_candle(self, candle: Dict) -> None:
        """Handle a completed candle."""
        self.predictor.update_feature_buffer(candle)

        X = self.predictor.compute_features()
        if X is None:
            return

        prediction = self.predictor.predict(X)
        if prediction:
            logger.info(
                f"[{self.interval}] {prediction['direction']} "
                f"P(UP)={prediction['P_UP']:.4f} "
                f"conf={prediction['confidence']:.4f}"
            )

        if self.drift_monitor:
            self.drift_monitor.add_sample(X.flatten(), self.predictor.feature_names)
            drift = self.drift_monitor.check_drift()
            if drift and drift.get("drift_detected"):
                logger.warning(f"Feature drift detected: PSI={drift['overall_psi']:.4f}")

        if self.alert_gen:
            self.alert_gen.check_prediction(prediction)

    async def _on_tick(self, tick: Dict) -> None:
        """Handle a tick-level update."""
        self.predictor.update_feature_buffer(tick)

    @staticmethod
    def _interval_to_seconds(interval: str) -> int:
        """Convert interval string to seconds."""
        mapping = {
            "1m": 60, "5m": 300, "15m": 900, "30m": 1800,
            "1h": 3600, "2h": 7200, "4h": 14400, "1d": 86400,
        }
        return mapping.get(interval, 3600)


async def run_stream(
    interval: str = "1h",
    target_window: str = "24h",
    model_type: str = "ensemble",
    duration: int = 0,
) -> None:
    """Run streaming pipeline.

    Args:
        interval: Data interval (15m, 1h, 1d)
        target_window: Prediction window
        model_type: Model to use (ensemble, lightgbm, etc.)
        duration: Run duration in seconds (0 = unlimited)
    """
    engine = StreamingEngine(interval, target_window, model_type)

    if not await engine.initialize():
        logger.error("Failed to initialize streaming engine")
        return

    await engine.start(duration)
