"""Real-time streaming, incremental inference, drift monitoring, and alerting package."""

from streaming.alert import AlertGenerator
from streaming.drift_monitor import DriftMonitor
from streaming.engine import StreamingEngine
from streaming.handler import (
    CandleAggregator,
    MessageHandler,
    MessageNormalizer,
)
from streaming.predictor import StreamPredictor

__all__ = [
    "StreamingEngine",
    "StreamPredictor",
    "AlertGenerator",
    "DriftMonitor",
    "MessageHandler",
    "MessageNormalizer",
    "CandleAggregator",
]
