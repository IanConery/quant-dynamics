"""Alert system for streaming predictions.

Generates alerts when prediction confidence or regime changes exceed thresholds.
"""

import json
import os
import time
from typing import Dict, List, Optional

import pandas as pd

from config.settings import ALERT_CONFIG
from utils.logger import setup_logger

logger = setup_logger(__name__)


class AlertGenerator:
    """Generates and manages alerts from streaming predictions.

    Alert conditions:
    1. High confidence prediction (P(UP) > threshold or P(UP) < 1-threshold)
    2. Large predicted return (|return| > threshold)
    3. Regime change detected
    4. Cooldown prevents alert spam
    """

    def __init__(self, config: Dict = None):
        self.config = config or ALERT_CONFIG
        self.min_confidence = self.config.get("min_confidence", 0.70)
        self.return_threshold = self.config.get("pred_return_threshold", 0.005)
        self.alert_on_regime = self.config.get("alert_on_regime_change", True)
        self.cooldown = self.config.get("cooldown_seconds", 300)
        self.alert_file = self.config.get("alert_file", "artifacts/alerts.jsonl")
        self._last_alert_time = 0
        self._last_regime = None
        self.alerts: List[Dict] = []

    def check_prediction(self, prediction: Dict) -> Optional[Dict]:
        """Check if prediction triggers an alert.

        Args:
            prediction: Dict with P_UP, direction, confidence, predicted_return

        Returns:
            Alert dict if triggered, None otherwise
        """
        now = time.time()

        if now - self._last_alert_time < self.cooldown:
            return None

        alert = None
        alert_type = None

        confidence = prediction.get("confidence", 0)
        p_up = prediction.get("P_UP", 0.5)

        if confidence >= self.min_confidence:
            alert_type = "high_confidence"
            alert = {
                "timestamp": str(pd.Timestamp.now(tz="UTC")),
                "type": alert_type,
                "P_UP": round(p_up, 6),
                "direction": prediction.get("direction", "?"),
                "confidence": round(confidence, 6),
                "message": f"{'BULLISH' if p_up >= 0.5 else 'BEARISH'} signal with {confidence:.1%} confidence",
            }

        pred_return = prediction.get("predicted_return")
        if pred_return is not None and abs(pred_return) >= self.return_threshold:
            alert_type = "large_return"
            alert = {
                "timestamp": str(pd.Timestamp.now(tz="UTC")),
                "type": alert_type,
                "predicted_return": round(pred_return, 6),
                "direction": "UP" if pred_return > 0 else "DOWN",
                "message": f"Large predicted return: {pred_return:+.2%}",
            }

        if self.alert_on_regime and prediction.get("regime"):
            current_regime = prediction["regime"]
            if self._last_regime and current_regime != self._last_regime:
                alert_type = "regime_change"
                alert = {
                    "timestamp": str(pd.Timestamp.now(tz="UTC")),
                    "type": alert_type,
                    "from_regime": self._last_regime,
                    "to_regime": current_regime,
                    "message": f"Regime change: {self._last_regime} → {current_regime}",
                }
            self._last_regime = current_regime

        if alert:
            self._last_alert_time = now
            self.alerts.append(alert)
            self._write_alert(alert)
            logger.info(f"ALERT [{alert_type}]: {alert['message']}")
            return alert

        return None

    def _write_alert(self, alert: Dict) -> None:
        """Write alert to JSONL file."""
        try:
            os.makedirs(os.path.dirname(self.alert_file) or ".", exist_ok=True)
            with open(self.alert_file, "a") as f:
                f.write(json.dumps(alert, default=str) + "\n")
        except Exception as e:
            logger.error(f"Failed to write alert: {e}")

    def get_recent_alerts(self, n: int = 20) -> List[Dict]:
        """Get recent alerts."""
        return self.alerts[-n:]

    def clear_alerts(self) -> None:
        """Clear in-memory alert history."""
        self.alerts.clear()
        self._last_alert_time = 0
        self._last_regime = None
