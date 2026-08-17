"""Population Stability Index (PSI) drift monitoring.

Detects feature distribution drift between reference and current data.
Triggers retraining alerts when drift exceeds threshold.
"""

import json
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.settings import DRIFT_CONFIG
from utils.logger import setup_logger

logger = setup_logger(__name__)


class DriftMonitor:
    """Monitors feature distribution drift using Population Stability Index (PSI).

    PSI interpretation:
    - < 0.1: No material change
    - 0.1 - 0.2: Moderate change
    - > 0.2: Significant change (action required)
    """

    def __init__(self, config: Dict = None):
        self.config = config or DRIFT_CONFIG
        self.psi_threshold = self.config.get("psi_threshold", 0.1)
        self.reference_window = self.config.get("reference_window", 1000)
        self.check_frequency = self.config.get("check_frequency", 100)
        self.auto_retrain = self.config.get("auto_retrain", False)
        self.drift_log = self.config.get("drift_log", "artifacts/drift_log.jsonl")

        self.reference_data: Dict[str, np.ndarray] = {}
        self.current_data: Dict[str, List[float]] = {}
        self._sample_count = 0
        self.drift_history: List[Dict] = []
        self._initialized = False

    def set_reference(self, X: np.ndarray, feature_names: List[str]) -> None:
        """Set reference distribution from training data.

        Args:
            X: Reference feature matrix (N, D)
            feature_names: Column names
        """
        for i, name in enumerate(feature_names):
            self.reference_data[name] = X[:, i]

        self._initialized = True
        logger.info(f"Drift reference set: {len(feature_names)} features, {len(X)} samples")

    def add_sample(self, X: np.ndarray, feature_names: List[str]) -> None:
        """Add a new sample to current data buffer.

        Args:
            X: Feature vector (D,)
            feature_names: Column names
        """
        if not self._initialized:
            return

        self._sample_count += 1
        for i, name in enumerate(feature_names):
            if name not in self.current_data:
                self.current_data[name] = []
            self.current_data[name].append(float(X[i]))

    def check_drift(self, force: bool = False) -> Optional[Dict]:
        """Check for drift in feature distributions.

        Returns:
            Drift report dict if drift detected, None otherwise
        """
        if not self._initialized:
            return None

        if not force and self._sample_count % self.check_frequency != 0:
            return None

        psi_scores = {}
        for name in self.reference_data:
            ref = self.reference_data[name]
            curr = np.array(self.current_data.get(name, []))

            if len(curr) < 10:
                continue

            psi = self._compute_psi(ref, curr)
            psi_scores[name] = psi

        overall_psi = np.mean(list(psi_scores.values())) if psi_scores else 0

        report = {
            "timestamp": str(pd.Timestamp.now(tz="UTC")),
            "sample_count": self._sample_count,
            "overall_psi": round(overall_psi, 6),
            "feature_psi": {k: round(v, 6) for k, v in psi_scores.items()},
            "threshold": self.psi_threshold,
            "drift_detected": overall_psi > self.psi_threshold,
        }

        if overall_psi > self.psi_threshold:
            report["message"] = f"Drift detected: PSI={overall_psi:.4f} > {self.psi_threshold}"
            report["action"] = "retrain" if self.auto_retrain else "alert"
            logger.warning(f"DRIFT: {report['message']}")
        else:
            report["message"] = f"No significant drift: PSI={overall_psi:.4f}"
            report["action"] = "none"

        self.drift_history.append(report)
        self._write_drift_log(report)
        return report

    @staticmethod
    def _compute_psi(
        reference: np.ndarray,
        current: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """Compute Population Stability Index between two distributions.

        PSI = sum((current_pct - reference_pct) * ln(current_pct / reference_pct))
        """
        eps = 1e-4

        min_val = min(reference.min(), current.min())
        max_val = max(reference.max(), current.max())

        if max_val - min_val < eps:
            return 0.0

        bin_edges = np.linspace(min_val, max_val, n_bins + 1)

        ref_counts, _ = np.histogram(reference, bins=bin_edges)
        curr_counts, _ = np.histogram(current, bins=bin_edges)

        ref_pct = (ref_counts + eps) / (len(reference) + eps * n_bins)
        curr_pct = (curr_counts + eps) / (len(current) + eps * n_bins)

        psi = np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct))
        return float(psi)

    def get_drift_report(self, n: int = 20) -> List[Dict]:
        """Get recent drift reports."""
        return self.drift_history[-n:]

    def _write_drift_log(self, report: Dict) -> None:
        """Write drift report to log file."""
        try:
            os.makedirs(os.path.dirname(self.drift_log) or ".", exist_ok=True)
            with open(self.drift_log, "a") as f:
                f.write(json.dumps(report, default=str) + "\n")
        except Exception as e:
            logger.error(f"Failed to write drift log: {e}")

    def reset_current(self) -> None:
        """Reset current data buffer (after retraining)."""
        self.current_data.clear()
        self._sample_count = 0
        logger.info("Drift monitor: current data reset")

    @staticmethod
    def compute_psi_array(reference: np.ndarray, current: np.ndarray) -> float:
        """Compute PSI between two arrays (convenience method)."""
        return DriftMonitor._compute_psi(reference, current)
