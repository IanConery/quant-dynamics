from typing import Optional, Tuple
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from utils.logger import setup_logger

logger = setup_logger(__name__)


def calibrate_platt(
    proba: np.ndarray,
    y_true: np.ndarray,
    method: str = "sigmoid",
) -> np.ndarray:
    """Calibrate classification probabilities using Platt scaling or isotonic regression.

    Args:
        proba: Raw predicted probabilities (N,) for binary or (N, C) for multi-class
        y_true: True labels (N,)
        method: 'sigmoid' for Platt scaling, 'isotonic' for isotonic regression

    Returns:
        Calibrated probabilities (N,) for binary, (N, C) for multi-class
    """
    if proba.ndim == 1:
        return _calibrate_binary(proba=proba, y_true=y_true, method=method)
    else:
        return _calibrate_multi(proba=proba, y_true=y_true, method=method)


def _calibrate_binary(proba: np.ndarray, y_true: np.ndarray, method: str) -> np.ndarray:
    """Calibrate binary probabilities."""
    X_input = proba.reshape(-1, 1)
    y_binary = (y_true > 0).astype(int)

    n_pos = float(np.sum(y_binary == 1))
    n_neg = float(np.sum(y_binary == 0))
    if n_pos == 0 or n_neg == 0:
        logger.warning("Calibration: single class in validation data, returning raw probabilities")
        return proba

    if method == "sigmoid":
        lr = LogisticRegression()
        lr.fit(X_input, y_binary)
        calibrated = lr.predict_proba(X_input)[:, 1]
    else:
        iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
        calibrated = iso.fit_transform(proba, y_binary)

    logger.info(f"Calibration ({method}): ECE before={_ece_binary(proba, y_binary):.4f}, after={_ece_binary(calibrated, y_binary):.4f}")
    return calibrated


def _calibrate_multi(proba: np.ndarray, y_true: np.ndarray, method: str) -> np.ndarray:
    """Calibrate multi-class probabilities."""
    n_classes = proba.shape[1]
    y_labels = y_true.astype(int)

    calibrated = np.zeros_like(proba)
    for c in range(n_classes):
        y_binary = (y_labels == c).astype(int)
        p_col = proba[:, c]

        n_pos = float(np.sum(y_binary == 1))
        n_neg = float(np.sum(y_binary == 0))
        if n_pos == 0 or n_neg == 0:
            calibrated[:, c] = p_col
            continue

        if method == "sigmoid":
            lr = LogisticRegression()
            lr.fit(p_col.reshape(-1, 1), y_binary)
            calibrated[:, c] = lr.predict_proba(p_col.reshape(-1, 1))[:, 1]
        else:
            iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
            calibrated[:, c] = iso.fit_transform(p_col, y_binary)

    calibrated = calibrated / calibrated.sum(axis=1, keepdims=True)
    logger.info(f"Calibration ({method}) multi-class: ECE before={_ece_multi(proba, y_true):.4f}, after={_ece_multi(calibrated, y_true):.4f}")
    return calibrated


def _ece_binary(proba: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error for binary classification."""
    y_binary = (y_true > 0).astype(int)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (proba >= bin_boundaries[i]) & (proba < bin_boundaries[i + 1])
        if i == n_bins - 1:
            mask = (proba >= bin_boundaries[i]) & (proba <= bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue
        bin_conf = proba[mask].mean()
        bin_acc = y_binary[mask].mean()
        ece += mask.sum() * abs(bin_conf - bin_acc)
    return float(ece / len(y_true))


def _ece_multi(proba: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error for multi-class classification."""
    y_labels = y_true.astype(int)
    n_classes = proba.shape[1]
    ece = 0.0

    for c in range(n_classes):
        y_binary = (y_labels == c).astype(int)
        p_col = proba[:, c]

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        for i in range(n_bins):
            mask = (p_col >= bin_boundaries[i]) & (p_col < bin_boundaries[i + 1])
            if i == n_bins - 1:
                mask = (p_col >= bin_boundaries[i]) & (p_col <= bin_boundaries[i + 1])
            if mask.sum() == 0:
                continue
            bin_conf = p_col[mask].mean()
            bin_acc = y_binary[mask].mean()
            ece += mask.sum() * abs(bin_conf - bin_acc)

    return float(ece / (len(y_true) * n_classes))


class ProbabilityCalibrator:
    """Wraps a model's predict_proba with calibration.

    Trains calibration on validation data, applies to any prediction.
    """

    def __init__(self, method: str = "sigmoid"):
        self.method = method
        self.is_binary = None
        self.fitted = False
        self.calibrators = []

    def fit(self, proba: np.ndarray, y_true: np.ndarray) -> "ProbabilityCalibrator":
        """Train calibration on validation set predictions and labels."""
        if proba.ndim == 1:
            self.is_binary = True
            n_pos = float(np.sum((y_true > 0).astype(int) == 1))
            n_neg = float(np.sum((y_true > 0).astype(int) == 0))
            if n_pos == 0 or n_neg == 0:
                logger.warning("Single class in calibration data, skipping calibration")
                self.fitted = False
                return self
            self._fit_binary(proba=proba, y_true=y_true)
        else:
            self.is_binary = False
            self._fit_multi(proba=proba, y_true=y_true)
        self.fitted = True
        return self

    def _fit_binary(self, proba: np.ndarray, y_true: np.ndarray) -> None:
        y_binary = (y_true > 0).astype(int)
        if self.method == "sigmoid":
            lr = LogisticRegression()
            lr.fit(proba.reshape(-1, 1), y_binary)
            self.calibrators = [lr]
        else:
            iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
            iso.fit(proba, y_binary)
            self.calibrators = [iso]

    def _fit_multi(self, proba: np.ndarray, y_true: np.ndarray) -> None:
        n_classes = proba.shape[1]
        self.calibrators = []
        y_labels = y_true.astype(int)
        for c in range(n_classes):
            y_binary = (y_labels == c).astype(int)
            n_pos = float(np.sum(y_binary == 1))
            n_neg = float(np.sum(y_binary == 0))
            if n_pos == 0 or n_neg == 0:
                self.calibrators.append(None)
                continue
            if self.method == "sigmoid":
                lr = LogisticRegression()
                lr.fit(proba[:, c].reshape(-1, 1), y_binary)
                self.calibrators.append(lr)
            else:
                iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
                iso.fit(proba[:, c], y_binary)
                self.calibrators.append(iso)

    def calibrate(self, proba: np.ndarray) -> np.ndarray:
        """Apply calibration to raw probabilities."""
        if not self.fitted:
            return proba

        if self.is_binary:
            return self._calibrate_binary(proba)
        return self._calibrate_multi(proba)

    def _calibrate_binary(self, proba: np.ndarray) -> np.ndarray:
        cal = self.calibrators[0]
        if cal is None:
            return proba
        if self.method == "sigmoid":
            return cal.predict_proba(proba.reshape(-1, 1))[:, 1]
        else:
            return cal.transform(proba)

    def _calibrate_multi(self, proba: np.ndarray) -> np.ndarray:
        n_classes = proba.shape[1]
        calibrated = np.zeros_like(proba)
        for c in range(n_classes):
            cal = self.calibrators[c]
            if cal is None:
                calibrated[:, c] = proba[:, c]
                continue
            if self.method == "sigmoid":
                calibrated[:, c] = cal.predict_proba(proba[:, c].reshape(-1, 1))[:, 1]
            else:
                calibrated[:, c] = cal.transform(proba[:, c])
        calibrated = calibrated / calibrated.sum(axis=1, keepdims=True)
        return calibrated
