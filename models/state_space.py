from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from config.settings import STATE_SPACE_CONFIG
from utils.logger import setup_logger

logger = setup_logger(__name__)


class KalmanFilter:
    """Kalman Filter for latent price process separation.

    Separates signal (trend) from noise in price data using a state-space
    model. Outputs filtered signal, smoothed estimates, and prediction
    intervals.
    """

    def __init__(self, config: Dict = None):
        self.config = config or STATE_SPACE_CONFIG.get("kalman", {})
        self.process_noise = self.config.get("process_noise", 1e-4)
        self.observation_noise = self.config.get("observation_noise", 1e-3)
        self.n_components = self.config.get("n_components", 2)
        self.state_mean = None
        self.state_cov = None
        self.filtered_states = None
        self.filtered_covs = None
        self.fitted = False

    def fit(
        self,
        prices: np.ndarray,
        initial_state: np.ndarray = None,
        initial_cov: np.ndarray = None,
    ) -> "KalmanFilter":
        """Fit Kalman Filter to price series.

        Uses a local linear trend model:
        - State: [level, trend]
        - Observation: level
        - Transition: level_t = level_{t-1} + trend_{t-1} + process_noise
        - Observation: y_t = level_t + observation_noise

        Args:
            prices: Price series (N,)
            initial_state: Initial state estimate [level, trend]
            initial_cov: Initial state covariance matrix
        """
        prices = np.asarray(prices, dtype=np.float64)
        n = len(prices)

        if initial_state is None:
            self.state_mean = np.array([prices[0], 0.0])
        else:
            self.state_mean = initial_state.copy()

        if initial_cov is None:
            self.state_cov = np.eye(self.n_components) * 1.0
        else:
            self.state_cov = initial_cov.copy()

        Q = np.eye(self.n_components) * self.process_noise
        R = self.observation_noise

        F = np.array([[1, 1], [0, 1]])
        H = np.array([1, 0])

        filtered_states = []
        filtered_covs = []

        for i in range(n):
            # Predict
            state_pred = F @ self.state_mean
            cov_pred = F @ self.state_cov @ F.T + Q

            # Update
            y = prices[i]
            y_pred = float(H @ state_pred)
            innovation = y - y_pred
            S = float(H @ cov_pred @ H) + R
            K = (cov_pred @ H) / (S + 1e-10)

            self.state_mean = state_pred + K * innovation
            I_mat = np.eye(self.n_components) - np.outer(K, H)
            self.state_cov = I_mat @ cov_pred

            filtered_states.append(self.state_mean.copy())
            filtered_covs.append(self.state_cov.copy())

        self.filtered_states = np.array(filtered_states)
        self.filtered_covs = np.array(filtered_covs)
        self.fitted = True
        logger.info(f"Kalman Filter: fitted on {n} observations, "
                     f"signal extracted ({self.n_components} components)")
        return self

    def get_signal(self) -> np.ndarray:
        """Extract filtered signal (level component)."""
        if not self.fitted:
            raise RuntimeError("KalmanFilter not fitted")
        return self.filtered_states[:, 0]

    def get_trend(self) -> np.ndarray:
        """Extract filtered trend component."""
        if not self.fitted:
            raise RuntimeError("KalmanFilter not fitted")
        return self.filtered_states[:, 1]

    def get_confidence_bounds(
        self,
        prices: np.ndarray,
        n_std: float = 2.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get confidence bounds around filtered signal."""
        if not self.fitted:
            raise RuntimeError("KalmanFilter not fitted")

        signal = self.get_signal()
        var = self.filtered_covs[:, 0, 0]
        margin = n_std * np.sqrt(var + self.observation_noise)
        return signal - margin, signal + margin

    def predict(
        self,
        horizon: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict future prices using current state.

        Returns:
            (predictions, prediction_variance)
        """
        if not self.fitted:
            raise RuntimeError("KalmanFilter not fitted")

        F = np.array([[1, 1], [0, 1]])
        Q = np.eye(self.n_components) * self.process_noise

        state = np.asarray(self.state_mean).ravel()
        cov = np.asarray(self.state_cov)
        predictions = []
        variances = []

        for _ in range(horizon):
            state = F @ state
            cov = F @ cov @ F.T + Q
            predictions.append(float(state[0]))
            variances.append(float(cov[0, 0] + self.observation_noise))

        return np.array(predictions), np.array(variances)

    def save(self, path: str) -> None:
        import os
        import joblib
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        joblib.dump({
            "config": self.config,
            "state_mean": self.state_mean,
            "state_cov": self.state_cov,
            "filtered_states": self.filtered_states,
            "filtered_covs": self.filtered_covs,
            "fitted": self.fitted,
        }, path)

    @classmethod
    def load(cls, path: str) -> "KalmanFilter":
        import joblib
        data = joblib.load(path)
        kf = KalmanFilter(config=data.get("config", {}))
        kf.state_mean = data["state_mean"]
        kf.state_cov = data["state_cov"]
        kf.filtered_states = data["filtered_states"]
        kf.filtered_covs = data["filtered_covs"]
        kf.fitted = data.get("fitted", False)
        return kf


class GaussianProcessRegressor:
    """Gaussian Process for price regression with uncertainty estimation.

    Uses sklearn's GaussianProcessRegressor with configurable kernel.
    """

    def __init__(self, config: Dict = None):
        self.config = config or STATE_SPACE_CONFIG.get("gaussian_process", {})
        self.gpr = None
        self.fitted = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> "GaussianProcessRegressor":
        """Fit GP to features X and targets y."""
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor as GPR
            from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
        except ImportError:
            raise ImportError(
                "sklearn Gaussian Process requires sklearn >= 0.24. "
                "Install with: pip install scikit-learn"
            )

        kernel_str = self.config.get("kernel", "RBF + WhiteKernel")
        n_restarts = self.config.get("n_restarts_optimizer", 10)
        random_state = self.config.get("random_state", 42)

        if "RBF" in kernel_str:
            kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3))
        else:
            kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3))

        if "WhiteKernel" in kernel_str:
            kernel += WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-10, 1e+1))

        if "ConstantKernel" in kernel_str:
            kernel = ConstantKernel(1.0, (1e-3, 1e3)) * kernel

        self.gpr = GPR(
            kernel=kernel,
            n_restarts_optimizer=n_restarts,
            random_state=random_state,
            normalize_y=True,
        )
        self.gpr.fit(X, y)
        self.fitted = True
        logger.info(f"GP: fitted on {len(X)} samples, "
                     f"log_marginal_likelihood={self.gpr.log_marginal_likelihood():.4f}")
        return self

    def predict(
        self,
        X: np.ndarray,
        return_std: bool = False,
    ):
        """Predict with uncertainty estimation."""
        if not self.fitted:
            raise RuntimeError("GaussianProcessRegressor not fitted")

        y_mean, y_std = self.gpr.predict(X, return_std=True)
        if return_std:
            return y_mean, y_std
        return y_mean

    def predict_intervals(
        self,
        X: np.ndarray,
        n_std: float = 2.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict with confidence intervals."""
        y_mean, y_std = self.predict(X, return_std=True)
        lower = y_mean - n_std * y_std
        upper = y_mean + n_std * y_std
        return y_mean, lower, upper

    def save(self, path: str) -> None:
        import os
        import joblib
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        joblib.dump({
            "config": self.config,
            "gpr": self.gpr,
            "fitted": self.fitted,
        }, path)

    @classmethod
    def load(cls, path: str) -> "GaussianProcessRegressor":
        import joblib
        data = joblib.load(path)
        gpr = cls(config=data.get("config", {}))
        gpr.gpr = data["gpr"]
        gpr.fitted = data.get("fitted", False)
        return gpr
