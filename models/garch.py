from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.settings import GARCH_CONFIG
from utils.logger import setup_logger

logger = setup_logger(__name__)


class GARCHModel:
    """GARCH family models for volatility forecasting.

    Supports GARCH(1,1), EGARCH(1,1), and GJR-GARCH(1,1) via the arch package.
    Outputs conditional volatility forecasts and regime probability estimates
    based on volatility levels.
    """

    def __init__(self, model_type: str = "GARCH", config: Dict = None):
        self.config = config or GARCH_CONFIG
        self.model_type = model_type
        self.arch_model = None
        self.arch_result = None
        self.fitted = False

    def _get_arch_class(self):
        """Import and return the appropriate arch model class."""
        try:
            from arch import arch_model
        except ImportError:
            raise ImportError(
                "The 'arch' package is required for GARCH models. "
                "Install with: pip install arch"
            )
        return arch_model

    def fit(
        self,
        returns: np.ndarray,
        p: int = None,
        q: int = None,
    ) -> "GARCHModel":
        """Fit GARCH model to return series.

        Args:
            returns: Array of returns (not prices)
            p: GARCH order (default from config)
            q: ARCH order (default from config)
        """
        if p is None:
            p = self.config.get("p", 1)
        if q is None:
            q = self.config.get("q", 1)

        dist = self.config.get("distribution", "Normal")
        mean_type = self.config.get("mean", "Zero")

        arch_model_fn = self._get_arch_class()

        if self.model_type == "EGARCH":
            self.arch_model = arch_model_fn(
                returns,
                vol="EGARCH",
                p=p,
                q=q,
                mean=mean_type,
                dist=dist,
            )
        elif self.model_type == "GJR-GARCH":
            raise ValueError("GJR-GARCH: use GARCH with power=1 for asymmetric modeling")
        else:
            self.arch_model = arch_model_fn(
                returns,
                vol="GARCH",
                p=p,
                q=q,
                mean=mean_type,
                dist=dist,
            )

        self.arch_result = self.arch_model.fit(disp="off")
        self.fitted = True
        logger.info(f"GARCH({self.model_type}): fitted on {len(returns)} observations")
        return self

    def forecast(
        self,
        horizon: int = 1,
    ) -> np.ndarray:
        """Forecast conditional volatility.

        Args:
            horizon: Number of periods to forecast ahead

        Returns:
            Array of forecasted variances (horizon,)
        """
        if not self.fitted:
            raise RuntimeError("GARCHModel not fitted")

        forecast = self.arch_result.forecast(horizon=horizon)
        var_forecast = forecast.variance.values[-horizon:]
        return var_forecast.flatten()

    def get_conditional_volatility(self) -> np.ndarray:
        """Get in-sample conditional volatility estimates."""
        if not self.fitted:
            raise RuntimeError("GARCHModel not fitted")

        vol = self.arch_result.conditional_volatility
        if hasattr(vol, 'values'):
            return vol.values
        return np.asarray(vol)

    def get_regime_probabilities(self, vol: np.ndarray = None) -> Dict[str, np.ndarray]:
        """Estimate regime probabilities from volatility levels.

        Low vol → sideways regime
        Medium vol → trending regime
        High vol → stressed regime

        Args:
            vol: Volatility series. If None, uses in-sample conditional vol.

        Returns:
            Dict with 'low', 'medium', 'high' probability arrays.
        """
        if vol is None:
            vol = self.get_conditional_volatility()

        vol_series = pd.Series(vol)
        q33 = vol_series.quantile(0.33)
        q66 = vol_series.quantile(0.66)

        low_prob = np.clip(1 - (vol - q33) / (q66 - q33 + 1e-10), 0, 1)
        high_prob = np.clip((vol - q33) / (q66 - q33 + 1e-10), 0, 1)
        medium_prob = 1 - low_prob - high_prob
        medium_prob = np.clip(medium_prob, 0, 1)

        total = low_prob + medium_prob + high_prob + 1e-10
        return {
            "low": low_prob / total,
            "medium": medium_prob / total,
            "high": high_prob / total,
        }

    def save(self, path: str) -> None:
        import os
        import joblib
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        joblib.dump({
            "model_type": self.model_type,
            "config": self.config,
            "arch_result": self.arch_result,
        }, path)
        logger.info(f"Saved GARCH model ({self.model_type}) -> {path}")

    @classmethod
    def load(cls, path: str) -> "GARCHModel":
        import joblib
        data = joblib.load(path)
        model = cls(model_type=data["model_type"], config=data.get("config", GARCH_CONFIG))
        model.arch_result = data["arch_result"]
        model.fitted = True
        return model


class GARCHEnsemble:
    """Ensemble of multiple GARCH models for robust volatility forecasting."""

    def __init__(self, config: Dict = None):
        self.config = config or GARCH_CONFIG
        self.models: Dict[str, GARCHModel] = {}

    def fit(
        self,
        returns: np.ndarray,
    ) -> "GARCHEnsemble":
        """Fit all GARCH model variants."""
        model_types = self.config.get("models", ["GARCH", "EGARCH", "GJR-GARCH"])

        for mt in model_types:
            try:
                gm = GARCHModel(model_type=mt, config=self.config)
                gm.fit(returns)
                self.models[mt] = gm
            except Exception as e:
                logger.warning(f"GARCH fit failed for {mt}: {e}")

        if not self.models:
            logger.error("No GARCH models could be fitted")
        else:
            logger.info(f"GARCH ensemble: fitted {len(self.models)} models")
        return self

    def forecast(self, horizon: int = 1) -> Dict[str, np.ndarray]:
        """Forecast volatility from all models."""
        forecasts = {}
        for name, model in self.models.items():
            try:
                forecasts[name] = model.forecast(horizon)
            except Exception as e:
                logger.warning(f"GARCH forecast failed for {name}: {e}")
        return forecasts

    def median_forecast(self, horizon: int = 1) -> np.ndarray:
        """Return median forecast across all GARCH models."""
        forecasts = self.forecast(horizon)
        if not forecasts:
            return np.array([0.0])

        all_fcasts = np.array(list(forecasts.values()))
        return np.median(all_fcasts, axis=0)

    def save(self, path: str) -> None:
        import os
        import joblib
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        joblib.dump({
            "config": self.config,
            "models": {
                name: {
                    "model_type": m.model_type,
                    "arch_result": m.arch_result,
                }
                for name, m in self.models.items()
            },
        }, path)
        logger.info(f"Saved GARCH ensemble -> {path}")

    @classmethod
    def load(cls, path: str) -> "GARCHEnsemble":
        import joblib
        data = joblib.load(path)
        ensemble = cls(config=data.get("config", GARCH_CONFIG))
        for name, mdata in data["models"].items():
            gm = GARCHModel(model_type=mdata["model_type"])
            gm.arch_result = mdata["arch_result"]
            gm.fitted = True
            ensemble.models[name] = gm
        return ensemble
