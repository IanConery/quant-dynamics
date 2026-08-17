from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score

from config.settings import MODEL_DIR, REGIME_CLASSIFIER
from models.regime_classifier import RegimeClassifier
from models.garch import GARCHModel, GARCHEnsemble
from models.state_space import KalmanFilter, GaussianProcessRegressor
from models.classical import ClassicalModel
from models.deep import DeepModel
from models.ensemble import StackingEnsemble
from utils.logger import setup_logger

logger = setup_logger(__name__)


class RegimeAwareTrainer:
    """Trains separate models per regime and ensembles by regime probability.

    Workflow:
    1. Classify each timestep into a regime (Bear/Sideways/Bull)
    2. Train separate prediction models per regime
    3. At inference time, weight predictions by current regime probability
    4. Fallback to default model when regime confidence is low
    """

    def __init__(self, config: Dict = None):
        self.config = config or REGIME_CLASSIFIER
        self.regime_classifier: Optional[RegimeClassifier] = None
        self.regime_models: Dict[str, Dict[str, Any]] = {}
        self.default_model: Optional[Any] = None
        self.garch_ensemble: Optional[GARCHEnsemble] = None
        self.kalman_filter: Optional[KalmanFilter] = None
        self.gpr: Optional[GaussianProcessRegressor] = None
        self.feature_names: List[str] = []
        self.task: str = "regression"

    def train_regime_classifier(
        self,
        df: pd.DataFrame,
        train_end: int,
        val_end: int,
    ) -> RegimeClassifier:
        """Step 1: Train the regime classifier."""
        self.regime_classifier = RegimeClassifier(config=self.config)
        self.regime_classifier.fit(df, train_end, val_end)
        return self.regime_classifier

    def get_regime_mask(
        self,
        df: pd.DataFrame,
        regime: str,
    ) -> np.ndarray:
        """Get boolean mask for a given regime."""
        if self.regime_classifier is None:
            raise RuntimeError("Regime classifier not trained")
        labels = self.regime_classifier.predict(df)
        regime_map = {"bear": 0, "sideways": 1, "bull": 2}
        target = regime_map.get(regime, -1)
        return labels == target

    def train_regime_specific_models(
        self,
        df: pd.DataFrame,
        data: Dict,
        interval: str,
        target_window: str,
        model_type: str = "lightgbm",
        task: str = "regression",
    ) -> Dict[str, ClassicalModel]:
        """Step 2: Train separate models for each regime.

        Uses regime labels to mask training data, fitting separate models
        for bear, sideways, and bull regimes.
        """
        if self.regime_classifier is None:
            raise RuntimeError("Train regime classifier first")

        self.task = task
        tw = target_window if "h" in str(target_window) else f"{target_window}h"
        short = "reg" if task == "regression" else "clf"

        labels = self.regime_classifier.predict(df)
        regime_names = ["bear", "sideways", "bull"]

        X_all = data["X_train"]
        y_all = data[f"y_{short}_train"]

        feature_names = list(data["feature_names"])
        self.feature_names = feature_names

        n_data = len(X_all)
        train_end = int(n_data * 0.8)
        val_end = min(train_end + int(n_data * 0.15), n_data)

        from config.settings import CLASSICAL_PARAMS
        params = dict(CLASSICAL_PARAMS.get(model_type, {}).get(task, {}))
        if model_type == "lightgbm" and task == "regression":
            params["objective"] = "regression"

        trained_models = {}

        for regime_idx, regime in enumerate(regime_names):
            train_mask = labels[:n_data] == regime_idx
            X_regime_train = X_all[train_mask]
            y_regime_train = y_all[train_mask]

            if len(X_regime_train) < 100:
                logger.warning(f"  Not enough data for regime '{regime}' ({len(X_regime_train)} samples), skipping")
                continue

            # Split regime data into train/val
            split = int(len(X_regime_train) * 0.8)
            X_val_regime = X_regime_train[split:]
            y_val_regime = y_regime_train[split:]
            X_tr_regime = X_regime_train[:split]
            y_tr_regime = y_regime_train[:split]

            if len(X_val_regime) < 10:
                n_val = min(10, len(X_regime_train))
                X_val_regime = X_regime_train[-n_val:]
                y_val_regime = y_regime_train[-n_val:]

            model = ClassicalModel(model_type, task, params)
            model.fit(X_tr_regime, y_tr_regime, X_val_regime, y_val_regime)
            model.feature_names = feature_names
            trained_models[regime] = model

            logger.info(f"  Regime '{regime}': trained on {len(X_tr_regime)} samples")

        self.regime_models[task] = trained_models
        return trained_models

    def train_default_model(
        self,
        data: Dict,
        model_type: str = "lightgbm",
        task: str = "regression",
    ) -> ClassicalModel:
        """Train a default model as fallback when regime confidence is low."""
        short = "reg" if task == "regression" else "clf"

        from config.settings import CLASSICAL_PARAMS
        params = dict(CLASSICAL_PARAMS.get(model_type, {}).get(task, {}))

        self.default_model = ClassicalModel(model_type, task, params)
        self.default_model.fit(
            data["X_train"], data[f"y_{short}_train"],
            data["X_val"], data[f"y_{short}_val"],
        )
        self.default_model.feature_names = list(data["feature_names"])
        logger.info("  Default (fallback) model trained")
        return self.default_model

    def train_volatility_models(
        self,
        df: pd.DataFrame,
    ) -> Tuple[Optional[GARCHEnsemble], Optional[KalmanFilter]]:
        """Train GARCH ensemble and Kalman Filter on price data."""
        returns = df["close"].pct_change().dropna().values

        # GARCH ensemble
        try:
            self.garch_ensemble = GARCHEnsemble()
            self.garch_ensemble.fit(returns)
            logger.info("  GARCH ensemble trained")
        except Exception as e:
            logger.warning(f"  GARCH ensemble failed: {e}")
            self.garch_ensemble = None

        # Kalman Filter
        try:
            prices = df["close"].dropna().values
            self.kalman_filter = KalmanFilter()
            self.kalman_filter.fit(prices)
            logger.info("  Kalman Filter trained")
        except Exception as e:
            logger.warning(f"  Kalman Filter failed: {e}")
            self.kalman_filter = None

        return self.garch_ensemble, self.kalman_filter

    def predict(
        self,
        df_test: pd.DataFrame,
        X_test: np.ndarray,
        regime_confidence_threshold: float = 0.4,
    ) -> Dict[str, np.ndarray]:
        """Regime-aware prediction.

        For each test sample:
        1. Predict regime probability
        2. If regime confidence > threshold, use regime-specific model
        3. Otherwise, fall back to default model

        Args:
            df_test: Test dataframe (for regime features)
            X_test: Test feature matrix
            regime_confidence_threshold: Minimum max probability to trust regime

        Returns:
            Dict with predictions, regime labels, and confidence
        """
        if self.regime_classifier is None:
            raise RuntimeError("Regime classifier not trained")

        regime_proba = self.regime_classifier.predict_proba(df_test)
        regime_labels = self.regime_classifier.predict(df_test)
        max_conf = regime_proba.max(axis=1)

        regime_names = ["bear", "sideways", "bull"]

        use_regime = max_conf >= regime_confidence_threshold

        n = len(X_test)
        predictions = np.zeros(n)
        used_regime = np.full(n, -1)

        task_models = self.regime_models.get(self.task, {})

        for i in range(n):
            if use_regime[i] and task_models:
                regime_idx = regime_labels[i]
                regime_name = regime_names[regime_idx]
                if regime_name in task_models:
                    try:
                        predictions[i] = task_models[regime_name].predict(X_test[i:i+1])[0]
                        used_regime[i] = regime_idx
                        continue
                    except Exception as e:
                        logger.warning(f"Regime model prediction failed for {regime_name}: {e}")

            if self.default_model is not None:
                try:
                    predictions[i] = self.default_model.predict(X_test[i:i+1])[0]
                except Exception as e:
                    logger.warning(f"Default model prediction failed: {e}")
                    predictions[i] = 0.0
            else:
                predictions[i] = 0.0

        stats = {}
        for ri, rn in enumerate(regime_names):
            mask = used_regime == ri
            stats[rn] = int(mask.sum())

        logger.info(f"Regime-aware prediction: {n} samples, "
                     f"regime usage: {stats}, "
                     f"default fallback: {int((used_regime == -1).sum())}")

        return {
            "predictions": predictions,
            "regime_labels": regime_labels,
            "regime_proba": regime_proba,
            "regime_confidence": max_conf,
            "used_regime": used_regime,
        }

    def save(
        self,
        interval: str,
        target_window: str,
    ) -> None:
        """Save all regime-aware components."""
        tw = target_window if "h" in str(target_window) else f"{target_window}h"
        output_dir = MODEL_DIR
        import os
        os.makedirs(output_dir, exist_ok=True)

        if self.regime_classifier:
            self.regime_classifier.save(
                os.path.join(output_dir, f"regime_classifier_{interval}_{tw}.pkl")
            )

        if self.garch_ensemble:
            self.garch_ensemble.save(
                os.path.join(output_dir, f"garch_ensemble_{interval}_{tw}.pkl")
            )

        if self.kalman_filter:
            self.kalman_filter.save(
                os.path.join(output_dir, f"kalman_filter_{interval}_{tw}.pkl")
            )

        if self.regime_models:
            for task, models in self.regime_models.items():
                for regime, model in models.items():
                    model.save(
                        os.path.join(output_dir,
                                     f"regime_{regime}_{task}_{interval}_{tw}.pkl")
                    )

        if self.default_model:
            self.default_model.save(
                os.path.join(output_dir,
                             f"default_model_{self.task}_{interval}_{tw}.pkl")
            )

        logger.info(f"All regime-aware models saved to {output_dir}")

    @classmethod
    def load(
        cls,
        interval: str,
        target_window: str,
    ) -> "RegimeAwareTrainer":
        """Load all regime-aware components."""
        tw = target_window if "h" in str(target_window) else f"{target_window}h"

        trainer = cls()

        import os
        output_dir = MODEL_DIR

        rc_path = os.path.join(output_dir, f"regime_classifier_{interval}_{tw}.pkl")
        if os.path.exists(rc_path):
            trainer.regime_classifier = RegimeClassifier.load(rc_path)

        garch_path = os.path.join(output_dir, f"garch_ensemble_{interval}_{tw}.pkl")
        if os.path.exists(garch_path):
            trainer.garch_ensemble = GARCHEnsemble.load(garch_path)

        kf_path = os.path.join(output_dir, f"kalman_filter_{interval}_{tw}.pkl")
        if os.path.exists(kf_path):
            trainer.kalman_filter = KalmanFilter.load(kf_path)

        return trainer
