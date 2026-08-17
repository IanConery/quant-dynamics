from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import joblib
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

from config.settings import DYNAMIC_ENSEMBLE
from utils.logger import setup_logger

logger = setup_logger(__name__)


class DynamicWeightingEnsemble:
    """Ensemble with dynamic weight adjustment based on recent model performance.

    Weights are recalculated periodically using recency-weighted performance
    metrics over a sliding window. Models that performed better recently
    receive higher weights.
    """

    def __init__(self, task: str, config: Dict = None):
        self.task = task
        self.config = config or DYNAMIC_ENSEMBLE
        self.model_keys: List[str] = []
        self.weights: Dict[str, float] = {}
        self.performance_history: Dict[str, List[float]] = {}
        self.lookback = self.config.get("lookback_periods", 100)
        self.decay = self.config.get("decay_factor", 0.95)
        self.min_weight = self.config.get("min_weight", 0.01)
        self.reweight_freq = self.config.get("reweight_frequency", 50)
        self.sample_count = 0

    def update_weights(
        self,
        model_preds: Dict[str, np.ndarray],
        y_true: np.ndarray,
    ) -> Dict[str, float]:
        """Update model weights based on recent performance.

        Args:
            model_preds: Dict mapping model name -> predictions
            y_true: Ground truth labels/values

        Returns:
            Updated weights dict
        """
        for key, preds in model_preds.items():
            if key not in self.performance_history:
                self.performance_history[key] = []

            n = min(len(preds), len(y_true))
            if n == 0:
                continue

            pred_aligned = preds[:n]
            true_aligned = y_true[:n]

            if self.task == "regression":
                perf = float(np.mean((pred_aligned - true_aligned) ** 2))
                perf = -perf
            else:
                pred_clf = (pred_aligned > 0.5).astype(int)
                acc = float(np.mean(pred_clf == true_aligned[:n]))
                perf = acc

            self.performance_history[key].append(perf)
            if len(self.performance_history[key]) > self.lookback:
                self.performance_history[key] = self.performance_history[key][-self.lookback:]

        self.sample_count += n
        if self.sample_count >= self.reweight_freq:
            self.sample_count = 0
            self._recalculate_weights()

        return dict(self.weights)

    def _recalculate_weights(self) -> None:
        """Recalculate weights using recency-decayed performance scores."""
        if not self.performance_history:
            n_models = max(len(self.model_keys), 1)
            self.weights = {k: 1.0 / n_models for k in self.model_keys}
            return

        scores = {}
        for key, history in self.performance_history.items():
            if not history:
                scores[key] = 0.0
                continue

            decayed = np.array(history) * (self.decay ** np.arange(len(history))[::-1])
            scores[key] = float(np.mean(decayed))

        total = sum(max(s, -np.inf) for s in scores.values())
        if total <= 0 or np.isinf(total):
            n_models = len(scores)
            self.weights = {k: 1.0 / n_models for k in scores}
            return

        self.weights = {k: max(s / total, self.min_weight) for k, s in scores.items()}

        weight_total = sum(self.weights.values())
        self.weights = {k: w / weight_total for k, w in self.weights.items()}

    def predict_weighted(
        self,
        model_preds: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Compute weighted average prediction.

        Args:
            model_preds: Dict mapping model name -> predictions

        Returns:
            Weighted ensemble predictions
        """
        if not self.weights:
            self.model_keys = list(model_preds.keys())
            self._recalculate_weights()

        stacked = []
        weights = []
        for key in model_preds:
            if key in self.weights:
                stacked.append(model_preds[key])
                weights.append(self.weights[key])

        if not stacked:
            return np.array(list(model_preds.values())[0])

        arr = np.column_stack(stacked)
        w = np.array(weights)
        w = w / w.sum()
        return arr @ w

    def save(self, path: str) -> None:
        import os
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"Saved dynamic ensemble -> {path}")

    @classmethod
    def load(cls, path: str) -> "DynamicWeightingEnsemble":
        return joblib.load(path)


class StackingEnsemble:
    """
    Stacking ensemble: base model predictions become features for a meta-learner.
    Out-of-fold predictions on validation set train the meta-learner.
    """

    def __init__(self, task: str, meta_learner_type: str = "ridge"):
        self.task = task
        self.meta_learner_type = meta_learner_type
        self.meta_model = self._create_meta_learner()
        self.base_model_keys: List[str] = []

    def _create_meta_learner(self):
        if self.task == "regression":
            if self.meta_learner_type == "ridge":
                return Ridge(alpha=1.0)
            elif self.meta_learner_type == "lightgbm":
                return GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)
        else:
            if self.meta_learner_type == "logistic_regression":
                return LogisticRegression(C=1.0, max_iter=1000)
            elif self.meta_learner_type == "lightgbm":
                return GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
        return Ridge(alpha=1.0) if self.task == "regression" else LogisticRegression(C=1.0)

    def fit(
        self,
        base_models: Dict[str, Any],
        X_val: np.ndarray,
        X_val_seq: Optional[np.ndarray],
        y_val: np.ndarray,
    ) -> "StackingEnsemble":
        self.base_model_keys = sorted(base_models.keys())

        meta_features = []
        n_seq = len(X_val_seq) if X_val_seq is not None else 0
        n_flat = len(X_val)
        min_n = min(n_seq, n_flat) if n_seq > 0 else n_flat

        for key in self.base_model_keys:
            model = base_models[key]
            is_seq = "lstm" in key or "itransformer" in key or "tft" in key
            X = X_val_seq if is_seq else X_val

            if is_seq:
                pred = model.predict(X).flatten()
            else:
                pred = model.predict(X)

            meta_features.append(pred[:min_n])

        meta_X = np.column_stack(meta_features)

        if np.isnan(meta_X).any():
            meta_X = np.nan_to_num(meta_X, nan=0.0)

        y_aligned = y_val[:len(meta_X)]
        self.meta_model.fit(meta_X, y_aligned)
        logger.info(f"  Ensemble ({self.task}): meta-learner trained on {len(self.base_model_keys)} base models")
        return self

    def predict(self, base_models: Dict[str, Any],
                X: np.ndarray, X_seq: Optional[np.ndarray]) -> np.ndarray:
        meta_features = []
        n_seq = len(X_seq) if X_seq is not None else 0
        n_flat = len(X)
        min_n = min(n_seq, n_flat) if n_seq > 0 else n_flat

        for key in self.base_model_keys:
            model = base_models[key]
            is_seq = "lstm" in key or "itransformer" in key or "tft" in key
            inp = X_seq if is_seq else X

            if is_seq:
                pred = model.predict(inp).flatten()
            else:
                pred = model.predict(inp)

            meta_features.append(pred[:min_n])

        meta_X = np.column_stack(meta_features)
        if np.isnan(meta_X).any():
            meta_X = np.nan_to_num(meta_X, nan=0.0)

        return self.meta_model.predict(meta_X)

    def predict_proba(self, base_models: Dict[str, Any],
                       X: np.ndarray, X_seq: Optional[np.ndarray]) -> np.ndarray:
        if self.task == "classification":
            preds = self.predict(base_models, X, X_seq)
            return preds.reshape(-1, 1)
        return self.predict(base_models, X, X_seq).reshape(-1, 1)

    def save(self, path: str) -> None:
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"  Saved ensemble ({self.task}) -> {path}")

    @classmethod
    def load(cls, path: str) -> "StackingEnsemble":
        return joblib.load(path)
