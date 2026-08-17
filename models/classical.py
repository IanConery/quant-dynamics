from typing import Any, Dict, List, Optional

import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import joblib

from config.settings import CLASSICAL_PARAMS, CUSTOM_OBJECTIVES
from utils.logger import setup_logger

logger = setup_logger(__name__)


# ─── Custom Objectives ─────────────────────────────────────────────────────


def huber_objective(y_pred: np.ndarray, y_true: np.ndarray, delta: float = 0.35):
    """Huber loss objective for LightGBM regression.

    Combines MSE for small errors and MAE for large errors (robust to outliers).
    """
    residual = y_pred - y_true
    abs_res = np.abs(residual)
    is_small = abs_res <= delta
    grad = np.where(is_small, residual, delta * np.sign(residual))
    hess = np.where(is_small, 1.0, 0.0)
    hess = np.where(hess == 0, 1e-8, hess)
    return grad, hess


def f1_objective(y_pred: np.ndarray, y_true: np.ndarray, threshold: float = 0.5):
    """F1-score optimized objective for LightGBM classification.

    Uses smooth approximation of F1 gradient for differentiable optimization.
    """
    y_pred_proba = 1.0 / (1.0 + np.exp(-y_pred))
    y_pred_proba = np.clip(y_pred_proba, 1e-7, 1 - 1e-7)
    y_true = y_true.astype(float)

    pred_binary = (y_pred_proba >= threshold).astype(float)
    tp = np.sum(pred_binary * y_true)
    fp = np.sum(pred_binary * (1 - y_true))
    fn = np.sum((1 - pred_binary) * y_true)

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    beta = 1.0
    beta_sq = beta ** 2

    f1 = (1 + beta_sq) * precision * recall / (beta_sq * precision + recall + 1e-8)

    grad = (y_true - y_pred_proba) * 2
    hess = y_pred_proba * (1 - y_pred_proba) * 2
    hess = np.maximum(hess, 1e-8)
    return grad, hess


def xgb_huber_objective(y_pred, y_true, delta: float = 0.35):
    """Huber loss objective for XGBoost."""
    y_true = y_true.get_label()
    residual = y_pred - y_true
    abs_res = np.abs(residual)
    grad = np.where(abs_res <= delta, residual, delta * np.sign(residual))
    hess = np.where(abs_res <= delta, 1.0, 1e-8)
    return grad, hess


class ClassicalModel:
    """Wrapper around sklearn-compatible models with unified interface."""

    def __init__(self, model_type: str, task: str, params: Dict, n_classes: int = 2):
        self.model_type = model_type
        self.task = task
        self.params = params
        self.n_classes = n_classes
        self.feature_names = None
        self.n_features = None
        self.model = self._instantiate()

    def _instantiate(self):
        if self.model_type == "xgboost":
            if self.task == "regression":
                return xgb.XGBRegressor(**self.params)
            if self.n_classes > 2:
                p = dict(self.params)
                p["objective"] = "multi:softprob"
                p["num_class"] = self.n_classes
                return xgb.XGBClassifier(**p)
            return xgb.XGBClassifier(**self.params)
        elif self.model_type == "lightgbm":
            if self.task == "regression":
                return lgb.LGBMRegressor(**self.params)
            if self.n_classes > 2:
                p = dict(self.params)
                p["objective"] = "multiclass"
                p["num_class"] = self.n_classes
                return lgb.LGBMClassifier(**p)
            return lgb.LGBMClassifier(**self.params)
        elif self.model_type == "random_forest":
            if self.task == "regression":
                return RandomForestRegressor(**self.params)
            return RandomForestClassifier(**self.params)
        raise ValueError(f"Unknown model type: {self.model_type}")

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray,
            custom_objective: str = None) -> "ClassicalModel":
        self.n_features = X_train.shape[1]
        if self.model_type == "random_forest":
            self.model.fit(X_train, y_train)
            logger.info(f"  {self.model_type}/{self.task}: trained (no early stopping)")
        else:
            eval_set = [(X_train, y_train), (X_val, y_val)]
            if self.model_type == "xgboost":
                if self.task == "classification" and self.n_classes == 2:
                    n_pos = int(y_train.sum())
                    n_neg = len(y_train) - n_pos
                    scale_pos = max(n_neg / max(n_pos, 1), 1)
                    self.model.fit(
                        X_train, y_train,
                        eval_set=eval_set,
                        verbose=False,
                        sample_weight=np.where(y_train == 1, scale_pos, 1.0),
                    )
                else:
                    self.model.fit(
                        X_train, y_train,
                        eval_set=eval_set,
                        verbose=False,
                    )
            else:
                if self.task == "classification" and self.n_classes == 2:
                    n_pos = int(y_train.sum())
                    n_neg = len(y_train) - n_pos
                    scale_pos = max(n_neg / max(n_pos, 1), 1)
                    self.model.fit(
                        X_train, y_train,
                        eval_set=eval_set,
                        sample_weight=np.where(y_train == 1, scale_pos, 1.0),
                    )
                else:
                    self.model.fit(
                        X_train, y_train,
                        eval_set=eval_set,
                    )
            best_iter = getattr(self.model, "best_iteration", None)
            logger.info(
                f"  {self.model_type}/{self.task}: trained "
                f"(best_iteration={best_iter})"
            )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.n_features and X.shape[1] != self.n_features:
            raise ValueError(
                f"Feature mismatch: model expects {self.n_features} features, got {X.shape[1]}. "
                f"Re-train the model with matching features or re-run feature selection."
            )
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.n_features and X.shape[1] != self.n_features:
            raise ValueError(
                f"Feature mismatch: model expects {self.n_features} features, got {X.shape[1]}. "
                f"Re-train the model with matching features or re-run feature selection."
            )
        if self.task == "classification":
            probs = self.model.predict_proba(X)
            if self.n_classes == 2:
                return probs[:, 1]
            return probs
        return self.predict(X).reshape(-1, 1)

    def get_feature_importances(self) -> np.ndarray:
        raw = getattr(self.model, "feature_importances_", None)
        if raw is None:
            return np.zeros(0)
        total = raw.sum()
        return raw / total if total > 0 else raw

    def save(self, path: str) -> None:
        joblib.dump(self, path)
        logger.info(f"  Saved {self.model_type}/{self.task} -> {path}")

    @classmethod
    def load(cls, path: str) -> "ClassicalModel":
        model = joblib.load(path)
        if not hasattr(model, "n_features"):
            model.n_features = None
        if not hasattr(model, "feature_names"):
            model.feature_names = None
        if not hasattr(model, "n_classes"):
            model.n_classes = 2
        return model


def get_all_classical_models(task: str, n_classes: int = 2) -> List[ClassicalModel]:
    models = []
    for model_type in CLASSICAL_PARAMS:
        params = CLASSICAL_PARAMS[model_type].get(task, {})
        if params:
            models.append(ClassicalModel(model_type, task, params, n_classes=n_classes))
    return models
