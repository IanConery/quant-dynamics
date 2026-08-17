from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score

from config.settings import REGIME_CLASSIFIER, MODEL_DIR
from utils.logger import setup_logger

logger = setup_logger(__name__)


class RegimeClassifier:
    """Multi-class regime classifier: Bear (0), Sideways (1), Bull (2).

    Labels are created from future returns over a lookahead window, with
    configurable thresholds. Model is a LightGBM multi-class classifier
    trained with walk-forward validation.
    """

    def __init__(self, config: Dict = None):
        self.config = config or REGIME_CLASSIFIER
        self.model = None
        self.feature_names: List[str] = []
        self.n_classes = self.config.get("n_classes", 3)
        self.thresholds = self.config.get("thresholds", {})

    @staticmethod
    def create_regime_labels(
        close: pd.Series,
        lookahead: int = 20,
        bear_threshold: float = -0.02,
        bull_threshold: float = 0.02,
    ) -> np.ndarray:
        """Create regime labels from future cumulative returns.

        0 = Bear (return < bear_threshold)
        1 = Sideways (bear_threshold <= return <= bull_threshold)
        2 = Bull (return > bull_threshold)
        """
        future_return = close.shift(-lookahead) / close - 1
        labels = np.select(
            [future_return < bear_threshold, future_return > bull_threshold],
            [0, 2],
            default=1,
        )
        return labels

    @staticmethod
    def extract_regime_features(
        df: pd.DataFrame,
        feature_cols: List[str] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """Extract regime-relevant features from processed dataframe.

        Uses a subset of features that are most relevant for regime detection:
        ADX, Bollinger Band width, volume ratio, rolling Sharpe, bull/bear flag,
        plus volatility and trend indicators.
        """
        if feature_cols is None:
            feature_cols = [
                "adx", "plus_di", "minus_di",
                "bb_width", "bb_percent",
                "vol_ratio", "rolling_sharpe",
                "is_trending", "is_bull",
                "rsi_14", "atr_14",
                "volume_ratio",
            ]

        available = [c for c in feature_cols if c in df.columns]
        if len(available) < 3:
            logger.warning(f"Insufficient regime features available: {available}")
            available = [c for c in df.columns
                         if c not in ["timestamp", "open", "high", "low", "close", "volume"]
                         and df[c].dtype in (np.float64, np.float32, np.int64)]
            available = available[:15]

        X = df[available].values.astype(np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X, available

    def fit(
        self,
        df: pd.DataFrame,
        train_end: int,
        val_end: int,
    ) -> "RegimeClassifier":
        """Train regime classifier on chronological split.

        Args:
            df: Processed dataframe with regime features
            train_end: Index to split train/val
            val_end: Index to split val/test
        """
        cfg = self.config
        thresholds = cfg.get("thresholds", {})
        lookahead = thresholds.get("lookahead", 20)
        bear_thresh = thresholds.get("bear_return", -0.02)
        bull_thresh = thresholds.get("bull_return", 0.02)

        close = df["close"]
        labels = self.create_regime_labels(close, lookahead, bear_thresh, bull_thresh)

        X_all, feature_names = self.extract_regime_features(df)
        self.feature_names = feature_names

        valid_mask = ~np.isnan(labels) & ~np.isnan(X_all).any(axis=1)
        X_all = X_all[valid_mask]
        labels = labels[valid_mask].astype(int)

        n = len(X_all)
        te = min(train_end, n)
        ve = min(val_end, n)

        X_train = X_all[:te]
        y_train = labels[:te]
        X_val = X_all[te:ve]
        y_val = labels[te:ve]

        if len(X_val) == 0:
            logger.warning("No validation data for regime classifier")
            return self

        class_counts = np.bincount(y_train, minlength=self.n_classes)
        logger.info(f"Regime class distribution (train): {class_counts}")

        params = dict(cfg.get("train_params", {}))
        params["objective"] = "multiclass"
        params["num_class"] = self.n_classes
        params["metric"] = "multi_logloss"

        self.model = lgb.LGBMClassifier(**params)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
        )

        val_pred = self.model.predict(X_val)
        val_proba = self.model.predict_proba(X_val)
        acc = accuracy_score(y_val, val_pred)
        f1 = f1_score(y_val, val_pred, average="macro", zero_division=0)
        bal_acc = balanced_accuracy_score(y_val, val_pred)

        logger.info(
            f"Regime classifier: val_acc={acc:.4f}, val_f1={f1:.4f}, "
            f"val_bal_acc={bal_acc:.4f}"
        )
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict regime labels for each row."""
        if self.model is None:
            raise RuntimeError("RegimeClassifier not fitted")
        X, _ = self.extract_regime_features(df)
        return self.model.predict(X)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predict regime probabilities."""
        if self.model is None:
            raise RuntimeError("RegimeClassifier not fitted")
        X, _ = self.extract_regime_features(df)
        return self.model.predict_proba(X)

    def get_regime_series(self, df: pd.DataFrame) -> pd.Series:
        """Get regime labels as a pandas Series aligned with df index."""
        labels = self.predict(df)
        regime_names = ["bear", "sideways", "bull"]
        return pd.Series(
            [regime_names.get(int(l), "unknown") for l in labels],
            index=df.index,
        )

    def save(self, path: str) -> None:
        import os
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"Saved regime classifier -> {path}")

    @classmethod
    def load(cls, path: str) -> "RegimeClassifier":
        return joblib.load(path)
