"""Real-time model predictor.

Loads trained models and runs inference on streaming data.
"""

import asyncio
import json
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from config.settings import MODEL_DIR
from models.classical import ClassicalModel
from models.deep import DeepModel
from models.ensemble import StackingEnsemble
from utils.logger import setup_logger

logger = setup_logger(__name__)


class StreamPredictor:
    """Real-time inference engine for streaming predictions.

    Maintains a rolling feature buffer and runs inference when new data arrives.
    """

    def __init__(
        self,
        interval: str,
        target_window: str,
        model_type: str = "ensemble",
        model_dir: str = None,
        feature_buffer_size: int = 200,
    ):
        self.interval = interval
        self.target_window = target_window if "h" in str(target_window) else f"{target_window}h"
        self.model_type = model_type
        self.model_dir = model_dir or os.path.join(MODEL_DIR, interval, self.target_window)
        self.feature_buffer_size = feature_buffer_size
        self.model = None
        self.base_models = {}
        self.feature_names: List[str] = []
        self.feature_buffer: List[Dict] = []
        self.last_prediction_time = 0
        self.prediction_history: List[Dict] = []
        self._loaded = False

    def load_models(self) -> bool:
        """Load trained models from disk."""
        try:
            from data.loader import load_processed, get_datasets

            df = load_processed(self.interval)
            data = get_datasets(self.interval, self.target_window)
            self._feature_columns = list(data["feature_names"])
            self.scaler = data.get("scaler", None)

            if self.model_type == "ensemble":
                return self._load_ensemble(df)
            else:
                return self._load_individual(df)
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            return False

    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get feature columns from processed data."""
        from data.processor import _get_feature_columns
        all_features = _get_feature_columns(df)
        return all_features

    def _load_ensemble(self, df: pd.DataFrame) -> bool:
        """Load ensemble model and base models."""
        ens_path = os.path.join(self.model_dir, "ensemble_clf.pkl")
        if not os.path.exists(ens_path):
            logger.error(f"Ensemble model not found: {ens_path}")
            return False

        try:
            self.model = StackingEnsemble.load(ens_path)

            n_features = len(self._feature_columns)
            for fname in os.listdir(self.model_dir):
                if "clf" in fname and "ensemble" not in fname:
                    fpath = os.path.join(self.model_dir, fname)
                    if fname.endswith(".pkl"):
                        try:
                            m = ClassicalModel.load(fpath)
                            actual_n = m.n_features or getattr(m.model, 'n_features_in_', None)
                            if actual_n is not None and actual_n != n_features:
                                logger.warning(f"Skipping {fname}: feature mismatch ({actual_n} vs {n_features})")
                                continue
                            self.base_models[fname] = m
                        except Exception as e:
                            logger.warning(f"Failed to load {fname}: {e}")
                    elif fname.endswith(".pth"):
                        try:
                            m = DeepModel.load(fpath)
                            if m.input_size != n_features:
                                logger.warning(f"Skipping {fname}: input_size mismatch ({m.input_size} vs {n_features})")
                                continue
                            self.base_models[fname] = m
                        except Exception as e:
                            logger.warning(f"Failed to load {fname}: {e}")

            self.feature_names = self._feature_columns
            self._loaded = True
            logger.info(f"Loaded ensemble with {len(self.base_models)} compatible base models, "
                         f"{len(self.feature_names)} features")
            return True
        except Exception as e:
            logger.error(f"Failed to load ensemble: {e}")
            return False

    def _load_individual(self, df: pd.DataFrame) -> bool:
        """Load individual model."""
        for ext in [".pkl", ".pth"]:
            mpath = os.path.join(self.model_dir, f"{self.model_type}_clf{ext}")
            if os.path.exists(mpath):
                try:
                    if ext == ".pkl":
                        self.model = ClassicalModel.load(mpath)
                    else:
                        self.model = DeepModel.load(mpath)
                    self.feature_names = self._feature_columns
                    self._loaded = True
                    logger.info(f"Loaded model: {self.model_type}")
                    return True
                except Exception as e:
                    logger.error(f"Failed to load {self.model_type}: {e}")
        return False

    def update_feature_buffer(self, candle: Dict) -> None:
        """Add a new candle to the rolling feature buffer.

        Args:
            candle: OHLCV candle dict with timestamp, open, high, low, close, volume
        """
        self.feature_buffer.append(candle)
        if len(self.feature_buffer) > self.feature_buffer_size:
            self.feature_buffer = self.feature_buffer[-self.feature_buffer_size:]

    def compute_features(self) -> Optional[np.ndarray]:
        """Compute features from the current buffer.

        Returns feature vector for the latest candle.
        """
        if len(self.feature_buffer) < 10:
            return None

        df = pd.DataFrame(self.feature_buffer)
        if "timestamp" not in df.columns:
            return None

        # Compute all indicators from buffer
        close = df["close"].values.astype(float)
        high = df["high"].values.astype(float) if "high" in df.columns else close
        low = df["low"].values.astype(float) if "low" in df.columns else close
        volume = df["volume"].values.astype(float) if "volume" in df.columns else np.zeros_like(close)
        n = len(close)

        features = {}
        features["close"] = close[-1]
        features["volume"] = volume[-1]

        # Returns
        for lag in [1, 2, 3, 5, 10, 20]:
            if n > lag:
                features[f"return_{lag}"] = (close[-1] / close[-lag]) - 1
                features[f"close_lag_{lag}"] = close[-lag]

        # SMA
        for w in [7, 14, 21, 50, 100, 200]:
            if n >= w:
                features[f"sma_{w}"] = float(np.mean(close[-w:]))
            else:
                features[f"sma_{w}"] = close[-1]

        # EMA
        for w in [7, 14, 21, 50]:
            if n >= w:
                alpha = 2.0 / (w + 1)
                ema = close[-w]
                for i in range(1, w):
                    ema = close[-w + i] * alpha + ema * (1 - alpha)
                features[f"ema_{w}"] = float(ema)
            else:
                features[f"ema_{w}"] = close[-1]

        # RSI
        for p in [7, 14, 21]:
            if n >= p + 1:
                deltas = np.diff(close[-(p+1):])
                gains = np.maximum(deltas, 0)
                losses = np.maximum(-deltas, 0)
                avg_gain = np.mean(gains)
                avg_loss = np.mean(losses)
                rs = avg_gain / (avg_loss + 1e-10)
                features[f"rsi_{p}"] = 100 - (100 / (1 + rs))
            else:
                features[f"rsi_{p}"] = 50

        # MACD
        if n >= 26:
            ema12 = self._ema(close, 12)
            ema26 = self._ema(close, 26)
            macd = ema12 - ema26
            features["macd"] = float(macd)
            if n >= 35:
                signal_period = min(9, n - 26)
                macd_series = []
                for i in range(max(0, n - 35), n):
                    e12 = self._ema(close[:i+1], 12)
                    e26 = self._ema(close[:i+1], 26) if i >= 26 else e12
                    macd_series.append(e12 - e26)
                if macd_series:
                    alpha_s = 2.0 / (signal_period + 1)
                    sig = macd_series[0]
                    for v in macd_series[1:]:
                        sig = v * alpha_s + sig * (1 - alpha_s)
                    features["macd_signal"] = float(sig)
                    features["macd_histogram"] = float(macd - sig)
                else:
                    features["macd_signal"] = 0
                    features["macd_histogram"] = 0
            else:
                features["macd_signal"] = 0
                features["macd_histogram"] = 0
        else:
            features["macd"] = 0
            features["macd_signal"] = 0
            features["macd_histogram"] = 0

        # Bollinger Bands
        if n >= 20:
            sma20 = np.mean(close[-20:])
            std20 = np.std(close[-20:])
            features["bb_upper"] = sma20 + 2 * std20
            features["bb_lower"] = sma20 - 2 * std20
            features["bb_width"] = (4 * std20) / (sma20 + 1e-10)
            features["bb_percent"] = (close[-1] - features["bb_lower"]) / (features["bb_upper"] - features["bb_lower"] + 1e-10)
        else:
            features["bb_upper"] = features["bb_lower"] = features["bb_width"] = features["bb_percent"] = 0

        # ATR
        if n >= 15:
            trs = []
            for i in range(1, n):
                tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
                trs.append(tr)
            features["atr_14"] = float(np.mean(trs[-14:]))
        else:
            features["atr_14"] = 0

        # Stochastic
        if n >= 14:
            lowest_low = np.min(low[-14:])
            highest_high = np.max(high[-14:])
            range_ = highest_high - lowest_low + 1e-10
            features["stoch_k"] = 100 * (close[-1] - lowest_low) / range_
            if n >= 17:
                k_vals = []
                for i in range(n - 14, n):
                    ll = np.min(low[max(0, i-13):i+1])
                    hh = np.max(high[max(0, i-13):i+1])
                    r = hh - ll + 1e-10
                    k_vals.append(100 * (close[i] - ll) / r)
                features["stoch_d"] = float(np.mean(k_vals[-3:]))
            else:
                features["stoch_d"] = features["stoch_k"]
        else:
            features["stoch_k"] = features["stoch_d"] = 50

        # ADX
        features["adx"] = 25.0
        features["plus_di"] = 25.0
        features["minus_di"] = 25.0

        # Volume SMA
        for w in [7, 14, 21]:
            if n >= w:
                features[f"volume_sma_{w}"] = float(np.mean(volume[-w:]))
            else:
                features[f"volume_sma_{w}"] = volume[-1] if n > 0 else 0

        if n >= 21:
            features["volume_ratio"] = volume[-1] / (np.mean(volume[-21:]) + 1e-10)
        else:
            features["volume_ratio"] = 1.0

        # Price patterns
        features["body_size"] = abs(close[-1] - close[-2]) / (close[-2] + 1e-10)
        features["upper_wick"] = (high[-1] - max(close[-1], close[-2])) / (close[-2] + 1e-10)
        features["lower_wick"] = (min(close[-1], close[-2]) - low[-1]) / (close[-2] + 1e-10)
        features["is_bullish"] = 1.0 if close[-1] > close[-2] else 0.0
        body_r = abs(close[-1] - close[-2]) / (high[-1] - low[-1] + 1e-10)
        features["body_ratio"] = body_r

        # Time features
        ts = df["timestamp"].iloc[-1]
        features["hour_sin"] = np.sin(2 * np.pi * ts.hour / 24)
        features["hour_cos"] = np.cos(2 * np.pi * ts.hour / 24)
        features["dow_sin"] = np.sin(2 * np.pi * ts.dayofweek / 7)
        features["dow_cos"] = np.cos(2 * np.pi * ts.dayofweek / 7)
        features["month_sin"] = np.sin(2 * np.pi * ts.month / 12)
        features["month_cos"] = np.cos(2 * np.pi * ts.month / 12)
        features["is_month_end"] = 1.0 if ts.day >= 28 else 0.0
        features["is_weekend"] = 1.0 if ts.dayofweek >= 5 else 0.0

        # Regime features
        if n >= 21:
            ret = np.diff(close[-21:]) / (close[-21:-1] + 1e-10)
            mean_ret = np.mean(ret)
            std_ret = np.std(ret)
            features["rolling_sharpe"] = mean_ret / (std_ret + 1e-10)
            features["vol_ratio"] = 1.0
        else:
            features["rolling_sharpe"] = 0
            features["vol_ratio"] = 1.0

        features["is_trending"] = 0.0 if features.get("adx", 25) <= 25 else 1.0
        features["is_bull"] = 1.0 if n >= 200 and close[-1] > np.mean(close[-200:]) else 0.5

        if not self.feature_names:
            return None

        X = np.zeros(len(self.feature_names))
        for i, fname in enumerate(self.feature_names):
            if fname in features:
                X[i] = features[fname]

        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X.reshape(1, -1)

    @staticmethod
    def _ema(arr: np.ndarray, span: int) -> float:
        """Compute EMA of an array."""
        if len(arr) < span:
            return float(arr[-1])
        alpha = 2.0 / (span + 1)
        ema = arr[-span]
        for i in range(1, span):
            ema = arr[-span + i] * alpha + ema * (1 - alpha)
        return float(ema)

    def predict(self, X: np.ndarray) -> Optional[Dict]:
        """Run inference on feature vector.

        Args:
            X: Feature matrix (1, n_features)

        Returns:
            Prediction dict with probabilities, direction, confidence
        """
        if self.model is None:
            return None

        try:
            proba = None

            if isinstance(self.model, StackingEnsemble):
                # Only use loaded base models
                loaded_keys = list(self.base_models.keys())
                if loaded_keys != self.model.base_model_keys:
                    self.model.base_model_keys = loaded_keys

                if len(self.base_models) >= 2:
                    X_seq = self._build_sequence(X)
                    proba = self.model.predict(self.base_models, X, X_seq)
                elif len(self.base_models) == 1:
                    # Fallback to single model
                    single_model = list(self.base_models.values())[0]
                    proba = single_model.predict_proba(X)
                    if proba.ndim > 1:
                        proba = proba[:, 1]
                else:
                    logger.warning("No compatible base models for ensemble")
                    proba = np.array([0.5])
            elif hasattr(self.model, "predict_proba"):
                proba = self.model.predict_proba(X)
                if proba.ndim > 1:
                    proba = proba[:, 1]
            else:
                proba = self.model.predict(X)

            proba = float(proba.flatten()[0]) if hasattr(proba, 'flatten') else float(proba)
            direction = "UP" if proba >= 0.5 else "DOWN"
            confidence = max(proba, 1 - proba)

            result = {
                "timestamp": pd.Timestamp.now(tz="UTC"),
                "P_UP": round(proba, 6),
                "direction": direction,
                "confidence": round(confidence, 6),
            }
            self.prediction_history.append(result)
            return result

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return None

    def _build_sequence(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Build sequence data from buffer for deep models."""
        if len(self.feature_buffer) < 60:
            return None

        sequences = []
        for i in range(-60, 0):
            candle = self.feature_buffer[i]
            feat = self._candle_to_features(candle)
            sequences.append(feat)

        return np.array(sequences).reshape(1, 60, -1)

    def _candle_to_features(self, candle: Dict) -> List[float]:
        """Convert a single candle to feature values."""
        return [
            candle.get("open", 0),
            candle.get("high", 0),
            candle.get("low", 0),
            candle.get("close", 0),
            candle.get("volume", 0),
        ]

    def get_prediction_history(self, n: int = 50) -> List[Dict]:
        """Get recent prediction history."""
        return self.prediction_history[-n:]

    def save_state(self, path: str = None) -> None:
        """Save feature buffer state to disk."""
        if path is None:
            from config.settings import STREAMING_CONFIG
            state_dir = STREAMING_CONFIG.get("state_dir", "artifacts/streaming_state")
            os.makedirs(state_dir, exist_ok=True)
            path = os.path.join(state_dir, f"buffer_{self.interval}.json")

        state = {
            "feature_buffer": [
                {k: str(v) if isinstance(v, pd.Timestamp) else v
                 for k, v in c.items()}
                for c in self.feature_buffer[-100:]
            ],
            "prediction_history": self.prediction_history[-100:],
        }

        with open(path, "w") as f:
            json.dump(state, f, default=str)
        logger.info(f"State saved: {path}")

    def load_state(self, path: str = None) -> bool:
        """Load feature buffer state from disk."""
        if path is None:
            from config.settings import STREAMING_CONFIG
            state_dir = STREAMING_CONFIG.get("state_dir", "artifacts/streaming_state")
            path = os.path.join(state_dir, f"buffer_{self.interval}.json")

        if not os.path.exists(path):
            return False

        try:
            with open(path) as f:
                state = json.load(f)

            for entry in state.get("feature_buffer", []):
                if "timestamp" in entry:
                    entry["timestamp"] = pd.Timestamp(entry["timestamp"])
                for key in ["open", "high", "low", "close", "volume"]:
                    if key in entry:
                        entry[key] = float(entry[key])
                self.feature_buffer.append(entry)

            self.prediction_history = state.get("prediction_history", [])
            logger.info(f"State loaded: {len(self.feature_buffer)} candles, {len(self.prediction_history)} predictions")
            return True
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            return False
