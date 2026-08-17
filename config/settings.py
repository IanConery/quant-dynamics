import os
import json
from pathlib import Path

# Load .env file if present
_env_file = Path(__file__).resolve().parent.parent / ".env"
if _env_file.exists():
    with open(_env_file) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _, _val = _line.partition("=")
                os.environ.setdefault(_key.strip(), _val.strip())


def _env(key: str, default: str = "") -> str:
    """Get environment variable."""
    return os.environ.get(key, default)


def _env_float(key: str, default: float = 0.0) -> float:
    """Get environment variable as float."""
    return float(os.environ.get(key, default))


def _env_int(key: str, default: int = 0) -> int:
    """Get environment variable as int."""
    return int(os.environ.get(key, default))


def _env_json(key: str, default: str = "{}") -> dict:
    """Get environment variable as JSON-parsed dict."""
    return json.loads(os.environ.get(key, default))

# API settings
EXCHANGE = "kraken"
SYMBOL = "XRP/USDT"

# Cross-asset symbols for correlation features
CROSS_ASSET_SYMBOLS = ["BTC/USDT", "ETH/USDT"]

# External data sources
EXTERNAL_DATA = {
    "fear_greed_index": {
        "api_url": "https://api.alternative.me/fng/?limit=365",
        "frequency": "daily",
        "value_column": "value",
        "timestamp_column": "timestamp",
    },
}

# Derivatives data config
DERIVATIVES_CONFIG = {
    "enabled": True,
    "exchanges": ["bybit", "binance", "okx"],
    "symbol": "XRP/USDT",
    "funding_rate": {
        "enabled": True,
        "timeframe": "8h",
        "features": ["funding_rate", "funding_rate_zscore", "funding_rate_abs"],
    },
    "open_interest": {
        "enabled": True,
        "features": ["open_interest", "oi_change_pct", "oi_change_24h_pct", "oi_volume_ratio"],
    },
    "liquidations": {
        "enabled": True,
        "features": ["total_liq", "liq_ratio"],
    },
}

# On-chain data config
ON_CHAIN_CONFIG = {
    "enabled": True,
    "sources": {
        "active_addresses": {
            "api": "blockchain.com",
            "features": ["n_requests", "hashrate"],
        },
        "exchange_net_flow": {
            "api": "coingecko",
            "features": ["total_volume_24h", "market_cap", "price_change_24h"],
        },
    },
}

# Macro data config
MACRO_CONFIG = {
    "enabled": True,
    "sources": {
        "vix": {"ticker": "^VIX", "frequency": "daily"},
        "dxy": {"ticker": "DX-Y.NYB", "frequency": "daily"},
        "us10y": {"ticker": "^TNX", "frequency": "daily"},
    },
    "features": ["vix_value", "vix_change", "dxy_value", "dxy_change", "us10y_value", "us10y_change"],
}

# Order book config
ORDER_BOOK_CONFIG = {
    "enabled": True,
    "exchange": "binance",
    "n_levels": 20,
    "features": ["order_book_imbalance", "depth_ratio", "spread_pct"],
}

# Data fetching
INTERVALS = ["15m", "1h", "4h", "1d"]
START_DATE = "2023-01-01"
DATA_DIR = "artifacts/raw_data"
PROCESSED_DIR = "artifacts/processed_data"
MODEL_DIR = "artifacts/models"
BACKTEST_DIR = "artifacts/backtest_results"

# Rate limiting for Binance API
FETCH_LIMIT = 1000
RATE_LIMIT_DELAY = 0.3
RETRY_ATTEMPTS = 3
RETRY_DELAY = 5

# Feature engineering
INDICATORS = {
    "sma": [7, 14, 21, 50, 100, 200],
    "ema": [7, 14, 21, 50],
    "rsi": [7, 14, 21],
    "macd": {"fast": 12, "slow": 26, "signal": 9},
    "bollinger": {"period": 20, "std": 2},
    "atr": [14],
    "stochastic": {"k": 14, "d": 3},
    "adx": [14],
    "obv": [],
    "vwap": [],
    "volume_sma": [7, 14, 21],
}

# Cross-asset feature configuration
CROSS_ASSET_FEATURES = {
    "ratio": True,
    "rolling_correlation": True,
    "market_beta": True,
    "correlation_windows": [20, 60],
    "beta_window": 60,
}

# BTC lead/lag features
BTC_LEAD_LAG = {
    "enabled": True,
    "lead_bars": [1, 2, 4, 8],
    "momentum_windows": [4, 8, 16, 24],
    "ratio_windows": [4, 8, 16],
    "vol_regimes": True,
    "vol_windows": [20, 60],
    "trend_strength": True,
    "adx_period": 14,
}

# Momentum-based features
MOMENTUM_TARGETS = {
    "enabled": True,
    "momentum_windows": [4, 8, 16, 24],
    "acceleration": True,   # momentum_accel_N: change in momentum
    "continuation": True,   # momentum_ratio_N: relative to rolling mean
    "continuation_window": 20,  # rolling mean window for ratio
}

# Volatility-adjusted momentum features
VOL_ADJ_MOMENTUM = {
    "enabled": True,
    "vol_window": 20,       # rolling volatility window
    "momentum_windows": [4, 8, 16, 24],
    "acceleration": True,   # vol_adj_accel_N
    "regime_flags": True,   # vol_regime: 0=low, 1=medium, 2=high
}

# Time-based feature configuration
TIME_FEATURES = {
    "hour_of_day": True,
    "day_of_week": True,
    "is_month_end": True,
    "is_week_end": True,
    "month": True,
    "year_month": True,
}

# Feature selection configuration
FEATURE_SELECTION = {
    "method": "permutation",
    "n_top_features": None,
    "min_importance_threshold": 0.001,
    "test_size_for_selection": 0.15,
}

# Target windows (hours)
PREDICTION_WINDOWS = {
    "15m": [1, 4, 12, 24],
    "1h": [1, 4, 12, 24, 72],
    "4h": [4, 12, 24, 72, 168],
    "1d": [24, 72, 168, 336, 720],
}

# Target definitions
CLASSIFICATION_THRESHOLD = 0.0

# Ternary classification config
TERNARY_CLASSIFICATION = {
    "enabled": True,
    "bands": {
        "15m": 0.001,
        "1h": 0.005,
        "4h": 0.003,
        "1d": 0.01,
    },
    # 0 = DOWN, 1 = SIDEWAYS, 2 = UP
}

# Volatility-normalized targets
VOL_NORMALIZED_TARGETS = {
    "enabled": True,
    "vol_window": 20,  # Rolling window for volatility calculation
}

# Regime features
REGIME_FEATURES = {
    "rolling_sharpe": True,
    "sharpe_window": 20,
    "adx_trend_flag": True,
    "adx_threshold": 25,
    "vol_clustering": True,
    "vol_window": 20,
    "bull_bear_flag": True,
    "bull_bear_sma": 200,
}

# Model hyperparameters
CLASSICAL_PARAMS = {
    "xgboost": {
        "regression": {
            "n_estimators": 1000,
            "max_depth": 8,
            "learning_rate": 0.01,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "objective": "reg:squarederror",
            "early_stopping_rounds": 50,
            "n_jobs": -1,
            "random_state": 42,
        },
        "classification": {
            "n_estimators": 1000,
            "max_depth": 8,
            "learning_rate": 0.01,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "objective": "binary:logistic",
            "early_stopping_rounds": 50,
            "n_jobs": -1,
            "random_state": 42,
        },
    },
    "lightgbm": {
        "regression": {
            "n_estimators": 1000,
            "max_depth": 8,
            "learning_rate": 0.01,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_samples": 20,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "objective": "regression",
            "early_stopping_round": 50,
            "n_jobs": -1,
            "verbose": -1,
            "random_state": 42,
        },
        "classification": {
            "n_estimators": 1000,
            "max_depth": 8,
            "learning_rate": 0.01,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_samples": 20,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "objective": "binary",
            "early_stopping_round": 50,
            "n_jobs": -1,
            "verbose": -1,
            "random_state": 42,
        },
    },
    "random_forest": {
        "regression": {
            "n_estimators": 500,
            "max_depth": 15,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "n_jobs": -1,
            "random_state": 42,
        },
        "classification": {
            "n_estimators": 500,
            "max_depth": 15,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "n_jobs": -1,
            "random_state": 42,
        },
    },
}

DEEP_PARAMS = {
    "lstm": {
        "input_size": "auto",
        "hidden_size": 128,
        "num_layers": 3,
        "dropout": 0.3,
        "seq_length": 60,
        "learning_rate": 0.001,
        "batch_size": 64,
        "epochs": 200,
        "patience": 20,
        "weight_decay": 1e-5,
    },
    "itransformer": {
        "input_size": "auto",
        "d_model": 128,
        "nhead": 8,
        "num_layers": 4,
        "dim_feedforward": 512,
        "dropout": 0.2,
        "seq_length": 60,
        "learning_rate": 0.0005,
        "batch_size": 64,
        "epochs": 200,
        "patience": 20,
        "weight_decay": 1e-5,
    },
    "tft": {
        "input_size": "auto",
        "hidden_size": 64,
        "num_gru_layers": 2,
        "nhead": 4,
        "dropout": 0.1,
        "seq_length": 60,
        "learning_rate": 0.0005,
        "batch_size": 128,
        "epochs": 200,
        "patience": 20,
        "weight_decay": 1e-5,
        "mc_dropout_samples": 10,
        "n_static_features": 8,
    },
}

ENSEMBLE_WEIGHTS = {
    "method": _env("ENSEMBLE_METHOD", "stacking"),
    "meta_learner": _env("ENSEMBLE_META_LEARNER", "ridge"),
    "weights": _env_json("ENSEMBLE_WEIGHTS_JSON", '{"xgboost_reg": 0.2, "lightgbm_reg": 0.2, "rf_reg": 0.2, "lstm_reg": 0.2, "itransformer_reg": 0.2}'),
}

# Time split configuration
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Hyperparameter tuning with Optuna
OPTUNA_CONFIG = {
    "enabled": True,
    "n_trials": 100,
    "timeout_minutes": 120,
    "direction": "maximize",
    "metric": "r2",
    "sampler": "tpe",
    "storage": None,
    "study_name_prefix": "xrp_",
}

# Backtesting config
BACKTEST_CONFIG = {
    "initial_capital": 10000.0,
    "commission_rate": 0.001,
    "slippage": 0.0005,
    "strategies": {
        "signal_based": {
            "buy_threshold": 0.55,
            "sell_threshold": 0.45,
            "hold": True,
        },
        "confidence_based": {
            "min_confidence": 0.70,
            "buy_threshold": 0.55,
            "sell_threshold": 0.45,
        },
    },
    "risk_management": {
        "position_sizing": "fixed_fractional",
        "risk_per_trade": _env_float("BT_RISK_PER_TRADE", 0.02),
        "stop_loss_pct": _env_float("BT_STOP_LOSS", 0.03),
        "take_profit_pct": _env_float("BT_TAKE_PROFIT", 0.06),
        "max_drawdown_circuit_breaker": _env_float("BT_MAX_DD", 0.15),
        "max_concurrent_trades": _env_int("BT_MAX_TRADES", 1),
        "trailing_stop_pct": _env_float("BT_TRAILING_STOP", 0.015),
    },
    "walk_forward": {
        "method": "expanding",
        "step_size": "1M",
        "min_train_samples": 1260,
        "min_test_samples": 720,
        "max_folds": 20,
    },
}

# Regime classification
REGIME_CLASSIFIER = {
    "enabled": True,
    "model_type": "lightgbm",
    "n_classes": 3,
    # 0 = Bear, 1 = Sideways, 2 = Bull
    "features": ["adx", "bb_width", "vol_ratio", "rolling_sharpe", "is_bull"],
    "label_window": 20,
    "thresholds": {
        "bear_return": -0.02,
        "bull_return": 0.02,
        "lookahead": 20,
    },
    "train_params": {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.01,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 20,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "n_jobs": -1,
        "random_state": 42,
    },
}

# GARCH models
GARCH_CONFIG = {
    "enabled": True,
    "models": ["GARCH", "EGARCH"],
    "p": 1,
    "q": 1,
    "distribution": "Normal",
    "power": 2,
    "mean": "Zero",
    "volatility": "GARCH",
}

# State-space models
STATE_SPACE_CONFIG = {
    "enabled": True,
    "kalman": {
        "process_noise": 1e-4,
        "observation_noise": 1e-3,
        "n_components": 2,
    },
    "gaussian_process": {
        "kernel": "RBF + WhiteKernel",
        "n_restarts_optimizer": 10,
        "random_state": 42,
    },
}

# Custom objectives
CUSTOM_OBJECTIVES = {
    "enabled": True,
    "regression": "huber",
    "classification": "f1",
    "huber_delta": 0.35,
}

# Dynamic ensemble weighting
DYNAMIC_ENSEMBLE = {
    "enabled": True,
    "method": "recency_weighted",
    "lookback_periods": 100,
    "decay_factor": 0.95,
    "min_weight": 0.01,
    "reweight_frequency": 50,
}

# Streaming / WebSocket config
STREAMING_CONFIG = {
    "enabled": True,
    "sources": {
        "kraken": {
            "url": "wss://ws.kraken.com",
            "enabled": True,
            "channels": ["ohlc", "trade"],
            "interval": "1h",
            "symbol": "XRP/USDT",
        },
        "coinbase": {
            "url": "wss://ws-feed.exchange.coinbase.com",
            "enabled": False,
            "channels": ["matches", "ticker"],
            "interval": "1h",
            "symbol": "XRP-USD",
        },
        "okx": {
            "url": "wss://ws.okx.com:8443/ws/v5/public",
            "enabled": False,
            "channels": ["funding-rate"],
            "symbol": "XRP-USDT-SWAP",
        },
    },
    "reconnect": {
        "max_retries": 10,
        "base_delay": 1.0,
        "max_delay": 60.0,
        "backoff_factor": 2.0,
    },
    "buffer": {
        "max_size": 10000,
        "flush_interval": 60,
    },
    "state_dir": "artifacts/streaming_state",
}

# Alert config
ALERT_CONFIG = {
    "enabled": True,
    "min_confidence": 0.70,
    "pred_return_threshold": 0.005,
    "alert_on_regime_change": True,
    "alert_file": "artifacts/alerts.jsonl",
    "cooldown_seconds": 300,
}

# Drift monitoring config
DRIFT_CONFIG = {
    "enabled": True,
    "psi_threshold": 0.1,
    "reference_window": 1000,
    "check_frequency": 100,
    "auto_retrain": False,
    "drift_log": "artifacts/drift_log.jsonl",
}

# Binance Vision historical data config
BINANCE_VISION = {
    "enabled": True,
    "base_url": "https://data.binance.vision/data/spot/monthly/klines",
    "symbols": {
        "XRPUSDT": {"start_year": 2019},
        "BTCUSDT": {"start_year": 2019},
        "ETHUSDT": {"start_year": 2019},
    },
    "intervals": ["15m", "1h", "4h", "1d"],
    "download_dir": "artifacts/raw_data/binance",
    "max_retries": 3,
    "retry_delay": 2,
    "rate_limit_delay": 0.5,
}

# Taker flow features from Binance klines
TAKER_FLOW_FEATURES = {
    "enabled": True,
    "features": ["taker_buy_ratio", "taker_buy_quote_ratio", "trade_count", "taker_buy_ratio_ma"],
    "ma_window": 20,
}

# Commission-adjusted training targets (TensorTrade insight 11.9)
TARGET_COST_ADJUSTMENT = {
    "enabled": False,
    "round_trip_commission": 0.0015,
    "round_trip_slippage": 0.001,
}

# Device
try:
    import torch as _torch
    DEVICE = "cuda" if _torch.cuda.is_available() else "cpu"
except ImportError:
    DEVICE = "cpu"

# Triple Barrier Method
TRIPLE_BARRIER = {
    "enabled": True,
    "upper_barrier": _env_float("TB_UPPER_BARRIER", 0.05),
    "lower_barrier": _env_float("TB_LOWER_BARRIER", -0.05),
    "time_horizon_bars": _env_int("TB_TIME_HORIZON", 168),
    "volatility_adjusted": False,
}

# Meta-Label Filter
META_LABEL = {
    "enabled": True,
    "stage1_model": "lightgbm",
    "stage1_features": ["rsi_14", "volume_ratio", "is_bull", "taker_buy_ratio"],
    "stage1_threshold": _env_float("META_LABEL_THRESHOLD", 0.55),
}

# Bayesian Updating Framework
BAYESIAN_UPDATING = {
    "enabled": True,
    "prior_method": "regime",
    "prior_up": _env_float("BAYES_PRIOR_UP", 0.5),
    "regime_base_rates": _env_json("BAYES_REGIME_RATES", '{"bull": 0.60, "sideways": 0.50, "bear": 0.40}'),
    "decay_factor": _env_float("BAYES_DECAY", 0.95),
    "min_lr_threshold": _env_float("BAYES_MIN_LR", 1.1),
    "max_confidence_weight": _env_float("BAYES_MAX_CONF_WEIGHT", 3.0),
}

# Regime-Specific Trading Parameters
REGIME_PARAMS = _env_json("REGIME_PARAMS_JSON", json.dumps({
    "bull": {
        "stop_loss_pct": 0.03,
        "take_profit_pct": 0.06,
        "risk_per_trade": 0.02,
        "trailing_stop_pct": 0.015,
        "max_concurrent_trades": 2,
    },
    "sideways": {
        "stop_loss_pct": 0.03,
        "take_profit_pct": 0.05,
        "risk_per_trade": 0.01,
        "trailing_stop_pct": 0.02,
        "max_concurrent_trades": 1,
    },
    "bear": {
        "stop_loss_pct": 0.03,
        "take_profit_pct": 0.05,
        "risk_per_trade": 0.01,
        "trailing_stop_pct": 0.02,
        "max_concurrent_trades": 1,
    },
}))


def validate_settings() -> None:
    """Validate core configuration settings and environment variables.

    Raises:
        ValueError: If any setting or environment variable is invalid.
    """
    if not EXCHANGE or not isinstance(EXCHANGE, str):
        raise ValueError(f"Invalid EXCHANGE setting: {EXCHANGE!r}. Must be a non-empty string.")
    if not SYMBOL or not isinstance(SYMBOL, str):
        raise ValueError(f"Invalid SYMBOL setting: {SYMBOL!r}. Must be a non-empty string.")

    # Triple Barrier validation
    tb_upper = TRIPLE_BARRIER.get("upper_barrier", 0.0)
    tb_lower = TRIPLE_BARRIER.get("lower_barrier", 0.0)
    tb_horizon = TRIPLE_BARRIER.get("time_horizon_bars", 0)
    if tb_upper <= 0:
        raise ValueError(f"TRIPLE_BARRIER upper_barrier must be > 0, got {tb_upper}")
    if tb_lower >= 0:
        raise ValueError(f"TRIPLE_BARRIER lower_barrier must be < 0, got {tb_lower}")
    if tb_horizon <= 0:
        raise ValueError(f"TRIPLE_BARRIER time_horizon_bars must be > 0, got {tb_horizon}")

    # Meta-label validation
    ml_thresh = META_LABEL.get("stage1_threshold", 0.0)
    if not (0.0 <= ml_thresh <= 1.0):
        raise ValueError(f"META_LABEL stage1_threshold must be in [0, 1], got {ml_thresh}")

    # Bayesian updating validation
    prior_up = BAYESIAN_UPDATING.get("prior_up", 0.5)
    if not (0.0 <= prior_up <= 1.0):
        raise ValueError(f"BAYESIAN_UPDATING prior_up must be in [0, 1], got {prior_up}")
    decay = BAYESIAN_UPDATING.get("decay_factor", 0.95)
    if not (0.0 < decay <= 1.0):
        raise ValueError(f"BAYESIAN_UPDATING decay_factor must be in (0, 1], got {decay}")
    min_lr = BAYESIAN_UPDATING.get("min_lr_threshold", 1.1)
    if min_lr < 1.0:
        raise ValueError(f"BAYESIAN_UPDATING min_lr_threshold must be >= 1.0, got {min_lr}")

    regime_rates = BAYESIAN_UPDATING.get("regime_base_rates", {})
    for regime, rate in regime_rates.items():
        if not (0.0 <= float(rate) <= 1.0):
            raise ValueError(f"BAYESIAN_UPDATING regime rate for '{regime}' must be in [0, 1], got {rate}")

    # Regime params validation
    for regime, params in REGIME_PARAMS.items():
        if not isinstance(params, dict):
            raise ValueError(f"REGIME_PARAMS for '{regime}' must be a dict, got {type(params)}")
        if params.get("stop_loss_pct", 0) <= 0:
            raise ValueError(f"REGIME_PARAMS['{regime}']['stop_loss_pct'] must be > 0")
        if params.get("take_profit_pct", 0) <= 0:
            raise ValueError(f"REGIME_PARAMS['{regime}']['take_profit_pct'] must be > 0")
        if params.get("risk_per_trade", 0) <= 0:
            raise ValueError(f"REGIME_PARAMS['{regime}']['risk_per_trade'] must be > 0")

    # Backtest risk management validation
    rm = BACKTEST_CONFIG.get("risk_management", {})
    if rm.get("risk_per_trade", 0) <= 0:
        raise ValueError("BACKTEST_CONFIG risk_per_trade must be > 0")
    if rm.get("stop_loss_pct", 0) <= 0:
        raise ValueError("BACKTEST_CONFIG stop_loss_pct must be > 0")
    if rm.get("take_profit_pct", 0) <= 0:
        raise ValueError("BACKTEST_CONFIG take_profit_pct must be > 0")
    if rm.get("max_drawdown_circuit_breaker", 0) <= 0:
        raise ValueError("BACKTEST_CONFIG max_drawdown_circuit_breaker must be > 0")

    # Ensemble weights validation
    weights = ENSEMBLE_WEIGHTS.get("weights", {})
    if isinstance(weights, dict):
        for model_name, weight in weights.items():
            if float(weight) < 0:
                raise ValueError(f"ENSEMBLE_WEIGHTS weight for '{model_name}' must be >= 0, got {weight}")


# Run validation on import
validate_settings()

