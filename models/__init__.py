"""Machine learning and deep learning models package."""

from models.bayes_updater import BayesUpdater
from models.calibration import (
    ProbabilityCalibrator,
    calibrate_platt,
)
from models.classical import ClassicalModel
from models.deep import (
    DeepModel,
    LSTMModel,
    iTransformerModel,
)
from models.ensemble import (
    DynamicWeightingEnsemble,
    StackingEnsemble,
)
from models.garch import (
    GARCHEnsemble,
    GARCHModel,
)
from models.regime_classifier import RegimeClassifier
from models.regime_trainer import RegimeAwareTrainer
from models.state_space import (
    GaussianProcessRegressor,
    KalmanFilter,
)
from models.tft import TFTModel
from models.trainer import (
    calibrate_model,
    evaluate_all,
    evaluate_model,
    feature_importance_report,
    optuna_tune,
    temporal_cv_evaluate,
    train_classical,
    train_deep,
    train_ensemble,
)

__all__ = [
    # Classical & Deep
    "ClassicalModel",
    "DeepModel",
    "LSTMModel",
    "iTransformerModel",
    "TFTModel",
    # Ensembles
    "StackingEnsemble",
    "DynamicWeightingEnsemble",
    # Regime & Volatility
    "RegimeClassifier",
    "RegimeAwareTrainer",
    "GARCHModel",
    "GARCHEnsemble",
    "KalmanFilter",
    "GaussianProcessRegressor",
    # Calibration & Bayesian
    "ProbabilityCalibrator",
    "calibrate_platt",
    "BayesUpdater",
    # Trainer functions
    "train_classical",
    "train_deep",
    "train_ensemble",
    "evaluate_model",
    "evaluate_all",
    "optuna_tune",
    "temporal_cv_evaluate",
    "calibrate_model",
    "feature_importance_report",
]
