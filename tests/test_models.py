"""Tests for machine learning models, ensembles, probability calibration, and Bayesian updater."""

import unittest
import numpy as np

from models.bayes_updater import BayesUpdater
from models.calibration import ProbabilityCalibrator
from models.classical import ClassicalModel
from models.ensemble import StackingEnsemble


class TestModels(unittest.TestCase):

    def setUp(self):
        """Create synthetic regression and classification dataset."""
        np.random.seed(42)
        n = 150
        X = np.random.randn(n, 5)
        y_reg = 0.5 * X[:, 0] - 0.3 * X[:, 1] + np.random.randn(n) * 0.1
        y_clf = (y_reg > 0).astype(int)

        self.data = {
            "X_train": X[:80],
            "y_train_reg": y_reg[:80],
            "y_train_clf": y_clf[:80],
            "X_val": X[80:110],
            "y_val_reg": y_reg[80:110],
            "y_val_clf": y_clf[80:110],
            "X_test": X[110:],
            "y_test_reg": y_reg[110:],
            "y_test_clf": y_clf[110:],
        }

    def test_classical_model_xgboost_regression(self):
        """Test XGBoost regression model training and prediction."""
        model = ClassicalModel(
            model_type="xgboost",
            task="regression",
            params={"n_estimators": 20, "max_depth": 3, "random_state": 42},
        )
        model.fit(
            self.data["X_train"], self.data["y_train_reg"],
            self.data["X_val"], self.data["y_val_reg"],
        )
        preds = model.predict(self.data["X_test"])
        self.assertEqual(len(preds), len(self.data["X_test"]))
        self.assertEqual(preds.ndim, 1)

    def test_classical_model_lightgbm_classification(self):
        """Test LightGBM classification model training and probability prediction."""
        model = ClassicalModel(
            model_type="lightgbm",
            task="classification",
            params={"n_estimators": 20, "max_depth": 3, "min_child_samples": 5, "random_state": 42, "verbose": -1},
        )
        model.fit(
            self.data["X_train"], self.data["y_train_clf"],
            self.data["X_val"], self.data["y_val_clf"],
        )
        preds = model.predict(self.data["X_test"])
        proba = model.predict_proba(self.data["X_test"])
        self.assertEqual(len(preds), len(self.data["X_test"]))
        self.assertEqual(proba.shape[0], len(self.data["X_test"]))

    def test_stacking_ensemble(self):
        """Test StackingEnsemble combining multiple base models."""
        xgb = ClassicalModel(
            model_type="xgboost",
            task="regression",
            params={"n_estimators": 10, "random_state": 42},
        )
        lgb_m = ClassicalModel(
            model_type="lightgbm",
            task="regression",
            params={"n_estimators": 10, "min_child_samples": 5, "random_state": 42, "verbose": -1},
        )
        rf = ClassicalModel(
            model_type="random_forest",
            task="regression",
            params={"n_estimators": 10, "random_state": 42},
        )

        xgb.fit(
            self.data["X_train"], self.data["y_train_reg"],
            self.data["X_val"], self.data["y_val_reg"],
        )
        lgb_m.fit(
            self.data["X_train"], self.data["y_train_reg"],
            self.data["X_val"], self.data["y_val_reg"],
        )
        rf.fit(
            self.data["X_train"], self.data["y_train_reg"],
            self.data["X_val"], self.data["y_val_reg"],
        )

        base_models = {
            "xgboost_reg": xgb,
            "lightgbm_reg": lgb_m,
            "rf_reg": rf,
        }

        ensemble = StackingEnsemble(task="regression", meta_learner_type="ridge")
        ensemble.fit(base_models, self.data["X_val"], None, self.data["y_val_reg"])

        preds = ensemble.predict(base_models, self.data["X_test"], None)
        self.assertEqual(len(preds), len(self.data["X_test"]))

    def test_probability_calibrator(self):
        """Test Platt scaling probability calibration."""
        np.random.seed(42)
        y_true = np.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 1])
        raw_probs = np.array([0.2, 0.3, 0.4, 0.7, 0.8, 0.9, 0.3, 0.6, 0.1, 0.85])

        calibrator = ProbabilityCalibrator(method="sigmoid")
        calibrator.fit(raw_probs, y_true)
        calibrated = calibrator.calibrate(raw_probs)

        self.assertEqual(len(calibrated), len(raw_probs))
        self.assertTrue(np.all(calibrated >= 0.0) and np.all(calibrated <= 1.0))

    def test_bayes_updater(self):
        """Test Bayesian probability updating based on evidence."""
        updater = BayesUpdater(config={"prior_method": "regime", "likelihood_ratio_threshold": 1.1})
        updater.set_prior(0.5)
        self.assertEqual(updater._current_prior, 0.5)

        # Update with evidence
        posterior = updater.update_with_evidence(model_proba=0.85, evidence_strength=1.0)
        self.assertGreater(posterior, 0.5)


if __name__ == "__main__":
    unittest.main()
