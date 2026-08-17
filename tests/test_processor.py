"""Tests for full data processing pipeline and feature matrix builders."""

import unittest
import numpy as np
import pandas as pd

from data.processor import (
    _get_feature_columns,
    audit_scale_invariance,
    build_feature_matrix,
    build_feature_matrix_triple_barrier,
    process_raw_data,
    select_features,
)


class TestProcessor(unittest.TestCase):

    def setUp(self):
        """Create a synthetic multi-asset dataset."""
        np.random.seed(42)
        n = 200
        dates = pd.date_range("2024-01-01", periods=n, freq="1h")
        xrp_close = 1.0 + np.cumsum(np.random.randn(n) * 0.01)
        btc_close = 50000.0 + np.cumsum(np.random.randn(n) * 100)
        eth_close = 3000.0 + np.cumsum(np.random.randn(n) * 10)

        self.df = pd.DataFrame({
            "timestamp": dates,
            "open": xrp_close - 0.005,
            "high": xrp_close + 0.01,
            "low": xrp_close - 0.01,
            "close": xrp_close,
            "volume": np.random.uniform(10000, 500000, n),
        })
        self.btc_df = pd.DataFrame({
            "timestamp": dates,
            "open": btc_close,
            "high": btc_close + 50,
            "low": btc_close - 50,
            "close": btc_close,
            "volume": np.random.uniform(100, 500, n),
        })
        self.eth_df = pd.DataFrame({
            "timestamp": dates,
            "open": eth_close,
            "high": eth_close + 5,
            "low": eth_close - 5,
            "close": eth_close,
            "volume": np.random.uniform(500, 2000, n),
        })
        self.fng_df = pd.DataFrame({
            "timestamp": dates[::24],
            "value": np.random.uniform(20, 80, len(dates[::24])),
        })

    def test_process_raw_data_pipeline(self):
        """Smoke test full process_raw_data pipeline."""
        processed = process_raw_data(self.df.copy(), btc_df=self.btc_df, eth_df=self.eth_df, fng_df=self.fng_df)

        self.assertEqual(len(processed), len(self.df))
        self.assertGreater(len(processed.columns), len(self.df.columns) + 20)

        feature_cols = _get_feature_columns(processed)
        self.assertGreater(len(feature_cols), 0)
        self.assertNotIn("close", feature_cols)
        self.assertNotIn("timestamp", feature_cols)

    def test_build_feature_matrix(self):
        """Test feature matrix extraction with targets."""
        processed = process_raw_data(self.df.copy(), btc_df=self.btc_df, eth_df=self.eth_df, fng_df=self.fng_df)

        from data.features.targets import create_targets
        processed = create_targets(processed, "1h", windows={"1h": [1, 4]})

        X, y_reg, y_clf, feature_cols = build_feature_matrix(processed, "1h")
        self.assertEqual(X.ndim, 2)
        self.assertEqual(len(X), len(y_reg))
        self.assertEqual(len(X), len(y_clf))
        self.assertEqual(X.shape[1], len(feature_cols))
        self.assertFalse(np.isnan(X).any())

    def test_build_feature_matrix_triple_barrier(self):
        """Test feature matrix building for triple barrier targets."""
        processed = process_raw_data(self.df.copy(), btc_df=self.btc_df, eth_df=self.eth_df, fng_df=self.fng_df)

        from data.features.targets import create_targets_triple_barrier
        processed = create_targets_triple_barrier(processed, "1h", windows={"1h": [1, 4]})

        X, y_tb, y_tb_reg, y_filter, feature_cols = build_feature_matrix_triple_barrier(processed, "1h")
        self.assertEqual(X.ndim, 2)
        self.assertIsNotNone(y_tb)
        self.assertEqual(len(X), len(y_tb))

    def test_select_features(self):
        """Test permutation importance feature selection."""
        np.random.seed(42)
        X = np.random.randn(100, 6)
        y = 2.0 * X[:, 0] - 1.5 * X[:, 1] + np.random.randn(100) * 0.1
        feature_names = [f"feat_{i}" for i in range(6)]

        X_sel, selected = select_features(X, y, feature_names, method="permutation", n_top=2)
        self.assertLessEqual(len(selected), 2)
        self.assertTrue("feat_0" in selected or "feat_1" in selected)

    def test_audit_scale_invariance(self):
        """Test scale invariance auditing report generation."""
        processed = process_raw_data(self.df.copy(), btc_df=self.btc_df, eth_df=self.eth_df, fng_df=self.fng_df)

        report = audit_scale_invariance(processed, close_col="close")
        self.assertIsInstance(report, pd.DataFrame)
        self.assertIn("feature", report.columns)
        self.assertIn("flagged", report.columns)


if __name__ == "__main__":
    unittest.main()
