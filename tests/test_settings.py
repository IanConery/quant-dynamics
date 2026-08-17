"""Tests for configuration settings, environment loaders, and validation."""

import unittest
import config.settings as s


class TestSettings(unittest.TestCase):

    def test_env_helper_functions(self):
        """Test environment variable casting functions."""
        self.assertEqual(s._env("NON_EXISTENT_KEY_12345", "fallback"), "fallback")
        self.assertEqual(s._env_float("NON_EXISTENT_KEY_12345", 3.14), 3.14)
        self.assertEqual(s._env_int("NON_EXISTENT_KEY_12345", 42), 42)
        self.assertEqual(s._env_json("NON_EXISTENT_KEY_12345", '{"a": 1}'), {"a": 1})

    def test_core_settings_exist(self):
        """Verify that essential configuration dicts and constants are defined."""
        self.assertTrue(isinstance(s.EXCHANGE, str) and len(s.EXCHANGE) > 0)
        self.assertTrue(isinstance(s.SYMBOL, str) and len(s.SYMBOL) > 0)
        self.assertIsInstance(s.CROSS_ASSET_SYMBOLS, list)
        self.assertIsInstance(s.INDICATORS, dict)
        self.assertIsInstance(s.PREDICTION_WINDOWS, dict)
        self.assertIsInstance(s.TRIPLE_BARRIER, dict)
        self.assertIsInstance(s.META_LABEL, dict)
        self.assertIsInstance(s.BAYESIAN_UPDATING, dict)
        self.assertIsInstance(s.REGIME_PARAMS, dict)
        self.assertIsInstance(s.BACKTEST_CONFIG, dict)

    def test_validate_settings_passes_default(self):
        """Test that the current settings pass validation."""
        s.validate_settings()

    def test_validate_settings_invalid_exchange(self):
        """Test validation raises on bad exchange."""
        original = s.EXCHANGE
        try:
            s.EXCHANGE = ""
            with self.assertRaises(ValueError):
                s.validate_settings()
        finally:
            s.EXCHANGE = original

    def test_validate_settings_invalid_triple_barrier(self):
        """Test validation raises on invalid triple barrier thresholds."""
        original_upper = s.TRIPLE_BARRIER["upper_barrier"]
        try:
            s.TRIPLE_BARRIER["upper_barrier"] = -0.05
            with self.assertRaises(ValueError):
                s.validate_settings()
        finally:
            s.TRIPLE_BARRIER["upper_barrier"] = original_upper

    def test_validate_settings_invalid_meta_label(self):
        """Test validation raises on invalid meta-label threshold."""
        original_threshold = s.META_LABEL["stage1_threshold"]
        try:
            s.META_LABEL["stage1_threshold"] = 1.5
            with self.assertRaises(ValueError):
                s.validate_settings()
        finally:
            s.META_LABEL["stage1_threshold"] = original_threshold

    def test_device_auto_detection(self):
        """Test DEVICE is either 'cuda' or 'cpu'."""
        self.assertIn(s.DEVICE, ("cuda", "cpu"))


if __name__ == "__main__":
    unittest.main()
