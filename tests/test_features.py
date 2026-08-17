"""Tests for technical indicator, cross-asset, time, and target feature engineering."""

import unittest
import numpy as np
import pandas as pd

from data.features.cross_asset import (
    add_btc_lead_lag_features,
    add_cross_asset_features,
    add_momentum_features,
    add_vol_adj_momentum_features,
)
from data.features.external import (
    add_regime_features,
    add_time_features,
)
from data.features.targets import (
    _interval_to_hours,
    adjust_targets_for_cost,
    create_pbr_targets,
    create_targets,
    create_targets_triple_barrier,
)
from data.features.technical import (
    add_adx,
    add_atr,
    add_bollinger,
    add_ema,
    add_lag_features,
    add_macd,
    add_obv,
    add_price_patterns,
    add_rsi,
    add_sma,
    add_stochastic,
    add_volume_sma,
    add_vwap,
)


class TestFeatures(unittest.TestCase):

    def setUp(self):
        """Create a reproducible synthetic OHLCV DataFrame."""
        np.random.seed(42)
        n = 250
        dates = pd.date_range("2024-01-01", periods=n, freq="1h")
        price = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
        high = price + np.random.uniform(0.1, 1.0, n)
        low = price - np.random.uniform(0.1, 1.0, n)
        open_p = price + np.random.uniform(-0.3, 0.3, n)
        close = price
        volume = np.random.uniform(1000, 50000, n)

        self.df = pd.DataFrame({
            "timestamp": dates,
            "open": open_p,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        })

    def test_technical_indicators(self):
        """Test all technical indicator additions."""
        df = self.df.copy()
        df = add_sma(df, windows=[7, 14])
        self.assertIn("sma_7", df.columns)
        self.assertIn("sma_14", df.columns)

        df = add_ema(df, windows=[7, 14])
        self.assertIn("ema_7", df.columns)
        self.assertIn("ema_14", df.columns)

        df = add_rsi(df, periods=[7, 14])
        self.assertIn("rsi_7", df.columns)
        self.assertIn("rsi_14", df.columns)
        self.assertTrue(df["rsi_14"].dropna().between(0, 100).all())

        df = add_macd(df)
        self.assertIn("macd", df.columns)
        self.assertIn("macd_signal", df.columns)
        self.assertIn("macd_histogram", df.columns)

        df = add_bollinger(df)
        self.assertIn("bb_upper", df.columns)
        self.assertIn("bb_lower", df.columns)
        self.assertIn("bb_width", df.columns)

        df = add_atr(df, periods=[14])
        self.assertIn("atr_14", df.columns)

        df = add_stochastic(df)
        self.assertIn("stoch_k", df.columns)
        self.assertIn("stoch_d", df.columns)

        df = add_adx(df)
        self.assertIn("adx", df.columns)
        self.assertIn("plus_di", df.columns)
        self.assertIn("minus_di", df.columns)

        df = add_obv(df)
        self.assertIn("obv", df.columns)

        df = add_vwap(df)
        self.assertIn("vwap", df.columns)

        df = add_volume_sma(df, windows=[7, 14])
        self.assertIn("volume_sma_7", df.columns)
        self.assertIn("volume_ratio", df.columns)

        df = add_lag_features(df, lags=[1, 2, 5])
        self.assertIn("close_lag_1", df.columns)
        self.assertIn("return_1", df.columns)

        df = add_price_patterns(df)
        self.assertIn("body_size", df.columns)
        self.assertIn("is_bullish", df.columns)

    def test_time_features(self):
        """Test cyclical time feature creation."""
        df = add_time_features(self.df.copy())
        self.assertIn("hour_sin", df.columns)
        self.assertIn("hour_cos", df.columns)
        self.assertIn("dow_sin", df.columns)
        self.assertIn("dow_cos", df.columns)
        self.assertIn("month_sin", df.columns)
        self.assertIn("month_cos", df.columns)
        self.assertIn("is_weekend", df.columns)

    def test_regime_features(self):
        """Test regime feature calculation."""
        df = add_regime_features(self.df.copy())
        self.assertIn("rolling_sharpe", df.columns)
        self.assertIn("vol_ratio", df.columns)
        self.assertIn("is_bull", df.columns)

    def test_cross_asset_and_momentum_features(self):
        """Test cross-asset features and momentum calculations."""
        btc_df = self.df.copy()
        btc_df["close"] = btc_df["close"] * 50.0
        eth_df = self.df.copy()
        eth_df["close"] = eth_df["close"] * 30.0

        df = add_cross_asset_features(self.df.copy(), btc_df, eth_df)
        self.assertIn("xrp_btc_ratio", df.columns)
        self.assertIn("btc_dominance", df.columns)

        df = add_momentum_features(df)
        self.assertIn("momentum_4", df.columns)
        self.assertIn("momentum_accel_4", df.columns)

        df = add_vol_adj_momentum_features(df)
        self.assertIn("vol_adj_momentum_4", df.columns)
        self.assertIn("vol_regime", df.columns)

    def test_interval_to_hours(self):
        """Test interval conversion function."""
        self.assertEqual(_interval_to_hours("15m"), 0.25)
        self.assertEqual(_interval_to_hours("1h"), 1.0)
        self.assertEqual(_interval_to_hours("4h"), 4.0)
        self.assertEqual(_interval_to_hours("1d"), 24.0)

    def test_targets_creation(self):
        """Test standard and triple barrier target generation."""
        windows = {"1h": [1, 4, 24]}
        df = create_targets(self.df.copy(), "1h", windows=windows)
        self.assertIn("reg_target_1h", df.columns)
        self.assertIn("clf_target_1h", df.columns)
        self.assertIn("ternary_target_1h", df.columns)

        df_tb = create_targets_triple_barrier(self.df.copy(), "1h", windows=windows)
        self.assertIn("tb_target_1h", df_tb.columns)
        self.assertIn("tb_reg_target_1h", df_tb.columns)
        self.assertTrue(set(df_tb["tb_target_1h"].unique()).issubset({0, 1, 2}))

        df_pbr = create_pbr_targets(self.df.copy(), "1h", windows=windows)
        self.assertIn("pbr_target_1h", df_pbr.columns)

    def test_adjust_targets_for_cost(self):
        """Test cost adjustment zeroing out small returns."""
        y = np.array([0.05, 0.001, -0.001, -0.04, 0.0005])
        adjusted = adjust_targets_for_cost(y, round_trip_cost=0.0025)
        self.assertEqual(adjusted[0], 0.05)
        self.assertEqual(adjusted[1], 0.0)
        self.assertEqual(adjusted[2], 0.0)
        self.assertEqual(adjusted[3], -0.04)
        self.assertEqual(adjusted[4], 0.0)


if __name__ == "__main__":
    unittest.main()
