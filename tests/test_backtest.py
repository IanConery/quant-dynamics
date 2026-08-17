"""Tests for backtesting metrics and execution engine."""

import unittest
import numpy as np
import pandas as pd

from backtest.metrics import (
    calmar_ratio,
    compute_backtest_metrics,
    compute_buy_and_hold_metrics,
    max_drawdown,
    max_drawdown_duration,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
)


class TestBacktest(unittest.TestCase):

    def test_financial_metrics(self):
        """Test standard financial ratio calculations."""
        returns = np.array([0.01, 0.02, -0.005, 0.015, -0.01, 0.025])
        equity = 10000.0 * np.cumprod(1.0 + returns)

        sr = sharpe_ratio(returns)
        self.assertIsInstance(sr, float)
        self.assertFalse(np.isnan(sr))

        sort_r = sortino_ratio(returns)
        self.assertIsInstance(sort_r, float)
        self.assertFalse(np.isnan(sort_r))

        dd = max_drawdown(equity)
        self.assertTrue(0.0 <= dd <= 1.0)

        dd_dur = max_drawdown_duration(equity)
        self.assertIsInstance(dd_dur, int)
        self.assertGreaterEqual(dd_dur, 0)

        calm = calmar_ratio(returns, equity)
        self.assertIsInstance(calm, float)

        pf = profit_factor(gross_profit=500.0, gross_loss=200.0)
        self.assertEqual(pf, 2.5)

    def test_compute_backtest_metrics(self):
        """Test full backtest metric report generation."""
        np.random.seed(42)
        n_bars = 100
        equity = 10000.0 + np.cumsum(np.random.randn(n_bars) * 50)

        trades_df = pd.DataFrame([
            {"pnl": 120.0, "return_pct": 0.012, "duration": 5, "commission": 1.0, "exit_reason": "take_profit"},
            {"pnl": -60.0, "return_pct": -0.006, "duration": 5, "commission": 1.0, "exit_reason": "stop_loss"},
            {"pnl": 200.0, "return_pct": 0.020, "duration": 10, "commission": 1.0, "exit_reason": "signal_exit"},
        ])

        metrics = compute_backtest_metrics(trades_df, equity, holding_period_days=30.0)
        self.assertEqual(metrics["total_trades"], 3)
        self.assertAlmostEqual(metrics["win_rate"], 2 / 3, delta=0.01)
        self.assertIn("sharpe_ratio", metrics)
        self.assertIn("max_drawdown", metrics)

    def test_buy_and_hold_metrics(self):
        """Test benchmark buy and hold metric calculations."""
        prices = np.array([1.0, 1.05, 1.02, 1.10, 1.08, 1.15])
        metrics = compute_buy_and_hold_metrics(prices, initial_capital=10000.0)
        self.assertAlmostEqual(metrics["total_return"], 0.15, delta=0.01)
        self.assertIn("sharpe_ratio", metrics)
        self.assertIn("max_drawdown", metrics)


if __name__ == "__main__":
    unittest.main()
