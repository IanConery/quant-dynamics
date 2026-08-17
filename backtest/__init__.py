"""Backtesting simulation and performance evaluation package."""

from backtest.engine import (
    compare_strategies,
    confidence_based_strategy,
    run_backtest,
    signal_based_strategy,
    walk_forward_backtest,
)
from backtest.metrics import (
    calmar_ratio,
    compute_backtest_metrics,
    compute_buy_and_hold_metrics,
    max_drawdown,
    max_drawdown_duration,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
    walk_forward_significance,
)

__all__ = [
    # Engine
    "run_backtest",
    "walk_forward_backtest",
    "compare_strategies",
    "signal_based_strategy",
    "confidence_based_strategy",
    # Metrics
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "max_drawdown_duration",
    "calmar_ratio",
    "profit_factor",
    "compute_backtest_metrics",
    "compute_buy_and_hold_metrics",
    "walk_forward_significance",
]
