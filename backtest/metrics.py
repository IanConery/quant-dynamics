import numpy as np
import pandas as pd


def sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0, periods_per_year: int = 365) -> float:
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    excess = returns - risk_free_rate / periods_per_year
    return float(excess.mean() / excess.std() * np.sqrt(periods_per_year))


def sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.0, periods_per_year: int = 365) -> float:
    if len(returns) < 2:
        return 0.0
    excess = returns - risk_free_rate / periods_per_year
    downside = excess[excess < 0]
    if len(downside) == 0 or np.std(downside) == 0:
        return 0.0
    downside_std = float(np.sqrt(np.mean(downside ** 2)))
    return float(excess.mean() / downside_std * np.sqrt(periods_per_year))


def max_drawdown(equity_curve: np.ndarray) -> float:
    if len(equity_curve) == 0:
        return 0.0
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (peak - equity_curve) / peak
    return float(np.max(drawdown))


def max_drawdown_duration(equity_curve: np.ndarray) -> int:
    if len(equity_curve) == 0:
        return 0
    peak = np.maximum.accumulate(equity_curve)
    in_drawdown = equity_curve < peak
    durations = []
    current = 0
    for d in in_drawdown:
        if d:
            current += 1
        else:
            if current > 0:
                durations.append(current)
            current = 0
    if current > 0:
        durations.append(current)
    return max(durations) if durations else 0


def calmar_ratio(returns: np.ndarray, equity_curve: np.ndarray, periods_per_year: int = 365) -> float:
    md = max_drawdown(equity_curve)
    if md == 0:
        return 0.0
    annual_return = float(returns.sum() * periods_per_year / len(returns))
    return annual_return / md


def profit_factor(gross_profit: float, gross_loss: float) -> float:
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0
    return gross_profit / abs(gross_loss)


def compute_backtest_metrics(
    trades: pd.DataFrame,
    equity_curve: np.ndarray,
    holding_period_days: float = 1.0,
) -> dict:
    if len(trades) == 0:
        return {
            "total_trades": 0,
            "trades_per_day": 0.0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "total_return": 0.0,
            "annualized_return": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
            "max_drawdown_duration": 0,
            "calmar_ratio": 0.0,
            "profit_factor": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "avg_trade_duration": 0.0,
            "total_commission": 0.0,
            "net_return": 0.0,
            "expectancy": 0.0,
            "best_trade": 0.0,
            "worst_trade": 0.0,
            "stop_loss_hits": 0,
            "take_profit_hits": 0,
            "trailing_stop_hits": 0,
            "circuit_breaker_triggered": False,
        }

    total_trades = len(trades)
    wins = trades[trades["pnl"] > 0]
    losses = trades[trades["pnl"] <= 0]
    winning_trades = len(wins)
    losing_trades = len(losses)
    win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

    gross_profit = float(wins["pnl"].sum()) if len(wins) > 0 else 0.0
    gross_loss = float(losses["pnl"].sum()) if len(losses) > 0 else 0.0

    returns = equity_curve[1:] / equity_curve[:-1] - 1 if len(equity_curve) > 1 else np.array([0.0])
    total_return = float((equity_curve[-1] / equity_curve[0]) - 1) if len(equity_curve) > 0 else 0.0
    n_periods = len(returns)
    annualized_return = total_return * (365 / holding_period_days) if holding_period_days > 0 else 0.0

    trade_returns = trades["return_pct"].values if "return_pct" in trades.columns else np.array([0.0])
    sr = sharpe_ratio(trade_returns) if len(trade_returns) > 0 else 0.0
    so = sortino_ratio(trade_returns) if len(trade_returns) > 0 else 0.0
    md = max_drawdown(equity_curve)
    mdd_dur = max_drawdown_duration(equity_curve)
    cr = calmar_ratio(trade_returns, equity_curve) if len(trade_returns) > 0 else 0.0
    pf = profit_factor(gross_profit, gross_loss)

    avg_win = float(wins["pnl"].mean()) if len(wins) > 0 else 0.0
    avg_loss = float(losses["pnl"].mean()) if len(losses) > 0 else 0.0
    avg_duration = float(trades["duration"].mean()) if "duration" in trades.columns and len(trades) > 0 else 0.0
    total_commission = float(trades["commission"].sum()) if "commission" in trades.columns else 0.0
    net_return = total_return - total_commission
    expectancy = float(trades["pnl"].mean()) if total_trades > 0 else 0.0
    best_trade = float(trades["pnl"].max()) if total_trades > 0 else 0.0
    worst_trade = float(trades["pnl"].min()) if total_trades > 0 else 0.0

    exit_reasons = trades.get("exit_reason", pd.Series(dtype=str)).value_counts()
    sl_hits = int(exit_reasons.get("stop_loss", 0))
    tp_hits = int(exit_reasons.get("take_profit", 0))
    ts_hits = int(exit_reasons.get("trailing_stop", 0))
    trades_per_day = total_trades / holding_period_days if holding_period_days > 0 else 0.0

    return {
        "total_trades": total_trades,
        "trades_per_day": round(trades_per_day, 4),
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "win_rate": round(win_rate, 4),
        "total_return": round(total_return, 6),
        "annualized_return": round(annualized_return, 6),
        "sharpe_ratio": round(sr, 4),
        "sortino_ratio": round(so, 4),
        "max_drawdown": round(md, 6),
        "max_drawdown_duration": mdd_dur,
        "calmar_ratio": round(cr, 4),
        "profit_factor": round(pf, 4),
        "avg_win": round(avg_win, 6),
        "avg_loss": round(avg_loss, 6),
        "avg_trade_duration": round(avg_duration, 2),
        "total_commission": round(total_commission, 4),
        "net_return": round(net_return, 6),
        "expectancy": round(expectancy, 6),
        "best_trade": round(best_trade, 6),
        "worst_trade": round(worst_trade, 6),
        "stop_loss_hits": sl_hits,
        "take_profit_hits": tp_hits,
        "trailing_stop_hits": ts_hits,
        "circuit_breaker_triggered": False,
    }


def compute_buy_and_hold_metrics(
    prices: np.ndarray,
    initial_capital: float = 10000.0,
    commission_rate: float = 0.001,
    holding_period_days: float = 1.0,
    periods_per_year: int = 365,
) -> dict:
    """Compute metrics for a simple buy-and-hold strategy.

    Buys at the first price and holds until the last price.
    Returns same metric structure as compute_backtest_metrics for easy comparison.
    """
    if len(prices) < 2:
        return _empty_bh_metrics()

    entry_price = prices[0]
    exit_price = prices[-1]
    total_return = float((exit_price / entry_price) - 1)
    price_returns = np.diff(prices) / prices[:-1]

    equity = initial_capital * (prices / entry_price)
    commission = initial_capital * commission_rate * 2
    equity[-1] -= commission
    equity_curve = np.concatenate([[initial_capital], equity])

    eq_returns = equity_curve[1:] / equity_curve[:-1] - 1
    sr = sharpe_ratio(eq_returns, periods_per_year=periods_per_year)
    so = sortino_ratio(eq_returns, periods_per_year=periods_per_year)
    md = max_drawdown(equity_curve)
    mdd_dur = max_drawdown_duration(equity_curve)
    annual_return = total_return * (periods_per_year / (len(prices) * holding_period_days)) if len(prices) > 0 else 0.0
    cr = calmar_ratio(eq_returns, equity_curve, periods_per_year) if md > 0 else 0.0

    return {
        "total_trades": 1,
        "winning_trades": 1 if total_return > 0 else 0,
        "losing_trades": 0 if total_return > 0 else 1,
        "win_rate": 1.0 if total_return > 0 else 0.0,
        "total_return": round(total_return, 6),
        "annualized_return": round(annual_return, 6),
        "sharpe_ratio": round(sr, 4),
        "sortino_ratio": round(so, 4),
        "max_drawdown": round(md, 6),
        "max_drawdown_duration": mdd_dur,
        "calmar_ratio": round(cr, 4),
        "profit_factor": 0.0,
        "avg_win": round(total_return * initial_capital, 6) if total_return > 0 else 0.0,
        "avg_loss": round(abs(total_return) * initial_capital, 6) if total_return < 0 else 0.0,
        "avg_trade_duration": round(float(len(prices) - 1), 2),
        "total_commission": round(commission, 4),
        "net_return": round(total_return - commission / initial_capital, 6),
        "expectancy": round(total_return * initial_capital - commission, 6),
        "best_trade": round((exit_price - entry_price) / entry_price * initial_capital, 6),
        "worst_trade": round((exit_price - entry_price) / entry_price * initial_capital, 6),
        "stop_loss_hits": 0,
        "take_profit_hits": 0,
        "trailing_stop_hits": 0,
        "circuit_breaker_triggered": False,
    }


def walk_forward_significance(all_metrics: list) -> dict:
    """Compute statistical significance of walk-forward results.

    Args:
        all_metrics: List of metric dicts from each fold.

    Returns:
        Dict with t_stat, p_value, sharpe_ci_low, sharpe_ci_high, n_folds.
    """
    from scipy import stats
    import numpy as np

    n_folds = len(all_metrics)
    if n_folds < 2:
        return {
            "n_folds": n_folds,
            "t_stat": 0.0,
            "p_value": 1.0,
            "sharpe_ci_low": 0.0,
            "sharpe_ci_high": 0.0,
            "significance_note": "Need 2+ folds for significance test",
        }

    differences = []
    sharpes = []
    for m in all_metrics:
        agent_ret = m.get("total_return", 0.0)
        bh_ret = m.get("bh_total_return", 0.0)
        differences.append(agent_ret - bh_ret)
        sharpes.append(m.get("sharpe_ratio", 0.0))

    differences = np.array(differences)
    sharpes = np.array(sharpes)

    t_stat, p_value = stats.ttest_1samp(differences, 0.0)

    n_boot = 1000
    boot_sharpe = []
    for _ in range(n_boot):
        sample = np.random.choice(sharpes, size=n_folds, replace=True)
        boot_sharpe.append(sample.mean())
    ci_low, ci_high = np.percentile(boot_sharpe, [2.5, 97.5])

    significant = p_value < 0.05 and np.mean(differences) > 0
    note = "Statistically significant outperformance vs B&H" if significant else "Cannot conclude agent beats B&H"

    return {
        "n_folds": n_folds,
        "t_stat": round(float(t_stat), 4),
        "p_value": round(float(p_value), 4),
        "sharpe_ci_low": round(float(ci_low), 4),
        "sharpe_ci_high": round(float(ci_high), 4),
        "mean_sharpe": round(float(sharpes.mean()), 4),
        "significant": significant,
        "significance_note": note,
    }


def _empty_bh_metrics() -> dict:
    return {
        "total_trades": 0,
        "trades_per_day": 0.0,
        "winning_trades": 0,
        "losing_trades": 0,
        "win_rate": 0.0,
        "total_return": 0.0,
        "annualized_return": 0.0,
        "sharpe_ratio": 0.0,
        "sortino_ratio": 0.0,
        "max_drawdown": 0.0,
        "max_drawdown_duration": 0,
        "calmar_ratio": 0.0,
        "profit_factor": 0.0,
        "avg_win": 0.0,
        "avg_loss": 0.0,
        "avg_trade_duration": 0.0,
        "total_commission": 0.0,
        "net_return": 0.0,
        "expectancy": 0.0,
        "best_trade": 0.0,
        "worst_trade": 0.0,
        "stop_loss_hits": 0,
        "take_profit_hits": 0,
        "trailing_stop_hits": 0,
        "circuit_breaker_triggered": False,
    }
