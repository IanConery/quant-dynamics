# Quant Dynamics

### Systematic Machine Learning, Market Regime Detection & Time-Series Backtesting Framework
*An End-to-End Quantitative Research Pipeline for Digital Asset Markets*

Quant Dynamics is a modular quantitative forecasting and strategy research engine for digital asset markets (configured for XRP/USDT with multi-asset cross-correlations). It features multi-horizon feature engineering, market regime classification, stacked ensembles (classical ML + deep learning), Bayesian belief updating, and walk-forward backtesting simulation.

## Architecture

```
quant-dynamics/
├── config/settings.py          # Configuration, model hyperparams, derivatives, macro, Binance Vision config
├── data/
│   ├── fetcher.py              # Multi-source data: Kraken (ccxt) + yfinance + alternative.me FNG + Binance Vision
│   ├── derivatives_fetcher.py  # Funding rates, OI, liquidations
│   ├── external_fetcher.py     # On-chain, macro (VIX/DXY/US10Y), order book
│   ├── features/               # Modular feature engineering: technical, cross-asset, external, targets, selection
│   ├── processor.py            # Feature engineering orchestration & feature matrix builders
│   └── loader.py               # Chronological split, sequences, temporal CV, dataset caching
├── models/
│   ├── classical.py            # ClassicalModel wrapper (XGBoost, LightGBM, Random Forest)
│   ├── deep.py                 # LSTMModel + iTransformerModel (PyTorch)
│   ├── tft.py                  # Temporal Fusion Transformer (multi-task, interpretable)
│   ├── ensemble.py             # StackingEnsemble (Ridge/LogisticRegression meta-learner)
│   ├── calibration.py          # Platt scaling / isotonic probability calibration
│   ├── regime_classifier.py    # Multi-class regime classifier (Bear/Sideways/Bull)
│   ├── garch.py                # GARCH(1,1), EGARCH volatility modeling
│   ├── state_space.py          # Kalman Filter, Gaussian Process regression
│   ├── regime_trainer.py       # Regime-aware training + fallback
│   └── trainer.py              # Training loops, Optuna tuning, temporal CV, calibration
├── backtest/
│   ├── engine.py               # Strategy execution, risk management, walk-forward, B&H benchmark
│   └── metrics.py              # Sharpe, Sortino, Calmar, max drawdown, expectancy, profit factor
├── streaming/
│   ├── engine.py               # Real-time WebSocket→normalize→predict→alert→drift pipeline
│   ├── predictor.py            # Incremental feature computation, streaming inference
│   ├── alert.py                # Confidence threshold, regime change, cooldown alerts
│   ├── drift_monitor.py        # Population Stability Index (PSI) drift monitoring
│   └── handler.py              # Message normalizer, candle aggregator
├── utils/logger.py             # Centralized structured logging
├── tests/                      # Unit and integration test suite
├── main.py                     # Unified CLI entry point
├── Makefile                    # Workflow automation (install, test, train, backtest)
├── Dockerfile                  # Containerized deployment specification
├── pyproject.toml              # Package configuration and tool settings
└── requirements.txt
```

## Quick Start

```bash
# Clone and setup environment
git clone https://github.com/your-username/quant-dynamics.git
cd quant-dynamics

conda create -n quant-dynamics python=3.11 -y
conda activate quant-dynamics
pip install -r requirements.txt
cp .env.example .env
```

## CLI Commands

### Data Pipeline

```bash
python main.py data fetch                    # Fetch OHLCV + FNG (XRP + BTC + ETH)
python main.py data fetch --refresh          # Incremental: only new candles
python main.py data fetch --force            # Full re-fetch from scratch
python main.py data fetch --binance          # Download Binance Vision historical klines (2019+) + merge
python main.py data fetch --derivatives      # Fetch derivatives data (funding, OI, liqs)
python main.py data fetch --external         # Fetch external data (on-chain, macro, order book)
python main.py data process                  # Process all intervals with features + targets
python main.py data process --intervals 1h   # Single interval
```

### Training

```bash
python main.py train --interval 1h --window 24h --model all --evaluate
python main.py train --interval 1h --window 24h --model classical
python main.py train --interval 1h --window 24h --model deep
python main.py train --interval 1h --window 24h --model ensemble
python main.py train --interval 1h --window 24h --tune    # Optuna hyperparameter tuning
```

### Predictions

```bash
python main.py predict --interval 1h --window 24h --model ensemble --latest 10
python main.py predict --interval 1d --window 72h --model lightgbm --latest 20
```

### Backtesting (includes buy-and-hold comparison)

```bash
python main.py backtest --interval 1h --window 24h --model ensemble --strategy signal_based
python main.py backtest --interval 1h --window 24h --model ensemble --strategy all      # Compare
python main.py backtest --interval 1d --window 168h --model ensemble --walk-forward     # Walk-forward
python main.py backtest --interval 1d --window 72h --model ensemble --zero-commission   # Zero-commission diagnostic
python main.py backtest --interval 1d --window 72h --model ensemble --min-confidence 0.70 --min-holding-bars 3
```

### Regime + Volatility

```bash
python main.py regime-train --interval 1d --window 168h --model lightgbm
python main.py regime-predict --interval 1d --window 168h --latest 10
python main.py volatility --interval 1d --window 168h
```

### Real-Time Streaming

```bash
python main.py stream start --interval 1h --window 24h --model ensemble
python main.py stream predict --interval 1d --window 168h --model ensemble
python main.py stream drift --interval 1d --window 168h --psi-threshold 0.1
python main.py stream config
```

### Bot Tracker

Detect and track algorithmic trading accounts. See `bot_tracker/SUMMARY.md` for full docs.

```bash
# Full pipeline (fetch, detect, profile, cluster, train, report)
bash bot_tracker/train_bot_tracker.sh

# Individual steps
python main.py bot fetch --days 365       # Historical trade data
python main.py bot fetch-onchain          # XRPL wallet transactions
python main.py bot scan                   # Detect bot patterns
python main.py bot profile                # Score wallets for bot-like behavior
python main.py bot cluster                # Group patterns into bot types
python main.py bot train --horizon 15     # Train prediction model
python main.py bot predict --horizon 15   # Predict bot activity
python main.py bot report                 # Summary report
```

### Evaluation + Advanced

```bash
python main.py evaluate --interval 1h --window 24h
python main.py cv --interval 1d --window 168h --model lightgbm --folds 5 --gap 3       # Temporal CV
python main.py calibrate --interval 1d --window 72h --model lightgbm --method sigmoid   # Platt scaling
```

## Data Sources

| Source | Data | Interval |
|--------|------|----------|
| Kraken (ccxt) | XRP/USDT, BTC/USDT, ETH/USDT OHLCV | 15m, 1h, 4h, 1d |
| Binance Vision | XRP/BTC/ETH historical klines (2019+) | 15m, 1h, 4h, 1d |
| yfinance | Deeper historical OHLCV, VIX, DXY, US10Y | 15m, 1h, 4h, 1d |
| alternative.me | Fear & Greed Index | daily |
| OKX | Funding rates, open interest, liquidations | 8h, daily |
| Blockchain.com | Active addresses, hashrate | point-in-time |

Fallback chain: Kraken → Coinbase → Bybit → OKX

## Data Scale

Historical data from 2019+ across multiple intervals (15m, 1h, 4h, 1d), sourced from Binance Vision and Kraken.

## Features

The system engineers 135+ candidate features across multiple categories:

- **Technical indicators**: Trend, momentum, volatility, and volume indicators
- **Price action**: Lag features, returns, candlestick patterns
- **Time**: Cyclical time encodings
- **Cross-asset**: Multi-asset correlations and relative metrics
- **Sentiment**: Fear & Greed Index derivatives
- **Regime**: Market regime classification features
- **Derivatives**: Funding rates, open interest, liquidation data
- **Macro**: Traditional market indicators
- **Microstructure**: Taker flow and order book features

Automatic feature selection via permutation importance reduces to the most predictive subset per window.

## Models

### Classical
- **XGBoost**, **LightGBM**, **Random Forest**
- Early stopping (50 rounds) for XGB/LGBM
- Sample weighting for class imbalance
- Saved as `.pkl` via joblib

### Deep
- **LSTM**: 3-layer, hidden=128, dropout=0.3
- **iTransformer**: Inverted transformer (features as tokens), d_model=128, 4 layers, 8 heads
- **TFT**: Temporal Fusion Transformer, variable selection, static encoder, GRU decoder, additive attention, multi-task (regression + binary + ternary classification), Kendall uncertainty-weighted loss, MC Dropout for uncertainty estimation
- AdamW optimizer, ReduceLROnPlateau, gradient clipping, early stopping (patience=20)
- Saved as `.pth` via torch.save

### Ensemble
- **Stacking**: 5 base models (XGB + LGBM + RF + LSTM + iTransformer)
- **Meta-learner**: Ridge (regression), LogisticRegression (classification)
- Saved as `.pkl` via joblib

## Prediction Windows

| Interval | Windows |
|----------|---------|
| 15m | 1h, 4h, 12h, 24h |
| 1h | 1h, 4h, 12h, 24h, 72h |
| 4h | 4h, 12h, 24h, 72h, 168h |
| 1d | 24h, 72h, 168h, 336h, 720h |

## Backtesting

### Strategies
- **Signal-based**: Enter when P(UP) ≥ 0.55, exit when P(UP) ≤ 0.45
- **Confidence-based**: Same thresholds but only trade when model confidence ≥ 0.70

### Risk Management
Configurable position sizing, stop-loss, take-profit, trailing stops, and drawdown circuit breaker. Parameters are loaded from environment configuration.

### Metrics
Sharpe, Sortino, Calmar, Max Drawdown, Profit Factor, Expectancy, Win Rate, Average Win/Loss, Stop-loss hits, Take-profit hits, Trailing stop hits

### Walk-Forward
Expanding window, 1-month step, min 1260 train / 720 test samples, up to 20 folds.

## Status

This is an active research project. Model performance varies by prediction window and market regime. See the backtesting module for evaluation tools.

## Configuration

All settings are in `config/settings.py`. Sensitive parameters (trading thresholds, ensemble weights, risk management) are loaded from a `.env` file. Copy `.env.example` to `.env` and adjust values for your setup.

## Device

Auto-detects CUDA (`torch.cuda.is_available()`), falls back to CPU.

## Bot Tracker

Standalone module for detecting and tracking algorithmic trading patterns in cryptocurrency markets. See `bot_tracker/` for details.

## Overnight Training Script

```bash
# Run all training: classical + deep (LSTM/iTransformer/TFT) + ensembles + Optuna tuning + backtest
# Covers all intervals: 15m, 1h, 4h, 1d
# Estimated time: 8-14 hours on GPU
bash train_all_deep.sh
```

