.PHONY: help install install-dev test lint fetch process train predict backtest stream clean

PYTHON ?= python

help:  ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install:  ## Install core dependencies
	$(PYTHON) -m pip install -r requirements.txt
	@if [ ! -f .env ]; then cp .env.example .env; echo "Created .env from .env.example"; fi

install-dev: install  ## Install development tools (pytest, ruff)
	$(PYTHON) -m pip install -e ".[dev]"

test:  ## Run full test suite
	$(PYTHON) -m unittest discover -s tests -p "test_*.py" -v

lint:  ## Run ruff linter
	ruff check .

fetch:  ## Fetch multi-source OHLCV and external data
	$(PYTHON) main.py data fetch --refresh

process:  ## Process all intervals and engineer features
	$(PYTHON) main.py data process

train:  ## Train all models on default 1h interval
	$(PYTHON) main.py train --interval 1h --window 24h --model all --evaluate

predict:  ## Run ensemble inference
	$(PYTHON) main.py predict --interval 1h --window 24h --model ensemble --latest 10

backtest:  ## Run backtesting simulation
	$(PYTHON) main.py backtest --interval 1h --window 24h --model ensemble --strategy all

stream:  ## Start WebSocket live data streaming and prediction
	$(PYTHON) main.py stream start --interval 1h --window 24h --model ensemble

clean:  ## Clean temporary caches and compilation files
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
