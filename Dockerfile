# Multi-stage / Production Dockerfile for Quant Dynamics
FROM python:3.11-slim

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies (compiler, git, build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency specifications
COPY requirements.txt requirements-lock.txt pyproject.toml ./

# Install python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Ensure example environment file exists if .env is omitted
RUN if [ ! -f .env ]; then cp .env.example .env; fi

# Create artifacts directory structure
RUN mkdir -p artifacts/raw_data artifacts/processed_data artifacts/models artifacts/backtest_results

# Default entrypoint runs CLI help
ENTRYPOINT ["python", "main.py"]
CMD ["--help"]
