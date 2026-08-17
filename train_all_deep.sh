#!/bin/bash
# =============================================================================
# Quant Dynamics — Full Deep Model Training Script
# Trains all models (classical + deep + ensemble) across all intervals/windows.
# Then runs Optuna tuning and backtest comparison.
# Run overnight. Estimated time: 6-10 hours on GPU (with 2019+ data).
# Robust: continues on per-step failures, logs everything.
# GPU-safe: deep models run sequentially; classical models can overlap.
# =============================================================================

cd "$(dirname "$0")"
PYTHON="${PYTHON:-python}"
LOGFILE="artifacts/training_log_$(date '+%Y%m%d_%H%M%S').log"
mkdir -p artifacts

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOGFILE"
}

run_cmd() {
    log ">>> $*"
    if $PYTHON "$@" >> "$LOGFILE" 2>&1; then
        log "    ✓ Success"
    else
        log "    ✗ Failed — continuing anyway"
    fi
}

run_cmd_bg() {
    log ">>> (bg) $*"
    $PYTHON "$@" >> "$LOGFILE" 2>&1 &
    echo $!
}

wait_all() {
    local pids=("$@")
    local failed=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid" 2>/dev/null; then
            failed=$((failed + 1))
        fi
    done
    if [ "$failed" -gt 0 ]; then
        log "    ✗ $failed background job(s) failed"
    else
        log "    ✓ All background jobs done"
    fi
}

# ---------- helper: get windows for interval -------------------------------
get_windows() {
    case "$1" in
        15m) echo "1h 4h 12h 24h" ;;
        1h)  echo "1h 4h 12h 24h 72h" ;;
        4h)  echo "4h 12h 24h 72h 168h" ;;
        1d)  echo "24h 72h 168h 336h 720h" ;;
    esac
}

# ---------- Step 0: Classical models (parallelize across windows) ----------
# Classical models are CPU-only, so we can run multiple in parallel.
log "=== Step 0: Retraining classical models (all intervals, parallelized) ==="
for interval in 15m 1h 4h 1d; do
    bg_pids=()
    for w in $(get_windows "$interval"); do
        pid=$(run_cmd_bg main.py train --interval "$interval" --window "$w" --model classical --evaluate)
        bg_pids+=("$pid")
    done
    log "  Waiting for $interval classical models ($((${#bg_pids[@]})) jobs)..."
    wait_all "${bg_pids[@]}"
done

# ---------- Step 1: Deep models (sequential — GPU-bound) -------------------
log "=== Step 1: Training deep models (LSTM + iTransformer + TFT, sequential) ==="
for interval in 15m 1h 4h 1d; do
    for w in $(get_windows "$interval"); do
        run_cmd main.py train --interval "$interval" --window "$w" --model deep --evaluate
    done
done

# ---------- Step 2: Ensembles (lightweight, parallelize) --------------------
# Ensembles don't train base models (they load from disk), just fit meta-learner.
log "=== Step 2: Training ensembles (all windows, parallelized) ==="
for interval in 15m 1h 4h 1d; do
    bg_pids=()
    for w in $(get_windows "$interval"); do
        pid=$(run_cmd_bg main.py train --interval "$interval" --window "$w" --model ensemble --evaluate)
        bg_pids+=("$pid")
    done
    log "  Waiting for $interval ensembles ($((${#bg_pids[@]})) jobs)..."
    wait_all "${bg_pids[@]}"
done

# ---------- Step 3: Optuna tuning (sequential — GPU for deep models) --------
log "=== Step 3: Optuna tuning (best 4 windows, sequential) ==="
for interval_w in "1d/168h" "1d/72h" "4h/72h" "1h/24h"; do
    interval="${interval_w%%/*}"
    window="${interval_w##*/}"
    log "  Tuning: ${interval} / ${window} (LSTM + iTransformer + TFT)"
    run_cmd main.py train --interval "$interval" --window "$window" --model deep --tune
done

# ---------- Step 4: Backtest comparison (CPU-only, parallelize) ------------
log "=== Step 4: Backtest comparison (best 4 windows, parallelized) ==="
bg_pids=()
for interval_w in "1d/168h" "1d/72h" "4h/72h" "1h/24h"; do
    interval="${interval_w%%/*}"
    window="${interval_w##*/}"
    pid=$(run_cmd_bg main.py backtest --interval "$interval" --window "$window" --model ensemble --strategy all)
    bg_pids+=("$pid")
done
wait_all "${bg_pids[@]}"

log "=== ALL DONE ==="
log "Full log: $LOGFILE"
