#!/bin/bash
# Retrain: Optuna tuning only (3 windows)
cd "$(dirname "$0")"
PYTHON="${PYTHON:-python}"
LOG="artifacts/optuna_retrain_$(date '+%Y%m%d_%H%M%S').log"
mkdir -p artifacts

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

run_cmd() {
    log ">>> $*"
    if $PYTHON "$@" >> "$LOG" 2>&1; then
        log "  ✓ Done"
    else
        log "  ✗ Failed"
    fi
}

log "=== Optuna tuning (LSTM + iTransformer + TFT) ==="
for iw in "1d/168h" "1d/72h" "1h/24h"; do
    interval="${iw%%/*}"
    window="${iw##*/}"
    run_cmd main.py train --interval "$interval" --window "$window" --model deep --tune
done
log "=== DONE (log: $LOG) ==="
