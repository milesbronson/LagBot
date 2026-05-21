#!/usr/bin/env bash
# All-night 8-bin training: anchor (2M vs random) → 6-gen self-play chain.
# Stdout/stderr go to logs/overnight_8bin_<timestamp>.log so progress is
# inspectable via `tail -f`. Each phase aborts the rest on failure.

set -euo pipefail
cd "$(dirname "$0")/.."

TS=$(date +%Y%m%d_%H%M%S)
LOG="logs/overnight_8bin_${TS}.log"
mkdir -p logs

echo "[$(date)] starting anchor (2M vs RandomAgent)" | tee -a "$LOG"
.venv/bin/python train.py \
  --config configs/heads_up_anchor_8bin_2M.yaml \
  --name heads_up_anchor_8bin_v1 \
  >> "$LOG" 2>&1

echo "[$(date)] anchor done; starting chain (6 gens × 2M)" | tee -a "$LOG"
.venv/bin/python train.py \
  --config configs/heads_up_chain_8bin_v1.yaml \
  --name heads_up_chain_8bin_v1 \
  >> "$LOG" 2>&1

echo "[$(date)] overnight chain done" | tee -a "$LOG"
