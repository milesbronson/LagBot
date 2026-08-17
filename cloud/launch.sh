#!/usr/bin/env bash
# LagBot cloud launcher — the ONLY sanctioned way to start a rented run.
#
# Order of operations (each exists because skipping it has cost money):
#   1. preflight locally (tests + smoke of the exact run command) — free
#   2. caffeinate the launcher so lid-close can't orphan the pod
#   3. sky launch with autostop + --down
#   4. poll job state -> fetch artifacts -> VERIFY -> only then sky down
#   5. on any launch failure: sky down the half-created cluster
#   6. NEVER auto-relaunch on a stream error — poll `sky queue` for truth
#
# Usage:
#   cloud/launch.sh <RUN_NAME> [CONFIG]
#   CONFIG defaults to configs/heads_up_real_equity_v1.yaml
set -euo pipefail
cd "$(dirname "$0")/.."

RUN_NAME="${1:?usage: cloud/launch.sh <RUN_NAME> [CONFIG]}"
CONFIG="${2:-configs/heads_up_real_equity_v1.yaml}"
CLUSTER="lagbot-${RUN_NAME}"
YAML="cloud/lagbot-train.sky.yaml"
FETCH_DIR="models/${RUN_NAME}"

echo "== [0/5] sanity =="
command -v sky >/dev/null || { echo "skypilot not installed: uv add 'skypilot[runpod]'"; exit 1; }
sky check runpod 2>&1 | grep -q "enabled" || { echo "runpod not enabled in sky check"; exit 1; }
[ -f "$CONFIG" ] || { echo "no such config: $CONFIG"; exit 1; }

echo "== [1/5] preflight (local, free) =="
uv run --no-sync python -m pytest -q || { echo "PREFLIGHT FAIL: tests red — not spending money on broken code"; exit 1; }
# Smoke the EXACT entry point with a tiny config, and grep the output for
# the live gauges we depend on — a metric that exists in code but never
# prints has burned this playbook 4x.
SMOKE_OUT=$(mktemp)
uv run --no-sync python train.py --config configs/smoke_config.yaml --name "_preflight_smoke" 2>&1 | tee "$SMOKE_OUT" | tail -5
grep -q "Avg Reward\|avg_reward" "$SMOKE_OUT" || { echo "PREFLIGHT FAIL: reward gauge never printed"; exit 1; }
rm -rf models/_preflight_smoke metrics/_preflight_smoke logs/_preflight_smoke
echo "preflight OK"

echo "== [2/5] launch (autostop 12h idle, --down) =="
LAUNCH_CMD=(sky launch -y -i 720 --down -c "$CLUSTER" "$YAML" \
  --env CONFIG="$CONFIG" --env RUN_NAME="$RUN_NAME" \
  --env GH_ARTIFACT_TOKEN="${GH_ARTIFACT_TOKEN:-}" \
  --env MAILBOX_REPO="${MAILBOX_REPO:-}")
if command -v caffeinate >/dev/null; then
  caffeinate -i "${LAUNCH_CMD[@]}" || { echo "LAUNCH FAILED — tearing down half-created cluster"; sky down -y "$CLUSTER" || true; exit 1; }
else
  "${LAUNCH_CMD[@]}" || { echo "LAUNCH FAILED — tearing down half-created cluster"; sky down -y "$CLUSTER" || true; exit 1; }
fi

echo "== [3/5] wait for job (poll queue — do NOT trust log streams) =="
while true; do
  STATE=$(sky queue "$CLUSTER" 2>/dev/null | awk 'NR>1 && $1==1 {print $(NF)}' | head -1 || true)
  echo "$(date '+%H:%M') job state: ${STATE:-unknown}"
  case "$STATE" in
    SUCCEEDED) break ;;
    FAILED|FAILED_SETUP|CANCELLED) echo "job ended: $STATE — leaving cluster up for sky logs; run 'sky down $CLUSTER' when done debugging"; exit 1 ;;
    *) sleep 120 ;;
  esac
done

echo "== [4/5] fetch -> verify -> down =="
mkdir -p "$FETCH_DIR" "metrics/${RUN_NAME}"
rsync -az "${CLUSTER}:~/sky_workdir/models/${RUN_NAME}/" "$FETCH_DIR/" || echo "model fetch failed"
rsync -az "${CLUSTER}:~/sky_workdir/metrics/${RUN_NAME}/" "metrics/${RUN_NAME}/" || echo "metrics fetch failed"
rsync -az "${CLUSTER}:~/sky_workdir/models/registry.json" "cloud/registry_${RUN_NAME}.json" || echo "registry fetch failed"
if [ ! -s "$FETCH_DIR/final_model.zip" ] && [ ! -s "$FETCH_DIR/best_model.zip" ]; then
  echo "VERIFY FAILED: no checkpoint fetched — NOT tearing down. Inspect with: ssh $CLUSTER / sky logs $CLUSTER"
  exit 1
fi
echo "verified: $(ls -la "$FETCH_DIR" | head -5)"
sky down -y "$CLUSTER"

echo "== [5/5] done =="
echo "Checkpoints: $FETCH_DIR"
echo "Pod-side registry snapshot: cloud/registry_${RUN_NAME}.json (merge cards into models/registry.json manually)"
