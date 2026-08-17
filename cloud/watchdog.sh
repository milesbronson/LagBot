#!/usr/bin/env bash
# Pod watchdog — run every 15 min from cron/launchd. Queries the RunPod
# API directly (provider truth — `sky status` loses track of pods) and:
#   - stays silent when no pods exist
#   - warns loudly if ANY pod exists
#   - kills any pod older than MAX_AGE_HOURS (generous cap; a legit run
#     hits its own 6h timeout + 12h autostop long before this)
#
# Requires RUNPOD_API_KEY in the environment or ~/.runpod/config.toml.
# Install: cloud/README.md § Watchdog.
set -euo pipefail

MAX_AGE_HOURS="${MAX_AGE_HOURS:-26}"
KEY="${RUNPOD_API_KEY:-$(grep -m1 'apikey' ~/.runpod/config.toml 2>/dev/null | sed 's/.*= *"\{0,1\}\([^"]*\)"\{0,1\}/\1/')}"
[ -n "$KEY" ] || { echo "watchdog: no RunPod API key found"; exit 1; }

PODS_JSON=$(curl -sf https://rest.runpod.io/v1/pods -H "Authorization: Bearer $KEY")

# One python pass: print a status line per pod, emit over-age pod ids on
# stdout lines prefixed KILL: for the shell to act on.
REPORT=$(echo "$PODS_JSON" | python3 - "$MAX_AGE_HOURS" <<'EOF'
import json, sys
from datetime import datetime, timezone
max_age_h = float(sys.argv[1])
pods = json.load(sys.stdin)
if not pods:
    sys.exit(0)
for p in pods:
    age_h = None
    try:
        dt = datetime.fromisoformat(p.get("createdAt", "").replace("Z", "+00:00"))
        age_h = (datetime.now(timezone.utc) - dt).total_seconds() / 3600
    except Exception:
        pass
    age_txt = f"{age_h:.1f}h" if age_h is not None else "?"
    print(f"POD: {p.get('id')} ({p.get('name','?')}) age={age_txt} status={p.get('desiredStatus')}")
    if age_h is not None and age_h > max_age_h:
        print(f"KILL: {p['id']}")
EOF
)

[ -n "$REPORT" ] || exit 0   # no pods — quiet exit

echo "watchdog: RunPod pods ALIVE at $(date):"
echo "$REPORT" | grep '^POD:' | sed 's/^POD:/ /'

echo "$REPORT" | awk '/^KILL:/ {print $2}' | while read -r pid; do
  curl -sf -X DELETE "https://rest.runpod.io/v1/pods/$pid" -H "Authorization: Bearer $KEY" \
    && echo "watchdog: killed $pid (over ${MAX_AGE_HOURS}h cap)" \
    || echo "watchdog: FAILED to kill $pid"
done
