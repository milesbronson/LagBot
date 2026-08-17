#!/usr/bin/env bash
# Manual sweep: list every RunPod pod + current balance from the API
# (provider truth, not `sky status`). With --kill, terminate them all.
#
#   cloud/cloud_audit.sh          # list
#   cloud/cloud_audit.sh --kill   # list, then terminate everything
set -euo pipefail

KEY="${RUNPOD_API_KEY:-$(grep -m1 'apikey' ~/.runpod/config.toml 2>/dev/null | sed 's/.*= *"\{0,1\}\([^"]*\)"\{0,1\}/\1/')}"
[ -n "$KEY" ] || { echo "no RunPod API key found (env RUNPOD_API_KEY or ~/.runpod/config.toml)"; exit 1; }
AUTH="Authorization: Bearer $KEY"

echo "== balance =="
curl -sf https://rest.runpod.io/v1/user -H "$AUTH" \
  | python3 -c 'import sys,json; d=json.load(sys.stdin); print(f"  credit: ${d.get(\"clientBalance\", d.get(\"balance\", \"?\"))}")' \
  || echo "  (balance endpoint unavailable)"

echo "== pods =="
PODS_JSON=$(curl -sf https://rest.runpod.io/v1/pods -H "$AUTH")
echo "$PODS_JSON" | python3 -c '
import sys, json
pods = json.load(sys.stdin)
if not pods: print("  none — clean")
for p in pods:
    print(f"  {p.get(\"id\")}  {p.get(\"name\",\"?\")}  {p.get(\"desiredStatus\")}  created={p.get(\"createdAt\")}  $/hr={p.get(\"costPerHr\",\"?\")}")'

if [ "${1:-}" = "--kill" ]; then
  echo "$PODS_JSON" | python3 -c 'import sys,json; [print(p["id"]) for p in json.load(sys.stdin)]' | while read -r pid; do
    curl -sf -X DELETE "https://rest.runpod.io/v1/pods/$pid" -H "$AUTH" \
      && echo "  killed $pid" || echo "  FAILED to kill $pid"
  done
fi
