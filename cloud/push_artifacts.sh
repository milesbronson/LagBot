#!/usr/bin/env bash
# Artifact mailbox — the pod's LAST act. Pushes a tarball of the run's
# checkpoints + metrics as a tagged release to a dedicated private repo,
# so results survive even if the local machine sleeps and every fetch
# fails. Fetch-before-teardown remains the primary path; this is the
# backstop.
#
# Requires (passed via sky --env, never committed):
#   GH_ARTIFACT_TOKEN  fine-grained PAT, Contents-RW on MAILBOX_REPO only
#   MAILBOX_REPO       e.g. milesbronson/lagbot-artifacts
set -euo pipefail

RUN_NAME="${1:?usage: push_artifacts.sh <RUN_NAME>}"
: "${GH_ARTIFACT_TOKEN:?GH_ARTIFACT_TOKEN unset}"
: "${MAILBOX_REPO:?MAILBOX_REPO unset}"

TAG="run-${RUN_NAME}-$(date -u +%Y%m%dT%H%M%SZ)"
TARBALL="/tmp/${TAG}.tar.gz"
tar czf "$TARBALL" \
  "models/${RUN_NAME}" \
  "metrics/${RUN_NAME}" \
  models/registry.json 2>/dev/null || tar czf "$TARBALL" "models/${RUN_NAME}"

API="https://api.github.com/repos/${MAILBOX_REPO}"
AUTH="Authorization: Bearer ${GH_ARTIFACT_TOKEN}"

RELEASE_ID=$(curl -sf -X POST "$API/releases" -H "$AUTH" \
  -d "{\"tag_name\":\"${TAG}\",\"name\":\"${TAG}\",\"prerelease\":true}" \
  | python3 -c 'import sys,json; print(json.load(sys.stdin)["id"])')

curl -sf -X POST \
  "https://uploads.github.com/repos/${MAILBOX_REPO}/releases/${RELEASE_ID}/assets?name=$(basename "$TARBALL")" \
  -H "$AUTH" -H "Content-Type: application/gzip" \
  --data-binary @"$TARBALL" >/dev/null

echo "mailbox: pushed $(du -h "$TARBALL" | cut -f1) as ${TAG} to ${MAILBOX_REPO}"
