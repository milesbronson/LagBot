# Cloud training (RunPod + SkyPilot)

Rented-GPU training for LagBot, built on a hard-won playbook: **assume
every step can fail silently and make each failure cost ≤ one small
increment.** Prepaid balance is the last-resort cap; the layers below
make the common failures cheap.

## Should you even rent compute? (honest answer)

**Not for the legacy lineages.** A 2M-step generation takes **~15 min
on the local M4 Pro** (measured across 100+ registry generations) — an
overnight local run buys 30 generations for free, and SB3's MLP PPO
barely uses a GPU anyway.

**Yes for the real-equity lineage and parallel experiments.**
`configs/heads_up_real_equity_v1.yaml` turns on genuine Monte-Carlo
post-flop equity (~200 rollouts per observation) — generations go from
minutes to hours, and that's when farming legs out is worth $1–2 each.
Same for running several chains (seeds/configs) side by side.

## Setup (once)

1. RunPod account → Settings → API key → `~/.runpod/config.toml`.
   **Load prepaid credit small ($10–25)**: an empty balance physically
   cannot bill.
2. `uv add "skypilot[runpod]"` then verify: `sky check runpod`.
3. Optional mailbox: create a private repo (e.g. `lagbot-artifacts`),
   mint a fine-grained token with Contents-RW on that repo only, and
   export `GH_ARTIFACT_TOKEN` + `MAILBOX_REPO=owner/repo` in the shell
   that launches. Never commit the token.
4. Install the watchdog (see below).

## Launching a run

```bash
# NEVER raw `sky launch` — the wrapper is the product:
cloud/launch.sh real_equity_gen0 configs/heads_up_real_equity_v1.yaml
```

`launch.sh` does, in order — each step exists because skipping it has
cost real money in the source playbook:

1. **Preflight locally, free**: full test suite + a smoke run of the
   exact `train.py` entry point on `configs/smoke_config.yaml`, with the
   output grepped for the reward gauge (a metric that exists in code but
   never prints reads as "job finished" with garbage results).
2. `caffeinate`s the launcher so a closed lid can't orphan the pod.
3. `sky launch -y -i 720 --down` — 12h idle autostop as the fallback
   cap behind the run's own 6h `timeout`.
4. Polls `sky queue` (never trusts log streams — a broken stream looks
   like a dead job; do not relaunch over a live cluster).
5. **Fetch → verify checkpoint exists → only then `sky down`.**
   Teardown-before-fetch has destroyed finished runs; if verification
   fails the cluster stays up for manual rescue.
6. On launch failure: tears down the possibly-half-created cluster
   (half-failed launches leave billing pods, sometimes INIT-wedged with
   no autostop — the single worst leak class).

The task yaml (`lagbot-train.sky.yaml`) fast-fails on ARM nodes and
throttled CDNs before any expensive install, hard-caps the run with
`timeout 6h`, and — as its **last act** — pushes a results tarball to
the mailbox so artifacts survive even if the laptop sleeps and every
fetch fails.

## Registry semantics on the pod

`.skyignore` keeps `models/` (multi-GB) off the pod, so the pod starts
with an **empty registry**: `seed_anchors.py` registers the 10 anchors,
then training builds a fresh lineage. `launch.sh` fetches the pod-side
`registry.json` as `cloud/registry_<run>.json`; merge those cards into
the local `models/registry.json` by hand (ids are distinct, so it's a
JSON merge, not a rewrite). Resuming a cloud chain from a local parent
would need the parent zip shipped via `file_mounts` — not wired up yet,
deliberately: fresh lineages don't need it.

## Watchdog (assume you forget everything)

```bash
# launchd, every 15 min (macOS):
sed "s|__REPO__|$(pwd)|" cloud/com.lagbot.podwatchdog.plist \
  > ~/Library/LaunchAgents/com.lagbot.podwatchdog.plist
launchctl load ~/Library/LaunchAgents/com.lagbot.podwatchdog.plist
```

`watchdog.sh` asks the RunPod API (provider truth, not `sky status`)
for live pods every 15 min, prints them, and kills anything older than
26h. `cloud_audit.sh` is the manual sweep: lists pods + balance, and
`--kill` terminates everything.

## Cost model

| leg | cost |
|---|---|
| 2M-step generation, A40 | ~$1–2 |
| same leg on a pricier fallback region | up to ~$6 — budget for it |
| ARM node draw with fast-fail | ~$0.05 |
| worst single incident with all guards | ≈ one leg, not the balance |

## Status

Scaffolding is complete and lint-clean but **has not been launched
against a live RunPod account** (needs an API key + prepaid credit).
First launch checklist: `sky check runpod` green → run
`cloud/launch.sh` with the default config → watch the first 10 minutes
of `sky logs` → confirm the watchdog sees the pod → walk away.
