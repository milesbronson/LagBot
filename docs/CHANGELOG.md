# Changelog

History of significant changes and fixes to LagBot.

> Note: entries are historical — numbers like observation dims reflect the
> code *at that date*. Current architecture (see README): 161-dim obs =
> 53 base + 9 opponents × 12 HUD features.

---

## 2026-08-17: RULES V2 — correctness over compatibility

Backwards compatibility with pre-v2 checkpoints was deliberately dropped
(owner's call). Every model in the registry trained under the old rules
and is now a historical artifact; the next lineage
(`configs/heads_up_real_equity_v1.yaml`) is the first correct-world chain.

- **Real Monte-Carlo equity always on** — the `real_postflop_equity`
  compatibility flag is gone; `hand_strength` is genuine on every street.
- **Heads-up rules corrected**: the button posts the small blind, acts
  first preflop and last post-flop. The old engine had the button in the
  big blind with inverted post-flop order — every HU hand ever trained or
  played was positionally backwards.
- **Betting-round completion rewritten** around an acted-since-last-raise
  set: the big blind actually gets its option (the old action-count quota
  closed the round early in any pot with a preflop fold), and all-in
  run-outs no longer force phantom decisions on dead streets.
- **Min-raise legality**: an undersized raise via the amount paths is a
  call (never a silent fold, never a 1-chip re-raise reopening the
  betting), and a short all-in no longer shrinks the min-raise ladder.
- **Observation fix**: `pos` is now button-relative (the raw seat index
  was a constant for the learner — no positional signal at all).
- Eval gates now mirror the training table rules (`min_raise_multiplier`,
  rake, all-in availability) via `env_overrides`; under per-episode
  rotation, `trained_against_ids` and the gate exclusion list cover every
  card the wrapper ever seated, not just the first draw.
- Locked in by `tests/test_env/test_rules_v2.py`.

---

## 2026-08-14: Audit fixes — silent equity bug, eval-gate action space, web UI

**Post-flop hand equity was a constant 0.5 in every observation** (silent
`treys` API bug: `deck.draw(1)` returns a list; the resulting `TypeError`
was swallowed by a bare `except` that returned 0.5). Every checkpoint in the
registry was trained on that constant. The Monte-Carlo path is fixed
(per-player cache key, both hands scored on the same completed board), but a
regression duel showed existing champions collapse from +3,384 to −1,822
BB/100 when suddenly fed real equity — so the fix is gated behind a new env
flag `real_postflop_equity` (config key of the same name), default **off**.
Flip it on only for a fresh training lineage.

**`BestCheckpointCallback` promoted checkpoints in the wrong action space** —
it built its `EvalGate` without `raise_bins`, so 8-bin candidates were scored
in a 6-action env where actions ≥ 6 silently became calls. `raise_bins` now
threads through from the training config.

**EvalGate duplicate (mirrored-seat) scoring finished**: per-hand deck
seeding replays identical decks with seats swapped so card luck and seat bias
cancel; global RNG state and agents' `deterministic` flags are saved/restored
around evals; `EvalResult` now records `passed`.

**Web UI**: Raise slider sent the raise portion without the call amount
("Raise" often executed as a call, and could silently fold after a short
all-in); showdown now reveals bot hole cards on the table; WebSocket guard is
hand-number-aware (opening bot actions of a new hand are no longer dropped);
reconnect payload reports the real hand state and replays the action log;
one shared model instance per session, loaded off the event loop; "Next
Hand" double-click guard; pot display no longer double-counts live bets.

---

## 2026-02-25: Multi-Player Frontend + Backend Overhaul

Complete rebuild of the frontend and backend for multi-player support, PostgreSQL hand history, and production Docker Compose. See `claude_workflow/frontend/work.md` for the full log.

**Backend:**
- Fixed model discovery path (now scans `models/` recursively by step count)
- Rewrote bot loop — player_id keyed agents, no broken index math
- Added `last_action` to WebSocket broadcasts
- Added PostgreSQL hand history (`backend/db/`)
- Added hand history REST endpoints

**Frontend:**
- Responsive poker table for 2-10 players
- Live hand history from WebSocket events
- Real opponent stats fetching and display
- Raise slider + quick bet buttons (½ Pot, Pot, 2x Pot)
- Bot action animation badges
- WebSocket reconnection with exponential backoff
- localStorage for game settings persistence

---

## 2026-02-09: Reward Calculation Bug Fixes (CRITICAL)

Two bugs were causing impossible reward values (e.g., -490 BB) that corrupted training.

**Bug 1: Mid-Hand Stack Reset**
- `reset_stacks_every_n_timesteps` could fire mid-hand in `step()`, injecting chips during play
- Fix: Moved stack reset logic from `step()` to `reset()` — resets only happen between hands

**Bug 2: Fragile Reward Calculation**
- Reward recomputed `starting_stack` from `stack + total_bet_this_hand` on every step
- Any mid-hand state change corrupted the value
- Fix: Use `player.starting_stack_this_hand` (stored once before blinds are posted)

**Before fix:** `agent/min_reward: -490` (impossible)
**After fix:** `agent/min_reward` stays within `[-100, 200]` range

**File:** `src/poker_env/texas_holdem_env.py`
**Tests added:** `tests/test_env/test_reward_bugs.py` (6 tests, all passing)

---

## 2026-02-07: Web Interface Initial Implementation

Created the complete React frontend + FastAPI backend from scratch. 65 files total.

**Backend (FastAPI):**
- REST API: 7 endpoints (create game, action, new hand, state, opponent stats, hand history, delete)
- WebSocket at `/ws/{session_id}` for real-time state updates
- `GameSession` wraps `TexasHoldemEnv` for web access
- `GameManager` singleton manages sessions by UUID
- Pydantic models for request/response validation
- CORS middleware, health check endpoint
- Docker containerization (`backend/Dockerfile`)

**Frontend (React 18 + TypeScript):**
- Elliptical poker table layout
- Real-time state updates via WebSocket
- Betting controls: Fold, Check/Call, Raise slider, All-In
- Opponent stats panel (VPIP, PFR, AF)
- Hand result modal
- New game configuration modal
- Tailwind CSS with poker-themed colors
- Docker containerization (`frontend/Dockerfile`)
- Docker Compose for full stack

**Scripts added:** `scripts/start_web.sh`, `scripts/check_setup.sh`

---

## 2026-02-06: Observation Space Expansion (108 → 125 dims)

Enhanced the observation space with richer features for better training signal.

**Card encoding:** 4 dims → 6 dims per card (added proper suit one-hot + present flag)

**New features added:**
- `hand_strength`: Monte Carlo equity estimate using Treys (~200 simulations)
- `pot_odds`: `amount_to_call / (pot + call)` normalized to [0, 1]
- `spr`: Stack-to-Pot Ratio, normalized and capped at 1.0

**Reward shaping:**
- Good folds (equity < 0.3): +0.1 reward
- Bad folds (equity > 0.6): -0.1 reward
- Terminal reward normalization changed from `/ (big_blind * 100)` to `/ starting_stack`

**Observation dims:** 108 → 125 (53 base + 72 opponent tracking)

**Files:** `src/poker_env/texas_holdem_env.py`, `train.py`

---

## 2026-02-05: All-In Side Pot Bug Fix

**Bug:** When a player tried to call with insufficient chips, they were forced to fold instead of going all-in. The scenario of two players going all-in with unequal stacks was broken.

**Fix in `src/poker_env/pot_manager.py`:**
```python
# Before: forced fold on insufficient chips
if amount < to_call:
    player.fold()

# After: allow all-in with whatever the player has
if amount >= player.stack and player.stack > 0 and player.stack < to_call:
    actual_bet = player.bet(player.stack)
    self.pots[0].add_chips(actual_bet)
    return actual_bet, "all-in"
```

**Result:** Players with shorter stacks can now go all-in correctly. Side pots are created and only the matched amount is contested.

---

## 2026-02-03: Initial Web Interface Architecture

Designed the initial architecture for the FastAPI + React web interface. See `claude_workflow/frontend/research.md` for full details.

Key decisions:
- FastAPI for backend (WebSocket support, fast, Python-native for game engine)
- React 18 + Zustand for frontend (lightweight state management)
- Vite dev server proxies `/api` and `/ws` to FastAPI — no CORS issues in dev
- `0.0.0.0` binding on both servers enables local network play without deployment
