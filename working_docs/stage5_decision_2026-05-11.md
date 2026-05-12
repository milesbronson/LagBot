# Stage 5 Decision Doc — Move auto-play into env / delete wrapper?

**Date:** 2026-05-11
**Status:** Decision required.
**Predecessor:** `working_docs/refactor_design_2026-05-10.md` §2.3 / §4 (Migration Path step 5).
**Decision owner:** you (project owner).

This doc isolates the **five sub-decisions** that hide inside Stage 5, lists what changes for each, and gives my recommendation. Stage 5 is *not* one decision — it's five entangled ones.

---

## TL;DR — my overall recommendation

**Do not move the auto-play loop into the env.** Keep the wrapper. Decouple env→tracker calls via event hooks instead (planning-doc §3.2). This collapses "Stage 5" from a 2-day reshape into a 2-hour cleanup, costs nothing structurally, and avoids forcing the backend's async/human seat lifecycle to fit a training-only abstraction.

Reasoning summary: training and backend have different lifecycles. Training is synchronous and batched — opponents are picked from a list and asked for an action immediately. Backend is async and renders frames between actions — humans answer via WebSocket. An env that auto-plays opponents in `step()` would force one lifecycle on the other. The wrapper is the right place for "training-specific opponent orchestration."

---

## Section 0 — How the environment is currently set up

Before deciding whether to change anything, the current shape needs to be laid out so the decision is informed by reality, not memory.

### 0.1 The three layers

```
SB3 PPO trainer
   └─ DummyVecEnv                       (SB3 vectorization, not ours)
       └─ Monitor                       (SB3 episode tracking, not ours)
           └─ OpponentAutoPlayWrapper   (ours — train.py)
               └─ TexasHoldemEnv        (ours — src/poker_env/texas_holdem_env.py)
                   └─ GameState         (ours — src/poker_env/game_state.py)
                       └─ Player[ ]     (ours — src/poker_env/player.py)
```

Only the bottom three (wrapper, env, game state) are ours. Everything above is SB3 infrastructure that expects a Gym-style `(obs, reward, terminated, truncated, info)` API on each `step()`.

### 0.2 `TexasHoldemEnv` — the pure game state machine

Lives in `src/poker_env/texas_holdem_env.py`. Responsibilities:

- Owns a `GameState` containing `Player` objects (one per seat).
- Action space: discrete (fold, call, raise bins, all-in).
- Observation space: 161 dims = 53 base features + (9 opponent slots × 12 features).
- `step(action)`: applies one action *to whoever is currently to act*, advances `game_state.current_player_idx`, returns the new observation.
- Reward attribution: **terminal reward** = `(player.stack - starting_stack) / starting_stack` for the seat at `self.learning_agent_id`. **Intermediate fold-shaping** = ±0.1 × hand-equity gap, fired only when the learning agent folds. Wrapper does NOT recompute these — env owns reward.
- `learning_agent_id` is now a **constructor parameter** (default 0). Validated to be in `[0, num_players)`. Used everywhere reward is attributed.
- Emits no events yet (planning-doc §3.2 is unimplemented). The env directly calls `self.opponent_tracker.record_action(...)` from within game-logic methods. This is the coupling Step 9 will fix.

What the env explicitly does NOT know:
- It does not know "which player is the learner" beyond `learning_agent_id` as an index. It treats every seat uniformly.
- It does not know agents exist. It only knows `Player` objects with stacks/cards/bets.
- It does not auto-play anyone. It advances exactly one action per `step()` call.

### 0.3 `Player` ↔ `BaseAgent` — the bidirectional link (just landed)

After the recent refactor, every `Player` can be married to a `BaseAgent`:

```python
class Player:
    agent: Optional[BaseAgent]              # back-reference, default None
    def seat_agent(self, agent): ...        # binds both sides

class BaseAgent:
    player_id: Optional[int]                # back-reference, default None
    def seat(self, player): ...             # mirror entry point
```

After `player.seat_agent(agent)` (or equivalently `agent.seat(player)`):
- `player.agent is agent`
- `agent.player_id == player.player_id`

Both directions resolve to the same binding. The wrapper uses this to look up "who plays in seat 3?" without doing index math on a list of opponents.

### 0.4 `OpponentAutoPlayWrapper` — the training orchestration layer

Lives in `train.py`. Responsibilities:

- At construction: takes `(env, opponents_list)`. Seats each opponent agent into its `Player` via `agent.seat(player)`. Builds `opponents_by_id: Dict[player_id, agent]`.
- `reset()`: calls `env.reset()`, then auto-plays opponents until it's the learner's turn. This handles button rotation — without it, the learner's action would land on the wrong seat after a button change.
- `step(learner_action)`:
  1. Apply learner's action via `env.step()`.
  2. Capture env's reward (env does attribution; wrapper preserves it).
  3. While hand isn't over and current seat isn't the learner: ask `opponents_by_id[current_seat].select_action(obs)`, step the env, **accumulate** the env's reward (so intermediate shaping survives across the opponent loop).
  4. On terminal: call `_record_opponent_profits()` to log per-opponent profit deltas.
  5. Return `(obs, accumulated_reward, terminated, truncated, info)`.
- Exposes the env's `observation_space`, `action_space`, `metadata` so SB3 sees a normal Gym env.

What the wrapper explicitly does NOT do anymore:
- It does not overwrite the env's reward. That bug (planning-doc bugs §1.2) is gone and `tests/test_env/test_reward_attribution.py` pins it.
- It does not hardcode "player 0 is the learner." It reads `env.learning_agent_id` and uses `opponents_by_id` for lookups.

### 0.5 Data flow on a single training step

For a heads-up game (`num_players=2`, `learning_agent_id=0`):

```
SB3 → wrapped_env.step(action=1)            # learner says "call"
  wrapped_env.env.step(1)                   # env applies action to player 0
    → returns (obs, env_reward_0, term, _, _)
  accumulated_reward = env_reward_0
  while current_player_idx != 0 and not term:
    opp = opponents_by_id[1]                # the seated opponent agent
    opp_action = opp.select_action(obs)     # opponent decides
    wrapped_env.env.step(opp_action)        # env applies to player 1
      → returns (obs, env_reward_1, term, _, _)
    accumulated_reward += env_reward_1
  return (obs, accumulated_reward, term, _, info)
```

Reward attribution is entirely the env's responsibility. The wrapper just sums.

### 0.6 Where things still aren't ideal

- The opponent tracker is called directly from game-logic methods (e.g. `_handle_fold` calls `self.opponent_tracker.record_action(...)`). Planning-doc Step 9 / §3.2 wants this decoupled via an event-listener hook so the env doesn't import tracker types.
- `Player` and `BaseAgent` are still separate classes that point at each other. Planning-doc §2 ultimately wants `Agent` to own a `SeatState` (composition, not bidirectional reference). That's Step 7 territory — not yet started.
- The wrapper hardcodes "all non-learner seats are bots with synchronous `select_action`." A human in a seat would need a different layer (the backend handles that today by NOT using this wrapper).

### 0.7 What this means for the five decisions below

Read Sections 0.2–0.4 before each decision. The current setup is more decoupled than the planning doc's "before" sketch implies — most of what Stage 5 sets out to fix has already been fixed by the bidirectional link + parameterized `learning_agent_id` + reward attribution + wrapper cleanup. The remaining question is whether to *delete the wrapper* or *let the env emit events to it*. That's what the five decisions are really asking.

---

## The five sub-decisions

### Decision 1: Env constructor shape

**Today:**
```python
env = TexasHoldemEnv(num_players=3, learning_agent_id=0, ...)
# Env owns Player construction internally.
```

**Planning doc §2.3 proposes:**
```python
agents = [PPOAgent(...), CallAgent(...), CallAgent(...)]
env = TexasHoldemEnv(agents=agents, learning_agent_id=0, ...)
# Caller owns Agent lifecycle; env derives num_players from len(agents).
```

**What this affects:**
- Every test file that builds an env directly: `tests/test_env/*` (≈18 files), `tests/test_training/*`, `tests/test_agents/*`.
- `train.py`, `play.py`, `backend/services/game_session.py`, `src/training/scenario_tree_env.py`.
- All YAML configs and scripts that talk about `num_players`.

**My recommendation: KEEP `num_players` AS-IS.** Add a separate `env.seat_agents(agents_by_id: Dict[int, Agent])` method that's *optional* — callers that don't seat agents (e.g. the backend, where humans are handled at the session layer) just don't call it. The env constructs Players first, then agents get bound to them. This:
- Doesn't break existing callers.
- Already matches what we did with `Player.seat_agent(agent)` in the current refactor.
- Keeps env construction cheap (no upfront agent instantiation needed for testing game logic in isolation).

---

### Decision 2: Where does the auto-play loop live?

**Three options:**

**A. In `env.step()` — env owns the loop, wrapper deleted.**
The env, after applying the learner's action, advances through opponents until it's the learner's turn or the hand ends.

**B. In the wrapper (status quo after refactor) — env stays a pure state machine.**
Env exposes only "one action at a time." Wrapper orchestrates opponent calls.

**C. Hybrid — env exposes `play_until(predicate)`, wrapper and backend call it.**
Env grows a helper method that advances until a condition is met; callers decide the predicate.

**What this affects:**
- Option A: `OpponentAutoPlayWrapper` deleted. Backend `game_session.py` has to re-implement its own loop because humans can't be `select_action()`'d synchronously. Scenario-tree env has to opt out of auto-play to do replay. Net: ~600 lines removed, ~200 added across backend + scenario env.
- Option B: zero changes — we keep the current shape from this refactor.
- Option C: env gains ~30-line helper, wrapper shrinks ~20 lines, backend can dedupe ~50 lines of opponent-loop code.

**My recommendation: B (status quo) for now, with C as a future option if we find ourselves duplicating the loop in three places.**

Rationale: option A's only real win is "one source of truth for whose-turn-is-it" — but that source of truth already exists (`env.game_state.current_player_idx`). The wrapper just *reads* it. Moving the loop into the env doesn't centralize anything that isn't already centralized; it just moves the orchestration from one file to another, and forces the backend to fight the new design.

---

### Decision 3: Profit-tracker integration

**Today:** Wrapper calls `_record_opponent_profits(learning_agent)` on terminal.

**Two ways to do this post-Stage-5:**

**A. Env owns the profit tracker.** Pass `profit_tracker` to env constructor; env records on hand end internally.

**B. Profit tracker subscribes to `HandEndEvent`.** Planning doc §3.2 event hook style — env emits, anyone interested listens.

**What this affects:**
- Option A: tighter coupling, simpler today. Env imports `OpponentProfitTracker` from `src/training/`.
- Option B: env stays training-agnostic. New event-hook plumbing needed. Future loggers and the backend can also subscribe.

**My recommendation: B (event hooks), but only when we actually do event hooks for the tracker too** (planning doc §3.2). If we're not doing the tracker decoupling in this round, keep the profit tracker in the wrapper exactly where it is. Don't half-build the listener system.

---

### Decision 4: Scenario-tree env compatibility

**Today:** `src/training/scenario_tree_env.py` wraps the env and *manually* re-plays from a saved decision point. It expects `env.step()` to advance exactly one action.

**Affected by:**
- If env auto-plays internally (Option 2A), scenario env breaks immediately — replaying a "flop decision" requires single-step control.
- If wrapper holds the loop (Option 2B/C), scenario env stays functional.

**What this affects:**
- Only matters if you pick Decision 2 = A.
- Fix would be a `step_single_action()` escape hatch on env. Doable but adds API surface area.

**My recommendation: not a problem if we keep Decision 2 = B.**

---

### Decision 5: Migration strategy

**Two strategies:**

**A. Big-bang.** One PR. Every caller updated. Tests rewritten. Old wrapper deleted in same commit.

**B. Incremental.** Env grows new APIs alongside old. Wrapper kept as thin shim. Callers migrate one at a time over multiple PRs. Old code deleted only after every caller is moved.

**What this affects:**
- Big-bang: faster to "done"; high risk because we have ~9 real failing tests right now (all-in / chip-accounting / preflop bugs). Reshaping the env while those bugs exist means we won't know whether a regression came from the reshape or from one of the pre-existing bugs.
- Incremental: slower; each step keeps the suite green; bugs stay separable.

**What this affects (concrete):**
- Test files: ≈25 if changing env signature.
- Production callers: `train.py`, `play.py`, `backend/services/game_session.py`, `src/training/scenario_tree_env.py`.

**My recommendation: B (incremental) regardless of which option you pick for Decision 2.** And **fix the chip-accounting bugs FIRST** so we're not refactoring on top of broken arithmetic.

---

## What's NOT affected by any of these decisions

- The Player↔Agent marriage already shipped (`Player.agent`, `BaseAgent.player_id`, `seat()` / `seat_agent()`). That's done.
- Reward attribution. The env owns it; the wrapper no longer overrides it; tests pin the behavior. Done.
- Opponent tracker centralization. Decision 3 (event hooks) is the only thing that touches it, and only optionally.
- The 161-dim observation space and 4 added opponent features.

---

## Recommended next action

**Decide Decision 2 first** — it's the load-bearing one. If you pick **B (keep wrapper)**, the other four collapse:
- Decision 1: keep `num_players`. No change.
- Decision 3: profit tracker stays in wrapper. No change.
- Decision 4: scenario env unaffected. No change.
- Decision 5: there's nothing left to migrate.

Stage 5 effectively becomes "fix the chip-accounting bugs the audit surfaced, then call it done." That's where I'd spend the next session.

If you pick **A (move into env)**, all five decisions need to be made and the migration is ≈2 days. We should also fix the chip-accounting bugs *before* starting.

---

## Open questions to answer before starting Stage 5

1. Is anyone currently relying on the env being able to construct without agents (e.g. for unit-testing game state directly)? If yes, Decision 1 must keep `num_players`.
2. Is the backend's `game_session.py` going to be rewritten anyway? If yes, "force backend to refit" (Decision 2A) is cheaper. If no, prefer 2B.
3. Are there any planned features (multi-PPO training, tournament eval, simultaneous human + bot in same session) that need env-level orchestration? If yes, lean toward 2A. If no, keep wrapper.
