# Bug Inventory + Training Loop Rewrite

**Status:** Working document — iterate on this. Update checkboxes as items are fixed; add notes inline.

**Source material**: `audit_2026-05-10.md`, `training_review_2026-05-10.md`, `refactor_design_2026-05-10.md`, `player_agent_architecture_2026-05-10.md`.

---

## Part 1 — Bug Inventory

Severity legend: **P0** = corrupts training right now / silently wrong • **P1** = wrong but constrained blast radius • **P2** = cosmetic, drift hazard, or future-proofing

### 1.1 Opponent Tracker (`src/poker_env/opponent_tracker.py`)

- [ ] **P0** — `_update_opponent_stats` line 456-462: `three_bet_opportunities` only increments when the player themselves raised. A 3-bet opportunity is "someone else raised before you, and now it's your turn." Denominator is wrong; feature index `[3]` going to the network is junk.
- [ ] **P0** — `_update_opponent_stats` line 512-517: Fold-to-c-bet only fires when the *same player* both bet/raised the flop and folded the flop. That's impossible. Counters are always equal or always zero. Feature index `[5]` is dead.
- [ ] **P0** — `_update_opponent_stats` line 504-509: C-bet detection has no preflop-aggressor check. Any flop bet/raise by anyone who raised preflop is counted as a c-bet, even if someone else bet first and they raised.
- [ ] **P0** — `_update_opponent_stats` line 465-468: WTSD heuristic excludes losers. Only counts players who won the hand OR folded ≥5 actions in. Players who reached showdown and lost are invisible.
- [ ] **P1** — `_update_opponent_stats` line 495-501: Squeeze detection checks only this player's own actions, not whether someone else raised + a third player called. After any hand where the player called then raised preflop, both `squeeze_opportunities` and `squeeze_attempts` increment.
- [ ] **P1** — `_update_opponent_stats` line 482-491: Fold-to-3-bet-after-raising uses `opponent.raised_preflop > 0` as a lifetime gate. After the player ever raises preflop, this branch fires on every subsequent hand.
- [ ] **P1** — `texas_holdem_env.py:262-267` `_calculate_player_positions` records `{player_id: list_index}`, not seat relative to dealer. Position never changes for a given player → every position-based stat is keyed by `player_id`.
- [ ] **P1** — `_recalculate_stats` line 230-238: AF computed from `recent_actions` deque (maxlen=50). VPIP/PFR are lifetime. Mixed regimes mean AF swings on recency while VPIP/PFR don't. Pick one regime.
- [ ] **P2** — `Action.BET` is unreachable: env's `_string_to_action_enum` matches "raise" before "bet" and the action space never produces a "bet" string. All tracker branches keyed on `Action.BET` are dead code; the `0.6` action-value encoding in `recent_actions` features can never be produced.

### 1.2 Environment + Reward (`src/poker_env/texas_holdem_env.py`)

- [ ] **P1** — `OpponentAutoPlayWrapper.step` (`train.py:103-106, 129-133`) overwrites env reward whenever the hand ends, discarding intermediate fold-shaping (±0.1) when the agent's own fold ends the hand. Fix: either delete the wrapper override (env already attributes correctly to `learning_agent_id`), or change `reward = ...` to `reward += ...`.
- [ ] **P2** — Dead duplicate method `step_with_raise()` still in the file but unreferenced.

### 1.3 Training scripts & wrapper

- [ ] **P0** — `create_opponents()` (`train.py:244`) does not validate `opp.model.observation_space.shape == env.observation_space.shape`. Older 68-dim or differently-ordered 125-dim checkpoints can be loaded against the current env and either crash or silently mis-map.
- [ ] **P0** — `OpponentPPO.__init__` defaults to `deterministic=False` (`opponent_ppo.py:43`); opponents are loaded **once** at run start and never refreshed. The "self-play" loop trains against frozen stochastic snapshots of the starting weights for the entire 3M-step run. Decide: refresh-every-N or call it "fine-tune vs snapshot."
- [ ] **P1** — `train_from_checkpoint.py:71` and `resume_training.py:64` use `PPO.load()` which **overwrites all YAML hyperparameters with whatever is in the zip**. Cannot change `learning_rate`, `ent_coef`, etc. on resume. Fix: pass `custom_objects={'learning_rate': ..., 'clip_range': ...}` to `PPO.load`.
- [ ] **P1** — `TrainingCallback._on_step` saves on `self.n_calls % save_freq` (`ppo_agent.py:186`). Correct for single env; wrong under vec env; `n_calls` resets on every `learn()` call so resumed runs save at wrong absolute steps. Use `self.num_timesteps`.
- [ ] **P1** — `MetricsCallback` action buckets (`callbacks.py:124`) hardcode `{2,3,4}=raise, 5=all-in` against `Discrete(6)`. Works only by coincidence with default `raise_bins=[0.5,1.0,2.0]`. Derive indices from `env.action_space.n` instead.
- [ ] **P1** — `MetricsCallback` "win rate" is just `reward > 0` (`callbacks.py:64`), not actual hand-win rate. Hands won for less than blinds-posted count as losses with normalized stack-delta reward. Either rename to `positive_reward_rate` or compute from `info`.
- [ ] **P1** — `OpponentProfitTracker` is wired into `train.py` only. `train_from_checkpoint.py`, `train_diverse_opponents.py`, `train_vs_two_bots.py`, `resume_training.py` all construct `OpponentAutoPlayWrapper` *without* a tracker — no per-opponent profit recorded.
- [ ] **P1** — `OpponentProfitTracker` splits hand profit equally across opponents in a 3-player game. Misleading attribution; one opponent actually contested the pot.
- [ ] **P2** — `train.py:332` (and every other entry-script) hardcodes `num_players=3`, ignoring `configs/default_config.yaml:6` (`num_players: 6`). Either read YAML or delete the field.
- [ ] **P2** — Bare `except Exception` in the wrapper (`train.py:78`, `184`) swallows real bugs; prints a warning once per hand and drops profit attribution.
- [ ] **P2** — `train_vs_two_bots.py:26` constructs env with `track_opponents=True` but no `min_raise_multiplier` or `reset_stacks_every_n_timesteps` — silently uses env defaults, diverging from YAML.
- [ ] **P2** — `PPO(..., env=env)` then later `set_env(wrapped_env)` rebuilds the rollout buffer on `set_env`. Harmless given matching spaces but load-bearing across the pipeline.

### 1.4 Reproducibility

- [ ] **P0** — No seeds anywhere: SB3, numpy, torch, random, `env.reset(seed=...)` all unseeded. Same command twice produces different trajectories.
- [ ] **P0** — No config snapshot in the run dir. `metrics/<run>/` has JSON metrics but no YAML, no git commit, no library versions.
- [ ] **P0** — No record of which opponent checkpoints were used in a run. Opponent identity is mtime-dependent — `touch`ing a file changes the opponents of the next run.
- [ ] **P1** — `random.sample` in `train_diverse_opponents.py:61` unseeded.
- [ ] **P2** — SB3/torch versions not embedded per checkpoint.

### 1.5 Self-play infrastructure

- [ ] **P0** — There is **no automated self-play loop**. Each "generation" is a manual `python train.py --name gen_N` invocation. No orchestration, no opponent pool, no evaluation gate. The "loop" is documentation, not code.
- [ ] **P0** — No evaluation gate. Nothing measures whether `gen_N+1` actually beats `gen_N`. Currently relying on vibes + TensorBoard.
- [ ] **P1** — Opponent pool is "two slots filled by mtime". Once a third run exists, oldest is silently dropped. No replay buffer. Classic non-transitive cycles possible (gen_3 beats gen_2 by exploit, gen_5 forgets and re-loses to gen_2-style play).
- [ ] **P1** — `train_from_checkpoint.py` is **mirror self-play against a frozen starting snapshot** — the weakest form of self-play, prone to collapse modes (everyone folds, everyone all-ins).

### 1.6 Test gaps

- [ ] **P0** — 3-bet detection across hand action sequence
- [ ] **P0** — Fold-to-c-bet across players (A bets, B folds)
- [ ] **P0** — C-bet only counts preflop aggressor opening the flop
- [ ] **P0** — WTSD includes losers who saw showdown
- [ ] **P1** — Position rotates with the button across hands
- [ ] **P1** — Wrapper preserves intermediate fold shaping when terminal
- [ ] **P1** — Hand strength (Treys MC) sanity: AA preflop > 0.8 equity; 72o < 0.4
- [ ] **P2** — Squeeze detection from hand sequence
- [ ] **P2** — Fold-to-3-bet-after-raising
- [ ] **P2** — `_string_to_action_enum` snapshot of every action source
- [ ] **P2** — Reset stacks happens only between hands across N timesteps

---

## Part 2 — Training Loop Rewrite (Target Architecture)

### 2.1 End state

- `Player` class **deleted**. Its data lives on `Agent.seat` as a `SeatState` (composition).
- `OpponentAutoPlayWrapper` **deleted**. Auto-play loop moves inside `env.step`.
- `learning_agent_id = 0` **no longer hardcoded**. Configurable on env construction; replaceable per `reset()` (enables seat rotation).
- `TexasHoldemEnv` **operates on agents**, not players. Constructor takes `List[Agent]` and `learning_agent_id`.
- Each `Agent` owns its own `OpponentMemory`. Events are broadcast to every seated agent; each maintains its own profiles.
- `OpponentTracker` (global, shared) **deleted** in its current form. The per-agent `OpponentMemory` replaces it.

### 2.2 New class shapes (sketch)

```python
@dataclass
class SeatState:
    stack: int
    starting_stack_this_hand: int
    hand: list[int]
    current_bet: int
    total_bet_this_hand: int
    is_active: bool
    is_all_in: bool
    is_sitting_out: bool
    total_winnings: int
    total_buy_in: int

    def reset_for_new_hand(self):
        self.hand = []
        self.current_bet = 0
        self.total_bet_this_hand = 0
        self.is_active = not self.is_sitting_out
        self.is_all_in = False
        self.starting_stack_this_hand = self.stack


@dataclass
class ActionEvent:
    actor_id: int
    street: Street
    action: Action
    amount: int
    pot_before: int
    stack_before: int
    stack_after: int
    position_relative_to_button: int
    board_at_action: tuple


class OpponentMemory:
    """One per agent. Subscribes to env events."""
    profiles: Dict[int, OpponentProfile]
    event_log: Dict[int, deque[ActionEvent]]  # raw, for future learned encoder

    def on_action(self, event: ActionEvent): ...
    def on_hand_end(self, result: HandResult): ...
    def get_features(self, opponent_id: int) -> list[float]: ...


class Agent:
    player_id: int
    name: str
    seat: SeatState
    memory: OpponentMemory

    def select_action(self, obs, valid_actions) -> int: ...
    def on_action(self, event: ActionEvent): self.memory.on_action(event)
    def on_hand_end(self, result): self.memory.on_hand_end(result)


class TexasHoldemEnv(gym.Env):
    def __init__(self, agents: list[Agent], learning_agent_id: int, ...):
        self.agents = agents
        self.learning_agent_id = learning_agent_id
        self.game_state = GameState(seats=[a.seat for a in agents], ...)

    def step(self, action):
        learner = self.agents[self.learning_agent_id]
        obs, reward, done, info = self._execute(learner, action)
        while not done and self._current_agent_id() != self.learning_agent_id:
            agent = self.agents[self._current_agent_id()]
            opp_action = agent.select_action(
                self._observation_for(agent),
                self._valid_actions()
            )
            obs, _, done, info = self._execute(agent, opp_action)
        return obs, reward, done, False, info

    def _execute(self, agent, action):
        # mutate game state, build event, broadcast to all agents
        event = ...
        for a in self.agents:
            a.on_action(event)
        ...
```

### 2.3 Construction (new API)

**Before:**
```python
env = TexasHoldemEnv(num_players=3, track_opponents=True, ...)
opponents = [('call', CallAgent()), ('rand', RandomAgent())]
wrapped = OpponentAutoPlayWrapper(env, opponents)
agent.model.set_env(wrapped)
```

**After:**
```python
agents = [
    PPOAgent(player_id=0, name='learner'),
    OpponentPPO(player_id=1, name='gen_2', path='models/gen_2/final_model.zip'),
    OpponentPPO(player_id=2, name='gen_3', path='models/gen_3/final_model.zip'),
]
env = TexasHoldemEnv(agents=agents, learning_agent_id=0, ...)
agent.model.set_env(env)
```

### 2.4 What this unlocks

- **Seat rotation between hands.** Shuffle agent-to-`player_id` assignment in `env.reset()`. The PPO learner sees every position.
- **Opponent pool sampling for self-play.** Each seat just gets an `Agent`; PFSP / fictitious self-play falls out naturally.
- **Multi-PPO training.** Multiple agents at the table can all learn (separate `learn()` calls or shared policy).
- **Information asymmetry (future).** When you want it, modify `_execute` to broadcast only to agents still in the hand.
- **Cleaner backend.** A `HumanAgent` in any seat is no different from a `PPOAgent` in any seat.

---

## Part 3 — Migration Sequence

Each step independent. Tests stay green between steps. If a step takes >½ day, split it.

- [ ] **Step 0 — Reproducibility scaffold.** Write `metrics/<run>/run_manifest.json` containing config, opponent paths + SHA256, git commit, SB3/torch versions, seed. Add `--seed` to all entry scripts; propagate to numpy/random/torch/SB3/env. **This unblocks every diagnosis that follows.**
- [ ] **Step 1 — Fix opponent tracker P0 bugs** (1.1: 3-bet, c-bet, fold-to-c-bet, WTSD). Write tests against desired behavior first; they should fail; then fix until they pass. Do this *before* the refactor so corrections port directly.
- [ ] **Step 2 — Validate opponent observation space at load time.** In `create_opponents`, refuse to use mismatched obs shapes. Surfaces silent mis-mapping.
- [ ] **Step 3 — Fix `train_from_checkpoint.py` resume semantics.** Use `PPO.load(path, env=wrapped, custom_objects={...})` to actually apply YAML overrides.
- [ ] **Step 4 — Introduce `SeatState`.** Thin wrapper around `Player`'s existing fields. Make `Player` a deprecated alias that delegates. Tests still pass.
- [ ] **Step 5 — Introduce `Agent.seat: SeatState`.** Game logic still reads `player.stack` etc., but those reads start migrating to `agent.seat.stack`.
- [ ] **Step 6 — Parameterize `learning_agent_id`.** Replace wrapper index math with `Dict[player_id, Agent]`. Delete the wrapper's reward override. Wrapper still exists, just thinner.
- [ ] **Step 7 — Move auto-play loop into env, delete `OpponentAutoPlayWrapper`.**
- [ ] **Step 8 — Introduce `OpponentMemory` per agent.** Env broadcasts `ActionEvent`/`HandResult`; each agent's memory consumes. Old `OpponentTracker` becomes a thin shim that aggregates from the learning agent's memory (for the dashboard) and then is deleted.
- [ ] **Step 9 — Delete `Player`** and any shim code.
- [ ] **Step 10 — Add seat rotation** in `env.reset()`. The learner now sees every position.
- [ ] **Step 11 — Build evaluation harness.** N hands of `gen_N+1` vs `gen_N` (deterministic + stochastic), report bb/100 with confidence interval, gate promotion of "best".
- [ ] **Step 12 — Replace mtime opponent selection with explicit manifest + pool sampling.** `configs/opponents.yaml` lists paths + sample weights; `train.py` reads it.
- [ ] **Step 13 — Fix remaining P1 callback bugs** (1.3: `n_calls` → `num_timesteps`, action bucket from `action_space.n`, rename win_rate or compute from info, wire profit_tracker to all entry scripts).

---

## Part 4 — Open Questions

- [ ] How big should `OpponentMemory.event_log` deque be? (Suggest 10k events.)
- [ ] When the env broadcasts events, should agents that folded see subsequent events in the hand? (Phase 1: yes, full information. Phase 2: enforce asymmetry.)
- [ ] Should `OpponentMemory` for opponents trained with old observation spaces be reconstructed from scratch when they're loaded as opponents? (Probably yes — saved policy weights don't include memory state.)
- [ ] Per-opponent profit attribution: keep equal-split, replace with last-aggressor heuristic, or compute from side-pot contestants?
- [ ] Eval harness location — new entry script `tournament.py` vs callback inside `train.py`?
- [ ] Should `learning_agent_id` be a list (multi-PPO) from day one, or scalar with a future migration?

---

## Part 5 — Out of scope for this rewrite (track separately)

- SkyPilot cloud training migration (see `audit_2026-05-10.md` §6)
- `SubprocVecEnv` rollout parallelism (separate refactor)
- Hydra/Pydantic config migration
- W&B / MLflow experiment tracking
- Learned opponent encoder (RNN/Transformer over `event_log`) — future work, kept feasible by Step 8's `event_log` storage
