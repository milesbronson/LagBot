# Refactor Design: Unified Agent + Player Analytics

**Date:** 2026-05-10
**Status:** Design doc — not yet implemented.
**Predecessor:** `working_docs/player_agent_architecture_2026-05-10.md` (which sketched Options A and B). This doc commits to **Option A** and extends it to cover opponent analytics.

---

## 1. Goal

Make one object the **single source of truth** for everything about a participant at the table:

- Identity (name, id)
- Decision logic (policy / behavior)
- Seat state (stack, hand, current bet, all-in status)
- Observed behavior of others (what *this* agent has seen *other* agents do)

The current architecture spreads these across three places — `Player` (in the env), `BaseAgent` subclasses (in `src/agents/`), and `OpponentTracker` (one shared instance in the env). They're glued together by list index. That gluing is the root cause of: the wrapper index math, the hardcoded `learning_agent_id = 0`, the redundant reward override, the inability to rotate seats, and (less obviously) the way opponent tracking is structurally coupled to "there is exactly one learning agent and it sits at index 0."

---

## 2. The new shape

### 2.1 `Agent` is the persistent owner

```python
class Agent:
    # Identity
    player_id: int
    name: str

    # Decision logic — depends on subclass
    # PPOAgent has self.model; CallAgent / RandomAgent have policies; HumanAgent waits for input
    def select_action(self, obs, valid_actions) -> int: ...

    # Per-hand mutable state — lives here so there's one owner
    seat: SeatState

    # What this agent has observed about others
    memory: OpponentMemory

    # Lifecycle
    def reset_for_new_hand(self): self.seat.reset()
    def on_action_observed(self, event: ActionEvent): self.memory.record(event)
    def on_hand_ended(self, result: HandResult): self.memory.finalize_hand(result)
```

### 2.2 `SeatState` — per-hand mutable state

Pulled out of the old `Player` class. Lives as a composed field on `Agent` rather than as a peer object. Reset every hand; the `Agent` (and its policy weights, and its `memory`) is untouched.

```python
@dataclass
class SeatState:
    stack: int
    starting_stack_this_hand: int
    hand: list[int]                  # hole cards (Treys ints)
    current_bet: int                 # this betting round
    total_bet_this_hand: int
    is_active: bool                  # still in the hand
    is_all_in: bool
    is_sitting_out: bool
    total_winnings: int              # session-level, not reset per hand
    total_buy_in: int                # session-level, not reset per hand

    def reset_for_new_hand(self):
        self.hand = []
        self.current_bet = 0
        self.total_bet_this_hand = 0
        self.is_active = not self.is_sitting_out
        self.is_all_in = False
        self.starting_stack_this_hand = self.stack
```

Everything `Player` did is here; everything `Player` was missing (link to behavior) is solved by virtue of `SeatState` being owned by `Agent`.

### 2.3 The env operates on agents, not players

```python
class TexasHoldemEnv(gym.Env):
    def __init__(self, agents: list[Agent], learning_agent_id: int, ...):
        self.agents = agents
        self.learning_agent_id = learning_agent_id   # configurable, not hardcoded 0
        self.game_state = GameState(seats=[a.seat for a in agents], ...)

    def step(self, action):
        # Only called when current_player is the learning agent
        learning_agent = self.agents[self.learning_agent_id]
        self._execute(learning_agent, action)

        # Auto-play non-learning agents until it's the learning agent's turn again
        while not self._is_hand_complete() and self._current_agent() is not learning_agent:
            agent = self._current_agent()
            opp_action = agent.select_action(self._observation_for(agent), self._valid_actions())
            self._execute(agent, opp_action)

        return self._observation_for(learning_agent), reward, done, truncated, info
```

The wrapper goes away. The "auto-play opponents" loop moves into the env, but with no index math — it just asks "whose turn is it?" and dispatches.

### 2.4 What dies in this refactor

- `OpponentAutoPlayWrapper` in `train.py` — deleted
- The reward override in the wrapper — deleted (env already does it right)
- `self.learning_agent_id = 0` hardcoded — replaced with constructor arg
- `opponent_idx = current_idx - 1` math — replaced with `player_id` lookups
- `Player` class — folded into `SeatState`
- The implicit "agent always in seat 0" assumption everywhere

---

## 3. Opponent analytics — centralized tracker, query interface on the agent

**REVISED 2026-05-10.** The earlier draft of this section recommended per-agent `OpponentMemory`. On reflection that was over-engineering. The corrected design below keeps a single centralized tracker (matching the current "perfect-information record keeper" design) but moves the *query interface* onto the agent. The tracker stays as the single source of truth.

### 3.1 Why centralized, not per-agent

The arguments I gave earlier for per-agent memory (information asymmetry, multi-PPO, theoretical purity) are real, but they're future concerns. The current feature set (VPIP, PFR, AF, 3-bet %, c-bet %, fold-to-c-bet %, WTSD %, confidence) is a set of **aggregate properties of the player** — not observer-relative beliefs. There is nothing asymmetric about them. One shared computation produces them; every consumer reads the same numbers.

Benefits of keeping it centralized:

- One source of truth. One place to debug, one place to test, one place to read events.
- Cheap and simple. Events arrive, counters update, queries return derived stats.
- The bugs are localized in `_update_opponent_stats`. Centralization isn't what's wrong; the logic is.
- The decoupling that DOES matter (separating tracking from game logic) is achievable without changing the tracker's centralization.

### 3.2 The actual change: decouple via event hooks, put query on Agent

```python
# Env emits events; tracker subscribes. Env doesn't import tracker types.
class TexasHoldemEnv(gym.Env):
    def __init__(self, ..., listeners: list[EnvEventListener] = ()):
        self._listeners = list(listeners)

    def register_listener(self, listener): self._listeners.append(listener)

    def _emit(self, event):
        for l in self._listeners:
            l.on_event(event)


class OpponentTracker(EnvEventListener):
    """Centralized, single source of truth. Subscribes to env events."""
    def on_event(self, event):
        if isinstance(event, HandStartEvent):  self._on_hand_start(event)
        elif isinstance(event, ActionEvent):   self._on_action(event)
        elif isinstance(event, HandEndEvent):  self._on_hand_end(event)

    def get_features(self, observer_id: int, opponent_id: int) -> list[float]:
        # observer_id is currently ignored — full information.
        # Hook is preserved so asymmetry can be added later without
        # changing callers.
        ...


class Agent:
    def __init__(self, ..., tracker: OpponentTracker):
        self._tracker = tracker   # injected, not owned

    def get_opponent_features(self, opponent_id: int) -> list[float]:
        return self._tracker.get_features(observer_id=self.player_id,
                                          opponent_id=opponent_id)
```

What changes structurally:

- Env no longer calls `self.opponent_tracker.record_action(...)` directly inside game-logic methods. Instead, env emits an `ActionEvent`; whoever subscribes (the tracker, the backend, future loggers) consumes it.
- Tracker stops being imported from `texas_holdem_env.py`. The game logic doesn't know what a tracker is.
- Agent has a `get_opponent_features(opponent_id)` interface that delegates to the injected tracker. Callers ask the agent, not the tracker.
- `observer_id` parameter on `tracker.get_features` is a forward-compatibility hook. Today it's ignored — every agent sees the same view. Tomorrow it can implement asymmetry without changing any caller.

The fix-the-bugs work (3-bet, c-bet, fold-to-c-bet, WTSD, position) is independent of this decoupling and happens first. Tests for those bugs are written against the tracker's public methods, which remain stable.

### 3.3 What data we should track

What we have today (after bug fixes):

| Stat | Meaning |
|---|---|
| VPIP | % hands voluntarily put money in pot |
| PFR | % hands raised preflop |
| AF | Aggression factor: (bets+raises)/calls |
| 3-bet % | When facing a raise, how often re-raise |
| C-bet % | When preflop raiser, how often bet flop |
| Fold-to-C-bet % | When facing a c-bet, how often fold |
| WTSD % | How often reach showdown |
| Confidence | min(hands_played / 100, 1.0) — a "how much do I trust this" weight |

Industry-standard adds worth considering (in rough order of value):

| Stat | Meaning | Why useful |
|---|---|---|
| W$SD | Won money at showdown | Distinguishes showdown winners from showdown losers |
| WWSF | Won when saw flop | Captures post-flop pressure |
| Turn c-bet % / River c-bet % | Aggression on later streets | Catches one-and-done c-bettors |
| Fold-to-turn-cbet %, fold-to-river-cbet % | | Captures who folds to barrels |
| Steal % | Raise from CO/BTN/SB when folded to | Captures positional aggression |
| Fold-to-steal % | When in BB and faced steal, fold % | Captures who defends blinds |
| Check-raise % | | A specific, exploitable behavior |
| Avg bet size / pot | By street | Bet sizing tells |
| Squeeze % | Raise after raise + caller | Captures pre-flop sophistication |

Better-structured options:

- **Per-street aggression matrix**: `{street: {bet_freq, raise_freq, call_freq, fold_freq, check_freq}}`. Replaces several of the above with one structured object.
- **Position-bucketed stats**: VPIP/PFR/AF broken out by position bucket (EP / MP / LP / SB / BB). Only meaningful once position is recorded correctly.
- **Stack-depth-bucketed stats**: short / medium / deep. Players behave very differently at 20bb vs 100bb.

### 3.4 The "high-dimensional tensor" idea

You're gesturing at something real, and worth designing the interface to allow.

The standard ML version of "rich opponent representation": maintain a **learned embedding per opponent** that gets updated by an encoder (RNN or Transformer) over the sequence of observed actions. The policy then conditions on `[hand_features, board_features, opponent_embedding]`.

Two reasons to design for this *now* even if you don't build it yet:

1. **The interface stays stable.** Today: `agent.memory.get_features(opponent_id) -> List[float]` returns 8 hand-crafted floats. Tomorrow: same call returns a 32-dim learned embedding. The policy doesn't care — the observation space stays the same shape.
2. **The action history is the input.** Both implementations are fed by the same `ActionEvent` stream. Don't throw away raw events when computing summarized stats — keep them in `OpponentMemory.event_history` so a future encoder can use them.

So my concrete advice: **keep hand-crafted stats for now (fixed and extended), but store raw action events alongside them so you can upgrade to learned embeddings later without rearchitecting.**

```python
class OpponentTracker(EnvEventListener):
    profiles: Dict[int, OpponentProfile]                    # summarized
    event_log: Dict[int, deque[ActionEvent]]                # raw, for future encoder

    def get_features(self, observer_id: int, opponent_id: int) -> list[float]:
        # Current implementation: read from profiles[opponent_id]
        # Future implementation: pass event_log[opponent_id] through an encoder
        ...
```

### 3.5 Anti-patterns to avoid

- **Don't put opponent features on `SeatState`.** Opponent stats are persistent across hands; seat state resets every hand. Different lifecycles → different objects.
- **Don't let env game logic import tracker types.** Use event hooks. The env emits, the tracker consumes. Decoupled lifecycles.
- **Don't compute features eagerly on every event.** Recompute lazily when the query is made. Today's tracker does some of both; pick one.
- **Don't return the same feature vector to every observer permanently.** Keep the `observer_id` parameter on `get_features` even though it's currently ignored — it's the migration hook for asymmetry later.
- **Don't tie the feature vector to a specific shape forever.** Wrap it in a method that returns a fixed-size array; let the implementation evolve.

---

## 4. Migration path

Strongly recommend this order. Step 1 is independent of the refactor and is blocking learning right now.

1. **Fix the P0 opponent-tracker bugs** in the *current* architecture. (See `working_docs/audit_2026-05-10.md` §2 — 3-bet, c-bet, fold-to-c-bet, WTSD.) Write the tests against desired behavior. Do this first because:
   - Tests written now port directly into the new structure
   - The model's input features stop being garbage
   - The refactor doesn't have to "fix bugs while moving code" (always painful)

2. **Introduce `SeatState`** as a thin wrapper around `Player`'s existing fields. Make `Player` a deprecated alias that delegates. Tests still pass.

3. **Introduce `Agent.seat`** on `BaseAgent`. Game logic still reads/writes `player.stack` etc. Eventually those reads/writes become `agent.seat.stack`.

4. **Parameterize `learning_agent_id`** in the env. Replace the wrapper's index math with `Dict[player_id, Agent]` lookups. Delete the wrapper's reward override.

5. **Move the auto-play loop into the env**, delete `OpponentAutoPlayWrapper`.

6. **Introduce `OpponentMemory`** per agent. Move the existing `OpponentTracker` logic onto it. Env broadcasts events; each agent's memory consumes them.

7. **Delete `Player` and the shared `OpponentTracker`** once nothing references them.

8. **Now you unlock**: seat rotation in `reset()`, real opponent pool sampling, multi-PPO training, learned embeddings later.

Each step keeps the test suite green. If a step takes more than half a day, you've probably bundled two steps together — split it.

---

## 5. Open questions

1. **How big is `event_log`?** Per-opponent deque of action events. At ~10 actions/hand × 10k hands = 100k events. Cheap. Set a maxlen (~10k events) to bound memory.
2. **Where do events come from?** Env emits `ActionEvent(actor_id, street, action_type, amount, pot_before, board_at_action)` on every action. Every seated agent gets a callback. Hand-end emits `HandResult(winners, winnings, showdown_hands_visible_to_observer)`.
3. **Showdown visibility.** When an agent doesn't reach showdown, do they see anyone's hole cards? In real poker: only if cards are tabled. The env should respect this — pass `revealed_to=[agent_ids_at_showdown]` so memories of folded agents don't get to peek.
4. **What about the backend?** `GameSession` will need updates. A human player becomes a `HumanAgent` instance; the WebSocket handler dispatches actions to the human's `select_action`-equivalent (probably a future/promise pattern since the human is async).
5. **Backwards compatibility with saved models?** PPO checkpoints save policy weights, not env structure. As long as the observation shape stays 125 dims, old models still load. Worth verifying with a smoke test before deleting old code paths.

---

## 6. Summary recommendation

- **Do the refactor.** The current Player/Agent split is the root cause of multiple bugs and blocks several real training improvements (seat rotation, pool sampling, multi-PPO).
- **Use Option A with composition**: `Agent` owns `SeatState` and `OpponentMemory`. Not bidirectional references — Agent is the root.
- **Fold opponent analytics into the refactor** the way you described: per-agent memory, not a shared tracker. This makes information asymmetry explicit and supports multi-agent training later.
- **Extend the stat set** modestly (W$SD, WWSF, per-street c-bet, steal %), but keep the interface (`get_features(opponent_id) -> fixed-size vector`) stable so you can swap in a learned encoder later without touching the policy.
- **Store raw action events** alongside summarized stats. They're cheap and they're the input to any future learned representation.
- **Sequence**: tracker bug fixes first (independent, blocking), then the refactor (half-day to two days), then leverage it (seat rotation, pool sampling, tournament eval).
