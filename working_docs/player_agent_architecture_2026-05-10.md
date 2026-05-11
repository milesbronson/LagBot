# Player / Agent Architecture Cleanup

**Date:** 2026-05-10
**Context:** Discussion about why the reward wrapper in `train.py` exists, the disjoint `Player` / `Agent` classes, and the hardcoded "learning agent = player 0" assumption.

---

## What you're seeing is real

The `Player` class and the `Agent` classes are two completely disjoint hierarchies, glued together by list index in the wrapper.

### `Player` — seat state

`src/poker_env/player.py` `Player` holds seat state:

- `player_id`
- `stack`
- `hand`
- `current_bet`
- `total_bet_this_hand`
- `starting_stack_this_hand`
- `is_active`, `is_all_in`, `is_sitting_out`

No reference to which agent is making the decisions for this seat.

### `Agent` — decision logic

`src/agents/base_agent.py` `BaseAgent` (and `PPOAgent`, `OpponentPPO`, `RandomAgent`, `CallAgent`, `HumanAgent`) holds decision logic:

- `name`
- `hands_played`
- `select_action(obs, valid_actions)`

No `player_id`, no reference back to a `Player`.

### The glue is list index

The two are connected only by index in `OpponentAutoPlayWrapper` — `train.py:115-120`:

```python
current_idx = self.env.game_state.current_player_idx
opponent_idx = current_idx - 1
opponent_type, opponent = self.opponents[opponent_idx]
```

So the architecture hard-assumes:

- Player 0 = learning agent
- Player 1 = `opponents[0]`
- Player 2 = `opponents[1]`

That assumption shows up in three places:

1. The wrapper index math (`current_idx - 1`)
2. The env's `self.learning_agent_id = 0` (`texas_holdem_env.py:100`)
3. The reward override using `players[0]` (`train.py:103-106, 129-133`)

---

## The wrapper reward override is now redundant

The history you're describing is real — but the env was later patched to do the right thing too.

`texas_holdem_env.py:240-245`:

```python
if done:
    winnings = self.game_state.determine_winners()
    learning_agent = self.game_state.players[self.learning_agent_id]
    agent_starting_stack = learning_agent.starting_stack_this_hand
    terminal_reward = (learning_agent.stack - agent_starting_stack) / self.starting_stack
    reward += terminal_reward
```

The env **already** attributes the terminal reward to the learning agent regardless of whose action ended the hand. So the wrapper's reward override block —

```python
if terminated or truncated:
    learning_agent = self.env.game_state.players[0]
    agent_starting_stack = learning_agent.starting_stack_this_hand
    reward = (learning_agent.stack - agent_starting_stack) / self.env.starting_stack
```

— is **recomputing the same number and overwriting it**. As a side effect it nukes the +0.1 / −0.1 fold-shaping when the agent's own fold ends the hand (because env adds shaping + terminal, wrapper replaces it with just terminal).

The wrapper was the original fix for the "wrong agent's reward came out" bug. The env-level fix made it redundant. Nobody deleted the wrapper code. Worth knowing as you clean up — you can probably delete that override entirely.

---

## What a clean architecture would look like

Sketch — not a plan, not a commitment to a specific direction.

1. **`Player` gets an `agent` field** (or `agent_id`). `BaseAgent` gets a `player_id` field, set when seated.
2. **`TexasHoldemEnv.__init__` takes a `learning_agent_id`** (not hardcoded to 0) and a `Dict[player_id, BaseAgent]` for the non-learning seats.
3. **`env.step(action)` only runs when it's the learning agent's turn**; internally the env auto-plays whichever seats have non-learning agents attached, looked up by `player_id`, not list index.
4. **The wrapper either disappears entirely** or becomes a thin Gym adapter (no reward arithmetic, no index math).

### Benefits

- PPO agent can sit in any seat
- CallAgent can be on the button while PPO is UTG
- Seat assignments can rotate between hands (unbiased training)
- The whole class of index-arithmetic bugs goes away
- The wrapper stops being load-bearing

### Tradeoff

This is invasive. `learning_agent_id = 0` is referenced in:

- The env (`learning_agent_id`, the reward attribution, the observation construction in `_get_observation`)
- The wrapper (index math, reward override, opponent profit tracking)
- The backend (`backend/services/game_session.py` almost certainly assumes the human is in a specific seat)
- The metrics callback
- Possibly the play CLI (`play.py`)

Probably a half-day of careful editing with tests as a safety net.

---

## Two paths

### Option A — Full marriage of `Player` and `Agent`

- `Player.agent: BaseAgent` field
- `BaseAgent.player_id: int` field
- Env owns the `Player[]` list, and each `Player` knows its decision-maker
- `env.step()` either takes the learning agent's action, or runs the env to the next learning-agent turn auto-playing seated non-learning agents along the way
- Clean, elegant, future-proof

### Option B — Lower-risk: kill index math and the player-0 assumption

- Keep `Player` and `Agent` as disjoint classes
- Parameterize `learning_agent_id` everywhere (no more hardcoded `= 0`)
- Replace index math (`opponent_idx = current_idx - 1`) with `Dict[player_id, Agent]` lookups in the wrapper
- Delete the redundant reward override in the wrapper
- Most of the architectural cleanup, none of the class hierarchy restructuring

---

## Open question

Which direction do you want to go — full marriage of `Player` and `Agent` (Option A), or just kill the index math and the `player_0` assumption (Option B)?
