# Bankroll Tracking Feature

## Goal
Track total money won/lost for each bot across all hands, like a real poker player's bankroll.

## Implementation Plan

### 1. Add to TexasHoldemEnv (`texas_holdem_env.py`)

In `__init__`:
```python
# Bankroll tracking (total profit/loss)
self.cumulative_bankroll = {}  # {player_id: total_profit}
for player in self.game_state.players:
    self.cumulative_bankroll[player.player_id] = 0
```

In `step()` when hand completes:
```python
if done:
    winnings = self.game_state.determine_winners()

    # Update cumulative bankroll for all players
    for player in self.game_state.players:
        starting = player.starting_stack_this_hand
        ending = player.stack
        profit = ending - starting
        self.cumulative_bankroll[player.player_id] += profit

    # Terminal reward calculation (existing code)
    learning_agent = self.game_state.players[self.learning_agent_id]
    ...
```

In `reset()`:
```python
# Don't reset cumulative_bankroll - it tracks across all hands!
# Only reset it if you want to start tracking from zero
```

### 2. Add to TensorBoard Logging

In `src/training/callbacks.py` (MetricsCallback or create new callback):
```python
class BankrollCallback(BaseCallback):
    def __init__(self, log_freq=1000):
        super().__init__()
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            # Get environment (handle VecEnv wrapper)
            env = self.training_env.envs[0].env

            # Get cumulative bankroll
            bankroll = env.cumulative_bankroll[0]
            big_blind = env.game_state.big_blind

            # Log to TensorBoard
            self.logger.record("agent/cumulative_bankroll", bankroll)
            self.logger.record("agent/bankroll_in_bb", bankroll / big_blind)

            # Calculate BB/100 hands
            hands_played = env.game_state.hand_number
            if hands_played > 0:
                bb_per_100 = (bankroll / big_blind) / (hands_played / 100)
                self.logger.record("agent/bb_per_100_hands", bb_per_100)

        return True
```

### 3. Add to Console Logs

In the agent stats print section (around line 9000 in training output):
```python
------------------------------------------
| agent/                  |              |
|    all_in_rate          | 0.142        |
|    avg_reward           | -0.212       |
|    cumulative_bankroll  | -2145        |  # NEW
|    bankroll_bb          | -214.5       |  # NEW
|    bb_per_100           | -8.2         |  # NEW
|    call_rate            | 0.202        |
|    episodes_completed   | 2627         |
|    fold_rate            | 0.159        |
|    max_reward           | 8.61         |
|    min_reward           | -21.4        |
|    raise_rate           | 0.497        |
|    win_rate             | 0.375        |
------------------------------------------
```

### 4. Integrate into Training Script

In `train.py`, add the callback:
```python
from src.training.callbacks import MetricsCallback, BankrollCallback

# Create callbacks
metrics_callback = MetricsCallback(metrics=metrics, log_freq=10000)
bankroll_callback = BankrollCallback(log_freq=1000)

# Train with callbacks
agent.model.learn(
    total_timesteps=training_config['total_timesteps'],
    callback=[save_callback, metrics_callback, bankroll_callback]
)
```

### 5. Example Output

**Console Logs:**
```
------------------------------------------
| agent/                  |              |
|    avg_reward           | -0.212       |
|    cumulative_bankroll  | -2,145       |  # Total $ lost
|    bankroll_bb          | -214.5       |  # In big blinds
|    bb_per_100           | -8.2         |  # Losing 8.2 BB per 100 hands
|    win_rate             | 0.375        |
|    episodes_completed   | 2627         |
------------------------------------------
```

**TensorBoard Charts:**
You'll see three new scalar graphs under "agent/":
- `agent/cumulative_bankroll` - Line going up (profit) or down (loss)
- `agent/bankroll_in_bb` - Same but in big blinds
- `agent/bb_per_100_hands` - Standard poker win rate metric

Example TensorBoard view:
```
agent/cumulative_bankroll    |     -2,145 ▼
agent/bankroll_bb            |    -214.5 ▼
agent/bb_per_100_hands       |      -8.2 ▼
agent/win_rate               |      0.375 →
```

## Why This is Useful

1. **Intuitive**: "Made $5,000" is clearer than "avg reward 0.05"
2. **Real poker metric**: Big blinds won per 100 hands (BB/100)
3. **Long-term trend**: Shows if agent is profitable overall
4. **Easy debugging**: If bankroll crashes, something is wrong

## Calculation

- **Cumulative Bankroll** = Sum of (ending_stack - starting_stack) for all hands
- **BB/100 hands** = (cumulative_bankroll / big_blind) / (hands_played / 100)
- Positive = profitable, Negative = losing money

## Reset Behavior

- **Don't reset** cumulative_bankroll between hands (that's the point!)
- **Only reset** when:
  - Starting completely new training run
  - Want to track from a specific checkpoint
  - Testing specific scenarios

## Notes

- This is separate from the reward signal (which is normalized)
- Reward guides learning, bankroll tracks human-readable profit
- Can track for all players, not just learning agent
- Useful for comparing different training runs

## Files to Modify (When Implementing)

1. `src/poker_env/texas_holdem_env.py` - Add tracking
2. `src/agents/ppo_agent.py` or callbacks - Add logging
3. `src/training/callbacks.py` - Display in training output
