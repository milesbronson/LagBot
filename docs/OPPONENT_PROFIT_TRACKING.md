# Opponent Profit Tracking Guide

Track which opponents your poker bot is making the most money from!

## What It Does

The opponent profit tracker records:
- **Per-opponent profit/loss** - How much the bot makes against each opponent
- **Hands played** - Number of hands against each opponent
- **Win rate** - Percentage of hands won against each opponent
- **Average profit per hand** - Efficiency of exploitation
- **Historical trends** - Profit evolution over training

## How to Use

### 1. Training (Automatic)

Opponent profit tracking is **automatically enabled** for all training runs. No changes needed!

```bash
python3 train.py --config configs/deep_architecture_3M.yaml --name my_run
```

Data is saved to: `metrics/my_run/opponent_profits.json`

### 2. View Results During Training

The callback prints a summary every 10k timesteps:

```
================================================================================
OPPONENT PROFIT ANALYSIS
================================================================================

Opponent             Type            Hands    Total $       Avg $/Hand    Win Rate
--------------------------------------------------------------------------------------
CallAgent            call            1234     +1.2345       +0.0010       78.5%
RandomAgent          random          1234     +0.8765       +0.0007       72.3%

================================================================================
🏆 Best Matchup:  CallAgent (call) → +1.2345 profit
⚠️  Worst Matchup: RandomAgent (random) → +0.8765 profit
================================================================================
```

### 3. Analyze After Training

**Text summary:**
```bash
python3 analyze_opponent_profits.py my_run
```

**Graphical analysis:**
```bash
# Show plots
python3 analyze_opponent_profits.py my_run --plot

# Save plots to file
python3 analyze_opponent_profits.py my_run --plot --save opponent_analysis.png
```

## What the Graphs Show

### Graph 1: Cumulative Profit Over Time
- Shows how profit vs each opponent evolves during training
- Upward slope = exploiting that opponent more over time
- Flat line = consistent performance
- Downward slope = opponent is adapting or bot struggling

### Graph 2: Total Profit Per Opponent
- Bar chart showing final profit against each opponent
- Green bars = profitable matchups (making money)
- Red bars = unprofitable matchups (losing money)
- Height = total profit magnitude

## Interpreting Results

### Positive Profit Against All Opponents ✅
```
CallAgent    → +1.23 profit (Exploiting well!)
RandomAgent  → +0.87 profit (Also exploiting)
```
**Interpretation:** Bot is beating both opponents. Higher profit = stronger exploitation.

### Mixed Results ⚠️
```
CallAgent    → +2.45 profit (Crushing it!)
ppo_gen_1    → -0.32 profit (Struggling)
```
**Interpretation:** Bot exploits weak opponents but struggles against stronger ones.

### Profit Variance Analysis

**High variance** (e.g., +2.0 vs -0.5):
- Bot has exploitable strategies that work against some opponents
- May indicate over-specialization
- Could struggle against novel opponent types

**Low variance** (e.g., +0.5 vs +0.4):
- Consistent performance across opponents
- More robust, generalizable strategy
- Less exploitation but more reliable

## Example Use Cases

### 1. Find Weakest Opponent
```bash
python3 analyze_opponent_profits.py my_run
```
Look for the opponent with highest profit - that's who the bot exploits best!

### 2. Track Improvement Over Time
```bash
python3 analyze_opponent_profits.py my_run --plot
```
Compare early vs late training slopes to see if bot learns to exploit better.

### 3. Compare Generations
```bash
python3 analyze_opponent_profits.py gen_1
python3 analyze_opponent_profits.py gen_2
```
See if newer generation exploits opponents better than previous.

## Data Format

Data is saved as JSON in `metrics/<run_name>/opponent_profits.json`:

```json
{
  "opponent_results": {
    "1": {
      "name": "CallAgent",
      "type": "call",
      "hands_played": 1234,
      "total_profit": 1.2345,
      "avg_profit": 0.0010,
      "win_count": 969,
      "loss_count": 265,
      "win_rate": 0.785
    },
    "2": { ... }
  },
  "history": {
    "timesteps": [10000, 20000, 30000, ...],
    "opponent_profits": [
      {"1": 0.123, "2": 0.098},
      {"1": 0.245, "2": 0.187},
      ...
    ]
  }
}
```

## Technical Details

### How Profit is Calculated

In a 3-player game (bot vs 2 opponents):

1. **Total profit** = (ending_stack - starting_stack) / starting_stack
2. **Per-opponent profit** = total_profit / 2 (split equally)

This is a simple approximation. A more sophisticated version could track:
- Which opponent folded first (attribute more profit)
- Showdown winners (attribute based on pot contribution)
- Position-based attribution

### When Profits Are Recorded

- After each hand completes (terminated or truncated)
- Checkpointed every 10,000 timesteps
- Final summary at end of training

### Multi-Opponent Attribution

Currently, profit is split equally among opponents. Future versions could:
- Weight by opponent's contribution to pot
- Track head-to-head profit (1v1 situations)
- Adjust for position effects

## Troubleshooting

### "No opponent profit data found"

Make sure:
1. You're using the latest training code with profit tracking
2. Training has run for at least one hand
3. The run name matches the directory in `metrics/`

### Graphs look flat

- May need more training time for trends to emerge
- Check if bot is actually making profit (could be near zero)
- Verify opponents are different types (not all the same)

### Negative profits against all opponents

- Bot is losing! This could indicate:
  - Early in training (hasn't learned yet)
  - Bug in reward calculation
  - Opponents are too strong
  - Configuration issue (e.g., very high rake)

---

## Next Steps

After analyzing opponent profits:

1. **Focus training** on weak matchups (negative profit)
2. **Generate diverse opponents** to avoid over-specialization
3. **Compare generations** to track improvement
4. **Adjust architecture** if exploitation is poor

**Happy hunting for exploitable opponents! 🎯**
