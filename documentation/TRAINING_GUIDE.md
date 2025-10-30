# Training & Dashboard Guide

## Quick Start

```bash
# 1. Start training
python train_with_metrics.py --name first_bot --config configs/default_config.yaml

# 2. In another terminal, monitor with Tensorboard
tensorboard --logdir logs/first_bot/

# 3. Generate dashboards (during or after training)
python dashboard_gen.py --all
```

## What Gets Tracked

Every 10,000 timesteps:
- **Rewards**: Episode reward + 100-episode average
- **Actions**: Fold rate, raise rate, all-in rate
- **Learning**: Loss curves, policy performance
- **Win Rate**: How often agent wins hands

Data saved to: `metrics/<run_name>/metrics.json`

## Dashboard Commands

```bash
# Generate dashboard for specific run
python dashboard_gen.py --run first_bot

# Generate comparison of all runs
python dashboard_gen.py --compare

# Generate HTML report (open in browser)
python dashboard_gen.py --report

# Generate everything
python dashboard_gen.py --all
```

## Files Generated

After training:
- `models/first_bot/final_model.zip` - Trained model
- `logs/first_bot/` - Tensorboard logs
- `metrics/first_bot/metrics.json` - Dashboard data
- `dashboard_first_bot.png` - Training graph
- `dashboard_comparison.png` - Multi-run comparison
- `training_report.html` - HTML report

## Dashboard Metrics Explained

**Learning Curve**: Shows reward over time. Upward trend = improving agent.

**Action Distribution**: Shows what % of actions are fold/raise/all-in. Should diversify over time.

**Loss Curves**: Policy loss + value loss. Should decrease = better learning.

**Win Rate**: Percentage of hands won. Target ~33% with good play.

## Config Tuning

Edit `configs/default_config.yaml`:

```yaml
training:
  total_timesteps: 1000000      # More = longer training
  learning_rate: 0.0003          # Higher = faster learning (but unstable)
  n_steps: 2048                  # Batch size per update
  batch_size: 64                 # Mini-batch size
```

## Tips

1. **Start small**: 100k timesteps (~10 min) to test
2. **Compare runs**: Train multiple configs, compare dashboards
3. **Plot often**: Generate dashboards every 50k steps
4. **Watch learning curve**: Flat = not learning; diving = bug
5. **Check action dist**: If all folds/calls, something's wrong

## Example Workflow

```bash
# Train for 10 minutes
python train_with_metrics.py --name test_run --config configs/default_config.yaml

# After 100k steps, check dashboard
python dashboard_gen.py --run test_run

# If looks good, continue training longer
python train_with_metrics.py --name production_run --config configs/default_config.yaml

# Compare runs
python dashboard_gen.py --compare
```

## Troubleshooting

**No data in metrics?**
- Check `metrics/` folder exists
- Training must run to 10k steps minimum

**Dashboard shows all zeros?**
- Metrics update every 10k steps, wait longer
- Check `metrics/<run_name>/metrics.json` exists

**Can't open HTML report?**
- Use: `python -m http.server` then open http://localhost:8000/training_report.html