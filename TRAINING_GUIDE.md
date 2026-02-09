# LagBot Training Guide

A complete guide to running training jobs for your Texas Hold'em poker bot.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Training Scripts Overview](#training-scripts-overview)
3. [Configuration Files](#configuration-files)
4. [Common Training Commands](#common-training-commands)
5. [Resuming Training](#resuming-training)
6. [Monitoring Training](#monitoring-training)
7. [Output Files](#output-files)
8. [Tips & Best Practices](#tips--best-practices)

---

## Quick Start

### Basic Training (Recommended for Beginners)

```bash
cd /Users/mbb/Developer/Personal_Projects/LagBot

# Train with default settings (evolving opponents)
python train.py

# Train with a specific config
python train.py --config configs/default_config.yaml

# Train with a custom name
python train.py --name my_experiment_v1
```

---

## Training Scripts Overview

### 1. `train.py` - Main Training Script (Recommended)

**What it does:** Trains against previous generation models (self-play evolution)

**How it works:**
- Loads the 2 most recent trained models as opponents
- If no models exist, uses CallAgent and RandomAgent
- Automatically evolves opponents as you train new generations

**When to use:** Most of the time - this is your main training script

**Command:**
```bash
python train.py --config configs/default_config.yaml --name gen_3
```

---

### 2. `train_diverse_opponents.py` - Diverse Opponent Training

**What it does:** Trains against random rule-based opponents (TightAgent, AggressiveAgent, PassiveAgent, ManiacAgent)

**How it works:**
- Randomly selects 2 opponents from a pool of 4 diverse agents
- Forces the bot to learn opponent modeling

**When to use:** When you want to train against varied playstyles

**Command:**
```bash
python train_diverse_opponents.py --config configs/shared_layers_3M.yaml --name diverse_v1
```

---

### 3. `resume_training.py` - Resume from Checkpoint

**What it does:** Continues training from a saved checkpoint

**How it works:**
- Loads a checkpoint (e.g., `model_500000_steps.zip`)
- Continues training for additional timesteps

**When to use:** When training was interrupted or you want to train longer

**Command:**
```bash
# Edit the script first to set:
# - checkpoint_path
# - remaining_steps
# - run_name

python resume_training.py
```

**Customization (edit the script):**
```python
if __name__ == "__main__":
    resume_training(
        checkpoint_path="./models/my_run/model_500000_steps.zip",
        remaining_steps=500000,
        run_name="my_run",
        config_path="configs/default_config.yaml"
    )
```

---

### 4. `train_from_checkpoint.py` - Self-Play from Checkpoint

**What it does:** Loads a checkpoint and trains against itself (self-play)

**How it works:**
- Loads a model checkpoint
- Creates 2 opponent copies of the same model
- Continues training against itself

**When to use:** For pure self-play training from an existing model

**Command:**
```bash
python train_from_checkpoint.py \
  --checkpoint ./models/gen_1/final_model.zip \
  --config configs/shared_layers_3M.yaml \
  --name self_play_v1
```

---

## Configuration Files

Located in `configs/` directory:

### Main Configs

| Config File | Description | Timesteps |
|------------|-------------|-----------|
| `default_config.yaml` | Baseline configuration | 1M |
| `shared_layers_3M.yaml` | Shared architecture layers | 3M |
| `opponent_prediction_3M.yaml` | Focus on opponent modeling | 3M |
| `aggressive_learning_3M.yaml` | Higher learning rate | 3M |
| `deep_architecture_2M.yaml` | Deeper neural network | 2M |
| `self_play_3M.yaml` | Self-play focused | 3M |

### Key Configuration Parameters

```yaml
environment:
  starting_stack: 1000          # Starting chips
  small_blind: 5                # Small blind amount
  big_blind: 10                 # Big blind amount
  reset_stacks_every_n_timesteps: 3000  # Reset chip counts

training:
  total_timesteps: 1000000      # How long to train
  learning_rate: 0.001          # Learning rate
  n_steps: 3000                 # Steps per update
  batch_size: 128               # Batch size
  gamma: 0.99                   # Discount factor
  ent_coef: 0.05                # Exploration bonus

logging:
  save_frequency: 50000         # Save checkpoints every N steps
  log_dir: "./logs/"
  model_dir: "./models/"
```

---

## Common Training Commands

### Start a New Training Run

```bash
# With default config
python train.py --name experiment_1

# With custom config
python train.py --config configs/shared_layers_3M.yaml --name shared_v1

# With diverse opponents
python train_diverse_opponents.py --config configs/aggressive_learning_3M.yaml --name diverse_agg_v1
```

### Continue Training (Method 1: Resume)

```bash
# Edit resume_training.py first, then:
python resume_training.py
```

### Continue Training (Method 2: Checkpoint Self-Play)

```bash
python train_from_checkpoint.py \
  --checkpoint ./models/gen_2/final_model.zip \
  --config configs/self_play_3M.yaml \
  --name gen_2_continued
```

---

## Resuming Training

### Option 1: Edit `resume_training.py`

Open `resume_training.py` and modify the bottom:

```python
if __name__ == "__main__":
    resume_training(
        checkpoint_path="./models/YOUR_RUN_NAME/model_500000_steps.zip",
        remaining_steps=500000,  # Additional timesteps to train
        run_name="YOUR_RUN_NAME",
        config_path="configs/default_config.yaml"
    )
```

Then run:
```bash
python resume_training.py
```

### Option 2: Use `train_from_checkpoint.py`

```bash
python train_from_checkpoint.py \
  --checkpoint ./models/gen_1/final_model.zip \
  --config configs/shared_layers_3M.yaml \
  --name gen_1_extended
```

---

## Monitoring Training

### TensorBoard (Real-time Metrics)

```bash
# Start TensorBoard
tensorboard --logdir logs/

# Then open browser to: http://localhost:6006
```

**Key Metrics to Watch:**
- `rollout/ep_rew_mean` - Average reward per episode
- `train/value_loss` - Value function loss (should decrease)
- `train/policy_loss` - Policy loss
- `train/entropy_loss` - Exploration (should be positive)

### Training Metrics

Metrics are saved in `metrics/YOUR_RUN_NAME/`:
- `episode_rewards.csv` - Rewards per episode
- `win_rates.csv` - Win rates over time
- `training_progress.json` - Overall progress

### Model Checkpoints

Models are saved in `models/YOUR_RUN_NAME/`:
- `model_50000_steps.zip` - Checkpoint at 50k steps
- `model_100000_steps.zip` - Checkpoint at 100k steps
- `final_model.zip` - Final trained model

---

## Output Files

After training, you'll have:

```
models/
  └── YOUR_RUN_NAME/
      ├── model_50000_steps.zip      # Checkpoint
      ├── model_100000_steps.zip     # Checkpoint
      └── final_model.zip            # Final model

logs/
  └── YOUR_RUN_NAME/
      └── PPO_*/                     # TensorBoard logs

metrics/
  └── YOUR_RUN_NAME/
      ├── episode_rewards.csv
      ├── win_rates.csv
      └── training_progress.json
```

---

## Tips & Best Practices

### 1. Naming Conventions

Use descriptive names with versions:
```bash
python train.py --name gen_1_baseline
python train.py --name gen_2_self_play
python train.py --name diverse_tight_v3
```

### 2. Training Duration

- **Quick test:** 100k - 500k timesteps (5-30 mins)
- **Standard training:** 1M - 3M timesteps (1-3 hours)
- **Long training:** 5M+ timesteps (5+ hours)

### 3. Checkpoint Frequency

Set `save_frequency` in config:
- **Fast iteration:** 25000 (saves every 25k steps)
- **Standard:** 50000 (saves every 50k steps)
- **Long runs:** 100000 (saves every 100k steps)

### 4. GPU vs CPU

The code automatically uses GPU if available (MPS on Mac, CUDA on Linux/Windows).

### 5. Opponent Evolution Strategy

For best results, train in generations:

```bash
# Generation 1: Start from scratch
python train.py --config configs/default_config.yaml --name gen_1

# Generation 2: Train against gen_1 (automatic)
python train.py --config configs/default_config.yaml --name gen_2

# Generation 3: Train against gen_1 and gen_2 (automatic)
python train.py --config configs/default_config.yaml --name gen_3
```

The `train.py` script automatically loads the 2 most recent models!

### 6. Experimentation

When trying new configs, use descriptive names:

```bash
python train.py --config configs/aggressive_learning_3M.yaml --name exp_aggressive_lr
python train.py --config configs/deep_architecture_2M.yaml --name exp_deep_network
```

### 7. Monitoring Long Runs

For overnight/long training runs:

```bash
# Run in background (Unix/Mac)
nohup python train.py --config configs/shared_layers_3M.yaml --name overnight_run > training.log 2>&1 &

# Or use screen/tmux
screen -S training
python train.py --config configs/shared_layers_3M.yaml --name overnight_run
# Ctrl+A, D to detach
```

---

## Common Workflows

### Workflow 1: Quick Experiment

```bash
# 1. Test with short run
python train.py --config configs/default_config.yaml --name test_quick

# 2. Monitor
tensorboard --logdir logs/

# 3. If good results, extend training
python train_from_checkpoint.py \
  --checkpoint ./models/test_quick/final_model.zip \
  --config configs/shared_layers_3M.yaml \
  --name test_extended
```

### Workflow 2: Generation Evolution

```bash
# Gen 1
python train.py --name gen_1

# Gen 2 (automatically uses gen_1 as opponent)
python train.py --name gen_2

# Gen 3 (automatically uses gen_1 and gen_2)
python train.py --name gen_3
```

### Workflow 3: Diverse Training

```bash
# Train with diverse opponents
python train_diverse_opponents.py --config configs/shared_layers_3M.yaml --name diverse_v1

# Then switch to self-play
python train_from_checkpoint.py \
  --checkpoint ./models/diverse_v1/final_model.zip \
  --config configs/self_play_3M.yaml \
  --name diverse_to_selfplay
```

---

## Troubleshooting

### Training is slow
- Check if GPU is being used (should see "cuda" or "mps" in output)
- Reduce `n_steps` or `batch_size` in config
- Use a shorter config (1M instead of 3M timesteps)

### Out of memory
- Reduce `batch_size` in config
- Reduce `n_steps` in config
- Use a smaller network architecture

### Agent not learning
- Check TensorBoard: is `ep_rew_mean` increasing?
- Try increasing `learning_rate` (e.g., 0.001 → 0.003)
- Try increasing `ent_coef` for more exploration
- Train for longer (3M+ timesteps)

### Can't find previous models
- Check `models/` directory exists
- Verify `final_model.zip` files exist in model directories
- Models must be in format: `models/gen_X/final_model.zip`

---

## Quick Reference

```bash
# Basic training
python train.py

# Custom training
python train.py --config configs/shared_layers_3M.yaml --name my_run

# Diverse opponents
python train_diverse_opponents.py --name diverse_run

# Resume (edit script first)
python resume_training.py

# Continue from checkpoint
python train_from_checkpoint.py --checkpoint ./models/gen_1/final_model.zip --config configs/self_play_3M.yaml --name continued

# Monitor
tensorboard --logdir logs/

# Background training (Unix/Mac)
nohup python train.py --name overnight > training.log 2>&1 &
```

---

**Happy Training! 🎰♠️♥️♣️♦️**
