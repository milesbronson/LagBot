# Quick Start Guide

## Setup

1. **Create a virtual environment:**
```bash
cd poker_rl_bot
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Install the package in development mode:**
```bash
pip install -e .
```

## Testing the Environment

### Run Tests
```bash
# Run all tests
pytest

# Run only environment tests
pytest tests/test_env/

# Run only agent tests
pytest tests/test_agents/

# Run with coverage report
pytest --cov=poker_env --cov=agents --cov-report=html
```

### Test the Environment Manually
```python
from poker_env import TexasHoldemEnv

# Create environment
env = TexasHoldemEnv(num_players=3)

# Reset and play a hand
obs = env.reset()
env.render()

done = False
while not done:
    action = env.action_space.sample()  # Random action
    obs, reward, done, info = env.step(action)
    env.render()

print(f"Hand complete! Winnings: {info.get('winnings', {})}")
```

## Training a Bot

### Basic Training
```bash
python train.py --config configs/default_config.yaml
```

### Custom Training Run
```bash
python train.py --config configs/default_config.yaml --name my_first_bot
```

### Monitor Training with Tensorboard
```bash
tensorboard --logdir logs/
```

Then open http://localhost:6006 in your browser.

## Playing Against the Bot

### Play against random opponents
```bash
python play.py --opponents 2
```

### Play against a trained bot
```bash
python play.py --model models/my_first_bot/final_model.zip --opponents 1
```

### Play against calling stations
```bash
python play.py --opponent-type call --opponents 3
```

## Project Structure Overview

```
poker_rl_bot/
├── poker_env/              # Core poker environment
│   ├── texas_holdem_env.py # Main Gym environment
│   ├── game_state.py       # Game logic
│   ├── hand_evaluator.py   # Hand rankings
│   ├── pot_manager.py      # Betting & pots
│   └── player.py           # Player class
│
├── agents/                 # AI agents
│   ├── base_agent.py       # Abstract base
│   ├── ppo_agent.py        # PPO implementation
│   ├── random_agent.py     # Random baseline
│   └── human_agent.py      # Human player
│
├── tests/                  # Test suite
│   ├── test_env/          # Environment tests
│   └── test_agents/       # Agent tests
│
├── train.py               # Training script
└── play.py                # CLI game interface
```

## Configuration

Edit `configs/default_config.yaml` to customize:

- **Environment settings**: Number of players, stack sizes, blinds
- **Training parameters**: Learning rate, batch size, timesteps
- **Rake settings**: Enable/disable rake, set percentages

Example configuration:
```yaml
environment:
  num_players: 6
  starting_stack: 1000
  small_blind: 5
  big_blind: 10
  rake_enabled: false

training:
  total_timesteps: 1000000
  learning_rate: 0.0003
```

## Next Steps

1. **Train a basic agent** - Start with a short training run (100k steps)
2. **Evaluate performance** - Play against it to see how it performs
3. **Tune hyperparameters** - Adjust learning rate, batch size, etc.
4. **Implement opponent modeling** - Add exploitative strategies
5. **Self-play evolution** - Train against previous versions

## Common Issues

### Import Errors
If you get import errors, make sure you've installed the package:
```bash
pip install -e .
```

### Treys Installation Issues
If treys fails to install:
```bash
pip install treys==0.1.8 --no-cache-dir
```

### GPU Support (Optional)
For faster training with GPU:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Tips for Training

1. **Start small** - Train with fewer players (2-3) first
2. **Use checkpoints** - The trainer saves models every 50k steps
3. **Monitor progress** - Use Tensorboard to watch reward curves
4. **Iterate quickly** - Short training runs help you test changes faster
5. **Compare strategies** - Train multiple agents and have them play each other

## Example Training Session

```bash
# 1. Quick test (5 minutes)
python train.py --name quick_test

# 2. Monitor in real-time
tensorboard --logdir logs/quick_test

# 3. Test the trained bot
python play.py --model models/quick_test/final_model.zip

# 4. Longer training (1 hour+)
python train.py --name full_training

# Auto-loads the most recent trained model
python play.py --opponents 2

# Or specify a model explicitly (as before)
python play.py --model models/run_20231115_143022/final_model.zip --opponents 2

# Random bots
python play.py --opponent-type random --opponents 3
```

Happy training! 🎰