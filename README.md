# Texas Hold'em RL Bot

A reinforcement learning bot for Texas Hold'em poker using PPO (Proximal Policy Optimization) trained via self-play.

## Project Overview

This project implements a complete Texas Hold'em poker environment compatible with OpenAI Gym, along with RL agents that learn optimal poker strategies through self-play. The bot can adapt its playstyle based on opponent weaknesses using machine learning.

## Features

- **Gym-compatible environment** supporting 2-10 players
- **Complete poker logic**: hand evaluation, betting rounds, side pots
- **Configurable rake** (can be enabled/disabled)
- **Dynamic player management** (players can join/leave)
- **Command-line interface** for human vs bot play
- **PPO-based RL agents** using Stable Baselines3
- **Comprehensive test suite** for environment and agents
- **Multiple agent strategies** (exploitative playstyles)

## Project Structure

```
poker_rl_bot/
├── poker_env/              # Poker environment implementation
│   ├── __init__.py
│   ├── texas_holdem_env.py # Main Gym environment
│   ├── game_state.py       # Game state management
│   ├── hand_evaluator.py   # Hand ranking logic
│   ├── pot_manager.py      # Betting and pot management
│   └── player.py           # Player abstraction
├── agents/                 # RL agents and strategies
│   ├── __init__.py
│   ├── base_agent.py       # Abstract base agent
│   ├── ppo_agent.py        # PPO agent implementation
│   ├── random_agent.py     # Random baseline agent
│   └── human_agent.py      # Human player interface
├── tests/                  # Test suite
│   ├── test_env/          # Environment tests
│   │   ├── test_hand_evaluator.py
│   │   ├── test_pot_manager.py
│   │   └── test_texas_holdem_env.py
│   └── test_agents/       # Agent tests
│       ├── test_ppo_agent.py
│       └── test_random_agent.py
├── configs/               # Configuration files
│   └── default_config.yaml
├── logs/                  # Training logs
├── models/                # Saved models
├── train.py              # Training script
├── play.py               # CLI for playing against bot
└── requirements.txt      # Dependencies

```

## Installation

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training a bot
```bash
python train.py --config configs/default_config.yaml
```

### Playing against the bot
```bash
python play.py --model models/best_model.zip
```

### Running tests
```bash
# All tests
pytest

# Environment tests only
pytest tests/test_env/

# Agent tests only
pytest tests/test_agents/

# With coverage
pytest --cov=poker_env --cov=agents
```

## Development Roadmap

- [x] Project setup and structure
- [ ] Core poker environment implementation
- [ ] Hand evaluation system
- [ ] Pot and betting management
- [ ] Multi-player support (2-10 players)
- [ ] Gym environment interface
- [ ] PPO agent implementation
- [ ] Self-play training loop
- [ ] Command-line interface
- [ ] Exploitative strategy variations
- [ ] Advanced opponent modeling

## Contributing

This is a personal project, but suggestions and improvements are welcome!

## License

MIT License