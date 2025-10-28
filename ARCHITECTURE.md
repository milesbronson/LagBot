# Project Architecture

## Component Interaction Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Interface                           │
│  ┌──────────────┐                           ┌──────────────┐   │
│  │   train.py   │                           │   play.py    │   │
│  │ (Training)   │                           │   (CLI Game) │   │
│  └──────┬───────┘                           └──────┬───────┘   │
└─────────┼───────────────────────────────────────────┼───────────┘
          │                                           │
          │                                           │
┌─────────▼───────────────────────────────────────────▼───────────┐
│                         Agents Layer                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  PPOAgent    │  │ RandomAgent  │  │ HumanAgent   │          │
│  │  (Learning)  │  │  (Baseline)  │  │ (Interactive)│          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         │                  │                  │                  │
│         └──────────────────┴──────────────────┘                  │
│                            │                                     │
│                   ┌────────▼─────────┐                          │
│                   │   BaseAgent      │                          │
│                   │  (Abstract)      │                          │
│                   └────────┬─────────┘                          │
└────────────────────────────┼──────────────────────────────────────┘
                             │
                             │ select_action(obs)
                             │
┌────────────────────────────▼──────────────────────────────────────┐
│                    Gym Environment Layer                          │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │               TexasHoldemEnv (Gym Interface)              │   │
│  │  • observation_space                                      │   │
│  │  • action_space                                           │   │
│  │  • reset() → observation                                  │   │
│  │  • step(action) → (obs, reward, done, info)              │   │
│  │  • render()                                               │   │
│  └───────────────────────────┬───────────────────────────────┘   │
└────────────────────────────────┼──────────────────────────────────┘
                                 │
                                 │ uses
                                 │
┌────────────────────────────────▼──────────────────────────────────┐
│                      Game Logic Layer                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │  GameState   │  │ PotManager   │  │HandEvaluator │           │
│  │              │  │              │  │              │           │
│  │• Players     │  │• Pots        │  │• Treys lib   │           │
│  │• Deck        │  │• Betting     │  │• Hand ranks  │           │
│  │• Community   │  │• Side pots   │  │• Comparison  │           │
│  │• Rounds      │  │• Rake        │  │              │           │
│  └──────┬───────┘  └──────┬───────┘  └──────────────┘           │
│         │                  │                                      │
│         └──────────┬───────┘                                      │
│                    │                                              │
│            ┌───────▼────────┐                                    │
│            │     Player      │                                    │
│            │  • Stack        │                                    │
│            │  • Hand         │                                    │
│            │  • Bets         │                                    │
│            │  • Status       │                                    │
│            └─────────────────┘                                    │
└───────────────────────────────────────────────────────────────────┘
```

## Data Flow

### Training Loop
```
1. train.py
   ↓
2. Load Config (YAML)
   ↓
3. Create TexasHoldemEnv
   ↓
4. Create PPOAgent (with Stable Baselines3)
   ↓
5. Training Loop:
   • Agent observes state
   • Agent selects action
   • Environment executes action
   • Environment returns (observation, reward, done, info)
   • Agent learns from experience
   ↓
6. Save Model
```

### Playing Loop
```
1. play.py
   ↓
2. Create TexasHoldemEnv
   ↓
3. Load/Create Agents (Human + Bots)
   ↓
4. Game Loop:
   • Reset environment (new hand)
   • For each player:
     - Show game state
     - Get action (human input or bot decision)
     - Execute action
     - Update game state
   • Determine winners
   • Distribute pots
   ↓
5. Show Statistics
```

## Observation Space Structure

```
Observation Vector (32 dimensions):
├── Hole Cards (8 dims)
│   └── 2 cards × 4 features each
├── Community Cards (20 dims)
│   └── 5 cards × 4 features each
├── Stack Info (4 dims)
│   ├── Player stack (normalized)
│   ├── Pot size (normalized)
│   ├── Current bet (normalized)
│   └── Amount to call (normalized)
└── Game Info (4 dims)
    ├── Number active players (normalized)
    ├── Position (normalized)
    ├── Betting round (normalized)
    └── Button position (normalized)
```

## Action Space

```
Discrete(3):
├── 0: Fold
├── 1: Check/Call
└── 2: Raise (minimum)
```

## File Dependencies

```
train.py
├── poker_env.texas_holdem_env
├── agents.ppo_agent
└── configs/default_config.yaml

play.py
├── poker_env.texas_holdem_env
├── agents.ppo_agent
├── agents.human_agent
└── agents.random_agent

texas_holdem_env.py
├── game_state.py
├── hand_evaluator.py
├── pot_manager.py
└── player.py

game_state.py
├── player.py
├── pot_manager.py
└── hand_evaluator.py

pot_manager.py
└── player.py

hand_evaluator.py
└── treys (external library)
```

## Testing Structure

```
tests/
├── test_env/
│   ├── test_hand_evaluator.py
│   │   └── Tests hand ranking logic
│   ├── test_pot_manager.py
│   │   └── Tests betting, pots, rake
│   └── test_texas_holdem_env.py
│       └── Tests full environment
└── test_agents/
    └── test_random_agent.py
        └── Tests agent behavior
```

## Configuration Flow

```
configs/default_config.yaml
        ↓
    train.py reads config
        ↓
    Creates environment with:
    • num_players
    • starting_stack
    • blinds
    • rake settings
        ↓
    Creates agent with:
    • learning_rate
    • batch_size
    • timesteps
    • etc.
```

## Model Saving/Loading

```
Training:
train.py → PPOAgent.train() → Save checkpoints → models/run_name/

Loading:
play.py → PPOAgent.load_agent(path) → Model ready for inference
```

## Extension Points

### Adding New Agents
```python
from agents.base_agent import BaseAgent

class MyAgent(BaseAgent):
    def select_action(self, observation, valid_actions):
        # Your strategy here
        return action
```

### Custom Observation Space
```python
# In texas_holdem_env.py
def _get_observation(self):
    # Modify observation encoding
    # Add new features
    return observation
```

### New Betting Actions
```python
# In texas_holdem_env.py
self.action_space = spaces.Discrete(5)
# 0: Fold, 1: Check, 2: Call, 3: Raise, 4: All-in
```

### Alternative RL Algorithms
```python
from stable_baselines3 import A2C, SAC, TD3

agent = A2C("MlpPolicy", env, ...)
# Or use any other SB3 algorithm
```