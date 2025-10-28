# Usage Examples

## Basic Usage Patterns

### 1. Simple Environment Testing

```python
from poker_env import TexasHoldemEnv

# Create a 3-player game
env = TexasHoldemEnv(num_players=3, starting_stack=1000)

# Play one hand with random actions
obs = env.reset()
print("Initial observation shape:", obs.shape)

done = False
step = 0
while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    env.render()
    step += 1
    
print(f"Hand finished in {step} steps")
print(f"Winners: {info.get('winnings', {})}")
```

### 2. Testing with Specific Agents

```python
from poker_env import TexasHoldemEnv
from agents.random_agent import RandomAgent, CallAgent, WeightedRandomAgent

# Create environment
env = TexasHoldemEnv(num_players=3)

# Create different agent types
agents = [
    RandomAgent(name="Random"),
    CallAgent(name="Caller"),
    WeightedRandomAgent(name="Weighted", fold_weight=0.1, call_weight=0.6, raise_weight=0.3)
]

# Simulate a session
num_hands = 10
for hand in range(num_hands):
    obs = env.reset()
    done = False
    
    while not done:
        current_idx = env.game_state.current_player_idx
        agent = agents[current_idx]
        
        action = agent.select_action(obs)
        obs, reward, done, info = env.step(action)
    
    print(f"Hand {hand+1} complete. Stacks: {[p.stack for p in env.game_state.players]}")
```

### 3. Training a Quick Test Agent

```python
from poker_env import TexasHoldemEnv
from agents.ppo_agent import PPOAgent

# Create environment
env = TexasHoldemEnv(num_players=2)  # Heads-up for faster training

# Create PPO agent with fast training settings
agent = PPOAgent(
    env=env,
    learning_rate=0.001,
    n_steps=512,
    batch_size=64,
    verbose=1
)

# Quick training run
print("Training for 10,000 steps...")
agent.train(total_timesteps=10000)

# Save the model
agent.save("test_model")
print("Model saved!")

# Test it
obs = env.reset()
for _ in range(10):
    action = agent.select_action_deterministic(obs)
    obs, reward, done, info = env.step(action)
    if done:
        break
```

### 4. Evaluating Agent Performance

```python
from poker_env import TexasHoldemEnv
from agents.ppo_agent import PPOAgent
from agents.random_agent import RandomAgent

def evaluate_agent(agent, opponent, num_hands=100):
    """Evaluate agent against an opponent"""
    env = TexasHoldemEnv(num_players=2)
    
    agent_winnings = 0
    opponent_winnings = 0
    
    for hand in range(num_hands):
        obs = env.reset()
        done = False
        
        while not done:
            current_idx = env.game_state.current_player_idx
            
            if current_idx == 0:
                action = agent.select_action_deterministic(obs)
            else:
                action = opponent.select_action(obs)
            
            obs, reward, done, info = env.step(action)
        
        # Track winnings
        final_stacks = [p.stack for p in env.game_state.players]
        agent_winnings += final_stacks[0] - 1000
        opponent_winnings += final_stacks[1] - 1000
    
    print(f"Results over {num_hands} hands:")
    print(f"Agent: {agent_winnings:+d} chips")
    print(f"Opponent: {opponent_winnings:+d} chips")
    print(f"Agent win rate: {(agent_winnings / (abs(agent_winnings) + abs(opponent_winnings))) * 100:.1f}%")

# Load trained agent
env = TexasHoldemEnv(num_players=2)
trained_agent = PPOAgent.load_agent("models/my_bot/final_model", env)

# Evaluate against random opponent
random_opponent = RandomAgent()
evaluate_agent(trained_agent, random_opponent, num_hands=1000)
```

### 5. Custom Training Callback

```python
from stable_baselines3.common.callbacks import BaseCallback
from agents.ppo_agent import PPOAgent
from poker_env import TexasHoldemEnv

class CustomPokerCallback(BaseCallback):
    def __init__(self, eval_freq=1000):
        super().__init__()
        self.eval_freq = eval_freq
        self.best_reward = -float('inf')
    
    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            # Custom evaluation logic
            print(f"\nStep {self.n_calls}: Evaluating...")
            # Add your evaluation code here
        return True

# Use custom callback
env = TexasHoldemEnv(num_players=3)
agent = PPOAgent(env=env)

callback = CustomPokerCallback(eval_freq=5000)
agent.train(total_timesteps=100000, callback=callback)
```

### 6. Multi-Agent Self-Play Setup

```python
from poker_env import TexasHoldemEnv
from agents.ppo_agent import PPOAgent
import copy

def self_play_training(num_iterations=5, steps_per_iteration=50000):
    """Train agents against previous versions"""
    env = TexasHoldemEnv(num_players=2)
    
    # Start with initial agent
    current_agent = PPOAgent(env=env, name="agent_v0")
    
    for iteration in range(num_iterations):
        print(f"\n{'='*60}")
        print(f"Training Iteration {iteration + 1}")
        print(f"{'='*60}")
        
        # Train current agent
        current_agent.train(total_timesteps=steps_per_iteration)
        
        # Save this version
        model_path = f"models/self_play/agent_v{iteration}"
        current_agent.save(model_path)
        
        # Create next generation agent
        if iteration < num_iterations - 1:
            current_agent = PPOAgent(env=env, name=f"agent_v{iteration + 1}")

# Run self-play training
self_play_training(num_iterations=5, steps_per_iteration=50000)
```

### 7. Analyzing Hand Histories

```python
from poker_env import TexasHoldemEnv
from poker_env.hand_evaluator import HandEvaluator

def analyze_hand(hole_cards_str, community_cards_str):
    """Analyze a specific poker hand"""
    evaluator = HandEvaluator()
    
    # Convert string cards to integers
    hole_cards = [evaluator.string_to_card(c) for c in hole_cards_str.split()]
    community_cards = [evaluator.string_to_card(c) for c in community_cards_str.split()]
    
    # Evaluate
    rank = evaluator.evaluate_hand(hole_cards, community_cards)
    hand_class = evaluator.get_rank_class(rank)
    hand_name = evaluator.class_to_string(hand_class)
    
    print(f"Hole cards: {hole_cards_str}")
    print(f"Board: {community_cards_str}")
    print(f"Hand: {hand_name}")
    print(f"Rank: {rank} (lower is better)")

# Example usage
analyze_hand("As Ks", "Ah Kh Kc Jh Tc")
# Output: Three of a Kind
```

### 8. Testing Side Pots

```python
from poker_env.pot_manager import PotManager
from poker_env.player import Player

def test_complex_side_pots():
    """Test complex side pot scenarios"""
    pot_manager = PotManager(small_blind=5, big_blind=10)
    
    # Create players with different stack sizes
    players = [
        Player(0, 1000, "Alice"),
        Player(1, 500, "Bob"),
        Player(2, 200, "Charlie")
    ]
    
    # Simulate betting where players go all-in at different levels
    players[2].total_bet_this_hand = 200  # Charlie all-in
    players[1].total_bet_this_hand = 500  # Bob all-in
    players[0].total_bet_this_hand = 500  # Alice calls
    
    # Calculate pots
    pots = pot_manager.calculate_side_pots(players)
    
    print("Side Pots Created:")
    for i, pot in enumerate(pots):
        print(f"  Pot {i+1}: ${pot.amount}, Eligible players: {pot.eligible_players}")

test_complex_side_pots()
```

### 9. Custom Reward Shaping

```python
from poker_env import TexasHoldemEnv
import numpy as np

class CustomRewardEnv(TexasHoldemEnv):
    """Environment with custom reward shaping"""
    
    def step(self, action):
        obs, reward, done, info = super().step(action)
        
        # Add custom reward components
        if not done:
            # Small reward for staying in hand
            reward += 0.01
            
            # Penalty for folding
            if action == 0:
                reward -= 0.05
        
        return obs, reward, done, info

# Use custom environment
env = CustomRewardEnv(num_players=3)
obs = env.reset()

# Continue training as normal
```

### 10. Batch Evaluation Script

```python
import numpy as np
from poker_env import TexasHoldemEnv
from agents.ppo_agent import PPOAgent
from agents.random_agent import RandomAgent, CallAgent

def comprehensive_evaluation(model_path, num_hands=1000):
    """Comprehensively evaluate a trained agent"""
    env = TexasHoldemEnv(num_players=2)
    agent = PPOAgent.load_agent(model_path, env)
    
    opponents = {
        'Random': RandomAgent(),
        'Caller': CallAgent(),
    }
    
    results = {}
    
    for opp_name, opponent in opponents.items():
        print(f"\nEvaluating against {opp_name}...")
        winnings = []
        
        for _ in range(num_hands):
            obs = env.reset()
            done = False
            
            initial_stack = env.game_state.players[0].stack
            
            while not done:
                current_idx = env.game_state.current_player_idx
                
                if current_idx == 0:
                    action = agent.select_action_deterministic(obs)
                else:
                    action = opponent.select_action(obs)
                
                obs, reward, done, info = env.step(action)
            
            final_stack = env.game_state.players[0].stack
            winnings.append(final_stack - initial_stack)
        
        results[opp_name] = {
            'mean': np.mean(winnings),
            'std': np.std(winnings),
            'min': np.min(winnings),
            'max': np.max(winnings),
            'win_rate': sum(1 for w in winnings if w > 0) / len(winnings)
        }
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    for opp_name, stats in results.items():
        print(f"\nVs {opp_name}:")
        print(f"  Mean: {stats['mean']:+.2f} chips/hand")
        print(f"  Std:  {stats['std']:.2f}")
        print(f"  Range: [{stats['min']:+.0f}, {stats['max']:+.0f}]")
        print(f"  Win Rate: {stats['win_rate']*100:.1f}%")

# Run evaluation
comprehensive_evaluation("models/my_bot/final_model", num_hands=1000)
```

## Command Line Examples

### Training

```bash
# Basic training
python train.py

# Custom config
python train.py --config configs/heads_up_config.yaml

# Named run
python train.py --name aggressive_bot_v1

# Monitor with tensorboard
tensorboard --logdir logs/
```

### Playing

```bash
# Play against random bot
python play.py

# Play against trained bot
python play.py --model models/my_bot/final_model.zip

# Play against multiple bots
python play.py --opponents 3

# Play against calling stations
python play.py --opponent-type call --opponents 2
```

### Testing

```bash
# Run all tests
pytest

# Specific test file
pytest tests/test_env/test_hand_evaluator.py

# With coverage
pytest --cov=poker_env --cov=agents

# Verbose output
pytest -v

# Stop on first failure
pytest -x
```

## Tips and Tricks

### 1. Quick Debugging
```python
# Enable verbose output
env = TexasHoldemEnv(num_players=2)
env.reset()
env.render()  # Shows current game state

# Check observation encoding
obs = env._get_observation()
print("Observation:", obs)
print("Shape:", obs.shape)
```

### 2. Profiling Training Speed
```python
import time

env = TexasHoldemEnv(num_players=2)
agent = PPOAgent(env=env, verbose=0)

start = time.time()
agent.train(total_timesteps=10000)
elapsed = time.time() - start

print(f"Training speed: {10000/elapsed:.0f} steps/second")
```

### 3. Saving Training Checkpoints
```python
from agents.ppo_agent import TrainingCallback

callback = TrainingCallback(
    save_freq=10000,  # Save every 10k steps
    save_path="models/checkpoints/"
)

agent.train(total_timesteps=100000, callback=callback)
```

### 4. Testing Environment Reset
```python
env = TexasHoldemEnv(num_players=3)

# Play multiple hands
for hand in range(5):
    obs = env.reset()
    print(f"Hand {hand+1}:")
    print(f"  Button position: {env.game_state.button_position}")
    print(f"  Player stacks: {[p.stack for p in env.game_state.players]}")
```

These examples should give you a solid foundation for working with the poker RL bot!