#!/usr/bin/env python3
"""
Quick test to verify opponent profit tracking works
"""

import sys
sys.path.insert(0, '/Users/mbb/Developer/Personal_Projects/LagBot')

from src.poker_env.texas_holdem_env import TexasHoldemEnv
from src.agents.ppo_agent import PPOAgent
from src.agents.random_agent import CallAgent, RandomAgent
from src.training.opponent_profit_tracker import OpponentProfitTracker
from src.training.opponent_autoplay_wrapper import OpponentAutoPlayWrapper

print("="*70)
print("TESTING OPPONENT PROFIT TRACKING")
print("="*70)

# Create environment
env = TexasHoldemEnv(
    num_players=3,
    starting_stack=1000,
    small_blind=5,
    big_blind=10,
    track_opponents=True
)

# Create opponents
opponents = [
    ('call', CallAgent(name="TestCallAgent")),
    ('random', RandomAgent(name="TestRandomAgent"))
]

# Create profit tracker
profit_tracker = OpponentProfitTracker("test_run", save_dir="metrics")

# Create wrapped environment
wrapped_env = OpponentAutoPlayWrapper(env, opponents, profit_tracker=profit_tracker)

# Create simple agent (random for testing)
print("\n1. Creating test agent...")
test_agent = RandomAgent(name="TestLearningAgent")

# Run a few hands
print("2. Running 10 test hands...")
for hand_num in range(10):
    obs, info = wrapped_env.reset()
    done = False
    steps = 0

    while not done and steps < 100:
        action = test_agent.select_action(obs)
        obs, reward, terminated, truncated, info = wrapped_env.step(action)
        done = terminated or truncated
        steps += 1

    print(f"   Hand {hand_num + 1}: {steps} steps, reward={reward:.3f}")

# Checkpoint the data
print("\n3. Saving checkpoint...")
profit_tracker.checkpoint(timestep=10)

# Print summary
print("\n4. Profit Summary:")
profit_tracker.print_summary()

# Check if data was recorded
print("\n5. Verification:")
if profit_tracker.opponent_results:
    print(f"   ✅ SUCCESS: Recorded data for {len(profit_tracker.opponent_results)} opponents")
    for opp_id, stats in profit_tracker.opponent_results.items():
        print(f"      - {stats['name']}: {stats['hands_played']} hands, "
              f"profit={stats['total_profit']:.4f}")
else:
    print("   ❌ FAILED: No opponent data recorded!")
    print("   Debug info:")
    print(f"      - Opponents: {[o[1].name for o in opponents]}")
    print(f"      - Profit tracker: {profit_tracker}")

print("\n" + "="*70)
print("TEST COMPLETE")
print("="*70)
