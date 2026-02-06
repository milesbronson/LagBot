"""
Test rewards through the wrapper (as training uses)
"""

import numpy as np
from src.poker_env.texas_holdem_env import TexasHoldemEnv
from src.agents.random_agent import CallAgent, RandomAgent
from train import OpponentAutoPlayWrapper


def test_wrapper_rewards():
    print("\n" + "="*80)
    print("TESTING REWARDS THROUGH WRAPPER")
    print("="*80)

    env = TexasHoldemEnv(
        num_players=3,
        starting_stack=1000,
        small_blind=5,
        big_blind=10,
        track_opponents=True
    )

    opponents = [
        ('call', CallAgent()),
        ('random', RandomAgent())
    ]

    wrapped_env = OpponentAutoPlayWrapper(env, opponents)

    episode_rewards = []
    zero_reward_wins = 0
    nonzero_reward_wins = 0

    print("\nPlaying 50 hands through wrapper...")

    for hand_num in range(50):
        obs, info = wrapped_env.reset()

        starting_stack = env.game_state.players[0].stack

        done = False
        total_reward = 0
        steps = 0

        while not done and steps < 100:
            # Learning agent always calls
            action = 1

            obs, reward, terminated, truncated, info = wrapped_env.step(action)
            total_reward += reward

            done = terminated or truncated
            steps += 1

        final_stack = env.game_state.players[0].stack
        actual_change = final_stack - starting_stack

        episode_rewards.append(total_reward)

        # Check if we won but got zero reward
        if actual_change > 0:
            if total_reward == 0:
                zero_reward_wins += 1
            else:
                nonzero_reward_wins += 1

        if hand_num < 5:
            print(f"  Hand {hand_num+1}: Change=${actual_change:+d}, "
                  f"Total Reward={total_reward:+.1f}")

    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}")

    print(f"\nReward statistics:")
    print(f"  Mean: {np.mean(episode_rewards):.2f}")
    print(f"  Std: {np.std(episode_rewards):.2f}")
    print(f"  Min: {np.min(episode_rewards):.2f}")
    print(f"  Max: {np.max(episode_rewards):.2f}")

    print(f"\nWinning hands:")
    print(f"  With correct reward: {nonzero_reward_wins}")
    print(f"  With ZERO reward (BUG!): {zero_reward_wins}")

    if zero_reward_wins > 0:
        print(f"\n❌ WRAPPER BUG: {zero_reward_wins} winning hands had zero reward!")
        print("   The wrapper is not correctly passing rewards to the learning agent")
        return False
    else:
        print(f"\n✅ Wrapper correctly passes all rewards")
        return True


if __name__ == "__main__":
    test_wrapper_rewards()
