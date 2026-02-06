"""
Test if rewards are being calculated correctly
"""

import numpy as np
from src.poker_env.texas_holdem_env import TexasHoldemEnv
from src.agents.random_agent import CallAgent, RandomAgent


def test_reward_calculation():
    """
    Verify rewards make sense
    """

    print("\n" + "="*80)
    print("TESTING REWARD CALCULATION")
    print("="*80)

    env = TexasHoldemEnv(
        num_players=3,
        starting_stack=1000,
        small_blind=5,
        big_blind=10,
        track_opponents=True
    )

    call_agent = CallAgent()
    random_agent = RandomAgent()

    # Track rewards
    episode_rewards = []
    player_0_wins = 0
    player_0_losses = 0
    total_hands = 0

    print("\nPlaying 50 hands and tracking rewards...")

    for hand_num in range(50):
        obs, info = env.reset()

        # Record starting stack
        starting_stack = env.game_state.players[0].stack

        done = False
        hand_reward = 0
        steps = 0

        while not done and steps < 100:
            current_idx = env.game_state.current_player_idx

            # Select action
            if current_idx == 0:
                action = 1  # Always call/check
            elif current_idx == 1:
                action = call_agent.select_action(obs)
            else:
                action = random_agent.select_action(obs)

            obs, reward, terminated, truncated, info = env.step(action)

            # Only track reward for player 0
            if current_idx == 0:
                hand_reward += reward

            done = terminated or truncated
            steps += 1

        # Get final stack
        final_stack = env.game_state.players[0].stack
        actual_change = final_stack - starting_stack

        episode_rewards.append(hand_reward)
        total_hands += 1

        if actual_change > 0:
            player_0_wins += 1
        elif actual_change < 0:
            player_0_losses += 1

        if hand_num < 5 or hand_num >= 45:  # Show first 5 and last 5
            print(f"  Hand {hand_num+1}: Start=${starting_stack}, End=${final_stack}, "
                  f"Change=${actual_change:+d}, Reward={hand_reward:+.1f}")

    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}")

    print(f"\nTotal hands: {total_hands}")
    print(f"Player 0 wins: {player_0_wins} ({player_0_wins/total_hands*100:.1f}%)")
    print(f"Player 0 losses: {player_0_losses} ({player_0_losses/total_hands*100:.1f}%)")
    print(f"Player 0 ties: {total_hands - player_0_wins - player_0_losses}")

    print(f"\nReward statistics:")
    print(f"  Mean: {np.mean(episode_rewards):.2f}")
    print(f"  Std: {np.std(episode_rewards):.2f}")
    print(f"  Min: {np.min(episode_rewards):.2f}")
    print(f"  Max: {np.max(episode_rewards):.2f}")

    # Analysis
    print(f"\n{'='*80}")
    print("ANALYSIS")
    print(f"{'='*80}")

    if player_0_wins / total_hands < 0.4:
        print("\n⚠️  Player 0 win rate is LOW!")
        print("   Even with call-only strategy, should beat CallAgent + RandomAgent")
        print("   Something might be wrong with game logic")

    if np.mean(episode_rewards) < -5:
        print("\n⚠️  Average reward is very negative!")
        print("   Suggests player 0 is losing money consistently")
    elif np.mean(episode_rewards) > 5:
        print("\n✓ Average reward is positive")
        print("  Player 0 is winning money on average")
    else:
        print("\n~ Average reward near zero")
        print("  Close to break-even")

    return episode_rewards


if __name__ == "__main__":
    test_reward_calculation()
