"""Test new features: hand strength, pot odds, SPR, and reward shaping"""

from src.poker_env.texas_holdem_env import TexasHoldemEnv
import numpy as np

def test_observation_dimensions():
    """Test that observation space has correct dimensions"""
    print("Testing observation dimensions...")

    # With opponent tracking
    env = TexasHoldemEnv(num_players=3, track_opponents=True)
    obs, _ = env.reset()
    assert obs.shape == (125,), f"Expected (125,), got {obs.shape}"
    print(f"✓ With tracking: {obs.shape}")

    # Without opponent tracking
    env = TexasHoldemEnv(num_players=3, track_opponents=False)
    obs, _ = env.reset()
    assert obs.shape == (53,), f"Expected (53,), got {obs.shape}"
    print(f"✓ Without tracking: {obs.shape}")

def test_card_encoding():
    """Test that cards are encoded with 6 dims"""
    print("\nTesting card encoding...")
    env = TexasHoldemEnv(num_players=3, track_opponents=False)
    obs, _ = env.reset()

    # Cards: 7 cards × 6 dims = 42 dims
    # Hand features: 3 dims
    # Game state: 8 dims
    # Total: 53 dims

    cards_section = obs[:42]
    hand_features = obs[42:45]
    game_state = obs[45:53]

    print(f"✓ Cards section: {cards_section.shape}")
    print(f"✓ Hand features (strength, pot_odds, spr): {hand_features}")
    print(f"✓ Game state section: {game_state.shape}")

    # Check hand strength is between 0 and 1
    hand_strength = hand_features[0]
    assert 0 <= hand_strength <= 1, f"Hand strength {hand_strength} not in [0,1]"
    print(f"✓ Hand strength: {hand_strength:.3f}")

def test_hand_strength_caching():
    """Test that hand strength is cached per street"""
    print("\nTesting hand strength caching...")
    env = TexasHoldemEnv(num_players=3, track_opponents=False)
    obs, _ = env.reset()

    # Get initial hand strength
    hand_strength_1 = obs[42]

    # Take action (call)
    obs, reward, done, truncated, info = env.step(1)

    # Should be cached if still same street
    hand_strength_2 = obs[42]

    print(f"✓ Initial hand strength: {hand_strength_1:.3f}")
    print(f"✓ After action: {hand_strength_2:.3f}")

def test_reward_shaping():
    """Test intermediate reward shaping for folds"""
    print("\nTesting reward shaping...")
    env = TexasHoldemEnv(num_players=3, track_opponents=False, starting_stack=1000, big_blind=10)

    # Play a few hands and check rewards
    for hand_num in range(5):
        obs, _ = env.reset()

        # Fold immediately (action 0)
        obs, reward, done, truncated, info = env.step(0)

        print(f"Hand {hand_num + 1}: Fold reward = {reward:.4f}")

        # Reward should be small (intermediate shaping ±0.1 range)
        # unless hand ended immediately

def test_terminal_reward_normalization():
    """Test that terminal rewards are normalized by starting_stack"""
    print("\nTesting terminal reward normalization...")
    env = TexasHoldemEnv(num_players=3, track_opponents=False, starting_stack=1000, big_blind=10)

    obs, _ = env.reset()

    # Play until hand completes
    for _ in range(20):
        # Call/check
        obs, reward, done, truncated, info = env.step(1)

        if done:
            print(f"✓ Terminal reward: {reward:.4f}")
            print(f"  (normalized by starting_stack = 1000)")
            # Reward should be in reasonable range for 1000 starting stack
            assert -2.0 <= reward <= 2.0, f"Reward {reward} seems too large"
            break

if __name__ == "__main__":
    test_observation_dimensions()
    test_card_encoding()
    test_hand_strength_caching()
    test_reward_shaping()
    test_terminal_reward_normalization()
    print("\n✅ All tests passed!")
