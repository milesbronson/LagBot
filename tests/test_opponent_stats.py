"""
Test if opponent stats are being calculated correctly
"""

import numpy as np
from src.poker_env.texas_holdem_env import TexasHoldemEnv
from src.agents.random_agent import CallAgent, RandomAgent


def test_opponent_stat_calculation():
    """
    Run hands and verify opponent stats are calculated correctly
    """

    print("\n" + "="*80)
    print("TESTING OPPONENT STAT CALCULATION")
    print("="*80)

    # Create environment with opponent tracking
    env = TexasHoldemEnv(
        num_players=3,
        starting_stack=1000,
        small_blind=5,
        big_blind=10,
        track_opponents=True
    )

    # Create known opponents
    call_agent = CallAgent(name="CallAgent")
    random_agent = RandomAgent(name="RandomAgent")

    print("\nOpponents:")
    print("  Player 1: CallAgent (should always call/check)")
    print("  Player 2: RandomAgent (random actions)")
    print()

    # Play 100 hands to gather stats
    num_hands = 100
    observations = []

    print(f"Playing {num_hands} hands to gather stats...")

    for hand_num in range(num_hands):
        obs, info = env.reset()
        done = False
        steps = 0

        while not done and steps < 100:
            current_player_idx = env.game_state.current_player_idx

            # Store observation for analysis
            if current_player_idx == 0:  # Learning agent's turn
                observations.append(obs.copy())

            # Select action based on player
            if current_player_idx == 0:
                # Learning agent - just call everything to observe opponents
                action = 1  # Call/check
            elif current_player_idx == 1:
                # CallAgent
                action = call_agent.select_action(obs)
            else:
                # RandomAgent
                action = random_agent.select_action(obs)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1

        if (hand_num + 1) % 20 == 0:
            print(f"  Completed {hand_num + 1} hands...")

    print(f"\n✓ Played {num_hands} hands")
    print(f"✓ Collected {len(observations)} observations")

    # Analyze the observations
    print("\n" + "="*80)
    print("OBSERVATION ANALYSIS")
    print("="*80)

    if len(observations) < 10:
        print("⚠️  Not enough observations collected!")
        return

    # Get a recent observation (after stats have accumulated)
    recent_obs = observations[-1]

    print(f"\nObservation shape: {recent_obs.shape}")
    print(f"Expected: (108,) = 36 base + 72 opponent stats")

    # Extract opponent features
    base_obs = recent_obs[:36]
    opp_features = recent_obs[36:]

    print(f"\nBase observation (36 dims): shape={base_obs.shape}")
    print(f"Opponent features (72 dims): shape={opp_features.shape}")

    # Parse opponent stats (8 features × 9 slots = 72)
    # We have 2 opponents, so slots 0-1 should be populated, rest should be zeros

    print("\n" + "-"*80)
    print("OPPONENT 1 STATS (CallAgent - Player 1)")
    print("-"*80)

    opp1_features = opp_features[0:8]
    feature_names = ["VPIP", "PFR", "Aggression", "3-Bet%", "C-Bet%", "Fold to C-Bet%", "Showdown%", "Confidence"]

    for i, (name, value) in enumerate(zip(feature_names, opp1_features)):
        print(f"  {name:20s}: {value:.4f}")

    print("\n" + "-"*80)
    print("OPPONENT 2 STATS (RandomAgent - Player 2)")
    print("-"*80)

    opp2_features = opp_features[8:16]

    for i, (name, value) in enumerate(zip(feature_names, opp2_features)):
        print(f"  {name:20s}: {value:.4f}")

    # Check if stats are all zeros
    print("\n" + "="*80)
    print("VALIDATION")
    print("="*80)

    if np.allclose(opp1_features, 0) and np.allclose(opp2_features, 0):
        print("❌ CRITICAL ERROR: All opponent stats are ZERO!")
        print("   The opponent tracker is not working!")
        return False

    # Check if stats are reasonable for CallAgent
    vpip_1 = opp1_features[0]
    pfr_1 = opp1_features[1]
    confidence_1 = opp1_features[7]

    print(f"\n✓ Opponent stats are NON-ZERO")
    print(f"\nCallAgent validation:")
    print(f"  VPIP: {vpip_1:.2%} (expected: HIGH, >80%)")
    print(f"  PFR: {pfr_1:.2%} (expected: LOW, <10%)")
    print(f"  Confidence: {confidence_1:.2f} (expected: ~1.0 after {num_hands} hands)")

    issues = []

    if vpip_1 < 0.5:
        issues.append("⚠️  CallAgent VPIP is too low! (should be >80%)")

    if pfr_1 > 0.2:
        issues.append("⚠️  CallAgent PFR is too high! (should be <10%)")

    if confidence_1 < 0.5:
        issues.append("⚠️  Confidence is too low after 100 hands!")

    if issues:
        print("\n" + "="*80)
        print("ISSUES FOUND:")
        print("="*80)
        for issue in issues:
            print(issue)
        return False

    print("\n" + "="*80)
    print("✅ OPPONENT STATS APPEAR CORRECT!")
    print("="*80)

    # Check variance across observations
    print("\n" + "="*80)
    print("CHECKING STAT VARIANCE OVER TIME")
    print("="*80)

    # Get first 10 and last 10 observations
    early_obs = observations[0:10] if len(observations) >= 10 else observations
    late_obs = observations[-10:]

    def get_opp1_vpip(obs):
        return obs[36]  # First opponent feature is VPIP

    early_vpips = [get_opp1_vpip(obs) for obs in early_obs]
    late_vpips = [get_opp1_vpip(obs) for obs in late_obs]

    print(f"\nCallAgent VPIP:")
    print(f"  Early hands (1-10): {np.mean(early_vpips):.2%} (std: {np.std(early_vpips):.3f})")
    print(f"  Late hands (90-100): {np.mean(late_vpips):.2%} (std: {np.std(late_vpips):.3f})")

    if np.std(late_vpips) < 0.01 and np.mean(late_vpips) > 0.7:
        print("  ✓ VPIP converged to stable high value (good for CallAgent)")
    else:
        print("  ⚠️  VPIP variance is unusual")

    return True


def test_observation_contains_stats():
    """
    Verify that observations actually contain opponent stats in the right place
    """

    print("\n" + "="*80)
    print("TESTING OBSERVATION STRUCTURE")
    print("="*80)

    env = TexasHoldemEnv(
        num_players=3,
        starting_stack=1000,
        small_blind=5,
        big_blind=10,
        track_opponents=True
    )

    obs, info = env.reset()

    print(f"\nObservation shape: {obs.shape}")
    print(f"Expected: (108,)")

    if obs.shape[0] != 108:
        print(f"❌ ERROR: Observation shape is {obs.shape}, expected (108,)")
        return False

    # Check structure
    print("\nObservation structure:")
    print(f"  [0:28]   - Hole + Community cards (7 cards × 4 features)")
    print(f"  [28:32]  - Stack, pot, bet, call amounts")
    print(f"  [32:36]  - Active players, position, round, button")
    print(f"  [36:44]  - Opponent 1 stats (8 features)")
    print(f"  [44:52]  - Opponent 2 stats (8 features)")
    print(f"  [52:108] - Empty slots (7 opponents × 8 features)")

    # Check if opponent section has any non-zero values
    opp_section = obs[36:108]

    print(f"\nOpponent section stats:")
    print(f"  Min: {opp_section.min():.4f}")
    print(f"  Max: {opp_section.max():.4f}")
    print(f"  Mean: {opp_section.mean():.4f}")
    print(f"  Non-zero values: {np.count_nonzero(opp_section)}/72")

    if np.allclose(opp_section, 0):
        print("\n⚠️  All opponent features are ZERO (no hands played yet)")
        print("   This is expected at reset - stats populate after hands are played")
    else:
        print("\n✓ Opponent features contain data")

    return True


def test_opponent_tracker_directly():
    """
    Test the OpponentTracker class directly
    """

    print("\n" + "="*80)
    print("TESTING OpponentTracker DIRECTLY")
    print("="*80)

    from src.poker_env.opponent_tracker import OpponentTracker, Action

    tracker = OpponentTracker()

    # Simulate some actions for player 1 (CallAgent behavior)
    print("\nSimulating CallAgent behavior (player 1):")
    print("  - Always calls/checks")
    print("  - Never raises")

    for i in range(20):
        # Player 1 calls preflop
        tracker.record_action(
            player_id=1,
            player_name="CallAgent",
            action=Action.CALL,
            amount=10,
            pot_size=30,
            stack_before=1000 - i*10,
            stack_after=990 - i*10,
            street="PREFLOP",
            position=1
        )

    # End hand to finalize stats
    tracker.end_hand(winners=[1], winnings={1: 30, 0: 0, 2: 0}, final_stacks={0: 990, 1: 1030, 2: 1000})

    # Get opponent info
    if 1 in tracker.opponents:
        opp_info = tracker.opponents[1]
        print(f"\nPlayer 1 stats after 20 calls:")
        print(f"  Hands played: {opp_info.hands_played}")
        print(f"  VPIP: {opp_info.vpip:.2%}")
        print(f"  PFR: {opp_info.pfr:.2%}")
    else:
        print("\n⚠️  Player 1 not tracked!")

    # Get observation features
    obs_features = tracker.get_observation_features(
        hero_id=0,
        opponent_ids=[1, 2],
        max_opponents=9,
        features_per_opponent=8
    )

    print(f"\nObservation features length: {len(obs_features)}")
    print(f"Expected: 72 (9 opponents × 8 features)")

    print(f"\nFirst 8 features (Player 1):")
    for i, val in enumerate(obs_features[:8]):
        feature_names = ["VPIP", "PFR", "AF", "3-Bet%", "C-Bet%", "Fold to C-Bet%", "Showdown%", "Confidence"]
        print(f"  {feature_names[i]:20s}: {val:.4f}")

    if obs_features[0] > 0:  # VPIP should be > 0
        print("\n✓ OpponentTracker is working correctly")
        return True
    else:
        print("\n❌ OpponentTracker stats are not being calculated!")
        return False


if __name__ == "__main__":
    print("\n" + "="*80)
    print("OPPONENT STAT DIAGNOSTIC TEST SUITE")
    print("="*80)

    # Test 1: Check observation structure
    print("\n[TEST 1] Checking observation structure...")
    test_observation_contains_stats()

    # Test 2: Test tracker directly
    print("\n[TEST 2] Testing OpponentTracker directly...")
    test_opponent_tracker_directly()

    # Test 3: Full integration test
    print("\n[TEST 3] Full integration test with actual gameplay...")
    result = test_opponent_stat_calculation()

    print("\n" + "="*80)
    print("DIAGNOSTIC COMPLETE")
    print("="*80)

    if result:
        print("\n✅ Opponent stats are being calculated correctly!")
        print("\n   This means the agent is CHOOSING to ignore them.")
        print("   → Solution: Use self-play or more diverse opponents")
    else:
        print("\n❌ Opponent stats have issues!")
        print("\n   This explains why the agent ignores them.")
        print("   → Solution: Fix the OpponentTracker implementation")
