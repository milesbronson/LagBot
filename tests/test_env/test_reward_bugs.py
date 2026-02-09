"""
Test cases for reward calculation and stack reset bugs
"""

import pytest
import numpy as np
from src.poker_env.texas_holdem_env import TexasHoldemEnv


def test_reward_never_exceeds_starting_stack():
    """Reward magnitude should never exceed starting_stack / big_blind

    Bug: Mid-hand stack resets caused impossible rewards like -490 BB
    Fix: Stack resets only happen in reset() method between hands
    """
    env = TexasHoldemEnv(
        num_players=3,
        starting_stack=1000,
        big_blind=10,
        reset_stacks_every_n_timesteps=100  # Enable to test the fix
    )
    max_reward_magnitude = 1000 / 10  # 100 BB

    for _ in range(100):
        env.reset()
        done = False
        while not done:
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            done = terminated or truncated
            if done:
                # Allow small tolerance for rounding
                assert abs(reward) <= max_reward_magnitude + 1, \
                    f"Impossible reward: {reward:.2f} (max should be ±{max_reward_magnitude} BB)"


def test_stack_reset_only_between_hands():
    """Stack reset should only happen in reset(), never during step()

    Bug: Stack reset in step() caused mid-hand chip injection
    Fix: Moved reset logic to reset() method
    """
    env = TexasHoldemEnv(
        num_players=3,
        starting_stack=1000,
        small_blind=5,
        big_blind=10,
        reset_stacks_every_n_timesteps=5  # Very low to trigger frequently
    )

    for _ in range(50):
        env.reset()
        done = False

        # Record initial total chips
        initial_total = sum(p.stack + p.total_bet_this_hand for p in env.game_state.players)

        while not done:
            # Record stacks + bets before step
            pre_total = sum(p.stack + p.total_bet_this_hand for p in env.game_state.players)
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            done = terminated or truncated

            if not done:
                # Mid-hand: total chips in play should be conserved (stacks + bets = constant)
                post_total = sum(p.stack + p.total_bet_this_hand for p in env.game_state.players)
                assert post_total == pre_total, \
                    f"Chips changed mid-hand! {pre_total} -> {post_total} (stack reset fired mid-hand?)"


def test_starting_stack_this_hand_set_before_blinds():
    """starting_stack_this_hand should reflect stack before blinds are posted

    Bug: If set after blinds, reward calculations would be off by blind amount
    Verify: This should already work correctly
    """
    env = TexasHoldemEnv(
        num_players=3,
        starting_stack=1000,
        small_blind=5,
        big_blind=10
    )
    env.reset()

    for player in env.game_state.players:
        # All players start with 1000, so starting_stack_this_hand should be 1000
        # (even for SB/BB who have already posted blinds)
        assert player.starting_stack_this_hand == 1000, \
            f"{player.name}: starting_stack_this_hand={player.starting_stack_this_hand}, expected 1000"


def test_reward_calculation_uses_stored_starting_stack():
    """Verify reward calculation uses starting_stack_this_hand, not recomputed value

    Bug: Recomputing starting_stack = stack + total_bet was fragile
    Fix: Use stored starting_stack_this_hand value
    """
    env = TexasHoldemEnv(
        num_players=3,
        starting_stack=1000,
        small_blind=5,
        big_blind=10
    )

    for _ in range(20):
        env.reset()
        done = False

        # Store starting stacks for all players
        starting_stacks = {p.player_id: p.starting_stack_this_hand for p in env.game_state.players}

        while not done:
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            done = terminated or truncated

            if done:
                # Verify reward is correct for learning agent
                learning_agent = env.game_state.players[env.learning_agent_id]
                expected_reward = (learning_agent.stack - starting_stacks[learning_agent.player_id]) / env.starting_stack

                # Allow small tolerance for intermediate reward shaping
                assert abs(reward - expected_reward) < 0.2, \
                    f"Reward mismatch: got {reward:.4f}, expected {expected_reward:.4f}"


def test_no_mid_hand_reset_messages():
    """Verify [RESET] messages only appear in reset(), never during step()

    Bug: Stack reset in step() could print [RESET] during a hand
    Fix: Reset only in reset() method
    """
    import sys
    from io import StringIO

    env = TexasHoldemEnv(
        num_players=3,
        starting_stack=1000,
        reset_stacks_every_n_timesteps=10
    )

    # Capture stdout
    captured_output = StringIO()
    old_stdout = sys.stdout

    try:
        sys.stdout = captured_output

        # Run many hands
        for _ in range(50):
            env.reset()

            # Clear buffer after reset (reset() is allowed to print [RESET])
            captured_output.truncate(0)
            captured_output.seek(0)

            done = False
            step_count = 0

            while not done:
                step_count += 1
                obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
                done = terminated or truncated

                # Check if [RESET] appeared during step() (should never happen)
                output = captured_output.getvalue()
                if '[RESET]' in output:
                    # [RESET] should never appear during step(), only in reset()
                    assert False, \
                        f"[RESET] message appeared during step() (step {step_count}, done={done})"
    finally:
        sys.stdout = old_stdout


def test_multiple_resets_preserve_chip_conservation():
    """Test that multiple stack resets don't break chip conservation

    Verify that chips are conserved within each hand, even with periodic stack resets
    """
    env = TexasHoldemEnv(
        num_players=3,
        starting_stack=1000,
        reset_stacks_every_n_timesteps=50
    )

    reset_occurred = False

    for hand_num in range(100):
        # Check stacks BEFORE reset (to verify reset happened on previous iteration)
        if reset_occurred:
            # After a reset in previous reset(), before new hand starts
            # All players should have starting_stack (before blinds posted)
            total_before_hand = sum(p.stack for p in env.game_state.players)
            expected_total = 1000 * 3  # 3 players × 1000 chips
            assert total_before_hand == expected_total, \
                f"Total chips after reset: {total_before_hand} != {expected_total}"

        env.reset()

        # Check if stack reset just happened
        reset_occurred = (env.timesteps_since_reset == 0 and env.total_timesteps > 0)

        done = False
        while not done:
            pre_total = sum(p.stack + p.total_bet_this_hand for p in env.game_state.players)
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            done = terminated or truncated

            if not done:
                post_total = sum(p.stack + p.total_bet_this_hand for p in env.game_state.players)
                assert abs(post_total - pre_total) < 0.01, \
                    f"Chip conservation violated: {pre_total} -> {post_total}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
