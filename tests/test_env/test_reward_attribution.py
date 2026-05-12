"""
Reward attribution regression tests.

These tests pin down the env's reward semantics so that the
"wrapper overwrites env reward and nukes intermediate shaping" bug
(documented in working_docs/bugs_and_training_rewrite_2026-05-10.md §1.2)
cannot silently regress.

Three things must remain true forever:

1. Env's terminal reward is always attributed to `learning_agent_id`
   (the player getting trained), regardless of who took the final action.

2. Env's intermediate fold-shaping reward is preserved alongside the
   terminal reward — the env adds them; nobody downstream should
   overwrite the sum.

3. The training wrapper passes env reward through unchanged. It must
   not recompute the terminal reward (the env already does that).

The first two are properties of the env alone. The third is a wrapper
property that the wrapper test below pins explicitly.
"""

import random

import numpy as np
import pytest

from src.poker_env.texas_holdem_env import TexasHoldemEnv
from src.agents.random_agent import CallAgent


# --------------------------------------------------------------------- helpers


def _seeded_env(num_players=2, **kwargs):
    """Build an env with deterministic dealing."""
    random.seed(0xC0FFEE)
    np.random.seed(0xC0FFEE)
    env = TexasHoldemEnv(num_players=num_players, starting_stack=1000,
                         small_blind=5, big_blind=10, **kwargs)
    env.reset(seed=0xC0FFEE)
    return env


def _patch_hand_strength(env, equity: float):
    """Pin hand-equity to a known value so fold shaping is deterministic."""
    env._calculate_hand_strength = lambda *args, **kwargs: equity


# --------------------------------------------------------------------- env reward tests


class TestEnvTerminalReward:
    def test_terminal_reward_attributed_to_learning_agent(self):
        """
        Env's terminal reward is always learner's stack delta / starting stack.
        Even when the final action is taken by a non-learning agent.
        """
        env = _seeded_env(num_players=2)
        learner_id = env.learning_agent_id
        learner = env.game_state.players[learner_id]
        initial_stack = learner.starting_stack_this_hand

        # Play the hand: every non-learner folds, learner always calls/checks.
        # Whichever player is forced to act fold-ends the hand.
        reward = 0.0
        terminated = False
        steps = 0
        while not terminated and steps < 50:
            current_idx = env.game_state.current_player_idx
            action = 1 if current_idx == learner_id else 0
            _, reward, terminated, _, _ = env.step(action)
            steps += 1

        expected = (learner.stack - initial_stack) / env.starting_stack
        # If the learner happened to be the one acting last, the action was a
        # call/check (no shaping). So reward should equal expected.
        assert reward == pytest.approx(expected, abs=1e-6)

    def test_intermediate_fold_shaping_survives_terminal(self):
        """
        When the learner's own fold ends the hand AND their hand has shapeable
        equity, the env's reward includes both the shaping AND the terminal
        stack delta. This is the property the wrapper override used to break.
        """
        env = _seeded_env(num_players=2)
        learner_id = env.learning_agent_id
        learner = env.game_state.players[learner_id]
        initial_stack = learner.starting_stack_this_hand

        # Pin equity at 0.1 — well below the 0.3 threshold where good-fold
        # shaping kicks in. Expected shaping = 0.1 * (0.3 - 0.1) / 0.3.
        _patch_hand_strength(env, equity=0.1)
        expected_shaping = 0.1 * (0.3 - 0.1) / 0.3

        # Make sure the learner is the one to act. If they're not, fold the
        # other player's seat until it's the learner's turn (HU has only one
        # other seat so this resolves in ≤1 step).
        terminated = False
        reward = 0.0
        while not terminated:
            current_idx = env.game_state.current_player_idx
            if current_idx == learner_id:
                # Learner folds — this should end the hand AND attract shaping.
                _, reward, terminated, _, _ = env.step(0)
                break
            else:
                # Push the other player to act so the learner can fold next.
                # If we end the hand here, the test is moot — re-seed and skip.
                _, reward, terminated, _, _ = env.step(1)  # call/check
                if terminated:
                    pytest.skip("Hand ended before learner could fold (seed sensitive)")

        terminal_only = (learner.stack - initial_stack) / env.starting_stack
        expected_total = terminal_only + expected_shaping

        assert terminated
        assert reward == pytest.approx(expected_total, abs=1e-4), (
            f"reward={reward}, expected terminal+shaping={expected_total}, "
            f"terminal_only={terminal_only}, shaping={expected_shaping}. "
            "If reward equals terminal_only, fold-shaping was silently dropped."
        )
        # Sanity: shaping is non-trivial (well above floating-point noise).
        assert abs(reward - terminal_only) > 0.05

    def test_no_shaping_when_non_learner_folds(self):
        """
        Shaping only fires for the learner. When a non-learner folds, the env
        returns just the terminal reward (their stack delta on the learner).
        """
        env = _seeded_env(num_players=2)
        learner_id = env.learning_agent_id
        learner = env.game_state.players[learner_id]
        initial_stack = learner.starting_stack_this_hand

        _patch_hand_strength(env, equity=0.1)

        terminated = False
        reward = 0.0
        steps = 0
        while not terminated and steps < 50:
            current_idx = env.game_state.current_player_idx
            if current_idx == learner_id:
                # Learner calls/checks — never folds.
                action = 1
            else:
                # Opponent folds.
                action = 0
            _, reward, terminated, _, _ = env.step(action)
            steps += 1

        # No fold was ever taken by the learner → no shaping reward.
        expected = (learner.stack - initial_stack) / env.starting_stack
        assert reward == pytest.approx(expected, abs=1e-6)


class TestEnvReceivesLearningAgentIdAtConstruction:
    """After parameterization, env attributes reward to the configured seat."""

    def test_env_accepts_learning_agent_id_arg(self):
        env = TexasHoldemEnv(num_players=3, learning_agent_id=2)
        assert env.learning_agent_id == 2

    def test_default_learning_agent_id_is_zero(self):
        env = TexasHoldemEnv(num_players=3)
        assert env.learning_agent_id == 0

    def test_terminal_reward_routes_to_configured_learner(self):
        """
        With learning_agent_id=1 (BB in HU), terminal reward should match
        player 1's stack delta — not player 0's.
        """
        random.seed(0xC0FFEE)
        np.random.seed(0xC0FFEE)
        env = TexasHoldemEnv(num_players=2, starting_stack=1000,
                             small_blind=5, big_blind=10,
                             learning_agent_id=1)
        env.reset(seed=0xC0FFEE)
        learner = env.game_state.players[1]
        initial_stack = learner.starting_stack_this_hand

        # Player 0 (button/SB in HU) folds → player 1 wins blinds.
        terminated = False
        reward = 0.0
        steps = 0
        while not terminated and steps < 50:
            current_idx = env.game_state.current_player_idx
            action = 0 if current_idx == 0 else 1
            _, reward, terminated, _, _ = env.step(action)
            steps += 1

        expected = (learner.stack - initial_stack) / env.starting_stack
        assert reward == pytest.approx(expected, abs=1e-6)
        # Learner is player 1 → they should have gained chips from player 0's
        # folded SB.
        assert learner.stack > initial_stack


# --------------------------------------------------------------------- wrapper tests


class TestWrapperReward:
    """The training wrapper must pass env reward through unchanged."""

    def _build_wrapper(self, num_players=2):
        from src.training.opponent_autoplay_wrapper import OpponentAutoPlayWrapper

        random.seed(0xC0FFEE)
        np.random.seed(0xC0FFEE)
        env = TexasHoldemEnv(num_players=num_players, starting_stack=1000,
                             small_blind=5, big_blind=10)
        opponents = [('call', CallAgent(name=f'C{i}'))
                     for i in range(num_players - 1)]
        wrapped = OpponentAutoPlayWrapper(env, opponents)
        wrapped.reset(seed=0xC0FFEE)
        return wrapped, env

    def test_wrapper_preserves_fold_shaping_when_learner_terminal_fold(self):
        """
        REGRESSION: the wrapper used to `reward = ...` on terminal, replacing
        the env's reward+shaping with just the terminal piece. After the fix,
        the wrapper passes env reward through.
        """
        wrapped, env = self._build_wrapper(num_players=2)
        learner = env.game_state.players[env.learning_agent_id]
        initial_stack = learner.starting_stack_this_hand
        _patch_hand_strength(env, equity=0.1)
        expected_shaping = 0.1 * (0.3 - 0.1) / 0.3

        # Learner folds first action.
        _, reward, terminated, _, _ = wrapped.step(0)

        terminal_only = (learner.stack - initial_stack) / env.starting_stack
        expected_total = terminal_only + expected_shaping

        assert terminated
        assert reward == pytest.approx(expected_total, abs=1e-4), (
            f"Wrapper returned reward={reward}; env would return "
            f"{expected_total}. The wrapper is mutating env reward — that's "
            "the bug we already fixed; do not re-introduce it."
        )

    def test_wrapper_matches_env_reward_when_hand_ends_on_opponent_fold(self):
        """
        When an opponent ends the hand by folding AFTER the learner acts, wrapper
        reward should equal env's terminal stack-delta reward for the learner.
        (No shaping, since learner didn't fold.)

        Set button_position so the learner is UTG and acts first. Otherwise
        opponents can fold during wrapper.reset()'s auto-play loop, ending the
        hand before the learner ever acts. In that case wrapper.step(1) lands on
        an already-terminal hand and (correctly, per PROD-1 fix) returns reward 0
        — but that's not what this test is verifying.
        """
        random.seed(0xDEADBEEF)
        np.random.seed(0xDEADBEEF)
        env = TexasHoldemEnv(num_players=3, starting_stack=1000,
                             small_blind=5, big_blind=10)

        from src.training.opponent_autoplay_wrapper import OpponentAutoPlayWrapper

        class FoldAgent(CallAgent):
            def select_action(self, obs, valid_actions=None):
                return 0

        opponents = [('fold', FoldAgent(name='F1')), ('fold', FoldAgent(name='F2'))]
        wrapped = OpponentAutoPlayWrapper(env, opponents)
        wrapped.reset(seed=0xDEADBEEF)
        # Force the learner (P0) to be UTG by setting button so P0 is the next
        # active player after BB. With 3 players and learner=P0: button=2 →
        # SB=P0, BB=P1, UTG=P2; button=1 → SB=P2, BB=P0, UTG=P1; button=0 →
        # SB=P1, BB=P2, UTG=P0. We want UTG=P0, so target button=0. Reset until
        # we land there (or set directly and re-do bookkeeping).
        while env.game_state.button_position != 0:
            wrapped.reset()

        learner = env.game_state.players[env.learning_agent_id]
        initial_stack = learner.starting_stack_this_hand
        assert env.game_state.current_player_idx == env.learning_agent_id, (
            "Setup: learner must be first to act"
        )

        # Learner calls; opponents fold; learner ends up winning blinds.
        _, reward, terminated, _, _ = wrapped.step(1)

        assert terminated
        expected = (learner.stack - initial_stack) / env.starting_stack
        assert reward == pytest.approx(expected, abs=1e-6)
        assert learner.stack > initial_stack
