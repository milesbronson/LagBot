"""Tests for RegretRewardWrapper.

The wrapper sits on top of OpponentAutoPlayWrapper and rewrites the
terminal reward in three modes: profit (passthrough), regret_blend
(subtract lambda * max_regret), regret_only (replace with -max_regret).

These tests use minimal fakes so the regret math can be checked
analytically without spinning up a real poker env.
"""

import gymnasium as gym
import numpy as np
import pytest

from src.training.regret_reward_wrapper import RegretRewardWrapper, MODES


class _FakePlayer:
    def __init__(self):
        self.total_bet_this_hand = 0.0
        self.starting_stack_this_hand = 1000.0


class _FakeGameState:
    def __init__(self):
        self.players = [_FakePlayer(), _FakePlayer()]


class _FakeBaseEnv:
    """Stand-in for TexasHoldemEnv exposing only the surface the
    RegretRewardWrapper reaches for."""

    def __init__(self):
        self.learning_agent_id = 0
        self.game_state = _FakeGameState()
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(4,))
        self.action_space = gym.spaces.Discrete(3)

    def reset(self, **kwargs):
        for p in self.game_state.players:
            p.total_bet_this_hand = 0.0
            p.starting_stack_this_hand = 1000.0
        return np.zeros(4, dtype=np.float32), {}


class _FakeOuterEnv(gym.Env):
    """Mimics OpponentAutoPlayWrapper: exposes ``learner_id`` and forwards
    ``env`` to a fake base env. The step() takes a sequence of
    (invested_now, reward, done) tuples scripted by the test."""

    def __init__(self, base: _FakeBaseEnv, script):
        super().__init__()
        self.env = base
        self.learner_id = base.learning_agent_id
        self.observation_space = base.observation_space
        self.action_space = base.action_space
        self._script = list(script)
        self._step_idx = 0

    def reset(self, **kwargs):
        self._step_idx = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        # The fake script tells us what to bake into the base env state
        # before returning (invested-after-action, reward, done).
        invested_after, reward, done = self._script[self._step_idx]
        self._step_idx += 1
        self.env.game_state.players[self.learner_id].total_bet_this_hand = invested_after
        obs = np.zeros(4, dtype=np.float32)
        return obs, float(reward), bool(done), False, {}


def _build(script, mode="profit", regret_lambda=1.0):
    base = _FakeBaseEnv()
    outer = _FakeOuterEnv(base, script)
    return RegretRewardWrapper(outer, mode=mode, regret_lambda=regret_lambda), base, outer


class TestConstruction:
    def test_invalid_mode_rejected(self):
        base = _FakeBaseEnv()
        outer = _FakeOuterEnv(base, [])
        with pytest.raises(ValueError, match="mode"):
            RegretRewardWrapper(outer, mode="bogus")

    def test_negative_lambda_rejected(self):
        base = _FakeBaseEnv()
        outer = _FakeOuterEnv(base, [])
        with pytest.raises(ValueError, match="regret_lambda"):
            RegretRewardWrapper(outer, regret_lambda=-0.1)

    def test_modes_constant_matches_signature(self):
        assert set(MODES) == {"profit", "regret_blend", "regret_only"}


class TestProfitPassthrough:
    def test_terminal_reward_unchanged(self):
        # Single-decision hand: invested 100, lost it all → reward -0.1.
        wrapper, _, _ = _build([(100.0, -0.1, True)], mode="profit")
        wrapper.reset()
        _, reward, term, trunc, info = wrapper.step(0)
        assert term and not trunc
        assert reward == pytest.approx(-0.1)
        # In profit mode regret_max may still be recorded for telemetry
        # but mode label must be "profit".
        assert info["regret_mode"] == "profit"


class TestRegretMath:
    def test_max_regret_uses_shallowest_invested_decision(self):
        # Three decisions; pre-action invested = 0 → 50 → 200.
        # Hand ends with -0.6 (loses 600/1000 of stack).
        # fold_value at d0 = -0/1000 = 0.00   regret = 0   - (-0.6) = 0.60
        # fold_value at d1 = -50/1000 = -0.05 regret = -0.05 - (-0.6) = 0.55
        # fold_value at d2 = -200/1000 = -0.20 regret = -0.20 - (-0.6) = 0.40
        # max regret = 0.60 — the earliest decision dominates because
        # folding earlier would have saved the most chips.
        script = [
            (50.0, 0.0, False),
            (200.0, 0.0, False),
            (600.0, -0.6, True),
        ]
        wrapper, _, _ = _build(script, mode="regret_blend", regret_lambda=1.0)
        wrapper.reset()
        wrapper.step(0)
        wrapper.step(0)
        _, reward, _, _, info = wrapper.step(0)
        assert info["regret_max"] == pytest.approx(0.60)
        # regret_blend: reward - lambda * max_regret = -0.6 - 1.0 * 0.60
        assert reward == pytest.approx(-0.6 - 0.60)

    def test_regret_clamped_at_zero_when_actual_beats_all_folds(self):
        # Profitable hand: invested 300, ended with +0.4 net.
        # fold_value = -0.3, regret = -0.3 - 0.4 = -0.7 → clamped to 0.
        script = [(300.0, 0.4, True)]
        wrapper, _, _ = _build(script, mode="regret_blend", regret_lambda=1.0)
        wrapper.reset()
        _, reward, _, _, info = wrapper.step(0)
        assert info["regret_max"] == pytest.approx(0.0)
        assert reward == pytest.approx(0.4)

    def test_pre_action_invested_snapshot(self):
        # The wrapper must snapshot invested BEFORE the action (i.e. the
        # chip total the learner would forfeit by folding instead). The
        # script encodes "invested-after-action" so the snapshot for
        # decision N is the invested-after from step N-1 (or 0 at start).
        script = [
            (200.0, 0.0, False),   # before step 0 → 0 invested
            (200.0, -0.2, True),   # before step 1 → 200 invested
        ]
        wrapper, _, _ = _build(script, mode="regret_blend", regret_lambda=1.0)
        wrapper.reset()
        wrapper.step(0)
        _, _, _, _, info = wrapper.step(0)
        # decision 0: invested=0,   fold_value=0,   regret = 0 - (-0.2) = 0.20
        # decision 1: invested=200, fold_value=-0.2, regret = 0
        assert info["regret_max"] == pytest.approx(0.20)


class TestRawStepReward:
    def test_raw_step_reward_is_unshaped_profit(self):
        # Terminal hand: invested 0 pre-action, ends -0.6. regret_blend
        # shapes reward to -0.6 - 0.6 = -1.2, but raw_step_reward must
        # report the UNSHAPED env reward (-0.6) for the money curve.
        wrapper, _, _ = _build([(600.0, -0.6, True)], mode="regret_blend",
                               regret_lambda=1.0)
        wrapper.reset()
        _, reward, _, _, info = wrapper.step(0)
        assert reward == pytest.approx(-1.2)                  # shaped
        assert info["raw_step_reward"] == pytest.approx(-0.6)  # unshaped

    def test_raw_step_reward_present_on_nonterminal(self):
        wrapper, _, _ = _build([(50.0, 0.05, False), (100.0, -0.1, True)],
                               mode="regret_only")
        wrapper.reset()
        _, _, _, _, info0 = wrapper.step(0)
        assert info0["raw_step_reward"] == pytest.approx(0.05)

    def test_raw_step_reward_unaffected_in_profit_mode(self):
        wrapper, _, _ = _build([(100.0, -0.1, True)], mode="profit")
        wrapper.reset()
        _, reward, _, _, info = wrapper.step(0)
        assert reward == pytest.approx(-0.1)
        assert info["raw_step_reward"] == pytest.approx(-0.1)


class TestRegretOnlyMode:
    def test_replaces_reward_with_negative_max_regret(self):
        script = [(500.0, -0.5, True)]
        wrapper, _, _ = _build(script, mode="regret_only")
        wrapper.reset()
        _, reward, _, _, info = wrapper.step(0)
        # decision 0: invested=0 (pre-action), fold_value=0,
        # regret = 0 - (-0.5) = 0.5
        assert info["regret_max"] == pytest.approx(0.5)
        assert reward == pytest.approx(-0.5)


class TestNonTerminalSteps:
    def test_intermediate_rewards_unchanged(self):
        script = [(50.0, 0.05, False), (100.0, -0.1, True)]
        wrapper, _, _ = _build(script, mode="regret_only")
        wrapper.reset()
        _, r0, term0, _, info0 = wrapper.step(0)
        assert not term0
        assert r0 == pytest.approx(0.05)
        # Non-terminal info should not carry regret keys.
        assert "regret_max" not in info0
        _, r1, term1, _, info1 = wrapper.step(0)
        assert term1
        assert "regret_max" in info1


class TestResetClearsState:
    def test_invested_history_cleared_between_hands(self):
        script1 = [(900.0, -0.9, True)]
        wrapper, _, _ = _build(script1, mode="regret_blend", regret_lambda=1.0)
        wrapper.reset()
        _, _, _, _, info1 = wrapper.step(0)
        # decision 0 invested=0 → regret = 0 - (-0.9) = 0.9
        assert info1["regret_max"] == pytest.approx(0.9)

        # Replace script for second hand: short, low regret.
        wrapper.env._script = [(100.0, 0.1, True)]
        wrapper.reset()
        _, reward, _, _, info2 = wrapper.step(0)
        # decision 0 invested=0, fold_value=0, regret = -0.1 → clamped to 0
        assert info2["regret_max"] == pytest.approx(0.0)
        assert reward == pytest.approx(0.1)


class TestAttributeForwarding:
    def test_unknown_attrs_forwarded_to_inner_env(self):
        base = _FakeBaseEnv()
        outer = _FakeOuterEnv(base, [])
        outer.snapshot_card_stats = lambda: {"sentinel": True}  # mimic OpponentAutoPlayWrapper
        wrapper = RegretRewardWrapper(outer, mode="profit")
        assert wrapper.snapshot_card_stats() == {"sentinel": True}
