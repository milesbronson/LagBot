"""Tests for EvalGate head-to-head shootout."""

import numpy as np
import pytest

from src.agents.base_agent import BaseAgent
from src.agents.random_agent import CallAgent
from src.training.eval_gate import EvalGate, EvalResult


class FoldAgent(BaseAgent):
    def select_action(self, observation, valid_actions=None):
        return 0


class FixedAgent(BaseAgent):
    """Picks the first valid action (deterministic, no RNG)."""

    def select_action(self, observation, valid_actions=None):
        if valid_actions is None:
            return 1
        return valid_actions[0]


class TestConstruction:
    def test_bad_num_hands_raises(self):
        with pytest.raises(ValueError):
            EvalGate(num_hands=0)
        with pytest.raises(ValueError):
            EvalGate(num_hands=-1)

    def test_defaults(self):
        gate = EvalGate()
        assert gate.num_hands == 1000
        assert gate.threshold == 0.0


class TestEvaluate:
    def test_returns_result_with_correct_metadata(self):
        gate = EvalGate(num_hands=5, seed=1)
        result = gate.evaluate(
            FixedAgent("a"),
            FixedAgent("b"),
            candidate_id="cand",
            predecessor_id="pred",
        )
        assert isinstance(result, EvalResult)
        assert result.candidate_id == "cand"
        assert result.predecessor_id == "pred"
        assert result.hands_played == 5
        assert result.big_blind == 10

    def test_fold_agent_loses_against_call_agent(self):
        gate = EvalGate(num_hands=30, seed=7, big_blind=10, small_blind=5)
        result = gate.evaluate(FoldAgent("folder"), CallAgent("caller"))
        # Folder forfeits the blinds every hand; profit should be clearly
        # negative.
        assert result.candidate_profit_chips < 0
        assert result.mbb_per_100 < 0
        assert result.candidate_wins == 0

    def test_mbb_per_100_formula(self):
        """profit/bb/hands * 100_000 should equal the reported mbb/100."""
        gate = EvalGate(num_hands=20, seed=3)
        result = gate.evaluate(FoldAgent("f"), CallAgent("c"))
        expected = (result.candidate_profit_chips / result.big_blind / result.hands_played) * 100_000.0
        assert result.mbb_per_100 == pytest.approx(expected)

    def test_deterministic_with_seed(self):
        gate1 = EvalGate(num_hands=10, seed=42)
        gate2 = EvalGate(num_hands=10, seed=42)
        r1 = gate1.evaluate(FixedAgent("a1"), FixedAgent("b1"))
        r2 = gate2.evaluate(FixedAgent("a2"), FixedAgent("b2"))
        assert r1.candidate_profit_chips == r2.candidate_profit_chips
        assert r1.mbb_per_100 == pytest.approx(r2.mbb_per_100)

    def test_different_seeds_can_differ(self):
        # Most pairs of seeds produce different traces; if they happen to
        # match it's not a bug, so just sanity-check the call works.
        r1 = EvalGate(num_hands=5, seed=1).evaluate(FixedAgent("a"), FixedAgent("b"))
        r2 = EvalGate(num_hands=5, seed=999).evaluate(FixedAgent("a"), FixedAgent("b"))
        assert r1.hands_played == r2.hands_played == 5


class TestPasses:
    def test_passes_when_above_threshold(self):
        gate = EvalGate(num_hands=1, threshold_mbb_per_100=0.0)
        result = EvalResult(
            candidate_id="c", predecessor_id="p",
            hands_played=1, candidate_profit_chips=10,
            big_blind=10, mbb_per_100=50.0,
            candidate_wins=1, candidate_losses=0,
        )
        assert gate.passes(result) is True

    def test_fails_when_below_threshold(self):
        gate = EvalGate(num_hands=1, threshold_mbb_per_100=10.0)
        result = EvalResult(
            candidate_id="c", predecessor_id="p",
            hands_played=1, candidate_profit_chips=0,
            big_blind=10, mbb_per_100=5.0,
            candidate_wins=0, candidate_losses=0,
        )
        assert gate.passes(result) is False

    def test_passes_at_threshold(self):
        gate = EvalGate(num_hands=1, threshold_mbb_per_100=10.0)
        result = EvalResult(
            candidate_id="c", predecessor_id="p",
            hands_played=1, candidate_profit_chips=0,
            big_blind=10, mbb_per_100=10.0,
            candidate_wins=0, candidate_losses=0,
        )
        assert gate.passes(result) is True


class TestSerialization:
    def test_to_dict_roundtrip(self):
        result = EvalResult(
            candidate_id="c", predecessor_id="p",
            hands_played=100, candidate_profit_chips=250.0,
            big_blind=10, mbb_per_100=250.0,
            candidate_wins=55, candidate_losses=40,
        )
        d = result.to_dict()
        assert d["candidate_id"] == "c"
        assert d["mbb_per_100"] == 250.0
        assert set(d.keys()) == {
            "candidate_id", "predecessor_id", "hands_played",
            "candidate_profit_chips", "big_blind", "mbb_per_100",
            "candidate_wins", "candidate_losses",
        }
