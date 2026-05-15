"""Tests for RegressionEval — sweeping a new card against prior cards."""

import json

from src.training.agent_card import AgentCard
from src.training.agent_registry import AgentRegistry
from src.training.regression_eval import RegressionEval


def _registry(tmp_path) -> AgentRegistry:
    return AgentRegistry(path=str(tmp_path / "reg.json"))


def test_evaluate_runs_against_all_prior_cards(tmp_path):
    registry = _registry(tmp_path)
    registry.register(AgentCard(id="r1", name="r1", kind="random", generation=0))
    registry.register(AgentCard(id="c1", name="c1", kind="call", generation=0))
    new_card = AgentCard(id="cand", name="cand", kind="random", generation=2)
    registry.register(new_card)

    reg = RegressionEval(num_hands=10, threshold_mbb_per_100=-1e12, seed=1)
    result = reg.evaluate(new_card, registry)

    assert len(result.results) == 2
    assert {r.opponent_id for r in result.results} == {"r1", "c1"}
    assert result.passed + result.regressed == 2
    assert all(r.opponent_id != new_card.id for r in result.results)


def test_evaluate_skips_self_and_unloadable_ppo(tmp_path):
    registry = _registry(tmp_path)
    registry.register(AgentCard(id="ghost", name="ghost", kind="ppo", path=None))
    registry.register(AgentCard(id="r1", name="r1", kind="random"))
    new_card = AgentCard(id="cand", name="cand", kind="call", generation=1)
    registry.register(new_card)

    reg = RegressionEval(num_hands=10, threshold_mbb_per_100=-1e12, seed=2)
    result = reg.evaluate(new_card, registry)

    opp_ids = [r.opponent_id for r in result.results]
    assert "ghost" not in opp_ids
    assert "cand" not in opp_ids
    assert "r1" in opp_ids


def test_include_fixtures_false_excludes_rule_cards(tmp_path):
    registry = _registry(tmp_path)
    registry.register(AgentCard(id="r1", name="r1", kind="random"))
    registry.register(AgentCard(id="c1", name="c1", kind="call"))
    new_card = AgentCard(id="cand", name="cand", kind="random", generation=1)
    registry.register(new_card)

    reg = RegressionEval(
        num_hands=10, threshold_mbb_per_100=-1e12, seed=0, include_fixtures=False,
    )
    result = reg.evaluate(new_card, registry)
    assert result.results == []


def test_threshold_flags_regressions(tmp_path):
    registry = _registry(tmp_path)
    registry.register(AgentCard(id="r1", name="r1", kind="random"))
    new_card = AgentCard(id="cand", name="cand", kind="random", generation=1)
    registry.register(new_card)

    # An impossibly high threshold forces every matchup to fail.
    reg = RegressionEval(num_hands=10, threshold_mbb_per_100=1e12, seed=0)
    result = reg.evaluate(new_card, registry)
    assert result.regressed == len(result.results)
    assert result.passed == 0
    assert all(not r.passed for r in result.results)


def test_save_writes_json(tmp_path):
    registry = _registry(tmp_path)
    registry.register(AgentCard(id="r1", name="r1", kind="random"))
    new_card = AgentCard(id="cand", name="cand", kind="random", generation=1)
    registry.register(new_card)

    reg = RegressionEval(num_hands=10, threshold_mbb_per_100=-1e12, seed=0)
    result = reg.evaluate(new_card, registry)

    out = tmp_path / "out" / "regression.json"
    reg.save(result, out)
    assert out.exists()
    data = json.loads(out.read_text())
    assert data["new_card_id"] == "cand"
    assert len(data["results"]) == 1
