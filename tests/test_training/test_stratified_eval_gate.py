"""Tests for StratifiedEvalGate + build_eval_pool."""

import random

import pytest

from src.agents.anchors import ALL_ANCHORS
from src.agents.random_agent import CallAgent, RandomAgent
from src.training.agent_card import AgentCard
from src.training.agent_registry import AgentRegistry
from src.training.stratified_eval_gate import (
    StratifiedEvalGate,
    build_eval_pool,
)


def _seed_anchors(registry):
    for cls in ALL_ANCHORS:
        registry.register(AgentCard(
            id=cls.ANCHOR_ID,
            name=cls.ARCHETYPE,
            kind="anchor",
            behavior_stats=dict(cls.CANONICAL_STATS),
        ))


def _shover_stats(scale=1.0):
    return {
        "hands_observed": 500,
        "vpip": 0.98, "pfr": 0.94, "af": 40.0,
        "three_bet_percent": 0.85, "cbet_percent": 0.0,
        "fold_to_cbet_percent": 0.0,
        "went_to_showdown_percent": 0.60, "win_at_showdown_percent": 0.45,
        "wwsf_percent": 0.78,
    }


def _tag_stats():
    return {
        "hands_observed": 500,
        "vpip": 0.65, "pfr": 0.50, "af": 3.0,
        "three_bet_percent": 0.50, "cbet_percent": 0.60,
        "fold_to_cbet_percent": 1.0,
        "went_to_showdown_percent": 0.25, "win_at_showdown_percent": 0.48,
        "wwsf_percent": 0.24,
    }


class TestBuildEvalPool:
    def test_excludes_training_pool_ids(self, tmp_path):
        r = AgentRegistry(path=str(tmp_path / "r.json"))
        _seed_anchors(r)
        r.register(AgentCard(
            id="ppo_a", name="ppo_a", kind="ppo", generation=3,
            path="/tmp/fake.zip", behavior_stats=_shover_stats(),
        ))
        r.register(AgentCard(
            id="ppo_b", name="ppo_b", kind="ppo", generation=4,
            path="/tmp/fake.zip", behavior_stats=_shover_stats(),
        ))
        pool = build_eval_pool(
            r,
            current_generation=5,
            training_pool_ids=["ppo_b"],
            rng=random.Random(0),
        )
        all_ids = [c.id for cards in pool.values() for c in cards]
        assert "ppo_b" not in all_ids

    def test_excludes_anchors_when_flag_off(self, tmp_path):
        r = AgentRegistry(path=str(tmp_path / "r.json"))
        _seed_anchors(r)
        r.register(AgentCard(
            id="ppo_a", name="ppo_a", kind="ppo", generation=3,
            path="/tmp/fake.zip", behavior_stats=_shover_stats(),
        ))
        pool = build_eval_pool(
            r,
            current_generation=5,
            training_pool_ids=[],
            rng=random.Random(0),
            include_anchors=False,
        )
        all_ids = {c.id for cards in pool.values() for c in cards}
        for cls in ALL_ANCHORS:
            assert cls.ANCHOR_ID not in all_ids

    def test_includes_anchors_by_default(self, tmp_path):
        r = AgentRegistry(path=str(tmp_path / "r.json"))
        _seed_anchors(r)
        # No PPO cards — only anchors. Default should still produce a pool.
        pool = build_eval_pool(
            r,
            current_generation=5,
            training_pool_ids=[],
            rng=random.Random(0),
        )
        all_ids = {c.id for cards in pool.values() for c in cards}
        # Every anchor's own stratum should be populated by the anchor itself.
        for cls in ALL_ANCHORS:
            assert cls.ANCHOR_ID in all_ids

    def test_excludes_ppo_without_path(self, tmp_path):
        r = AgentRegistry(path=str(tmp_path / "r.json"))
        _seed_anchors(r)
        r.register(AgentCard(
            id="ppo_nopath", name="ppo_nopath", kind="ppo", generation=3,
            path=None, behavior_stats=_shover_stats(),
        ))
        pool = build_eval_pool(
            r,
            current_generation=5,
            training_pool_ids=[],
            rng=random.Random(0),
        )
        all_ids = {c.id for cards in pool.values() for c in cards}
        assert "ppo_nopath" not in all_ids

    def test_recency_weight_favours_newer(self, tmp_path):
        r = AgentRegistry(path=str(tmp_path / "r.json"))
        _seed_anchors(r)
        # Two ppo cards in the same stratum, very different generations.
        r.register(AgentCard(
            id="ppo_old", name="ppo_old", kind="ppo", generation=1,
            path="/tmp/fake.zip", behavior_stats=_shover_stats(),
        ))
        r.register(AgentCard(
            id="ppo_new", name="ppo_new", kind="ppo", generation=50,
            path="/tmp/fake.zip", behavior_stats=_shover_stats(),
        ))
        # Many draws; cards_per_stratum=1 so only one shover-stratum
        # card per call. Count how often each is picked.
        counts = {"ppo_old": 0, "ppo_new": 0}
        for seed in range(200):
            pool = build_eval_pool(
                r,
                current_generation=60,
                training_pool_ids=[],
                rng=random.Random(seed),
                cards_per_stratum=1,
                include_anchors=False,
            )
            shover_cards = pool.get("anchor_shover_v0", [])
            assert len(shover_cards) <= 1
            for c in shover_cards:
                counts[c.id] += 1
        # New: weight = 1 / (60 - 50 + 1) = 1/11
        # Old: weight = 1 / (60 - 1 + 1) = 1/60
        # Expect new to be picked ~5.5x as often. Allow slack: 3x.
        assert counts["ppo_new"] > counts["ppo_old"] * 3

    def test_empty_strata_omitted(self, tmp_path):
        r = AgentRegistry(path=str(tmp_path / "r.json"))
        _seed_anchors(r)
        # Only one PPO card, in the shover stratum.
        r.register(AgentCard(
            id="ppo_a", name="ppo_a", kind="ppo", generation=3,
            path="/tmp/fake.zip", behavior_stats=_shover_stats(),
        ))
        pool = build_eval_pool(
            r,
            current_generation=5,
            training_pool_ids=[],
            rng=random.Random(0),
            include_anchors=False,
        )
        # Only the shover stratum should appear; all others empty.
        assert set(pool.keys()) == {"anchor_shover_v0"}

    def test_multiple_strata_populated(self, tmp_path):
        r = AgentRegistry(path=str(tmp_path / "r.json"))
        _seed_anchors(r)
        r.register(AgentCard(
            id="ppo_shovery", name="ppo_shovery", kind="ppo", generation=3,
            path="/tmp/fake.zip", behavior_stats=_shover_stats(),
        ))
        r.register(AgentCard(
            id="ppo_tag", name="ppo_tag", kind="ppo", generation=4,
            path="/tmp/fake.zip", behavior_stats=_tag_stats(),
        ))
        pool = build_eval_pool(
            r,
            current_generation=5,
            training_pool_ids=[],
            rng=random.Random(0),
            include_anchors=False,
        )
        assert len(pool) == 2
        assert "anchor_shover_v0" in pool
        assert "anchor_tight_aggressive_v0" in pool


class TestStratifiedEvalGate:
    def test_evaluate_runs_end_to_end_with_simple_agents(self, tmp_path):
        """Use CallAgent vs RandomAgent so we exercise the full code
        path without requiring trained PPO models."""
        r = AgentRegistry(path=str(tmp_path / "r.json"))
        _seed_anchors(r)
        # Two non-anchor cards, different strata. We're using random/call
        # agents as the opponents — irrelevant which "kind" they are
        # since the factory below ignores kind.
        r.register(AgentCard(
            id="opp_random", name="opp_random", kind="random", generation=1,
            behavior_stats=_shover_stats(),
        ))
        r.register(AgentCard(
            id="opp_call", name="opp_call", kind="call", generation=2,
            behavior_stats=_tag_stats(),
        ))

        gate = StratifiedEvalGate(
            num_hands=30,  # small for test speed
            threshold_mbb_per_100=-1e9,  # always pass per-stratum
            min_strata_to_pass=0,
            aggregate_threshold_mbb_per_100=-1e9,
            seed=42,
            include_anchors=False,
        )

        def factory(card):
            if card.kind == "call":
                return CallAgent(name=card.name)
            return RandomAgent(name=card.name)

        candidate = CallAgent(name="candidate")
        result = gate.evaluate(
            candidate=candidate,
            candidate_card_id="candidate",
            registry=r,
            training_pool_ids=[],
            current_generation=3,
            opponent_factory=factory,
        )

        # Both strata should produce one result each.
        assert result.strata_evaluated == 2
        # Threshold is -1e9, so everything passes per-stratum.
        assert all(r.passed for r in result.results)
        assert gate.passes(result)

    def test_passes_requires_both_per_stratum_and_aggregate(self, tmp_path):
        r = AgentRegistry(path=str(tmp_path / "r.json"))
        _seed_anchors(r)
        r.register(AgentCard(
            id="opp_random", name="opp_random", kind="random", generation=1,
            behavior_stats=_shover_stats(),
        ))

        gate = StratifiedEvalGate(
            num_hands=20,
            threshold_mbb_per_100=-1e9,
            min_strata_to_pass=0,
            aggregate_threshold_mbb_per_100=1e9,  # impossible
            seed=42,
        )

        def factory(card):
            return RandomAgent(name=card.name)

        result = gate.evaluate(
            candidate=CallAgent(name="candidate"),
            candidate_card_id="candidate",
            registry=r,
            training_pool_ids=[],
            current_generation=2,
            opponent_factory=factory,
        )
        # Per-stratum passes but aggregate threshold is unreachable → FAIL.
        assert not gate.passes(result)
