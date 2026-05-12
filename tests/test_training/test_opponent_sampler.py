"""Tests for OpponentSampler strategies."""

import random

import pytest

from src.training.agent_card import AgentCard
from src.training.agent_registry import AgentRegistry
from src.training.opponent_sampler import OpponentSampler


def _make_registry(tmp_path, specs):
    """specs: iterable of (id, kind, generation)."""
    r = AgentRegistry(path=str(tmp_path / "registry.json"))
    for aid, kind, gen in specs:
        r.register(AgentCard(id=aid, name=aid, kind=kind, generation=gen))
    return r


def _sampler(registry, seed=0):
    return OpponentSampler(registry, rng=random.Random(seed))


class TestLatestStrategy:
    def test_returns_top_n_by_generation(self, tmp_path):
        r = _make_registry(tmp_path, [
            ("a", "ppo", 0), ("b", "ppo", 5), ("c", "ppo", 3), ("d", "ppo", 9),
        ])
        s = _sampler(r)
        ids = [c.id for c in s.sample(n=2, strategy="latest")]
        assert ids == ["d", "b"]

    def test_kind_filter(self, tmp_path):
        r = _make_registry(tmp_path, [
            ("ppo1", "ppo", 1), ("rule1", "rule", 9), ("ppo2", "ppo", 2),
        ])
        s = _sampler(r)
        ids = [c.id for c in s.sample(n=5, strategy="latest", kind="ppo")]
        assert ids == ["ppo2", "ppo1"]

    def test_exclude_ids(self, tmp_path):
        r = _make_registry(tmp_path, [
            ("a", "ppo", 5), ("b", "ppo", 4), ("c", "ppo", 3),
        ])
        s = _sampler(r)
        ids = [c.id for c in s.sample(n=2, strategy="latest", exclude_ids=["a"])]
        assert ids == ["b", "c"]

    def test_returns_full_pool_if_smaller_than_n(self, tmp_path):
        r = _make_registry(tmp_path, [("a", "ppo", 0)])
        s = _sampler(r)
        ids = [c.id for c in s.sample(n=5, strategy="latest")]
        assert ids == ["a"]


class TestRandomStrategy:
    def test_no_replacement_unique(self, tmp_path):
        r = _make_registry(tmp_path, [(f"a{i}", "ppo", i) for i in range(10)])
        s = _sampler(r, seed=42)
        picks = s.sample(n=4, strategy="random")
        ids = [c.id for c in picks]
        assert len(set(ids)) == 4

    def test_with_replacement_can_repeat(self, tmp_path):
        r = _make_registry(tmp_path, [("a", "ppo", 0), ("b", "ppo", 0)])
        s = _sampler(r, seed=0)
        picks = s.sample(n=10, strategy="random", with_replacement=True)
        assert len(picks) == 10

    def test_deterministic_with_seed(self, tmp_path):
        r = _make_registry(tmp_path, [(f"a{i}", "ppo", 0) for i in range(8)])
        s1 = _sampler(r, seed=123)
        s2 = _sampler(r, seed=123)
        a = [c.id for c in s1.sample(n=3, strategy="random")]
        b = [c.id for c in s2.sample(n=3, strategy="random")]
        assert a == b

    def test_kind_filter(self, tmp_path):
        r = _make_registry(tmp_path, [
            ("p1", "ppo", 0), ("r1", "rule", 0), ("p2", "ppo", 0),
        ])
        s = _sampler(r, seed=1)
        picks = s.sample(n=5, strategy="random", kind="ppo")
        assert all(c.kind == "ppo" for c in picks)


class TestWeightedRecency:
    def test_favors_newer_in_expectation(self, tmp_path):
        # one ancient, one fresh
        r = _make_registry(tmp_path, [("old", "ppo", 0), ("new", "ppo", 99)])
        # With weights 1 vs 100, new should dominate over many trials
        counts = {"old": 0, "new": 0}
        for seed in range(200):
            s = _sampler(r, seed=seed)
            picked = s.sample(n=1, strategy="weighted_recency", with_replacement=True)
            counts[picked[0].id] += 1
        assert counts["new"] > counts["old"] * 5

    def test_no_replacement_yields_unique(self, tmp_path):
        r = _make_registry(tmp_path, [(f"a{i}", "ppo", i) for i in range(6)])
        s = _sampler(r, seed=7)
        picks = s.sample(n=4, strategy="weighted_recency")
        assert len({c.id for c in picks}) == 4

    def test_returns_pool_if_n_exceeds(self, tmp_path):
        r = _make_registry(tmp_path, [("a", "ppo", 0), ("b", "ppo", 5)])
        s = _sampler(r, seed=0)
        picks = s.sample(n=10, strategy="weighted_recency")
        assert len(picks) == 2
        assert {c.id for c in picks} == {"a", "b"}


class TestFixedStrategy:
    def test_returns_in_order(self, tmp_path):
        r = _make_registry(tmp_path, [
            ("a", "ppo", 0), ("b", "ppo", 0), ("c", "ppo", 0),
        ])
        s = _sampler(r)
        picks = s.sample(n=3, strategy="fixed", ids=["c", "a"])
        assert [c.id for c in picks] == ["c", "a"]

    def test_missing_id_raises(self, tmp_path):
        r = _make_registry(tmp_path, [("a", "ppo", 0)])
        s = _sampler(r)
        with pytest.raises(KeyError):
            s.sample(n=1, strategy="fixed", ids=["ghost"])

    def test_exclude_filters_out(self, tmp_path):
        r = _make_registry(tmp_path, [("a", "ppo", 0), ("b", "ppo", 0)])
        s = _sampler(r)
        picks = s.sample(n=2, strategy="fixed", ids=["a", "b"], exclude_ids=["a"])
        assert [c.id for c in picks] == ["b"]


class TestErrors:
    def test_unknown_strategy(self, tmp_path):
        r = _make_registry(tmp_path, [])
        s = _sampler(r)
        with pytest.raises(ValueError):
            s.sample(n=1, strategy="bogus")

    def test_negative_n(self, tmp_path):
        r = _make_registry(tmp_path, [])
        s = _sampler(r)
        with pytest.raises(ValueError):
            s.sample(n=-1, strategy="latest")

    def test_empty_registry_returns_empty(self, tmp_path):
        r = _make_registry(tmp_path, [])
        s = _sampler(r)
        for strat in ("latest", "random", "weighted_recency"):
            assert s.sample(n=3, strategy=strat) == []

    def test_zero_n_returns_empty(self, tmp_path):
        r = _make_registry(tmp_path, [("a", "ppo", 0)])
        s = _sampler(r)
        assert s.sample(n=0, strategy="latest") == []
