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


class TestStratifiedByAnchor:
    """``stratified_by_anchor`` requires anchor cards in the registry.
    Cards are quantised to nearest anchor, then a stratum is picked
    uniformly and a card drawn from it."""

    def _seed_anchors(self, registry, *, with_anchor_stats=True):
        from src.agents.anchors import ALL_ANCHORS
        for cls in ALL_ANCHORS:
            stats = dict(cls.CANONICAL_STATS) if with_anchor_stats else {"hands_observed": 0}
            card = AgentCard(
                id=cls.ANCHOR_ID,
                name=cls.ARCHETYPE,
                kind="anchor",
                behavior_stats=stats,
            )
            registry.register(card)

    def test_missing_anchors_raises(self, tmp_path):
        r = _make_registry(tmp_path, [("a", "ppo", 0)])
        s = _sampler(r)
        with pytest.raises(RuntimeError, match="anchor"):
            s.sample(n=1, strategy="stratified_by_anchor")

    def test_assigns_card_to_nearest_anchor(self, tmp_path):
        r = AgentRegistry(path=str(tmp_path / "registry.json"))
        self._seed_anchors(r)
        # PPO card with shover-ish stats — should fall in the shover stratum.
        shover_like = AgentCard(
            id="ppo_shover_like", name="shover_like", kind="ppo", generation=1,
            behavior_stats={
                "hands_observed": 500,
                "vpip": 0.98, "pfr": 0.94, "af": 40.0,
                "three_bet_percent": 0.85, "cbet_percent": 0.0,
                "fold_to_cbet_percent": 0.0,
                "went_to_showdown_percent": 0.60, "win_at_showdown_percent": 0.45,
                "wwsf_percent": 0.78,
            },
        )
        r.register(shover_like)
        s = _sampler(r)
        strata = s.assign_strata(r.all())
        assert "anchor_shover_v0" in strata
        ids_in_shover = {c.id for c in strata["anchor_shover_v0"]}
        assert "ppo_shover_like" in ids_in_shover

    def test_unobserved_cards_get_unstratified_bucket(self, tmp_path):
        r = AgentRegistry(path=str(tmp_path / "registry.json"))
        self._seed_anchors(r)
        r.register(AgentCard(
            id="fresh", name="fresh", kind="ppo", generation=1,
            behavior_stats={"hands_observed": 0},
        ))
        s = _sampler(r)
        strata = s.assign_strata(r.all())
        assert "unstratified" in strata
        assert {c.id for c in strata["unstratified"]} == {"fresh"}

    def test_sample_returns_cards_from_strata(self, tmp_path):
        r = AgentRegistry(path=str(tmp_path / "registry.json"))
        self._seed_anchors(r)
        s = _sampler(r, seed=0)
        picks = s.sample(n=3, strategy="stratified_by_anchor")
        assert len(picks) == 3
        assert len({c.id for c in picks}) == 3  # no replacement

    def test_sample_covers_multiple_strata_with_replacement(self, tmp_path):
        r = AgentRegistry(path=str(tmp_path / "registry.json"))
        self._seed_anchors(r)
        s = _sampler(r, seed=0)
        # Over many draws, every stratum should appear at least once.
        from src.agents.anchors import ALL_ANCHORS
        expected = len(ALL_ANCHORS)
        seen_strata = set()
        for _ in range(200):
            pick = s.sample(n=1, strategy="stratified_by_anchor", with_replacement=True)[0]
            seen_strata.add(pick.id)  # ids are anchor ids since only anchors exist
        # All seeded anchors should appear at least once within 200 draws.
        assert len(seen_strata) == expected

    def test_exclude_ids_removes_from_stratum(self, tmp_path):
        r = AgentRegistry(path=str(tmp_path / "registry.json"))
        self._seed_anchors(r)
        s = _sampler(r, seed=0)
        excluded = "anchor_shover_v0"
        for _ in range(20):
            picks = s.sample(
                n=5, strategy="stratified_by_anchor",
                exclude_ids=[excluded], with_replacement=True,
            )
            assert all(c.id != excluded for c in picks)

    def _add_shover_like_ppo(self, r):
        r.register(AgentCard(
            id="ppo_shover_like", name="shover_like", kind="ppo", generation=1,
            behavior_stats={
                "hands_observed": 500,
                "vpip": 0.98, "pfr": 0.94, "af": 40.0,
                "three_bet_percent": 0.85, "cbet_percent": 0.0,
                "fold_to_cbet_percent": 0.0,
                "went_to_showdown_percent": 0.60, "win_at_showdown_percent": 0.45,
                "wwsf_percent": 0.78,
            },
        ))

    def test_anchors_seed_only_prefers_ppo_in_occupied_stratum(self, tmp_path):
        # When a stratum holds a PPO occupant, anchors_seed_only must drop
        # the scripted anchor so the learner trains against the PPO self,
        # never the hand-coded archetype.
        r = AgentRegistry(path=str(tmp_path / "registry.json"))
        self._seed_anchors(r)
        self._add_shover_like_ppo(r)
        s = _sampler(r, seed=0)
        for _ in range(50):
            pick = s.sample(
                n=1, strategy="stratified_by_anchor",
                anchors_seed_only=True, with_replacement=True,
            )[0]
            # The shover anchor must never be drawn now that a PPO bot
            # occupies its bucket; every other anchor still seeds its own
            # (empty-of-PPO) bucket.
            assert pick.id != "anchor_shover_v0"

    def test_anchors_seed_only_keeps_anchor_in_empty_stratum(self, tmp_path):
        # A stratum with no PPO occupant keeps its anchor (the seeding case).
        r = AgentRegistry(path=str(tmp_path / "registry.json"))
        self._seed_anchors(r)
        self._add_shover_like_ppo(r)
        s = _sampler(r, seed=0)
        seen = set()
        for _ in range(300):
            pick = s.sample(
                n=1, strategy="stratified_by_anchor",
                anchors_seed_only=True, with_replacement=True,
            )[0]
            seen.add(pick.id)
        # tight_passive has no PPO occupant -> its anchor should still appear;
        # shover's anchor should be absent (replaced by the PPO occupant).
        assert "anchor_tight_passive_v0" in seen
        assert "ppo_shover_like" in seen
        assert "anchor_shover_v0" not in seen


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
