"""Tests for AgentRegistry + AgentCard persistence and aggregation."""

import json

import pytest

from src.training.agent_card import AgentCard
from src.training.agent_registry import AgentRegistry


def _card(aid: str, *, kind: str = "ppo", generation: int = 0, **kw) -> AgentCard:
    return AgentCard(id=aid, name=aid, kind=kind, generation=generation, **kw)


class TestAgentCardRoundtrip:
    def test_to_from_dict_preserves_fields(self):
        c = _card("a1", path="models/a1.zip", generation=3, parent_id="a0")
        c.behavior_stats = {"hands_observed": 50, "vpip": 0.25}
        c.matchup_history = {"obs": {"hands_played": 10, "total_profit": 5.0}}
        restored = AgentCard.from_dict(c.to_dict())
        assert restored == c


class TestRegistryBasics:
    def test_empty_registry_when_file_missing(self, tmp_path):
        r = AgentRegistry(path=str(tmp_path / "registry.json"))
        assert r.all() == []
        assert r.next_generation() == 0

    def test_register_and_get_roundtrip(self, tmp_path):
        r = AgentRegistry(path=str(tmp_path / "registry.json"))
        c = _card("a1")
        r.register(c)
        assert r.get("a1") is c
        assert r.all() == [c]

    def test_register_duplicate_raises(self, tmp_path):
        r = AgentRegistry(path=str(tmp_path / "registry.json"))
        r.register(_card("a1"))
        with pytest.raises(ValueError):
            r.register(_card("a1"))

    def test_persist_and_reload(self, tmp_path):
        path = str(tmp_path / "registry.json")
        r1 = AgentRegistry(path=path)
        r1.register(_card("a1", generation=2, path="models/a1.zip"))
        r1.register(_card("a2", kind="rule", generation=0))

        r2 = AgentRegistry(path=path)
        assert {c.id for c in r2.all()} == {"a1", "a2"}
        a1 = r2.get("a1")
        assert a1.generation == 2
        assert a1.path == "models/a1.zip"
        assert r2.get("a2").kind == "rule"

    def test_save_writes_schema_version(self, tmp_path):
        path = tmp_path / "registry.json"
        r = AgentRegistry(path=str(path))
        r.register(_card("a1"))
        data = json.loads(path.read_text())
        assert data["schema_version"] == AgentRegistry.SCHEMA_VERSION
        assert "saved_at" in data
        assert "a1" in data["agents"]


class TestLatestAndGeneration:
    def test_latest_orders_by_generation_then_created_at(self, tmp_path):
        r = AgentRegistry(path=str(tmp_path / "registry.json"))
        r.register(_card("g0", generation=0))
        r.register(_card("g2", generation=2))
        r.register(_card("g1", generation=1))
        ids = [c.id for c in r.latest(n=3)]
        assert ids == ["g2", "g1", "g0"]

    def test_latest_filters_by_kind(self, tmp_path):
        r = AgentRegistry(path=str(tmp_path / "registry.json"))
        r.register(_card("ppo1", kind="ppo", generation=1))
        r.register(_card("rule1", kind="rule", generation=5))
        r.register(_card("ppo2", kind="ppo", generation=2))
        ids = [c.id for c in r.latest(n=5, kind="ppo")]
        assert ids == ["ppo2", "ppo1"]

    def test_latest_caps_at_n(self, tmp_path):
        r = AgentRegistry(path=str(tmp_path / "registry.json"))
        for i in range(5):
            r.register(_card(f"a{i}", generation=i))
        assert len(r.latest(n=2)) == 2

    def test_next_generation(self, tmp_path):
        r = AgentRegistry(path=str(tmp_path / "registry.json"))
        assert r.next_generation() == 0
        r.register(_card("a", generation=0))
        r.register(_card("b", generation=3))
        assert r.next_generation() == 4


class TestUpdateMatchup:
    def test_accumulates_on_opponent_card(self, tmp_path):
        r = AgentRegistry(path=str(tmp_path / "registry.json"))
        r.register(_card("learner"))
        r.register(_card("opp"))

        r.update_matchup("learner", "opp", hands=100, profit=50.0, timestep=1000)
        entry = r.get("opp").matchup_history["learner"]
        assert entry["hands_played"] == 100
        assert entry["total_profit"] == 50.0
        assert entry["avg_profit"] == pytest.approx(0.5)
        assert entry["last_seen_timestep"] == 1000

    def test_accumulates_across_calls(self, tmp_path):
        r = AgentRegistry(path=str(tmp_path / "registry.json"))
        r.register(_card("learner"))
        r.register(_card("opp"))

        r.update_matchup("learner", "opp", hands=100, profit=50.0, timestep=1000)
        r.update_matchup("learner", "opp", hands=50, profit=-10.0, timestep=2000)
        entry = r.get("opp").matchup_history["learner"]
        assert entry["hands_played"] == 150
        assert entry["total_profit"] == pytest.approx(40.0)
        assert entry["avg_profit"] == pytest.approx(40.0 / 150)
        assert entry["last_seen_timestep"] == 2000

    def test_unknown_opponent_raises(self, tmp_path):
        r = AgentRegistry(path=str(tmp_path / "registry.json"))
        r.register(_card("learner"))
        with pytest.raises(KeyError):
            r.update_matchup("learner", "ghost", hands=1, profit=1.0, timestep=1)

    def test_multiple_observers_kept_separate(self, tmp_path):
        r = AgentRegistry(path=str(tmp_path / "registry.json"))
        r.register(_card("a"))
        r.register(_card("b"))
        r.register(_card("opp"))
        r.update_matchup("a", "opp", hands=10, profit=5.0, timestep=1)
        r.update_matchup("b", "opp", hands=20, profit=-2.0, timestep=2)
        history = r.get("opp").matchup_history
        assert history["a"]["hands_played"] == 10
        assert history["b"]["hands_played"] == 20

    def test_persists_across_reload(self, tmp_path):
        path = str(tmp_path / "registry.json")
        r1 = AgentRegistry(path=path)
        r1.register(_card("learner"))
        r1.register(_card("opp"))
        r1.update_matchup("learner", "opp", hands=10, profit=5.0, timestep=100)

        r2 = AgentRegistry(path=path)
        entry = r2.get("opp").matchup_history["learner"]
        assert entry["hands_played"] == 10
        assert entry["total_profit"] == 5.0


class TestUpdateBehaviorStats:
    def test_first_observation_sets_values(self, tmp_path):
        r = AgentRegistry(path=str(tmp_path / "registry.json"))
        r.register(_card("a"))
        r.update_behavior_stats("a", {"vpip": 0.30, "pfr": 0.20}, hands_observed=100)
        stats = r.get("a").behavior_stats
        assert stats["hands_observed"] == 100
        assert stats["vpip"] == pytest.approx(0.30)
        assert stats["pfr"] == pytest.approx(0.20)

    def test_weighted_average_merging(self, tmp_path):
        r = AgentRegistry(path=str(tmp_path / "registry.json"))
        r.register(_card("a"))
        r.update_behavior_stats("a", {"vpip": 0.40}, hands_observed=100)
        r.update_behavior_stats("a", {"vpip": 0.20}, hands_observed=300)
        stats = r.get("a").behavior_stats
        assert stats["hands_observed"] == 400
        # (0.40*100 + 0.20*300) / 400 = (40 + 60)/400 = 0.25
        assert stats["vpip"] == pytest.approx(0.25)

    def test_zero_hands_is_noop_value_wise(self, tmp_path):
        r = AgentRegistry(path=str(tmp_path / "registry.json"))
        r.register(_card("a"))
        r.update_behavior_stats("a", {}, hands_observed=0)
        stats = r.get("a").behavior_stats
        assert stats == {"hands_observed": 0}

    def test_new_key_treated_as_zero_in_old(self, tmp_path):
        r = AgentRegistry(path=str(tmp_path / "registry.json"))
        r.register(_card("a"))
        r.update_behavior_stats("a", {"vpip": 0.40}, hands_observed=100)
        r.update_behavior_stats("a", {"agg": 0.5}, hands_observed=100)
        stats = r.get("a").behavior_stats
        # vpip in second batch absent -> 0; (0.40*100 + 0*100)/200 = 0.20
        assert stats["vpip"] == pytest.approx(0.20)
        # agg absent in first batch; (0*100 + 0.5*100)/200 = 0.25
        assert stats["agg"] == pytest.approx(0.25)
        assert stats["hands_observed"] == 200

    def test_unknown_agent_raises(self, tmp_path):
        r = AgentRegistry(path=str(tmp_path / "registry.json"))
        with pytest.raises(KeyError):
            r.update_behavior_stats("ghost", {"vpip": 0.3}, hands_observed=10)

    def test_persists_across_reload(self, tmp_path):
        path = str(tmp_path / "registry.json")
        r1 = AgentRegistry(path=path)
        r1.register(_card("a"))
        r1.update_behavior_stats("a", {"vpip": 0.3}, hands_observed=50)

        r2 = AgentRegistry(path=path)
        stats = r2.get("a").behavior_stats
        assert stats["hands_observed"] == 50
        assert stats["vpip"] == pytest.approx(0.3)


class TestAtomicSave:
    def test_no_tmp_file_left_after_save(self, tmp_path):
        path = tmp_path / "registry.json"
        r = AgentRegistry(path=str(path))
        r.register(_card("a"))
        leftovers = list(tmp_path.glob("*.tmp"))
        assert leftovers == []
        assert path.exists()
