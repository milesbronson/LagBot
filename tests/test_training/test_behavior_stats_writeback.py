"""Lock the train.py write-back into AgentRegistry.update_behavior_stats.
The helper now takes card-keyed snapshots so it works under per-episode
opponent rotation — multiple cards can pass through the same seat in one
run, so we cannot key by seat id."""

from src.training.agent_card import AgentCard
from src.training.agent_registry import AgentRegistry
from train import _fold_behavior_stats_into_registry


def _make_snapshot(**overrides):
    """Default-valued stat dict so individual tests only specify what
    they care about."""
    base = {
        "hands_observed": 30, "vpip": 0.4, "pfr": 0.2, "af": 1.5,
        "three_bet_percent": 0.1, "cbet_percent": 0.5,
        "fold_to_cbet_percent": 0.5, "went_to_showdown_percent": 0.25,
        "win_at_showdown_percent": 0.5, "wwsf_percent": 0.45,
        "fold_to_3bet_after_raise_percent": 0.6, "squeeze_percent": 0.05,
    }
    base.update(overrides)
    return base


def test_writes_ppo_opponents_and_skips_fixtures(tmp_path):
    reg = AgentRegistry(path=str(tmp_path / "reg.json"))
    reg.register(AgentCard(id="ppo_a", name="ppo_a", kind="ppo", path="x.zip"))
    reg.register(AgentCard(id="fixture_call", name="c", kind="call"))

    _fold_behavior_stats_into_registry(reg, {
        "ppo_a": _make_snapshot(vpip=0.4, pfr=0.2),
        "fixture_call": _make_snapshot(vpip=0.9, pfr=0.0),
    })

    ppo = reg.get("ppo_a").behavior_stats
    assert ppo["hands_observed"] == 30
    assert ppo["vpip"] == 0.4
    assert ppo["pfr"] == 0.2

    fixture = reg.get("fixture_call").behavior_stats
    assert fixture == {"hands_observed": 0}


def test_zero_hands_observed_is_skipped(tmp_path):
    reg = AgentRegistry(path=str(tmp_path / "reg.json"))
    reg.register(AgentCard(id="ppo_a", name="ppo_a", kind="ppo", path="x.zip"))

    _fold_behavior_stats_into_registry(reg, {
        "ppo_a": {"hands_observed": 0, "vpip": 0.5},
    })
    assert reg.get("ppo_a").behavior_stats == {"hands_observed": 0}


def test_unknown_card_id_is_silently_skipped(tmp_path):
    """Snapshot keys that aren't in the registry (e.g. ad-hoc agent ids
    used during testing) should not blow up the write-back."""
    reg = AgentRegistry(path=str(tmp_path / "reg.json"))
    reg.register(AgentCard(id="ppo_a", name="ppo_a", kind="ppo", path="x.zip"))

    _fold_behavior_stats_into_registry(reg, {
        "ppo_a": _make_snapshot(vpip=0.3),
        "stranger": _make_snapshot(vpip=0.9),
    })

    assert reg.get("ppo_a").behavior_stats["vpip"] == 0.3
    assert reg.get("stranger") is None


def test_repeated_runs_accumulate_weighted_average(tmp_path):
    """Two runs of equal weight should leave the registry at the
    arithmetic mean of the two snapshots."""
    reg = AgentRegistry(path=str(tmp_path / "reg.json"))
    reg.register(AgentCard(id="ppo_a", name="ppo_a", kind="ppo", path="x.zip"))

    _fold_behavior_stats_into_registry(
        reg, {"ppo_a": {"hands_observed": 50, "vpip": 0.2}},
    )
    _fold_behavior_stats_into_registry(
        reg, {"ppo_a": {"hands_observed": 50, "vpip": 0.8}},
    )

    assert reg.get("ppo_a").behavior_stats["hands_observed"] == 100
    assert reg.get("ppo_a").behavior_stats["vpip"] == 0.5
