"""Tests for src/training/registry_report.py.

Builds a synthetic registry + metrics dir in tmp, then asserts the
rollup table and matchup matrix report what we expect."""

import json
import os
from pathlib import Path

import pytest

from src.training.agent_card import AgentCard
from src.training.agent_registry import AgentRegistry
from src.training.registry_report import (
    load_run_metrics,
    load_street_breakdown,
    load_value_calibration,
    matchup_matrix,
    print_summary,
    summary_table,
)


@pytest.fixture
def populated(tmp_path):
    """A registry with 3 generations of PPOs (gen0 → gen1 → gen2) plus
    one rule fixture, paired with a metrics/<run>/metrics.json for each."""
    reg = AgentRegistry(str(tmp_path / "registry.json"))
    reg.register(AgentCard(id="rule_call_v0", name="CallAgent", kind="call"))

    reg.register(AgentCard(
        id="gen0", name="gen0", kind="ppo",
        generation=0, total_timesteps=10_000,
        trained_against_ids=["rule_call_v0"],
    ))
    reg.register(AgentCard(
        id="gen1", name="gen1", kind="ppo",
        generation=1, parent_id="gen0", total_timesteps=10_000,
        trained_against_ids=["gen0"],
        eval_stats={"mbb_per_100": 12.5, "passed": True, "hands": 1000},
    ))
    reg.register(AgentCard(
        id="gen2", name="gen2", kind="ppo",
        generation=2, parent_id="gen1", total_timesteps=10_000,
        trained_against_ids=["gen1"],
        eval_stats={"mbb_per_100": -3.0, "passed": False, "hands": 1000},
    ))

    # gen1 played against gen0 and ended up +200/hand on average
    reg.update_matchup(
        observer_id="gen1", opponent_id="gen0",
        hands=500, profit=100_000, timestep=10_000,
    )
    # gen2 played against gen1 and lost on average
    reg.update_matchup(
        observer_id="gen2", opponent_id="gen1",
        hands=500, profit=-25_000, timestep=10_000,
    )

    metrics_dir = tmp_path / "metrics"
    for run in ("gen0", "gen1", "gen2"):
        d = metrics_dir / run
        d.mkdir(parents=True)
        with open(d / "metrics.json", "w") as f:
            json.dump({
                "timesteps": [1000, 5000, 10000],
                "rewards": [-0.1, 0.2, 0.5],
                "avg_reward_100": [-0.05, 0.15, 0.4],
                "win_rate": [0.3, 0.45, 0.55],
            }, f)

    return reg, str(metrics_dir)


def test_summary_table_returns_rows_sorted_by_generation(populated):
    reg, mdir = populated
    rows = summary_table(reg, metrics_dir=mdir)
    assert [r["id"] for r in rows] == ["gen0", "gen1", "gen2"]
    assert [r["generation"] for r in rows] == [0, 1, 2]


def test_summary_table_excludes_non_ppo_kinds_by_default(populated):
    reg, mdir = populated
    rows = summary_table(reg, metrics_dir=mdir)
    assert all(r["id"] != "rule_call_v0" for r in rows)


def test_summary_table_carries_eval_stats(populated):
    reg, mdir = populated
    rows = summary_table(reg, metrics_dir=mdir)
    by_id = {r["id"]: r for r in rows}
    assert by_id["gen0"]["eval_mbb_per_100"] is None
    assert by_id["gen1"]["eval_mbb_per_100"] == 12.5
    assert by_id["gen1"]["eval_passed"] is True
    assert by_id["gen2"]["eval_passed"] is False


def test_summary_table_pulls_last_metrics_from_per_run_file(populated):
    reg, mdir = populated
    rows = summary_table(reg, metrics_dir=mdir)
    by_id = {r["id"]: r for r in rows}
    assert by_id["gen2"]["last_avg_reward_100"] == 0.4
    assert by_id["gen2"]["last_win_rate"] == 0.55


def test_summary_table_handles_missing_metrics_dir(tmp_path):
    reg = AgentRegistry(str(tmp_path / "registry.json"))
    reg.register(AgentCard(id="x", name="x", kind="ppo", generation=0))
    rows = summary_table(reg, metrics_dir=str(tmp_path / "nowhere"))
    assert len(rows) == 1
    assert rows[0]["last_avg_reward_100"] is None
    assert rows[0]["last_win_rate"] is None


def test_matchup_matrix_lists_only_ppo_ids(populated):
    reg, _ = populated
    ids, _matrix = matchup_matrix(reg)
    assert ids == ["gen0", "gen1", "gen2"]
    assert "rule_call_v0" not in ids


def test_matchup_matrix_reflects_recorded_profits(populated):
    reg, _ = populated
    ids, matrix = matchup_matrix(reg)
    by = {(ids[i], ids[j]): matrix[i][j]
          for i in range(len(ids)) for j in range(len(ids))}
    # gen1 observed +100_000 over 500 hands vs gen0 → avg 200
    assert by[("gen1", "gen0")] == pytest.approx(200.0)
    # gen2 observed -25_000 over 500 hands vs gen1 → avg -50
    assert by[("gen2", "gen1")] == pytest.approx(-50.0)
    # nothing recorded for the inverse direction
    assert by[("gen0", "gen1")] is None
    assert by[("gen1", "gen2")] is None


def test_print_summary_writes_header_and_rows(populated, capsys):
    reg, mdir = populated
    rows = summary_table(reg, metrics_dir=mdir)
    print_summary(rows)
    out = capsys.readouterr().out
    assert "gen" in out and "mbb/100" in out
    assert "gen1" in out
    # gen0 has no eval stats → printed as a dash
    assert " - " in out


def test_print_summary_handles_empty_rows(capsys):
    print_summary([])
    out = capsys.readouterr().out
    assert "no PPO cards" in out


def test_load_value_calibration_reads_pairs(tmp_path):
    run = tmp_path / "run_x"
    run.mkdir()
    payload = {"gamma": 0.99, "pairs": [
        {"timestep": 1000, "value": 0.4, "actual_return": 0.5},
    ]}
    (run / "value_calibration.json").write_text(json.dumps(payload))
    data = load_value_calibration("run_x", str(tmp_path))
    assert data["gamma"] == 0.99
    assert len(data["pairs"]) == 1
    assert data["pairs"][0]["value"] == 0.4


def test_load_value_calibration_returns_none_if_missing(tmp_path):
    assert load_value_calibration("nope", str(tmp_path)) is None


def test_load_street_breakdown_reads_distributions(tmp_path):
    run = tmp_path / "run_y"
    run.mkdir()
    payload = {
        "timesteps": [1000, 2000],
        "distributions": [
            {"preflop": {"fold": 0.1, "call": 0.2, "raise": 0.6, "all_in": 0.1, "count": 50},
             "flop": {"fold": 0, "call": 0, "raise": 0, "all_in": 0, "count": 0},
             "turn": {"fold": 0, "call": 0, "raise": 0, "all_in": 0, "count": 0},
             "river": {"fold": 0, "call": 0, "raise": 0, "all_in": 0, "count": 0}},
            {"preflop": {"fold": 0.05, "call": 0.15, "raise": 0.7, "all_in": 0.1, "count": 100},
             "flop": {"fold": 0, "call": 0, "raise": 0, "all_in": 0, "count": 0},
             "turn": {"fold": 0, "call": 0, "raise": 0, "all_in": 0, "count": 0},
             "river": {"fold": 0, "call": 0, "raise": 0, "all_in": 0, "count": 0}},
        ],
    }
    (run / "street_breakdown.json").write_text(json.dumps(payload))
    data = load_street_breakdown("run_y", str(tmp_path))
    assert data["timesteps"] == [1000, 2000]
    assert data["distributions"][1]["preflop"]["raise"] == 0.7


def test_load_run_metrics_returns_none_if_missing(tmp_path):
    assert load_run_metrics("nope", str(tmp_path)) is None
