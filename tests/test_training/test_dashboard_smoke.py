"""End-to-end smoke test for scripts/dashboard.py.

Uses streamlit's in-process `AppTest` runner to confirm the dashboard
script renders without exceptions when handed a synthetic registry +
metrics fixture. We don't assert on visual content — just that the
plotting code paths in all four tabs execute cleanly.
"""

import json
import os
from pathlib import Path

import pytest

from src.training.agent_card import AgentCard
from src.training.agent_registry import AgentRegistry

streamlit_testing = pytest.importorskip("streamlit.testing.v1")
AppTest = streamlit_testing.AppTest

DASHBOARD = str(Path(__file__).resolve().parents[2] / "scripts" / "dashboard.py")


def _build_fixture(tmp_path: Path) -> tuple[str, str]:
    reg = AgentRegistry(str(tmp_path / "registry.json"))
    reg.register(AgentCard(id="rule_call_v0", name="c", kind="call"))
    reg.register(AgentCard(
        id="gen0", name="gen0", kind="ppo", generation=0, total_timesteps=5000,
        trained_against_ids=["rule_call_v0"],
    ))
    reg.register(AgentCard(
        id="gen1", name="gen1", kind="ppo", generation=1, parent_id="gen0",
        total_timesteps=5000, trained_against_ids=["gen0"],
        eval_stats={"mbb_per_100": 8.2, "passed": True, "hands": 1000},
    ))
    reg.update_matchup(
        observer_id="gen1", opponent_id="gen0",
        hands=200, profit=4000, timestep=5000,
    )

    mdir = tmp_path / "metrics"
    for run in ("gen0", "gen1"):
        rd = mdir / run
        rd.mkdir(parents=True)
        (rd / "metrics.json").write_text(json.dumps({
            "timesteps": [1000, 3000, 5000],
            "rewards": [-0.1, 0.1, 0.3],
            "avg_reward_100": [-0.05, 0.15, 0.3],
            "win_rate": [0.3, 0.4, 0.45],
            "policy_loss": [0.5, 0.3, 0.15],
            "value_loss": [0.8, 0.5, 0.2],
            "entropy": [1.0, 0.9, 0.7],
            "fold_rate": [0.5, 0.3, 0.25],
            "raise_rate": [0.3, 0.4, 0.5],
            "all_in_rate": [0.05, 0.05, 0.05],
        }))
        (rd / "value_calibration.json").write_text(json.dumps({
            "gamma": 0.99,
            "pairs": [
                {"timestep": 1000 + i * 200,
                 "value": 0.1 + i * 0.01,
                 "actual_return": 0.05 + i * 0.012}
                for i in range(20)
            ],
        }))
        (rd / "street_breakdown.json").write_text(json.dumps({
            "timesteps": [1000, 3000, 5000],
            "distributions": [
                {s: {"fold": 0.2, "call": 0.3, "raise": 0.45, "all_in": 0.05, "count": 100}
                 for s in ("preflop", "flop", "turn", "river")},
                {s: {"fold": 0.15, "call": 0.3, "raise": 0.5, "all_in": 0.05, "count": 100}
                 for s in ("preflop", "flop", "turn", "river")},
                {s: {"fold": 0.1, "call": 0.3, "raise": 0.55, "all_in": 0.05, "count": 100}
                 for s in ("preflop", "flop", "turn", "river")},
            ],
        }))
    return str(tmp_path / "registry.json"), str(mdir)


def test_dashboard_renders_all_tabs_without_exception(tmp_path):
    registry_path, metrics_dir = _build_fixture(tmp_path)

    at = AppTest.from_file(DASHBOARD, default_timeout=30)
    at.run()
    # First run uses default sidebar inputs (which point at "models/registry.json").
    # Override and re-run so the fixture data is what gets rendered.
    at.sidebar.text_input[0].set_value(registry_path)
    at.sidebar.text_input[1].set_value(metrics_dir)
    at.run()

    assert len(at.exception) == 0, f"dashboard raised: {at.exception}"
    assert len(at.error) == 0, f"dashboard surfaced errors: {at.error}"
    assert len(at.tabs) == 4, f"expected 4 tabs, got {len(at.tabs)}"

    headers = [h.value for h in at.header]
    assert any("Registry overview" in h for h in headers)
    assert any("Training curves" in h for h in headers)
    assert any("Per-street action breakdown" in h for h in headers)
    assert any("Critic calibration" in h for h in headers)


def test_dashboard_shows_error_when_registry_missing(tmp_path):
    """If no registry exists, the dashboard should stop cleanly with an
    error — NOT raise."""
    at = AppTest.from_file(DASHBOARD, default_timeout=30)
    at.run()
    at.sidebar.text_input[0].set_value(str(tmp_path / "does_not_exist.json"))
    at.run()
    assert len(at.exception) == 0
    assert len(at.error) >= 1, "missing registry should surface an st.error"
