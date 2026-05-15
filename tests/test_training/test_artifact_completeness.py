"""Artifact-completeness contract for train.py.

Every observability tab in the Streamlit dashboard reads from one of
these files. If a future change drops any of them, the dashboard tab
goes blank silently. This test runs a tiny end-to-end training cycle
and asserts that every expected artifact lands on disk with the
expected shape.

The run is intentionally TINY (a few thousand steps) so it's still
acceptable in the suite, but we run it as a slow/integration test."""

import json
import os
from pathlib import Path

import pytest
import yaml


# Tiny but real training cycle. Marked slow because it actually invokes
# SB3 PPO — most other tests stub the model.
pytestmark = pytest.mark.slow


@pytest.fixture
def tiny_config(tmp_path):
    cfg = {
        "environment": {
            "num_players": 3,
            "starting_stack": 1000,
            "small_blind": 5,
            "big_blind": 10,
            "min_raise_multiplier": 2.0,
            "rake_enabled": False,
            "rake_percent": 0.0,
            "rake_cap": 0,
            "reset_stacks_every_n_timesteps": 1000,
        },
        "training": {
            "total_timesteps": 2048,
            "learning_rate": 0.001,
            "n_steps": 512,
            "batch_size": 64,
            "n_epochs": 2,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.05,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
        },
        "opponents": {"strategy": "latest", "kind": None, "fixed_ids": []},
        "continuation": {"resume_from": None, "generations": 1},
        "eval_gate": {"enabled": False, "num_hands": 100,
                      "threshold_mbb_per_100": 0.0, "seed": 0},
        "ppo": {"policy": "MlpPolicy", "verbose": 0,
                "tensorboard_log": str(tmp_path / "logs")},
        "logging": {
            "log_dir": str(tmp_path / "logs"),
            "save_frequency": 5000,
            "model_dir": str(tmp_path / "models"),
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_training_cycle_produces_all_observability_artifacts(tiny_config, tmp_path, monkeypatch):
    """Run a tiny generation and check every file the dashboard reads."""
    # Train into a tmp metrics/models layout so we don't touch the real repo.
    monkeypatch.chdir(tmp_path)
    # Re-point registry + per-run metrics into tmp.
    (tmp_path / "metrics").mkdir(exist_ok=True)
    (tmp_path / "models").mkdir(exist_ok=True)
    # train.py opens AgentRegistry at the default "models/registry.json"
    # which is now relative to tmp_path thanks to monkeypatch.chdir.

    from train import train as train_fn

    train_fn(str(tiny_config), run_name="art_smoke")

    run_dir = tmp_path / "metrics" / "art_smoke"
    expected_files = [
        "metrics.json",
        "action_history.json",
        "street_breakdown.json",
        "value_calibration.json",
        "opponent_profits.json",
    ]
    for fname in expected_files:
        path = run_dir / fname
        assert path.exists(), f"missing observability artifact: {path}"
        data = json.loads(path.read_text())
        assert isinstance(data, dict), f"{fname} not a JSON object"

    # Registry got a card.
    reg_path = tmp_path / "models" / "registry.json"
    assert reg_path.exists()
    reg = json.loads(reg_path.read_text())
    assert "art_smoke" in reg["agents"], "trained card was not registered"

    # Street breakdown should have at least one snapshot with the canonical 4 streets.
    sb = json.loads((run_dir / "street_breakdown.json").read_text())
    assert sb["timesteps"], "no street_breakdown snapshots recorded"
    sample = sb["distributions"][0]
    assert set(sample.keys()) >= {"preflop", "flop", "turn", "river"}

    # Value calibration should have at least one (V, G_0) pair.
    vc = json.loads((run_dir / "value_calibration.json").read_text())
    assert vc["pairs"], "no value-calibration pairs recorded"
    p = vc["pairs"][0]
    assert {"timestep", "value", "actual_return"} <= set(p.keys())

    # Collapsed action rates in metrics.json must not be silently stuck at 0
    # — that would mean MetricsCallback dropped them on the floor before
    # handing agent_stats to TrainingMetrics.record_step, and the dashboard's
    # "Action rate totals" panel would render flat zero lines.
    m = json.loads((run_dir / "metrics.json").read_text())
    for key in ("fold_rate", "raise_rate", "all_in_rate"):
        rates = m.get(key, [])
        assert rates, f"{key} missing from metrics.json"
        assert any(r > 0 for r in rates), \
            f"{key} stayed at 0 across the whole run — wiring is broken"

    # policy_loss used to be silently stuck at 0 because the callback read
    # the wrong SB3 logger key ('train/policy_loss' vs 'train/policy_gradient_loss').
    # If this regresses the losses panel in the dashboard goes half-blank.
    pls = m.get("policy_loss", [])
    assert pls, "policy_loss missing from metrics.json"
    assert any(abs(p) > 0 for p in pls), \
        "policy_loss stayed at 0 across the run — likely SB3 logger key drift"

    # rewards (per-window raw) and avg_reward_100 (trailing-100 smoothed)
    # used to collapse to identical values because the callback handed the
    # already-trailing-100 slice to metrics.log_step. The dashboard plots
    # them as separate series — collapse means the lines overlap.
    rew = m.get("rewards", [])
    avg = m.get("avg_reward_100", [])
    assert rew and avg, "rewards/avg_reward_100 missing from metrics.json"
    # When there's more than one snapshot the series must not be identical
    # (they were before — both ended up as mean of the same trailing-100
    # slice). The tiny test cycle here may only produce one snapshot, in
    # which case the assert is trivially satisfied by the single-element
    # comparison; the longer smoke run is what actually exercises the split.
    if len(rew) >= 2 and len(avg) >= 2:
        assert any(r != a for r, a in zip(rew, avg)), \
            "rewards and avg_reward_100 are identical at every snapshot — the " \
            "per-window vs trailing-100 split is broken"
