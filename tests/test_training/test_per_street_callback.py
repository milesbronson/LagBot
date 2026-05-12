"""Tests for MetricsCallback's per-street action breakdown.

We don't drive a real PPO/env stack — too slow. Instead we instantiate
the callback, hand it a mock model + a fake `self.locals` dict, call
`_on_step` repeatedly with chosen (action, street) pairs, and check that
the per-street aggregation comes out right.
"""

import json
import os
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.training.callbacks import MetricsCallback
from src.training.metrics import TrainingMetrics


@pytest.fixture
def callback(tmp_path):
    metrics = TrainingMetrics("test_run", save_dir=str(tmp_path))
    cb = MetricsCallback(metrics=metrics, log_freq=10000, verbose=0)
    cb.model = MagicMock()
    # Force the default 6-action layout (Mock unwrap path falls through).
    cb._action_buckets = {
        "fold": [0], "call": [1], "raise": [2, 3, 4], "all_in": [5],
    }
    cb.episode_actions = []
    cb.episode_action_streets = []
    cb.episode_rewards = []
    cb.episode_wins = 0
    cb.episode_count = 0
    cb.current_episode_reward = 0
    cb.last_logged_step = 0
    return cb


def _step(cb, action, street):
    """Simulate one SB3 callback step with the given (action, street)."""
    cb.locals = {
        "actions": np.array([action]),
        "infos": [{"learner_action": "x", "learner_street": street}],
        "dones": np.array([False]),
        "rewards": np.array([0.0]),
    }
    cb._on_step()


def test_pairs_each_action_with_its_street(callback):
    _step(callback, 0, "preflop")   # fold
    _step(callback, 2, "flop")      # raise
    _step(callback, 1, "turn")      # call
    _step(callback, 5, "river")     # all_in
    assert callback.episode_action_streets == [
        (0, "preflop"), (2, "flop"), (1, "turn"), (5, "river"),
    ]


def test_per_street_rates_isolate_by_street(callback):
    # 2 actions on preflop: 1 fold, 1 raise → 0.5/0.5
    _step(callback, 0, "preflop")
    _step(callback, 2, "preflop")
    # 1 action on flop: all-in → 0/0/0/1.0
    _step(callback, 5, "flop")

    rates = callback._per_street_action_rates(callback._action_buckets)
    assert rates["preflop"]["fold"] == pytest.approx(0.5)
    assert rates["preflop"]["raise"] == pytest.approx(0.5)
    assert rates["preflop"]["count"] == 2

    assert rates["flop"]["all_in"] == pytest.approx(1.0)
    assert rates["flop"]["count"] == 1

    # Streets with no actions report all zeros and count=0.
    assert rates["turn"]["count"] == 0
    assert rates["river"]["count"] == 0


def test_unknown_street_actions_are_dropped(callback):
    _step(callback, 0, "preflop")
    _step(callback, 2, None)            # learner_street missing
    _step(callback, 1, "showdown")      # not in canonical 4

    rates = callback._per_street_action_rates(callback._action_buckets)
    assert rates["preflop"]["count"] == 1
    assert sum(r["count"] for r in rates.values()) == 1


def test_record_street_breakdown_writes_json(tmp_path):
    metrics = TrainingMetrics("t", save_dir=str(tmp_path))
    per_street = {
        "preflop": {"fold": 0.1, "call": 0.2, "raise": 0.6, "all_in": 0.1, "count": 100},
        "flop": {"fold": 0.5, "call": 0.3, "raise": 0.2, "all_in": 0.0, "count": 40},
        "turn": {"fold": 0, "call": 0, "raise": 0, "all_in": 0, "count": 0},
        "river": {"fold": 0, "call": 0, "raise": 0, "all_in": 0, "count": 0},
    }
    metrics.record_street_breakdown(50_000, per_street)
    path = os.path.join(str(tmp_path), "t", "street_breakdown.json")
    assert os.path.exists(path)
    with open(path) as f:
        data = json.load(f)
    assert data["timesteps"] == [50_000]
    assert data["distributions"][0]["preflop"]["raise"] == pytest.approx(0.6)


def test_log_metrics_logs_per_street_to_tensorboard(callback):
    # Feed a few steps then force a log dump.
    _step(callback, 0, "preflop")
    _step(callback, 2, "preflop")
    _step(callback, 5, "flop")
    # We need at least one completed episode for the early-return guard
    # in _log_metrics to pass.
    callback.episode_rewards = [0.1]
    callback.episode_count = 1
    callback._log_metrics()

    logger = callback.model.logger
    recorded = {call.args[0]: call.args[1] for call in logger.record.call_args_list}
    assert recorded["agent/preflop/fold_rate"] == pytest.approx(0.5)
    assert recorded["agent/preflop/raise_rate"] == pytest.approx(0.5)
    assert recorded["agent/flop/all_in_rate"] == pytest.approx(1.0)
    # Turn and river had no actions → not logged (count=0 short-circuit).
    assert "agent/turn/fold_rate" not in recorded
    assert "agent/river/fold_rate" not in recorded
