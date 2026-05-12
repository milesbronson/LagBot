"""Tests for CriticCalibrationCallback.

We don't run SB3. Instead we instantiate the callback, set self.locals
manually, and walk through a couple of synthetic episodes verifying:

- V(s_0) is captured at the FIRST step of an episode and never updated
  mid-episode (a critic snapshot, not a running average).
- The recorded actual_return equals sum_t gamma^t * r_t.
- A new episode after `done=True` resets the snapshot and re-captures V.
- Flushing writes value_calibration.json with the expected pairs.
"""

import json
import os

import numpy as np
import pytest
from unittest.mock import MagicMock

from src.training.callbacks import CriticCalibrationCallback


def _make_callback(tmp_path, gamma=0.5):
    cb = CriticCalibrationCallback(save_dir=str(tmp_path), gamma=gamma, flush_freq=10**9)
    cb.model = MagicMock()
    cb.model.gamma = gamma
    # Manually run _on_training_start to set internal state.
    cb._on_training_start()
    return cb


def _step(cb, reward, done, value, num_timesteps):
    cb.num_timesteps = num_timesteps
    cb.locals = {
        "rewards": np.array([reward], dtype=np.float32),
        "dones": np.array([done]),
        "values": np.array([value], dtype=np.float32),
    }
    cb._on_step()


def test_value_is_snapshot_at_step_0_only(tmp_path):
    cb = _make_callback(tmp_path)
    _step(cb, reward=0.0, done=False, value=0.7, num_timesteps=1)
    _step(cb, reward=0.0, done=False, value=99.0, num_timesteps=2)  # ignored
    _step(cb, reward=1.0, done=True, value=99.0, num_timesteps=3)   # ignored
    assert len(cb._pairs) == 1
    assert cb._pairs[0]["value"] == pytest.approx(0.7)


def test_actual_return_uses_discount(tmp_path):
    cb = _make_callback(tmp_path, gamma=0.5)
    # rewards: [0, 1, 4] → G_0 = 0 + 0.5*1 + 0.25*4 = 1.5
    _step(cb, reward=0.0, done=False, value=10.0, num_timesteps=1)
    _step(cb, reward=1.0, done=False, value=10.0, num_timesteps=2)
    _step(cb, reward=4.0, done=True, value=10.0, num_timesteps=3)
    assert cb._pairs[0]["actual_return"] == pytest.approx(1.5)


def test_multiple_episodes_independent(tmp_path):
    cb = _make_callback(tmp_path, gamma=1.0)
    # Episode 1: V_0=0.2, reward=1 → G_0=1
    _step(cb, reward=1.0, done=True, value=0.2, num_timesteps=1)
    # Episode 2: V_0=-0.3, rewards=[2, -1] → G_0=1
    _step(cb, reward=2.0, done=False, value=-0.3, num_timesteps=2)
    _step(cb, reward=-1.0, done=True, value=-0.3, num_timesteps=3)
    assert len(cb._pairs) == 2
    assert cb._pairs[0]["value"] == pytest.approx(0.2)
    assert cb._pairs[0]["actual_return"] == pytest.approx(1.0)
    assert cb._pairs[1]["value"] == pytest.approx(-0.3)
    assert cb._pairs[1]["actual_return"] == pytest.approx(1.0)


def test_flush_writes_json(tmp_path):
    cb = _make_callback(tmp_path, gamma=1.0)
    _step(cb, reward=1.0, done=True, value=0.5, num_timesteps=1)
    cb._flush()
    path = os.path.join(str(tmp_path), "value_calibration.json")
    assert os.path.exists(path)
    with open(path) as f:
        data = json.load(f)
    assert data["gamma"] == 1.0
    assert len(data["pairs"]) == 1
    assert data["pairs"][0]["value"] == pytest.approx(0.5)
    assert data["pairs"][0]["timestep"] == 1


def test_no_values_in_locals_skips_capture(tmp_path):
    """Defensive: if for some reason SB3 didn't put 'values' in locals
    (early reset, mock VecEnv, etc.) we shouldn't crash — we should
    record nothing."""
    cb = _make_callback(tmp_path)
    cb.num_timesteps = 1
    cb.locals = {
        "rewards": np.array([1.0]),
        "dones": np.array([True]),
        # 'values' deliberately absent
    }
    cb._on_step()
    assert cb._pairs == []
