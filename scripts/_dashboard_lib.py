"""Shared helpers for the LagBot Streamlit dashboard.

Each page under ``scripts/pages/`` calls into here for:
- sidebar wiring (registry path, metrics dir, run selector) shared via
  ``st.session_state``
- behaviour-stat vector / radar / PCA helpers (single source of truth
  with ``OpponentSampler``)
- PPO action-distribution + observation construction utilities for the
  scenario lab and policy-compare pages

No Streamlit page logic lives here — only reusable building blocks.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import streamlit as st

# Make `src` importable when the dashboard runs via `streamlit run`.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.training.agent_registry import AgentRegistry
from src.training.opponent_sampler import (
    _AF_SCALE, _STAT_KEYS, _stat_distance, _stat_vector,
)


RADAR_AXES = list(_STAT_KEYS) + ["af_scaled"]


# ---------------------------------------------------------------------
# Sidebar / session state
# ---------------------------------------------------------------------


def setup_sidebar() -> Tuple[AgentRegistry, str]:
    """Render the global sidebar inputs and return ``(registry, metrics_dir)``.

    Values are persisted in ``st.session_state`` so every page sees the
    same selection. Raises a Streamlit stop if the registry is missing.
    """
    st.sidebar.title("LagBot dashboard")
    default_reg = st.session_state.get("registry_path", "models/registry.json")
    default_metrics = st.session_state.get("metrics_dir", "metrics")
    registry_path = st.sidebar.text_input("Registry path", value=default_reg)
    metrics_dir = st.sidebar.text_input("Metrics directory", value=default_metrics)
    st.session_state["registry_path"] = registry_path
    st.session_state["metrics_dir"] = metrics_dir

    if not os.path.exists(registry_path):
        st.error(
            f"Registry not found at {registry_path!r}. Train at least one "
            "generation first."
        )
        st.stop()

    registry = AgentRegistry(registry_path)
    return registry, metrics_dir


def select_run(rows: List[Dict], key: str = "selected_run") -> Optional[str]:
    """Sidebar selectbox for picking which run's metrics to view."""
    if not rows:
        return None
    run_ids = [r["id"] for r in rows]
    default_idx = run_ids.index(st.session_state[key]) if key in st.session_state and st.session_state[key] in run_ids else len(run_ids) - 1
    selected = st.sidebar.selectbox("Run", options=run_ids, index=default_idx, key=key)
    return selected


# ---------------------------------------------------------------------
# Behaviour-stat space helpers
# ---------------------------------------------------------------------


def card_vector(card) -> np.ndarray:
    return np.array(_stat_vector(card.behavior_stats or {}), dtype=float)


def stat_distance(a, b) -> float:
    return _stat_distance(a.behavior_stats or {}, b.behavior_stats or {})


def pca_2d(vectors: np.ndarray) -> np.ndarray:
    """Classic PCA via SVD; returns (N, 2) projection."""
    if len(vectors) < 2:
        return np.zeros((len(vectors), 2))
    centered = vectors - vectors.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:2].T


def radar_plot(ax, vectors, labels, colors, title: str = ""):
    """Spider chart on a polar axis. Each row of `vectors` is a card."""
    n = len(RADAR_AXES)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]
    for vec, label, color in zip(vectors, labels, colors):
        v = list(vec) + [vec[0]]
        ax.plot(angles, v, color=color, linewidth=2, label=label)
        ax.fill(angles, v, color=color, alpha=0.15)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(RADAR_AXES, fontsize=8)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=7)
    ax.set_ylim(0, 1.0)
    if title:
        ax.set_title(title, fontsize=10, pad=12)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)


# ---------------------------------------------------------------------
# Action space / PPO inference
# ---------------------------------------------------------------------


def action_labels(raise_bins: List[float]) -> List[str]:
    """Pretty labels for the Discrete(2 + len(raise_bins) + 1) action space."""
    labels = ["fold", "call/check"]
    for b in raise_bins:
        labels.append(f"raise {int(b * 100)}% pot")
    labels.append("all-in")
    return labels


@st.cache_resource(show_spinner=False)
def load_ppo_model(model_path: str):
    """Cache the PPO model in memory across reruns and pages."""
    from stable_baselines3 import PPO
    return PPO.load(model_path, device="cpu")


def action_distribution(model, obs: np.ndarray) -> np.ndarray:
    """Return categorical action probabilities for a single observation.

    Pulls the distribution from PPO's policy directly so the dashboard
    can chart the full softmax — not just the sampled action.
    """
    import torch

    obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        dist = model.policy.get_distribution(obs_t)
        # SB3 wraps the actual torch.distributions.Categorical here.
        probs = dist.distribution.probs.squeeze(0).cpu().numpy()
    return probs


def get_observation(env) -> np.ndarray:
    """Public wrapper around the env's protected ``_get_observation``."""
    return env._get_observation()


# ---------------------------------------------------------------------
# TensorBoard scalar reader
# ---------------------------------------------------------------------


def _latest_event_mtime(tb_root: Path) -> float:
    """Highest mtime across all event files under ``tb_root``.

    Used as the cache key so a live training run that's still writing
    events invalidates the cached scalars as new data lands.
    """
    if not tb_root.exists():
        return 0.0
    mtimes = [p.stat().st_mtime for p in tb_root.rglob("events.out.tfevents*")]
    return max(mtimes) if mtimes else 0.0


def _load_tb_scalars_raw(tb_root: str, _mtime: float) -> Dict[str, List[Tuple[int, float]]]:
    """Read every scalar tag under ``tb_root`` (recursively).

    ``_mtime`` is in the signature purely to bust ``st.cache_data`` when
    the event file grows mid-training. Returns ``{tag: [(step, value), ...]}``.
    """
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    root = Path(tb_root)
    # SB3 writes to ``<log_dir>/PPO_N/events.out.tfevents…``. We aggregate
    # across every PPO_* subdir so an interrupted run with PPO_1 + PPO_2
    # shows up as one stream.
    series: Dict[str, List[Tuple[int, float]]] = {}
    subdirs = sorted([p for p in root.iterdir() if p.is_dir()])
    if not subdirs:
        # Fallback: events written at the root of tb_root.
        subdirs = [root]
    for sd in subdirs:
        ea = EventAccumulator(str(sd), size_guidance={"scalars": 0})
        try:
            ea.Reload()
        except Exception:
            continue
        for tag in ea.Tags().get("scalars", []):
            evs = ea.Scalars(tag)
            series.setdefault(tag, []).extend((e.step, float(e.value)) for e in evs)
    # Sort each series by step in case multiple PPO_* dirs got merged out of order.
    for tag in series:
        series[tag].sort(key=lambda p: p[0])
    return series


def load_tb_scalars(run_name: str, log_dir: str = "logs") -> Dict[str, List[Tuple[int, float]]]:
    """Cached TensorBoard scalar reader. Returns ``{}`` if no events exist."""
    tb_root = Path(log_dir) / run_name
    if not tb_root.exists():
        return {}
    mtime = _latest_event_mtime(tb_root)
    if mtime == 0.0:
        return {}
    cached = st.cache_data(show_spinner=False)(_load_tb_scalars_raw)
    return cached(str(tb_root), mtime)


def tb_xy(series: Dict[str, List[Tuple[int, float]]], tag: str) -> Tuple[List[int], List[float]]:
    """Unzip a tag's series into parallel step/value lists. ``([], [])`` if absent."""
    points = series.get(tag, [])
    if not points:
        return [], []
    steps, values = zip(*points)
    return list(steps), list(values)
