"""Training curves — TensorBoard scalars charted in-app.

Replaces the need to spin up `tensorboard --logdir logs`. Reads the
SB3 PPO event files under ``logs/<run>/PPO_*/`` and charts every scalar
tag organised by namespace (``rollout/``, ``train/``, ``agent/``,
``agent/<street>/``). Falls back to ``metrics.json`` if no TB events
exist for the selected run.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import streamlit as st  # noqa: E402

from scripts._dashboard_lib import (  # noqa: E402
    load_tb_scalars, select_run, setup_sidebar, tb_xy,
)
from src.training.registry_report import (  # noqa: E402
    load_run_metrics, summary_table,
)


st.set_page_config(page_title="Training curves", layout="wide")
registry, metrics_dir = setup_sidebar()
log_dir = st.sidebar.text_input("TensorBoard log dir", value="logs")
rows = summary_table(registry, metrics_dir)

# Augment with in-progress runs: dirs under log_dir that have TB events
# but no registry card yet. Lets us watch live training before its gate
# fires and registers the card.
registered_ids = {r["id"] for r in rows}
log_root = Path(log_dir)
if log_root.exists():
    for run_path in sorted(log_root.iterdir()):
        if not run_path.is_dir() or run_path.name in registered_ids:
            continue
        if any(run_path.rglob("events.out.tfevents*")):
            rows.append({"id": run_path.name, "generation": -1, "_in_progress": True})

selected_run = select_run(rows)
if selected_run and any(r.get("_in_progress") and r["id"] == selected_run for r in rows):
    st.sidebar.caption(f"⏳ `{selected_run}` is still training (not yet in registry)")

st.title(f"Training curves — {selected_run}")
if not selected_run:
    st.stop()

tb = load_tb_scalars(selected_run, log_dir=log_dir)


def _plot(tag: str, *, color: str = "#1f77b4", ylim=None, title: str = None,
          ax=None, smoothing: int = 0):
    """Plot one tag onto an axes. Adds optional EMA-style smoothing overlay."""
    xs, ys = tb_xy(tb, tag)
    if not xs:
        if ax is not None:
            ax.text(0.5, 0.5, f"no data for\n{tag}", ha="center", va="center",
                    transform=ax.transAxes, color="#999", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
        return False
    own_ax = ax is None
    if own_ax:
        fig, ax = plt.subplots(figsize=(6, 3.3))
    ax.plot(xs, ys, color=color, alpha=(0.35 if smoothing else 1.0), linewidth=1)
    if smoothing and len(ys) > smoothing:
        w = smoothing
        avg = [sum(ys[max(0, i - w):i + 1]) / min(w + 1, i + 1) for i in range(len(ys))]
        ax.plot(xs, avg, color=color, linewidth=1.8)
    ax.set_title(title or tag, fontsize=9)
    ax.set_xlabel("step", fontsize=8)
    ax.grid(True, alpha=0.3)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.tick_params(labelsize=7)
    if own_ax:
        st.pyplot(fig)
        plt.close(fig)
    return True


# --- TB-driven path ---------------------------------------------------
if tb:
    n_tags = len(tb)
    sample_tag = next(iter(tb))
    last_step = tb[sample_tag][-1][0] if tb[sample_tag] else 0
    st.caption(
        f"Reading TensorBoard events from `{log_dir}/{selected_run}/`. "
        f"{n_tags} scalar tags · latest step ≈ {last_step:,}."
    )
    smoothing = st.slider("Smoothing window (rollouts)", 0, 50, 10)
    if st.button("Refresh from disk"):
        st.cache_data.clear()
        st.rerun()

    # --- Rollout ------------------------------------------------------
    st.subheader("Rollout")
    cols = st.columns(3)
    for col, (tag, color, ylim) in zip(cols, [
        ("rollout/ep_rew_mean", "#1f77b4", None),
        ("rollout/ep_len_mean", "#ff7f0e", None),
        ("time/fps",            "#7f7f7f", None),
    ]):
        with col:
            fig, ax = plt.subplots(figsize=(5, 3))
            _plot(tag, color=color, ylim=ylim, ax=ax, smoothing=smoothing)
            st.pyplot(fig); plt.close(fig)

    # --- Train --------------------------------------------------------
    st.subheader("PPO update stats")
    panel = [
        ("train/value_loss",           "#ff7f0e", None),
        ("train/policy_gradient_loss", "#1f77b4", None),
        ("train/entropy_loss",         "#2ca02c", None),
        ("train/approx_kl",            "#9467bd", None),
        ("train/clip_fraction",        "#8c564b", (0, 1)),
        ("train/explained_variance",   "#17becf", None),
    ]
    rows_ = [panel[:3], panel[3:]]
    for row in rows_:
        cols = st.columns(3)
        for col, (tag, color, ylim) in zip(cols, row):
            with col:
                fig, ax = plt.subplots(figsize=(5, 3))
                _plot(tag, color=color, ylim=ylim, ax=ax, smoothing=smoothing)
                st.pyplot(fig); plt.close(fig)

    # --- Agent (env-level rewards / win rate / action mix) ------------
    # avg_reward is the SHAPED training signal (profit minus regret
    # penalty, so it sits negative under regret_blend); avg_raw_profit is
    # the unshaped per-hand winnings — the real money curve.
    st.subheader("Agent — reward & win rate")
    cols = st.columns(4)
    for col, (tag, color, ylim) in zip(cols, [
        ("agent/avg_reward",      "#1f77b4", None),
        ("agent/avg_raw_profit",  "#2ca02c", None),
        ("agent/win_rate",        "#9467bd", (0, 1)),
        ("agent/episodes_completed", "#7f7f7f", None),
    ]):
        with col:
            fig, ax = plt.subplots(figsize=(5, 3))
            _plot(tag, color=color, ylim=ylim, ax=ax, smoothing=smoothing)
            st.pyplot(fig); plt.close(fig)

    st.subheader("Agent — overall action rates")
    cols = st.columns(4)
    for col, (tag, color) in zip(cols, [
        ("agent/fold_rate",   "#d62728"),
        ("agent/call_rate",   "#7f7f7f"),
        ("agent/raise_rate",  "#1f77b4"),
        ("agent/all_in_rate", "#ff7f0e"),
    ]):
        with col:
            fig, ax = plt.subplots(figsize=(4, 3))
            _plot(tag, color=color, ylim=(0, 1), ax=ax, smoothing=smoothing)
            st.pyplot(fig); plt.close(fig)

    # --- Per-street action rates -------------------------------------
    st.subheader("Per-street action rates")
    streets = ("preflop", "flop", "turn", "river")
    actions = [
        ("fold_rate",   "#d62728"),
        ("call_rate",   "#7f7f7f"),
        ("raise_rate",  "#1f77b4"),
        ("all_in_rate", "#ff7f0e"),
    ]
    cols = st.columns(2)
    for i, street in enumerate(streets):
        with cols[i % 2]:
            st.markdown(f"**{street.capitalize()}**")
            fig, ax = plt.subplots(figsize=(6, 3.2))
            for action, color in actions:
                _plot(f"agent/{street}/{action}", color=color, ylim=(0, 1),
                      ax=ax, smoothing=smoothing, title="")
            ax.set_title(street.capitalize(), fontsize=10)
            ax.legend([a[0] for a in actions], loc="upper right", fontsize=7)
            st.pyplot(fig); plt.close(fig)

    # --- Everything else (catch-all for tags we didn't categorise) ---
    known = {
        "rollout/ep_rew_mean", "rollout/ep_len_mean", "time/fps",
        "train/value_loss", "train/policy_gradient_loss", "train/entropy_loss",
        "train/approx_kl", "train/clip_fraction", "train/explained_variance",
        "train/loss", "train/learning_rate", "train/clip_range",
        "agent/avg_reward", "agent/avg_raw_profit", "agent/win_rate",
        "agent/episodes_completed",
        "agent/fold_rate", "agent/call_rate", "agent/raise_rate", "agent/all_in_rate",
        "agent/max_reward", "agent/min_reward",
    }
    for s in streets:
        for a, _ in actions:
            known.add(f"agent/{s}/{a}")
        known.add(f"agent/{s}/action_count")
    extras = sorted(set(tb) - known)
    if extras:
        with st.expander(f"Other tags ({len(extras)})"):
            cols = st.columns(3)
            for i, tag in enumerate(extras):
                with cols[i % 3]:
                    fig, ax = plt.subplots(figsize=(4.5, 2.6))
                    _plot(tag, ax=ax, smoothing=smoothing)
                    st.pyplot(fig); plt.close(fig)

    st.stop()


# --- Fallback: metrics.json (legacy view) ----------------------------
st.warning(
    f"No TensorBoard events found under `{log_dir}/{selected_run}/`. "
    "Falling back to metrics.json."
)
metrics = load_run_metrics(selected_run, metrics_dir)
if not metrics or not metrics.get("timesteps"):
    st.warning(f"No metrics.json for run {selected_run!r} either.")
    st.stop()

ts = metrics["timesteps"]
col1, col2 = st.columns(2)
with col1:
    st.subheader("Reward")
    fig, ax = plt.subplots(figsize=(7, 4))
    if metrics.get("rewards"):
        ax.plot(ts, metrics["rewards"], label="shaped (per-window)", alpha=0.4)
    if metrics.get("avg_reward_100"):
        ax.plot(ts, metrics["avg_reward_100"], label="shaped avg(100)", linewidth=2)
    raw100 = metrics.get("avg_raw_profit_100")
    if raw100 and len(raw100) == len(ts):
        ax.plot(ts, raw100, label="raw profit avg(100)", linewidth=2, color="#2ca02c")
    ax.axhline(0, color="#999", lw=0.8, ls="--")
    ax.set_xlabel("training step")
    ax.set_ylabel("reward / profit")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig); plt.close(fig)

    st.subheader("Win rate")
    fig, ax = plt.subplots(figsize=(7, 4))
    if metrics.get("win_rate"):
        ax.plot(ts, metrics["win_rate"], color="#2ca02c", linewidth=2)
        ax.set_ylim(0, 1)
    ax.set_xlabel("training step")
    ax.set_ylabel("win rate")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig); plt.close(fig)

with col2:
    st.subheader("Losses")
    fig, ax = plt.subplots(figsize=(7, 4))
    for key, color in [("policy_loss", "#1f77b4"),
                       ("value_loss", "#ff7f0e"),
                       ("entropy", "#2ca02c")]:
        vals = metrics.get(key)
        if vals:
            ax.plot(ts, vals, label=key, color=color)
    ax.set_xlabel("training step")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig); plt.close(fig)
