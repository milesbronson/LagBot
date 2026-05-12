"""LagBot training dashboard — single Streamlit app consolidating every
view we have over the registry + per-run metrics.

Usage:
    streamlit run scripts/dashboard.py

Sidebar lets you pick:
- which registry file to load
- which metrics directory to scan
- which run (for per-run tabs)
- whether to auto-refresh

Tabs:
1. Registry overview — summary table + matchup heatmap + per-gen mbb/100
2. Run training curves — reward, win-rate, losses, learning rate
3. Per-street action breakdown — fold/call/raise/all-in stacked over time, per street
4. Critic calibration — V(s_0) vs G_0 scatter

All loaders are imported from src.training.registry_report so this file
is presentation-only — no business logic lives here.
"""

import os
import sys
import time
from pathlib import Path

# Make `src` importable when running via `streamlit run scripts/dashboard.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
# Use a non-GUI backend so figures render correctly inside Streamlit's
# worker threads (the default macOS backend aborts there).
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.training.agent_registry import AgentRegistry
from src.training.registry_report import (
    load_run_metrics,
    load_street_breakdown,
    load_value_calibration,
    matchup_matrix,
    summary_table,
)


st.set_page_config(page_title="LagBot training dashboard", layout="wide")


# ---------------------------------------------------------------------
# Sidebar — global selectors
# ---------------------------------------------------------------------

st.sidebar.title("LagBot dashboard")

registry_path = st.sidebar.text_input("Registry path", value="models/registry.json")
metrics_dir = st.sidebar.text_input("Metrics directory", value="metrics")
auto_refresh = st.sidebar.checkbox("Auto-refresh every 30s", value=False)

if not os.path.exists(registry_path):
    st.error(f"Registry not found at {registry_path!r}. Train at least one generation first.")
    st.stop()

registry = AgentRegistry(registry_path)
rows = summary_table(registry, metrics_dir)

if not rows:
    st.warning("Registry has no PPO cards yet. Run `python train.py --config ...` first.")
    st.stop()

run_ids = [r["id"] for r in rows]
selected_run = st.sidebar.selectbox(
    "Run (for per-run tabs)",
    options=run_ids,
    index=len(run_ids) - 1,
)


# ---------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------

tab_overview, tab_curves, tab_streets, tab_critic = st.tabs(
    ["Registry overview", "Training curves", "Per-street actions", "Critic calibration"]
)


# ---------- Tab 1: registry overview --------------------------------

with tab_overview:
    st.header("Registry overview")
    st.caption(
        "All PPO checkpoints in lineage order. mbb/100 is the EvalGate verdict vs parent; "
        "absent for fresh or pre-gate runs."
    )

    df = pd.DataFrame(rows)
    # Pretty-format the columns we display.
    df_display = df[[
        "generation", "id", "parent_id", "total_timesteps",
        "eval_mbb_per_100", "eval_passed",
        "last_avg_reward_100", "last_win_rate",
    ]].rename(columns={
        "generation": "gen",
        "parent_id": "parent",
        "total_timesteps": "steps",
        "eval_mbb_per_100": "mbb/100",
        "eval_passed": "passed",
        "last_avg_reward_100": "avg_reward(100)",
        "last_win_rate": "win_rate",
    })
    st.dataframe(df_display, width="stretch", hide_index=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("EvalGate verdict per generation")
        gens = [r["generation"] for r in rows if r["eval_mbb_per_100"] is not None]
        mbbs = [r["eval_mbb_per_100"] for r in rows if r["eval_mbb_per_100"] is not None]
        if gens:
            fig, ax = plt.subplots(figsize=(6, 4))
            colors = ["#2ca02c" if v >= 0 else "#d62728" for v in mbbs]
            ax.bar(gens, mbbs, color=colors)
            ax.axhline(0, color="black", linewidth=0.7)
            ax.set_xlabel("generation")
            ax.set_ylabel("mbb / 100 vs parent")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("No eval stats recorded yet (enable `eval_gate` in config).")

    with col_b:
        st.subheader("Matchup matrix (observer vs opponent)")
        ids, matrix = matchup_matrix(registry)
        if ids:
            data = np.array([[v if v is not None else np.nan for v in row] for row in matrix])
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(data, cmap="RdYlGn", aspect="auto")
            ax.set_xticks(range(len(ids)))
            ax.set_yticks(range(len(ids)))
            ax.set_xticklabels([i[:14] for i in ids], rotation=45, ha="right", fontsize=8)
            ax.set_yticklabels([i[:14] for i in ids], fontsize=8)
            ax.set_xlabel("opponent")
            ax.set_ylabel("observer")
            fig.colorbar(im, ax=ax, label="avg chip profit / hand")
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("No matchups recorded yet.")


# ---------- Tab 2: training curves ----------------------------------

with tab_curves:
    st.header(f"Training curves — {selected_run}")
    run_metrics = load_run_metrics(selected_run, metrics_dir)
    if not run_metrics or not run_metrics.get("timesteps"):
        st.warning(f"No metrics.json for run {selected_run!r}.")
    else:
        ts = run_metrics["timesteps"]

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Reward")
            fig, ax = plt.subplots(figsize=(7, 4))
            if run_metrics.get("rewards"):
                ax.plot(ts, run_metrics["rewards"], label="raw", alpha=0.4)
            if run_metrics.get("avg_reward_100"):
                ax.plot(ts, run_metrics["avg_reward_100"], label="avg(100)", linewidth=2)
            ax.set_xlabel("training step")
            ax.set_ylabel("reward")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)

            st.subheader("Win rate")
            fig, ax = plt.subplots(figsize=(7, 4))
            if run_metrics.get("win_rate"):
                ax.plot(ts, run_metrics["win_rate"], color="#2ca02c", linewidth=2)
                ax.set_ylim(0, 1)
            ax.set_xlabel("training step")
            ax.set_ylabel("win rate")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)

        with col2:
            st.subheader("Losses")
            fig, ax = plt.subplots(figsize=(7, 4))
            for key, color in [("policy_loss", "#1f77b4"),
                               ("value_loss", "#ff7f0e"),
                               ("entropy", "#2ca02c")]:
                vals = run_metrics.get(key)
                if vals:
                    ax.plot(ts, vals, label=key, color=color)
            ax.set_xlabel("training step")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)

            st.subheader("Action rate totals")
            fig, ax = plt.subplots(figsize=(7, 4))
            for key, color in [("fold_rate", "#d62728"),
                               ("raise_rate", "#1f77b4"),
                               ("all_in_rate", "#ff7f0e")]:
                vals = run_metrics.get(key)
                if vals:
                    ax.plot(ts, vals, label=key, color=color)
            ax.set_xlabel("training step")
            ax.set_ylabel("rate")
            ax.set_ylim(0, 1)
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)


# ---------- Tab 3: per-street action breakdown ----------------------

with tab_streets:
    st.header(f"Per-street action breakdown — {selected_run}")
    sb = load_street_breakdown(selected_run, metrics_dir)
    if not sb or not sb.get("timesteps"):
        st.warning(
            "No street_breakdown.json yet. Per-street data is recorded by the "
            "current MetricsCallback — run a fresh training cycle to populate it."
        )
    else:
        ts = sb["timesteps"]
        streets = ("preflop", "flop", "turn", "river")
        actions = ("fold", "call", "raise", "all_in")
        action_colors = {
            "fold": "#d62728",
            "call": "#7f7f7f",
            "raise": "#1f77b4",
            "all_in": "#ff7f0e",
        }

        col1, col2 = st.columns(2)
        for i, street in enumerate(streets):
            col = col1 if i % 2 == 0 else col2
            with col:
                st.subheader(street.capitalize())
                fig, ax = plt.subplots(figsize=(6, 3.5))
                # Stacked area: action rates over training time on this street.
                stack = np.array([
                    [snap[street][a] for snap in sb["distributions"]]
                    for a in actions
                ])
                ax.stackplot(
                    ts, stack,
                    labels=actions,
                    colors=[action_colors[a] for a in actions],
                    alpha=0.85,
                )
                ax.set_xlabel("training step")
                ax.set_ylabel("rate")
                ax.set_ylim(0, 1)
                ax.legend(loc="upper right", fontsize=8)
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close(fig)


# ---------- Tab 4: critic calibration -------------------------------

with tab_critic:
    st.header(f"Critic calibration — {selected_run}")
    calib = load_value_calibration(selected_run, metrics_dir)
    if not calib or not calib.get("pairs"):
        st.warning(
            "No value_calibration.json yet. The CriticCalibrationCallback emits "
            "this — run a fresh training cycle (current train.py wires it in)."
        )
    else:
        pairs = calib["pairs"]
        v = np.array([p["value"] for p in pairs])
        g = np.array([p["actual_return"] for p in pairs])
        bias = float((v - g).mean())
        mae = float(np.abs(v - g).mean())

        st.caption(
            f"gamma = {calib.get('gamma', '?')}   "
            f"n = {len(pairs)}   "
            f"mean bias (V − G_0) = {bias:+.4f}   "
            f"MAE = {mae:.4f}"
        )

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("V(s_0) vs G_0 scatter")
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(v, g, s=10, alpha=0.4)
            lo, hi = float(min(v.min(), g.min())), float(max(v.max(), g.max()))
            ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="V = G_0 (ideal)")
            ax.set_xlabel("predicted V(s_0)")
            ax.set_ylabel("actual discounted return G_0")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)

        with col2:
            st.subheader("Residual (V − G_0) over training time")
            ts = [p["timestep"] for p in pairs]
            residuals = v - g
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(ts, residuals, s=8, alpha=0.4)
            ax.axhline(0, color="black", linewidth=0.7)
            ax.set_xlabel("training step")
            ax.set_ylabel("V(s_0) − G_0")
            ax.set_title("Above 0 = critic over-estimates; below = under-estimates")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)


# ---------------------------------------------------------------------
# Optional auto-refresh.
# ---------------------------------------------------------------------

if auto_refresh:
    time.sleep(30)
    st.rerun()
