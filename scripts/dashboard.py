"""LagBot training dashboard — multi-page Streamlit app.

Usage:
    streamlit run scripts/dashboard.py

This file is the landing page (Registry overview). Every other view
lives under ``scripts/pages/`` and Streamlit picks them up automatically
into the sidebar. Shared helpers (sidebar wiring, behaviour-stat math,
PPO action-distribution utilities) live in ``scripts/_dashboard_lib.py``.

To add a new view, drop a new file in ``scripts/pages/`` — name it
``N_Title_With_Underscores.py`` and Streamlit will show it as "Title
With Underscores" in the sidebar in numeric order.
"""

import sys
import time
from pathlib import Path

# Make `src` importable when running via `streamlit run scripts/dashboard.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from scripts._dashboard_lib import setup_sidebar  # noqa: E402
from src.training.registry_report import (  # noqa: E402
    matchup_matrix,
    summary_table,
)


st.set_page_config(page_title="LagBot dashboard", layout="wide")

registry, metrics_dir = setup_sidebar()
rows = summary_table(registry, metrics_dir)

auto_refresh = st.sidebar.checkbox("Auto-refresh every 30s", value=False)


# ---------------------------------------------------------------------
# Landing page — Registry overview
# ---------------------------------------------------------------------

st.title("Registry overview")
st.caption(
    "All cards in the registry: PPO checkpoints, anchors, and rule "
    "fixtures. Use the sidebar to navigate to per-card and per-policy "
    "deep dives."
)

if not rows:
    st.warning("Registry has no PPO cards yet. Run a training cycle first.")
    st.stop()

import re as _re  # noqa: E402


def _run_of(card_id: str) -> str:
    """Strip trailing `_genN` so cards group by training run.

    Handles both id styles in the registry: `..._v6_gen29` and
    `run_20260521_004723_gen13` (the old `_v\\d+`-anchored regex missed the
    latter, exploding the landing page into one expander per card)."""
    return _re.sub(r"_gen\d+$", "", card_id or "")


df = pd.DataFrame(rows)
df["run"] = df["id"].map(_run_of)
df_display = df[[
    "run", "generation", "id", "parent_id", "total_timesteps",
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

view = st.radio(
    "Grouping",
    options=["By run (expandable)", "Flat table"],
    horizontal=True,
    index=0,
)

if view == "Flat table":
    st.dataframe(df_display.sort_values(["run", "gen"]), width="stretch", hide_index=True)
else:
    runs_sorted = sorted(df_display["run"].unique())
    for run in runs_sorted:
        sub = df_display[df_display["run"] == run].sort_values("gen")
        n_cards = len(sub)
        passed = int(sub["passed"].fillna(False).astype(bool).sum())
        gens = sub["gen"].tolist()
        gen_range = f"gen {min(gens)}–{max(gens)}" if len(gens) > 1 else f"gen {gens[0]}"
        header = f"**{run}** — {n_cards} cards · {gen_range} · {passed} passed gate"
        with st.expander(header, expanded=(run == runs_sorted[-1])):
            st.dataframe(
                sub.drop(columns=["run"]),
                width="stretch", hide_index=True,
            )

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
    st.caption("Cell = observer's avg chip profit per hand vs that opponent. Green = wins, red = loses.")
    ids, matrix = matchup_matrix(registry)
    if ids:
        import os as _os
        from matplotlib.colors import TwoSlopeNorm  # noqa: E402

        data = np.array([[v if v is not None else np.nan for v in row] for row in matrix], dtype=float)

        # Drop rows/cols that are entirely empty so the grid isn't mostly blank.
        row_keep = ~np.isnan(data).all(axis=1)
        col_keep = ~np.isnan(data).all(axis=0)
        keep = row_keep | col_keep  # symmetric: keep card if it appears as either side
        if keep.sum() < len(ids):
            data = data[np.ix_(keep, keep)]
            ids = [i for i, k in zip(ids, keep) if k]

        # Strip a common prefix shared by every label (e.g. "heads_up_chain_8bin_v1_")
        # so the remaining tail is what actually distinguishes cards.
        common = _os.path.commonprefix(ids)
        short = [i[len(common):] or i for i in ids] if len(common) > 4 else list(ids)

        finite = data[np.isfinite(data)]
        vmax = float(np.nanmax(np.abs(finite))) if finite.size else 1.0
        vmax = max(vmax, 1e-6)
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

        # Scale figure with card count so labels have room to breathe.
        side = max(6, min(14, 0.32 * len(ids) + 4))
        fig, ax = plt.subplots(figsize=(side, side * 0.85))
        im = ax.imshow(data, cmap="RdYlGn", norm=norm, aspect="auto")
        ax.set_xticks(range(len(ids)))
        ax.set_yticks(range(len(ids)))
        font = 9 if len(ids) <= 20 else (7 if len(ids) <= 35 else 6)
        ax.set_xticklabels(short, rotation=45, ha="right", fontsize=font)
        ax.set_yticklabels(short, fontsize=font)
        ax.set_xlabel("opponent")
        ax.set_ylabel("observer")
        if common and len(common) > 4:
            ax.set_title(f"labels relative to: {common}", fontsize=8, color="#555")

        # Annotate cells with values only when the grid is small enough to read them.
        if len(ids) <= 16:
            for r in range(len(ids)):
                for c in range(len(ids)):
                    v = data[r, c]
                    if np.isfinite(v):
                        ax.text(c, r, f"{v:+.1f}",
                                ha="center", va="center",
                                color="black" if abs(v) < 0.6 * vmax else "white",
                                fontsize=max(6, font - 1))

        fig.colorbar(im, ax=ax, label="avg chip profit / hand")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("No matchups recorded yet.")


if auto_refresh:
    time.sleep(30)
    st.rerun()
