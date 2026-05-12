"""
Registry rollup — read AgentRegistry + per-run TrainingMetrics and produce
a generation-by-generation view of how training is progressing.

For each PPO card in the registry, we have:
- generation, parent_id, trained_against_ids: lineage
- eval_stats: mbb/100 and pass/fail from EvalGate (if a parent existed)
- matchup_history: chip-profit per observer recorded by the wrapper

Per-run TrainingMetrics (metrics/<run_name>/metrics.json) adds the
intra-run reward trajectory.

Two outputs:
1. summary_table(): list of per-gen dicts (also printable).
2. plot_generation_summary(out_path): matplotlib figure with eval mbb/100
   per gen, intra-run avg reward(100), and matchup profit heatmap.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.training.agent_card import AgentCard
from src.training.agent_registry import AgentRegistry


def load_run_metrics(run_name: str, metrics_dir: str = "metrics") -> Optional[dict]:
    path = Path(metrics_dir) / run_name / "metrics.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_value_calibration(run_name: str, metrics_dir: str = "metrics") -> Optional[dict]:
    path = Path(metrics_dir) / run_name / "value_calibration.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_street_breakdown(run_name: str, metrics_dir: str = "metrics") -> Optional[dict]:
    path = Path(metrics_dir) / run_name / "street_breakdown.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def summary_table(
    registry: AgentRegistry,
    metrics_dir: str = "metrics",
    kind: str = "ppo",
) -> List[Dict]:
    """One row per agent of the given kind, sorted by generation.

    Each row carries lineage, eval-gate verdict, and (if available) the
    last reward(100) the per-run TrainingMetrics recorded."""
    rows: List[Dict] = []
    cards = [c for c in registry.all() if c.kind == kind]
    cards.sort(key=lambda c: (c.generation, c.created_at))

    for card in cards:
        eval_stats = card.eval_stats or {}
        run_metrics = load_run_metrics(card.id, metrics_dir)
        last_avg_reward = None
        last_win_rate = None
        timesteps = 0
        if run_metrics:
            avg100 = run_metrics.get("avg_reward_100", [])
            win = run_metrics.get("win_rate", [])
            ts = run_metrics.get("timesteps", [])
            last_avg_reward = avg100[-1] if avg100 else None
            last_win_rate = win[-1] if win else None
            timesteps = ts[-1] if ts else 0

        rows.append({
            "id": card.id,
            "generation": card.generation,
            "parent_id": card.parent_id,
            "trained_against": card.trained_against_ids,
            "total_timesteps": card.total_timesteps or timesteps,
            "eval_mbb_per_100": eval_stats.get("mbb_per_100"),
            "eval_passed": eval_stats.get("passed"),
            "last_avg_reward_100": last_avg_reward,
            "last_win_rate": last_win_rate,
        })
    return rows


def print_summary(rows: List[Dict]) -> None:
    """Pretty-print the table to stdout. No external deps."""
    if not rows:
        print("(no PPO cards in registry)")
        return

    headers = ["gen", "id", "parent", "steps", "mbb/100", "passed", "avg_r(100)", "win"]
    widths = [3, 26, 20, 10, 10, 6, 11, 6]

    def fmt(v, w):
        if v is None:
            s = "-"
        elif isinstance(v, float):
            s = f"{v:.3g}"
        else:
            s = str(v)
        if len(s) > w:
            s = s[: w - 1] + "…"
        return s.ljust(w)

    print(" ".join(fmt(h, w) for h, w in zip(headers, widths)))
    print(" ".join("-" * w for w in widths))
    for r in rows:
        line = [
            r["generation"], r["id"], r["parent_id"] or "(fresh)",
            r["total_timesteps"],
            r["eval_mbb_per_100"],
            "yes" if r["eval_passed"] else ("no" if r["eval_passed"] is False else "-"),
            r["last_avg_reward_100"],
            r["last_win_rate"],
        ]
        print(" ".join(fmt(v, w) for v, w in zip(line, widths)))


def matchup_matrix(
    registry: AgentRegistry,
    kind: str = "ppo",
) -> Tuple[List[str], List[List[Optional[float]]]]:
    """Square avg-profit matrix indexed by ppo agent ids.

    cell[i][j] = average chip profit per hand of observer i vs opponent j,
    pulled from opponent_j.matchup_history[observer_i.id]. None if no
    matchup recorded."""
    cards = [c for c in registry.all() if c.kind == kind]
    cards.sort(key=lambda c: (c.generation, c.created_at))
    ids = [c.id for c in cards]
    matrix: List[List[Optional[float]]] = [[None] * len(ids) for _ in ids]
    for j, opp in enumerate(cards):
        for i, obs_id in enumerate(ids):
            entry = opp.matchup_history.get(obs_id)
            if entry and entry.get("hands_played", 0) > 0:
                matrix[i][j] = entry.get("avg_profit")
    return ids, matrix


def plot_generation_summary(
    registry: AgentRegistry,
    out_path: str,
    metrics_dir: str = "metrics",
    kind: str = "ppo",
) -> Optional[str]:
    """Render a 3-panel figure: eval mbb/100 per gen, reward curves
    overlaid by gen, matchup heatmap. Returns the saved path or None if
    matplotlib is unavailable."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not available; skipping plot")
        return None

    rows = summary_table(registry, metrics_dir, kind)
    if not rows:
        print("(nothing to plot)")
        return None

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    fig.suptitle("Registry rollup", fontsize=14, fontweight="bold")

    ax = axes[0]
    gens = [r["generation"] for r in rows if r["eval_mbb_per_100"] is not None]
    mbbs = [r["eval_mbb_per_100"] for r in rows if r["eval_mbb_per_100"] is not None]
    if gens:
        ax.bar(gens, mbbs)
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_xlabel("generation")
        ax.set_ylabel("mbb / 100 hands vs parent")
        ax.set_title("EvalGate verdict per generation")
    else:
        ax.text(0.5, 0.5, "no eval stats yet", ha="center", va="center")
        ax.set_axis_off()

    ax = axes[1]
    plotted_any = False
    for r in rows:
        run = load_run_metrics(r["id"], metrics_dir)
        if not run:
            continue
        ts = run.get("timesteps", [])
        avg100 = run.get("avg_reward_100", [])
        if ts and avg100:
            ax.plot(ts, avg100, label=f"gen{r['generation']} {r['id'][:14]}", alpha=0.8)
            plotted_any = True
    if plotted_any:
        ax.set_xlabel("training step")
        ax.set_ylabel("avg reward (last 100 episodes)")
        ax.set_title("Intra-run learning curves")
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "no per-run metrics found", ha="center", va="center")
        ax.set_axis_off()

    ax = axes[2]
    ids, matrix = matchup_matrix(registry, kind)
    if ids:
        data = np.array([[v if v is not None else np.nan for v in row] for row in matrix])
        im = ax.imshow(data, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(ids)))
        ax.set_yticks(range(len(ids)))
        ax.set_xticklabels([i[:12] for i in ids], rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels([i[:12] for i in ids], fontsize=8)
        ax.set_xlabel("opponent")
        ax.set_ylabel("observer")
        ax.set_title("Avg chip profit per hand (observer vs opponent)")
        fig.colorbar(im, ax=ax)
    else:
        ax.text(0.5, 0.5, "no matchups recorded", ha="center", va="center")
        ax.set_axis_off()

    # Critic calibration scatter — use the LATEST gen's run as the proxy
    # since calibration is a per-run diagnostic (overlaying generations
    # would clutter the view).
    ax = axes[3]
    latest_with_calib = None
    for r in reversed(rows):
        calib = load_value_calibration(r["id"], metrics_dir)
        if calib and calib.get("pairs"):
            latest_with_calib = (r, calib)
            break
    if latest_with_calib:
        r, calib = latest_with_calib
        pairs = calib["pairs"]
        v = np.array([p["value"] for p in pairs])
        g = np.array([p["actual_return"] for p in pairs])
        ax.scatter(v, g, s=8, alpha=0.5)
        lo, hi = float(min(v.min(), g.min())), float(max(v.max(), g.max()))
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="V = G_0 (ideal)")
        ax.set_xlabel("predicted V(s_0)")
        ax.set_ylabel("actual discounted return G_0")
        ax.set_title(f"Critic calibration — gen{r['generation']} {r['id'][:16]}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "no critic calibration data", ha="center", va="center")
        ax.set_axis_off()

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
