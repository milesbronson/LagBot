"""Sweep every intermediate checkpoint of every chain generation through the
EvalGate vs its training parent, then report which checkpoint had the peak
performance.

Why: PPO doesn't monotonically improve. The final_model.zip we register may
have collapsed past an earlier peak — and for gens that failed the gate
outright (v3_gen0, v4_gen1) there is probably *some* mid-training checkpoint
that would have passed, salvaging ~2M timesteps of wasted compute.

Output: prints per-checkpoint mbb/100 as it goes, writes a structured
summary to metrics/checkpoint_sweep.json.
"""

import json
import re
import time
from pathlib import Path

from src.agents.opponent_ppo import OpponentPPO
from src.training.eval_gate import EvalGate

NUM_HANDS = 1000
SEED = 0
OUTPUT_PATH = Path("metrics/checkpoint_sweep.json")


CHAIN_GENS = [
    ("heads_up_chain_v1_gen0", "heads_up_anchor_v1"),
    ("heads_up_chain_v2_gen0", "heads_up_chain_v1_gen0"),
    ("heads_up_chain_v2_gen1", "heads_up_chain_v2_gen0"),
    ("heads_up_chain_v3_gen0", "heads_up_chain_v2_gen1"),  # failed gate
    ("heads_up_chain_v3_gen1", "heads_up_chain_v2_gen1"),
    ("heads_up_chain_v4_gen0", "heads_up_chain_v3_gen1"),
    ("heads_up_chain_v4_gen1", "heads_up_chain_v4_gen0"),  # failed gate
]


def list_checkpoints(gen_dir: Path):
    """Return [(steps, path)], sorted by steps. 'final' sorts last."""
    items = []
    step_re = re.compile(r"model_(\d+)_steps\.zip$")
    for p in gen_dir.glob("model_*_steps.zip"):
        m = step_re.search(p.name)
        if m:
            items.append((int(m.group(1)), p))
    items.sort()
    final = gen_dir / "final_model.zip"
    if final.exists():
        items.append(("final", final))
    return items


def sweep_one_gen(gen_id: str, parent_id: str):
    gen_dir = Path("models") / gen_id
    if not gen_dir.exists():
        print(f"[skip] {gen_id}: directory missing")
        return None

    parent_path = f"models/{parent_id}/final_model.zip"
    if not Path(parent_path).exists():
        print(f"[skip] {gen_id}: parent {parent_path} missing")
        return None

    checkpoints = list_checkpoints(gen_dir)
    if not checkpoints:
        print(f"[skip] {gen_id}: no checkpoints in {gen_dir}")
        return None

    print(f"\n=== {gen_id}  (parent: {parent_id})  {len(checkpoints)} checkpoints ===")

    predecessor = OpponentPPO(parent_path, name=f"parent_{parent_id}")
    gate = EvalGate(num_hands=NUM_HANDS, threshold_mbb_per_100=-100.0, seed=SEED)

    results = []
    for steps, cp_path in checkpoints:
        t0 = time.time()
        candidate = OpponentPPO(str(cp_path), name=f"cand_{gen_id}_{steps}")
        result = gate.evaluate(
            candidate, predecessor,
            candidate_id=cp_path.stem, predecessor_id=parent_id,
        )
        elapsed = time.time() - t0
        steps_label = f"{steps:>8}" if isinstance(steps, int) else f"{steps:>8}"
        print(f"  {steps_label}  mbb/100={result.mbb_per_100:+12.0f}  "
              f"profit_chips={result.candidate_profit_chips:+9.0f}  "
              f"w/l={result.candidate_wins}/{result.candidate_losses}  "
              f"({elapsed:.1f}s)", flush=True)
        results.append({
            "steps": steps,
            "path": str(cp_path),
            "mbb_per_100": result.mbb_per_100,
            "profit_chips": result.candidate_profit_chips,
            "wins": result.candidate_wins,
            "losses": result.candidate_losses,
        })

    peak = max(results, key=lambda r: r["mbb_per_100"])
    final = next((r for r in results if r["steps"] == "final"), None)
    print(f"  PEAK: steps={peak['steps']}  mbb/100={peak['mbb_per_100']:+.0f}")
    if final and peak["steps"] != "final":
        delta = peak["mbb_per_100"] - final["mbb_per_100"]
        print(f"  FINAL: mbb/100={final['mbb_per_100']:+.0f}  (peak beats final by {delta:+.0f})")

    return {
        "gen_id": gen_id,
        "parent_id": parent_id,
        "checkpoints": results,
        "peak": peak,
        "final": final,
    }


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary = []
    for gen_id, parent_id in CHAIN_GENS:
        r = sweep_one_gen(gen_id, parent_id)
        if r is not None:
            summary.append(r)
        with open(OUTPUT_PATH, "w") as f:
            json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'gen':<28} {'peak_steps':>12} {'peak_mbb':>14} {'final_mbb':>14} {'delta':>12}")
    for r in summary:
        peak = r["peak"]
        final = r["final"]
        peak_steps = peak["steps"]
        peak_mbb = peak["mbb_per_100"]
        final_mbb = final["mbb_per_100"] if final else float("nan")
        delta = peak_mbb - final_mbb if final else float("nan")
        flag = " *" if final and peak["steps"] != "final" and delta > 0 else ""
        print(f"{r['gen_id']:<28} {str(peak_steps):>12} {peak_mbb:>+14.0f} {final_mbb:>+14.0f} {delta:>+12.0f}{flag}")

    print(f"\nWrote: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
