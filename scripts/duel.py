#!/usr/bin/env python
"""
duel.py — stand-alone head-to-head evaluator. Pits CHALLENGER against
TESTER in the same code path the training-time eval gate uses, with
automatic bridging when the two models were trained on different raise-bin
sets (e.g. an 8-bin chain vs the older 3-bin lineage).

Usage:
  python scripts/duel.py CHALLENGER TESTER [options]

CHALLENGER / TESTER may be any of:
  - path to a model .zip file
  - path to a directory containing final_model.zip
  - "random" or "call" (rule agents)

Bin detection priority for each side:
  1. --{role}-bins "0.25,0.5,..."  (explicit CLI)
  2. --{role}-config configs/foo.yaml  (read environment.raise_bins)
  3. infer count from model.action_space and match against KNOWN_BINSETS;
     error out if ambiguous.

Env bin set defaults to the challenger's; override with --env-bins.

Output: one-line + table summary. mbb/100 from the challenger's seat.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import yaml

# Allow `python scripts/duel.py ...` from repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.agents.base_agent import BaseAgent
from src.agents.random_agent import CallAgent, RandomAgent
from src.agents.opponent_ppo import OpponentPPO
from src.training.cross_bin_agent import CrossBinAgent
from src.training.eval_gate import EvalGate


# Bin sets we know we've trained against. Used only when --{role}-bins and
# --{role}-config aren't given and we have to infer from action-space size.
KNOWN_BINSETS = {
    3: [0.5, 1.0, 2.0],
    6: [0.5, 1.0, 1.5, 2.0, 3.0, 5.0],
    8: [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0],
}


def _resolve_model_path(spec: str) -> str:
    p = Path(spec)
    if p.is_file():
        return str(p)
    if p.is_dir():
        for cand in ("best_model.zip", "final_model.zip"):
            if (p / cand).is_file():
                return str(p / cand)
        raise FileNotFoundError(
            f"{spec}: directory has no best_model.zip or final_model.zip"
        )
    repo_candidate = REPO_ROOT / "models" / spec
    if repo_candidate.is_dir():
        return _resolve_model_path(str(repo_candidate))
    raise FileNotFoundError(f"could not resolve model from spec {spec!r}")


def _parse_bins(s: str) -> List[float]:
    bins = [float(x.strip()) for x in s.split(",") if x.strip()]
    if not bins:
        raise ValueError(f"empty bin list: {s!r}")
    if any(b <= 0 for b in bins):
        raise ValueError(f"bins must be > 0: {bins}")
    if bins != sorted(bins):
        raise ValueError(f"bins must be ascending: {bins}")
    return bins


def _bins_from_config(path: str) -> List[float]:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    bins = cfg.get("environment", {}).get("raise_bins")
    if not bins:
        raise ValueError(f"config {path} has no environment.raise_bins")
    return [float(b) for b in bins]


def _load_agent(
    spec: str,
    explicit_bins: Optional[List[float]],
    config_path: Optional[str],
    role: str,
) -> Tuple[BaseAgent, Optional[List[float]]]:
    if spec.lower() == "random":
        return RandomAgent(name=f"{role}=Random"), None
    if spec.lower() == "call":
        return CallAgent(name=f"{role}=Call"), None

    model_path = _resolve_model_path(spec)
    agent = OpponentPPO(model_path, name=f"{role}={Path(model_path).parent.name}")
    if not agent.is_loaded():
        raise RuntimeError(f"failed to load PPO model at {model_path}")

    if explicit_bins is not None:
        bins = explicit_bins
    elif config_path is not None:
        bins = _bins_from_config(config_path)
    else:
        n_actions = int(agent.model.action_space.n)
        bin_count = n_actions - 3
        if bin_count not in KNOWN_BINSETS:
            raise ValueError(
                f"{role}: model has Discrete({n_actions}) → {bin_count} bins, "
                f"which is not in KNOWN_BINSETS. Pass --{role}-bins explicitly."
            )
        bins = KNOWN_BINSETS[bin_count]

    expected_n = 2 + len(bins) + 1
    actual_n = int(agent.model.action_space.n)
    if expected_n != actual_n:
        raise ValueError(
            f"{role}: bin count {len(bins)} implies Discrete({expected_n}) "
            f"but model is Discrete({actual_n}). Bin spec is wrong."
        )
    return agent, bins


def _maybe_wrap(
    agent: BaseAgent,
    agent_bins: Optional[List[float]],
    env_bins: List[float],
    rng: random.Random,
    role: str,
) -> BaseAgent:
    if agent_bins is None or agent_bins == env_bins:
        return agent
    return CrossBinAgent(
        inner=agent,
        inner_bins=agent_bins,
        env_bins=env_bins,
        rng=rng,
        name=f"{role}=CrossBin({agent.name})",
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("challenger")
    p.add_argument("tester")
    p.add_argument("--challenger-bins", type=_parse_bins, default=None)
    p.add_argument("--tester-bins", type=_parse_bins, default=None)
    p.add_argument("--challenger-config", default=None)
    p.add_argument("--tester-config", default=None)
    p.add_argument("--env-bins", type=_parse_bins, default=None,
                   help="override env bin set; defaults to challenger's")
    p.add_argument("--num-hands", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--starting-stack", type=int, default=1000)
    p.add_argument("--small-blind", type=int, default=5)
    p.add_argument("--big-blind", type=int, default=10)
    args = p.parse_args()

    challenger, ch_bins = _load_agent(
        args.challenger, args.challenger_bins, args.challenger_config, "challenger"
    )
    tester, te_bins = _load_agent(
        args.tester, args.tester_bins, args.tester_config, "tester"
    )

    if args.env_bins is not None:
        env_bins = args.env_bins
    elif ch_bins is not None:
        env_bins = ch_bins
    elif te_bins is not None:
        env_bins = te_bins
    else:
        env_bins = KNOWN_BINSETS[3]

    rng_ch = random.Random(args.seed)
    rng_te = random.Random(args.seed + 1)
    challenger = _maybe_wrap(challenger, ch_bins, env_bins, rng_ch, "challenger")
    tester = _maybe_wrap(tester, te_bins, env_bins, rng_te, "tester")

    print(f"challenger : {args.challenger}  bins={ch_bins}")
    print(f"tester     : {args.tester}  bins={te_bins}")
    print(f"env bins   : {env_bins}")
    print(f"hands      : {args.num_hands}  seed={args.seed}")
    print()

    gate = EvalGate(
        num_hands=args.num_hands,
        threshold_mbb_per_100=0.0,
        starting_stack=args.starting_stack,
        small_blind=args.small_blind,
        big_blind=args.big_blind,
        seed=args.seed,
        raise_bins=env_bins,
    )
    result = gate.evaluate(
        challenger,
        tester,
        candidate_id=Path(args.challenger).name,
        predecessor_id=Path(args.tester).name,
    )

    margin = result.mbb_per_100
    wins, losses = result.candidate_wins, result.candidate_losses
    ties = result.hands_played - wins - losses
    profit = result.candidate_profit_chips
    print(f"challenger mbb/100 : {margin:>+12,.1f}")
    print(f"challenger profit  : {profit:>+12,.1f} chips ({profit / args.big_blind:+,.1f} BB)")
    print(f"hands won / lost   : {wins} / {losses}  (ties: {ties})")
    verdict = "challenger wins" if margin > 0 else "tester wins" if margin < 0 else "tie"
    print(f"verdict            : {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
