"""CLI: print and plot a registry-driven cross-generation rollup.

Usage:
    python scripts/registry_report.py
    python scripts/registry_report.py --plot out.png
    python scripts/registry_report.py --registry models/registry.json --metrics-dir metrics
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.agent_registry import AgentRegistry
from src.training.registry_report import (
    plot_generation_summary,
    print_summary,
    summary_table,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--registry", default="models/registry.json")
    p.add_argument("--metrics-dir", default="metrics")
    p.add_argument("--plot", default=None, help="if set, save figure to this path")
    p.add_argument("--kind", default="ppo")
    args = p.parse_args()

    registry = AgentRegistry(args.registry)
    rows = summary_table(registry, args.metrics_dir, args.kind)
    print_summary(rows)

    if args.plot:
        saved = plot_generation_summary(
            registry, args.plot, args.metrics_dir, args.kind,
        )
        if saved:
            print(f"\nfigure saved: {saved}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
