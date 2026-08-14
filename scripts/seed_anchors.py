#!/usr/bin/env python
"""
seed_anchors.py — idempotently register the ten hand-coded anchor
archetypes into the AgentRegistry.

The anchors form the backbone of the opponent pool and the centroids of
the stratified-by-anchor sampling strategy. They are not retrained per
run, so seeding is a one-time bootstrap (or a refresh when an anchor's
canonical stats change).

Usage:
  python scripts/seed_anchors.py
  python scripts/seed_anchors.py --registry models/registry.json
  python scripts/seed_anchors.py --refresh   # overwrite canonical stats
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.agents.anchors import ALL_ANCHORS
from src.training.agent_card import AgentCard
from src.training.agent_registry import AgentRegistry


def seed_anchors(registry_path: str, refresh: bool = False) -> dict:
    registry = AgentRegistry(path=registry_path)
    added, refreshed, skipped = [], [], []

    for cls in ALL_ANCHORS:
        existing = registry.get(cls.ANCHOR_ID)
        if existing is None:
            card = AgentCard(
                id=cls.ANCHOR_ID,
                name=cls.ARCHETYPE,
                kind="anchor",
                path=None,
                generation=0,
                parent_id=None,
                trained_against_ids=[],
                training_config={"archetype": cls.ARCHETYPE},
                total_timesteps=0,
                behavior_stats=dict(cls.CANONICAL_STATS),
            )
            registry.register(card)
            added.append(cls.ANCHOR_ID)
        elif refresh:
            existing.behavior_stats = dict(cls.CANONICAL_STATS)
            existing.training_config = {"archetype": cls.ARCHETYPE}
            registry.save()
            refreshed.append(cls.ANCHOR_ID)
        else:
            skipped.append(cls.ANCHOR_ID)

    return {"added": added, "refreshed": refreshed, "skipped": skipped}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--registry",
        default=AgentRegistry.DEFAULT_PATH,
        help=f"path to registry JSON (default: {AgentRegistry.DEFAULT_PATH})",
    )
    ap.add_argument(
        "--refresh",
        action="store_true",
        help="overwrite canonical behavior_stats on anchors that already exist",
    )
    args = ap.parse_args()

    summary = seed_anchors(args.registry, refresh=args.refresh)
    if summary["added"]:
        print(f"Added: {', '.join(summary['added'])}")
    if summary["refreshed"]:
        print(f"Refreshed: {', '.join(summary['refreshed'])}")
    if summary["skipped"]:
        print(f"Already seeded (use --refresh to update): {', '.join(summary['skipped'])}")
    if not any(summary.values()):
        print("No anchors processed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
