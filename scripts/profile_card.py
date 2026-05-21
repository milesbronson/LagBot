"""Profile a registry card by playing N hands against a fixed panel.

Behaviour stats normally accumulate when a card is sampled as an
opponent during a later training run, but newly-registered cards can
sit at hands_observed=0 for several gens before that happens. This
script forces a profile by seating the card as the opponent in a
short headless run vs a Random+Call panel, then writing the resulting
stats back to the registry.

Usage:
    python scripts/profile_card.py --card-id heads_up_chain_8bin_v2_gen5
    python scripts/profile_card.py --all-unprofiled
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agents.opponent_ppo import OpponentPPO
from src.agents.random_agent import CallAgent, RandomAgent
from src.poker_env.texas_holdem_env import TexasHoldemEnv
from src.training.agent_card import AgentCard
from src.training.agent_registry import AgentRegistry
from src.training.opponent_autoplay_wrapper import OpponentAutoPlayWrapper


def _instantiate(card: AgentCard):
    if card.kind == "ppo":
        return OpponentPPO(card.path, name=card.name)
    if card.kind == "anchor":
        from src.agents.anchors import ALL_ANCHORS
        for cls in ALL_ANCHORS:
            if cls.ANCHOR_ID == card.id:
                return cls(name=card.name)
        raise ValueError(f"anchor card {card.id!r} has no matching class")
    if card.kind == "call":
        return CallAgent(name=card.name)
    if card.kind == "random":
        return RandomAgent(name=card.name)
    raise ValueError(f"unknown agent kind {card.kind!r}")


def profile_card(
    card: AgentCard,
    hands_per_panel: int = 500,
    raise_bins: list[float] | None = None,
) -> dict:
    """Play `hands_per_panel` hands against each panel agent. Returns
    the aggregated snapshot dict ready for ``update_behavior_stats``."""
    target = _instantiate(card)

    env = TexasHoldemEnv(
        num_players=2,
        starting_stack=1000,
        small_blind=5,
        big_blind=10,
        rake_percent=0.0,
        rake_cap=0,
        min_raise_multiplier=2.0,
        reset_stacks_every_n_timesteps=3000,
        raise_bins=raise_bins or [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0],
        track_opponents=True,
    )

    wrapper = OpponentAutoPlayWrapper(
        env,
        opponents_list=[(card.kind, target)],
        seat_to_card={1: card},
    )

    panel = [
        ("random", RandomAgent(name="profile_random")),
        ("call", CallAgent(name="profile_call")),
    ]
    for _kind, learner in panel:
        learner.seat(env.game_state.players[0])
        learner.bind_env(env)

        for _ in range(hands_per_panel):
            obs, _ = wrapper.reset()
            terminated = truncated = False
            while not (terminated or truncated):
                action = learner.select_action(obs)
                obs, _, terminated, truncated, _ = wrapper.step(action)

    snap = wrapper.snapshot_card_stats()
    return snap.get(card.id, {})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--registry", default="models/registry.json")
    p.add_argument("--card-id", default=None,
                   help="Profile a specific card.")
    p.add_argument("--all-unprofiled", action="store_true",
                   help="Profile every PPO/anchor card with hands_observed=0.")
    p.add_argument("--hands-per-panel", type=int, default=500)
    args = p.parse_args()

    if not (args.card_id or args.all_unprofiled):
        p.error("pass --card-id or --all-unprofiled")

    registry = AgentRegistry(args.registry)
    if args.card_id:
        targets = [registry.get(args.card_id)]
        if targets[0] is None:
            raise SystemExit(f"card {args.card_id!r} not in registry")
    else:
        targets = [
            c for c in registry.all()
            if c.kind in ("ppo", "anchor")
            and int((c.behavior_stats or {}).get("hands_observed", 0)) == 0
            and (c.kind == "anchor" or c.path)
        ]

    if not targets:
        print("No targets — every relevant card already has hands_observed > 0.")
        return

    print(f"Profiling {len(targets)} card(s) with "
          f"{2 * args.hands_per_panel} hands each (random + call panel)…")
    for card in targets:
        stats = profile_card(card, hands_per_panel=args.hands_per_panel)
        hands = int(stats.get("hands_observed", 0))
        if hands == 0:
            print(f"  {card.id}  SKIPPED (no hands recorded)")
            continue
        rates = {k: v for k, v in stats.items() if k != "hands_observed"}
        registry.update_behavior_stats(card.id, rates, hands)
        print(
            f"  {card.id}  hands={hands}  vpip={stats['vpip']:.3f}  "
            f"pfr={stats['pfr']:.3f}  af={stats['af']:.2f}  "
            f"3bet={stats['three_bet_percent']:.3f}  "
            f"cbet={stats['cbet_percent']:.3f}"
        )

    print("Done.")


if __name__ == "__main__":
    main()
