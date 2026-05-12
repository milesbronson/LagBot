"""
AgentRegistry — persistent JSON-backed catalogue of every agent we know
about, plus accumulated observations of how each plays.

Storage: a single JSON file (default: ``models/registry.json``). Writes
are atomic (tempfile + rename) so a crash mid-save cannot corrupt the
file. The registry is the central artifact of the continuous training
loop: each generation reads opponents from it and writes the new
checkpoint back.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from src.training.agent_card import AgentCard


class AgentRegistry:
    DEFAULT_PATH = "models/registry.json"
    SCHEMA_VERSION = 1

    def __init__(self, path: str = DEFAULT_PATH):
        self.path = Path(path)
        self.cards: Dict[str, AgentCard] = {}
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        with open(self.path) as f:
            data = json.load(f)
        self.cards = {
            aid: AgentCard.from_dict(card_data)
            for aid, card_data in data.get("agents", {}).items()
        }

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        payload = {
            "schema_version": self.SCHEMA_VERSION,
            "saved_at": datetime.now().isoformat(),
            "agents": {aid: card.to_dict() for aid, card in self.cards.items()},
        }
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2)
        tmp.replace(self.path)

    def register(self, card: AgentCard) -> None:
        if card.id in self.cards:
            raise ValueError(f"Agent id {card.id!r} is already registered")
        self.cards[card.id] = card
        self.save()

    def get(self, agent_id: str) -> Optional[AgentCard]:
        return self.cards.get(agent_id)

    def all(self) -> List[AgentCard]:
        return list(self.cards.values())

    def latest(self, n: int = 1, kind: Optional[str] = None) -> List[AgentCard]:
        """Return up to N cards, most recent first by (generation, created_at).
        Filtered by ``kind`` if given (e.g. ``"ppo"``)."""
        cards = self.all()
        if kind is not None:
            cards = [c for c in cards if c.kind == kind]
        cards.sort(key=lambda c: (c.generation, c.created_at), reverse=True)
        return cards[:n]

    def next_generation(self) -> int:
        """One past the highest generation in the registry (0 if empty)."""
        if not self.cards:
            return 0
        return max(c.generation for c in self.cards.values()) + 1

    def update_matchup(
        self,
        observer_id: str,
        opponent_id: str,
        *,
        hands: int,
        profit: float,
        timestep: int,
    ) -> None:
        """Accumulate observer's results vs opponent. Stored on the OPPONENT's
        card (so a query "what do we know about Y?" gathers every observer)."""
        if opponent_id not in self.cards:
            raise KeyError(f"opponent {opponent_id!r} not in registry")
        card = self.cards[opponent_id]
        entry = card.matchup_history.setdefault(
            observer_id,
            {"hands_played": 0, "total_profit": 0.0, "last_seen_timestep": 0},
        )
        entry["hands_played"] += hands
        entry["total_profit"] += profit
        entry["avg_profit"] = entry["total_profit"] / max(entry["hands_played"], 1)
        entry["last_seen_timestep"] = timestep
        self.save()

    def update_behavior_stats(
        self,
        agent_id: str,
        new_stats: Dict[str, Any],
        hands_observed: int,
    ) -> None:
        """Fold a fresh aggregate of behavioural observations into an agent's
        running estimate via weighted average over hand counts.

        ``new_stats`` must be the aggregate from this run (already averaged
        over ``hands_observed``). Existing stats are merged so that more
        hands of evidence pull the running estimate toward the new value."""
        if agent_id not in self.cards:
            raise KeyError(f"agent {agent_id!r} not in registry")
        card = self.cards[agent_id]
        old = card.behavior_stats
        old_hands = int(old.get("hands_observed", 0))
        new_hands_total = old_hands + hands_observed
        merged: Dict[str, Any] = {"hands_observed": new_hands_total}
        if new_hands_total == 0:
            card.behavior_stats = merged
            self.save()
            return
        keys = set(old.keys()) | set(new_stats.keys())
        keys.discard("hands_observed")
        for key in keys:
            old_val = float(old.get(key, 0.0))
            new_val = float(new_stats.get(key, 0.0))
            merged[key] = (
                old_val * old_hands + new_val * hands_observed
            ) / new_hands_total
        card.behavior_stats = merged
        self.save()
