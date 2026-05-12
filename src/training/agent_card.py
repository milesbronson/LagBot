"""
AgentCard — one entry in the AgentRegistry.

A card holds everything we know about a single agent (a trained
checkpoint or a rule-based fixture) plus our accumulated observations of
how it plays. The `behavior_stats` and `matchup_history` fields are the
"remember how a friend plays" substrate: each time an agent participates
in a hand, its card's stats are updated. They persist across training
runs.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, List, Dict, Any


@dataclass
class AgentCard:
    id: str
    name: str
    kind: str
    path: Optional[str] = None
    generation: int = 0
    parent_id: Optional[str] = None
    trained_against_ids: List[str] = field(default_factory=list)
    training_config: Dict[str, Any] = field(default_factory=dict)
    total_timesteps: int = 0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # Populated by EvalGate when this agent was promoted. None for agents
    # that bypassed the gate (seed checkpoints, rule-based fixtures).
    eval_stats: Optional[Dict[str, Any]] = None

    # Aggregated behavioral observations, accumulated across every hand
    # this agent has been observed playing (learner OR opponent). Stored
    # as running weighted averages plus the hand count, so additional
    # observations refine the estimate without re-storing every hand.
    # Substrate for opponent embeddings later.
    behavior_stats: Dict[str, Any] = field(
        default_factory=lambda: {"hands_observed": 0}
    )

    # Per-matchup ledger keyed by the OBSERVER agent's id. When agent X
    # plays N hands against this card's agent Y, Y.matchup_history[X.id]
    # accumulates X's results vs Y. Lives on the observed agent so a
    # query "what do we know about Y?" returns every observer's record.
    matchup_history: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentCard":
        return cls(**data)
