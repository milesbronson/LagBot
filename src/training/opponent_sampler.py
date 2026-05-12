"""
OpponentSampler — picks opponents from the AgentRegistry for a training
run, according to a configurable strategy.

Strategies
----------
- ``latest``: the N most recent cards by (generation, created_at).
- ``random``: uniform sample without replacement.
- ``weighted_recency``: random sample with probability ∝ (generation + 1),
  so newer agents are favored but older ones are still reachable.
- ``fixed``: explicit list of agent ids passed via ``ids=[...]``.

All strategies honour ``kind`` (filter by AgentCard.kind) and
``exclude_ids`` (commonly the learner itself). If the candidate pool is
smaller than ``n``, the sampler returns the full pool unless
``with_replacement=True``.
"""

import random
from typing import List, Optional, Sequence

from src.training.agent_card import AgentCard
from src.training.agent_registry import AgentRegistry


STRATEGIES = ("latest", "random", "weighted_recency", "fixed")


class OpponentSampler:
    def __init__(self, registry: AgentRegistry, rng: Optional[random.Random] = None):
        self.registry = registry
        self.rng = rng if rng is not None else random.Random()

    def sample(
        self,
        n: int,
        *,
        strategy: str = "latest",
        kind: Optional[str] = None,
        exclude_ids: Optional[Sequence[str]] = None,
        ids: Optional[Sequence[str]] = None,
        with_replacement: bool = False,
    ) -> List[AgentCard]:
        if strategy not in STRATEGIES:
            raise ValueError(
                f"unknown strategy {strategy!r}; choose from {STRATEGIES}"
            )
        if n < 0:
            raise ValueError("n must be >= 0")

        if strategy == "fixed":
            return self._sample_fixed(ids or [], exclude_ids or ())

        pool = self._candidate_pool(kind, exclude_ids or ())
        if not pool or n == 0:
            return []

        if strategy == "latest":
            pool.sort(key=lambda c: (c.generation, c.created_at), reverse=True)
            return pool[:n]

        if strategy == "random":
            if with_replacement:
                return [self.rng.choice(pool) for _ in range(n)]
            return self.rng.sample(pool, min(n, len(pool)))

        # weighted_recency
        weights = [c.generation + 1 for c in pool]
        if with_replacement:
            return self.rng.choices(pool, weights=weights, k=n)
        return self._weighted_sample_no_replace(pool, weights, min(n, len(pool)))

    def _candidate_pool(
        self, kind: Optional[str], exclude_ids: Sequence[str]
    ) -> List[AgentCard]:
        excluded = set(exclude_ids)
        cards = [
            c for c in self.registry.all()
            if c.id not in excluded and (kind is None or c.kind == kind)
        ]
        return cards

    def _sample_fixed(
        self, ids: Sequence[str], exclude_ids: Sequence[str]
    ) -> List[AgentCard]:
        excluded = set(exclude_ids)
        out: List[AgentCard] = []
        for aid in ids:
            if aid in excluded:
                continue
            card = self.registry.get(aid)
            if card is None:
                raise KeyError(f"agent id {aid!r} not in registry")
            out.append(card)
        return out

    def _weighted_sample_no_replace(
        self, pool: List[AgentCard], weights: List[float], n: int
    ) -> List[AgentCard]:
        remaining = list(zip(pool, weights))
        out: List[AgentCard] = []
        for _ in range(n):
            if not remaining:
                break
            cards, ws = zip(*remaining)
            pick = self.rng.choices(cards, weights=ws, k=1)[0]
            out.append(pick)
            remaining = [(c, w) for c, w in remaining if c.id != pick.id]
        return out
