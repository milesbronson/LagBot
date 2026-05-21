"""
OpponentSampler — picks opponents from the AgentRegistry for a training
run, according to a configurable strategy.

Strategies
----------
- ``latest``: the N most recent cards by (generation, created_at).
- ``random``: uniform sample without replacement.
- ``weighted_recency``: random sample with probability ∝ (generation + 1),
  so newer agents are favored but older ones are still reachable.
- ``stratified_by_anchor``: each card is quantised to its nearest
  anchor archetype (by behaviour-stat distance). Sampling first picks
  a stratum uniformly, then draws a card from inside it. This is the
  diversity guarantee that the ``latest`` strategy can't give us:
  even if the latest 20 checkpoints are all mirror-shovers, we still
  see tight-passive / calling-station / etc. opponents because every
  stratum is reachable. Anchors themselves participate as the seed
  population of their own strata.
- ``fixed``: explicit list of agent ids passed via ``ids=[...]``.

All strategies honour ``kind`` (filter by AgentCard.kind) and
``exclude_ids`` (commonly the learner itself). If the candidate pool is
smaller than ``n``, the sampler returns the full pool unless
``with_replacement=True``.
"""

import math
import random
from typing import Dict, List, Optional, Sequence

from src.training.agent_card import AgentCard
from src.training.agent_registry import AgentRegistry


STRATEGIES = (
    "latest",
    "random",
    "weighted_recency",
    "stratified_by_anchor",
    "fixed",
)

# Behaviour-stat keys used for stratum assignment. All are fractions in
# [0, 1] except ``af`` which is a ratio. AF gets divided by 10 and capped
# at 1.0 so the Shover's AF=49 can't swamp every other feature.
_STAT_KEYS = (
    "vpip",
    "pfr",
    "three_bet_percent",
    "cbet_percent",
    "fold_to_cbet_percent",
    "went_to_showdown_percent",
    "win_at_showdown_percent",
    "wwsf_percent",
)
_AF_SCALE = 10.0  # AF/_AF_SCALE clipped to [0, 1]


def _stat_vector(stats: Dict) -> List[float]:
    """Project a card's behaviour_stats into a normalised feature vector."""
    out = [float(stats.get(k, 0.0)) for k in _STAT_KEYS]
    af = float(stats.get("af", 0.0))
    out.append(min(af / _AF_SCALE, 1.0))
    return out


def _stat_distance(a: Dict, b: Dict) -> float:
    va, vb = _stat_vector(a), _stat_vector(b)
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(va, vb)))


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
        anchors_seed_only: bool = False,
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

        if strategy == "weighted_recency":
            weights = [c.generation + 1 for c in pool]
            if with_replacement:
                return self.rng.choices(pool, weights=weights, k=n)
            return self._weighted_sample_no_replace(pool, weights, min(n, len(pool)))

        # stratified_by_anchor
        return self._sample_stratified(
            pool, n, with_replacement, anchors_seed_only=anchors_seed_only
        )

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

    # --- stratified ---------------------------------------------------

    def assign_strata(
        self, cards: Sequence[AgentCard]
    ) -> Dict[str, List[AgentCard]]:
        """Group cards by nearest-anchor stratum.

        Anchors (``kind == "anchor"``) define their own strata. Every
        non-anchor card with non-empty ``behavior_stats`` is assigned to
        the anchor minimising Euclidean distance in the normalised
        feature vector. Cards with no observations (hands_observed == 0)
        get the ``"unstratified"`` bucket — callers can choose to skip
        or fold them in.

        Returns:
            ``{stratum_id: [cards...]}``. Stratum id is the anchor's id
            (or ``"unstratified"``). Empty strata are omitted.
        """
        anchors = [c for c in self.registry.all() if c.kind == "anchor"]
        if not anchors:
            raise RuntimeError(
                "stratified_by_anchor needs anchor cards in the registry; "
                "run scripts/seed_anchors.py first"
            )

        # Anchors always define the strata, but only appear as members
        # if they're in the input ``cards`` (so exclude_ids filtering
        # upstream actually takes effect).
        input_ids = {c.id for c in cards}
        strata: Dict[str, List[AgentCard]] = {a.id: [] for a in anchors}
        for a in anchors:
            if a.id in input_ids:
                strata[a.id].append(a)

        for card in cards:
            if card.kind == "anchor":
                continue
            stats = card.behavior_stats or {}
            if int(stats.get("hands_observed", 0)) <= 0:
                strata.setdefault("unstratified", []).append(card)
                continue
            best = min(
                anchors,
                key=lambda a: _stat_distance(stats, a.behavior_stats),
            )
            strata[best.id].append(card)

        return {k: v for k, v in strata.items() if v}

    def _sample_stratified(
        self,
        pool: List[AgentCard],
        n: int,
        with_replacement: bool,
        anchors_seed_only: bool = False,
    ) -> List[AgentCard]:
        strata = self.assign_strata(pool)
        if not strata:
            return []

        if anchors_seed_only:
            # Anchors exist only to *define* the style buckets and to seed
            # buckets that have no learned occupant yet. Once a stratum
            # contains any PPO card, drop the scripted anchor from it so the
            # learner trains against past versions of itself (real self-play)
            # rather than the hand-coded archetype. A stratum holding only
            # its anchor keeps it — that's the seeding case.
            pruned: Dict[str, List[AgentCard]] = {}
            for sid, bucket in strata.items():
                non_anchor = [c for c in bucket if c.kind != "anchor"]
                pruned[sid] = non_anchor if non_anchor else bucket
            strata = pruned

        stratum_ids = sorted(strata.keys())  # deterministic given rng
        out: List[AgentCard] = []
        seen: set = set()

        while len(out) < n:
            available = [
                sid for sid in stratum_ids
                if with_replacement or any(c.id not in seen for c in strata[sid])
            ]
            if not available:
                break
            sid = self.rng.choice(available)
            bucket = strata[sid]
            if not with_replacement:
                bucket = [c for c in bucket if c.id not in seen]
            pick = self.rng.choice(bucket)
            out.append(pick)
            if not with_replacement:
                seen.add(pick.id)
        return out
