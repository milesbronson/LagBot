"""
StratifiedEvalGate — diverse-pool head-to-head evaluation.

Why this exists
---------------
The default ``EvalGate`` plays the candidate against a single
predecessor. A candidate can "pass" that gate by learning to exploit
one stale opponent while still being terrible against the rest of the
play-style manifold. The continuous-training pattern then promotes a
narrow specialist that the *next* generation learns to mirror, and the
chain collapses to one archetype.

This gate replaces the single-predecessor shootout with:
1. An eval pool sampled across the same anchor-stratum partition the
   training pool uses, EXCLUDING the cards the learner just trained
   against (so we measure generalisation, not memorisation).
   Anchors are included by default (``include_anchors=True``) so the
   gate has coverage of every stratum during early generations when
   the registry hasn't yet populated all archetypes. Flip the flag
   off once the live pool covers every stratum and you want the bar
   to be "beat the live pool, not the scaffolding".
2. Within each stratum, a recency weight ``1 / (current_gen - card_gen + 1)``
   so newer checkpoints get more weight than ancient ones (user
   preference: stronger opponents anchor the gate).
3. A configurable per-stratum pass criterion ("beat threshold in K of
   the M strata that had any eligible cards"). Pure aggregate mbb/100
   lets a candidate that crushes one stratum mask total failure
   elsewhere — strata-by-strata makes pool-wide skill the bar.

``passes()`` returns True only if both the per-stratum and aggregate
thresholds are met. The training loop owns the retry behaviour.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from src.agents.base_agent import BaseAgent
from src.training.agent_card import AgentCard
from src.training.agent_registry import AgentRegistry
from src.training.eval_gate import EvalGate, EvalResult
from src.training.opponent_sampler import OpponentSampler


@dataclass
class StratumResult:
    stratum_id: str
    opponent_id: str
    mbb_per_100: float
    profit_chips: float
    candidate_wins: int
    candidate_losses: int
    passed: bool


@dataclass
class StratifiedEvalResult:
    candidate_id: str
    num_hands_per_opponent: int
    threshold_mbb_per_100: float
    min_strata_to_pass: int
    aggregate_threshold_mbb_per_100: float
    results: List[StratumResult] = field(default_factory=list)

    @property
    def strata_passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def strata_evaluated(self) -> int:
        return len(self.results)

    @property
    def aggregate_mbb_per_100(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.mbb_per_100 for r in self.results) / len(self.results)

    def to_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "num_hands_per_opponent": self.num_hands_per_opponent,
            "threshold_mbb_per_100": self.threshold_mbb_per_100,
            "min_strata_to_pass": self.min_strata_to_pass,
            "aggregate_threshold_mbb_per_100": self.aggregate_threshold_mbb_per_100,
            "strata_passed": self.strata_passed,
            "strata_evaluated": self.strata_evaluated,
            "aggregate_mbb_per_100": self.aggregate_mbb_per_100,
            "results": [
                {
                    "stratum_id": r.stratum_id,
                    "opponent_id": r.opponent_id,
                    "mbb_per_100": r.mbb_per_100,
                    "profit_chips": r.profit_chips,
                    "candidate_wins": r.candidate_wins,
                    "candidate_losses": r.candidate_losses,
                    "passed": r.passed,
                }
                for r in self.results
            ],
        }


def _recency_weight(card_gen: int, current_gen: int) -> float:
    """User specified: newer cards get more weight within a stratum."""
    return 1.0 / max(1, (current_gen - card_gen + 1))


def build_eval_pool(
    registry: AgentRegistry,
    *,
    current_generation: int,
    training_pool_ids: Sequence[str],
    rng: random.Random,
    cards_per_stratum: int = 1,
    include_anchors: bool = True,
) -> Dict[str, List[AgentCard]]:
    """Sample an eval pool stratified by anchor.

    Exclusions
    ----------
    - Cards used in the just-finished training pool (avoid measuring
      memorisation; we want generalisation).
    - PPO cards without a saved path (can't be loaded).
    - Anchor cards if ``include_anchors=False`` (turn off once the live
      pool covers every stratum and you want the bar to be "beat the
      live pool, not the scaffolding").

    Within each non-empty stratum, sample ``cards_per_stratum`` cards
    weighted by recency. Because anchors have generation 0, real PPO
    cards in the same stratum outweigh them automatically — anchors
    therefore mostly surface in strata with no PPO coverage yet.

    Returns ``{stratum_id: [cards...]}``; strata with no eligible cards
    are omitted entirely from the output (and from the pass denominator).
    """
    excluded = set(training_pool_ids)
    sampler = OpponentSampler(registry, rng=rng)

    all_cards = [
        c for c in registry.all()
        if c.id not in excluded
        and (include_anchors or c.kind != "anchor")
        and (c.kind != "ppo" or c.path)
    ]
    if not all_cards:
        return {}

    strata = sampler.assign_strata(all_cards)
    out: Dict[str, List[AgentCard]] = {}
    for sid, cards in strata.items():
        eligible = list(cards)
        if not eligible:
            continue

        weights = [
            _recency_weight(c.generation, current_generation) for c in eligible
        ]
        k = min(cards_per_stratum, len(eligible))
        # Weighted sample without replacement within the stratum.
        picks: List[AgentCard] = []
        remaining = list(zip(eligible, weights))
        for _ in range(k):
            cards_only, ws = zip(*remaining)
            pick = rng.choices(cards_only, weights=ws, k=1)[0]
            picks.append(pick)
            remaining = [(c, w) for c, w in remaining if c.id != pick.id]
        out[sid] = picks

    return out


class StratifiedEvalGate:
    def __init__(
        self,
        num_hands: int = 2000,
        threshold_mbb_per_100: float = -100.0,
        starting_stack: int = 1000,
        small_blind: int = 5,
        big_blind: int = 10,
        seed: int = 0,
        raise_bins: Optional[List[float]] = None,
        cards_per_stratum: int = 1,
        min_strata_to_pass: int = 5,
        aggregate_threshold_mbb_per_100: float = -200.0,
        include_anchors: bool = True,
    ):
        if num_hands <= 0:
            raise ValueError("num_hands must be > 0")
        self.num_hands = num_hands
        self.threshold = threshold_mbb_per_100
        self.aggregate_threshold = aggregate_threshold_mbb_per_100
        self.starting_stack = starting_stack
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.seed = seed
        self.raise_bins = raise_bins
        self.cards_per_stratum = cards_per_stratum
        self.min_strata_to_pass = min_strata_to_pass
        self.include_anchors = include_anchors

    def evaluate(
        self,
        candidate: BaseAgent,
        candidate_card_id: str,
        registry: AgentRegistry,
        training_pool_ids: Sequence[str],
        current_generation: int,
        opponent_factory,
    ) -> StratifiedEvalResult:
        """Run the candidate vs one drawn card per stratum.

        Args:
            candidate: the freshly trained learner to gate.
            candidate_card_id: id we will register under (used for logs).
            registry: source of opponent cards and anchor strata.
            training_pool_ids: ids the learner just trained against
                (excluded from the eval pool).
            current_generation: used for the recency weighting.
            opponent_factory: callable ``(AgentCard) -> BaseAgent`` that
                instantiates an opponent for a card. Caller owns this so
                we don't have to import ppo / rule classes here.
        """
        rng = random.Random(self.seed)
        pool = build_eval_pool(
            registry,
            current_generation=current_generation,
            training_pool_ids=training_pool_ids,
            rng=rng,
            cards_per_stratum=self.cards_per_stratum,
            include_anchors=self.include_anchors,
        )

        result = StratifiedEvalResult(
            candidate_id=candidate_card_id,
            num_hands_per_opponent=self.num_hands,
            threshold_mbb_per_100=self.threshold,
            min_strata_to_pass=self.min_strata_to_pass,
            aggregate_threshold_mbb_per_100=self.aggregate_threshold,
        )

        # Stable iteration order so the same seed -> same matchup order.
        for matchup_seed, sid in enumerate(sorted(pool.keys())):
            for card in pool[sid]:
                opponent = opponent_factory(card)
                gate = EvalGate(
                    num_hands=self.num_hands,
                    threshold_mbb_per_100=self.threshold,
                    starting_stack=self.starting_stack,
                    small_blind=self.small_blind,
                    big_blind=self.big_blind,
                    seed=self.seed + matchup_seed,
                    raise_bins=self.raise_bins,
                )
                eval_result: EvalResult = gate.evaluate(
                    candidate,
                    opponent,
                    candidate_id=candidate_card_id,
                    predecessor_id=card.id,
                )
                result.results.append(StratumResult(
                    stratum_id=sid,
                    opponent_id=card.id,
                    mbb_per_100=float(eval_result.mbb_per_100),
                    profit_chips=float(eval_result.candidate_profit_chips),
                    candidate_wins=eval_result.candidate_wins,
                    candidate_losses=eval_result.candidate_losses,
                    passed=eval_result.mbb_per_100 >= self.threshold,
                ))

        return result

    def passes(self, result: StratifiedEvalResult) -> bool:
        # Aggregate score keeps a true catastrophe from sneaking through
        # by clearing the per-stratum minimum on a technicality.
        #
        # Clamp the required-strata bar to what was actually evaluable:
        # with a thin pool (early generations, aggressive excludes) fewer
        # strata exist than the configured minimum, and an unclamped bar
        # made every candidate unpassable — burning max_retries full
        # trainings and stalling the chain permanently.
        required = min(self.min_strata_to_pass, result.strata_evaluated)
        return (
            result.strata_passed >= required
            and result.aggregate_mbb_per_100 >= self.aggregate_threshold
        )

    def report(self, result: StratifiedEvalResult) -> str:
        verdict = "PASS" if self.passes(result) else "FAIL"
        lines = [
            f"\nStratifiedEvalGate for {result.candidate_id}: {verdict}",
            f"  hands/opponent: {result.num_hands_per_opponent}",
            f"  per-stratum threshold: {result.threshold_mbb_per_100:+.0f} mbb/100",
            f"  aggregate threshold:    {result.aggregate_threshold_mbb_per_100:+.0f} mbb/100",
            f"  strata passed: {result.strata_passed}/{result.strata_evaluated} "
            f"(need {result.min_strata_to_pass})",
            f"  aggregate mbb/100: {result.aggregate_mbb_per_100:+.0f}",
            "  ---",
        ]
        for r in result.results:
            mark = "OK " if r.passed else "FAIL"
            lines.append(
                f"  [{mark}] {r.stratum_id:<32} vs {r.opponent_id:<36} "
                f"mbb/100={r.mbb_per_100:+10.0f}  "
                f"w/l={r.candidate_wins:>4}/{r.candidate_losses:<4}"
            )
        return "\n".join(lines)
