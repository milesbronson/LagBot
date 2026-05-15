"""
RegressionEval — run the just-registered card against every prior PPO
card in the registry to detect catastrophic forgetting.

A strict-latest self-play chain can drift: the new generation may learn
to exploit its immediate predecessor while losing the moves that beat
earlier ancestors. This sweep produces an mbb/100 score for the new
card against every prior card and flags negative-margin matchups as
regressions.

Reuses EvalGate's deterministic 1000-hand shootout machinery; the
shootout is reduced to ``num_hands=500`` here by default because it
runs once per gen and across many opponents.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

from src.agents.base_agent import BaseAgent
from src.agents.opponent_ppo import OpponentPPO
from src.agents.random_agent import CallAgent, RandomAgent
from src.training.agent_card import AgentCard
from src.training.agent_registry import AgentRegistry
from src.training.eval_gate import EvalGate


@dataclass
class MatchupResult:
    opponent_id: str
    opponent_kind: str
    mbb_per_100: float
    profit_chips: float
    candidate_wins: int
    candidate_losses: int
    passed: bool


@dataclass
class RegressionResult:
    new_card_id: str
    num_hands: int
    threshold_mbb_per_100: float
    results: List[MatchupResult] = field(default_factory=list)
    passed: int = 0
    regressed: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


class RegressionEval:
    def __init__(
        self,
        num_hands: int = 500,
        threshold_mbb_per_100: float = -100.0,
        starting_stack: int = 1000,
        small_blind: int = 5,
        big_blind: int = 10,
        seed: int = 0,
        include_fixtures: bool = True,
    ):
        if num_hands <= 0:
            raise ValueError("num_hands must be > 0")
        self.num_hands = num_hands
        self.threshold = threshold_mbb_per_100
        self.starting_stack = starting_stack
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.seed = seed
        self.include_fixtures = include_fixtures

    def _instantiate(self, card: AgentCard) -> BaseAgent:
        if card.kind == "ppo":
            return OpponentPPO(card.path, name=card.name)
        if card.kind == "call":
            return CallAgent(name=card.name)
        if card.kind == "random":
            return RandomAgent(name=card.name)
        raise ValueError(f"unknown agent kind {card.kind!r}")

    def evaluate(
        self,
        new_card: AgentCard,
        registry: AgentRegistry,
    ) -> RegressionResult:
        pool: List[AgentCard] = []
        for card in registry.all():
            if card.id == new_card.id:
                continue
            if card.kind in ("call", "random") and not self.include_fixtures:
                continue
            if card.kind == "ppo" and not card.path:
                continue
            pool.append(card)
        # Sort oldest → newest so the report reads chronologically.
        pool.sort(key=lambda c: (c.generation, c.created_at))

        candidate = self._instantiate(new_card)

        gate = EvalGate(
            num_hands=self.num_hands,
            threshold_mbb_per_100=self.threshold,
            starting_stack=self.starting_stack,
            small_blind=self.small_blind,
            big_blind=self.big_blind,
            seed=self.seed,
        )

        result = RegressionResult(
            new_card_id=new_card.id,
            num_hands=self.num_hands,
            threshold_mbb_per_100=self.threshold,
        )

        for opp_card in pool:
            opponent = self._instantiate(opp_card)
            eval_result = gate.evaluate(
                candidate, opponent,
                candidate_id=new_card.id, predecessor_id=opp_card.id,
            )
            mbb = eval_result.mbb_per_100
            ok = mbb >= self.threshold
            result.results.append(MatchupResult(
                opponent_id=opp_card.id,
                opponent_kind=opp_card.kind,
                mbb_per_100=float(mbb),
                profit_chips=float(eval_result.candidate_profit_chips),
                candidate_wins=eval_result.candidate_wins,
                candidate_losses=eval_result.candidate_losses,
                passed=ok,
            ))
            if ok:
                result.passed += 1
            else:
                result.regressed += 1

        return result

    def report(self, result: RegressionResult) -> str:
        total = len(result.results)
        lines = [
            f"\nRegression eval for {result.new_card_id} "
            f"against {total} prior cards ({self.num_hands} hands each):"
        ]
        for r in result.results:
            mark = "OK " if r.passed else "REGRESSION"
            lines.append(
                f"  vs {r.opponent_id:<36} "
                f"mbb/100={r.mbb_per_100:+10.0f}  "
                f"w/l={r.candidate_wins:>4}/{r.candidate_losses:<4}  "
                f"{mark}"
            )
        lines.append(
            f"  TOTAL: {result.passed}/{total} passed  "
            f"({result.regressed}/{total} regressed) "
            f"@ threshold {result.threshold_mbb_per_100:+.0f}"
        )
        return "\n".join(lines)

    def save(self, result: RegressionResult, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
