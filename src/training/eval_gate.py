"""
EvalGate — head-to-head shootout that gates whether a freshly trained
candidate is allowed to replace its predecessor as the latest checkpoint
in the registry.

Plays a fixed number of hands between two seated agents in a heads-up
TexasHoldemEnv with deterministic seeding. Reports the candidate's
profit in mbb/100 (milli-big-blinds per 100 hands), the standard poker
performance unit, and a pass/fail decision against a configured
threshold.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from src.agents.base_agent import BaseAgent
from src.poker_env.texas_holdem_env import TexasHoldemEnv


@dataclass
class EvalResult:
    candidate_id: str
    predecessor_id: str
    hands_played: int
    candidate_profit_chips: float
    big_blind: int
    mbb_per_100: float
    candidate_wins: int
    candidate_losses: int
    passed: bool | None = None

    def to_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "predecessor_id": self.predecessor_id,
            "hands_played": self.hands_played,
            "candidate_profit_chips": self.candidate_profit_chips,
            "big_blind": self.big_blind,
            "mbb_per_100": self.mbb_per_100,
            "candidate_wins": self.candidate_wins,
            "candidate_losses": self.candidate_losses,
            "passed": self.passed,
        }


class EvalGate:
    def __init__(
        self,
        num_hands: int = 1000,
        threshold_mbb_per_100: float = 0.0,
        starting_stack: int = 1000,
        small_blind: int = 5,
        big_blind: int = 10,
        seed: int = 0,
        raise_bins: list[float] | None = None,
        duplicate: bool = True,
        env_overrides: dict | None = None,
    ):
        if num_hands <= 0:
            raise ValueError("num_hands must be > 0")
        self.num_hands = num_hands
        self.threshold = threshold_mbb_per_100
        self.starting_stack = starting_stack
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.seed = seed
        self.raise_bins = raise_bins
        # Extra TexasHoldemEnv kwargs (min_raise_multiplier, rake_percent,
        # rake_cap, ...) so the gate judges candidates under the SAME
        # table rules they were trained with.
        self.env_overrides = env_overrides or {}
        # Duplicate (mirrored) scoring: each dealt hand is played twice with
        # the candidate in seat 0 then seat 1, on the IDENTICAL deck. Card
        # luck and any positional/seat advantage both cancel in the sum,
        # leaving a near-pure skill differential at a fraction of the
        # variance. A mirror of identical policies nets ~0 (the symmetry
        # check). Disable only for debugging the single-seat path.
        self.duplicate = duplicate

    def _build_env(self) -> TexasHoldemEnv:
        # track_opponents=True is mandatory: training (train.py:_build_env)
        # builds the env with tracking on, so saved PPO models expect a
        # 161-dim obs. Mismatching here makes every loaded PPO opponent
        # crash inside predict() and silently fall back to "call",
        # producing a meaningless gate result. Same applies to raise_bins:
        # the gate env must mirror training's action space.
        return TexasHoldemEnv(
            num_players=2,
            starting_stack=self.starting_stack,
            small_blind=self.small_blind,
            big_blind=self.big_blind,
            track_opponents=True,
            learning_agent_id=0,
            raise_bins=self.raise_bins,
            **self.env_overrides,
        )

    def _seed_hand(self, hand_idx: int) -> None:
        """Seed the deck deterministically for hand ``hand_idx``.

        The deck shuffle in game_state uses Python's GLOBAL ``random``
        (not np.random, which is all env.reset(seed=) touches), so we must
        seed it here for the deal to be reproducible. Seeding both RNGs
        immediately before reset() — the shuffle is the first global-random
        consumer inside start_new_hand — pins the exact cards for this hand.
        Re-seeding to the same value in the mirror pass reproduces the deck
        identically while only the seat assignment differs.
        """
        h = self.seed + hand_idx
        random.seed(h)
        np.random.seed(h % (2**32 - 1))

    def _play_pass(
        self, candidate: BaseAgent, predecessor: BaseAgent, candidate_seat: int
    ) -> tuple[float, int, int]:
        """Play ``num_hands`` with the candidate in ``candidate_seat`` (0 or
        1) and measure the candidate's chip profit from that seat. The deck
        + button sequence are pinned per-hand, so two passes with swapped
        seats see identical cards."""
        opp_seat = 1 - candidate_seat
        env = self._build_env()
        env.game_state.players[candidate_seat].seat_agent(candidate)
        env.game_state.players[opp_seat].seat_agent(predecessor)
        candidate.player_id = candidate_seat
        predecessor.player_id = opp_seat
        candidate.bind_env(env)
        predecessor.bind_env(env)
        agents = [None, None]
        agents[candidate_seat] = candidate
        agents[opp_seat] = predecessor

        self._seed_hand(0)
        obs, _ = env.reset(seed=self.seed)

        profit_chips = 0.0
        wins = 0
        losses = 0
        for hand_idx in range(self.num_hands):
            stack_before = env.game_state.players[candidate_seat].starting_stack_this_hand
            done = False
            while not done:
                current = env.game_state.get_current_player()
                agent = agents[current.player_id]
                valid = env.get_valid_actions()
                action = agent.select_action(obs, valid)
                obs, _, terminated, _, _ = env.step(action)
                done = terminated

            delta = env.game_state.players[candidate_seat].stack - stack_before
            profit_chips += delta
            if delta > 0:
                wins += 1
            elif delta < 0:
                losses += 1

            if hand_idx < self.num_hands - 1:
                self._seed_hand(hand_idx + 1)
                obs, _ = env.reset(seed=self.seed + hand_idx + 1)

        return profit_chips, wins, losses

    def evaluate(
        self,
        candidate: BaseAgent,
        predecessor: BaseAgent,
        *,
        candidate_id: str = "candidate",
        predecessor_id: str = "predecessor",
    ) -> EvalResult:
        # Pass A: candidate in seat 0. Pass B (duplicate only): candidate in
        # seat 1 on the IDENTICAL per-hand decks/buttons. Summing cancels
        # card luck and seat bias.
        #
        # _seed_hand reseeds the process-global RNGs, and this gate runs
        # mid-training (BestCheckpointCallback), so both states must be
        # restored on exit or every eval restarts training's deck stream
        # from the same deterministic point. Same for the agents'
        # `deterministic` flag: opponent instances are cached across
        # generations, so a leaked True would freeze their exploration.
        py_state = random.getstate()
        np_state = np.random.get_state()
        flag_holders = [a for a in (candidate, predecessor) if hasattr(a, "deterministic")]
        saved_flags = [a.deterministic for a in flag_holders]
        # Greedy actions during eval: with deterministic policies the mirror
        # pass is a literal replay with seats swapped, so card luck and seat
        # bias cancel EXACTLY; sampling would leave residual noise.
        for a in flag_holders:
            a.deterministic = True
        try:
            pa, wa, la = self._play_pass(candidate, predecessor, candidate_seat=0)
            if self.duplicate:
                pb, wb, lb = self._play_pass(candidate, predecessor, candidate_seat=1)
            else:
                pb, wb, lb = 0.0, 0, 0
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)
            for a, flag in zip(flag_holders, saved_flags):
                a.deterministic = flag

        profit_chips = pa + pb
        wins = wa + wb
        losses = la + lb
        hands_played = self.num_hands * (2 if self.duplicate else 1)

        mbb_per_100 = (profit_chips / self.big_blind / hands_played) * 100_000.0

        return EvalResult(
            passed=mbb_per_100 >= self.threshold,
            candidate_id=candidate_id,
            predecessor_id=predecessor_id,
            hands_played=hands_played,
            candidate_profit_chips=float(profit_chips),
            big_blind=self.big_blind,
            mbb_per_100=mbb_per_100,
            candidate_wins=wins,
            candidate_losses=losses,
        )

    def passes(self, result: EvalResult) -> bool:
        return result.mbb_per_100 >= self.threshold
