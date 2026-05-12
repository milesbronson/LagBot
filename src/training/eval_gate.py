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

from dataclasses import dataclass

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
    ):
        if num_hands <= 0:
            raise ValueError("num_hands must be > 0")
        self.num_hands = num_hands
        self.threshold = threshold_mbb_per_100
        self.starting_stack = starting_stack
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.seed = seed

    def evaluate(
        self,
        candidate: BaseAgent,
        predecessor: BaseAgent,
        *,
        candidate_id: str = "candidate",
        predecessor_id: str = "predecessor",
    ) -> EvalResult:
        env = TexasHoldemEnv(
            num_players=2,
            starting_stack=self.starting_stack,
            small_blind=self.small_blind,
            big_blind=self.big_blind,
            track_opponents=False,
            learning_agent_id=0,
        )
        env.game_state.players[0].seat_agent(candidate)
        env.game_state.players[1].seat_agent(predecessor)
        candidate.player_id = 0
        predecessor.player_id = 1
        agents = [candidate, predecessor]

        obs, _ = env.reset(seed=self.seed)

        profit_chips = 0.0
        wins = 0
        losses = 0

        for hand_idx in range(self.num_hands):
            # starting_stack_this_hand is the pre-blind stack the env
            # itself uses for reward accounting, so the SB/BB forfeits
            # are correctly attributed to this hand.
            stack_before = env.game_state.players[0].starting_stack_this_hand

            done = False
            while not done:
                current = env.game_state.get_current_player()
                agent = agents[current.player_id]
                valid = env.get_valid_actions()
                action = agent.select_action(obs, valid)
                obs, _, terminated, _, _ = env.step(action)
                done = terminated

            stack_after = env.game_state.players[0].stack
            delta = stack_after - stack_before
            profit_chips += delta
            if delta > 0:
                wins += 1
            elif delta < 0:
                losses += 1

            if hand_idx < self.num_hands - 1:
                obs, _ = env.reset(seed=self.seed + hand_idx + 1)

        mbb_per_100 = (profit_chips / self.big_blind / self.num_hands) * 100_000.0

        return EvalResult(
            candidate_id=candidate_id,
            predecessor_id=predecessor_id,
            hands_played=self.num_hands,
            candidate_profit_chips=float(profit_chips),
            big_blind=self.big_blind,
            mbb_per_100=mbb_per_100,
            candidate_wins=wins,
            candidate_losses=losses,
        )

    def passes(self, result: EvalResult) -> bool:
        return result.mbb_per_100 >= self.threshold
