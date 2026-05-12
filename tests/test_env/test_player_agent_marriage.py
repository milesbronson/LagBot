"""
Bidirectional Player <-> Agent link regression tests.

Pins down the contract introduced in the Step-2/3 refactor
(`working_docs/refactor_design_2026-05-10.md` §2): each Player can be seated
to a BaseAgent, and after seating the two objects can find each other by
either direction. The wrapper relies on this to look up opponent agents by
player_id instead of doing `current_idx - 1` math.

Three properties must hold:

1. A fresh agent has player_id = None and a fresh player has agent = None.
2. `agent.seat(player)` AND `player.seat_agent(agent)` both produce the
   same bidirectional binding — calling either end works.
3. The binding lets the wrapper resolve "the agent in seat X" without
   relying on list-index ordering of an opponents list.
"""

import numpy as np
import pytest

from src.agents.random_agent import CallAgent, RandomAgent
from src.poker_env.player import Player
from src.poker_env.texas_holdem_env import TexasHoldemEnv


class TestFreshObjectsHaveNoBinding:
    def test_fresh_player_has_no_agent(self):
        p = Player(player_id=3, stack=1000, name="Seat3")
        assert p.agent is None

    def test_fresh_agent_has_no_player_id(self):
        a = CallAgent(name="C")
        assert a.player_id is None


class TestBidirectionalBinding:
    def test_player_seat_agent_sets_both_sides(self):
        p = Player(player_id=2, stack=1000, name="Seat2")
        a = CallAgent(name="C")
        p.seat_agent(a)
        assert p.agent is a
        assert a.player_id == 2

    def test_agent_seat_sets_both_sides(self):
        """`agent.seat(player)` is the mirror entry point; same effect."""
        p = Player(player_id=4, stack=1000, name="Seat4")
        a = RandomAgent(name="R")
        a.seat(p)
        assert p.agent is a
        assert a.player_id == 4

    def test_reseating_overwrites_cleanly(self):
        """Seating a second agent into the same seat replaces the first."""
        p = Player(player_id=1, stack=1000, name="Seat1")
        first = CallAgent(name="first")
        second = CallAgent(name="second")
        p.seat_agent(first)
        p.seat_agent(second)
        assert p.agent is second
        assert second.player_id == 1
        # First agent retains its old player_id even though it's no longer in
        # the seat — that's fine; the env's source of truth is player.agent.


class TestWrapperUsesBidirectionalLink:
    """The wrapper's Dict[player_id, agent] is built via seat(); seated
    agents are reachable both from the wrapper map AND from player.agent."""

    def test_wrapper_seats_opponents_at_construction(self):
        from src.training.opponent_autoplay_wrapper import OpponentAutoPlayWrapper

        env = TexasHoldemEnv(num_players=3, starting_stack=1000,
                             small_blind=5, big_blind=10)
        opp_a = CallAgent(name="A")
        opp_b = CallAgent(name="B")
        wrapped = OpponentAutoPlayWrapper(env, [('call', opp_a), ('call', opp_b)])

        # Both opponents seated into the env's players, NOT player 0 (learner).
        non_learner_ids = [pid for pid in (0, 1, 2) if pid != env.learning_agent_id]
        seated_agents = [env.game_state.players[pid].agent for pid in non_learner_ids]
        assert opp_a in seated_agents
        assert opp_b in seated_agents
        # Each seated agent reports the right seat back.
        for pid in non_learner_ids:
            agent = env.game_state.players[pid].agent
            assert agent.player_id == pid

        # And the wrapper's lookup table agrees with what's on the players.
        for pid, agent in wrapped.opponents_by_id.items():
            assert env.game_state.players[pid].agent is agent

    def test_wrapper_rejects_mismatched_opponents_count(self):
        from src.training.opponent_autoplay_wrapper import OpponentAutoPlayWrapper

        env = TexasHoldemEnv(num_players=3, starting_stack=1000,
                             small_blind=5, big_blind=10)
        # 3 players → 2 non-learner seats; passing 1 opponent should error.
        with pytest.raises(ValueError, match="non-learner seats"):
            OpponentAutoPlayWrapper(env, [('call', CallAgent(name="only_one"))])


class TestLearningAgentIdValidation:
    def test_negative_id_rejected(self):
        with pytest.raises(ValueError, match="learning_agent_id"):
            TexasHoldemEnv(num_players=3, learning_agent_id=-1)

    def test_id_out_of_range_rejected(self):
        with pytest.raises(ValueError, match="learning_agent_id"):
            TexasHoldemEnv(num_players=3, learning_agent_id=3)

    def test_id_at_upper_bound_minus_one_accepted(self):
        env = TexasHoldemEnv(num_players=3, learning_agent_id=2)
        assert env.learning_agent_id == 2
