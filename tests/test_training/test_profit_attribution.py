"""Per-opponent profit attribution contract.

OpponentAutoPlayWrapper._record_opponent_profits used to record
total_profit/N to every opponent, which made opponent_profits.json
identical across opponents by construction. These tests pin down the
chip-contribution-weighted attribution that replaced it."""

import pytest

from src.poker_env.texas_holdem_env import TexasHoldemEnv
from src.agents.random_agent import CallAgent, RandomAgent
from src.training.opponent_autoplay_wrapper import OpponentAutoPlayWrapper
from src.training.opponent_profit_tracker import OpponentProfitTracker


def _make_wrapper(tmp_path):
    env = TexasHoldemEnv(
        num_players=3, starting_stack=1000,
        small_blind=5, big_blind=10, track_opponents=False,
    )
    opps = [
        ("call", CallAgent(name="CallA")),
        ("random", RandomAgent(name="RandB")),
    ]
    tracker = OpponentProfitTracker("attr_test", save_dir=str(tmp_path))
    return env, opps, tracker, OpponentAutoPlayWrapper(env, opps, profit_tracker=tracker)


def test_attribution_proportional_to_chip_contribution(tmp_path):
    """When opponents put in different amounts, profit splits the same way."""
    env, opps, tracker, wrapper = _make_wrapper(tmp_path)
    wrapper.reset()
    learner_id = env.learning_agent_id
    opp_ids = [pid for pid in (0, 1, 2) if pid != learner_id]

    # Manually pose: learner won 100 chips, opp_ids[0] put in 30, opp_ids[1] put in 70.
    # We have to bypass live play; just set hand_starting_stack and stacks.
    wrapper.hand_starting_stack = env.starting_stack - 100  # learner is +100
    env.game_state.players[learner_id].stack = env.starting_stack  # net +100
    env.game_state.players[opp_ids[0]].total_bet_this_hand = 30
    env.game_state.players[opp_ids[1]].total_bet_this_hand = 70

    wrapper._record_opponent_profits(env.game_state.players[learner_id])

    results = {pid: tracker.opponent_results[pid] for pid in opp_ids}
    # total_profit normalized = 100 / 1000 = 0.1
    # share[0] = 30/100 = 0.3, share[1] = 0.7
    assert results[opp_ids[0]]["total_profit"] == pytest.approx(0.1 * 0.3)
    assert results[opp_ids[1]]["total_profit"] == pytest.approx(0.1 * 0.7)
    assert (results[opp_ids[0]]["total_profit"] +
            results[opp_ids[1]]["total_profit"]) == pytest.approx(0.1)


def test_attribution_uniform_when_no_opponent_contributed(tmp_path):
    """Edge case: degenerate hand where no opponent put chips in.
    Both opponents should get equal credit (uniform fallback)."""
    env, opps, tracker, wrapper = _make_wrapper(tmp_path)
    wrapper.reset()
    learner_id = env.learning_agent_id
    opp_ids = [pid for pid in (0, 1, 2) if pid != learner_id]

    wrapper.hand_starting_stack = env.starting_stack
    env.game_state.players[learner_id].stack = env.starting_stack + 50
    for pid in opp_ids:
        env.game_state.players[pid].total_bet_this_hand = 0

    wrapper._record_opponent_profits(env.game_state.players[learner_id])

    a = tracker.opponent_results[opp_ids[0]]["total_profit"]
    b = tracker.opponent_results[opp_ids[1]]["total_profit"]
    assert a == pytest.approx(b), "uniform fallback should split evenly"
    assert (a + b) == pytest.approx(0.05)


def test_attribution_loss_when_one_opponent_pressured(tmp_path):
    """If learner lost 100 and only one opponent put chips in, that
    opponent should absorb the entire loss attribution."""
    env, opps, tracker, wrapper = _make_wrapper(tmp_path)
    wrapper.reset()
    learner_id = env.learning_agent_id
    opp_ids = [pid for pid in (0, 1, 2) if pid != learner_id]

    wrapper.hand_starting_stack = env.starting_stack
    env.game_state.players[learner_id].stack = env.starting_stack - 100
    env.game_state.players[opp_ids[0]].total_bet_this_hand = 100  # the bettor
    env.game_state.players[opp_ids[1]].total_bet_this_hand = 0    # folded immediately

    wrapper._record_opponent_profits(env.game_state.players[learner_id])

    pressured = tracker.opponent_results[opp_ids[0]]["total_profit"]
    folded   = tracker.opponent_results[opp_ids[1]]["total_profit"]
    assert pressured == pytest.approx(-0.1)
    assert folded == pytest.approx(0.0)


def test_attribution_sum_equals_total_profit(tmp_path):
    """Invariant: per-opponent shares always sum to learner total profit."""
    env, opps, tracker, wrapper = _make_wrapper(tmp_path)
    wrapper.reset()
    learner_id = env.learning_agent_id
    opp_ids = [pid for pid in (0, 1, 2) if pid != learner_id]

    wrapper.hand_starting_stack = env.starting_stack
    env.game_state.players[learner_id].stack = env.starting_stack + 250
    env.game_state.players[opp_ids[0]].total_bet_this_hand = 80
    env.game_state.players[opp_ids[1]].total_bet_this_hand = 170

    wrapper._record_opponent_profits(env.game_state.players[learner_id])

    total = sum(tracker.opponent_results[pid]["total_profit"] for pid in opp_ids)
    assert total == pytest.approx(0.25)
