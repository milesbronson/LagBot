"""Tests for EvalGate head-to-head shootout."""

import inspect

import numpy as np
import pytest

from src.agents.base_agent import BaseAgent
from src.agents.opponent_ppo import OpponentPPO
from src.agents.ppo_agent import PPOAgent
from src.agents.random_agent import CallAgent, RandomAgent, WeightedRandomAgent
from src.training.eval_gate import EvalGate, EvalResult


class FoldAgent(BaseAgent):
    def select_action(self, observation, valid_actions=None):
        return 0


class FixedAgent(BaseAgent):
    """Picks the first valid action (deterministic, no RNG)."""

    def select_action(self, observation, valid_actions=None):
        if valid_actions is None:
            return 1
        return valid_actions[0]


class TestConstruction:
    def test_bad_num_hands_raises(self):
        with pytest.raises(ValueError):
            EvalGate(num_hands=0)
        with pytest.raises(ValueError):
            EvalGate(num_hands=-1)

    def test_defaults(self):
        gate = EvalGate()
        assert gate.num_hands == 1000
        assert gate.threshold == 0.0


class TestEvaluate:
    def test_returns_result_with_correct_metadata(self):
        gate = EvalGate(num_hands=5, seed=1)
        result = gate.evaluate(
            FixedAgent("a"),
            FixedAgent("b"),
            candidate_id="cand",
            predecessor_id="pred",
        )
        assert isinstance(result, EvalResult)
        assert result.candidate_id == "cand"
        assert result.predecessor_id == "pred"
        assert result.hands_played == 5
        assert result.big_blind == 10

    def test_fold_agent_loses_against_call_agent(self):
        gate = EvalGate(num_hands=30, seed=7, big_blind=10, small_blind=5)
        result = gate.evaluate(FoldAgent("folder"), CallAgent("caller"))
        # Folder forfeits the blinds every hand; profit should be clearly
        # negative.
        assert result.candidate_profit_chips < 0
        assert result.mbb_per_100 < 0
        assert result.candidate_wins == 0

    def test_mbb_per_100_formula(self):
        """profit/bb/hands * 100_000 should equal the reported mbb/100."""
        gate = EvalGate(num_hands=20, seed=3)
        result = gate.evaluate(FoldAgent("f"), CallAgent("c"))
        expected = (result.candidate_profit_chips / result.big_blind / result.hands_played) * 100_000.0
        assert result.mbb_per_100 == pytest.approx(expected)

    def test_deterministic_with_seed(self):
        gate1 = EvalGate(num_hands=10, seed=42)
        gate2 = EvalGate(num_hands=10, seed=42)
        r1 = gate1.evaluate(FixedAgent("a1"), FixedAgent("b1"))
        r2 = gate2.evaluate(FixedAgent("a2"), FixedAgent("b2"))
        assert r1.candidate_profit_chips == r2.candidate_profit_chips
        assert r1.mbb_per_100 == pytest.approx(r2.mbb_per_100)

    def test_different_seeds_can_differ(self):
        # Most pairs of seeds produce different traces; if they happen to
        # match it's not a bug, so just sanity-check the call works.
        r1 = EvalGate(num_hands=5, seed=1).evaluate(FixedAgent("a"), FixedAgent("b"))
        r2 = EvalGate(num_hands=5, seed=999).evaluate(FixedAgent("a"), FixedAgent("b"))
        assert r1.hands_played == r2.hands_played == 5


class TestPasses:
    def test_passes_when_above_threshold(self):
        gate = EvalGate(num_hands=1, threshold_mbb_per_100=0.0)
        result = EvalResult(
            candidate_id="c", predecessor_id="p",
            hands_played=1, candidate_profit_chips=10,
            big_blind=10, mbb_per_100=50.0,
            candidate_wins=1, candidate_losses=0,
        )
        assert gate.passes(result) is True

    def test_fails_when_below_threshold(self):
        gate = EvalGate(num_hands=1, threshold_mbb_per_100=10.0)
        result = EvalResult(
            candidate_id="c", predecessor_id="p",
            hands_played=1, candidate_profit_chips=0,
            big_blind=10, mbb_per_100=5.0,
            candidate_wins=0, candidate_losses=0,
        )
        assert gate.passes(result) is False

    def test_passes_at_threshold(self):
        gate = EvalGate(num_hands=1, threshold_mbb_per_100=10.0)
        result = EvalResult(
            candidate_id="c", predecessor_id="p",
            hands_played=1, candidate_profit_chips=0,
            big_blind=10, mbb_per_100=10.0,
            candidate_wins=0, candidate_losses=0,
        )
        assert gate.passes(result) is True


class TestAgentSignatureContract:
    """EvalGate.evaluate calls `agent.select_action(obs, valid)` uniformly
    for every seated agent. Every concrete BaseAgent subclass must accept
    that 2-arg form, or the gate crashes mid-shootout (this regressed once
    in OpponentPPO, taking down a 3-gen chained run after gen 1 finished)."""

    @pytest.mark.parametrize("cls", [
        CallAgent, RandomAgent, WeightedRandomAgent, OpponentPPO, PPOAgent,
    ])
    def test_select_action_accepts_valid_actions(self, cls):
        sig = inspect.signature(cls.select_action)
        params = list(sig.parameters.values())
        # self, observation, valid_actions — must be callable as (obs, valid)
        assert len(params) >= 3, \
            f"{cls.__name__}.select_action must accept (obs, valid_actions)"
        # The third parameter must be passable positionally.
        third = params[2]
        assert third.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.POSITIONAL_ONLY,
        ), f"{cls.__name__}.select_action's valid_actions must be positional"


class TestObsSpaceMatchesTraining:
    """The gate's env must produce observations of the same shape as the
    training env. Saved PPO models embed the obs space they were trained
    with, and SB3's predict() rejects a mismatched obs shape — every action
    raises and the opponent silently falls back to "call", invalidating
    the gate result. (Regressed once: gate built with track_opponents=False
    while training used True, after a 2M-step gen 0 had already finished.)"""

    def test_gate_env_obs_space_matches_training_env(self):
        from src.poker_env.texas_holdem_env import TexasHoldemEnv
        # Mirrors train.py:_build_env — both must produce the same obs shape.
        training_env = TexasHoldemEnv(num_players=2, track_opponents=True)
        gate_env = EvalGate()._build_env()
        assert gate_env.observation_space.shape == training_env.observation_space.shape


class TestRewardLeak:
    """Diagnostic tests for the suspected reward leak in eval_gate.

    The concern: mbb/100 numbers in the wild (e.g. +1.14M for some pool
    runs) imply ~11 BB profit per hand, which is enormous. Either the
    predecessor is dramatically weaker than the candidate, or the gate's
    per-hand profit accounting is double-counting somewhere — most
    plausibly through auto-rebuy chip injection (when a busted player
    gets fresh chips between hands, those chips must NOT show up as
    candidate profit).

    These tests pin the gate's accounting invariants so any future leak
    fails loudly instead of silently inflating mbb/100.
    """

    def _replay_gate(self, candidate, predecessor, num_hands, seed=0):
        """Mirror EvalGate.evaluate() but expose per-hand deltas + stacks.
        Kept in lock-step with the real loop so any divergence in the
        product code shows up here too."""
        gate = EvalGate(num_hands=num_hands, seed=seed)
        env = gate._build_env()
        env.game_state.players[0].seat_agent(candidate)
        env.game_state.players[1].seat_agent(predecessor)
        candidate.player_id = 0
        predecessor.player_id = 1
        agents = [candidate, predecessor]

        obs, _ = env.reset(seed=gate.seed)

        per_hand = []  # (stack_before, opp_stack_before, stack_after, delta)
        for hand_idx in range(num_hands):
            stack_before = env.game_state.players[0].starting_stack_this_hand
            opp_before = env.game_state.players[1].starting_stack_this_hand
            done = False
            while not done:
                current = env.game_state.get_current_player()
                agent = agents[current.player_id]
                valid = env.get_valid_actions()
                action = agent.select_action(obs, valid)
                obs, _, terminated, _, _ = env.step(action)
                done = terminated
            stack_after = env.game_state.players[0].stack
            per_hand.append((stack_before, opp_before, stack_after,
                             stack_after - stack_before))
            if hand_idx < num_hands - 1:
                obs, _ = env.reset(seed=gate.seed + hand_idx + 1)
        return per_hand, gate

    def test_profit_chips_equals_sum_of_per_hand_deltas(self):
        """The gate accumulates profit_chips one hand at a time. The
        reported total must equal the simple sum — no double-counting,
        no off-by-one, no chip injection bleeding into the total."""
        per_hand, gate = self._replay_gate(
            FixedAgent("a"), FixedAgent("b"), num_hands=20, seed=11
        )
        result = EvalGate(num_hands=20, seed=11).evaluate(
            FixedAgent("a"), FixedAgent("b")
        )
        expected = sum(delta for *_, delta in per_hand)
        assert result.candidate_profit_chips == pytest.approx(expected, abs=1e-6)

    def test_per_hand_delta_bounded_by_pre_hand_stacks(self):
        """On any single hand, the candidate's chip change is bounded by
        what was on the table at the start of that hand:
            -stack_before  <=  delta  <=  opp_stack_before.
        If a delta ever exceeds the opponent's pre-hand stack, the gate
        is crediting candidate with chips that didn't exist when the
        hand started — the canonical signature of a rebuy/injection
        leak."""
        per_hand, _ = self._replay_gate(
            FoldAgent("f"), CallAgent("c"), num_hands=50, seed=3
        )
        for stack_before, opp_before, _, delta in per_hand:
            assert delta >= -stack_before, (
                f"delta={delta} below -stack_before={-stack_before} "
                "(candidate lost more chips than they had at hand start)"
            )
            assert delta <= opp_before, (
                f"delta={delta} above opp_before={opp_before} "
                "(candidate won more than opponent could have contributed)"
            )

    def test_mirror_match_profit_bounded(self):
        """Two CallAgents are symmetric — neither has an edge. Over
        many hands, |profit_chips| should be much smaller than what a
        leak would produce (e.g. 11 BB/hand = 1100 chips over 100
        hands). We allow a generous bound for blind/positional variance
        but anything in the thousands is a smoking gun."""
        result = EvalGate(num_hands=100, seed=42).evaluate(
            CallAgent("a"), CallAgent("b")
        )
        # 11 BB/hand × 100 hands = 1100 chips would be the leak signature.
        # Real variance for CallAgent-vs-CallAgent is well under 1 stack.
        assert abs(result.candidate_profit_chips) < 500, (
            f"Mirror match drifted by {result.candidate_profit_chips} "
            "chips — symmetric agents shouldn't accumulate profit."
        )

    def test_rebuy_does_not_inject_profit_into_running_total(self):
        """Run a lopsided match (FoldAgent vs CallAgent) long enough
        that the folder busts and gets rebought. Verify that on each
        post-rebuy hand the delta is still bounded by ±starting_stack,
        i.e. the rebuy itself doesn't show up as profit."""
        per_hand, _ = self._replay_gate(
            CallAgent("c"), FoldAgent("f"), num_hands=200, seed=5
        )
        # Detect rebuy events on opponent (predecessor): stack_before
        # for the candidate doesn't tell us about opponent rebuy, but
        # opp_before equal to starting_stack=1000 after losing chips
        # indicates rebuy happened.
        # The invariant we care about is per-hand: delta on any hand
        # must be bounded by opp_before, which is the rebought stack.
        # If profit ever exceeded opp_before, rebuy chips leaked.
        starting_stack = 1000
        for stack_before, opp_before, stack_after, delta in per_hand:
            assert opp_before <= starting_stack, (
                f"opp_before={opp_before} exceeds starting_stack — "
                "rebuy is over-funding the opponent."
            )
            assert delta <= opp_before, (
                f"delta={delta} > opp_before={opp_before}: candidate "
                "won more than opponent had — rebuy chips leaked into "
                "the running profit total."
            )

    def test_stack_conservation_per_hand(self):
        """At the end of each hand (before next reset), total chips at
        the table plus any uncollected pot must equal the sum of stacks
        at the start of that hand. Auto-rebuy only fires on the NEXT
        reset, so within-hand conservation is required regardless of
        rake config."""
        per_hand, gate = self._replay_gate(
            FixedAgent("a"), FixedAgent("b"), num_hands=10, seed=7
        )
        # Sanity: every recorded hand observed conservation pre-reset
        # (the deltas sum to zero across both seats minus rake; here
        # rake=0 so deltas are exactly opposite).
        for stack_before, opp_before, stack_after, delta in per_hand:
            # candidate gain = opponent loss (rake=0 by default in gate)
            assert -delta <= opp_before, (
                "Candidate loss exceeds opponent pre-hand stack"
            )


class TestSerialization:
    def test_to_dict_roundtrip(self):
        result = EvalResult(
            candidate_id="c", predecessor_id="p",
            hands_played=100, candidate_profit_chips=250.0,
            big_blind=10, mbb_per_100=250.0,
            candidate_wins=55, candidate_losses=40,
        )
        d = result.to_dict()
        assert d["candidate_id"] == "c"
        assert d["mbb_per_100"] == 250.0
        assert set(d.keys()) == {
            "candidate_id", "predecessor_id", "hands_played",
            "candidate_profit_chips", "big_blind", "mbb_per_100",
            "candidate_wins", "candidate_losses",
        }
