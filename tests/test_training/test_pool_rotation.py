"""
Tests for per-episode opponent rotation in OpponentAutoPlayWrapper —
the "league" knob. A factory swaps fresh opponents into each non-learner
seat on every reset(), and the wrapper is responsible for stashing /
restoring per-card OpponentTracker profiles so the same card resumes
its accumulated stats whenever it cycles back in.
"""

import pytest

from src.agents.random_agent import CallAgent, RandomAgent
from src.poker_env.texas_holdem_env import TexasHoldemEnv
from src.training.agent_card import AgentCard
from src.training.opponent_autoplay_wrapper import OpponentAutoPlayWrapper


def _play_one_hand(wrapped_env, max_steps=100):
    """Bash through a hand using the call/check action so the test
    finishes deterministically. Returns whether the hand actually ended."""
    wrapped_env.reset()
    for _ in range(max_steps):
        _, _, terminated, truncated, _ = wrapped_env.step(1)
        if terminated or truncated:
            return True
    return False


@pytest.fixture
def env():
    return TexasHoldemEnv(
        num_players=3, starting_stack=1000,
        small_blind=1, big_blind=2, track_opponents=True,
    )


class _ScriptedFactory:
    """Cycle through a fixed list of (kind, agent_factory, card) tuples
    per seat. Each call returns a fresh instantiated agent so the test
    doesn't need to share BaseAgent instances across seats."""

    def __init__(self, schedule):
        self.schedule = schedule
        self.calls_per_seat = {}

    def __call__(self, seat_id):
        idx = self.calls_per_seat.get(seat_id, 0)
        seat_schedule = self.schedule[seat_id]
        kind, agent_cls, card = seat_schedule[idx % len(seat_schedule)]
        self.calls_per_seat[seat_id] = idx + 1
        return kind, agent_cls(name=card.id), card


class TestConstructorValidation:
    def test_rejects_both_modes(self, env):
        with pytest.raises(ValueError, match="exactly one"):
            OpponentAutoPlayWrapper(
                env,
                opponents_list=[('call', CallAgent()), ('call', CallAgent())],
                opponent_factory=lambda pid: ('call', CallAgent(), AgentCard(
                    id="x", name="x", kind="call",
                )),
            )

    def test_rejects_neither_mode(self, env):
        with pytest.raises(ValueError, match="exactly one"):
            OpponentAutoPlayWrapper(env)


class TestFactoryInvocation:
    def test_factory_called_on_each_reset(self, env):
        """The factory fires once per seat on each reset(). It does NOT
        fire at __init__ — calling it there too would burn one draw per
        seat without ever playing it, shifting any rotation schedule."""
        card = AgentCard(id="c", name="c", kind="call")
        calls = {1: 0, 2: 0}

        def factory(seat_id):
            calls[seat_id] += 1
            return 'call', CallAgent(name="c"), card

        wrapped = OpponentAutoPlayWrapper(env, opponent_factory=factory)
        assert calls == {1: 0, 2: 0}

        wrapped.reset()
        assert calls == {1: 1, 2: 1}

        wrapped.reset()
        assert calls == {1: 2, 2: 2}

    def test_static_mode_does_not_invoke_factory(self, env):
        """Sanity check: passing opponents_list keeps the old behavior."""
        wrapped = OpponentAutoPlayWrapper(
            env,
            opponents_list=[
                ('call', CallAgent()),
                ('call', CallAgent()),
            ],
        )
        assert wrapped.opponent_factory is None
        # Wrapper should still play through a hand without error.
        assert _play_one_hand(wrapped)


class TestProfilePersistence:
    def test_same_card_resumes_profile_after_rotation(self, env):
        """After A occupies a seat and accumulates some live stats,
        rotating B in then back to A should restore A's profile rather
        than starting over from zero."""
        card_a = AgentCard(id="a", name="a", kind="call")
        card_b = AgentCard(id="b", name="b", kind="call")

        # Seat 1 alternates A, B, A. Seat 2 stays on a third card.
        card_c = AgentCard(id="c", name="c", kind="call")
        factory = _ScriptedFactory({
            1: [('call', CallAgent, card_a),
                ('call', CallAgent, card_b),
                ('call', CallAgent, card_a)],
            2: [('call', CallAgent, card_c)],
        })

        wrapped = OpponentAutoPlayWrapper(env, opponent_factory=factory)

        # Hand 1: A in seat 1.
        assert _play_one_hand(wrapped)
        tracker = env.opponent_tracker
        assert 1 in tracker.opponents
        a_hands_after_h1 = tracker.opponents[1].hands_played
        assert a_hands_after_h1 >= 1

        # Hand 2: B rotates into seat 1. A's profile should be stashed.
        assert _play_one_hand(wrapped)
        assert wrapped.card_profiles["a"].hands_played == a_hands_after_h1
        b_hands_after_h2 = tracker.opponents[1].hands_played
        # B is fresh in this seat, so live profile starts from this hand.
        assert b_hands_after_h2 >= 1

        # Hand 3: A returns. Tracker should resume A's earlier stats.
        wrapped.reset()
        # The wrapper restores the cached profile BEFORE env.reset() runs
        # — but env.reset() calls tracker.start_hand which mutates the
        # live profile. So we assert via the cumulative hands_played not
        # dropping back to zero.
        assert tracker.opponents[1].hands_played >= a_hands_after_h1, (
            "A's stashed profile should have been restored on rotation"
        )

    def test_stash_uses_card_id_not_seat_id(self, env):
        """If the same card occupies different seats across rotations,
        the cache should still find its profile by card_id."""
        card_x = AgentCard(id="x", name="x", kind="call")
        card_y = AgentCard(id="y", name="y", kind="call")

        # Round 1: X in seat 1, Y in seat 2.
        # Round 2: Y in seat 1, X in seat 2.
        factory = _ScriptedFactory({
            1: [('call', CallAgent, card_x), ('call', CallAgent, card_y)],
            2: [('call', CallAgent, card_y), ('call', CallAgent, card_x)],
        })

        wrapped = OpponentAutoPlayWrapper(env, opponent_factory=factory)
        assert _play_one_hand(wrapped)

        # Both cards should now have stashed/live profiles somewhere.
        # After rotation, the previous occupants get parked under their
        # card_id, then the new occupants' cached profiles (if any) get
        # restored. Either way, both card ids should be tracked.
        assert _play_one_hand(wrapped)

        # After the rotation finishes, every card the factory has ever
        # surfaced should have made it into the cache.
        assert "x" in wrapped.card_profiles or "x" in {
            wrapped.seat_to_card_id[pid] for pid in wrapped.non_learner_ids
        }
        assert "y" in wrapped.card_profiles or "y" in {
            wrapped.seat_to_card_id[pid] for pid in wrapped.non_learner_ids
        }


class TestPriorSeeding:
    def test_seed_priors_called_with_card_behavior_stats(self, env):
        """A card carrying nontrivial behavior_stats should land in the
        OpponentTracker.priors dict for its seat."""
        card = AgentCard(
            id="seeded", name="seeded", kind="call",
            behavior_stats={"hands_observed": 50, "vpip": 0.7, "pfr": 0.3},
        )
        plain = AgentCard(id="plain", name="plain", kind="call")

        factory = _ScriptedFactory({
            1: [('call', CallAgent, card)],
            2: [('call', CallAgent, plain)],
        })
        wrapped = OpponentAutoPlayWrapper(env, opponent_factory=factory)
        wrapped.reset()

        assert 1 in env.opponent_tracker.priors
        assert env.opponent_tracker.priors[1]["vpip"] == 0.7
        # The card with default empty behavior_stats should NOT seed priors.
        assert 2 not in env.opponent_tracker.priors


class TestSnapshotCardStats:
    def test_rotation_snapshot_keyed_by_card_id(self, env):
        card_a = AgentCard(id="a", name="a", kind="call")
        card_b = AgentCard(id="b", name="b", kind="call")
        card_c = AgentCard(id="c", name="c", kind="call")

        factory = _ScriptedFactory({
            1: [('call', CallAgent, card_a), ('call', CallAgent, card_b)],
            2: [('call', CallAgent, card_c)],
        })
        wrapped = OpponentAutoPlayWrapper(env, opponent_factory=factory)

        assert _play_one_hand(wrapped)  # A, C
        assert _play_one_hand(wrapped)  # B, C

        snap = wrapped.snapshot_card_stats()
        # Every card that ever occupied a seat should show up.
        assert {"a", "b", "c"}.issubset(snap.keys())
        for entry in snap.values():
            assert "hands_observed" in entry
            assert "vpip" in entry

    def test_static_mode_with_seat_to_card_returns_card_keyed(self, env):
        """Static mode populated with seat_to_card should still produce
        a card-keyed snapshot so train.py can use the same write-back
        path in both modes."""
        card_a = AgentCard(id="static_a", name="a", kind="call")
        card_b = AgentCard(id="static_b", name="b", kind="call")
        wrapped = OpponentAutoPlayWrapper(
            env,
            opponents_list=[('call', CallAgent()), ('call', CallAgent())],
            seat_to_card={1: card_a, 2: card_b},
        )
        assert _play_one_hand(wrapped)

        snap = wrapped.snapshot_card_stats()
        assert "static_a" in snap
        assert "static_b" in snap

    def test_static_mode_without_seat_to_card_returns_empty(self, env):
        wrapped = OpponentAutoPlayWrapper(
            env,
            opponents_list=[('call', CallAgent()), ('call', CallAgent())],
        )
        assert _play_one_hand(wrapped)
        assert wrapped.snapshot_card_stats() == {}


class TestProfitTrackerSkipsRotation:
    def test_no_profit_recording_in_rotation_mode(self, env):
        """Per-seat profit attribution aliases multiple cards into the
        same bucket under rotation, so the wrapper must skip it even
        if a profit_tracker is supplied."""
        from src.training.opponent_profit_tracker import OpponentProfitTracker

        card = AgentCard(id="x", name="x", kind="call")
        factory = _ScriptedFactory({
            1: [('call', CallAgent, card)],
            2: [('call', CallAgent, card)],
        })
        tracker = OpponentProfitTracker("test_rotation_skip", save_dir="/tmp")
        wrapped = OpponentAutoPlayWrapper(
            env, opponent_factory=factory, profit_tracker=tracker,
        )
        assert _play_one_hand(wrapped)
        assert _play_one_hand(wrapped)

        # No opponent_results should accumulate when the factory is set.
        assert tracker.opponent_results == {} or all(
            v["hands_played"] == 0 for v in tracker.opponent_results.values()
        )
