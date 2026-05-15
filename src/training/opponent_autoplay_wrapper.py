"""
OpponentAutoPlayWrapper — gym wrapper around TexasHoldemEnv that drives
the non-learner seats automatically.

Used by the unified training entrypoint. SB3 only needs to step the
learner; this wrapper takes care of:

- Seating each opponent agent into its Player (bidirectional link).
- Auto-playing every non-learner seat after the learner acts (and at
  the top of each hand) until it is the learner's turn or the hand
  ends.
- Accumulating env-attributed rewards across opponent steps.
- (Optionally) recording per-opponent chip profit into an
  OpponentProfitTracker.
- (Optionally) rotating opponents per episode via an
  ``opponent_factory`` callable. Rotation parks the previous occupant's
  OpponentTracker profile under its card_id and either restores the
  newly-seated card's cached profile or seeds fresh priors from the
  registry. This is the "league" knob: lets one training run see many
  opponents so the policy can learn opponent-conditional play.
"""

from typing import Callable, Dict, List, Optional, Tuple

import gymnasium as gym

from src.poker_env.opponent_tracker import OpponentProfile
from src.poker_env.texas_holdem_env import TexasHoldemEnv
from src.training.agent_card import AgentCard
from src.training.opponent_profit_tracker import OpponentProfitTracker


# Factory signature: factory(seat_id) -> (kind, agent, card)
OpponentFactory = Callable[[int], Tuple[str, object, AgentCard]]


class OpponentAutoPlayWrapper(gym.Env):
    def __init__(
        self,
        env: TexasHoldemEnv,
        opponents_list: Optional[List[Tuple[str, object]]] = None,
        profit_tracker: Optional[OpponentProfitTracker] = None,
        opponent_factory: Optional[OpponentFactory] = None,
        seat_to_card: Optional[Dict[int, AgentCard]] = None,
    ):
        """
        Args:
            env: TexasHoldemEnv instance.
            opponents_list: Static seating: list of (kind, agent) tuples
                in seat order for non-learner seats. Use this OR
                ``opponent_factory`` (mutually exclusive).
            profit_tracker: Optional tracker for per-opponent profits.
                Per-seat attribution becomes meaningless under rotation,
                so the wrapper does not record profits when a factory is
                supplied.
            opponent_factory: Per-episode opponent sampling. Called on
                every ``reset()`` for each non-learner seat to obtain a
                fresh ``(kind, agent, card)`` triple. The wrapper handles
                stashing and restoring per-card OpponentTracker profiles
                across rotations.
            seat_to_card: Static-mode card mapping. Required when
                ``opponents_list`` is used and the caller wants card-keyed
                snapshots from ``snapshot_card_stats()``.
        """
        super().__init__()
        if (opponents_list is None) == (opponent_factory is None):
            raise ValueError(
                "must pass exactly one of opponents_list or opponent_factory"
            )

        self.env = env
        self.profit_tracker = profit_tracker
        self.opponent_factory = opponent_factory

        self.learner_id = env.learning_agent_id
        self.non_learner_ids = [
            p.player_id for p in env.game_state.players
            if p.player_id != self.learner_id
        ]

        # Per-card persistent state across rotations.
        self.card_profiles: Dict[str, OpponentProfile] = {}
        self.card_metadata: Dict[str, AgentCard] = {}
        self.agent_cache: Dict[str, object] = {}
        self.seat_to_card_id: Dict[int, Optional[str]] = {
            pid: None for pid in self.non_learner_ids
        }

        self.opponents_by_id: Dict[int, object] = {}
        self.opponent_types_by_id: Dict[int, str] = {}

        if opponents_list is not None:
            if len(opponents_list) != len(self.non_learner_ids):
                raise ValueError(
                    f"opponents_list has {len(opponents_list)} entries but env "
                    f"has {len(self.non_learner_ids)} non-learner seats"
                )
            for pid, (kind, agent) in zip(self.non_learner_ids, opponents_list):
                agent.seat(env.game_state.players[pid])
                self.opponents_by_id[pid] = agent
                self.opponent_types_by_id[pid] = kind
            self.opponents = opponents_list

            if seat_to_card is not None:
                for pid, card in seat_to_card.items():
                    self.seat_to_card_id[pid] = card.id
                    self.card_metadata[card.id] = card
        else:
            # Rotation mode: defer factory invocation to reset(). Calling
            # it in __init__ as well would burn one draw per seat without
            # ever playing it, shifting any test schedule by one.
            self.opponents = None

        self.hand_starting_stack = None

        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.metadata = env.metadata

    def _rotate_in_new_opponents(self) -> None:
        """Stash current per-seat OpponentTracker profiles under their
        card_id, then sample a new opponent per seat and either restore
        that card's cached profile or seed priors from the registry."""
        tracker = self.env.opponent_tracker
        for pid in self.non_learner_ids:
            prev_card_id = self.seat_to_card_id.get(pid)
            if prev_card_id is not None and pid in tracker.opponents:
                # Park the previous opponent's accumulated profile so a
                # later rotation that re-seats this card resumes its
                # stats rather than starting from zero.
                self.card_profiles[prev_card_id] = tracker.opponents[pid]

            kind, agent, card = self.opponent_factory(pid)
            self.seat_to_card_id[pid] = card.id
            self.card_metadata[card.id] = card
            # Cache instantiated agents — loading a PPO model from disk
            # every episode would dominate runtime. The factory is
            # expected to return cached agents, but we hold a backup
            # reference here too.
            self.agent_cache[card.id] = agent

            if card.id in self.card_profiles:
                tracker.opponents[pid] = self.card_profiles[card.id]
            else:
                tracker.opponents.pop(pid, None)

            prior_stats = card.behavior_stats or {}
            if int(prior_stats.get("hands_observed", 0)) > 0:
                tracker.priors[pid] = dict(prior_stats)
            else:
                tracker.priors.pop(pid, None)

            agent.seat(self.env.game_state.players[pid])
            self.opponents_by_id[pid] = agent
            self.opponent_types_by_id[pid] = kind

    def reset(self, **kwargs):
        if self.opponent_factory is not None:
            self._rotate_in_new_opponents()

        obs, info = self.env.reset(**kwargs)

        if self.profit_tracker and self.opponent_factory is None:
            learning_player = self.env.game_state.players[self.learner_id]
            self.hand_starting_stack = learning_player.starting_stack_this_hand

        # Without this the first step() would apply the learner's action
        # to an opponent seat when the button rotates.
        obs, _, terminated, truncated, info = self._auto_play_opponents(
            obs, terminated=False, truncated=False, info=info
        )
        return obs, info

    def step(self, action: int) -> Tuple:
        """Env owns reward attribution. The wrapper passes env reward
        through unchanged — do not recompute it here."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Stash the learner-specific step info before auto-play overwrites
        # the dict with opponent actions. Per-street action breakdown in
        # MetricsCallback reads these keys to know which street the
        # learner's action happened on.
        learner_info = {
            "learner_action": info.get("action"),
            "learner_street": info.get("street"),
        }

        if terminated or truncated:
            self._record_opponent_profits(
                self.env.game_state.players[self.learner_id]
            )
            info.update(learner_info)
            return obs, reward, terminated, truncated, info

        obs, opp_reward, terminated, truncated, info = self._auto_play_opponents(
            obs, terminated, truncated, info
        )
        reward = reward + opp_reward
        info.update(learner_info)

        if terminated or truncated:
            self._record_opponent_profits(
                self.env.game_state.players[self.learner_id]
            )

        return obs, reward, terminated, truncated, info

    def _auto_play_opponents(self, obs, terminated, truncated, info):
        accumulated_reward = 0.0
        while not (terminated or truncated) and \
                self.env.game_state.current_player_idx != self.learner_id:
            current_idx = self.env.game_state.current_player_idx
            agent = self.opponents_by_id.get(current_idx)
            if agent is None:
                break
            opponent_action = agent.select_action(obs)
            obs, opp_reward, terminated, truncated, info = \
                self.env.step(opponent_action)
            accumulated_reward += opp_reward
        return obs, accumulated_reward, terminated, truncated, info

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self):
        return self.env.close()

    def snapshot_card_stats(self) -> Dict[str, Dict]:
        """Per-card behavior-stat snapshot suitable for
        ``AgentRegistry.update_behavior_stats``. Flushes the live
        tracker profiles into the per-card cache first so the currently
        seated opponents' freshest stats are included.

        Returns:
            Dict mapping card_id -> {hands_observed, vpip, pfr, af, ...}.
            Empty dict if no card metadata is known (e.g. static mode
            constructed without seat_to_card).
        """
        tracker = self.env.opponent_tracker
        for pid, card_id in self.seat_to_card_id.items():
            if card_id is None:
                continue
            if pid in tracker.opponents:
                self.card_profiles[card_id] = tracker.opponents[pid]

        out: Dict[str, Dict] = {}
        for card_id, profile in self.card_profiles.items():
            out[card_id] = {
                "hands_observed": int(profile.hands_played),
                "vpip": float(profile.vpip),
                "pfr": float(profile.pfr),
                "af": float(profile.af),
                "three_bet_percent": float(profile.three_bet_percent),
                "cbet_percent": float(profile.cbet_percent),
                "fold_to_cbet_percent": float(profile.fold_to_cbet_percent),
                "went_to_showdown_percent": float(profile.went_to_showdown_percent),
                "win_at_showdown_percent": float(profile.win_at_showdown_percent),
                "wwsf_percent": float(profile.wwsf_percent),
                "fold_to_3bet_after_raise_percent": float(
                    profile.fold_to_3bet_after_raise_percent
                ),
                "squeeze_percent": float(profile.squeeze_percent),
            }
        return out

    def _record_opponent_profits(self, learning_agent):
        if not self.profit_tracker or self.hand_starting_stack is None:
            return
        # Skip profit attribution when opponents rotate — per-seat
        # tracking aliases multiple cards into the same bucket and
        # corrupts matchup_history. Per-card profit attribution is a
        # follow-on feature.
        if self.opponent_factory is not None:
            return

        try:
            total_profit = (learning_agent.stack - self.hand_starting_stack) / self.env.starting_stack

            # Attribute the learner's profit per opponent by chips-into-pot share.
            # When learner wins: opponent who paid more chips contributed more
            # to the win, so they get more credit. When learner loses: opponent
            # who pressured the pot harder gets more of the blame.
            # The previous "uniform total_profit / N" scheme made all opponents
            # carry identical numbers by construction, so opponent_profits.json
            # could never show per-opponent differentiation.
            contributions = {
                opp_id: self.env.game_state.players[opp_id].total_bet_this_hand
                for opp_id in self.opponents_by_id
            }
            total_opp_contribution = sum(contributions.values())

            for opponent_id, agent in self.opponents_by_id.items():
                opponent_type = self.opponent_types_by_id[opponent_id]
                opponent_name = getattr(agent, 'name', f'Opponent_{opponent_id}')

                if opponent_id not in self.profit_tracker.opponent_types:
                    self.profit_tracker.register_opponent(
                        opponent_id=opponent_id,
                        name=opponent_name,
                        opponent_type=opponent_type,
                    )

                if total_opp_contribution > 0:
                    share = contributions[opponent_id] / total_opp_contribution
                else:
                    # No opponent put chips in — degenerate hand. Fall back to
                    # uniform so the bookkeeping still balances.
                    share = 1.0 / len(self.opponents_by_id)

                self.profit_tracker.record_hand_result(
                    opponent_id=opponent_id,
                    profit=total_profit * share,
                )
        except Exception as e:
            print(f"Warning: Failed to record opponent profit: {e}")
