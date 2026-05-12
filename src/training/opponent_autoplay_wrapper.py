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
"""

from typing import List, Tuple, Optional

import gymnasium as gym

from src.poker_env.texas_holdem_env import TexasHoldemEnv
from src.training.opponent_profit_tracker import OpponentProfitTracker


class OpponentAutoPlayWrapper(gym.Env):
    def __init__(
        self,
        env: TexasHoldemEnv,
        opponents_list: List[Tuple[str, object]],
        profit_tracker: Optional[OpponentProfitTracker] = None,
    ):
        """
        Args:
            env: TexasHoldemEnv instance.
            opponents_list: List of (opponent_type, agent) tuples in
                seat order for the non-learner seats (i.e. every
                player whose player_id != env.learning_agent_id).
            profit_tracker: Optional tracker for per-opponent profits.
        """
        super().__init__()
        self.env = env
        self.profit_tracker = profit_tracker

        self.learner_id = env.learning_agent_id

        non_learner_ids = [p.player_id for p in env.game_state.players
                           if p.player_id != self.learner_id]
        if len(opponents_list) != len(non_learner_ids):
            raise ValueError(
                f"opponents_list has {len(opponents_list)} entries but env "
                f"has {len(non_learner_ids)} non-learner seats"
            )

        self.opponents_by_id = {}
        self.opponent_types_by_id = {}
        for player_id, (opponent_type, agent) in zip(non_learner_ids, opponents_list):
            agent.seat(env.game_state.players[player_id])
            self.opponents_by_id[player_id] = agent
            self.opponent_types_by_id[player_id] = opponent_type

        self.opponents = opponents_list

        self.hand_starting_stack = None

        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.metadata = env.metadata

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        if self.profit_tracker:
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

    def _record_opponent_profits(self, learning_agent):
        if not self.profit_tracker or self.hand_starting_stack is None:
            return

        try:
            total_profit = (learning_agent.stack - self.hand_starting_stack) / self.env.starting_stack
            per_opponent_profit = total_profit / len(self.opponents_by_id)

            for opponent_id, agent in self.opponents_by_id.items():
                opponent_type = self.opponent_types_by_id[opponent_id]
                opponent_name = getattr(agent, 'name', f'Opponent_{opponent_id}')

                if opponent_id not in self.profit_tracker.opponent_types:
                    self.profit_tracker.register_opponent(
                        opponent_id=opponent_id,
                        name=opponent_name,
                        opponent_type=opponent_type,
                    )

                self.profit_tracker.record_hand_result(
                    opponent_id=opponent_id,
                    profit=per_opponent_profit,
                )
        except Exception as e:
            print(f"Warning: Failed to record opponent profit: {e}")
