"""Counterfactual-fold-regret reward augmentation.

The default profit reward returned by TexasHoldemEnv is the chip delta
over the hand, normalised by the starting stack. A pure-profit signal
biases the critic toward the on-policy distribution: if the policy
never folds, V(s) reflects only the value of the chosen (raise/call)
trajectory and the gradient cannot tell the policy that folding was
the better choice in expectation.

This wrapper injects the counterfactual signal CFR uses. At each
agent decision point we record the learner's chip investment. At
hand-end we compute, for each prior decision d:

    fold_value(d) = -invested(d) / starting_stack
    regret(d)     = max(0, fold_value(d) - actual_reward)

and aggregate by taking the maximum (the single worst missed fold).
Aggregating by max — not sum — matches the credit-assignment intent:
the deepest missed fold dominates and avoids double-counting nested
sub-decisions.

The augmented reward is applied at the terminal step; GAE then
distributes the signal back to every action in the hand.

Modes:
  - profit       : pass through (identical to baseline training).
  - regret_blend : reward - lambda * max_regret.  **Recommended.**
                   Matches the StepScorer / PRM family of hybrid
                   regret-shaping methods: keep the profit gradient
                   ("winning more is better") and add a dense penalty
                   for missed-fold decisions. lambda controls strength;
                   start at 1.0 for our LAG-collapse case (reward and
                   max_regret are both in [-1, +1] / [0, 1]).
  - regret_only  : -max_regret.  Drops profit entirely. Use ONLY as a
                   diagnostic — without the profit gradient the bot
                   cannot tell "won big" from "broke even" (both map
                   to 0) and converges to fold-everything play. Not a
                   defensible curriculum reward.
"""

from typing import List, Optional, Tuple

import gymnasium as gym


MODES = ("profit", "regret_blend", "regret_only")


class RegretRewardWrapper(gym.Env):
    def __init__(
        self,
        inner_env: gym.Env,
        mode: str = "profit",
        regret_lambda: float = 1.0,
    ):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if regret_lambda < 0:
            raise ValueError("regret_lambda must be >= 0")

        self.env = inner_env
        self.mode = mode
        self.regret_lambda = float(regret_lambda)

        self.observation_space = inner_env.observation_space
        self.action_space = inner_env.action_space
        self.metadata = getattr(inner_env, "metadata", {})

        self._invested_at_decision: List[float] = []
        self._starting_stack: Optional[float] = None
        self._learner_id: int = self._resolve_learner_id()

    def _resolve_learner_id(self) -> int:
        """Walk the wrapper chain to find learning_agent_id on the
        underlying TexasHoldemEnv."""
        cur = self.env
        for _ in range(8):
            if hasattr(cur, "learning_agent_id"):
                return cur.learning_agent_id
            if hasattr(cur, "learner_id"):
                return cur.learner_id
            cur = getattr(cur, "env", None)
            if cur is None:
                break
        raise AttributeError(
            "could not find learner id on inner env chain"
        )

    def _base_env(self):
        cur = self.env
        for _ in range(8):
            if hasattr(cur, "game_state"):
                return cur
            cur = getattr(cur, "env", None)
            if cur is None:
                break
        raise AttributeError("could not find base env with game_state")

    def _learner_invested_now(self) -> float:
        base = self._base_env()
        player = base.game_state.players[self._learner_id]
        return float(player.total_bet_this_hand)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._invested_at_decision = []
        # Normalize fold-value by the CONFIG starting stack, matching the
        # env's own profit term (texas_holdem_env divides by the config
        # value). Using the per-hand stack put the two regret_blend terms
        # on different scales whenever stacks drifted from the default.
        self._starting_stack = float(self._base_env().starting_stack)
        return obs, info

    def step(self, action: int) -> Tuple:
        # Snapshot pre-action investment — the chip total the learner
        # would forfeit by folding instead of taking ``action``.
        self._invested_at_decision.append(self._learner_invested_now())

        obs, reward, terminated, truncated, info = self.env.step(action)

        # Snapshot the unshaped env reward (per-hand profit) before any
        # regret penalty is applied. The MetricsCallback sums this across a
        # hand to chart raw winnings alongside the shaped training reward,
        # so a negative shaped reward (profit minus an always->=0 regret
        # penalty) isn't mistaken for losing money.
        raw_step_reward = float(reward)

        if terminated or truncated:
            max_regret = self._compute_max_regret(actual_reward=reward)
            info["regret_max"] = max_regret
            info["regret_lambda"] = self.regret_lambda
            info["regret_mode"] = self.mode

            if self.mode == "regret_blend":
                reward = reward - self.regret_lambda * max_regret
            elif self.mode == "regret_only":
                reward = -max_regret
            # "profit" -> unchanged

        info["raw_step_reward"] = raw_step_reward
        return obs, reward, terminated, truncated, info

    def _compute_max_regret(self, actual_reward: float) -> float:
        if not self._invested_at_decision or self._starting_stack in (None, 0):
            return 0.0
        worst = 0.0
        denom = float(self._starting_stack)
        for invested in self._invested_at_decision:
            fold_value = -invested / denom
            regret = fold_value - actual_reward
            if regret > worst:
                worst = regret
        return worst

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self):
        return self.env.close()

    def __getattr__(self, name):
        # Forward unknown attribute lookups to the inner env so callers
        # that reach for ``snapshot_card_stats`` / ``opponent_tracker``
        # etc. keep working transparently.
        if name == "env":
            raise AttributeError(name)
        return getattr(self.env, name)
