"""
Enhanced training callbacks for metrics collection
"""

import json
import os
from typing import Optional

from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

from src.training.metrics import TrainingMetrics
from src.training.opponent_profit_tracker import OpponentProfitTracker


class MetricsCallback(BaseCallback):
    """Callback that logs training metrics to dashboard"""

    STREETS = ("preflop", "flop", "turn", "river")

    def __init__(self, metrics: TrainingMetrics, log_freq: int = 10000, verbose: int = 0):
        super().__init__(verbose)
        self.metrics = metrics
        self.log_freq = log_freq
        self.episode_rewards = []
        self.current_episode_reward = 0  # Track current episode reward
        self.episode_actions = []  # Track actions taken
        # Paired (action, street) records for per-street breakdown. Same
        # length as episode_actions when both fields are populated; entries
        # whose street is unknown are bucketed under "unknown".
        self.episode_action_streets = []
        self.episode_wins = 0
        self.episode_count = 0
        self.last_logged_step = 0
        # Cached after _on_training_start: action-bucket indices read off
        # the env's actual action space so we don't hard-code 6-action
        # layout. Populated lazily because env isn't bound yet at __init__.
        self._action_buckets = None
            

    def set_model(self, model) -> None:
        """Bind the SB3 model. BaseCallback exposes `self.model` as a simple
        attribute (no setter), so plain assignment is the correct hook."""
        self.model = model

    def _on_step(self) -> bool:
        """Called at each step"""
        # Get info from the last step
        infos = self.locals.get('infos', [])
        dones = self.locals.get('dones', [])
        rewards = self.locals.get('rewards', [])

        # Track actions taken (with street, if the wrapper exposed it).
        # Each call to _on_step corresponds to ONE learner action, so we
        # pair the action with the street from infos[0]['learner_street'].
        if 'actions' in self.locals:
            actions = self.locals['actions']
            if isinstance(actions, np.ndarray):
                action_list = actions.flatten().tolist()
            else:
                action_list = [int(actions)]
            street = None
            if infos and isinstance(infos[0], dict):
                # learner_street was stashed by OpponentAutoPlayWrapper
                # before opponents acted, so it survives even after the
                # auto-play loop overwrote 'street' for opponent steps.
                street = infos[0].get('learner_street')
            for a in action_list:
                self.episode_actions.append(int(a))
                self.episode_action_streets.append((int(a), street))

        # Track rewards
        if isinstance(rewards, np.ndarray):
            self.current_episode_reward += float(rewards[0]) if len(rewards) > 0 else 0
        else:
            self.current_episode_reward += float(rewards) if rewards is not None else 0

        # Track episode completion. Prefer SB3 Monitor's `info['episode']['r']`
        # (authoritative episode return) when present; fall back to the
        # local accumulator only when there's no Monitor wrapper. Doing both
        # used to double-count every completed episode.
        for i, info in enumerate(infos):
            if isinstance(dones, np.ndarray):
                done = dones[i] if i < len(dones) else False
            else:
                done = dones if i == 0 else False

            if 'episode' in info:
                episode_reward = info['episode'].get('r', 0)
                self.episode_rewards.append(episode_reward)
                self.episode_count += 1
                if episode_reward > 0:
                    self.episode_wins += 1
                self.current_episode_reward = 0
            elif done:
                self.episode_rewards.append(self.current_episode_reward)
                self.episode_count += 1
                if self.current_episode_reward > 0:
                    self.episode_wins += 1
                self.current_episode_reward = 0

        # Log periodically
        if self.num_timesteps - self.last_logged_step >= self.log_freq:
            self._log_metrics()
            self.last_logged_step = self.num_timesteps

        return True

    def _on_training_start(self) -> None:
        """Called at training start"""
        self.episode_rewards = []
        self.current_episode_reward = 0
        self.episode_actions = []
        self.episode_action_streets = []
        self.episode_wins = 0
        self.episode_count = 0
        self.last_logged_step = 0
        self._action_buckets = self._resolve_action_buckets()

    def _resolve_action_buckets(self):
        """Read the env's actual action layout off the SB3 model. Falls
        back to the historical 3-bin + all-in layout if the env can't
        be unwrapped (e.g. when a test passes a Mock model)."""
        default_layout = {
            "fold": [0],
            "call": [1],
            "raise": [2, 3, 4],
            "all_in": [5],
        }
        try:
            env = self.model.get_env() if hasattr(self.model, "get_env") else None
            unwrapped = env
            for attr in ("envs", "env"):
                inner = getattr(unwrapped, attr, None)
                if inner is None:
                    continue
                unwrapped = inner[0] if isinstance(inner, (list, tuple)) else inner
            while hasattr(unwrapped, "env") and not hasattr(unwrapped, "raise_bins"):
                unwrapped = unwrapped.env

            raise_bins = getattr(unwrapped, "raise_bins", None)
            include_all_in = getattr(unwrapped, "include_all_in", True)
            if not isinstance(raise_bins, list):
                return default_layout
            raise_indices = list(range(2, 2 + len(raise_bins)))
            all_in_idx = 2 + len(raise_bins) if include_all_in else None
            return {
                "fold": [0],
                "call": [1],
                "raise": raise_indices,
                "all_in": [all_in_idx] if all_in_idx is not None else [],
            }
        except Exception:
            return default_layout

    def _per_street_action_rates(self, buckets):
        """Break episode_action_streets into per-street fold/call/raise/all_in
        rates plus a count of actions taken on that street.

        Returns {street: {"fold": x, "call": y, "raise": z, "all_in": w,
        "count": n}} for the four canonical streets. Streets with zero
        recorded actions report zeroes."""
        fold_set = set(buckets["fold"])
        call_set = set(buckets["call"])
        raise_set = set(buckets["raise"])
        all_in_set = set(buckets["all_in"])

        per_street = {s: {"fold": 0, "call": 0, "raise": 0, "all_in": 0, "count": 0}
                      for s in self.STREETS}
        for action, street in self.episode_action_streets:
            if street not in per_street:
                # Skip None/"unknown" / showdown streets — they're not in
                # the canonical 4-street layout we report on.
                continue
            entry = per_street[street]
            entry["count"] += 1
            if action in fold_set:
                entry["fold"] += 1
            elif action in call_set:
                entry["call"] += 1
            elif action in raise_set:
                entry["raise"] += 1
            elif action in all_in_set:
                entry["all_in"] += 1

        for s, e in per_street.items():
            n = max(e["count"], 1)
            e["fold"] /= n
            e["call"] /= n
            e["raise"] /= n
            e["all_in"] /= n
        return per_street

    def _on_training_end(self) -> None:
        """Called at training end"""
        # Log final metrics
        self._log_metrics()

    def _log_metrics(self) -> None:
        """Log collected metrics to both custom metrics and TensorBoard"""
        if not self.episode_rewards:
            if self.verbose > 0:
                print(f"[{self.num_timesteps}] No episodes completed yet")
            return

        # Calculate statistics
        avg_reward = np.mean(self.episode_rewards)
        max_reward = np.max(self.episode_rewards)
        min_reward = np.min(self.episode_rewards)
        
        win_rate = self.episode_wins / max(self.episode_count, 1)

        # Action distribution statistics
        fold_rate = 0
        raise_rate = 0
        all_in_rate = 0
        call_rate = 0

        buckets = self._action_buckets or self._resolve_action_buckets()
        per_street = None
        if self.episode_actions:
            total_actions = len(self.episode_actions)
            fold_set = set(buckets["fold"])
            call_set = set(buckets["call"])
            raise_set = set(buckets["raise"])
            all_in_set = set(buckets["all_in"])

            fold_count = sum(1 for a in self.episode_actions if a in fold_set)
            call_count = sum(1 for a in self.episode_actions if a in call_set)
            raise_count = sum(1 for a in self.episode_actions if a in raise_set)
            all_in_count = sum(1 for a in self.episode_actions if a in all_in_set)

            fold_rate = fold_count / total_actions
            call_rate = call_count / total_actions
            raise_rate = raise_count / total_actions
            all_in_rate = all_in_count / total_actions

            per_street = self._per_street_action_rates(buckets)

        # Extract learning metrics from model if available
        policy_loss = 0.0
        value_loss = 0.0
        entropy_loss = 0.0

        if hasattr(self.model, 'logger') and self.model.logger:
            if hasattr(self.model.logger, 'name_to_value'):
                policy_loss = self.model.logger.name_to_value.get('train/policy_loss', 0.0)
                value_loss = self.model.logger.name_to_value.get('train/value_loss', 0.0)
                entropy_loss = self.model.logger.name_to_value.get('train/entropy_loss', 0.0)

        # Prepare agent stats
        agent_stats = {
            'win_rate': float(win_rate),
            'episodes': int(self.episode_count),
            'avg_reward': float(avg_reward),
            'max_reward': float(max_reward),
            'min_reward': float(min_reward),
        }

        # Prepare learning metrics
        learning_metrics = {
            'learning_rate': float(self.model.learning_rate) if hasattr(self.model, 'learning_rate') else 0.0,
            'policy_loss': float(policy_loss),
            'value_loss': float(value_loss),
            'entropy': float(entropy_loss)
        }

        # Log to custom metrics system
        self.metrics.log_step(
            self.num_timesteps,
            self.episode_rewards[-100:] if self.episode_rewards else [],
            agent_stats,
            learning_metrics
        )

        # Log to TensorBoard
        if hasattr(self.model, 'logger') and self.model.logger:
            # Agent performance metrics
            self.model.logger.record("agent/win_rate", win_rate)
            self.model.logger.record("agent/avg_reward", avg_reward)
            self.model.logger.record("agent/max_reward", max_reward)
            self.model.logger.record("agent/min_reward", min_reward)
            
            # Action distribution metrics
            self.model.logger.record("agent/fold_rate", fold_rate)
            self.model.logger.record("agent/call_rate", call_rate)
            self.model.logger.record("agent/raise_rate", raise_rate)
            self.model.logger.record("agent/all_in_rate", all_in_rate)

            # Per-street breakdown: e.g. agent/preflop/raise_rate.
            # Watching how aggression shifts between preflop and river is
            # a much sharper diagnostic than the collapsed totals above.
            if per_street:
                for street, entry in per_street.items():
                    if entry["count"] == 0:
                        continue
                    self.model.logger.record(f"agent/{street}/fold_rate", entry["fold"])
                    self.model.logger.record(f"agent/{street}/call_rate", entry["call"])
                    self.model.logger.record(f"agent/{street}/raise_rate", entry["raise"])
                    self.model.logger.record(f"agent/{street}/all_in_rate", entry["all_in"])
                    self.model.logger.record(f"agent/{street}/action_count", entry["count"])

            # Episode tracking
            self.model.logger.record("agent/episodes_completed", self.episode_count)

            # Dump to TensorBoard file
            self.model.logger.dump(self.num_timesteps)

        # Record actions for action distribution tracking
        if self.episode_actions:
            self.metrics.record_actions(self.episode_actions)
            self.metrics.checkpoint_actions(self.num_timesteps)
            if per_street is not None:
                self.metrics.record_street_breakdown(self.num_timesteps, per_street)

        # Reset tracking for next logging period
        self.episode_actions = []
        self.episode_action_streets = []
        self.episode_wins = 0
        self.episode_count = 0

        if self.verbose > 0:
            print(f"[{self.num_timesteps}] Metrics - Win Rate: {win_rate:.2%}, Fold: {fold_rate:.2%}, "
                  f"Call: {call_rate:.2%}, Raise: {raise_rate:.2%}, Avg Reward: {avg_reward:.2f}")


class SimpleMetricsCallback(BaseCallback):
    """Simpler callback for basic metric tracking"""

    def __init__(self, metrics: TrainingMetrics, log_freq: int = 10000):
        super().__init__()
        self.metrics = metrics
        self.log_freq = log_freq
        self.steps_since_log = 0
        self.episode_rewards = []
        self.current_episode_reward = 0
        self.episode_actions = []
        self.episode_wins = 0
        self.episode_count = 0
        self.last_logged_step = 0

    def set_model(self, model) -> None:
        """Bind the SB3 model. BaseCallback exposes `self.model` as a simple
        attribute (no setter), so plain assignment is the correct hook."""
        self.model = model

    def _on_step(self) -> bool:
        """Called at each step"""
        self.steps_since_log += 1

        # Get info from the last step
        infos = self.locals.get('infos', [])
        dones = self.locals.get('dones', [])
        rewards = self.locals.get('rewards', [])

        # Track actions taken
        if 'actions' in self.locals:
            actions = self.locals['actions']
            if isinstance(actions, np.ndarray):
                self.episode_actions.extend(actions.flatten().tolist())
            else:
                self.episode_actions.append(int(actions))

        # Track rewards
        if isinstance(rewards, np.ndarray):
            self.current_episode_reward += float(rewards[0]) if len(rewards) > 0 else 0
        else:
            self.current_episode_reward += float(rewards) if rewards is not None else 0

        # Track episode completion
        for i, info in enumerate(infos):
            if isinstance(dones, np.ndarray):
                done = dones[i] if i < len(dones) else False
            else:
                done = dones if i == 0 else False

            if done:
                self.episode_rewards.append(self.current_episode_reward)
                self.episode_count += 1

                if self.current_episode_reward > 0:
                    self.episode_wins += 1

                self.current_episode_reward = 0

        # Log periodically
        if self.steps_since_log >= self.log_freq:
            self._log_metrics()
            self.steps_since_log = 0

        return True

    def _on_training_start(self) -> None:
        """Called at training start"""
        self.episode_rewards = []
        self.current_episode_reward = 0
        self.episode_actions = []
        self.episode_wins = 0
        self.episode_count = 0

    def _log_metrics(self) -> None:
        """Log collected metrics"""
        if not self.episode_rewards:
            return

        avg_reward = np.mean(self.episode_rewards)
        win_rate = self.episode_wins / max(self.episode_count, 1)

        # Record actions
        if self.episode_actions:
            self.metrics.record_actions(self.episode_actions)

        # Reset tracking
        self.episode_actions = []
        self.episode_wins = 0
        self.episode_count = 0

class CriticCalibrationCallback(BaseCallback):
    """Capture (V(s_0), actual discounted return G_0) pairs per episode.

    SB3 PPO already computes V(s) during action selection and exposes it
    via `self.locals['values']`. We snapshot V at the first step of each
    episode, then accumulate the discounted return until done. On done,
    we append the (V_0, G_0) pair to an in-memory buffer and flush to
    disk every `flush_freq` steps.

    Why s_0 only? It's the cheapest informative slice: if V(s_0) is
    consistently far from G_0, the critic is mis-estimating expected
    return, which is the textbook PPO value-loss diagnostic. Sampling at
    every step is possible but quadratic, and not needed for a first
    calibration view.
    """

    def __init__(
        self,
        save_dir: str,
        gamma: Optional[float] = None,
        flush_freq: int = 10_000,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.save_dir = save_dir
        self.flush_freq = flush_freq
        self._gamma = gamma  # if None, read off model at training start
        self._pairs = []  # list of dicts: {timestep, value, actual_return}
        self._current_ep_value = None
        self._current_ep_return = 0.0
        self._current_ep_step_in_ep = 0
        self._last_flush_step = 0

    def _on_training_start(self) -> None:
        if self._gamma is None:
            self._gamma = float(getattr(self.model, "gamma", 0.99))
        os.makedirs(self.save_dir, exist_ok=True)
        self._pairs = []
        self._current_ep_value = None
        self._current_ep_return = 0.0
        self._current_ep_step_in_ep = 0
        self._last_flush_step = 0

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards")
        dones = self.locals.get("dones")
        values = self.locals.get("values")

        if rewards is None or dones is None:
            return True

        reward = float(rewards[0]) if hasattr(rewards, "__len__") and len(rewards) > 0 \
            else float(rewards or 0.0)
        done = bool(dones[0]) if hasattr(dones, "__len__") and len(dones) > 0 \
            else bool(dones)

        # Capture V(s_0) the first time we see a step of a new episode.
        if self._current_ep_value is None and values is not None:
            try:
                v0 = values[0]
                # SB3 hands us a torch tensor; cast defensively.
                if hasattr(v0, "item"):
                    v0 = v0.item()
                self._current_ep_value = float(v0)
            except (IndexError, TypeError):
                self._current_ep_value = None

        # Accumulate discounted return G_0 = sum_t gamma^t r_t.
        self._current_ep_return += (self._gamma ** self._current_ep_step_in_ep) * reward
        self._current_ep_step_in_ep += 1

        if done:
            if self._current_ep_value is not None:
                self._pairs.append({
                    "timestep": int(self.num_timesteps),
                    "value": self._current_ep_value,
                    "actual_return": self._current_ep_return,
                })
            self._current_ep_value = None
            self._current_ep_return = 0.0
            self._current_ep_step_in_ep = 0

        if self.num_timesteps - self._last_flush_step >= self.flush_freq:
            self._flush()
            self._last_flush_step = self.num_timesteps
        return True

    def _on_training_end(self) -> None:
        self._flush()

    def _flush(self) -> None:
        path = os.path.join(self.save_dir, "value_calibration.json")
        with open(path, "w") as f:
            json.dump({"pairs": self._pairs, "gamma": self._gamma}, f, indent=2)


class OpponentProfitCallback(BaseCallback):
    """Callback to periodically checkpoint opponent profit data"""

    def __init__(self, profit_tracker: OpponentProfitTracker, checkpoint_freq: int = 10000):
        super().__init__()
        self.profit_tracker = profit_tracker
        self.checkpoint_freq = checkpoint_freq
        self.last_checkpoint = 0

    def _on_step(self) -> bool:
        """Called at each step"""
        if self.num_timesteps - self.last_checkpoint >= self.checkpoint_freq:
            self.profit_tracker.checkpoint(self.num_timesteps)
            self.last_checkpoint = self.num_timesteps

            # Print summary periodically
            if self.verbose > 0:
                self.profit_tracker.print_summary()

        return True

    def _on_training_end(self) -> None:
        """Save final checkpoint at end of training"""
        self.profit_tracker.checkpoint(self.num_timesteps)
        print("\nFinal Opponent Profit Summary:")
        self.profit_tracker.print_summary()
