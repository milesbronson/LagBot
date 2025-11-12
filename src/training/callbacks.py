"""
Enhanced training callbacks for metrics collection
"""

from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from src.training.metrics import TrainingMetrics


class MetricsCallback(BaseCallback):
    """Callback that logs training metrics to dashboard"""

    def __init__(self, metrics: TrainingMetrics, log_freq: int = 10000, verbose: int = 0):
        super().__init__(verbose)
        self.metrics = metrics
        self.log_freq = log_freq
        self.episode_rewards = []
        self.episode_actions = []  # Track actions taken
        self.episode_wins = 0
        self.episode_count = 0
        self.last_logged_step = 0

    def _on_step(self) -> bool:
        """Called at each step"""
        # Get info from the last step
        infos = self.locals.get('infos', [])

        # Track actions taken
        if 'actions' in self.locals:
            actions = self.locals['actions']
            if isinstance(actions, np.ndarray):
                self.episode_actions.extend(actions.flatten().tolist())
            else:
                self.episode_actions.append(int(actions))

        # Track episode completion
        for info in infos:
            if 'episode' in info:
                # Episode finished, record reward
                episode_reward = info['episode']['r']
                self.episode_rewards.append(episode_reward)
                self.episode_count += 1

                # Check if agent won (positive reward typically indicates win)
                if episode_reward > 0:
                    self.episode_wins += 1

        # Log periodically
        if self.num_timesteps - self.last_logged_step >= self.log_freq:
            self._log_metrics()
            self.last_logged_step = self.num_timesteps

        return True
    
    def _on_training_start(self) -> None:
        """Called at training start"""
        pass
    
    def _on_training_end(self) -> None:
        """Called at training end"""
        self._log_metrics()
    
    def _log_metrics(self):
        """Log current metrics"""
        # Calculate actual win rate from tracked episodes
        win_rate = self.episode_wins / self.episode_count if self.episode_count > 0 else 0.0

        # Calculate action distribution from tracked actions
        # Actions: 0=Fold, 1=Check/Call, 2-4=Raise, 5=All-in (typical poker env)
        total_actions = len(self.episode_actions)
        if total_actions > 0:
            action_counts = np.bincount(self.episode_actions, minlength=6)
            fold_rate = action_counts[0] / total_actions
            call_rate = action_counts[1] / total_actions
            # Raise rate: sum of raise actions (typically actions 2, 3, 4)
            raise_rate = (action_counts[2] + action_counts[3] + action_counts[4]) / total_actions if len(action_counts) > 4 else 0.0
            all_in_rate = action_counts[5] / total_actions if len(action_counts) > 5 else 0.0
        else:
            fold_rate = call_rate = raise_rate = all_in_rate = 0.0

        agent_stats = {
            'win_rate': float(win_rate),
            'fold_rate': float(fold_rate),
            'raise_rate': float(raise_rate),
            'all_in_rate': float(all_in_rate)
        }

        # Extract REAL training losses from the PPO model's logger
        policy_loss = 0.0
        value_loss = 0.0
        entropy_loss = 0.0

        if hasattr(self.model, 'logger') and self.model.logger is not None:
            # Access the logger's name_to_value dictionary
            logger_dict = self.model.logger.name_to_value
            policy_loss = logger_dict.get('train/policy_loss', 0.0)
            value_loss = logger_dict.get('train/value_loss', 0.0)
            entropy_loss = logger_dict.get('train/entropy_loss', 0.0)

        learning_metrics = {
            'learning_rate': float(self.model.learning_rate) if hasattr(self.model, 'learning_rate') else 0.0,
            'policy_loss': float(policy_loss),
            'value_loss': float(value_loss),
            'entropy': float(entropy_loss)
        }

        # Log the metrics
        self.metrics.log_step(
            self.num_timesteps,
            self.episode_rewards[-100:] if self.episode_rewards else [],
            agent_stats,
            learning_metrics
        )

        # Record actions for action distribution tracking
        if self.episode_actions:
            self.metrics.record_actions(self.episode_actions)
            self.metrics.checkpoint_actions(self.num_timesteps)

        # Reset tracking for next logging period
        self.episode_actions = []
        self.episode_wins = 0
        self.episode_count = 0

        if self.verbose > 0:
            print(f"[{self.num_timesteps}] Metrics - Win Rate: {win_rate:.2%}, Fold: {fold_rate:.2%}, "
                  f"Raise: {raise_rate:.2%}, Policy Loss: {policy_loss:.4f}")


class SimpleMetricsCallback(BaseCallback):
    """Simpler callback for basic metric tracking"""

    def __init__(self, metrics: TrainingMetrics, log_freq: int = 10000):
        super().__init__()
        self.metrics = metrics
        self.log_freq = log_freq
        self.steps_since_log = 0
        self.episode_rewards = []
        self.episode_actions = []
        self.episode_wins = 0
        self.episode_count = 0

    def _on_step(self) -> bool:
        self.steps_since_log += 1

        # Get info from the last step
        infos = self.locals.get('infos', [])

        # Track actions taken
        if 'actions' in self.locals:
            actions = self.locals['actions']
            if isinstance(actions, np.ndarray):
                self.episode_actions.extend(actions.flatten().tolist())
            else:
                self.episode_actions.append(int(actions))

        # Track episode completion
        for info in infos:
            if 'episode' in info:
                episode_reward = info['episode']['r']
                self.episode_rewards.append(episode_reward)
                self.episode_count += 1
                if episode_reward > 0:
                    self.episode_wins += 1

        # Log periodically
        if self.steps_since_log >= self.log_freq:
            # Calculate actual statistics
            win_rate = self.episode_wins / self.episode_count if self.episode_count > 0 else 0.0

            total_actions = len(self.episode_actions)
            if total_actions > 0:
                action_counts = np.bincount(self.episode_actions, minlength=6)
                fold_rate = action_counts[0] / total_actions
                raise_rate = (action_counts[2] + action_counts[3] + action_counts[4]) / total_actions if len(action_counts) > 4 else 0.0
                all_in_rate = action_counts[5] / total_actions if len(action_counts) > 5 else 0.0
            else:
                fold_rate = raise_rate = all_in_rate = 0.0

            agent_stats = {
                'win_rate': float(win_rate),
                'fold_rate': float(fold_rate),
                'raise_rate': float(raise_rate),
                'all_in_rate': float(all_in_rate)
            }

            self.metrics.log_step(
                self.num_timesteps,
                self.episode_rewards[-100:] if self.episode_rewards else [],
                agent_stats
            )

            # Record actions for distribution tracking
            if self.episode_actions:
                self.metrics.record_actions(self.episode_actions)
                self.metrics.checkpoint_actions(self.num_timesteps)

            # Reset counters
            self.episode_actions = []
            self.episode_wins = 0
            self.episode_count = 0
            self.steps_since_log = 0

        return True