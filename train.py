"""
Training script for PPO poker agent with opponent evolution.

Trains main PPO agent against:
- The last trained model (if available)
- The second-to-last trained model (if available)  
- Or CallAgent and RandomAgent as fallback

No MultiAgentWrapper class - opponents are handled inline in OpponentAutoPlayWrapper.
"""

import argparse
import yaml
import os
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional

from src.poker_env.texas_holdem_env import TexasHoldemEnv
from src.agents.ppo_agent import PPOAgent, TrainingCallback
from src.agents.random_agent import CallAgent, RandomAgent
from src.agents.opponent_ppo import OpponentPPO
from src.training.metrics import TrainingMetrics
from src.training.callbacks import MetricsCallback


# ============================================================================
# OPPONENT AUTO-PLAY WRAPPER - Handles opponent loop inline
# ============================================================================

class OpponentAutoPlayWrapper:
    """
    Wraps TexasHoldemEnv to automatically play opponent moves.
    
    This wrapper:
    1. Takes learning agent's action
    2. Automatically plays all opponents until main agent's turn
    3. Returns observation to learning agent
    
    Compatible with SB3's API.
    """
    
    def __init__(self, env: TexasHoldemEnv, opponents_list: List[Tuple[str, object]]):
        """
        Args:
            env: TexasHoldemEnv instance
            opponents_list: List of (opponent_type, agent) tuples
                           opponent_type: str (e.g., 'call', 'random', 'ppo_gen_1')
                           agent: Agent instance with select_action(obs) -> int method
        """
        self.env = env
        self.opponents = opponents_list
        
        # Expose environment properties for SB3
        self.observation_space = env.observation_space
        self.action_space = env.action_space
    
    def reset(self, **kwargs):
        """Reset environment and return initial observation"""
        obs, info = self.env.reset(**kwargs)
        return obs, info
    
    def step(self, action: int) -> Tuple:
        """
        Execute learning agent action and auto-play opponents.
        
        CRITICAL FLOW:
        1. Learning agent (player 0) takes action
        2. Environment executes action, records in tracker
        3. Loop plays all opponents (players 1, 2) until:
           - Main agent's turn comes up again (current_player_idx == 0)
           - Hand ends (terminated or truncated)
        4. Return observation with updated opponent stats
        
        This is where opponents interact with the environment!
        """
        
        # Step 1: Learning agent acts
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Step 2: Automatically play all opponents until main agent's turn or hand ends
        while not (terminated or truncated) and self.env.game_state.current_player_idx != 0:

            # Get current player index (should be 1 or 2)
            current_idx = self.env.game_state.current_player_idx
            opponent_idx = current_idx - 1  # opponent index in self.opponents list
            
            # Safety check
            if opponent_idx < len(self.opponents):
                opponent_type, opponent = self.opponents[opponent_idx]
                
                # ← OPPONENTS GET 68-DIM OBSERVATION WITH OPPONENT STATS
                opponent_action = opponent.select_action(obs)
                
                # ← OPPONENT AFFECTS ENVIRONMENT (chips move, stats recorded, etc.)
                obs, _, terminated, truncated, info = self.env.step(opponent_action)
            else:
                break
        
        return obs, reward, terminated, truncated, info
    
    def render(self, *args, **kwargs):
        """Render the environment"""
        return self.env.render(*args, **kwargs)
    
    def close(self):
        """Close the environment"""
        return self.env.close()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_config(config_path: str) -> dict:
    """Load YAML configuration file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def find_latest_models(models_dir: str = "models", count: int = 2) -> List[Optional[str]]:
    """
    Find the most recent trained models by modification time.
    
    Args:
        models_dir: Directory containing trained models
        count: Number of models to find (typically 2)
    
    Returns:
        List of model paths [latest, second_latest, ...], None for missing slots
        
    Example:
        ['/path/to/gen_2/final_model.zip', '/path/to/gen_1/final_model.zip']
    """
    models_path = Path(models_dir)
    
    if not models_path.exists():
        return [None] * count
    
    # Find all final_model.zip files with their modification times
    model_files = []
    for model_zip in models_path.glob("*/final_model.zip"):
        model_files.append({
            'path': str(model_zip),
            'mtime': model_zip.stat().st_mtime,
            'dir': model_zip.parent.name
        })
    
    if not model_files:
        return [None] * count
    
    # Sort by modification time (newest first)
    model_files.sort(key=lambda x: x['mtime'], reverse=True)
    
    # Build result list
    result = []
    for i in range(count):
        if i < len(model_files):
            result.append(model_files[i]['path'])
        else:
            result.append(None)
    
    return result


def create_opponents(models_dir: str = "models") -> List[Tuple[str, object]]:
    """
    Create opponent list for training.
    
    Strategy:
    - If 2+ models found: Use last 2 models
    - If 1 model found: Use it + CallAgent
    - If 0 models found: Use CallAgent + RandomAgent
    
    Args:
        models_dir: Directory containing trained models
    
    Returns:
        List of (opponent_type_str, agent_object) tuples
    """
    opponents = []
    latest_models = find_latest_models(models_dir, count=2)
    
    print("\n" + "="*70)
    print("OPPONENT SELECTION")
    print("="*70)
    
    models_loaded = 0
    
    # Try to load previous generation models
    for i, model_path in enumerate(latest_models):
        if model_path is not None:
            try:
                opponent = OpponentPPO(model_path)
                if opponent.is_loaded():
                    model_name = Path(model_path).parent.name
                    print(f"✓ Loaded opponent {i+1}: {model_name}")
                    opponents.append((f"ppo_{model_name}", opponent))
                    models_loaded += 1
                else:
                    print(f"✗ Failed to load opponent from {model_path}")
            except Exception as e:
                print(f"✗ Error loading opponent from {model_path}: {e}")
    
    # Fill remaining slots with CallAgent and RandomAgent
    if models_loaded == 0:
        print("\nNo trained models found. Using default opponents:")
        call_agent = CallAgent(name="CallAgent")
        random_agent = RandomAgent(name="RandomAgent")
        
        opponents = [
            ('call', call_agent),
            ('random', random_agent)
        ]
        
        print(f"✓ Using CallAgent")
        print(f"✓ Using RandomAgent")
    
    elif models_loaded == 1:
        print("\nOnly one model found. Adding CallAgent:")
        call_agent = CallAgent(name="CallAgent")
        opponents.append(('call', call_agent))
        print(f"✓ Added CallAgent")
    
    print("="*70)
    
    return opponents


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train(config_path: str, run_name: str = None):
    """Train PPO agent against evolving opponents"""
    
    config = load_config(config_path)
    
    if run_name is None:
        run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print("="*70)
    print(f"Training PPO Poker Agent: {run_name}")
    print("="*70)
    print(f"Configuration: {config_path}")
    print()
    
    # =========================================================================
    # ENVIRONMENT SETUP
    # =========================================================================
    
    env_config = config['environment']
    env = TexasHoldemEnv(
        num_players=3,
        starting_stack=env_config['starting_stack'],
        small_blind=env_config['small_blind'],
        big_blind=env_config['big_blind'],
        rake_percent=env_config['rake_percent'] if env_config['rake_enabled'] else 0.0,
        rake_cap=env_config.get('rake_cap', 0),
        min_raise_multiplier=env_config.get('min_raise_multiplier', 1.0),
        reset_stacks_every_n_timesteps=env_config.get('reset_stacks_every_n_timesteps'),
        track_opponents=True  # ← CRITICAL: Enable opponent tracking!
    )
    
    print(f"Environment: 3 players")
    print(f"  Starting stack: ${env_config['starting_stack']}")
    print(f"  Blinds: ${env_config['small_blind']}/${env_config['big_blind']}")
    print(f"  Opponent tracking: Enabled")
    print(f"  Observation space: {env.observation_space.shape}")  # Should be (68,)
    print()
    
    # =========================================================================
    # OPPONENT SETUP
    # =========================================================================
    
    opponents = create_opponents(models_dir="models")
    
    print(f"\nOpponents:")
    for i, (opponent_type, opponent) in enumerate(opponents, 1):
        print(f"  Player {i}: {opponent_type} - {opponent.name}")
    print()
    
    # =========================================================================
    # METRICS & LOGGING
    # =========================================================================
    
    metrics = TrainingMetrics(run_name, save_dir="metrics")
    training_config = config['training']
    log_dir = os.path.join(config['logging']['log_dir'], run_name)
    model_dir = os.path.join(config['logging']['model_dir'], run_name)
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"Metrics: metrics/{run_name}/")
    print(f"Logs: {log_dir}")
    print(f"Models: {model_dir}")
    print()
    
    # =========================================================================
    # AGENT CREATION
    # =========================================================================

    # Get policy_kwargs if specified in config
    policy_kwargs = training_config.get('policy_kwargs', None)

    agent = PPOAgent(
        env=env,
        name=f"PPO_{run_name}",
        learning_rate=training_config['learning_rate'],
        n_steps=training_config['n_steps'],
        batch_size=training_config['batch_size'],
        n_epochs=training_config['n_epochs'],
        gamma=training_config['gamma'],
        gae_lambda=training_config['gae_lambda'],
        clip_range=training_config['clip_range'],
        ent_coef=training_config.get('ent_coef', 0.01),  # Entropy bonus for exploration
        vf_coef=training_config.get('vf_coef', 0.5),     # Value function coefficient
        max_grad_norm=training_config.get('max_grad_norm', 0.5),  # Gradient clipping
        tensorboard_log=log_dir,
        policy_kwargs=policy_kwargs  # Pass custom architecture if specified
    )
    
    print(f"Agent: {agent.name}")
    print(f"Learning rate: {training_config['learning_rate']}")
    print(f"Total timesteps: {training_config['total_timesteps']:,}")
    if policy_kwargs:
        print(f"Architecture: {policy_kwargs}")
    print()
    
    # =========================================================================
    # WRAP ENVIRONMENT WITH OPPONENT AUTO-PLAY
    # =========================================================================
    
    wrapped_env = OpponentAutoPlayWrapper(env, opponents)
    
    # =========================================================================
    # CREATE CALLBACKS
    # =========================================================================
    
    save_freq = config['logging']['save_frequency']
    save_callback = TrainingCallback(
        save_freq=save_freq,
        save_path=model_dir
    )
    
    metrics_callback = MetricsCallback(
        metrics=metrics,
        log_freq=10000
    )
    
    # =========================================================================
    # TRAIN
    # =========================================================================
    
    print("Starting training...")
    print(f"Tensorboard: tensorboard --logdir {log_dir}")
    print()
    
    agent.model.learn(
        total_timesteps=training_config['total_timesteps'],
        callback=[save_callback, metrics_callback]
    )
    
    # =========================================================================
    # SAVE FINAL MODEL
    # =========================================================================
    
    final_model_path = os.path.join(model_dir, "final_model")
    agent.save(final_model_path)
    
    print()
    print("="*70)
    print("Training Complete!")
    print("="*70)
    print(f"Model saved to: {final_model_path}")
    print(f"Tensorboard: tensorboard --logdir {log_dir}")
    print(f"Metrics: metrics/{run_name}/")
    print()


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train PPO poker bot against evolving opponents"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Custom run name"
    )
    
    args = parser.parse_args()
    train(args.config, args.name)