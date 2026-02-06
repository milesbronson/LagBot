#!/usr/bin/env python3
"""
Resume training from a checkpoint
"""

import os
import yaml
from src.agents.ppo_agent import PPOAgent, TrainingCallback
from src.poker_env.texas_holdem_env import TexasHoldemEnv
from src.training.callbacks import MetricsCallback
from src.training.metrics import TrainingMetrics
from train import OpponentAutoPlayWrapper, create_opponents

def resume_training(
    checkpoint_path: str,
    remaining_steps: int,
    run_name: str,
    config_path: str = "configs/default_config.yaml"
):
    """Resume training from a checkpoint"""

    print("=" * 70)
    print(f"RESUMING TRAINING: {run_name}")
    print("=" * 70)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Remaining steps: {remaining_steps:,}")
    print()

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Setup environment
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
        track_opponents=True
    )

    print("Environment configured:")
    print(f"  Players: 3")
    print(f"  Starting stack: ${env_config['starting_stack']}")
    print(f"  Blinds: ${env_config['small_blind']}/${env_config['big_blind']}")
    print(f"  Observation space: {env.observation_space.shape}")
    print()

    # Create opponents
    opponents = create_opponents(models_dir="models")
    print(f"\nOpponents configured:")
    for i, (opponent_type, opponent) in enumerate(opponents, 1):
        print(f"  Player {i}: {opponent_type} - {opponent.name}")
    print()

    # Create agent from checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    agent = PPOAgent(env, name=f"PPO_{run_name}_resumed", device="auto")
    agent.load(checkpoint_path)
    print(f"✓ Model loaded successfully")
    print()

    # Wrap environment with opponents
    wrapped_env = OpponentAutoPlayWrapper(env, opponents)
    print(f"✓ Environment wrapped with opponents")
    print()

    # Setup logging
    log_dir = os.path.join(config['logging']['log_dir'], run_name)
    model_dir = os.path.join(config['logging']['model_dir'], run_name)

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    print(f"Logs: {log_dir}")
    print(f"Models: {model_dir}")
    print()

    # Setup callbacks
    save_freq = config['logging']['save_frequency']
    save_callback = TrainingCallback(
        save_freq=save_freq,
        save_path=model_dir
    )

    metrics = TrainingMetrics(run_name, save_dir="metrics")
    metrics_callback = MetricsCallback(
        metrics=metrics,
        log_freq=10000
    )

    # Resume training
    print("=" * 70)
    print("RESUMING TRAINING...")
    print("=" * 70)
    print(f"Remaining timesteps: {remaining_steps:,}")
    print(f"Tensorboard: tensorboard --logdir {log_dir}")
    print()

    # Set the wrapped environment on the model before training
    agent.model.set_env(wrapped_env)

    agent.model.learn(
        total_timesteps=remaining_steps,
        callback=[save_callback, metrics_callback],
        reset_num_timesteps=False,  # Don't reset the timestep counter
        tb_log_name="PPO_resumed"
    )

    # Save final model
    final_model_path = os.path.join(model_dir, "final_model")
    agent.save(final_model_path)

    print()
    print("=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"Final model saved to: {final_model_path}")
    print(f"Total timesteps: 1,000,000")
    print(f"Tensorboard: tensorboard --logdir {log_dir}")
    print(f"Metrics: metrics/{run_name}/")
    print()

if __name__ == "__main__":
    resume_training(
        checkpoint_path="./models/gpu_fixed_v1/model_500000_steps.zip",
        remaining_steps=500000,
        run_name="gpu_fixed_v1",
        config_path="configs/default_config.yaml"
    )
