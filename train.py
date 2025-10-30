"""
Training script with metrics dashboard support
"""

import argparse
import yaml
import os
from datetime import datetime
import numpy as np

from src.poker_env.texas_holdem_env import TexasHoldemEnv
from src.agents.ppo_agent import PPOAgent, TrainingCallback
from src.training.metrics import TrainingMetrics
from src.training.callbacks import SimpleMetricsCallback


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train(config_path: str, run_name: str = None):
    """Train a PPO agent with metrics collection"""
    
    config = load_config(config_path)
    
    if run_name is None:
        run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print("="*60)
    print(f"Training PPO Poker Agent: {run_name}")
    print("="*60)
    print(f"Configuration: {config_path}")
    print()
    
    # Create environment
    env_config = config['environment']
    env = TexasHoldemEnv(
        num_players=env_config['num_players'],
        starting_stack=env_config['starting_stack'],
        small_blind=env_config['small_blind'],
        big_blind=env_config['big_blind'],
        rake_percent=env_config['rake_percent'] if env_config['rake_enabled'] else 0.0,
        rake_cap=env_config.get('rake_cap', 0),
        min_raise_multiplier=env_config.get('min_raise_multiplier', 1.0)
    )
    
    print(f"Environment: {env_config['num_players']} players, "
          f"Starting stack: ${env_config['starting_stack']}, "
          f"Blinds: ${env_config['small_blind']}/${env_config['big_blind']}")
    print()
    
    # Create metrics
    metrics = TrainingMetrics(run_name, save_dir="metrics")
    print(f"Metrics will be saved to: metrics/{run_name}/")
    print()
    
    # Create agent
    training_config = config['training']
    
    log_dir = os.path.join(config['logging']['log_dir'], run_name)
    model_dir = os.path.join(config['logging']['model_dir'], run_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
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
        tensorboard_log=log_dir
    )
    
    print(f"Agent: {agent.name}")
    print(f"Learning rate: {training_config['learning_rate']}")
    print(f"Total timesteps: {training_config['total_timesteps']:,}")
    print()
    
    # Create callbacks
    save_freq = config['logging']['save_frequency']
    save_callback = TrainingCallback(
        save_freq=save_freq,
        save_path=model_dir
    )
    
    metrics_callback = SimpleMetricsCallback(
        metrics=metrics,
        log_freq=10000
    )
    
    # Train
    print("Starting training...")
    print(f"Tensorboard logs: {log_dir}")
    print(f"Models will be saved to: {model_dir}")
    print(f"Metrics will be saved to: metrics/{run_name}/")
    print()
    print("To monitor training:")
    print(f"  1. Tensorboard: tensorboard --logdir {log_dir}")
    print(f"  2. Dashboard: python -c \"from src.training.dashboard import TrainingDashboard; d = TrainingDashboard(); d.plot_single_run('{run_name}', 'dashboard.png')\"")
    print(f"  3. Report: python dashboard_gen.py")
    print()
    
    agent.train(
        total_timesteps=training_config['total_timesteps'],
        callback=[save_callback, metrics_callback]
    )
    
    # Save final model
    final_model_path = os.path.join(model_dir, "final_model")
    agent.save(final_model_path)
    
    # Generate summary
    summary = metrics.get_summary()
    
    print()
    print("="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Final model: {final_model_path}")
    print(f"Metrics: metrics/{run_name}/metrics.json")
    print()
    print("Summary:")
    print(f"  Total Timesteps: {summary.get('total_timesteps', 0):,}")
    print(f"  Final Reward: {summary.get('current_reward', 0):.2f}")
    print(f"  Avg Reward (100): {summary.get('avg_reward_100', 0):.2f}")
    print(f"  Win Rate: {summary.get('win_rate', 0):.1%}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Train a PPO poker agent")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--name',
        type=str,
        default=None,
        help='Name for this training run'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        return
    
    train(args.config, args.name)


if __name__ == "__main__":
    main()