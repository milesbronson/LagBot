"""
Training script for PPO poker agents
"""

import argparse
import yaml
import os
from datetime import datetime

from src.poker_env.texas_holdem_env import TexasHoldemEnv
from src.agents.ppo_agent import PPOAgent, TrainingCallback


def load_config(config_path: str) -> dict:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train(config_path: str, run_name: str = None):
    """
    Train a PPO agent
    
    Args:
        config_path: Path to configuration file
        run_name: Optional name for this training run
    """
    # Load configuration
    config = load_config(config_path)
    
    # Create run name with timestamp
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
    print(f"Min raise multiplier: {env_config.get('min_raise_multiplier', 1.0)}x")
    print()
    
    # Create agent
    training_config = config['training']
    ppo_config = config['ppo']
    
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
    
    # Create callback
    save_freq = config['logging']['save_frequency']
    callback = TrainingCallback(
        save_freq=save_freq,
        save_path=model_dir
    )
    
    # Train
    print("Starting training...")
    print(f"Tensorboard logs: {log_dir}")
    print(f"Models will be saved to: {model_dir}")
    print()
    print("To monitor training, run:")
    print(f"  tensorboard --logdir {log_dir}")
    print()
    
    agent.train(
        total_timesteps=training_config['total_timesteps'],
        callback=callback
    )
    
    # Save final model
    final_model_path = os.path.join(model_dir, "final_model")
    agent.save(final_model_path)
    
    print()
    print("="*60)
    print("Training complete!")
    print(f"Final model saved to: {final_model_path}")
    print("="*60)


def main():
    """Main entry point"""
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
    
    # Check if config exists
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        return
    
    train(args.config, args.name)


if __name__ == "__main__":
    main()