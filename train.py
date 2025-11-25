"""
Training script for PPO poker agent with fixed opponents
3-player environment:
- Player 0: Main PPO Agent (learning)
- Player 1: CallAgent (fixed)
- Player 2: RandomAgent (fixed)
"""

import argparse
import yaml
import os
from datetime import datetime
import numpy as np
from typing import List, Tuple

from src.poker_env.texas_holdem_env import TexasHoldemEnv
from src.agents.ppo_agent import PPOAgent, TrainingCallback
from src.agents.random_agent import CallAgent, RandomAgent
from src.training.metrics import TrainingMetrics
from src.training.callbacks import MetricsCallback


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


class MultiAgentWrapper:
    """
    Wraps environment to handle non-learning agents automatically.
    Only the main agent (player 0) learns.
    """
    
    def __init__(self, env: TexasHoldemEnv, 
                 main_agent: PPOAgent,
                 opponents: List[Tuple[str, object]]):
        """
        Args:
            env: TexasHoldemEnv with 3 players
            main_agent: PPO agent to train
            opponents: [(type, agent), ...] for players 1-2
        """
        self.env = env
        self.main_agent = main_agent
        self.opponents = opponents
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        
        print(f"Training setup:")
        print(f"  Player 0: Main PPO Agent")
        for i, (atype, agent) in enumerate(opponents, 1):
            print(f"  Player {i}: {atype.upper()} - {agent.name}")
    
    def reset(self):
        """Reset and return observation for main agent"""
        obs, info = self.env.reset()
        return obs, info
    
    def step(self, action: int):
        """
        Execute main agent action, then handle opponent turns
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Auto-play opponents until main agent's turn or hand ends
        while not (terminated or truncated) and self.env.game_state.get_current_player_index() != 0:
            current_idx = self.env.game_state.get_current_player_index()
            opponent_idx = current_idx - 1
            
            if opponent_idx < len(self.opponents):
                atype, opponent = self.opponents[opponent_idx]
                action = opponent.select_action(obs)
                obs, _, terminated, truncated, info = self.env.step(action)
            else:
                break
        
        return obs, reward, terminated, truncated, info
    
    def render(self):
        self.env.render()
    
    def close(self):
        self.env.close()


def train(config_path: str, run_name: str = None):
    """Train PPO agent against CallAgent and RandomAgent"""
    
    config = load_config(config_path)
    
    if run_name is None:
        run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print("="*60)
    print(f"Training PPO Poker Agent: {run_name}")
    print("="*60)
    print(f"Configuration: {config_path}")
    print()
    
    # Create 3-player environment
    env_config = config['environment']
    env = TexasHoldemEnv(
        num_players=3,
        starting_stack=env_config['starting_stack'],
        small_blind=env_config['small_blind'],
        big_blind=env_config['big_blind'],
        rake_percent=env_config['rake_percent'] if env_config['rake_enabled'] else 0.0,
        rake_cap=env_config.get('rake_cap', 0),
        min_raise_multiplier=env_config.get('min_raise_multiplier', 1.0),
        reset_stacks_every_n_timesteps=env_config.get('reset_stacks_every_n_timesteps')
    )
    
    print(f"Environment: 3 players")
    print(f"  Starting stack: ${env_config['starting_stack']}")
    print(f"  Blinds: ${env_config['small_blind']}/${env_config['big_blind']}")
    print()
    
    # Create metrics
    metrics = TrainingMetrics(run_name, save_dir="metrics")
    print(f"Metrics: metrics/{run_name}/")
    print()
    
    # Create directories
    training_config = config['training']
    log_dir = os.path.join(config['logging']['log_dir'], run_name)
    model_dir = os.path.join(config['logging']['model_dir'], run_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # Create main agent
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
    
    # Create opponents
    call_agent = CallAgent(name="CallAgent")
    random_agent = RandomAgent(name="RandomAgent")
    opponents = [
        ('call', call_agent),
        ('random', random_agent)
    ]
    
    # Wrap environment
    wrapped_env = MultiAgentWrapper(env, agent, opponents)
    
    # Create callbacks
    save_freq = config['logging']['save_frequency']
    save_callback = TrainingCallback(
        save_freq=save_freq,
        save_path=model_dir
    )
    
    metrics_callback = MetricsCallback(
        metrics=metrics,
        log_freq=10000
    )
    
    # Train
    print("Starting training...")
    print(f"Tensorboard: tensorboard --logdir {log_dir}")
    print()
    
    agent.train(
        total_timesteps=training_config['total_timesteps'],
        callback=[save_callback, metrics_callback]
    )
    
    # Save final model
    final_model_path = os.path.join(model_dir, "final_model")
    agent.save(final_model_path)
    
    print()
    print("="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Model saved to: {final_model_path}")
    print(f"Tensorboard: tensorboard --logdir {log_dir}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO poker bot")
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