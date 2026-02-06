"""
Continue training from checkpoint with self-play opponents
"""

import argparse
import yaml
from src.poker_env.texas_holdem_env import TexasHoldemEnv
from src.agents.ppo_agent import PPOAgent
from src.agents.opponent_ppo import OpponentPPO
from src.training.metrics import TrainingMetrics
from src.training.callbacks import MetricsCallback
from train import OpponentAutoPlayWrapper, TrainingCallback
import os
from datetime import datetime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--name", default=None, help="Run name")
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config))
    run_name = args.name or f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print("="*70)
    print(f"Continuing Training: {run_name}")
    print("="*70)
    print(f"Loading from: {args.checkpoint}")
    print(f"Config: {args.config}")
    print()

    # Create environment
    env_config = config['environment']
    env = TexasHoldemEnv(
        num_players=3,
        starting_stack=env_config['starting_stack'],
        small_blind=env_config['small_blind'],
        big_blind=env_config['big_blind'],
        min_raise_multiplier=env_config.get('min_raise_multiplier', 1.0),
        reset_stacks_every_n_timesteps=env_config.get('reset_stacks_every_n_timesteps'),
        track_opponents=True
    )

    # Load existing model
    print("Loading checkpoint...")
    training_config = config['training']

    # Get policy_kwargs from config
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
        ent_coef=training_config.get('ent_coef', 0.01),
        vf_coef=training_config.get('vf_coef', 0.5),
        max_grad_norm=training_config.get('max_grad_norm', 0.5),
        tensorboard_log=f"./logs/{run_name}",
        policy_kwargs=policy_kwargs
    )

    # Load the checkpoint weights
    agent.model = agent.model.load(args.checkpoint, env=env)
    print(f"✓ Loaded checkpoint from {args.checkpoint}")

    # Create self-play opponents using the loaded model
    print("\nSetting up self-play opponents...")

    opp1 = OpponentPPO(
        model_path=args.checkpoint,
        env=env,
        name="SelfPlay_Opp1"
    )

    opp2 = OpponentPPO(
        model_path=args.checkpoint,
        env=env,
        name="SelfPlay_Opp2"
    )

    opponents = [
        ('ppo_self', opp1),
        ('ppo_self', opp2)
    ]

    print(f"✓ Opponents: Playing against 2 copies of itself")

    # Wrap environment
    wrapped_env = OpponentAutoPlayWrapper(env, opponents)

    # Setup metrics and callbacks
    model_dir = os.path.join(config['logging']['model_dir'], run_name)
    os.makedirs(model_dir, exist_ok=True)

    metrics = TrainingMetrics(run_name, save_dir="metrics")

    save_callback = TrainingCallback(
        save_freq=config['logging']['save_frequency'],
        save_path=model_dir
    )

    metrics_callback = MetricsCallback(
        metrics=metrics,
        log_freq=10000
    )

    # Continue training
    print("\nContinuing training against self...")
    print(f"Total timesteps: {training_config['total_timesteps']:,}")
    print(f"Architecture: {policy_kwargs if policy_kwargs else 'Default'}")
    print()

    agent.model.set_env(wrapped_env)
    agent.model.learn(
        total_timesteps=training_config['total_timesteps'],
        callback=[save_callback, metrics_callback],
        reset_num_timesteps=False  # Continue from current timestep count
    )

    # Save final model
    final_model_path = os.path.join(model_dir, "final_model")
    agent.save(final_model_path)

    print()
    print("="*70)
    print("Training Complete!")
    print("="*70)
    print(f"Model saved to: {final_model_path}")
    print(f"Metrics: metrics/{run_name}/")
    print()


if __name__ == "__main__":
    main()
