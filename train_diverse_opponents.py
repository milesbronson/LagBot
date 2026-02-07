"""
Train with diverse rule-based opponents
Forces agent to use opponent modeling
"""

import argparse
import yaml
from datetime import datetime
import os
from src.poker_env.texas_holdem_env import TexasHoldemEnv
from src.agents.ppo_agent import PPOAgent
from train import OpponentAutoPlayWrapper, TrainingCallback
from src.training.metrics import TrainingMetrics
from src.training.callbacks import MetricsCallback
from create_diverse_opponents import TightAgent, AggressiveAgent, PassiveAgent, ManiacAgent
import random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/shared_layers_3M.yaml")
    parser.add_argument("--name", default=None)
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config))
    run_name = args.name or f"diverse_opponents_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print("="*70)
    print(f"Training with DIVERSE Opponents: {run_name}")
    print("="*70)

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

    print(f"\nEnvironment: 3 players, opponent tracking ENABLED")

    # Create diverse opponent pool
    opponent_pool = [
        TightAgent("Tight"),
        AggressiveAgent("Aggressive"),
        PassiveAgent("Passive"),
        ManiacAgent("Maniac")
    ]

    print(f"\nOpponent Pool ({len(opponent_pool)} agents):")
    print("  - TightAgent: Folds a lot, only plays premium hands")
    print("  - AggressiveAgent: Raises frequently, very aggressive")
    print("  - PassiveAgent: Calls a lot, rarely raises (calling station)")
    print("  - ManiacAgent: All-in or fold, no middle ground")

    # Randomly select 2 opponents for this training session
    selected_opponents = random.sample(opponent_pool, 2)
    opponents = [(f'{opp.name.lower()}', opp) for opp in selected_opponents]

    print(f"\nSelected opponents for this session:")
    for i, (opp_type, opp) in enumerate(opponents, 1):
        print(f"  Player {i}: {opp.name}")

    # Create agent
    training_config = config['training']
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

    print(f"\nAgent: {agent.name}")
    print(f"Architecture: {policy_kwargs if policy_kwargs else 'Default'}")
    print(f"Total timesteps: {training_config['total_timesteps']:,}")

    # Wrap environment
    wrapped_env = OpponentAutoPlayWrapper(env, opponents)

    # Setup callbacks
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

    # Train
    print("\nStarting training against DIVERSE opponents...")
    print("Agent MUST learn opponent modeling to succeed!\n")

    agent.model.set_env(wrapped_env)
    agent.model.learn(
        total_timesteps=training_config['total_timesteps'],
        callback=[save_callback, metrics_callback]
    )

    # Save final
    final_model_path = os.path.join(model_dir, "final_model")
    agent.save(final_model_path)

    print(f"\n{'='*70}")
    print("Training Complete!")
    print(f"{'='*70}")
    print(f"Model saved to: {final_model_path}")
    print(f"Metrics: metrics/{run_name}/")


if __name__ == "__main__":
    main()
