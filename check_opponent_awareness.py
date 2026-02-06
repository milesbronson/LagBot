"""
Check if the agent is responding to different opponent behaviors
"""

import numpy as np
import torch
from src.poker_env.texas_holdem_env import TexasHoldemEnv
from src.agents.ppo_agent import PPOAgent
from src.agents.random_agent import CallAgent, RandomAgent
import matplotlib.pyplot as plt


def load_trained_agent(model_path):
    """Load the trained PPO agent"""
    env = TexasHoldemEnv(num_players=3, starting_stack=1000, small_blind=5, big_blind=10, track_opponents=True)
    agent = PPOAgent(env=env, name="TestAgent")
    agent.model = agent.model.load(model_path)
    return agent, env


def create_mock_observation_with_opponent_stats(base_obs, opponent_stats):
    """
    Create observation with specific opponent stats

    Observation structure (108 dims):
    - Base game state: 36 dims
    - Opponent 1 stats: 36 dims
    - Opponent 2 stats: 36 dims

    Opponent stats per player (36 dims each):
    - VPIP, PFR, aggression, fold_to_3bet, check_fold, check_call, check_raise, bet_fold, bet_call, bet_raise, etc.
    """
    obs = base_obs.copy()
    # Inject opponent stats into observation (dims 36-72 for opponent 1, 72-108 for opponent 2)
    obs[36:72] = opponent_stats[0]  # Opponent 1
    obs[72:108] = opponent_stats[1]  # Opponent 2
    return obs


def test_opponent_differentiation(agent, env):
    """
    Test if agent's policy changes based on opponent stats

    Strategy:
    1. Create identical game states
    2. Vary only opponent stats in observation
    3. Check if agent's action probabilities change
    """

    print("\n" + "="*80)
    print("TESTING OPPONENT AWARENESS")
    print("="*80)

    # Get a sample observation
    obs, _ = env.reset()
    base_obs = obs[:36]  # Just game state, no opponent stats

    # Define different opponent profiles
    opponent_profiles = {
        "Tight-Passive": {
            "vpip": 0.15,
            "pfr": 0.08,
            "aggression": 0.3,
            "fold_to_3bet": 0.7,
        },
        "Loose-Aggressive": {
            "vpip": 0.45,
            "pfr": 0.35,
            "aggression": 0.8,
            "fold_to_3bet": 0.3,
        },
        "Calling Station": {
            "vpip": 0.5,
            "pfr": 0.1,
            "aggression": 0.2,
            "fold_to_3bet": 0.2,
        },
    }

    # Create full opponent stat vectors (36 dims each)
    def create_opponent_vector(profile):
        vec = np.zeros(36)
        vec[0] = profile["vpip"]
        vec[1] = profile["pfr"]
        vec[2] = profile["aggression"]
        vec[3] = profile["fold_to_3bet"]
        return vec

    results = {}

    for profile_name, profile in opponent_profiles.items():
        print(f"\nTesting against: {profile_name}")
        print(f"  VPIP: {profile['vpip']:.1%}, PFR: {profile['pfr']:.1%}, Aggression: {profile['aggression']:.1%}")

        # Create observation with this opponent profile
        opp1_vec = create_opponent_vector(profile)
        opp2_vec = create_opponent_vector(profile)

        # Create full observation
        full_obs = np.concatenate([base_obs, opp1_vec, opp2_vec])

        # Get action probabilities from agent
        obs_tensor = torch.FloatTensor(full_obs).unsqueeze(0)
        with torch.no_grad():
            action_probs = agent.model.policy.get_distribution(obs_tensor).distribution.probs.numpy()[0]

        results[profile_name] = action_probs

        # Display action probabilities
        action_names = ["Fold", "Call/Check", "Raise 50%", "Raise 100%", "Raise 200%", "All-in"]
        print(f"  Action probabilities:")
        for i, (name, prob) in enumerate(zip(action_names, action_probs)):
            print(f"    {name:15s}: {prob:6.1%}")

    # Compare differences
    print("\n" + "="*80)
    print("DIFFERENTIATION ANALYSIS")
    print("="*80)

    # Calculate KL divergence between policies
    from scipy.stats import entropy

    profiles = list(opponent_profiles.keys())
    for i in range(len(profiles)):
        for j in range(i+1, len(profiles)):
            kl = entropy(results[profiles[i]], results[profiles[j]])
            print(f"\nKL Divergence ({profiles[i]} vs {profiles[j]}): {kl:.4f}")
            if kl < 0.01:
                print(f"  ⚠️  Agent is NOT differentiating between these opponent types!")
            elif kl < 0.1:
                print(f"  ⚡ Agent shows SLIGHT differentiation")
            else:
                print(f"  ✅ Agent shows STRONG differentiation")

    # Visualize
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(action_names))
    width = 0.25

    for i, (profile_name, probs) in enumerate(results.items()):
        ax.bar(x + i*width, probs, width, label=profile_name)

    ax.set_xlabel('Action')
    ax.set_ylabel('Probability')
    ax.set_title('Agent Action Probabilities vs Opponent Type')
    ax.set_xticks(x + width)
    ax.set_xticklabels(action_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('opponent_differentiation.png', dpi=150)
    print(f"\n📊 Saved visualization to: opponent_differentiation.png")

    return results


def check_observation_usage(agent, env):
    """
    Check if opponent features in observation are actually being used
    by examining gradient flow
    """
    print("\n" + "="*80)
    print("CHECKING IF OPPONENT FEATURES ARE USED")
    print("="*80)

    obs, _ = env.reset()
    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).requires_grad_(True)

    # Forward pass
    action_probs = agent.model.policy.get_distribution(obs_tensor).distribution.probs

    # Compute gradients for each action
    for action_idx in range(action_probs.shape[1]):
        obs_tensor.grad = None
        action_probs[0, action_idx].backward(retain_graph=True)

        if obs_tensor.grad is not None:
            # Check gradients for different parts of observation
            base_grad = obs_tensor.grad[0, :36].abs().mean()
            opp1_grad = obs_tensor.grad[0, 36:72].abs().mean()
            opp2_grad = obs_tensor.grad[0, 72:108].abs().mean()

            print(f"\nAction {action_idx}:")
            print(f"  Base game state gradient: {base_grad:.6f}")
            print(f"  Opponent 1 stats gradient: {opp1_grad:.6f}")
            print(f"  Opponent 2 stats gradient: {opp2_grad:.6f}")

            if opp1_grad < base_grad * 0.01 and opp2_grad < base_grad * 0.01:
                print(f"  ⚠️  Opponent features may not be used much!")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python check_opponent_awareness.py <model_path>")
        print("Example: python check_opponent_awareness.py ./models/separate_actor_critic_2M_FINAL/model_350000_steps.zip")
        sys.exit(1)

    model_path = sys.argv[1]

    print("Loading agent...")
    agent, env = load_trained_agent(model_path)

    print("Testing opponent differentiation...")
    results = test_opponent_differentiation(agent, env)

    print("\nChecking feature usage...")
    check_observation_usage(agent, env)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
