"""
Diagnostic script to verify that PPO agent is actually updating its policy.
"""

import numpy as np
import torch as th
from src.poker_env.texas_holdem_env import TexasHoldemEnv
from src.agents.ppo_agent import PPOAgent


def test_policy_is_updating():
    """Test that policy weights change during training"""
    
    print("\n" + "="*70)
    print("TEST 1: Policy Weight Changes")
    print("="*70)
    
    env = TexasHoldemEnv(num_players=3)
    agent = PPOAgent(
        env,
        name="test_agent",
        learning_rate=0.001,  # Higher LR to see changes faster
        n_steps=256,  # Smaller for faster updates
        tensorboard_log="./logs/test_policy/"
    )
    
    # Get initial policy weights
    initial_weights = []
    for param in agent.model.policy.net_arch:
        if hasattr(param, 'weight'):
            initial_weights.append(param.weight.data.clone())
    
    print("Initial policy weights (first layer, first 5 values):")
    print(agent.model.policy.features_extractor.net[0].weight.data[0, :5])
    
    # Train for a bit
    print("\nTraining for 10,000 steps...")
    agent.train(total_timesteps=10000)
    
    # Get new weights
    print("\nFinal policy weights (first layer, first 5 values):")
    print(agent.model.policy.features_extractor.net[0].weight.data[0, :5])
    
    # Check if weights changed
    final_weights = agent.model.policy.features_extractor.net[0].weight.data
    initial_w = agent.model.policy.features_extractor.net[0].weight.data.clone()
    
    weight_change = (final_weights - initial_w).abs().sum().item()
    print(f"\nTotal weight change: {weight_change:.6f}")
    
    if weight_change > 0.01:
        print("✓ PASS: Weights are updating!")
        return True
    else:
        print("✗ FAIL: Weights are NOT changing")
        return False


def test_action_distribution_changes():
    """Test that action distribution changes as agent learns"""
    
    print("\n" + "="*70)
    print("TEST 2: Action Distribution Changes")
    print("="*70)
    
    env = TexasHoldemEnv(num_players=3)
    agent = PPOAgent(
        env,
        name="test_agent2",
        learning_rate=0.0003,
        n_steps=512,
    )
    
    # Collect initial action distribution
    print("Collecting initial action distribution (1000 steps)...")
    initial_actions = collect_action_distribution(env, agent, n_steps=1000)
    print("Initial action distribution:")
    print(f"  Fold: {initial_actions['fold']:.1%}")
    print(f"  Call: {initial_actions['call']:.1%}")
    print(f"  Raise: {initial_actions['raise']:.1%}")
    print(f"  All-in: {initial_actions['all_in']:.1%}")
    
    # Train
    print("\nTraining for 50,000 steps...")
    agent.train(total_timesteps=50000)
    
    # Collect final action distribution
    print("\nCollecting final action distribution (1000 steps)...")
    final_actions = collect_action_distribution(env, agent, n_steps=1000)
    print("Final action distribution:")
    print(f"  Fold: {final_actions['fold']:.1%}")
    print(f"  Call: {final_actions['call']:.1%}")
    print(f"  Raise: {final_actions['raise']:.1%}")
    print(f"  All-in: {final_actions['all_in']:.1%}")
    
    # Check if distribution changed
    distribution_shift = sum([
        abs(initial_actions['fold'] - final_actions['fold']),
        abs(initial_actions['call'] - final_actions['call']),
        abs(initial_actions['raise'] - final_actions['raise']),
    ])
    
    print(f"\nTotal distribution shift: {distribution_shift:.3f}")
    
    if distribution_shift > 0.05:
        print("✓ PASS: Action distribution changed significantly!")
        return True
    else:
        print("✗ FAIL: Action distribution barely changed")
        return False


def test_gradient_flow():
    """Test that gradients are flowing through the network"""
    
    print("\n" + "="*70)
    print("TEST 3: Gradient Flow")
    print("="*70)
    
    env = TexasHoldemEnv(num_players=3)
    agent = PPOAgent(
        env,
        name="test_agent3",
        learning_rate=0.0003,
        n_steps=512,
    )
    
    # Run one training step and check gradients
    obs, _ = env.reset()
    obs_tensor = th.FloatTensor(obs).unsqueeze(0)
    
    # Get action and value
    with th.no_grad():
        action, _ = agent.model.predict(obs_tensor)
        action_logits, value = agent.model.policy(obs_tensor)
    
    print("Checking gradient computation...")
    
    # Compute a simple loss to see gradients
    loss = value.mean()
    loss.backward()
    
    # Check if gradients exist
    param_count = 0
    has_grad_count = 0
    
    for name, param in agent.model.policy.named_parameters():
        if param.requires_grad:
            param_count += 1
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad_count += 1
    
    print(f"Parameters with gradients: {has_grad_count}/{param_count}")
    
    if has_grad_count > param_count * 0.8:  # At least 80% have gradients
        print("✓ PASS: Gradients flowing through network!")
        return True
    else:
        print("✗ FAIL: Gradient flow is blocked")
        return False


def test_loss_decreases():
    """Test that loss decreases during training"""
    
    print("\n" + "="*70)
    print("TEST 4: Training Loss Decreases")
    print("="*70)
    
    env = TexasHoldemEnv(num_players=3)
    agent = PPOAgent(
        env,
        name="test_agent4",
        learning_rate=0.001,
        n_steps=512,
        n_epochs=10,
    )
    
    print("Training and monitoring loss...")
    
    # Train and capture losses from tensorboard
    agent.train(total_timesteps=20000)
    
    print("✓ Training completed. Check tensorboard for loss curves:")
    print("  tensorboard --logdir ./logs/test_policy/test_agent4/")
    print("\nLook for:")
    print("  - train/policy_loss: Should decrease")
    print("  - train/value_loss: Should decrease")
    
    return True


def test_prediction_determinism():
    """Test that predictions become more deterministic"""
    
    print("\n" + "="*70)
    print("TEST 5: Prediction Determinism")
    print("="*70)
    
    env = TexasHoldemEnv(num_players=3)
    agent = PPOAgent(
        env,
        name="test_agent5",
        learning_rate=0.0003,
    )
    
    obs, _ = env.reset()
    
    print("Before training:")
    print("Taking 20 predictions from same observation...")
    initial_predictions = []
    for i in range(20):
        action, _ = agent.model.predict(obs, deterministic=False)
        initial_predictions.append(action)
    
    initial_entropy = len(set(initial_predictions)) / 20
    print(f"Entropy (diversity): {initial_entropy:.2f} (1.0 = fully random)")
    print(f"Action counts: {np.bincount(initial_predictions)}")
    
    print("\nTraining for 50,000 steps...")
    agent.train(total_timesteps=50000)
    
    print("\nAfter training:")
    print("Taking 20 predictions from same observation...")
    final_predictions = []
    for i in range(20):
        action, _ = agent.model.predict(obs, deterministic=False)
        final_predictions.append(action)
    
    final_entropy = len(set(final_predictions)) / 20
    print(f"Entropy (diversity): {final_entropy:.2f}")
    print(f"Action counts: {np.bincount(final_predictions)}")
    
    entropy_decrease = initial_entropy - final_entropy
    print(f"\nEntropy decrease: {entropy_decrease:.2f}")
    
    if entropy_decrease > 0.2:
        print("✓ PASS: Agent is becoming more deterministic (learning!)")
        return True
    else:
        print("✗ FAIL: Agent entropy unchanged (not learning)")
        return False


def collect_action_distribution(env, agent, n_steps=1000):
    """Collect distribution of actions taken"""
    
    action_counts = {'fold': 0, 'call': 0, 'raise': 0, 'all_in': 0}
    total = 0
    
    obs, _ = env.reset()
    done = False
    
    for step in range(n_steps):
        if done:
            obs, _ = env.reset()
            done = False
        
        action, _ = agent.model.predict(obs, deterministic=False)
        
        if action == 0:
            action_counts['fold'] += 1
        elif action == 1:
            action_counts['call'] += 1
        elif action >= 5:  # Last action is all-in
            action_counts['all_in'] += 1
        else:
            action_counts['raise'] += 1
        
        obs, reward, done, truncated, info = env.step(action)
        done = done or truncated
        total += 1
    
    # Convert to percentages
    for key in action_counts:
        action_counts[key] = action_counts[key] / total
    
    return action_counts


def run_all_tests():
    """Run all diagnostic tests"""
    
    print("\n" + "="*70)
    print("POLICY UPDATE DIAGNOSTICS")
    print("="*70)
    
    results = {}
    
    try:
        results['weight_changes'] = test_policy_is_updating()
    except Exception as e:
        print(f"✗ TEST FAILED: {e}")
        results['weight_changes'] = False
    
    try:
        results['action_distribution'] = test_action_distribution_changes()
    except Exception as e:
        print(f"✗ TEST FAILED: {e}")
        results['action_distribution'] = False
    
    try:
        results['gradient_flow'] = test_gradient_flow()
    except Exception as e:
        print(f"✗ TEST FAILED: {e}")
        results['gradient_flow'] = False
    
    try:
        results['loss_decreases'] = test_loss_decreases()
    except Exception as e:
        print(f"✗ TEST FAILED: {e}")
        results['loss_decreases'] = False
    
    try:
        results['prediction_determinism'] = test_prediction_determinism()
    except Exception as e:
        print(f"✗ TEST FAILED: {e}")
        results['prediction_determinism'] = False
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    total_passed = sum(results.values())
    total_tests = len(results)
    
    print(f"\nTotal: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("\n🎉 All tests passed! Agent is learning.")
    elif total_passed >= total_tests * 0.6:
        print("\n⚠️  Partial success. Some learning happening but check issues.")
    else:
        print("\n❌ Most tests failed. Agent is NOT learning properly.")
        print("\nCommon causes:")
        print("1. Learning rate too low")
        print("2. Network too small")
        print("3. Observation space issue (check env.reset())")
        print("4. Action space mismatch")


if __name__ == "__main__":
    # Run just one test or all
    run_all_tests()
    
    # Or run individual tests:
    # test_policy_is_updating()
    # test_action_distribution_changes()
    # test_gradient_flow()