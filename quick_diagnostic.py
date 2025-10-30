#!/usr/bin/env python3
"""
Quick Diagnostic Tool for Poker RL Environment

This script quickly checks if your environment is properly set up and working.
Run this first before doing comprehensive testing.
"""

import sys
import os

def check_python_version():
    """Check Python version"""
    print("🔍 Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"   ✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"   ✗ Python {version.major}.{version.minor}.{version.micro} (need 3.8+)")
        return False


def check_dependencies():
    """Check if key dependencies are installed"""
    print("\n🔍 Checking dependencies...")
    dependencies = {
        'gym': '0.26.2',
        'stable_baselines3': '2.2.1',
        'torch': '2.1.0',
        'treys': '0.1.8',
        'numpy': '1.24.3',
        'pytest': '7.4.3',
    }
    
    all_good = True
    for package, expected_version in dependencies.items():
        try:
            module = __import__(package.replace('-', '_'))
            version = getattr(module, '__version__', 'unknown')
            print(f"   ✓ {package} ({version})")
        except ImportError:
            print(f"   ✗ {package} (not installed)")
            all_good = False
    
    return all_good


def check_file_structure():
    """Check if project structure is correct"""
    print("\n🔍 Checking file structure...")
    
    required_dirs = [
        'src/poker_env',
        'src/agents',
        'tests/test_env',
        'tests/test_agents',
        'configs',
    ]
    
    required_files = [
        'src/poker_env/__init__.py',
        'src/poker_env/texas_holdem_env.py',
        'src/poker_env/player.py',
        'src/poker_env/pot_manager.py',
        'src/poker_env/hand_evaluator.py',
        'src/poker_env/game_state.py',
        'src/agents/__init__.py',
        'src/agents/base_agent.py',
        'src/agents/ppo_agent.py',
        'src/agents/random_agent.py',
        'train.py',
        'play.py',
        'requirements.txt',
    ]
    
    all_good = True
    
    for dir_path in required_dirs:
        if os.path.isdir(dir_path):
            print(f"   ✓ {dir_path}/")
        else:
            print(f"   ✗ {dir_path}/ (missing)")
            all_good = False
    
    for file_path in required_files:
        if os.path.isfile(file_path):
            print(f"   ✓ {file_path}")
        else:
            print(f"   ✗ {file_path} (missing)")
            all_good = False
    
    return all_good


def check_imports():
    """Check if modules can be imported"""
    print("\n🔍 Checking imports...")
    
    imports = [
        ('src.poker_env', 'TexasHoldemEnv'),
        ('src.poker_env.player', 'Player'),
        ('src.poker_env.pot_manager', 'PotManager'),
        ('src.poker_env.hand_evaluator', 'HandEvaluator'),
        ('src.poker_env.game_state', 'GameState'),
        ('src.agents.base_agent', 'BaseAgent'),
        ('src.agents.random_agent', 'RandomAgent'),
        ('src.agents.ppo_agent', 'PPOAgent'),
    ]
    
    all_good = True
    for module_name, class_name in imports:
        try:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
            print(f"   ✓ {module_name}.{class_name}")
        except ImportError as e:
            print(f"   ✗ {module_name}.{class_name} (import error)")
            all_good = False
        except AttributeError as e:
            print(f"   ✗ {module_name}.{class_name} (class not found)")
            all_good = False
    
    return all_good


def check_environment_basics():
    """Check if environment can be created and used"""
    print("\n🔍 Checking environment basics...")
    
    try:
        from src.poker_env import TexasHoldemEnv
        
        # Test creation
        env = TexasHoldemEnv(num_players=3)
        print("   ✓ Environment created")
        
        # Test reset
        obs = env.reset()
        print(f"   ✓ Reset works (obs shape: {obs.shape})")
        
        # Test observation space
        if env.observation_space.contains(obs):
            print("   ✓ Observation space valid")
        else:
            print("   ✗ Observation space invalid")
            return False
        
        # Test action space
        if env.action_space.n == 3:
            print("   ✓ Action space valid (3 actions)")
        else:
            print(f"   ✗ Action space invalid ({env.action_space.n} actions)")
            return False
        
        # Test step
        obs, reward, done, info = env.step(1)
        print("   ✓ Step works")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Environment test failed: {e}")
        return False


def check_agent_basics():
    """Check if agents work"""
    print("\n🔍 Checking agent basics...")
    
    try:
        from src.poker_env import TexasHoldemEnv
        from src.agents.random_agent import RandomAgent
        
        env = TexasHoldemEnv(num_players=2)
        agent = RandomAgent()
        
        obs = env.reset()
        action = agent.select_action(obs)
        
        if action in [0, 1, 2]:
            print("   ✓ RandomAgent works")
            return True
        else:
            print(f"   ✗ RandomAgent returned invalid action: {action}")
            return False
            
    except Exception as e:
        print(f"   ✗ Agent test failed: {e}")
        return False


def run_quick_game():
    """Run a quick game to verify everything works"""
    print("\n🔍 Running quick game simulation...")
    
    try:
        from src.poker_env import TexasHoldemEnv
        
        env = TexasHoldemEnv(num_players=3)
        obs = env.reset()
        
        done = False
        steps = 0
        max_steps = 50
        
        while not done and steps < max_steps:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            steps += 1
        
        if done and steps < max_steps:
            print(f"   ✓ Game completed in {steps} steps")
            if 'winnings' in info:
                winners = [pid for pid, amt in info['winnings'].items() if amt > 0]
                print(f"   ✓ Winners determined: Player(s) {winners}")
            return True
        else:
            print(f"   ✗ Game didn't complete (steps: {steps})")
            return False
            
    except Exception as e:
        print(f"   ✗ Game simulation failed: {e}")
        return False


def check_tests_exist():
    """Check if test files exist and can run"""
    print("\n🔍 Checking test suite...")
    
    test_files = [
        'tests/test_env/test_hand_evaluator.py',
        'tests/test_env/test_pot_manager.py',
        'tests/test_env/test_texas_holdem_env.py',
        'tests/test_agents/test_random_agent.py',
    ]
    
    all_exist = True
    for test_file in test_files:
        if os.path.isfile(test_file):
            print(f"   ✓ {test_file}")
        else:
            print(f"   ✗ {test_file} (missing)")
            all_exist = False
    
    if all_exist:
        print("\n   To run tests: pytest")
        print("   For verbose output: pytest -v")
        print("   For coverage: pytest --cov=src")
    
    return all_exist


def print_summary(results):
    """Print diagnostic summary"""
    print("\n" + "="*70)
    print("DIAGNOSTIC SUMMARY")
    print("="*70)
    
    total = len(results)
    passed = sum(results.values())
    
    print(f"\nChecks passed: {passed}/{total}")
    
    for check_name, passed in results.items():
        status = "✓" if passed else "✗"
        print(f"  {status} {check_name}")
    
    print("\n" + "="*70)
    
    if passed == total:
        print("🎉 ALL CHECKS PASSED! 🎉")
        print("\nYour environment is ready to use!")
        print("\nNext steps:")
        print("  1. Run tests: pytest")
        print("  2. Train a bot: python train.py")
        print("  3. Play: python play.py")
    else:
        print("⚠️  SOME CHECKS FAILED")
        print("\nTo fix issues:")
        print("  1. Install package: pip install -e .")
        print("  2. Install dependencies: pip install -r requirements.txt")
        print("  3. Check file structure matches FILE_INDEX.md")
        print("  4. Run this diagnostic again")
        print("  5. See TESTING_GUIDE.md for detailed help")
    
    print("="*70 + "\n")


def main():
    """Main diagnostic function"""
    print("\n" + "="*70)
    print("POKER RL BOT - QUICK DIAGNOSTIC")
    print("="*70)
    print("\nThis will quickly check if your environment is set up correctly.\n")
    
    # Run all checks
    results = {
        #'Python Version': check_python_version(),
        'Dependencies': check_dependencies(),
        'File Structure': check_file_structure(),
        'Imports': check_imports(),
        'Environment Basics': check_environment_basics(),
        'Agent Basics': check_agent_basics(),
        'Quick Game': run_quick_game(),
        'Test Suite': check_tests_exist(),
    }
    
    # Print summaryd
    print_summary(results)
    
    # Return exit code
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)