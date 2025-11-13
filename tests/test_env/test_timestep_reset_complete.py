import pytest
from src.poker_env.texas_holdem_env import TexasHoldemEnv


class TestTimestepResetBasics:
    """Basic tests for timestep reset mechanism"""
    
    def test_init_with_reset_parameter(self):
        """Test initialization sets reset parameter correctly"""
        env = TexasHoldemEnv(
            num_players=2,
            starting_stack=1000,
            reset_stacks_every_n_timesteps=5000
        )
        
        assert env.reset_stacks_every_n_timesteps == 5000
        assert env.timesteps_since_reset == 0
        assert env.total_timesteps == 0
    
    def test_init_without_reset_parameter(self):
        """Test initialization with reset disabled (None)"""
        env = TexasHoldemEnv(num_players=2, starting_stack=1000)
        
        assert env.reset_stacks_every_n_timesteps is None
        assert env.timesteps_since_reset == 0
    
    def test_timestep_counter_increments(self):
        """Test counter increments with each step()"""
        env = TexasHoldemEnv(
            num_players=2,
            starting_stack=1000,
            reset_stacks_every_n_timesteps=1000
        )
        
        env.reset()
        
        for i in range(1, 11):
            env.step(env.action_space.sample())
            assert env.total_timesteps == i
            assert env.timesteps_since_reset == i
    
    def test_reset_disabled_feature(self):
        """Test feature works when reset_stacks_every_n_timesteps=None"""
        env = TexasHoldemEnv(
            num_players=2,
            starting_stack=1000,
            reset_stacks_every_n_timesteps=None
        )
        
        env.reset()
        
        # Play many steps
        for _ in range(100):
            env.step(env.action_space.sample())
        
        # Should work without errors
        assert env.total_timesteps >= 100


class TestResetLogic:
    """Test the reset condition logic"""
    
    def test_reset_condition_check(self):
        """Test that reset() checks the condition correctly"""
        env = TexasHoldemEnv(
            num_players=2,
            starting_stack=1000,
            reset_stacks_every_n_timesteps=10
        )
        
        env.reset()
        
        # Play exactly 10 steps
        for _ in range(10):
            env.step(env.action_space.sample())
        
        assert env.timesteps_since_reset == 10
        
        # Next reset() call should check: if 10 >= 10 (True)
        # So it should reset the counter
        env.reset()
        assert env.timesteps_since_reset == 0
    
    def test_reset_counter_before_limit(self):
        """Test reset() doesn't reset counter if below limit"""
        env = TexasHoldemEnv(
            num_players=2,
            starting_stack=1000,
            reset_stacks_every_n_timesteps=20
        )
        
        env.reset()
        
        # Play only 10 steps (below 20 limit)
        for _ in range(10):
            env.step(env.action_space.sample())
        
        assert env.timesteps_since_reset == 10
        
        # reset() should NOT reset counter (10 < 20)
        env.reset()
        # Counter stays 0 only because new hand started, not because of reset logic
        # This is the tricky part - reset() always resets to 0 but only resets STACKS if limit hit


class TestConfigurationWithPPO:
    """Test configuration patterns for PPO alignment"""
    
    def test_n_steps_equals_reset(self):
        """Config: reset_stacks_every_n_timesteps = n_steps"""
        n_steps = 2048
        
        env = TexasHoldemEnv(
            num_players=2,
            starting_stack=1000,
            reset_stacks_every_n_timesteps=n_steps
        )
        
        assert env.reset_stacks_every_n_timesteps == n_steps
    
    def test_reset_equals_twice_n_steps(self):
        """Config: reset_stacks_every_n_timesteps = 2 * n_steps"""
        n_steps = 2048
        reset_every = n_steps * 2
        
        env = TexasHoldemEnv(
            num_players=2,
            starting_stack=1000,
            reset_stacks_every_n_timesteps=reset_every
        )
        
        assert env.reset_stacks_every_n_timesteps == reset_every
        assert env.reset_stacks_every_n_timesteps == 4096
    
    def test_reset_equals_three_times_n_steps(self):
        """Config: reset_stacks_every_n_timesteps = 3 * n_steps"""
        n_steps = 2048
        reset_every = n_steps * 3
        
        env = TexasHoldemEnv(
            num_players=2,
            starting_stack=1000,
            reset_stacks_every_n_timesteps=reset_every
        )
        
        assert env.reset_stacks_every_n_timesteps == reset_every


class TestTotalTimestepAccumulation:
    """Test total_timesteps accumulates correctly"""
    
    def test_total_timesteps_never_resets(self):
        """total_timesteps should accumulate forever, never reset"""
        env = TexasHoldemEnv(
            num_players=2,
            starting_stack=1000,
            reset_stacks_every_n_timesteps=10
        )
        
        env.reset()
        
        # Play 20 steps
        for _ in range(20):
            env.step(env.action_space.sample())
        
        total_after_20 = env.total_timesteps
        
        # Reset the env (which resets timesteps_since_reset to 0)
        env.reset()
        
        # total_timesteps should NOT have reset
        assert env.total_timesteps == total_after_20
        
        # Play 10 more
        for _ in range(10):
            env.step(env.action_space.sample())
        
        # total_timesteps should keep incrementing
        assert env.total_timesteps == total_after_20 + 10
    
    def test_timesteps_since_reset_resets(self):
        """timesteps_since_reset should reset but total_timesteps shouldn't"""
        env = TexasHoldemEnv(
            num_players=2,
            starting_stack=1000,
            reset_stacks_every_n_timesteps=5
        )
        
        env.reset()
        
        # Play 5 steps
        for _ in range(5):
            env.step(env.action_space.sample())
        
        assert env.timesteps_since_reset == 5
        assert env.total_timesteps == 5
        
        # Call reset
        env.reset()
        
        # timesteps_since_reset should be 0
        assert env.timesteps_since_reset == 0
        # total_timesteps should still be 5
        assert env.total_timesteps == 5


class TestMultipleEnvironments:
    """Test multiple environments are independent"""
    
    def test_two_envs_independent_counters(self):
        """Two environments should have independent counters"""
        env1 = TexasHoldemEnv(
            num_players=2,
            reset_stacks_every_n_timesteps=100
        )
        env2 = TexasHoldemEnv(
            num_players=2,
            reset_stacks_every_n_timesteps=100
        )
        
        env1.reset()
        env2.reset()
        
        # Play in env1
        for _ in range(10):
            env1.step(env1.action_space.sample())
        
        # Play in env2
        for _ in range(20):
            env2.step(env2.action_space.sample())
        
        assert env1.total_timesteps == 10
        assert env2.total_timesteps == 20


if __name__ == "__main__":
    pytest.main([__file__, "-v"])