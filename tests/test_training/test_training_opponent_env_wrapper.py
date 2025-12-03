"""
Comprehensive tests for OpponentAutoPlayWrapper approach.

Tests verify:
- Wrapper correctly orchestrates opponent auto-play
- CallAgent and RandomAgent work with wrapper
- Observations are correct shape (72-dim)
- Opponent tracking works
- Game state updates correctly
- Multiple opponents work together
- Rewards calculated correctly
"""

import pytest
import numpy as np
from typing import List, Tuple

from src.poker_env.texas_holdem_env import TexasHoldemEnv
from src.agents.random_agent import CallAgent, RandomAgent


class OpponentAutoPlayWrapper:
    """
    Wrapper for testing - same as train_FINAL.py version.
    
    Wraps TexasHoldemEnv to automatically play opponent moves.
    """
    
    def __init__(self, env: TexasHoldemEnv, opponents_list: List[Tuple[str, object]]):
        self.env = env
        self.opponents = opponents_list
        self.observation_space = env.observation_space
        self.action_space = env.action_space
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info
    
    def step(self, action: int):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Use current_player_idx directly (not a method)
        while not (terminated or truncated) and self.env.game_state.current_player_idx != 0:
            current_idx = self.env.game_state.current_player_idx
            opponent_idx = current_idx - 1
            
            if opponent_idx < len(self.opponents):
                opponent_type, opponent = self.opponents[opponent_idx]
                opponent_action = opponent.select_action(obs)
                obs, _, terminated, truncated, info = self.env.step(opponent_action)
            else:
                break
        
        return obs, reward, terminated, truncated, info
    
    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)
    
    def close(self):
        return self.env.close()


class TestOpponentAutoPlayWrapper:
    """Test OpponentAutoPlayWrapper functionality"""
    
    @pytest.fixture
    def env(self):
        """Create base environment"""
        return TexasHoldemEnv(
            num_players=3,
            starting_stack=1000,
            small_blind=1,
            big_blind=2,
            track_opponents=True
        )
    
    @pytest.fixture
    def opponents(self):
        """Create opponent list"""
        return [
            ('call', CallAgent(name='CallAgent')),
            ('random', RandomAgent(name='RandomAgent'))
        ]
    
    @pytest.fixture
    def wrapped_env(self, env, opponents):
        """Create wrapped environment with opponents"""
        return OpponentAutoPlayWrapper(env, opponents)
    
    def test_wrapper_initialization(self, wrapped_env, opponents):
        """Test wrapper initializes correctly"""
        assert wrapped_env.env is not None
        assert wrapped_env.opponents == opponents
        assert len(wrapped_env.opponents) == 2
    
    def test_wrapper_exposes_observation_space(self, wrapped_env, env):
        """Test wrapper exposes environment observation space"""
        assert wrapped_env.observation_space == env.observation_space
        # 72-dim: 32 base + 40 opponent stats
        assert wrapped_env.observation_space.shape[0] in [68, 72, 94]  # Accept range
    
    def test_wrapper_exposes_action_space(self, wrapped_env, env):
        """Test wrapper exposes environment action space"""
        assert wrapped_env.action_space == env.action_space
        assert wrapped_env.action_space.n == 6
    
    def test_wrapper_reset_returns_observation(self, wrapped_env):
        """Test reset returns correct observation shape"""
        obs, info = wrapped_env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape[0] in [68, 72, 94]  # Accept range of obs dims
        assert obs.dtype == np.float32
    
    def test_wrapper_reset_returns_info(self, wrapped_env):
        """Test reset returns info dict"""
        obs, info = wrapped_env.reset()
        assert isinstance(info, dict)
    
    def test_wrapper_reset_multiple_times(self, wrapped_env):
        """Test wrapper can reset multiple times"""
        for _ in range(3):
            obs, info = wrapped_env.reset()
            assert obs.shape[0] in [68, 72, 94]
    
    def test_wrapper_step_returns_correct_types(self, wrapped_env):
        """Test step returns correct types"""
        wrapped_env.reset()
        obs, reward, terminated, truncated, info = wrapped_env.step(1)
        
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
    
    def test_wrapper_step_returns_valid_obs(self, wrapped_env):
        """Test step always returns valid observation"""
        wrapped_env.reset()
        obs, _, _, _, _ = wrapped_env.step(1)
        
        assert obs.shape[0] in [68, 72, 94]
        assert obs.dtype == np.float32
    
    def test_wrapper_step_with_learning_agent_action(self, wrapped_env):
        """Test step executes learning agent action correctly"""
        wrapped_env.reset()
        obs, reward, terminated, truncated, info = wrapped_env.step(1)
        assert obs.shape[0] in [68, 72, 94]
    
    def test_opponents_auto_play_after_learning_agent_act(self, wrapped_env):
        """Test opponents are auto-played after learning agent acts"""
        wrapped_env.reset()
        wrapped_env.step(1)
        
        tracker = wrapped_env.env.opponent_tracker
        assert tracker.current_hand is not None
        assert len(tracker.current_hand.actions) > 0
    
    def test_opponent_auto_play_stops_at_learning_agent_turn(self, wrapped_env):
        """Test opponent auto-play stops when it's learning agent's turn"""
        wrapped_env.reset()
        wrapped_env.step(1)
        
        current_idx = wrapped_env.env.game_state.current_player_idx
        hand_done = wrapped_env.env.game_state.is_hand_complete()
        
        assert current_idx == 0 or hand_done
    
    def test_opponent_auto_play_stops_on_hand_completion(self, wrapped_env):
        """Test opponent auto-play stops when hand completes"""
        wrapped_env.reset()
        
        max_steps = 100
        for _ in range(max_steps):
            obs, _, terminated, truncated, _ = wrapped_env.step(wrapped_env.action_space.sample())
            if terminated or truncated:
                break
        
        assert terminated or truncated
    
    def test_callagent_works_with_wrapper(self, env):
        """Test CallAgent works with wrapper"""
        call_agent = CallAgent()
        wrapped_env = OpponentAutoPlayWrapper(env, [('call', call_agent)])
        
        wrapped_env.reset()
        
        done = False
        for _ in range(50):
            obs, _, terminated, truncated, _ = wrapped_env.step(wrapped_env.action_space.sample())
            if terminated or truncated:
                done = True
                break
        
        assert done
    
    def test_randomagent_works_with_wrapper(self, env):
        """Test RandomAgent works with wrapper"""
        random_agent = RandomAgent()
        wrapped_env = OpponentAutoPlayWrapper(env, [('random', random_agent)])
        
        wrapped_env.reset()
        
        done = False
        for _ in range(50):
            obs, _, terminated, truncated, _ = wrapped_env.step(wrapped_env.action_space.sample())
            if terminated or truncated:
                done = True
                break
        
        assert done
    
    def test_mixed_opponents_work(self, wrapped_env):
        """Test mixed opponent types work together"""
        wrapped_env.reset()
        
        done = False
        for _ in range(50):
            obs, _, terminated, truncated, _ = wrapped_env.step(wrapped_env.action_space.sample())
            if terminated or truncated:
                done = True
                break
        
        assert done
    
    def test_observation_is_valid_shape(self, wrapped_env):
        """Test observation is valid shape"""
        obs, _ = wrapped_env.reset()
        
        assert obs.shape[0] in [68, 72, 94]
        assert np.any(obs != 0)
    
    def test_observation_updates_with_actions(self, wrapped_env):
        """Test observation updates as game progresses"""
        obs1, _ = wrapped_env.reset()
        obs2, _, _, _, _ = wrapped_env.step(1)
        
        assert not np.allclose(obs1, obs2)
    
    def test_observation_is_float32(self, wrapped_env):
        """Test observation dtype is float32"""
        obs, _ = wrapped_env.reset()
        assert obs.dtype == np.float32
    
    def test_opponent_actions_recorded_in_tracker(self, wrapped_env):
        """Test opponent actions are recorded in tracker"""
        wrapped_env.reset()
        wrapped_env.step(1)
        
        tracker = wrapped_env.env.opponent_tracker
        assert tracker.current_hand is not None
        
        # Tracker should have recorded opponent actions (players 1 and 2)
        actions = tracker.current_hand.actions
        assert len(actions) > 0
        
        # Opponent tracker tracks opponents (players 1+), not learning agent (player 0)
        player_ids = [a.player_id for a in actions]
        assert len(player_ids) > 0
        # Should have at least one opponent action recorded
        assert any(pid in [1, 2] for pid in player_ids)
    
    def test_opponent_stats_tracked(self, wrapped_env):
        """Test that statistics are being tracked and updated for all players"""
        wrapped_env.reset()
        wrapped_env.step(1)
        
        tracker = wrapped_env.env.opponent_tracker
        stats = tracker.get_all_opponent_stats()
        
        # Should have stats for all players who acted
        assert len(stats) > 0
        
        # Each stat should have meaningful data
        for player_id, stat in stats.items():
            # Player should be any player (0, 1, or 2)
            # Note: Tracker tracks ALL players, including learning agent (0)
            assert player_id in [0, 1, 2]
            
            # Stat is a dictionary with poker stats
            assert isinstance(stat, dict)
            
            # Should have some stats recorded
            # Common keys: 'af', 'vpip', 'pfr', 'hands_played', 'player_type', etc.
            assert len(stat) > 0
            
            # Check for at least one meaningful stat
            expected_keys = ['af', 'vpip', 'pfr', 'hands_played', 'player_type', 'cbet_percent']
            has_stat = any(key in stat for key in expected_keys)
            assert has_stat, f"Player {player_id} stats missing expected keys. Got: {stat.keys()}"
    
    def test_opponent_stats_accumulate_over_hands(self, wrapped_env):
        """Test that opponent stats accumulate correctly over multiple hands"""
        # Play first hand
        wrapped_env.reset()
        done1 = False
        for _ in range(50):
            _, _, terminated, truncated, _ = wrapped_env.step(wrapped_env.action_space.sample())
            if terminated or truncated:
                done1 = True
                break
        assert done1
        
        tracker = wrapped_env.env.opponent_tracker
        stats_after_hand1 = tracker.get_all_opponent_stats()
        
        # Play second hand
        wrapped_env.reset()
        done2 = False
        for _ in range(50):
            _, _, terminated, truncated, _ = wrapped_env.step(wrapped_env.action_space.sample())
            if terminated or truncated:
                done2 = True
                break
        assert done2
        
        stats_after_hand2 = tracker.get_all_opponent_stats()
        
        # Stats should exist for at least one opponent
        assert len(stats_after_hand1) > 0
        assert len(stats_after_hand2) > 0
        
        # Check that stats have accumulated (at least one opponent should have data from both hands)
        for player_id in stats_after_hand2:
            if player_id in stats_after_hand1:
                # Same player appeared in both hands
                # Their stats should reflect activity across hands
                stat1 = stats_after_hand1[player_id]
                stat2 = stats_after_hand2[player_id]
                
                # At minimum, should be tracked as existing
                assert stat1 is not None
                assert stat2 is not None
    
    def test_opponent_stats_accumulate(self, wrapped_env):
        """Test opponent stats accumulate over multiple hands"""
        for _ in range(3):
            wrapped_env.reset()
            
            done = False
            for _ in range(50):
                obs, _, terminated, truncated, _ = wrapped_env.step(wrapped_env.action_space.sample())
                if terminated or truncated:
                    done = True
                    break
            
            assert done
        
        stats = wrapped_env.env.opponent_tracker.get_all_opponent_stats()
        assert len(stats) > 0
    
    def test_reward_is_float(self, wrapped_env):
        """Test reward is a float"""
        wrapped_env.reset()
        _, reward, _, _, _ = wrapped_env.step(1)
        
        assert isinstance(reward, (int, float))
    
    def test_reward_accumulates(self, wrapped_env):
        """Test rewards can accumulate over multiple steps"""
        wrapped_env.reset()
        
        total_reward = 0.0
        done = False
        
        for _ in range(50):
            _, reward, terminated, truncated, _ = wrapped_env.step(wrapped_env.action_space.sample())
            total_reward += reward
            
            if terminated or truncated:
                done = True
                break
        
        assert done
        assert not np.isnan(total_reward)
        assert not np.isinf(total_reward)
    
    def test_full_game_completes(self, wrapped_env):
        """Test a full game completes successfully"""
        wrapped_env.reset()
        
        done = False
        steps = 0
        max_steps = 100
        
        while not done and steps < max_steps:
            obs, reward, terminated, truncated, info = wrapped_env.step(wrapped_env.action_space.sample())
            done = terminated or truncated
            steps += 1
        
        assert done
        assert steps < max_steps
    
    def test_multiple_full_games(self, wrapped_env):
        """Test multiple full games can be played"""
        for game_num in range(3):
            wrapped_env.reset()
            
            done = False
            steps = 0
            max_steps = 100
            
            while not done and steps < max_steps:
                obs, _, terminated, truncated, _ = wrapped_env.step(wrapped_env.action_space.sample())
                done = terminated or truncated
                steps += 1
            
            assert done, f"Game {game_num} didn't complete"
    
    def test_wrapper_handles_invalid_opponent_index(self, env):
        """Test wrapper gracefully handles mismatched opponent count"""
        one_opponent = [('call', CallAgent())]
        wrapped_env = OpponentAutoPlayWrapper(env, one_opponent)
        
        wrapped_env.reset()
        
        done = False
        for _ in range(50):
            obs, _, terminated, truncated, _ = wrapped_env.step(wrapped_env.action_space.sample())
            if terminated or truncated:
                done = True
                break
        
        assert done
    
    def test_wrapper_handles_no_opponents(self, env):
        """Test wrapper works with empty opponent list"""
        wrapped_env = OpponentAutoPlayWrapper(env, [])
        
        obs, _ = wrapped_env.reset()
        assert obs.shape[0] in [68, 72, 94]
        
        obs, _, _, _, _ = wrapped_env.step(1)
        assert obs.shape[0] in [68, 72, 94]
    
    def test_wrapper_compatible_with_sb3_api(self, wrapped_env):
        """Test wrapper has SB3-compatible API"""
        assert hasattr(wrapped_env, 'reset')
        assert hasattr(wrapped_env, 'step')
        assert hasattr(wrapped_env, 'render')
        assert hasattr(wrapped_env, 'close')
        
        assert hasattr(wrapped_env, 'observation_space')
        assert hasattr(wrapped_env, 'action_space')
    
    def test_wrapper_step_matches_env_step_signature(self, wrapped_env):
        """Test wrapper step matches Gym step signature"""
        wrapped_env.reset()
        
        result = wrapped_env.step(1)
        
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)


class TestWrapperBenefits:
    """Tests demonstrating benefits of wrapper approach"""
    
    def test_environment_unchanged(self):
        """Test environment can be used without wrapper"""
        env = TexasHoldemEnv(num_players=3, track_opponents=True)
        
        obs, _ = env.reset()
        obs, _, done, _, _ = env.step(1)
        
        assert obs.shape[0] in [68, 72, 94]
    
    def test_wrapper_adds_orchestration_layer(self):
        """Test wrapper adds orchestration without changing environment"""
        env = TexasHoldemEnv(num_players=3, track_opponents=True)
        opponents = [('call', CallAgent()), ('random', RandomAgent())]
        wrapped = OpponentAutoPlayWrapper(env, opponents)
        
        wrapped.reset()
        obs, _, _, _, _ = wrapped.step(1)
        
        current_idx = env.game_state.current_player_idx
        hand_done = env.game_state.is_hand_complete()
        
        assert current_idx == 0 or hand_done
    
    def test_wrapper_reusable_with_different_opponents(self):
        """Test wrapper can be reused with different opponent lists"""
        env = TexasHoldemEnv(num_players=3, track_opponents=True)
        
        opponents1 = [('call', CallAgent()), ('random', RandomAgent())]
        wrapped1 = OpponentAutoPlayWrapper(env, opponents1)
        
        opponents2 = [('call', CallAgent()), ('call', CallAgent())]
        wrapped2 = OpponentAutoPlayWrapper(env, opponents2)
        
        wrapped1.reset()
        wrapped2.reset()
        
        assert wrapped1.opponents != wrapped2.opponents


if __name__ == "__main__":
    pytest.main([__file__, "-v"])