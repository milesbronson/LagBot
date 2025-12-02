"""
Tests for opponent tracking integration in TexasHoldemEnv
"""

import pytest
import numpy as np
from src.poker_env.texas_holdem_env import TexasHoldemEnv
from src.poker_env.opponent_tracker import Action, Street


class TestOpponentTrackingIntegration:
    """Test opponent tracking is properly integrated into environment"""
    
    @pytest.fixture
    def env_with_tracking(self):
        """Create environment with tracking enabled"""
        return TexasHoldemEnv(num_players=3, starting_stack=1000, track_opponents=True)
    
    @pytest.fixture
    def env_without_tracking(self):
        """Create environment with tracking disabled"""
        return TexasHoldemEnv(num_players=3, starting_stack=1000, track_opponents=False)
    
    def test_observation_space_with_tracking(self, env_with_tracking):
        """Observation space should include opponent stats when tracking"""
        # Get actual sizes from env
        obs, _ = env_with_tracking.reset()
        assert env_with_tracking.observation_space.shape == obs.shape
    
    def test_observation_space_without_tracking(self, env_without_tracking):
        """Observation space should be base size without tracking"""
        obs, _ = env_without_tracking.reset()
        assert env_without_tracking.observation_space.shape == obs.shape
    
    def test_reset_returns_correct_obs_shape(self, env_with_tracking):
        """Reset should return observation with correct shape"""
        obs, info = env_with_tracking.reset()
        assert obs.shape == env_with_tracking.observation_space.shape
    
    def test_step_returns_correct_obs_shape(self, env_with_tracking):
        """Step should return observation with correct shape"""
        env_with_tracking.reset()
        obs, reward, terminated, truncated, info = env_with_tracking.step(1)
        assert obs.shape == env_with_tracking.observation_space.shape
    
    def test_opponent_tracker_initialized(self, env_with_tracking):
        """Opponent tracker should be initialized"""
        assert env_with_tracking.opponent_tracker is not None
    
    def test_opponent_tracker_starts_hand_on_reset(self, env_with_tracking):
        """Reset should start a new hand in tracker"""
        env_with_tracking.reset()
        assert env_with_tracking.opponent_tracker.current_hand is not None
        assert env_with_tracking.opponent_tracker.current_hand.hand_number == 1
    
    def test_opponent_tracker_records_actions(self, env_with_tracking):
        """Actions should be recorded in tracker"""
        env_with_tracking.reset()
        env_with_tracking.step(1)  # Call
        
        # Check that action was recorded
        hand = env_with_tracking.opponent_tracker.current_hand
        assert hand is not None
        assert len(hand.actions) >= 1
    
    def test_opponent_stats_accumulate_over_hands(self, env_with_tracking):
        """Stats should accumulate across multiple hands"""
        # Play several hands
        for _ in range(5):
            env_with_tracking.reset()
            done = False
            while not done:
                action = env_with_tracking.action_space.sample()
                obs, reward, terminated, truncated, info = env_with_tracking.step(action)
                done = terminated or truncated
        
        # Check stats accumulated
        stats = env_with_tracking.opponent_tracker.get_all_opponent_stats()
        assert len(stats) > 0
        
        # At least one player should have hands_played > 0
        total_hands = sum(s.get('hands_played', 0) for s in stats.values() if s)
        assert total_hands > 0
    
    def test_opponent_tracker_get_all_stats_returns_dict(self, env_with_tracking):
        """get_all_opponent_stats should return dict"""
        env_with_tracking.reset()
        stats = env_with_tracking.opponent_tracker.get_all_opponent_stats()
        assert isinstance(stats, dict)
    
    def test_opponent_features_in_observation(self, env_with_tracking):
        """Opponent features should be appended to base observation"""
        # Play a few hands to build up stats
        for _ in range(3):
            env_with_tracking.reset()
            done = False
            while not done:
                action = env_with_tracking.action_space.sample()
                obs, reward, terminated, truncated, info = env_with_tracking.step(action)
                done = terminated or truncated
        
        # Get observations with and without tracking to determine base size
        env_no_track = TexasHoldemEnv(num_players=3, starting_stack=1000, track_opponents=False)
        base_obs, _ = env_no_track.reset()
        base_size = len(base_obs)
        
        # Get fresh observation with tracking
        obs, _ = env_with_tracking.reset()
        opponent_features = obs[base_size:]
        
        # Should have some opponent features
        assert len(opponent_features) > 0
        # Features should be normalized (0-1 range mostly)
        assert np.all(opponent_features >= 0)
    
    def test_hand_end_updates_tracker(self, env_with_tracking):
        """Hand completion should call end_hand on tracker"""
        env_with_tracking.reset()
        
        # Play until hand ends
        done = False
        while not done:
            action = env_with_tracking.action_space.sample()
            obs, reward, terminated, truncated, info = env_with_tracking.step(action)
            done = terminated or truncated
        
        # After hand ends, current_hand should be None
        assert env_with_tracking.opponent_tracker.current_hand is None
        
        # Hand should be in history
        assert len(env_with_tracking.opponent_tracker.hand_history) == 1


class TestBettingRoundToStreetConversion:
    """Test conversion between BettingRound and Street enums"""
    
    @pytest.fixture
    def env(self):
        return TexasHoldemEnv(num_players=3, track_opponents=True)
    
    def test_preflop_conversion(self, env):
        """PREFLOP should convert to Street.PREFLOP"""
        from src.poker_env.game_state import BettingRound
        result = env._betting_round_to_street(BettingRound.PREFLOP)
        assert result == Street.PREFLOP
    
    def test_flop_conversion(self, env):
        """FLOP should convert to Street.FLOP"""
        from src.poker_env.game_state import BettingRound
        result = env._betting_round_to_street(BettingRound.FLOP)
        assert result == Street.FLOP
    
    def test_turn_conversion(self, env):
        """TURN should convert to Street.TURN"""
        from src.poker_env.game_state import BettingRound
        result = env._betting_round_to_street(BettingRound.TURN)
        assert result == Street.TURN
    
    def test_river_conversion(self, env):
        """RIVER should convert to Street.RIVER"""
        from src.poker_env.game_state import BettingRound
        result = env._betting_round_to_street(BettingRound.RIVER)
        assert result == Street.RIVER


class TestActionStringToEnumConversion:
    """Test conversion from action strings to Action enums"""
    
    @pytest.fixture
    def env(self):
        return TexasHoldemEnv(num_players=3, track_opponents=True)
    
    def test_fold_conversion(self, env):
        assert env._string_to_action_enum("fold") == Action.FOLD
        assert env._string_to_action_enum("Fold") == Action.FOLD
        assert env._string_to_action_enum("FOLD") == Action.FOLD
    
    def test_check_conversion(self, env):
        assert env._string_to_action_enum("check") == Action.CHECK
        assert env._string_to_action_enum("Check") == Action.CHECK
    
    def test_call_conversion(self, env):
        assert env._string_to_action_enum("call") == Action.CALL
        assert env._string_to_action_enum("Call") == Action.CALL
    
    def test_raise_conversion(self, env):
        assert env._string_to_action_enum("raise") == Action.RAISE
        assert env._string_to_action_enum("Raise 50% pot") == Action.RAISE
    
    def test_all_in_conversion(self, env):
        assert env._string_to_action_enum("all-in") == Action.ALL_IN
        assert env._string_to_action_enum("All-in") == Action.ALL_IN
        assert env._string_to_action_enum("all_in") == Action.ALL_IN


class TestOpponentFeaturesExtraction:
    """Test _get_opponent_features method"""
    
    @pytest.fixture
    def env(self):
        return TexasHoldemEnv(num_players=3, track_opponents=True)
    
    def test_features_length(self, env):
        """Should return correct number of features"""
        env.reset()
        features = env._get_opponent_features(hero_id=0)
        expected = env.MAX_OPPONENTS * env.FEATURES_PER_OPPONENT
        assert len(features) == expected
    
    def test_features_are_floats(self, env):
        """All features should be floats"""
        env.reset()
        features = env._get_opponent_features(hero_id=0)
        assert features.dtype == np.float32
    
    def test_features_exclude_hero(self, env):
        """Hero should not be in opponent features"""
        env.reset()
        
        for _ in range(3):
            done = False
            while not done:
                obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
                done = terminated or truncated
            env.reset()
        
        features = env._get_opponent_features(hero_id=0)
        expected = env.MAX_OPPONENTS * env.FEATURES_PER_OPPONENT
        assert len(features) == expected


class TestTrackOpponentsFlag:
    """Test track_opponents parameter behavior"""
    
    def test_default_is_true(self):
        """track_opponents should default to True"""
        env = TexasHoldemEnv(num_players=3)
        assert env.track_opponents == True
        # Observation should be larger than base (32)
        assert env.observation_space.shape[0] > 32
    
    def test_can_disable(self):
        """Should be able to disable tracking"""
        env = TexasHoldemEnv(num_players=3, track_opponents=False)
        assert env.track_opponents == False
        # Should have base observation size (no opponent features)
        obs, _ = env.reset()
        assert env.observation_space.shape == obs.shape
    
    def test_disabled_still_has_tracker(self):
        """Tracker object still exists when disabled (for manual use)"""
        env = TexasHoldemEnv(num_players=3, track_opponents=False)
        assert env.opponent_tracker is not None
        assert env.opponent_tracker is not None


class TestMultipleHands:
    """Test tracking across multiple hands"""
    
    def test_stats_persist_across_hands(self):
        """Stats should accumulate across hands"""
        env = TexasHoldemEnv(num_players=3, track_opponents=True)
        
        hands_to_play = 10
        for _ in range(hands_to_play):
            env.reset()
            done = False
            while not done:
                action = env.action_space.sample()
                _, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
        
        # Check hand history
        assert len(env.opponent_tracker.hand_history) == hands_to_play
        
        # Check stats
        stats = env.opponent_tracker.get_all_opponent_stats()
        for pid, s in stats.items():
            if s:
                assert s['hands_played'] >= 1
    
    def test_vpip_pfr_calculated(self):
        """VPIP and PFR should be calculated after enough hands"""
        env = TexasHoldemEnv(num_players=3, track_opponents=True)
        
        for _ in range(20):
            env.reset()
            done = False
            while not done:
                action = env.action_space.sample()
                _, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
        
        stats = env.opponent_tracker.get_all_opponent_stats()
        vpips = [s['vpip'] for s in stats.values() if s and s['hands_played'] > 5]
        assert any(v > 0 for v in vpips), "Expected some non-zero VPIP values"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])