"""
Test suite for opponent tracking and enhanced environment

Tests cover:
- OpponentProfile statistics calculation
- Hand history recording
- Position tracking
- Stack ratio calculations
- Exploitability detection
"""

import pytest
from src.poker_env.opponent_tracker import (
    OpponentTracker, OpponentProfile, Action, Street,
    ActionRecord, StackRatioTracker, StackRatio
)
from datetime import datetime


class TestOpponentProfile:
    """Test individual opponent profile tracking"""
    
    def test_player_type_classification(self):
        """Test opponent classification based on stats"""
        profile = OpponentProfile(player_id=1, player_name="TestPlayer")
        
        # Unknown (not enough hands)
        assert profile.get_player_type() == "UNKNOWN"
        
        # Tight
        profile.hands_played = 20
        profile.vpip = 0.20
        profile.pfr = 0.15
        profile.af = 0.8
        assert profile.get_player_type() == "TIGHT"
        
        # Loose aggressive
        profile.vpip = 0.60
        profile.pfr = 0.50
        profile.af = 2.5
        assert profile.get_player_type() == "LOOSE_AGGRESSIVE"
        
        # Tight aggressive
        profile.vpip = 0.20
        profile.pfr = 0.18
        profile.af = 2.5
        assert profile.get_player_type() == "TIGHT_AGGRESSIVE"
    
    def test_profile_to_dict(self):
        """Test serialization of profile"""
        profile = OpponentProfile(
            player_id=1,
            player_name="Player1",
            hands_played=100,
            vpip=0.35,
            pfr=0.25,
            af=1.5
        )
        
        stats_dict = profile.to_dict()
        
        assert stats_dict['player_id'] == 1
        assert stats_dict['player_name'] == "Player1"
        assert stats_dict['hands_played'] == 100
        assert 'vpip' in stats_dict
        assert 'player_type' in stats_dict


class TestOpponentTracker:
    """Test multi-opponent tracking system"""
    
    def setup_method(self):
        """Create tracker for each test"""
        self.tracker = OpponentTracker(max_history_hands=100)
    
    def test_start_hand_initialization(self):
        """Test starting a new hand"""
        players = [
            {'id': 0, 'name': 'Agent', 'stack': 1000},
            {'id': 1, 'name': 'Player1', 'stack': 1000},
            {'id': 2, 'name': 'Player2', 'stack': 1000},
        ]
        
        self.tracker.start_hand(
            hand_number=1,
            players=players,
            dealer_position=0,
            small_blind=1,
            big_blind=2
        )
        
        assert self.tracker.hand_number == 1
        assert self.tracker.current_hand is not None
        assert 0 in self.tracker.opponents
        assert 1 in self.tracker.opponents
        assert 2 in self.tracker.opponents
    
    def test_record_action(self):
        """Test recording player actions"""
        players = [
            {'id': 0, 'name': 'Agent', 'stack': 1000},
            {'id': 1, 'name': 'Player1', 'stack': 1000},
        ]
        
        self.tracker.start_hand(
            hand_number=1,
            players=players,
            dealer_position=0,
            small_blind=1,
            big_blind=2
        )
        
        # Record action - position is now integer (0 = dealer/button)
        self.tracker.record_action(
            player_id=1,
            player_name='Player1',
            action=Action.RAISE,
            amount=10,
            pot_size=15,
            stack_before=1000,
            stack_after=990,
            street=Street.PREFLOP,
            position=0  # Button position as integer
        )
        
        assert len(self.tracker.current_hand.actions) == 1
        assert self.tracker.current_hand.actions[0].action == Action.RAISE
    
    def test_end_hand_updates_stats(self):
        """Test that ending hand updates opponent statistics"""
        players = [
            {'id': 0, 'name': 'Agent', 'stack': 1000},
            {'id': 1, 'name': 'Player1', 'stack': 1000},
        ]
        
        self.tracker.start_hand(
            hand_number=1,
            players=players,
            dealer_position=0,
            small_blind=1,
            big_blind=2
        )
        
        # Add some actions
        for _ in range(3):
            self.tracker.record_action(
                player_id=1,
                player_name='Player1',
                action=Action.CALL,
                amount=5,
                pot_size=10,
                stack_before=1000,
                stack_after=995,
                street=Street.PREFLOP,
                position=0  # Button position as integer
            )
        
        # End hand
        self.tracker.end_hand(
            winners=[0],
            winnings={0: 20, 1: -20},
            final_stacks={0: 1020, 1: 980}
        )
        
        # Check opponent profile updated
        assert self.tracker.opponents[1].hands_played >= 1
    
    def test_hand_history_recording(self):
        """Test that hand history is maintained"""
        players = [
            {'id': 0, 'name': 'Agent', 'stack': 1000},
            {'id': 1, 'name': 'Player1', 'stack': 1000},
        ]
        
        for hand_num in range(5):
            self.tracker.start_hand(
                hand_number=hand_num,
                players=players,
                dealer_position=0,
                small_blind=1,
                big_blind=2
            )
            
            self.tracker.record_action(
                player_id=1,
                player_name='Player1',
                action=Action.FOLD,
                amount=0,
                pot_size=3,
                stack_before=1000,
                stack_after=1000,
                street=Street.PREFLOP,
                position=0  # Button position as integer
            )
            
            self.tracker.end_hand(
                winners=[0],
                winnings={0: 3, 1: -1},
                final_stacks={0: 1003, 1: 997}
            )
        
        assert len(self.tracker.hand_history) == 5
        recent = self.tracker.get_recent_hands(opponent_id=1, limit=10)
        assert len(recent) == 5


class TestOpponentFeatures:
    """Test feature generation for ML models"""
    
    def setup_method(self):
        """Create tracker with some data"""
        self.tracker = OpponentTracker()
        
        players = [
            {'id': 0, 'name': 'Agent', 'stack': 1000},
            {'id': 1, 'name': 'Opponent1', 'stack': 1000},
            {'id': 2, 'name': 'Opponent2', 'stack': 1000},
        ]
        
        self.tracker.start_hand(
            hand_number=1,
            players=players,
            dealer_position=0,
            small_blind=1,
            big_blind=2
        )
    
    def test_opponent_features_vector_size(self):
        """Test that feature vector has correct size"""
        features = self.tracker.get_opponent_features(
            opponent_id=1,
            num_seats=3,
            recent_hand_window=10
        )
        
        # Should be 5 base + 10 recent = 15
        assert len(features) == 15
        assert all(isinstance(f, float) for f in features)
    
    def test_opponent_features_normalized(self):
        """Test that features are normalized 0-1"""
        features = self.tracker.get_opponent_features(
            opponent_id=1,
            num_seats=3,
            recent_hand_window=5
        )
        
        # Most should be 0-1 (except AF which can exceed 1)
        for feature in features:
            assert 0.0 <= feature <= 1.5  # AF can exceed 1
    
    def test_all_opponents_features(self):
        """Test getting features for all opponents"""
        all_features = self.tracker.get_all_opponents_features(
            learning_agent_id=0,
            num_seats=3
        )
        
        # Should have 2 opponents (excluding agent)
        assert len(all_features) == 2
        assert 1 in all_features
        assert 2 in all_features
        assert 0 not in all_features  # Agent should be excluded


class TestPositionTracking:
    """Test position detection and tracking"""
    
    def test_position_as_integer(self):
        """Test that positions are tracked as integers"""
        # Positions are now integers: 0=button, 1=SB, 2=BB, etc.
        position_values = {
            0: "Button",
            1: "Small Blind",
            2: "Big Blind",
            3: "Early",
            5: "Cutoff"
        }
        
        for pos, name in position_values.items():
            assert isinstance(pos, int)
    
    def test_position_in_action_record(self):
        """Test position stored in action record as integer"""
        action = ActionRecord(
            street=Street.PREFLOP,
            player_id=1,
            player_name="Player1",
            action=Action.RAISE,
            amount=10,
            pot_size=15,
            stack_before=1000,
            stack_after=990,
            position=5,  # Cutoff position as integer
            timestamp=datetime.now().timestamp()
        )
        
        assert action.position == 5
        assert isinstance(action.position, int)


class TestStackRatios:
    """Test stack ratio calculations"""
    
    def test_stack_ratio_calculation(self):
        """Test basic stack ratio calculation"""
        ratio = StackRatioTracker.get_stack_ratios(
            player_stack=2000,
            opponent_stack=1500,
            bb=50,
            pot_size=150
        )
        
        assert ratio.player_stack == 40  # 2000 / 50
        assert ratio.opponent_stack == 30  # 1500 / 50
        assert ratio.pot_size == 3  # 150 / 50
        assert ratio.bb == 50
    
    def test_stack_depth_classification(self):
        """Test stack depth classification"""
        assert StackRatioTracker.classify_stack_depth(5.0) == "SHALLOW"
        assert StackRatioTracker.classify_stack_depth(25.0) == "MEDIUM"
        assert StackRatioTracker.classify_stack_depth(75.0) == "DEEP"
        assert StackRatioTracker.classify_stack_depth(150.0) == "VERY_DEEP"


class TestExploitability:
    """Test exploitability detection"""
    
    def setup_method(self):
        """Create tracker with pre-made opponent"""
        self.tracker = OpponentTracker()
        
        # Create a known loose-aggressive opponent
        self.tracker.opponents[1] = OpponentProfile(
            player_id=1,
            player_name="Loose_Aggressive",
            hands_played=50,
            vpip=0.65,
            pfr=0.55,
            af=3.5
        )
        
        # Create a very tight opponent
        self.tracker.opponents[2] = OpponentProfile(
            player_id=2,
            player_name="Very_Tight",
            hands_played=50,
            vpip=0.10,
            pfr=0.08,
            af=0.5
        )
    
    def test_identify_loose_vpip(self):
        """Test identification of loose VPIP opponent"""
        exploitable = self.tracker.get_exploitable_opponents(threshold_hands=10)
        
        assert len(exploitable) > 0
        loose_opp = next((o for o in exploitable if o['opponent_id'] == 1), None)
        assert loose_opp is not None
        assert any(e['type'] == 'loose_vpip' for e in loose_opp['exploits'])
    
    def test_identify_tight_vpip(self):
        """Test identification of tight VPIP opponent"""
        exploitable = self.tracker.get_exploitable_opponents(threshold_hands=10)
        
        tight_opp = next((o for o in exploitable if o['opponent_id'] == 2), None)
        assert tight_opp is not None
        assert any(e['type'] == 'tight_vpip' for e in tight_opp['exploits'])
    
    def test_identify_aggression_exploit(self):
        """Test identification of aggression-based exploits"""
        exploitable = self.tracker.get_exploitable_opponents(threshold_hands=10)
        
        loose_agg = next((o for o in exploitable if o['opponent_id'] == 1), None)
        assert loose_agg is not None
        assert any(e['type'] == 'aggressive' for e in loose_agg['exploits'])
        
        tight_passive = next((o for o in exploitable if o['opponent_id'] == 2), None)
        assert tight_passive is not None
        assert any(e['type'] == 'passive' for e in tight_passive['exploits'])


class TestHandHistoryStorage:
    """Test hand history storage and retrieval"""
    
    def setup_method(self):
        self.tracker = OpponentTracker(max_history_hands=10)
    
    def test_hand_history_limit(self):
        """Test that hand history respects max size"""
        for i in range(20):
            players = [
                {'id': 0, 'name': 'Agent', 'stack': 1000},
                {'id': 1, 'name': 'Opp1', 'stack': 1000},
            ]
            
            self.tracker.start_hand(i, players, 0, 1, 2)
            self.tracker.end_hand([], {}, {})
        
        # Should only keep last 10
        assert len(self.tracker.hand_history) == 10
    
    def test_get_recent_hands(self):
        """Test retrieving recent hands for opponent"""
        players = [
            {'id': 0, 'name': 'Agent', 'stack': 1000},
            {'id': 1, 'name': 'Opp1', 'stack': 1000},
        ]
        
        for i in range(5):
            self.tracker.start_hand(i, players, 0, 1, 2)
            self.tracker.end_hand([], {}, {})
        
        recent = self.tracker.get_recent_hands(opponent_id=1, limit=3)
        assert len(recent) == 3


def test_integration_full_hand_workflow():
    """Integration test: full hand tracking workflow"""
    tracker = OpponentTracker()
    
    # Setup hand
    players = [
        {'id': 0, 'name': 'Agent', 'stack': 1000},
        {'id': 1, 'name': 'Player1', 'stack': 1000},
        {'id': 2, 'name': 'Player2', 'stack': 1000},
    ]
    
    tracker.start_hand(1, players, 0, 1, 2)
    
    # Preflop action - positions as integers
    tracker.record_action(1, 'Player1', Action.RAISE, 10, 15, 1000, 990,
                         Street.PREFLOP, 0)  # Button = 0
    tracker.record_action(2, 'Player2', Action.CALL, 10, 25, 1000, 990,
                         Street.PREFLOP, 1)  # Small Blind = 1
    
    # Flop action
    tracker.record_action(1, 'Player1', Action.BET, 20, 45, 990, 970,
                         Street.FLOP, 0)  # Button = 0
    tracker.record_action(2, 'Player2', Action.FOLD, 0, 45, 990, 990,
                         Street.FLOP, 1)  # Small Blind = 1
    
    # End hand - positions as integers in record_positions
    tracker.record_positions({0: 2, 1: 0, 2: 1})  # 0=BB, 1=Button, 2=SB
    tracker.end_hand([1], {1: 45, 0: 0, 2: -5}, {0: 998, 1: 1045, 2: 957})
    
    # Verify
    assert tracker.opponents[1].hands_played >= 1
    assert tracker.opponents[2].hands_played >= 1
    assert len(tracker.hand_history) == 1
    
    # Get features
    features_1 = tracker.get_opponent_features(1)
    assert len(features_1) > 0
    assert all(isinstance(f, float) for f in features_1)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])