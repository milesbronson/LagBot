"""
Test that opponent statistics are calculated correctly.

This test simulates specific game scenarios and verifies that
stats like VPIP, PFR, AF, etc. are computed accurately.
"""

import pytest
from src.poker_env.opponent_tracker import OpponentTracker, Action, Street


class TestOpponentStatsCalculation:
    """Test opponent statistics are calculated correctly"""
    
    @pytest.fixture
    def tracker(self):
        """Create fresh tracker for each test"""
        return OpponentTracker()
    
    def test_vpip_calculation_single_hand(self, tracker):
        """Test VPIP (Voluntarily Put $ In Pot) calculation - single hand"""
        # Setup: 3 players
        players = [
            {'id': 0, 'name': 'Agent', 'stack': 1000},
            {'id': 1, 'name': 'Player1', 'stack': 1000},
            {'id': 2, 'name': 'Player2', 'stack': 1000},
        ]
        
        # Hand 1: Player 1 CALLS (puts $ in pot voluntarily)
        tracker.start_hand(1, players, dealer_position=0, small_blind=1, big_blind=2)
        tracker.record_action(
            player_id=1, player_name='Player1',
            action=Action.CALL, amount=2,
            pot_size=3, stack_before=1000, stack_after=998,
            street=Street.PREFLOP, position=0
        )
        tracker.end_hand(winners=[0], winnings={0: 3, 1: -1, 2: 0}, final_stacks={0: 1003, 1: 999, 2: 1000})
        
        # Hand 2: Player 1 FOLDS (doesn't put $ in)
        tracker.start_hand(2, players, dealer_position=1, small_blind=1, big_blind=2)
        tracker.record_action(
            player_id=1, player_name='Player1',
            action=Action.FOLD, amount=0,
            pot_size=3, stack_before=999, stack_after=999,
            street=Street.PREFLOP, position=1
        )
        tracker.end_hand(winners=[0], winnings={0: 2, 1: 0, 2: -2}, final_stacks={0: 1005, 1: 999, 2: 998})
        
        # Get stats
        stats = tracker.get_all_opponent_stats()
        player1_stats = stats[1]
        
        # VPIP = hands with $ in / total hands = 1 / 2 = 50%
        assert player1_stats['vpip'] == 0.5, f"Expected VPIP 0.5, got {player1_stats['vpip']}"
        assert player1_stats['hands_played'] == 2
    
    def test_pfr_calculation(self, tracker):
        """Test PFR (Pre-Flop Raise) calculation"""
        players = [
            {'id': 0, 'name': 'Agent', 'stack': 1000},
            {'id': 1, 'name': 'Player1', 'stack': 1000},
        ]
        
        # Hand 1: Player 1 RAISES preflop
        tracker.start_hand(1, players, dealer_position=0, small_blind=1, big_blind=2)
        tracker.record_action(
            player_id=1, player_name='Player1',
            action=Action.RAISE, amount=10,
            pot_size=13, stack_before=1000, stack_after=990,
            street=Street.PREFLOP, position=0
        )
        tracker.end_hand(winners=[1], winnings={0: -10, 1: 10}, final_stacks={0: 990, 1: 1010})
        
        # Hand 2: Player 1 CALLS (not a raise)
        tracker.start_hand(2, players, dealer_position=1, small_blind=1, big_blind=2)
        tracker.record_action(
            player_id=1, player_name='Player1',
            action=Action.CALL, amount=2,
            pot_size=3, stack_before=1010, stack_after=1008,
            street=Street.PREFLOP, position=1
        )
        tracker.end_hand(winners=[0], winnings={0: 3, 1: -3}, final_stacks={0: 993, 1: 1007})
        
        # Hand 3: Player 1 FOLDS (no PFR)
        tracker.start_hand(3, players, dealer_position=0, small_blind=1, big_blind=2)
        tracker.record_action(
            player_id=1, player_name='Player1',
            action=Action.FOLD, amount=0,
            pot_size=3, stack_before=1007, stack_after=1007,
            street=Street.PREFLOP, position=0
        )
        tracker.end_hand(winners=[0], winnings={0: 2, 1: -1}, final_stacks={0: 995, 1: 1007})
        
        stats = tracker.get_all_opponent_stats()
        player1_stats = stats[1]
        
        # PFR = raises preflop / hands played = 1 / 3 = 33.33%
        expected_pfr = round(1/3, 3)
        assert player1_stats['pfr'] == expected_pfr, f"Expected PFR {expected_pfr}, got {player1_stats['pfr']}"
        assert player1_stats['hands_played'] == 3
    
    def test_aggression_factor_calculation(self, tracker):
        """Test AF (Aggression Factor) calculation"""
        players = [
            {'id': 0, 'name': 'Agent', 'stack': 1000},
            {'id': 1, 'name': 'Player1', 'stack': 1000},
        ]
        
        # Hand 1: Player 1 RAISES then RAISES again = 2 aggressive actions
        tracker.start_hand(1, players, dealer_position=0, small_blind=1, big_blind=2)
        tracker.record_action(
            player_id=1, player_name='Player1',
            action=Action.RAISE, amount=10,
            pot_size=13, stack_before=1000, stack_after=990,
            street=Street.PREFLOP, position=0
        )
        tracker.record_action(
            player_id=1, player_name='Player1',
            action=Action.RAISE, amount=20,
            pot_size=33, stack_before=990, stack_after=970,
            street=Street.FLOP, position=0
        )
        tracker.end_hand(winners=[1], winnings={0: -50, 1: 50}, final_stacks={0: 950, 1: 1050})
        
        # Hand 2: Player 1 CALLS = 1 passive action
        tracker.start_hand(2, players, dealer_position=1, small_blind=1, big_blind=2)
        tracker.record_action(
            player_id=1, player_name='Player1',
            action=Action.CALL, amount=5,
            pot_size=8, stack_before=1050, stack_after=1045,
            street=Street.PREFLOP, position=1
        )
        tracker.end_hand(winners=[0], winnings={0: 10, 1: -10}, final_stacks={0: 960, 1: 1040})
        
        stats = tracker.get_all_opponent_stats()
        player1_stats = stats[1]
        
        # AF = (raises + bets) / calls = 2 / 1 = 2.0
        expected_af = 2.0
        assert player1_stats['af'] == expected_af, f"Expected AF {expected_af}, got {player1_stats['af']}"
    
    def test_aggression_factor_zero_when_no_aggression(self, tracker):
        """Test AF is 0 when player never bets/raises"""
        players = [
            {'id': 0, 'name': 'Agent', 'stack': 1000},
            {'id': 1, 'name': 'Player1', 'stack': 1000},
        ]
        
        # Hand 1: Player 1 only calls
        tracker.start_hand(1, players, dealer_position=0, small_blind=1, big_blind=2)
        tracker.record_action(
            player_id=1, player_name='Player1',
            action=Action.CALL, amount=2,
            pot_size=4, stack_before=1000, stack_after=998,
            street=Street.PREFLOP, position=0
        )
        tracker.end_hand(winners=[0], winnings={0: 4, 1: -2}, final_stacks={0: 1004, 1: 998})
        
        stats = tracker.get_all_opponent_stats()
        player1_stats = stats[1]
        
        # AF = 0 / 1 = 0 (no aggressive actions)
        assert player1_stats['af'] == 0.0, f"Expected AF 0.0, got {player1_stats['af']}"
    
    def test_went_to_showdown_percentage(self, tracker):
        """Test showdown percentage calculation"""
        players = [
            {'id': 0, 'name': 'Agent', 'stack': 1000},
            {'id': 1, 'name': 'Player1', 'stack': 1000},
        ]
        
        # Hand 1: Player 1 goes to showdown (winner)
        tracker.start_hand(1, players, dealer_position=0, small_blind=1, big_blind=2)
        tracker.record_action(
            player_id=1, player_name='Player1',
            action=Action.CALL, amount=2,
            pot_size=4, stack_before=1000, stack_after=998,
            street=Street.PREFLOP, position=0
        )
        tracker.end_hand(winners=[1], winnings={0: -10, 1: 10}, final_stacks={0: 990, 1: 1010})
        
        # Hand 2: Player 1 goes to showdown (loser)
        tracker.start_hand(2, players, dealer_position=1, small_blind=1, big_blind=2)
        tracker.record_action(
            player_id=1, player_name='Player1',
            action=Action.CALL, amount=2,
            pot_size=4, stack_before=1010, stack_after=1008,
            street=Street.PREFLOP, position=1
        )
        tracker.end_hand(winners=[0], winnings={0: 4, 1: -2}, final_stacks={0: 994, 1: 1008})
        
        # Hand 3: Player 1 folds (not showdown)
        tracker.start_hand(3, players, dealer_position=0, small_blind=1, big_blind=2)
        tracker.record_action(
            player_id=1, player_name='Player1',
            action=Action.FOLD, amount=0,
            pot_size=3, stack_before=1008, stack_after=1008,
            street=Street.PREFLOP, position=0
        )
        tracker.end_hand(winners=[0], winnings={0: 3, 1: 0}, final_stacks={0: 997, 1: 1008})
        
        stats = tracker.get_all_opponent_stats()
        player1_stats = stats[1]
        
        # Showdown % = 2 / 3 = 66.67%
        expected_showdown = round(2/3, 3)
        assert player1_stats['went_to_showdown_percent'] == expected_showdown, \
            f"Expected showdown {expected_showdown}, got {player1_stats['went_to_showdown_percent']}"
    
    def test_multiple_players_independent_stats(self, tracker):
        """Test that stats for different players are independent"""
        players = [
            {'id': 0, 'name': 'Agent', 'stack': 1000},
            {'id': 1, 'name': 'Player1', 'stack': 1000},
            {'id': 2, 'name': 'Player2', 'stack': 1000},
        ]
        
        # Hand 1: Player 1 raises, Player 2 calls
        tracker.start_hand(1, players, dealer_position=0, small_blind=1, big_blind=2)
        tracker.record_action(
            player_id=1, player_name='Player1',
            action=Action.RAISE, amount=10,
            pot_size=13, stack_before=1000, stack_after=990,
            street=Street.PREFLOP, position=0
        )
        tracker.record_action(
            player_id=2, player_name='Player2',
            action=Action.CALL, amount=10,
            pot_size=23, stack_before=1000, stack_after=990,
            street=Street.PREFLOP, position=1
        )
        tracker.end_hand(winners=[1], winnings={0: -20, 1: 20, 2: 0}, final_stacks={0: 980, 1: 1020, 2: 990})
        
        # Hand 2: Player 1 folds, Player 2 raises
        tracker.start_hand(2, players, dealer_position=1, small_blind=1, big_blind=2)
        tracker.record_action(
            player_id=1, player_name='Player1',
            action=Action.FOLD, amount=0,
            pot_size=3, stack_before=1020, stack_after=1020,
            street=Street.PREFLOP, position=1
        )
        tracker.record_action(
            player_id=2, player_name='Player2',
            action=Action.RAISE, amount=10,
            pot_size=13, stack_before=990, stack_after=980,
            street=Street.PREFLOP, position=2
        )
        tracker.end_hand(winners=[2], winnings={0: 0, 1: -1, 2: 1}, final_stacks={0: 980, 1: 1019, 2: 991})
        
        stats = tracker.get_all_opponent_stats()
        player1_stats = stats[1]
        player2_stats = stats[2]
        
        # Player 1: 1 raise, 1 fold = VPIP 50%, PFR 50%
        assert player1_stats['vpip'] == 0.5
        assert player1_stats['pfr'] == 0.5
        assert player1_stats['hands_played'] == 2
        
        # Player 2: 1 call, 1 raise = VPIP 100%, PFR 50%
        assert player2_stats['vpip'] == 1.0
        assert player2_stats['pfr'] == 0.5
        assert player2_stats['hands_played'] == 2
    
    def test_all_stats_calculated_together(self, tracker):
        """Integration test: verify all stats for one player across multiple hands"""
        players = [
            {'id': 0, 'name': 'Agent', 'stack': 1000},
            {'id': 1, 'name': 'TestPlayer', 'stack': 1000},
        ]
        
        # Simulate 3 hands with known player 1 actions
        # Hand 1: Player 1 raises (VPIP, PFR yes, goes to showdown and loses)
        tracker.start_hand(1, players, dealer_position=0, small_blind=1, big_blind=2)
        tracker.record_action(
            player_id=1, player_name='TestPlayer',
            action=Action.RAISE, amount=10,
            pot_size=13, stack_before=1000, stack_after=990,
            street=Street.PREFLOP, position=0
        )
        tracker.end_hand(winners=[0], winnings={0: 10, 1: -5}, final_stacks={0: 1010, 1: 995})
        
        # Hand 2: Player 1 calls (VPIP yes, PFR no, goes to showdown and wins)
        tracker.start_hand(2, players, dealer_position=1, small_blind=1, big_blind=2)
        tracker.record_action(
            player_id=1, player_name='TestPlayer',
            action=Action.CALL, amount=2,
            pot_size=4, stack_before=995, stack_after=993,
            street=Street.PREFLOP, position=1
        )
        tracker.end_hand(winners=[1], winnings={0: -5, 1: 5}, final_stacks={0: 1005, 1: 1000})
        
        # Hand 3: Player 1 folds (VPIP no, PFR no, no showdown)
        tracker.start_hand(3, players, dealer_position=0, small_blind=1, big_blind=2)
        tracker.record_action(
            player_id=1, player_name='TestPlayer',
            action=Action.FOLD, amount=0,
            pot_size=3, stack_before=1000, stack_after=1000,
            street=Street.PREFLOP, position=0
        )
        tracker.end_hand(winners=[0], winnings={0: 3, 1: 0}, final_stacks={0: 1008, 1: 1000})
        
        stats = tracker.get_all_opponent_stats()
        player1_stats = stats[1]
        
        # Verify all stats
        assert player1_stats['hands_played'] == 3, "Should play 3 hands"
        assert player1_stats['vpip'] == round(2/3, 3), "VPIP should be 2/3 (raised and called, not folded)"
        assert player1_stats['pfr'] == round(1/3, 3), "PFR should be 1/3 (only raised once)"
        assert player1_stats['went_to_showdown_percent'] == round(2/3, 3), "Showdown 2/3 (2 hands went to showdown)"
        assert player1_stats['player_type'] is not None, "Should have player type classification"
        assert 'af' in player1_stats, "Should have aggression factor"
        assert 'confidence' in player1_stats, "Should have confidence score"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])