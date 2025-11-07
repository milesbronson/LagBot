"""
Comprehensive tests for hand completion logic
Tests various scenarios where hands should complete vs. continue
Helps debug the 17,800 steps issue
"""

import pytest
from src.poker_env.game_state import GameState, BettingRound
from src.poker_env.player import Player


class TestHandCompletionBasics:
    """Test basic hand completion scenarios"""
    
    @pytest.fixture
    def game(self):
        """Create a game for testing"""
        return GameState(
            num_players=3,
            starting_stack=1000,
            small_blind=5,
            big_blind=10
        )
    
    def test_hand_complete_all_but_one_fold_preflop(self, game):
        """Hand should complete when all but one player folds pre-flop"""
        game.start_new_hand()
        
        # Player 0 folds
        game.execute_action(0)  # Fold
        assert not game.is_hand_complete(), "Hand shouldn't complete after 1 fold"
        
        # Player 1 calls
        game.execute_action(1)  # Call
        assert not game.is_hand_complete(), "Hand shouldn't complete with 2 players left"
        
        # Player 2 folds
        game.execute_action(0)  # Fold
        assert game.is_hand_complete(), "Hand SHOULD complete when all but one fold!"
    
    def test_hand_complete_all_but_one_fold_flop(self, game):
        """Hand should complete when all but one fold post-flop"""
        game.start_new_hand()
        
        # Pre-flop: everyone calls
        game.execute_action(1)  # Player 0 calls
        game.execute_action(1)  # Player 1 calls
        game.execute_action(1)  # Player 2 calls
        
        # Should advance to flop
        game.advance_betting_round()
        assert game.betting_round == BettingRound.FLOP
        
        # On flop: first player folds
        game.execute_action(0)  # Fold
        assert not game.is_hand_complete()
        
        # Second player folds
        game.execute_action(0)  # Fold
        assert game.is_hand_complete(), "Hand should complete when only 1 player left on flop"
    
    def test_hand_complete_showdown(self, game):
        """Hand should complete at showdown"""
        game.start_new_hand()
        
        # Play through all streets without anyone folding
        # Pre-flop
        game.execute_action(1)  # Call
        game.execute_action(1)  # Call
        game.execute_action(1)  # Call
        
        game.advance_betting_round()
        assert game.betting_round == BettingRound.FLOP
        
        # Flop
        game.execute_action(1)  # Check
        game.execute_action(1)  # Check
        game.execute_action(1)  # Check
        
        game.advance_betting_round()
        assert game.betting_round == BettingRound.TURN
        
        # Turn
        game.execute_action(1)  # Check
        game.execute_action(1)  # Check
        game.execute_action(1)  # Check
        
        game.advance_betting_round()
        assert game.betting_round == BettingRound.RIVER
        
        # River
        game.execute_action(1)  # Check
        game.execute_action(1)  # Check
        game.execute_action(1)  # Check
        
        game.advance_betting_round()
        assert game.betting_round == BettingRound.SHOWDOWN
        assert game.is_hand_complete(), "Hand should be complete at showdown"
    
    def test_hand_not_complete_mid_preflop(self, game):
        """Hand should NOT complete mid pre-flop"""
        game.start_new_hand()
        
        assert not game.is_hand_complete(), "Hand should not be complete right after start"
        
        game.execute_action(1)  # First action
        assert not game.is_hand_complete(), "Hand should not be complete after 1 action"
    
    def test_hand_not_complete_mid_betting_round(self, game):
        """Hand should NOT complete while betting round is ongoing"""
        game.start_new_hand()
        
        # Play through first few actions
        game.execute_action(1)  # Call
        game.execute_action(1)  # Call
        
        # Player 2 hasn't acted yet, but game is checking completion
        assert not game.is_hand_complete()


class TestBettingRoundCompletion:
    """Test when betting rounds should complete"""
    
    @pytest.fixture
    def game(self):
        return GameState(num_players=3, starting_stack=1000, small_blind=5, big_blind=10)
    
    def test_betting_round_complete_everyone_checked(self, game):
        """Betting round completes when everyone checks"""
        game.start_new_hand()
        game.advance_betting_round()  # Skip to flop (where checks are common)
        
        # Flop: everyone checks
        game.execute_action(1)  # Check
        assert not game.is_betting_round_complete()
        
        game.execute_action(1)  # Check
        assert not game.is_betting_round_complete()
        
        game.execute_action(1)  # Check
        assert game.is_betting_round_complete(), "Betting round should complete when all check"
    
    def test_betting_round_complete_called_down(self, game):
        """Betting round completes when all match current bet"""
        game.start_new_hand()
        
        # Pre-flop: player 0 raises
        game.execute_action(2, raise_amount=20)  # Raise 20
        current_bet = game.pot_manager.current_bet
        
        # Player 1 calls
        game.execute_action(1)  # Call
        
        # Player 2 must also match the bet
        assert not game.is_betting_round_complete()
        
        game.execute_action(1)  # Call
        assert game.is_betting_round_complete(), "Round should complete when all call same bet"
    
    def test_betting_round_not_complete_pending_calls(self, game):
        """Betting round NOT complete if someone hasn't called"""
        game.start_new_hand()
        
        game.execute_action(1)  # Call big blind
        game.execute_action(2, raise_amount=50)  # Raise
        
        assert not game.is_betting_round_complete(), "Round not complete - waiting for calls"
        
        game.execute_action(1)  # First player must act on raise
        assert not game.is_betting_round_complete()
    
    def test_betting_round_complete_all_but_one_folded(self, game):
        """Betting round complete when only one player left"""
        game.start_new_hand()
        
        game.execute_action(0)  # Fold
        assert not game.is_betting_round_complete()
        
        game.execute_action(0)  # Fold
        assert game.is_betting_round_complete(), "Round complete - only 1 player left"


class TestAllInScenarios:
    """Test hand completion with all-in situations"""
    
    @pytest.fixture
    def game(self):
        return GameState(num_players=3, starting_stack=100, small_blind=5, big_blind=10)
    
    def test_hand_not_complete_one_player_all_in(self, game):
        """Hand should NOT complete just because one player is all-in"""
        game.start_new_hand()
        
        # Get player 0 all-in
        game.players[0].stack = 5
        game.execute_action(0)  # Would be small blind
        game.players[0].bet(5)
        game.players[0].is_all_in = True
        
        # Other players still need to act
        game.execute_action(1)  # Player 1 acts
        
        # Hand shouldn't be complete just from all-in
        assert not game.is_hand_complete(), "Hand not complete with all-in + other active players"
    
    def test_hand_complete_all_in_no_more_raises(self, game):
        """Hand should complete when all-in and no more raises possible"""
        game.start_new_hand()
        
        # Make player 0 all-in
        game.players[0].stack = 5
        game.players[0].bet(5)
        game.players[0].is_all_in = True
        
        # Player 1 calls
        game.players[1].current_bet = 5
        
        # Player 2 calls
        game.players[2].current_bet = 5
        
        # Now only all-in player left who can't raise
        # In this state, hand MIGHT complete depending on logic
        # (This depends on your implementation)
    
    def test_multiple_all_ins_hand_continues_to_showdown(self, game):
        """Hand should continue to showdown with multiple all-ins"""
        game.start_new_hand()
        
        # Both active players go all-in
        game.players[0].is_all_in = True
        game.players[1].is_all_in = True
        game.players[2].stack = 1000  # Fold
        game.players[2].is_active = False
        
        # Hand should continue through remaining streets to showdown
        assert not game.is_hand_complete()
        
        # Advance to showdown
        game.betting_round = BettingRound.SHOWDOWN
        assert game.is_hand_complete()


class TestFoldLogic:
    """Test fold-related completion scenarios"""
    
    @pytest.fixture
    def game(self):
        return GameState(num_players=4, starting_stack=1000, small_blind=5, big_blind=10)
    
    def test_only_one_active_player_remaining(self, game):
        """Hand completes when only one player is active"""
        game.start_new_hand()
        
        # Make 3 players fold
        for _ in range(3):
            game.execute_action(0)  # Fold
            if game.is_hand_complete():
                break
        
        assert game.is_hand_complete(), "Should complete when only 1 active player"
    
    def test_inactive_players_not_counted(self, game):
        """Inactive players should not be counted as active"""
        game.start_new_hand()
        
        # Player 0 folds
        game.execute_action(0)
        game.players[0].is_active = False
        
        # Player 1 folds  
        game.execute_action(0)
        game.players[1].is_active = False
        
        game.execute_action(0)

        game.display_hand_history()
        # Now only 2 active, should complete
        assert game.is_hand_complete()
    
    def test_fold_count_vs_active_count(self, game):
        """Verify fold count matches expected behavior"""
        game.start_new_hand()
        
        initial_active = len(game.get_active_players())
        assert initial_active == 4
        
        # One fold
        game.execute_action(0)
        assert len(game.get_active_players()) == 3
        assert not game.is_hand_complete()
        
        # Two folds
        game.execute_action(0)
        assert len(game.get_active_players()) == 2
        assert not game.is_hand_complete()
        
        # Three folds
        game.execute_action(0)
        assert len(game.get_active_players()) == 1
        assert game.is_hand_complete(), "Hand should complete when only 1 active left"


class TestBettingRoundAdvancement:
    """Test when hands advance to next betting round without completing"""
    
    @pytest.fixture
    def game(self):
        return GameState(num_players=3, starting_stack=1000, small_blind=5, big_blind=10)
    
    def test_betting_round_advances_preflop_to_flop(self, game):
        """Betting round should advance from pre-flop to flop"""
        game.start_new_hand()
        initial_round = game.betting_round
        
        # Everyone calls
        game.execute_action(1)
        game.execute_action(1)
        game.execute_action(1)
        
        # Should be able to advance
        game.advance_betting_round()
        
        assert game.betting_round == BettingRound.FLOP
        assert game.betting_round != initial_round
    
    def test_community_cards_appear_on_flop(self, game):
        """Community cards should appear when advancing to flop"""
        game.start_new_hand()
        
        # Everyone calls pre-flop
        for _ in range(3):
            game.execute_action(1)
        
        assert len(game.community_cards) == 0, "No community cards pre-flop"
        
        game.advance_betting_round()
        
        assert len(game.community_cards) == 3, "Should have 3 cards on flop"
    
    def test_each_betting_round_proper_progression(self, game):
        """Test proper progression through all rounds"""
        game.start_new_hand()
        
        # Pre-flop
        assert game.betting_round == BettingRound.PREFLOP
        for _ in range(3):
            game.execute_action(1)
        game.advance_betting_round()
        
        # Flop
        assert game.betting_round == BettingRound.FLOP
        for _ in range(3):
            game.execute_action(1)
        game.advance_betting_round()
        
        # Turn
        assert game.betting_round == BettingRound.TURN
        assert len(game.community_cards) == 4
        for _ in range(3):
            game.execute_action(1)
        game.advance_betting_round()
        
        # River
        assert game.betting_round == BettingRound.RIVER
        assert len(game.community_cards) == 5
        for _ in range(3):
            game.execute_action(1)
        game.advance_betting_round()
        
        # Showdown
        assert game.betting_round == BettingRound.SHOWDOWN
        assert game.is_hand_complete()


class TestEdgeCases:
    """Edge cases and unusual scenarios"""
    
    @pytest.fixture
    def game(self):
        return GameState(num_players=2, starting_stack=1000, small_blind=5, big_blind=10)
    
    def test_heads_up_two_players(self, game):
        """Test hand completion in heads-up (2 player) scenario"""
        game.start_new_hand()
        
        # One player folds
        game.execute_action(0)  # Fold
        
        # Should complete immediately
        assert game.is_hand_complete(), "Heads-up hand should complete when one folds"
    
    def test_immediate_fold_preflop(self, game):
        """First action is fold"""
        game.start_new_hand()
        
        game.execute_action(0)  # First action is fold
        
        assert game.is_hand_complete(), "Should complete if all but 1 fold immediately"
    
    @pytest.fixture
    def game_many_players(self):
        return GameState(num_players=10, starting_stack=1000, small_blind=5, big_blind=10)
    
    def test_many_players_scenario(self, game_many_players):
        """Test with maximum players (10)"""
        game = game_many_players
        game.start_new_hand()
        
        # Fold 9 players
        for i in range(9):
            game.execute_action(0)
            if i < 8:  # Don't check on last iteration yet
                assert not game.is_hand_complete(), f"Should not complete after {i+1} folds"
        
        # After 9 folds, should complete
        assert game.is_hand_complete(), "Should complete when only 1 player left"
    
    def test_hand_complete_called_multiple_times(self, game):
        """Calling is_hand_complete() multiple times should be consistent"""
        game.start_new_hand()
        game.execute_action(0)  # Someone folds
        
        result1 = game.is_hand_complete()
        result2 = game.is_hand_complete()
        result3 = game.is_hand_complete()
        
        assert result1 == result2 == result3, "is_hand_complete() should be consistent"


class TestStepCounter:
    """Tests to verify hand completion doesn't cause infinite loops"""
    
    @pytest.fixture
    def game(self):
        return GameState(num_players=3, starting_stack=1000, small_blind=5, big_blind=10)
    
    def test_hand_completes_within_reasonable_steps(self, game):
        """A complete hand should finish in reasonable number of steps"""
        game.start_new_hand()
        
        steps = 0
        max_steps = 1000  # Way more than needed for a real hand
        
        while not game.is_hand_complete() and steps < max_steps:
            # Random action
            action = game.current_player_idx % 2  # Simple pattern
            game.execute_action(action)
            
            # Advance betting round if needed
            if game.is_betting_round_complete() and not game.is_hand_complete():
                game.advance_betting_round()
            
            steps += 1
        
        assert game.is_hand_complete(), f"Hand should complete in {max_steps} steps"
        assert steps < max_steps, f"Hand took too many steps: {steps}"
    
    def test_action_loop_completes_hand(self, game):
        """Playing through all natural actions should complete hand"""
        game.start_new_hand()
        
        # Simulate natural play: everyone calls pre-flop, checks down
        # Pre-flop: 3 players, 3 actions
        game.execute_action(1)
        game.execute_action(1)
        game.execute_action(1)
        
        assert game.is_betting_round_complete()
        game.advance_betting_round()
        
        # Flop: 3 checks
        game.execute_action(1)
        game.execute_action(1)
        game.execute_action(1)
        
        assert game.is_betting_round_complete()
        game.advance_betting_round()
        
        # Turn: 3 checks
        game.execute_action(1)
        game.execute_action(1)
        game.execute_action(1)
        
        assert game.is_betting_round_complete()
        game.advance_betting_round()
        
        # River: 3 checks
        game.execute_action(1)
        game.execute_action(1)
        game.execute_action(1)
        
        assert game.is_betting_round_complete()
        game.advance_betting_round()
        
        # Should be at showdown now
        assert game.betting_round == BettingRound.SHOWDOWN
        assert game.is_hand_complete()


class TestDebugInfo:
    """Tests to help debug when hands don't complete"""
    
    @pytest.fixture
    def game(self):
        return GameState(num_players=3, starting_stack=1000, small_blind=5, big_blind=10)
    
    def test_debug_output_hand_state(self, game):
        """Print debug info about hand state"""
        game.start_new_hand()
        
        # Helper to print state
        def print_state(label):
            active = len(game.get_active_players())
            can_act = [p for p in game.get_active_players() if not p.is_all_in]
            print(f"\n{label}:")
            print(f"  Active players: {active}")
            print(f"  Players who can act: {len(can_act)}")
            print(f"  Betting round: {game.betting_round.name}")
            print(f"  Is betting round complete? {game.is_betting_round_complete()}")
            print(f"  Is hand complete? {game.is_hand_complete()}")
        
        print_state("Initial state")
        
        game.execute_action(1)
        print_state("After 1st action")
        
        game.execute_action(0)  # Fold
        print_state("After fold")
        
        game.execute_action(1)
        print_state("After call")
    
    def test_verify_active_player_count(self, game):
        """Verify active player count changes correctly"""
        game.start_new_hand()
        
        initial_count = len(game.get_active_players())
        assert initial_count == 3
        
        # Fold one
        game.execute_action(0)
        count_after_fold = len(game.get_active_players())
        
        # Should be one less
        assert count_after_fold == initial_count - 1 or game.is_hand_complete()
    
    def test_current_player_tracking(self, game):
        """Verify current player index updates correctly"""
        game.start_new_hand()
        
        initial_player = game.current_player_idx
        
        game.execute_action(1)  # Current player acts
        
        # Should move to next player (or stay if hand complete)
        if not game.is_hand_complete():
            assert game.current_player_idx != initial_player or game.num_players == 1


if __name__ == "__main__":
    # Run with: pytest test_hand_completion.py -v
    pytest.main([__file__, "-v", "-s"])  # -s to show print statements