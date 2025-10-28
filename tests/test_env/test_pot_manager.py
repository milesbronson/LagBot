"""
Tests for pot manager
"""

import pytest
from src.poker_env.pot_manager import PotManager, Pot
from src.poker_env.player import Player


class TestPot:
    """Test cases for Pot class"""
    
    def test_pot_creation(self):
        """Test pot initialization"""
        pot = Pot()
        assert pot.amount == 0
        assert pot.eligible_players == []
    
    def test_add_chips(self):
        """Test adding chips to pot"""
        pot = Pot()
        pot.add_chips(100)
        assert pot.amount == 100
        
        pot.add_chips(50)
        assert pot.amount == 150


class TestPotManager:
    """Test cases for PotManager class"""
    
    @pytest.fixture
    def pot_manager(self):
        """Create a pot manager instance"""
        return PotManager(small_blind=5, big_blind=10)
    
    @pytest.fixture
    def players(self):
        """Create test players"""
        return [
            Player(0, 1000, "Alice"),
            Player(1, 1000, "Bob"),
            Player(2, 1000, "Charlie")
        ]
    
    def test_initialization(self, pot_manager):
        """Test pot manager initialization"""
        assert pot_manager.small_blind == 5
        assert pot_manager.big_blind == 10
        assert pot_manager.current_bet == 0
        assert pot_manager.min_raise == 10
    
    def test_start_new_hand(self, pot_manager):
        """Test starting a new hand"""
        pot_manager.start_new_hand()
        assert len(pot_manager.pots) == 1
        assert pot_manager.current_bet == 0
    
    def test_post_blinds(self, pot_manager, players):
        """Test posting blinds"""
        pot_manager.start_new_hand()
        pot_manager.post_blinds(players[0], players[1])
        
        assert players[0].stack == 995  # Paid small blind
        assert players[1].stack == 990  # Paid big blind
        assert pot_manager.pots[0].amount == 15
        assert pot_manager.current_bet == 10
    
    def test_place_bet_call(self, pot_manager, players):
        """Test calling a bet"""
        pot_manager.start_new_hand()
        pot_manager.current_bet = 10
        
        amount, action = pot_manager.place_bet(players[0], 10)
        
        assert amount == 10
        assert action == "call"
        assert players[0].stack == 990
        assert pot_manager.pots[0].amount == 10
    
    def test_place_bet_raise(self, pot_manager, players):
        """Test raising"""
        pot_manager.start_new_hand()
        pot_manager.current_bet = 10
        
        amount, action = pot_manager.place_bet(players[0], 30)
        
        assert amount == 30
        assert action == "raise"
        assert players[0].stack == 970
        assert pot_manager.current_bet == 30
    
    def test_place_bet_fold(self, pot_manager, players):
        """Test folding"""
        pot_manager.start_new_hand()
        pot_manager.current_bet = 10
        
        amount, action = pot_manager.place_bet(players[0], 0)
        
        assert amount == 0
        assert action == "fold"
        assert not players[0].is_active
    
    def test_place_bet_check(self, pot_manager, players):
        """Test checking"""
        pot_manager.start_new_hand()
        pot_manager.current_bet = 0
        
        amount, action = pot_manager.place_bet(players[0], 0)
        
        assert amount == 0
        assert action == "check"
    
    def test_place_bet_all_in(self, pot_manager, players):
        """Test going all-in"""
        pot_manager.start_new_hand()
        pot_manager.current_bet = 10
        
        # Player has only 50 chips
        players[0].stack = 50
        amount, action = pot_manager.place_bet(players[0], 50)
        
        assert amount == 50
        assert action == "all-in"
        assert players[0].is_all_in
        assert players[0].stack == 0
    
    def test_calculate_side_pots_no_all_in(self, pot_manager, players):
        """Test pot calculation with no all-ins"""
        # All players bet the same amount
        for player in players:
            player.total_bet_this_hand = 100
        
        pots = pot_manager.calculate_side_pots(players)
        
        assert len(pots) == 1
        assert pots[0].amount == 300
        assert len(pots[0].eligible_players) == 3
    
    def test_calculate_side_pots_with_all_in(self, pot_manager, players):
        """Test pot calculation with one all-in"""
        # Player 0 all-in for 50, others bet 100
        players[0].total_bet_this_hand = 50
        players[1].total_bet_this_hand = 100
        players[2].total_bet_this_hand = 100
        
        pots = pot_manager.calculate_side_pots(players)
        
        # Should create 2 pots: main pot (150) and side pot (100)
        assert len(pots) == 2
        assert pots[0].amount == 150  # 50 from each player
        assert len(pots[0].eligible_players) == 3
        assert pots[1].amount == 100  # 50 from players 1 and 2
        assert len(pots[1].eligible_players) == 2
    
    def test_calculate_side_pots_multiple_all_ins(self, pot_manager, players):
        """Test pot calculation with multiple all-ins at different levels"""
        players[0].total_bet_this_hand = 30
        players[1].total_bet_this_hand = 60
        players[2].total_bet_this_hand = 100
        
        pots = pot_manager.calculate_side_pots(players)
        
        # Should create 3 pots
        assert len(pots) == 3
        assert pots[0].amount == 90   # 30 from each
        assert pots[1].amount == 60   # 30 from players 1 and 2
        assert pots[2].amount == 40   # 40 from player 2
    
    def test_distribute_pots_single_winner(self, pot_manager, players):
        """Test distributing pot to single winner"""
        for player in players:
            player.total_bet_this_hand = 100
        
        # Player 0 has best hand (rank 100), others have rank 200
        hand_ranks = {0: 100, 1: 200, 2: 200}
        
        winnings = pot_manager.distribute_pots(players, hand_ranks)
        
        assert winnings[0] == 300
        assert winnings[1] == 0
        assert winnings[2] == 0
    
    def test_distribute_pots_split(self, pot_manager, players):
        """Test splitting pot between multiple winners"""
        for player in players:
            player.total_bet_this_hand = 100
        
        # Players 0 and 1 tie with best hand
        hand_ranks = {0: 100, 1: 100, 2: 200}
        
        winnings = pot_manager.distribute_pots(players, hand_ranks)
        
        assert winnings[0] == 150
        assert winnings[1] == 150
        assert winnings[2] == 0
    
    def test_distribute_pots_with_rake(self):
        """Test rake application"""
        pot_manager = PotManager(
            small_blind=5,
            big_blind=10,
            rake_percent=0.05,  # 5% rake
            rake_cap=10
        )
        
        players = [
            Player(0, 1000, "Alice"),
            Player(1, 1000, "Bob"),
            Player(2, 1000, "Charlie")
        ]
        
        # Each player bets 100 (total pot 300)
        for player in players:
            player.total_bet_this_hand = 100
        
        hand_ranks = {0: 100, 1: 200, 2: 200}
        
        winnings = pot_manager.distribute_pots(players, hand_ranks)
        
        # 5% rake on 300 = 15, but capped at 10
        # Winner gets 300 - 10 = 290
        assert winnings[0] == 290


if __name__ == "__main__":
    pytest.main([__file__, "-v"])