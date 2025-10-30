"""
Tests for hand evaluator
"""

import pytest
from treys import Card
from src.poker_env.hand_evaluator import HandEvaluator


class TestHandEvaluator:
    """Test cases for HandEvaluator class"""
    
    @pytest.fixture
    def evaluator(self):
        """Create a hand evaluator instance"""
        return HandEvaluator()
    
    # def test_royal_flush(self, evaluator):
    #     """Test royal flush detection"""
    #     board = [
    #         Card.new('Ah'),
    #         Card.new('Kh'),
    #         Card.new('Qh'),
    #         Card.new('Jh'),
    #         Card.new('Th')
    #     ]
    #     hand = [Card.new('9h'), Card.new('8h')]
        
    #     rank = evaluator.evaluate_hand(hand, board)
    #     hand_class = evaluator.get_rank_class(rank)
        
    #     # Royal flush is class 1 (Straight Flush)
    #     assert hand_class == 1
    
    def test_four_of_a_kind(self, evaluator):
        """Test four of a kind detection"""
        board = [
            Card.new('Ah'),
            Card.new('Ad'),
            Card.new('Ac'),
            Card.new('Kh'),
            Card.new('Qh')
        ]
        hand = [Card.new('As'), Card.new('2h')]
        
        rank = evaluator.evaluate_hand(hand, board)
        hand_class = evaluator.get_rank_class(rank)
        
        # Four of a kind is class 2
        assert hand_class == 2
    
    def test_full_house(self, evaluator):
        """Test full house detection"""
        board = [
            Card.new('Ah'),
            Card.new('Ad'),
            Card.new('Kh'),
            Card.new('Kd'),
            Card.new('Qh')
        ]
        hand = [Card.new('Ac'), Card.new('2h')]
        
        rank = evaluator.evaluate_hand(hand, board)
        hand_class = evaluator.get_rank_class(rank)
        
        # Full house is class 3
        assert hand_class == 3
    
    def test_straight(self, evaluator):
        """Test straight detection"""
        board = [
            Card.new('9h'),
            Card.new('8d'),
            Card.new('7c'),
            Card.new('6h'),
            Card.new('2s')
        ]
        hand = [Card.new('5s'), Card.new('4h')]
        
        rank = evaluator.evaluate_hand(hand, board)
        hand_class = evaluator.get_rank_class(rank)
        
        # Straight is class 5
        assert hand_class == 5
    
    def test_compare_hands(self, evaluator):
        """Test comparing two hands"""
        board = [
            Card.new('Ah'),
            Card.new('Kd'),
            Card.new('Qc'),
            Card.new('Jh'),
            Card.new('2s')
        ]
        
        # Straight
        hand1 = [Card.new('Ts'), Card.new('9h')]
        # Pair of aces
        hand2 = [Card.new('As'), Card.new('3h')]
        
        rank1 = evaluator.evaluate_hand(hand1, board)
        rank2 = evaluator.evaluate_hand(hand2, board)
        
        result = evaluator.compare_hands(rank1, rank2)
        
        # Hand 1 (straight) should win
        assert result == -1
    
    def test_tie(self, evaluator):
        """Test when two hands tie"""
        board = [
            Card.new('Ah'),
            Card.new('Kd'),
            Card.new('Qc'),
            Card.new('Jh'),
            Card.new('Ts')
        ]
        
        # Both hands make the same straight on the board
        hand1 = [Card.new('9s'), Card.new('8h')]
        hand2 = [Card.new('9c'), Card.new('8d')]
        
        rank1 = evaluator.evaluate_hand(hand1, board)
        rank2 = evaluator.evaluate_hand(hand2, board)
        
        result = evaluator.compare_hands(rank1, rank2)
        
        # Should be a tie
        assert result == 0
    
    def test_preflop_evaluation(self, evaluator):
        """Test evaluation with no community cards"""
        hand = [Card.new('As'), Card.new('Ah')]
        board = []
        
        rank = evaluator.evaluate_hand(hand, board)
        
        # Should return worst possible rank when < 3 community cards
        assert rank == 7462
    
    def test_card_conversion(self, evaluator):
        """Test card string conversion"""
        card_str = 'Ah'
        card_int = HandEvaluator.string_to_card(card_str)
        converted_back = HandEvaluator.card_to_string(card_int)
        
        assert converted_back == card_str
    
    def test_create_deck(self, evaluator):
        """Test deck creation"""
        deck = HandEvaluator.create_deck()
        
        # Standard deck has 52 cards
        assert len(deck) == 52
        print(len(deck))
        # All cards should be unique
        assert len(set(deck)) == 52



if __name__ == "__main__":
    pytest.main([__file__, "-v"])