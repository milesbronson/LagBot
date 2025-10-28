"""
Hand evaluation using the treys library
"""

from treys import Card, Evaluator
from typing import List, Tuple


class HandEvaluator:
    """
    Wrapper around treys library for hand evaluation
    """
    
    def __init__(self):
        self.evaluator = Evaluator()
        
    def evaluate_hand(self, hole_cards: List[int], community_cards: List[int]) -> int:
        """
        Evaluate a poker hand
        
        Args:
            hole_cards: Player's hole cards (2 cards)
            community_cards: Community cards (0-5 cards)
            
        Returns:
            Hand rank (lower is better, 1 is Royal Flush)
        """
        if len(community_cards) < 3:
            # Pre-flop or not enough cards
            return 7462  # Worst possible hand
            
        return self.evaluator.evaluate(community_cards, hole_cards)
    
    def get_rank_class(self, hand_rank: int) -> int:
        """
        Get the hand class (1-9)
        1 = Straight Flush, 2 = Four of a Kind, ..., 9 = High Card
        
        Args:
            hand_rank: Hand rank from evaluate_hand
            
        Returns:
            Hand class (1-9)
        """
        return self.evaluator.get_rank_class(hand_rank)
    
    def class_to_string(self, hand_class: int) -> str:
        """
        Convert hand class to readable string
        
        Args:
            hand_class: Hand class (1-9)
            
        Returns:
            Human-readable hand name
        """
        return self.evaluator.class_to_string(hand_class)
    
    def compare_hands(self, hand1_rank: int, hand2_rank: int) -> int:
        """
        Compare two hands
        
        Args:
            hand1_rank: Rank of first hand
            hand2_rank: Rank of second hand
            
        Returns:
            -1 if hand1 wins, 1 if hand2 wins, 0 if tie
        """
        if hand1_rank < hand2_rank:
            return -1
        elif hand1_rank > hand2_rank:
            return 1
        else:
            return 0
            
    @staticmethod
    def card_to_string(card: int) -> str:
        """Convert card integer to string representation"""
        return Card.int_to_str(card)
    
    @staticmethod
    def string_to_card(card_str: str) -> int:
        """Convert card string to integer representation"""
        return Card.new(card_str)
    
    @staticmethod
    def create_deck() -> List[int]:
        """Create a full deck of 52 cards"""
        return Card.get_full_deck()
    
    @staticmethod
    def print_hand(hole_cards: List[int], community_cards: List[int]):
        """Print cards in readable format"""
        print("Hole cards:", [Card.int_to_str(c) for c in hole_cards])
        print("Community cards:", [Card.int_to_str(c) for c in community_cards])


# Example usage and testing
if __name__ == "__main__":
    evaluator = HandEvaluator()
    
    # Create some sample hands
    board = [
        Card.new('Ah'),
        Card.new('Kd'),
        Card.new('Jc'),
        Card.new('Qh'),
        Card.new('Ts')
    ]
    
    hand1 = [Card.new('9s'), Card.new('8s')]  # Straight
    hand2 = [Card.new('As'), Card.new('Ac')]  # Three of a kind
    
    rank1 = evaluator.evaluate_hand(hand1, board)
    rank2 = evaluator.evaluate_hand(hand2, board)
    
    print(f"Hand 1 rank: {rank1} - {evaluator.class_to_string(evaluator.get_rank_class(rank1))}")
    print(f"Hand 2 rank: {rank2} - {evaluator.class_to_string(evaluator.get_rank_class(rank2))}")
    
    winner = evaluator.compare_hands(rank1, rank2)
    if winner == -1:
        print("Hand 1 wins!")
    elif winner == 1:
        print("Hand 2 wins!")
    else:
        print("Tie!")