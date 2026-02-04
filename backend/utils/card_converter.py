"""
Card conversion utilities for frontend compatibility.
Uses existing HandEvaluator for Treys int <-> string conversion.
"""
from typing import List
from src.poker_env.hand_evaluator import HandEvaluator


def convert_cards_for_frontend(card_ints: List[int]) -> List[str]:
    """
    Convert list of Treys card integers to string representation.

    Args:
        card_ints: List of card integers (e.g., [268440965, 134236965])

    Returns:
        List of card strings (e.g., ["Ah", "Kd"])
    """
    return [HandEvaluator.card_to_string(c) for c in card_ints]


def convert_card_for_frontend(card_int: int) -> str:
    """
    Convert single Treys card integer to string representation.

    Args:
        card_int: Card integer

    Returns:
        Card string (e.g., "Ah")
    """
    return HandEvaluator.card_to_string(card_int)


def convert_card_from_frontend(card_str: str) -> int:
    """
    Convert card string to Treys integer representation.

    Args:
        card_str: Card string (e.g., "Ah")

    Returns:
        Card integer
    """
    return HandEvaluator.string_to_card(card_str)
