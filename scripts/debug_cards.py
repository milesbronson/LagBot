"""Debug card encoding"""

from src.poker_env.texas_holdem_env import TexasHoldemEnv
from treys import Card

env = TexasHoldemEnv(num_players=3, track_opponents=False)
obs, _ = env.reset()

player = env.game_state.get_current_player()
print("Player's hole cards:")
for i, card in enumerate(player.hand):
    print(f"  Card {i}: {card} (0x{card:x})")
    print(f"    Card string: {Card.int_to_str(card)}")
    print(f"    Rank >> 8: {(card >> 8) & 0xFF}")
    print(f"    Suit >> 12: {(card >> 12) & 0xF}")
    print()

# Test encoding
encoded = env._encode_cards(player.hand)
print(f"Encoded: {encoded}")
