"""Debug observation structure"""

from src.poker_env.texas_holdem_env import TexasHoldemEnv
import numpy as np

env = TexasHoldemEnv(num_players=3, track_opponents=False)
obs, _ = env.reset()

print(f"Total observation shape: {obs.shape}")
print()

# Break down observation
idx = 0

# Hole cards (2 cards × 6 dims = 12)
hole_end = idx + 12
hole_cards = obs[idx:hole_end]
print(f"Hole cards [{idx}:{hole_end}]: {hole_cards}")
idx = hole_end

# Community cards (5 cards × 6 dims = 30)
comm_end = idx + 30
comm_cards = obs[idx:comm_end]
print(f"Community cards [{idx}:{comm_end}]: {comm_cards[:18]}... (showing first 3 cards)")
idx = comm_end

# Hand features (3 dims)
hand_feat_end = idx + 3
hand_features = obs[idx:hand_feat_end]
print(f"\nHand features [{idx}:{hand_feat_end}]:")
print(f"  hand_strength = {hand_features[0]}")
print(f"  pot_odds = {hand_features[1]}")
print(f"  spr = {hand_features[2]}")
idx = hand_feat_end

# Game state (8 dims)
game_state = obs[idx:]
print(f"\nGame state [{idx}:end]:")
print(f"  stack = {game_state[0]}")
print(f"  pot = {game_state[1]}")
print(f"  bet = {game_state[2]}")
print(f"  call = {game_state[3]}")
print(f"  active = {game_state[4]}")
print(f"  pos = {game_state[5]}")
print(f"  rnd = {game_state[6]}")
print(f"  btn = {game_state[7]}")

# Direct calculation test
player = env.game_state.get_current_player()
hand_strength = env._calculate_hand_strength(player.hand, env.game_state.community_cards)
pot_odds = env._calculate_pot_odds(player)
spr = env._calculate_spr(player)

print(f"\nDirect calculations:")
print(f"  hand_strength = {hand_strength}")
print(f"  pot_odds = {pot_odds}")
print(f"  spr = {spr}")
