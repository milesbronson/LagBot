from src.poker_env.game_state import GameState

# Create game
game = GameState(num_players=3, starting_stack=1000, 
                 small_blind=5, big_blind=10)

# Start hand
game.start_new_hand()
print(f"Players: {len(game.players)}")
print(f"Pot: ${game.pot_manager.get_pot_total()}")

# Simulate actions
game.execute_action(1)  # Call
game.execute_action(1)  # Call
game.execute_action(1)  # Call

print(f"Betting round complete: {game.is_betting_round_complete()}")