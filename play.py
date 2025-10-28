"""
Play Texas Hold'em against a trained bot via command line
"""

import argparse
import os

from src.poker_env.texas_holdem_env import TexasHoldemEnv
from src.agents.ppo_agent import PPOAgent
from src.agents.human_agent import HumanAgent
from src.agents.random_agent import RandomAgent, CallAgent


def play_game(model_path: str = None, num_opponents: int = 1, opponent_type: str = "random"):
    """
    Play poker against bots
    
    Args:
        model_path: Path to trained model (if None, plays against random agents)
        num_opponents: Number of bot opponents (1-9)
        opponent_type: Type of opponent ("random", "call", "trained")
    """
    num_players = num_opponents + 1  # Human + opponents
    
    if not 2 <= num_players <= 10:
        print(f"Error: Number of players must be between 2 and 10 (you requested {num_players})")
        return
    
    print("\n" + "="*60)
    print("Texas Hold'em Poker")
    print("="*60)
    print(f"Players: You vs {num_opponents} bot(s)")
    print()
    
    # Create environment
    env = TexasHoldemEnv(
        num_players=num_players,
        starting_stack=1000,
        small_blind=5,
        big_blind=10
    )
    
    # Set the human as agent 0
    env.current_agent = 0
    
    # Create agents
    agents = [HumanAgent(name="You")]
    
    if model_path and os.path.exists(model_path):
        print(f"Loading trained model from: {model_path}")
        for i in range(num_opponents):
            agent = PPOAgent.load_agent(model_path, env, name=f"Bot_{i+1}")
            agents.append(agent)
    else:
        if model_path:
            print(f"Warning: Model not found at {model_path}, using {opponent_type} agents")
        
        for i in range(num_opponents):
            if opponent_type == "call":
                agent = CallAgent(name=f"CallBot_{i+1}")
            else:
                agent = RandomAgent(name=f"RandomBot_{i+1}")
            agents.append(agent)
    
    print()
    print("Commands:")
    print("  0 - Fold")
    print("  1 - Check/Call")
    print("  2 - Raise (minimum)")
    print()
    print("Press Ctrl+C to quit anytime")
    print("="*60)
    
    # Play hands
    hand_number = 0
    
    try:
        while True:
            hand_number += 1
            
            # Check if human still has chips
            if env.game_state.players[0].stack <= 0:
                print("\n" + "="*60)
                print("You're out of chips! Game over.")
                print("="*60)
                break
            
            # Reset environment
            obs = env.reset()
            
            print(f"\n{'='*60}")
            print(f"HAND #{hand_number}")
            print(f"{'='*60}\n")
            
            # Render initial state
            env.render()
            
            done = False
            step_count = 0
            
            while not done:
                current_player_idx = env.game_state.current_player_idx
                current_agent = agents[current_player_idx]
                
                # Get action from appropriate agent
                if isinstance(current_agent, HumanAgent):
                    action = current_agent.select_action(obs)
                else:
                    # Bot's turn
                    print(f"\n{current_agent.name}'s turn...")
                    action = current_agent.select_action(obs)
                    print(f"{current_agent.name} chooses: {['Fold', 'Check/Call', 'Raise'][action]}")
                
                # Execute action
                obs, reward, done, info = env.step(action)
                
                # Render state after action
                if not done:
                    env.render()
                
                step_count += 1
                
                # Safety check
                if step_count > 100:
                    print("Warning: Hand taking too long, forcing completion")
                    break
            
            # Show final results
            print("\n" + "="*60)
            print("HAND COMPLETE")
            print("="*60)
            env.render()
            
            if 'winnings' in info:
                print("\nResults:")
                for player_id, amount in info['winnings'].items():
                    if amount > 0:
                        player = env.game_state.players[player_id]
                        print(f"  {player.name} wins ${amount}!")
            
            # Show stacks
            print("\nChip Stacks:")
            for player in env.game_state.players:
                print(f"  {player.name}: ${player.stack}")
            
            # Ask if user wants to continue
            if hand_number % 5 == 0:  # Ask every 5 hands
                print()
                continue_game = input("Continue playing? (y/n): ").strip().lower()
                if continue_game != 'y':
                    break
    
    except KeyboardInterrupt:
        print("\n\nGame interrupted by user.")
    
    # Final statistics
    print("\n" + "="*60)
    print("GAME SUMMARY")
    print("="*60)
    print(f"Hands played: {hand_number}")
    print("\nFinal chip stacks:")
    for player in env.game_state.players:
        profit = player.stack - env.starting_stack
        print(f"  {player.name}: ${player.stack} ({profit:+d})")
    print("="*60)
    print("\nThanks for playing!")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Play poker against a bot")
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Path to trained model'
    )
    parser.add_argument(
        '--opponents',
        type=int,
        default=1,
        help='Number of opponents (1-9)'
    )
    parser.add_argument(
        '--opponent-type',
        type=str,
        choices=['random', 'call', 'trained'],
        default='random',
        help='Type of opponent (if not using trained model)'
    )
    
    args = parser.parse_args()
    
    play_game(args.model, args.opponents, args.opponent_type)


if __name__ == "__main__":
    main()