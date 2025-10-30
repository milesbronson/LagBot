"""
Play Texas Hold'em against bots with custom bet amounts
"""

import argparse
import os

from src.poker_env.texas_holdem_env import TexasHoldemEnv
from src.agents.ppo_agent import PPOAgent
from src.agents.human_agent import HumanAgent
from src.agents.random_agent import RandomAgent, CallAgent


class InteractiveHumanAgent(HumanAgent):
    """Human agent with custom bet amount support"""
    
    def __init__(self, env, name: str = "Human"):
        super().__init__(name)
        self.env = env
    
    def select_action_with_amount(self, observation):
        """
        Get action and bet amount from human
        
        Returns:
            (action, bet_amount) tuple where bet_amount is total contribution
        """
        current_player = self.env.game_state.get_current_player()
        pot_manager = self.env.game_state.pot_manager
        to_call = pot_manager.current_bet - current_player.current_bet
        min_raise, max_raise = self.env.get_valid_raise_range()
        
        print("\n" + "="*60)
        print("YOUR TURN!")
        print("="*60)
        print(f"Your stack: ${current_player.stack}")
        print(f"Your current bet this round: ${current_player.current_bet}")
        print(f"Pot total: ${pot_manager.get_pot_total()}")
        print(f"Min raise amount: ${pot_manager.min_raise}")
        print(f"Amount to call: ${to_call}")
        
        if max_raise > 0:
            total_bet_to_raise = to_call + min_raise
            total_bet_to_max = to_call + max_raise
            print(f"Min total bet (min raise): ${total_bet_to_raise}")
            print(f"Max total bet (all-in): ${total_bet_to_max}")
        
        print("\nActions:")
        print("  0 - Fold")
        print(f"  1 - {'Check' if to_call == 0 else f'Call ${to_call}'}")
        if max_raise > 0:
            print(f"  2 - Bet/Raise (custom amount)")
        print("="*60)
        
        while True:
            try:
                action_input = input("Enter action (0=Fold, 1=Call, 2=Bet): ").strip()
                action = int(action_input)
                
                if action not in [0, 1, 2]:
                    print("Invalid action. Choose 0, 1, or 2.")
                    continue
                
                # If not betting, return immediately
                if action != 2:
                    return action, None
                
                # Handle bet/raise
                if max_raise <= 0:
                    print("You don't have enough chips to raise. Calling instead.")
                    return 1, None
                
                # Get bet amount
                min_raise_amount = pot_manager.min_raise
                to_call_amount = to_call
                min_total_bet = to_call_amount + min_raise_amount
                max_total_bet = to_call_amount + max_raise
                
                print(f"\nBet amount (min ${min_raise_amount}, max ${max_raise}) to add on top of call")
                print(f"Total contribution will be: call ${to_call_amount} + bet amount")
                bet_input = input(f"Enter bet amount [${min_raise_amount}]: ").strip()
                
                if bet_input == "":
                    bet_amount = min_raise_amount
                else:
                    bet_amount = int(bet_input)
                
                # Validate bet amount
                if bet_amount < min_raise_amount:
                    print(f"Bet too small! Using minimum: ${min_raise_amount}")
                    bet_amount = min_raise_amount
                elif bet_amount > max_raise:
                    print(f"Bet too large! Using maximum (all-in): ${max_raise}")
                    bet_amount = max_raise
                
                # Convert bet amount to total contribution
                total_contribution = to_call_amount + bet_amount
                
                print(f"\nYou will contribute ${total_contribution} total (call ${to_call_amount} + bet ${bet_amount})")
                
                return action, total_contribution
                
            except ValueError:
                print("Invalid input. Please enter a number.")
            except (KeyboardInterrupt, EOFError):
                print("\nDefaulting to fold.")
                return 0, None


def play_game(model_path: str = None, num_opponents: int = 1, opponent_type: str = "random"):
    """Play poker with custom bet amounts"""
    
    num_players = num_opponents + 1
    
    if not 2 <= num_players <= 10:
        print(f"Error: Number of players must be between 2 and 10")
        return
    
    print("\n" + "="*60)
    print("Texas Hold'em Poker - Custom Bet Edition")
    print("="*60)
    print(f"Players: You vs {num_opponents} bot(s)")
    print()
    
    # Create environment
    env = TexasHoldemEnv(
        num_players=num_players,
        starting_stack=1000,
        small_blind=5,
        big_blind=10,
        min_raise_multiplier=1.0
    )
    
    env.current_agent = 0
    
    # Create agents
    agents = [InteractiveHumanAgent(env, name="You")]
    
    if model_path and os.path.exists(model_path):
        print(f"Loading trained model: {model_path}")
        for i in range(num_opponents):
            agent = PPOAgent.load_agent(model_path, env, name=f"Bot_{i+1}")
            agents.append(agent)
    else:
        if model_path:
            print(f"Model not found, using {opponent_type} agents")
        
        for i in range(num_opponents):
            if opponent_type == "call":
                agent = CallAgent(name=f"CallBot_{i+1}")
            else:
                agent = RandomAgent(name=f"RandomBot_{i+1}")
            agents.append(agent)
    
    print("\n" + "="*60)
    print("Press Ctrl+C to quit anytime")
    print("="*60)
    
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
            
            obs = env.reset()
            
            print(f"\n{'='*60}")
            print(f"HAND #{hand_number}")
            print(f"{'='*60}\n")
            
            env.render()
            
            done = False
            step_count = 0
            
            while not done:
                current_player_idx = env.game_state.current_player_idx
                current_agent = agents[current_player_idx]
                
                if isinstance(current_agent, InteractiveHumanAgent):
                    # Human player
                    action, bet_amount = current_agent.select_action_with_amount(obs)
                    obs, reward, done, info = env.step_with_raise(action, bet_amount)
                    
                    # Show what happened
                    action_names = {0: "folded", 1: "called/checked", 2: "bet"}
                    action_name = action_names.get(action, "acted")
                    if action == 2 and bet_amount:
                        to_call = env.game_state.pot_manager.current_bet - current_agent.env.game_state.players[0].current_bet
                        bet_only = bet_amount - to_call
                        print(f"\nYou {action_name} ${bet_amount} total (${to_call} call + ${bet_only} bet)")
                    else:
                        print(f"\nYou {action_name}")
                else:
                    # Bot player
                    print(f"\n{current_agent.name}'s turn...")
                    action = current_agent.select_action(obs)
                    obs, reward, done, info = env.step(action)
                    
                    action_names = {0: "folds", 1: "calls/checks", 2: "raises"}
                    print(f"{current_agent.name} {action_names.get(action, 'acts')}")
                
                if not done:
                    env.render()
                
                step_count += 1
                
                if step_count > 100:
                    print("Warning: Hand taking too long, forcing completion")
                    break
            
            # Show results
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
            
            print("\nChip Stacks:")
            for player in env.game_state.players:
                profit = player.stack - env.starting_stack
                print(f"  {player.name}: ${player.stack} ({profit:+d})")
            
            # Ask to continue every 5 hands
            if hand_number % 5 == 0:
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
    parser = argparse.ArgumentParser(description="Play poker with custom bet amounts")
    parser.add_argument('--model', type=str, default=None, help='Path to trained model')
    parser.add_argument('--opponents', type=int, default=1, help='Number of opponents (1-9)')
    parser.add_argument('--opponent-type', type=str, choices=['random', 'call'], 
                       default='random', help='Type of opponent')
    
    args = parser.parse_args()
    play_game(args.model, args.opponents, args.opponent_type)


if __name__ == "__main__":
    main()