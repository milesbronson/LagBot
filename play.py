"""
Play with flexible human betting and discrete bot actions (including all-in)
"""

import argparse
import os

from src.poker_env.texas_holdem_env import TexasHoldemEnv
from src.agents.ppo_agent import PPOAgent
from src.agents.human_agent import HumanAgent
from src.agents.random_agent import RandomAgent, CallAgent


class FlexibleHumanAgent(HumanAgent):
    """Human agent that can raise any amount"""
    
    def __init__(self, env, name: str = "Human"):
        super().__init__(name)
        self.env = env
    
    def select_action_with_custom_amount(self, observation):
        """Get action and custom bet amount from human"""
        current_player = self.env.game_state.get_current_player()
        pot_manager = self.env.game_state.pot_manager
        to_call = pot_manager.current_bet - current_player.current_bet
        
        print("\n" + "="*60)
        print("YOUR TURN!")
        print("="*60)
        print(f"Your stack: ${current_player.stack}")
        print(f"Your current bet: ${current_player.current_bet}")
        print(f"Pot total: ${pot_manager.get_pot_total()}")
        print(f"Amount to call: ${to_call}")
        print(f"Min raise: ${pot_manager.min_raise}")
        
        print("\nActions:")
        print("  0 - Fold")
        print(f"  1 - {'Check' if to_call == 0 else f'Call ${to_call}'}")
        print("  2 - Bet/Raise (custom amount)")
        print("  3 - All-in")
        print("="*60)
        
        while True:
            try:
                action_input = input("Enter action (0=Fold, 1=Call, 2=Bet, 3=All-in): ").strip()
                action = int(action_input)
                
                if action not in [0, 1, 2, 3]:
                    print("Invalid action. Choose 0, 1, 2, or 3.")
                    continue
                
                if action == 0:
                    return action, None
                elif action == 1:
                    return action, None
                elif action == 3:
                    # All-in
                    return 2, current_player.stack
                else:
                    # Custom bet/raise
                    print(f"\nHow much do you want to raise?")
                    print(f"Minimum raise: ${pot_manager.min_raise}")
                    print(f"Maximum (all-in): ${current_player.stack + to_call}")
                    
                    while True:
                        try:
                            raise_input = input("Enter raise amount: $").strip()
                            raise_amount = int(raise_input)
                            
                            if raise_amount < pot_manager.min_raise:
                                print(f"Raise too small. Minimum is ${pot_manager.min_raise}")
                                continue
                            
                            if raise_amount > current_player.stack:
                                print(f"Not enough chips. Using all-in: ${current_player.stack}")
                                raise_amount = current_player.stack
                            
                            total_contribution = to_call + raise_amount
                            print(f"\nYou will contribute ${total_contribution} total")
                            print(f"(Call ${to_call} + Raise ${raise_amount})")
                            
                            return action, total_contribution
                        
                        except ValueError:
                            print("Invalid amount. Enter a number.")
                
            except ValueError:
                print("Invalid input. Please enter a number.")
            except (KeyboardInterrupt, EOFError):
                print("\nDefaulting to fold.")
                return 0, None


class BotWithDiscreteActions:
    """Wrapper for bots using discrete actions"""
    
    def __init__(self, agent, env, name: str = None):
        self.agent = agent
        self.env = env
        self.name = name or agent.name
    
    def select_discrete_action(self, observation):
        """Bot selects from discrete raise bins + all-in"""
        return self.agent.select_action(observation)


def play_game(model_path: str = None, num_opponents: int = 1, opponent_type: str = "random"):
    """Play poker with flexible betting"""
    
    num_players = num_opponents + 1
    
    if not 2 <= num_players <= 10:
        print(f"Error: Number of players must be between 2 and 10")
        return
    
    print("\n" + "="*60)
    print("Texas Hold'em Poker - Flexible Betting + All-in")
    print("="*60)
    print(f"Players: You vs {num_opponents} bot(s)")
    print(f"Bot actions: Fold, Call, Raise 50%/100%/200% pot, All-in")
    print(f"Human actions: Fold, Call, Custom raise, All-in")
    print()
    
    env = TexasHoldemEnv(
        num_players=num_players,
        starting_stack=1000,
        small_blind=5,
        big_blind=10,
        raise_bins=[0.5, 1.0, 2.0],
        include_all_in=True
    )
    
    env.current_agent = 0
    human_agent = FlexibleHumanAgent(env, name="You")
    agents = [human_agent]
    
    if model_path and os.path.exists(model_path):
        print(f"Loading trained model: {model_path}")
        for i in range(num_opponents):
            agent = PPOAgent.load_agent(model_path, env, name=f"Bot_{i+1}")
            agents.append(BotWithDiscreteActions(agent, env))
    else:
        if model_path:
            print(f"Model not found, using {opponent_type} agents")
        
        for i in range(num_opponents):
            if opponent_type == "call":
                agent = CallAgent(name=f"CallBot_{i+1}")
            else:
                agent = RandomAgent(name=f"RandomBot_{i+1}")
            agents.append(BotWithDiscreteActions(agent, env))
    
    print("\n" + "="*60)
    print("Press Ctrl+C to quit anytime")
    print("="*60)
    
    hand_number = 0
    
    try:
        while True:
            hand_number += 1
            
            if env.game_state.players[0].stack <= 0:
                print("\n" + "="*60)
                print("You're out of chips! Game over.")
                print("="*60)
                break
            
            obs, info = env.reset()
            
            print(f"\n{'='*60}")
            print(f"HAND #{hand_number}")
            print(f"{'='*60}\n")
            
            env.render()
            done = False
            step_count = 0
            
            while not done:
                current_player_idx = env.game_state.current_player_idx
                current_agent = agents[current_player_idx]
                
                if isinstance(current_agent, FlexibleHumanAgent):
                    action, custom_amount = current_agent.select_action_with_custom_amount(obs)
                    
                    if action == 0:
                        obs, reward, done, info = env.step(0)
                        print(f"\nYou folded")
                    elif action == 1:
                        obs, reward, done, info = env.step(1)
                        print(f"\nYou called/checked")
                    else:
                        obs, reward, done, info = env.step_with_raise(2, custom_amount)
                        print(f"\nYou raised/went all-in")
                
                else:
                    print(f"\n{current_agent.name}'s turn...")
                    valid_actions = env.get_valid_actions()
                    discrete_action = current_agent.agent.select_action(obs, valid_actions)
                    obs, reward, done, info = env.step(discrete_action)
                    
                    action_desc = env.get_action_description(discrete_action)
                    print(f"{current_agent.name} {action_desc}")
                
                if not done:
                    env.render()
                
                step_count += 1
                if step_count > 100:
                    print("Warning: Hand taking too long, forcing completion")
                    break
            
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
            
            if hand_number % 5 == 0:
                print()
                continue_game = input("Continue playing? (y/n): ").strip().lower()
                if continue_game != 'y':
                    break
    
    except KeyboardInterrupt:
        print("\n\nGame interrupted by user.")
    
    print("\n" + "="*60)
    print("GAME SUMMARY")
    print("="*60)
    print(f"Hands played: {hand_number}")
    print("\nFinal chip stacks:")
    for player in env.game_state.players:
        profit = player.stack - env.starting_stack
        print(f"  {player.name}: ${player.stack} ({profit:+d})")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Play poker with flexible betting")
    parser.add_argument('--model', type=str, default=None, help='Path to trained model')
    parser.add_argument('--opponents', type=int, default=1, help='Number of opponents (1-9)')
    parser.add_argument('--opponent-type', type=str, choices=['random', 'call'], 
                       default='random', help='Type of opponent')
    
    args = parser.parse_args()
    play_game(args.model, args.opponents, args.opponent_type)


if __name__ == "__main__":
    main()