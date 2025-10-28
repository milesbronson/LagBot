"""
Human agent for command-line play
"""

import numpy as np
from src.agents.base_agent import BaseAgent


class HumanAgent(BaseAgent):
    """
    Human player agent for interactive play via command line
    """
    
    def __init__(self, name: str = "Human"):
        """
        Initialize human agent
        
        Args:
            name: Agent name
        """
        super().__init__(name)
    
    def select_action(self, observation: np.ndarray, valid_actions: list = None) -> int:
        """
        Get action from human input
        
        Args:
            observation: Current game observation (displayed separately)
            valid_actions: List of valid action indices
            
        Returns:
            Selected action index
        """
        if valid_actions is None:
            valid_actions = [0, 1, 2]
        
        action_names = {0: "Fold", 1: "Check/Call", 2: "Raise"}
        
        while True:
            print("\n" + "="*40)
            print("Your turn! Choose an action:")
            for action in valid_actions:
                print(f"  {action}: {action_names[action]}")
            print("="*40)
            
            try:
                action_input = input("Enter action number: ").strip()
                action = int(action_input)
                
                if action in valid_actions:
                    return action
                else:
                    print(f"Invalid action. Please choose from {valid_actions}")
            except (ValueError, KeyboardInterrupt):
                print("\nInvalid input. Please enter a number.")
            except EOFError:
                print("\nInput closed. Defaulting to fold.")
                return 0  # Default to fold