"""
Create simple rule-based opponents with VERY different strategies
to force the agent to differentiate
"""

from src.agents.base_agent import BaseAgent
import numpy as np


class TightAgent(BaseAgent):
    """
    Plays VERY tight - only plays premium hands
    VPIP: ~15%, PFR: ~12%
    """
    def __init__(self, name="TightAgent"):
        super().__init__(name)

    def select_action(self, observation, valid_actions=None):
        # Extract hole cards from observation (first 8 values)
        hole_cards = observation[:8]

        # Check if we have good cards (high ranks)
        has_high_cards = np.max(hole_cards) > 0.7  # High cards

        # Position (observation[33])
        position = observation[33] if len(observation) > 33 else 0.5
        late_position = position > 0.6

        # Check if there's a bet to call (observation[31])
        to_call = observation[31] if len(observation) > 31 else 0
        facing_bet = to_call > 0.01

        if not has_high_cards:
            # Fold weak hands
            return 0
        elif facing_bet:
            # Call with strong hands
            return 1
        elif late_position:
            # Raise in late position with good hands
            return 2  # Raise 50% pot
        else:
            # Check/call otherwise
            return 1


class AggressiveAgent(BaseAgent):
    """
    Plays VERY aggressive - raises a lot
    VPIP: ~60%, PFR: ~40%, Aggression: High
    """
    def __init__(self, name="AggressiveAgent"):
        super().__init__(name)

    def select_action(self, observation, valid_actions=None):
        # Random aggression
        action_probs = [0.1, 0.2, 0.5, 0.2]  # [fold, call, raise, all-in]

        # Check if we can raise (have enough chips)
        stack = observation[28] if len(observation) > 28 else 1.0
        to_call = observation[31] if len(observation) > 31 else 0

        if stack < 0.1:  # Short stack
            # All-in more often
            return np.random.choice([1, 5], p=[0.3, 0.7])  # Call or all-in

        if to_call > 0.5:  # Big bet
            # Fold sometimes, raise sometimes
            return np.random.choice([0, 1, 2], p=[0.3, 0.3, 0.4])

        # Default: very aggressive
        return np.random.choice([1, 2, 3], p=[0.2, 0.5, 0.3])


class PassiveAgent(BaseAgent):
    """
    Plays passive - calls a lot, rarely raises
    VPIP: ~50%, PFR: ~5%, Aggression: Low
    """
    def __init__(self, name="PassiveAgent"):
        super().__init__(name)

    def select_action(self, observation, valid_actions=None):
        # Check if facing a big bet
        to_call = observation[31] if len(observation) > 31 else 0
        stack = observation[28] if len(observation) > 28 else 1.0

        if to_call > stack * 0.8:  # Bet is >80% of stack
            # Fold to huge bets
            return 0
        elif to_call > 0.01:
            # Call most bets
            return 1
        else:
            # Check when possible, occasionally raise small
            return np.random.choice([1, 2], p=[0.9, 0.1])


class ManiacAgent(BaseAgent):
    """
    MANIAC - all-in or fold, no middle ground
    VPIP: ~40%, PFR: ~35%, All-in: ~25%
    """
    def __init__(self, name="ManiacAgent"):
        super().__init__(name)

    def select_action(self, observation, valid_actions=None):
        # 40% all-in, 30% fold, 30% call
        action_probs = [0.3, 0.3, 0.0, 0.0, 0.0, 0.4]  # Heavy on fold and all-in
        return np.random.choice(range(6), p=action_probs)


if __name__ == "__main__":
    print("Diverse opponent agents created!")
    print("\nAgent profiles:")
    print("  TightAgent: VPIP ~15%, PFR ~12% (plays only premium hands)")
    print("  AggressiveAgent: VPIP ~60%, PFR ~40% (raises a lot)")
    print("  PassiveAgent: VPIP ~50%, PFR ~5% (calling station)")
    print("  ManiacAgent: VPIP ~40%, All-in ~25% (all-in or fold)")
    print("\nThese agents have VERY different strategies!")
    print("The learning agent MUST differentiate them to win.")
