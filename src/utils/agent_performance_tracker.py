"""
Agent Performance Tracker - Tracks individual agent statistics during gameplay sessions
"""

from dataclasses import dataclass, asdict, field
from typing import Dict
from datetime import datetime
import json


@dataclass
class AgentStats:
    """Statistics for a single agent"""
    
    name: str
    agent_id: int
    net_winnings: int = 0  # Directly from player.total_winnings
    hands_played: int = 0
    hands_won: int = 0
    hands_lost: int = 0
    biggest_win: int = 0
    biggest_loss: int = 0
    folds: int = 0
    
    @property
    def win_rate(self) -> float:
        """Win rate as percentage"""
        if self.hands_played == 0:
            return 0.0
        return (self.hands_won / self.hands_played) * 100
    
    @property
    def avg_win(self) -> float:
        """Average win amount when winning"""
        if self.hands_won == 0:
            return 0.0
        return self.biggest_win / self.hands_won if self.biggest_win > 0 else 0.0
    
    @property
    def avg_loss(self) -> float:
        """Average loss amount when losing"""
        if self.hands_lost == 0:
            return 0.0
        return abs(self.biggest_loss) / self.hands_lost if self.biggest_loss < 0 else 0.0
    
    @property
    def profit_factor(self) -> float:
        """Profit factor: wins / losses (higher is better)"""
        if self.hands_lost == 0 or self.hands_won == 0:
            return 0.0
        total_won = self.biggest_win * self.hands_won
        total_lost = abs(self.biggest_loss) * self.hands_lost
        if total_lost == 0:
            return 0.0
        return total_won / total_lost
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'name': self.name,
            'agent_id': self.agent_id,
            'net_winnings': self.net_winnings,
            'hands_played': self.hands_played,
            'hands_won': self.hands_won,
            'hands_lost': self.hands_lost,
            'biggest_win': self.biggest_win,
            'biggest_loss': self.biggest_loss,
            'folds': self.folds,
            'win_rate': round(self.win_rate, 2),
            'avg_win': round(self.avg_win, 2),
            'avg_loss': round(self.avg_loss, 2),
            'profit_factor': round(self.profit_factor, 2)
        }


class SessionTracker:
    """Tracks agent performance during a gameplay session"""
    
    def __init__(self, session_name: str = None):
        """
        Initialize session tracker
        
        Args:
            session_name: Name for this session (default: timestamp)
        """
        self.session_name = session_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.agents: Dict[int, AgentStats] = {}
        self.hand_count = 0
        self.session_start_stacks: Dict[int, int] = {}
    
    def register_agent(self, agent_id: int, agent_name: str, starting_stack: int):
        """
        Register an agent at the start of a session
        
        Args:
            agent_id: Unique ID for the agent (usually their player position)
            agent_name: Name of the agent
            starting_stack: Starting chip stack (for reference only)
        """
        self.agents[agent_id] = AgentStats(
            name=agent_name,
            agent_id=agent_id,
            net_winnings=0
        )
        self.session_start_stacks[agent_id] = starting_stack
    
    def record_hand_start(self):
        """Called at the start of each hand"""
        self.hand_count += 1
        for agent in self.agents.values():
            agent.hands_played += 1
    
    def record_hand_result(self, agent_id: int, hand_won: bool = False):
        """
        Record the result of a hand for an agent
        
        Args:
            agent_id: ID of the agent
            hand_won: Whether the agent won the hand
        """
        if agent_id not in self.agents:
            return
        
        agent = self.agents[agent_id]
        
        if hand_won:
            agent.hands_won += 1
        else:
            agent.hands_lost += 1
    
    def update_agent_winnings(self, agent_id: int, player_total_winnings: int):
        """
        Update agent's net winnings from player.total_winnings
        
        Args:
            agent_id: ID of the agent
            player_total_winnings: player.total_winnings from the Player object
        """
        if agent_id in self.agents:
            self.agents[agent_id].net_winnings = player_total_winnings
    
    def record_fold(self, agent_id: int):
        """Record a fold action"""
        if agent_id in self.agents:
            self.agents[agent_id].folds += 1
    
    def get_agent_stats(self, agent_id: int) -> AgentStats:
        """Get stats for a specific agent"""
        return self.agents.get(agent_id)
    
    def get_all_stats(self) -> Dict[int, AgentStats]:
        """Get all agent stats"""
        return self.agents.copy()
    
    def print_session_summary(self):
        """Print summary of entire session"""
        print("\n" + "="*80)
        print(f"SESSION SUMMARY: {self.session_name}")
        print("="*80)
        print(f"Total hands played: {self.hand_count}\n")
        
        print(f"{'Agent':<20} {'Hands':<8} {'Won':<6} {'Lost':<6} {'Win %':<8} {'Net Winnings':<12}")
        print("-"*80)
        
        for agent_id in sorted(self.agents.keys()):
            agent = self.agents[agent_id]
            print(f"{agent.name:<20} {agent.hands_played:<8} {agent.hands_won:<6} "
                  f"{agent.hands_lost:<6} {agent.win_rate:>6.1f}% ${agent.net_winnings:>10,}")
        
        print("="*80)
    
    def print_rankings(self):
        """Print final rankings by profit"""
        if not self.agents:
            return
        
        # Sort by net winnings
        sorted_agents = sorted(
            self.agents.values(),
            key=lambda a: a.net_winnings,
            reverse=True
        )
        
        print("\n" + "="*80)
        print("FINAL RANKINGS")
        print("="*80)
        
        for rank, agent in enumerate(sorted_agents, 1):
            print(f"{rank}. {agent.name:<20} ${agent.net_winnings:>10,}  "
                  f"({agent.win_rate:>5.1f}%)")
        
        print("="*80)
    
    def print_detailed_stats(self):
        """Print detailed statistics for each agent"""
        print("\n" + "="*80)
        print("DETAILED STATISTICS")
        print("="*80)
        
        for agent_id in sorted(self.agents.keys()):
            agent = self.agents[agent_id]
            
            print(f"\n{agent.name}:")
            print(f"  Hands played:    {agent.hands_played}")
            print(f"  Hands won:       {agent.hands_won}")
            print(f"  Hands lost:      {agent.hands_lost}")
            print(f"  Win rate:        {agent.win_rate:.1f}%")
            print(f"  Total winnings:  ${agent.total_winnings:,}")
            print(f"  Total losses:    ${agent.total_losses:,}")
            print(f"  Net profit:      ${agent.net_winnings:,}")
            print(f"  Average win:     ${agent.avg_win:.2f}")
            print(f"  Average loss:    ${agent.avg_loss:.2f}")
            print(f"  Biggest win:     ${agent.biggest_win:,}")
            print(f"  Biggest loss:    ${agent.biggest_loss:,}")
            print(f"  Profit factor:   {agent.profit_factor:.2f}x")
            print(f"  Folds:           {agent.folds}")
        
        print("\n" + "="*80)
    
    def to_json(self, filename: str = None) -> str:
        """
        Export session to JSON
        
        Args:
            filename: Output filename (default: session_name.json)
            
        Returns:
            JSON string
        """
        if filename is None:
            filename = f"session_stats_{self.session_name}.json"
        
        data = {
            'session_name': self.session_name,
            'total_hands': self.hand_count,
            'agents': {
                str(agent_id): agent.to_dict()
                for agent_id, agent in self.agents.items()
            }
        }
        
        json_str = json.dumps(data, indent=2)
        
        with open(filename, 'w') as f:
            f.write(json_str)
        
        return json_str


if __name__ == "__main__":
    # Example usage
    tracker = SessionTracker("demo_session")
    
    # Register agents
    tracker.register_agent(0, "You", 1000)
    tracker.register_agent(1, "Bot_RandomAgent", 1000)
    tracker.register_agent(2, "Bot_CallAgent", 1000)
    
    # Simulate some hands
    for hand in range(10):
        tracker.record_hand_start()
        
        # Simulate random results
        import random
        for agent_id in range(3):
            result = random.randint(-200, 300)
            if result != 0:
                tracker.record_hand_result(agent_id, result, hand_won=(result > 0))
    
    # Print summary
    tracker.print_session_summary()
    tracker.print_rankings()
    tracker.print_detailed_stats()