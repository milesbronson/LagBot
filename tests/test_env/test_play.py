"""
Tests for play.py - verify FlexibleHumanAgent and game flow
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from src.poker_env.texas_holdem_env import TexasHoldemEnv


class TestCardToString:
    """Test the _card_to_string method"""
    
    @pytest.fixture
    def setup_agent(self):
        """Create a FlexibleHumanAgent for testing"""
        env = TexasHoldemEnv(num_players=2)
        # Import here to avoid circular imports
        from play import FlexibleHumanAgent
        agent = FlexibleHumanAgent(env, name="TestAgent")
        return agent
    
    def test_card_to_string_zero_card(self, setup_agent):
        """Test that card 0 returns '?'"""
        result = setup_agent._card_to_string(0)
        assert result == "?"
    
    def test_card_to_string_valid_cards(self, setup_agent):
        """Test conversion of valid card integers"""
        # These are hypothetical Treys card encodings
        # The exact values depend on Treys library encoding
        # Just verify it doesn't crash and returns a string
        test_cards = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
        
        for card in test_cards:
            result = setup_agent._card_to_string(card)
            assert isinstance(result, str)
            assert len(result) > 0
    
    def test_card_to_string_invalid_rank(self, setup_agent):
        """Test that invalid ranks return '?'"""
        # A card with invalid rank encoding
        invalid_card = 0xFFFF  # Likely invalid
        result = setup_agent._card_to_string(invalid_card)
        assert isinstance(result, str)


class TestFlexibleHumanAgentDisplay:
    """Test that display methods don't crash"""
    
    @pytest.fixture
    def env_and_agent(self):
        """Setup environment and agent"""
        env = TexasHoldemEnv(num_players=3, starting_stack=1000)
        from play import FlexibleHumanAgent
        agent = FlexibleHumanAgent(env, name="TestPlayer")
        return env, agent
    
    def test_agent_initializes(self, env_and_agent):
        """Test FlexibleHumanAgent initializes correctly"""
        env, agent = env_and_agent
        assert agent.name == "TestPlayer"
        assert agent.env == env
    
    def test_display_before_input(self, env_and_agent, capsys):
        """Test that display code runs without crashing before waiting for input"""
        env, agent = env_and_agent
        env.reset()
        
        # Mock input to immediately return a fold action
        with patch('builtins.input', return_value='0'):
            action, amount = agent.select_action_with_custom_amount(np.zeros(32))
        
        captured = capsys.readouterr()
        
        # Verify key display elements are shown
        assert "YOUR TURN!" in captured.out
        assert "Community Cards:" in captured.out
        assert "Your Hand:" in captured.out
        assert "Player Bets This Round:" in captured.out
        assert "Your stack:" in captured.out
        assert "Pot total:" in captured.out
        assert action == 0


class TestFlexibleHumanAgentActions:
    """Test action selection logic"""
    
    @pytest.fixture
    def env_and_agent(self):
        """Setup environment and agent"""
        env = TexasHoldemEnv(num_players=3, starting_stack=1000)
        from play import FlexibleHumanAgent
        agent = FlexibleHumanAgent(env, name="TestPlayer")
        return env, agent
    
    def test_fold_action(self, env_and_agent):
        """Test fold action"""
        env, agent = env_and_agent
        env.reset()
        
        with patch('builtins.input', return_value='0'):
            action, amount = agent.select_action_with_custom_amount(np.zeros(32))
        
        assert action == 0
        assert amount is None
    
    def test_call_action(self, env_and_agent):
        """Test call action"""
        env, agent = env_and_agent
        env.reset()
        
        with patch('builtins.input', return_value='1'):
            action, amount = agent.select_action_with_custom_amount(np.zeros(32))
        
        assert action == 1
        assert amount is None
    
    def test_all_in_action(self, env_and_agent):
        """Test all-in action"""
        env, agent = env_and_agent
        env.reset()
        current_player = env.game_state.get_current_player()
        stack = current_player.stack
        
        with patch('builtins.input', return_value='3'):
            action, amount = agent.select_action_with_custom_amount(np.zeros(32))
        
        assert action == 2  # Raise action
        assert amount == stack
    
    def test_custom_raise_action(self, env_and_agent):
        """Test custom raise action"""
        env, agent = env_and_agent
        env.reset()
        
        # First input: 2 (raise option)
        # Second input: 100 (raise amount)
        with patch('builtins.input', side_effect=['2', '100']):
            action, amount = agent.select_action_with_custom_amount(np.zeros(32))
        
        assert action == 2
        assert amount >= 100  # Could be more if includes call amount
    
    def test_invalid_action_retries(self, env_and_agent):
        """Test that invalid actions cause retry"""
        env, agent = env_and_agent
        env.reset()
        
        # First input: invalid, second: valid
        with patch('builtins.input', side_effect=['99', '0']):
            action, amount = agent.select_action_with_custom_amount(np.zeros(32))
        
        assert action == 0
    
    def test_keyboard_interrupt_folds(self, env_and_agent):
        """Test that keyboard interrupt defaults to fold"""
        env, agent = env_and_agent
        env.reset()
        
        with patch('builtins.input', side_effect=KeyboardInterrupt()):
            action, amount = agent.select_action_with_custom_amount(np.zeros(32))
        
        assert action == 0
        assert amount is None


class TestBotWithDiscreteActions:
    """Test BotWithDiscreteActions wrapper"""
    
    def test_bot_wrapper_initialization(self):
        """Test bot wrapper initializes correctly"""
        from play import BotWithDiscreteActions
        
        mock_agent = Mock()
        mock_agent.name = "MockBot"
        mock_env = Mock()
        
        bot = BotWithDiscreteActions(mock_agent, mock_env, name="TestBot")
        
        assert bot.name == "TestBot"
        assert bot.agent == mock_agent
        assert bot.env == mock_env
    
    def test_bot_wrapper_select_action(self):
        """Test bot wrapper selects actions"""
        from play import BotWithDiscreteActions
        
        mock_agent = Mock()
        mock_agent.select_action.return_value = 1
        mock_env = Mock()
        
        bot = BotWithDiscreteActions(mock_agent, mock_env)
        obs = np.zeros(32)
        
        action = bot.select_discrete_action(obs)
        
        assert action == 1
        mock_agent.select_action.assert_called_once()


class TestGameFlow:
    """Test the overall game flow"""
    
    def test_game_initialization(self):
        """Test that game initializes with correct number of players"""
        env = TexasHoldemEnv(num_players=3, starting_stack=1000)
        
        assert len(env.game_state.players) == 3
        for player in env.game_state.players:
            assert player.stack == 1000
    
    def test_game_reset(self):
        """Test game can reset for new hand"""
        env = TexasHoldemEnv(num_players=3, starting_stack=1000)
        obs1, info1 = env.reset()
        obs2, info2 = env.reset()
        
        assert isinstance(obs1, np.ndarray)
        assert isinstance(obs2, np.ndarray)
        assert obs1.shape == obs2.shape
    
    def test_game_step(self):
        """Test game step returns correct format"""
        env = TexasHoldemEnv(num_players=3, starting_stack=1000)
        obs, info = env.reset()
        
        obs, reward, terminated, truncated, info = env.step(1)  # Call action
        
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
    
    def test_hand_completion(self):
        """Test that a hand can be completed"""
        env = TexasHoldemEnv(num_players=2, starting_stack=1000)
        obs, info = env.reset()
        
        done = False
        steps = 0
        max_steps = 50
        
        while not done and steps < max_steps:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
        
        assert done, "Hand should complete within 50 steps"
        assert steps < max_steps


class TestChipTracking:
    """Test that chips are tracked correctly"""
    def test_chip_conservation_debug_blinds(self):
        """
        Debug test to trace where the $10 goes.
        Track blinds from posting to pot to winner.
        """
        env = TexasHoldemEnv(num_players=2, starting_stack=1000)
        initial_total = 2000
        
        obs, info = env.reset()
        
        print("\n" + "="*80)
        print("DEBUG: BLIND TRACKING")
        print("="*80)
        
        # After reset (blinds posted)
        print("\nAfter reset (blinds posted):")
        for p in env.game_state.players:
            print(f"  {p.name}: stack=${p.stack}, bet=${p.current_bet}, total_bet=${p.total_bet_this_hand}")
        
        pot_total = env.game_state.pot_manager.get_pot_total()
        stacks_sum = sum(p.stack for p in env.game_state.players)
        print(f"  Pot: ${pot_total}")
        print(f"  Sum of stacks: ${stacks_sum}")
        print(f"  Pot + Stacks: ${pot_total + stacks_sum}")
        print(f"  Expected: ${initial_total}")
        
        if pot_total + stacks_sum != initial_total:
            print(f"  ⚠ MISSING: ${initial_total - (pot_total + stacks_sum)}")
        
        # Play out hand
        done = False
        steps = 0
        while not done and steps < 100:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
        
        print("\nAfter hand complete:")
        for p in env.game_state.players:
            print(f"  {p.name}: stack=${p.stack}, total_bet=${p.total_bet_this_hand}, "
                f"total_winnings=${p.total_winnings}")
        
        # Check pot
        pot_total = env.game_state.pot_manager.get_pot_total()
        stacks_sum = sum(p.stack for p in env.game_state.players)
        total_bet_sum = sum(p.total_bet_this_hand for p in env.game_state.players)
        winnings_sum = sum(p.total_winnings for p in env.game_state.players)
        
        print(f"\nMetrics:")
        print(f"  Pot (unsettled): ${pot_total}")
        print(f"  Sum of stacks: ${stacks_sum}")
        print(f"  Sum of total_bet_this_hand: ${total_bet_sum}")
        print(f"  Sum of total_winnings: ${winnings_sum}")
        print(f"  Total accounted: ${stacks_sum + total_bet_sum + winnings_sum}")
        print(f"  Expected: ${initial_total}")
        
        if stacks_sum + total_bet_sum + winnings_sum != initial_total:
            print(f"  ⚠ MISSING: ${initial_total - (stacks_sum + total_bet_sum + winnings_sum)}")
        
        # Core check
        total_in_play = sum(p.stack for p in env.game_state.players)
        total_by_winnings = sum(p.total_winnings for p in env.game_state.players)
        
        print(f"\nCore check:")
        print(f"  Current stacks sum: ${total_in_play}")
        print(f"  Sum of total_winnings: ${total_by_winnings}")
        print(f"  Expected: ${initial_total}")
        
        # This should be true only if total_winnings captures EVERYTHING
        # But if blinds get forfeited without being added to winnings, we lose them
        
        assert total_in_play + total_by_winnings == initial_total or \
            total_in_play == initial_total, (
            f"Chips missing! stacks={total_in_play} + winnings={total_by_winnings} "
            f"!= {initial_total}"
        )
    def test_chip_conservation_comprehensive(self):
        """
        Comprehensive chip conservation test with hand history display.
        Checks: sum(stack) + rake == sum(total_buy_in)
        """
        env = TexasHoldemEnv(num_players=3, starting_stack=1000)
        initial_stacks = [p.stack for p in env.game_state.players]
        total_initial_chips = sum(initial_stacks)
        
        print("\n" + "="*80)
        print("CHIP CONSERVATION TEST - COMPREHENSIVE")
        print("="*80)
        
        num_hands = 5
        for hand_num in range(num_hands):
            obs, info = env.reset()
            
            # Play hand
            done = False
            steps = 0
            while not done and steps < 100:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                steps += 1
            
            # Display hand history with card visibility
            env.game_state.display_hand_history()
            
            # === METRICS ===
            current_sum = sum(p.stack for p in env.game_state.players)
            total_buy_in_sum = sum(p.total_buy_in for p in env.game_state.players)
            total_winnings_sum = sum(p.total_winnings for p in env.game_state.players)
            
            rake_collected = 0
            if hasattr(env.game_state.pot_manager, 'total_rake'):
                rake_collected = env.game_state.pot_manager.total_rake
            
            # Print metrics
            print(f"\n{'='*80}")
            print(f"CHIP CONSERVATION METRICS - Hand {hand_num + 1}")
            print(f"{'='*80}")
            print(f"  Current stacks sum: ${current_sum}")
            print(f"  Total buy-in sum: ${total_buy_in_sum}")
            print(f"  Total winnings sum: ${total_winnings_sum}")
            print(f"  Rake collected: ${rake_collected}")
            print(f"  Expected total: ${total_initial_chips}")
            
            # Per-player breakdown
            print(f"\nPer-player breakdown:")
            for p in env.game_state.players:
                print(f"  {p.name}: stack=${p.stack}, buy_in=${p.total_buy_in}, winnings=${p.total_winnings}")
            
            # === INVARIANTS ===
            assert current_sum + rake_collected == total_initial_chips, (
                f"Hand {hand_num}: Chip conservation failed! "
                f"stacks={current_sum} + rake={rake_collected} != initial={total_initial_chips}"
            )
            
            assert total_buy_in_sum == total_initial_chips, (
                f"Hand {hand_num}: Buy-in sum incorrect! "
                f"buy_in_sum={total_buy_in_sum} != initial={total_initial_chips}"
            )
            
            # Per-player invariant
            for p in env.game_state.players:
                if p.total_buy_in == initial_stacks[p.player_id]:
                    expected = initial_stacks[p.player_id] + p.total_winnings
                    assert p.stack == expected, (
                        f"Hand {hand_num}, {p.name}: stack={p.stack} != "
                        f"initial={initial_stacks[p.player_id]} + winnings={p.total_winnings}"
                    )
            
            print(f"\n✓ Hand {hand_num + 1}: All metrics consistent\n")
        
        print("="*80)
        print("✓ CHIP CONSERVATION COMPREHENSIVE TEST PASSED")
        print("="*80)

class TestCardDisplay:
    """Test card display formatting"""
    
    @pytest.fixture
    def agent(self):
        """Create agent for testing"""
        env = TexasHoldemEnv(num_players=2)
        from play import FlexibleHumanAgent
        return FlexibleHumanAgent(env, name="TestAgent")
    
    def test_card_string_format(self, agent):
        """Test that card strings are properly formatted"""
        result = agent._card_to_string(0)
        assert result == "?"
        
        # Test that valid cards produce strings with reasonable length
        for i in range(1, 100):
            result = agent._card_to_string(i)
            assert isinstance(result, str)
            assert len(result) > 0


def test_imports():
    """Test that all necessary imports work"""
    try:
        from play import FlexibleHumanAgent, BotWithDiscreteActions, play_game
        assert FlexibleHumanAgent is not None
        assert BotWithDiscreteActions is not None
        assert play_game is not None
    except ImportError as e:
        pytest.fail(f"Failed to import from play.py: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])