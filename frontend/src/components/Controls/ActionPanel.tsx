import React, { useState } from 'react';
import { GameState } from '../../types/game';
import { useGameStore } from '../../stores/gameStore';
import { QuickBetButtons } from './QuickBetButtons';
import { BetSlider } from './BetSlider';
import { formatCurrency } from '../../utils/formatting';

interface ActionPanelProps {
  gameState: GameState;
  className?: string;
}

export const ActionPanel: React.FC<ActionPanelProps> = ({ gameState, className = '' }) => {
  const { submitPlayerAction, isLoading } = useGameStore();
  const [raiseAmount, setRaiseAmount] = useState(gameState.min_raise);

  const humanPlayer = gameState.players.find((p) => p.is_human);
  if (!humanPlayer) return null;

  const canRaise = humanPlayer.stack > gameState.current_bet - humanPlayer.bet;
  const minRaise = gameState.min_raise;
  const maxRaise = humanPlayer.stack;

  const handleAction = async (actionType: number, customRaise?: number) => {
    await submitPlayerAction({
      action_type: actionType,
      raise_amount: customRaise,
    });
  };

  if (!gameState.is_human_turn) {
    return (
      <div className={`bg-gray-800 rounded-lg p-6 ${className}`}>
        <div className="text-white text-center text-lg">
          Waiting for other players...
        </div>
      </div>
    );
  }

  return (
    <div className={`bg-gray-800 rounded-lg p-6 ${className}`}>
      <div className="flex flex-col gap-4">
        <div className="text-white text-xl font-bold text-center">
          Your Turn
        </div>

        {/* Quick action buttons */}
        <QuickBetButtons
          gameState={gameState}
          onAction={handleAction}
          disabled={isLoading}
        />

        {/* Raise controls */}
        {canRaise && (
          <div className="flex flex-col gap-3">
            <div className="border-t border-gray-600 pt-4">
              <BetSlider
                min={minRaise}
                max={maxRaise}
                value={raiseAmount}
                onChange={setRaiseAmount}
                step={gameState.big_blind}
              />
            </div>

            <div className="flex gap-3">
              <button
                onClick={() => handleAction(2, raiseAmount)}
                disabled={isLoading || raiseAmount < minRaise}
                className="flex-1 px-6 py-3 bg-yellow-600 hover:bg-yellow-700 text-white font-bold rounded-lg
                           disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                Raise {formatCurrency(raiseAmount)}
              </button>

              <button
                onClick={() => handleAction(2, humanPlayer.stack)}
                disabled={isLoading}
                className="px-6 py-3 bg-red-700 hover:bg-red-800 text-white font-bold rounded-lg
                           disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                All-In
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
