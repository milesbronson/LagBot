import React from 'react';
import { GameState } from '../../types/game';
import { formatCurrency } from '../../utils/formatting';

interface QuickBetButtonsProps {
  gameState: GameState;
  onAction: (actionType: number, raiseAmount?: number) => void;
  disabled?: boolean;
  className?: string;
}

export const QuickBetButtons: React.FC<QuickBetButtonsProps> = ({
  gameState,
  onAction,
  disabled = false,
  className = '',
}) => {
  const humanPlayer = gameState.players.find((p) => p.is_human);
  if (!humanPlayer) return null;

  const canCheck = gameState.current_bet === 0 || humanPlayer.bet === gameState.current_bet;
  const callAmount = gameState.current_bet - humanPlayer.bet;

  return (
    <div className={`flex gap-3 ${className}`}>
      {/* Fold button */}
      <button
        onClick={() => onAction(0)}
        disabled={disabled}
        className="px-6 py-3 bg-red-600 hover:bg-red-700 text-white font-bold rounded-lg
                   disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
      >
        Fold
      </button>

      {/* Check/Call button */}
      <button
        onClick={() => onAction(1)}
        disabled={disabled}
        className="px-6 py-3 bg-green-600 hover:bg-green-700 text-white font-bold rounded-lg
                   disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
      >
        {canCheck ? 'Check' : `Call ${formatCurrency(callAmount)}`}
      </button>
    </div>
  );
};
