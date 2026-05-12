import React, { useState, useEffect } from 'react';
import { GameState } from '../../types/game';
import { useGameStore } from '../../stores/gameStore';
import { BetSlider } from './BetSlider';
import { formatCurrency } from '../../utils/formatting';

interface ActionPanelProps {
  gameState: GameState;
  className?: string;
}

export const ActionPanel: React.FC<ActionPanelProps> = ({ gameState, className = '' }) => {
  const { submitPlayerAction, isLoading } = useGameStore();

  const humanPlayer = gameState.players.find((p) => p.is_human);
  if (!humanPlayer) return null;

  const callAmount = Math.max(0, gameState.current_bet - humanPlayer.bet);
  const canCheck = callAmount === 0;
  const minRaise = Math.max(gameState.min_raise, gameState.big_blind);
  const maxRaise = humanPlayer.stack;
  const canRaise = maxRaise > callAmount;

  const [raiseAmount, setRaiseAmount] = useState(minRaise);

  useEffect(() => {
    setRaiseAmount(minRaise);
  }, [minRaise]);

  const handleAction = async (actionType: number) => {
    await submitPlayerAction({ action_type: actionType });
  };

  const handleRaise = async (amount: number) => {
    await submitPlayerAction({ action_type: 2, raise_amount: amount });
  };

  if (gameState.hand_complete) {
    return null;
  }

  if (!gameState.is_human_turn) {
    return (
      <div className={`bg-gray-800 rounded-lg p-6 ${className}`}>
        <div className="text-white text-center text-lg animate-pulse">
          Waiting for other players...
        </div>
      </div>
    );
  }

  const potSize = gameState.pot;
  const quickBets = [
    { label: '½ Pot', amount: Math.max(minRaise, Math.floor(potSize * 0.5)) },
    { label: 'Pot', amount: Math.max(minRaise, potSize) },
    { label: '2x Pot', amount: Math.max(minRaise, potSize * 2) },
  ].filter((b) => b.amount <= maxRaise);

  return (
    <div className={`bg-gray-800 rounded-lg p-6 ${className}`}>
      <div className="flex flex-col gap-4">
        <div className="text-white text-xl font-bold text-center">Your Turn</div>

        {/* Primary actions */}
        <div className="flex gap-3">
          <button
            onClick={() => handleAction(0)}
            disabled={isLoading}
            className="px-6 py-3 bg-red-600 hover:bg-red-700 text-white font-bold rounded-lg
                       disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            Fold
          </button>

          <button
            onClick={() => handleAction(1)}
            disabled={isLoading}
            className="flex-1 px-6 py-3 bg-green-600 hover:bg-green-700 text-white font-bold rounded-lg
                       disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {canCheck ? 'Check' : `Call ${formatCurrency(callAmount)}`}
          </button>

          <button
            onClick={() => handleRaise(maxRaise)}
            disabled={isLoading || !canRaise}
            className="px-6 py-3 bg-red-700 hover:bg-red-800 text-white font-bold rounded-lg
                       disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            All-In {formatCurrency(maxRaise)}
          </button>
        </div>

        {/* Raise controls */}
        {canRaise && (
          <div className="border-t border-gray-600 pt-4 flex flex-col gap-3">
            {/* Quick bet buttons */}
            <div className="flex gap-2">
              {quickBets.map((qb) => (
                <button
                  key={qb.label}
                  onClick={() => setRaiseAmount(qb.amount)}
                  className="px-3 py-1 bg-gray-700 hover:bg-gray-600 text-yellow-300 text-sm rounded
                             transition-colors"
                >
                  {qb.label}
                </button>
              ))}
            </div>

            {/* Slider */}
            <BetSlider
              min={minRaise}
              max={maxRaise}
              value={raiseAmount}
              onChange={setRaiseAmount}
              step={gameState.big_blind}
            />

            <button
              onClick={() => handleRaise(raiseAmount)}
              disabled={isLoading || raiseAmount < minRaise}
              className="w-full px-6 py-3 bg-yellow-600 hover:bg-yellow-700 text-white font-bold rounded-lg
                         disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              Raise to {formatCurrency(raiseAmount)}
            </button>
          </div>
        )}
      </div>
    </div>
  );
};
