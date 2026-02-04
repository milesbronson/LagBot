import React from 'react';
import { GameState } from '../../types/game';

interface HandHistoryProps {
  gameState: GameState;
  className?: string;
}

export const HandHistory: React.FC<HandHistoryProps> = ({ gameState, className = '' }) => {
  return (
    <div className={`bg-gray-800 rounded-lg p-4 ${className}`}>
      <h3 className="text-white font-bold text-lg mb-3">Hand History</h3>
      <div className="flex flex-col gap-2 text-sm text-gray-300">
        <div className="border-b border-gray-700 pb-2">
          <div>Hand #{gameState.hand_number}</div>
          <div>Round: {gameState.betting_round}</div>
        </div>
        <div className="text-xs text-gray-500">
          Action history will be displayed here as the hand progresses.
        </div>
      </div>
    </div>
  );
};
