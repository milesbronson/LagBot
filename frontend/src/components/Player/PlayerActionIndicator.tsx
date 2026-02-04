import React from 'react';
import { Player } from '../../types/game';

interface PlayerActionIndicatorProps {
  player: Player;
  isCurrentPlayer: boolean;
  className?: string;
}

export const PlayerActionIndicator: React.FC<PlayerActionIndicatorProps> = ({
  player,
  isCurrentPlayer,
  className = '',
}) => {
  if (player.is_folded) {
    return (
      <div className={`px-2 py-1 rounded bg-gray-600 text-white text-xs ${className}`}>
        Folded
      </div>
    );
  }

  if (player.is_all_in) {
    return (
      <div className={`px-2 py-1 rounded bg-red-600 text-white text-xs font-bold ${className}`}>
        All-In
      </div>
    );
  }

  if (isCurrentPlayer) {
    return (
      <div className={`px-2 py-1 rounded bg-yellow-500 text-black text-xs font-bold animate-pulse ${className}`}>
        Thinking...
      </div>
    );
  }

  return null;
};
