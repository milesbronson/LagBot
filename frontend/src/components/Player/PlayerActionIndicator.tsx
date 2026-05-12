import React from 'react';
import { Player } from '../../types/game';
import { useGameStore } from '../../stores/gameStore';
import { formatCurrency } from '../../utils/formatting';

interface PlayerActionIndicatorProps {
  player: Player;
  isCurrentPlayer: boolean;
  className?: string;
}

const actionBadgeStyle: Record<string, string> = {
  fold: 'bg-red-600',
  check: 'bg-gray-600',
  call: 'bg-green-600',
  raise: 'bg-yellow-600',
  bet: 'bg-yellow-600',
  'all-in': 'bg-red-700',
};

export const PlayerActionIndicator: React.FC<PlayerActionIndicatorProps> = ({
  player,
  isCurrentPlayer,
  className = '',
}) => {
  const { lastAction } = useGameStore();

  if (lastAction && lastAction.player_id === player.player_id) {
    const style = actionBadgeStyle[lastAction.action.toLowerCase()] || 'bg-gray-600';
    return (
      <div className={`px-2 py-1 rounded ${style} text-white text-xs font-bold animate-bounce ${className}`}>
        {lastAction.action}
        {lastAction.amount > 0 && ` ${formatCurrency(lastAction.amount)}`}
      </div>
    );
  }

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
