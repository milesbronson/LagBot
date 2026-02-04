import React from 'react';
import { Player } from '../../types/game';
import { formatCurrency } from '../../utils/formatting';

interface PlayerInfoProps {
  player: Player;
  className?: string;
}

export const PlayerInfo: React.FC<PlayerInfoProps> = ({ player, className = '' }) => {
  return (
    <div
      className={`
        bg-gray-800 text-white rounded-lg p-3 min-w-[140px]
        border-2 ${player.is_active ? 'border-yellow-400' : 'border-gray-600'}
        ${className}
      `}
    >
      <div className="flex flex-col gap-1">
        <div className="flex items-center justify-between">
          <span className="font-bold text-sm">{player.name}</span>
          {player.is_dealer && (
            <span className="bg-yellow-500 text-black px-2 py-0.5 rounded-full text-xs font-bold">
              D
            </span>
          )}
        </div>

        <div className="text-sm">
          <span className="text-gray-400">Stack: </span>
          <span className="font-semibold">{formatCurrency(player.stack)}</span>
        </div>

        {player.bet > 0 && (
          <div className="text-sm">
            <span className="text-gray-400">Bet: </span>
            <span className="font-semibold text-yellow-400">
              {formatCurrency(player.bet)}
            </span>
          </div>
        )}

        {player.is_all_in && (
          <div className="text-xs text-red-400 font-bold">ALL-IN</div>
        )}

        {player.is_folded && (
          <div className="text-xs text-gray-500 font-bold">FOLDED</div>
        )}
      </div>
    </div>
  );
};
