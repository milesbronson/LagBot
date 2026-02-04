import React from 'react';
import { Player } from '../../types/game';
import { PlayerStats } from '../Player/PlayerStats';

interface OpponentStatsProps {
  players: Player[];
  className?: string;
}

export const OpponentStats: React.FC<OpponentStatsProps> = ({ players, className = '' }) => {
  const opponents = players.filter((p) => !p.is_human);

  return (
    <div className={`bg-gray-800 rounded-lg p-4 ${className}`}>
      <h3 className="text-white font-bold text-lg mb-3">Opponent Stats</h3>
      <div className="flex flex-col gap-3">
        {opponents.map((opponent) => (
          <div key={opponent.player_id} className="border-b border-gray-700 pb-2">
            <div className="text-white font-semibold mb-1">{opponent.name}</div>
            <PlayerStats stats={null} />
            <div className="text-xs text-gray-500 mt-1">
              Stats will accumulate as you play more hands
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
