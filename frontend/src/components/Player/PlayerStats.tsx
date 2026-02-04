import React from 'react';
import { OpponentStats } from '../../types/game';
import { formatPercentage, formatDecimal } from '../../utils/formatting';

interface PlayerStatsProps {
  stats: OpponentStats | null;
  className?: string;
}

export const PlayerStats: React.FC<PlayerStatsProps> = ({ stats, className = '' }) => {
  if (!stats || stats.hands === 0) {
    return null;
  }

  return (
    <div className={`text-xs text-gray-400 ${className}`}>
      <div className="flex gap-2">
        <span title="Voluntarily Put money In Pot">
          VPIP: {formatPercentage(stats.vpip)}
        </span>
        <span title="Pre-Flop Raise">
          PFR: {formatPercentage(stats.pfr)}
        </span>
        <span title="Aggression Factor">
          AF: {formatDecimal(stats.af, 1)}
        </span>
      </div>
      <div className="text-xs opacity-70">
        {stats.hands} hands
      </div>
    </div>
  );
};
