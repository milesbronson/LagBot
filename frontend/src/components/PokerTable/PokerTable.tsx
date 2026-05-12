import React from 'react';
import { GameState } from '../../types/game';
import { PokerTableSeat } from './PokerTableSeat';
import { CommunityCards } from './CommunityCards';
import { PotDisplay } from './PotDisplay';
import { calculateSeatPosition, calculateBetPosition } from '../../utils/positioning';
import { formatCurrency } from '../../utils/formatting';

interface PokerTableProps {
  gameState: GameState;
  className?: string;
}

export const PokerTable: React.FC<PokerTableProps> = ({ gameState, className = '' }) => {
  const playerCount = gameState.players.length;
  const compact = playerCount > 6;

  const tableWidth = compact ? 900 : 800;
  const tableHeight = compact ? 560 : 500;

  return (
    <div className={`relative ${className}`} style={{ width: `${tableWidth}px`, height: `${tableHeight}px` }}>
      {/* Poker table felt */}
      <div
        className="absolute inset-0 bg-poker-felt rounded-full border-8 border-poker-rail shadow-2xl"
      />

      {/* Center area */}
      <div
        className="absolute flex flex-col items-center justify-center gap-3"
        style={{
          left: '50%',
          top: '45%',
          transform: 'translate(-50%, -50%)',
        }}
      >
        <CommunityCards cards={gameState.community_cards} />
        {gameState.pot > 0 && <PotDisplay amount={gameState.pot} />}
        <div className="text-white text-xs opacity-60">
          {gameState.betting_round}
        </div>
      </div>

      {/* Bet chips */}
      {gameState.players.map((player, index) => {
        if (player.bet <= 0) return null;
        const seatPos = calculateSeatPosition(index, playerCount, tableWidth, tableHeight);
        const betPos = calculateBetPosition(seatPos, tableWidth, tableHeight);
        return (
          <div
            key={`bet-${player.player_id}`}
            className="absolute z-10 flex items-center gap-1"
            style={{
              left: `${betPos.x}px`,
              top: `${betPos.y}px`,
              transform: 'translate(-50%, -50%)',
            }}
          >
            <div className="w-4 h-4 rounded-full bg-poker-chip border border-yellow-600 shadow" />
            <span className="text-yellow-300 text-xs font-bold whitespace-nowrap">
              {formatCurrency(player.bet)}
            </span>
          </div>
        );
      })}

      {/* Player seats */}
      {gameState.players.map((player, index) => {
        const position = calculateSeatPosition(index, playerCount, tableWidth, tableHeight);
        return (
          <PokerTableSeat
            key={player.player_id}
            player={player}
            isCurrentPlayer={gameState.current_player_idx === player.player_id}
            position={position}
            compact={compact}
          />
        );
      })}
    </div>
  );
};
