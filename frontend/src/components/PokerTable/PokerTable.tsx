import React from 'react';
import { GameState } from '../../types/game';
import { PokerTableSeat } from './PokerTableSeat';
import { CommunityCards } from './CommunityCards';
import { PotDisplay } from './PotDisplay';
import { calculateSeatPosition } from '../../utils/positioning';

interface PokerTableProps {
  gameState: GameState;
  className?: string;
}

export const PokerTable: React.FC<PokerTableProps> = ({ gameState, className = '' }) => {
  const tableWidth = 800;
  const tableHeight = 500;

  return (
    <div className={`relative ${className}`}>
      {/* Poker table felt */}
      <div
        className="bg-poker-felt rounded-full border-8 border-poker-rail shadow-2xl"
        style={{ width: `${tableWidth}px`, height: `${tableHeight}px` }}
      >
        {/* Center area with community cards and pot */}
        <div
          className="absolute flex flex-col items-center justify-center gap-4"
          style={{
            left: '50%',
            top: '50%',
            transform: 'translate(-50%, -50%)',
          }}
        >
          <CommunityCards cards={gameState.community_cards} />
          {gameState.pot > 0 && <PotDisplay amount={gameState.pot} />}
          <div className="text-white text-sm opacity-75">
            {gameState.betting_round}
          </div>
        </div>

        {/* Player seats */}
        {gameState.players.map((player, index) => {
          const position = calculateSeatPosition(
            index,
            gameState.players.length,
            tableWidth,
            tableHeight
          );

          return (
            <PokerTableSeat
              key={player.player_id}
              player={player}
              isCurrentPlayer={gameState.current_player_idx === player.player_id}
              position={position}
            />
          );
        })}
      </div>
    </div>
  );
};
