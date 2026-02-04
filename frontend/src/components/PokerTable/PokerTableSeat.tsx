import React from 'react';
import { Player } from '../../types/game';
import { PlayerInfo } from '../Player/PlayerInfo';
import { PlayerActionIndicator } from '../Player/PlayerActionIndicator';
import { HoleCards } from '../Cards/HoleCards';
import { DealerButton } from './DealerButton';

interface PokerTableSeatProps {
  player: Player;
  isCurrentPlayer: boolean;
  position: { x: number; y: number };
  className?: string;
}

export const PokerTableSeat: React.FC<PokerTableSeatProps> = ({
  player,
  isCurrentPlayer,
  position,
  className = '',
}) => {
  return (
    <div
      className={`absolute ${className}`}
      style={{
        left: `${position.x}px`,
        top: `${position.y}px`,
        transform: 'translate(-50%, -50%)',
      }}
    >
      <div className="flex flex-col items-center gap-2">
        {player.is_dealer && <DealerButton />}

        <HoleCards
          cards={player.hole_cards}
          showCards={player.is_human || !player.is_active}
        />

        <PlayerInfo player={player} />

        <PlayerActionIndicator
          player={player}
          isCurrentPlayer={isCurrentPlayer}
        />
      </div>
    </div>
  );
};
