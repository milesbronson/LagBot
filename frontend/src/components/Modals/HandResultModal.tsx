import React from 'react';
import { GameState } from '../../types/game';
import { formatCurrency } from '../../utils/formatting';
import { useGameStore } from '../../stores/gameStore';

interface HandResultModalProps {
  gameState: GameState;
  onNextHand: () => void;
}

export const HandResultModal: React.FC<HandResultModalProps> = ({
  gameState,
  onNextHand,
}) => {
  if (!gameState.hand_complete || !gameState.winner_info) {
    return null;
  }

  const winners = Object.entries(gameState.winner_info)
    .filter(([_, amount]) => amount > 0)
    .map(([playerId, amount]) => {
      const player = gameState.players.find((p) => p.player_id === Number(playerId));
      return { player, amount };
    });

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-gray-800 rounded-lg p-8 max-w-md w-full">
        <h2 className="text-white text-2xl font-bold mb-6 text-center">
          Hand Complete
        </h2>

        <div className="flex flex-col gap-4 mb-6">
          <div className="text-white text-center text-lg">
            Pot: {formatCurrency(gameState.pot)}
          </div>

          {winners.map(({ player, amount }) => (
            <div
              key={player?.player_id}
              className="bg-gray-700 rounded p-4 text-white"
            >
              <div className="font-bold text-lg">{player?.name}</div>
              <div className="text-green-400">
                Won {formatCurrency(amount)}
              </div>
              {player?.hole_cards && (
                <div className="text-sm mt-2">
                  Cards: {player.hole_cards.join(', ')}
                </div>
              )}
            </div>
          ))}
        </div>

        <button
          onClick={onNextHand}
          className="w-full px-6 py-3 bg-green-600 hover:bg-green-700 text-white font-bold rounded-lg"
        >
          Next Hand
        </button>
      </div>
    </div>
  );
};
