import React, { useState } from 'react';
import { useGameStore } from '../../stores/gameStore';

interface NewGameModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const NewGameModal: React.FC<NewGameModalProps> = ({ isOpen, onClose }) => {
  const { createNewGame, isLoading } = useGameStore();
  const [numOpponents, setNumOpponents] = useState(2);
  const [opponentType, setOpponentType] = useState('trained');
  const [startingStack, setStartingStack] = useState(1000);
  const [smallBlind, setSmallBlind] = useState(5);
  const [bigBlind, setBigBlind] = useState(10);

  if (!isOpen) return null;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    await createNewGame({
      num_opponents: numOpponents,
      opponent_type: opponentType,
      starting_stack: startingStack,
      small_blind: smallBlind,
      big_blind: bigBlind,
    });
    onClose();
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-gray-800 rounded-lg p-8 max-w-md w-full">
        <h2 className="text-white text-2xl font-bold mb-6">New Game</h2>

        <form onSubmit={handleSubmit} className="flex flex-col gap-4">
          <div>
            <label className="text-white block mb-2">Number of Opponents</label>
            <input
              type="number"
              min="1"
              max="9"
              value={numOpponents}
              onChange={(e) => setNumOpponents(Number(e.target.value))}
              className="w-full px-3 py-2 bg-gray-700 text-white rounded"
            />
          </div>

          <div>
            <label className="text-white block mb-2">Opponent Type</label>
            <select
              value={opponentType}
              onChange={(e) => setOpponentType(e.target.value)}
              className="w-full px-3 py-2 bg-gray-700 text-white rounded"
            >
              <option value="trained">Trained AI</option>
              <option value="call">Call Bot</option>
              <option value="random">Random Bot</option>
              <option value="mixed">Mixed Bots</option>
            </select>
          </div>

          <div>
            <label className="text-white block mb-2">Starting Stack</label>
            <input
              type="number"
              min="100"
              value={startingStack}
              onChange={(e) => setStartingStack(Number(e.target.value))}
              className="w-full px-3 py-2 bg-gray-700 text-white rounded"
            />
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="text-white block mb-2">Small Blind</label>
              <input
                type="number"
                min="1"
                value={smallBlind}
                onChange={(e) => setSmallBlind(Number(e.target.value))}
                className="w-full px-3 py-2 bg-gray-700 text-white rounded"
              />
            </div>

            <div>
              <label className="text-white block mb-2">Big Blind</label>
              <input
                type="number"
                min="2"
                value={bigBlind}
                onChange={(e) => setBigBlind(Number(e.target.value))}
                className="w-full px-3 py-2 bg-gray-700 text-white rounded"
              />
            </div>
          </div>

          <div className="flex gap-3 mt-4">
            <button
              type="button"
              onClick={onClose}
              className="flex-1 px-4 py-2 bg-gray-600 hover:bg-gray-700 text-white rounded"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={isLoading}
              className="flex-1 px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded
                         disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Start Game
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};
