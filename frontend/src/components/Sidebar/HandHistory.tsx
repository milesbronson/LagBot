import React, { useRef, useEffect } from 'react';
import { GameState } from '../../types/game';
import { useGameStore } from '../../stores/gameStore';
import { formatCurrency } from '../../utils/formatting';

interface HandHistoryProps {
  gameState: GameState;
  className?: string;
}

const actionColor: Record<string, string> = {
  fold: 'text-red-400',
  check: 'text-gray-300',
  call: 'text-green-400',
  raise: 'text-yellow-400',
  'all-in': 'text-red-300 font-bold',
  bet: 'text-yellow-400',
};

export const HandHistory: React.FC<HandHistoryProps> = ({ gameState, className = '' }) => {
  const { currentHandActions } = useGameStore();
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [currentHandActions.length]);

  return (
    <div className={`bg-gray-800 rounded-lg p-4 ${className}`}>
      <h3 className="text-white font-bold text-lg mb-3">
        Hand #{gameState.hand_number}
      </h3>
      <div
        ref={scrollRef}
        className="flex flex-col gap-1 text-sm max-h-48 overflow-y-auto"
      >
        {currentHandActions.length === 0 ? (
          <div className="text-xs text-gray-500">
            Waiting for actions...
          </div>
        ) : (
          currentHandActions.map((entry, i) => {
            const color = actionColor[entry.action.toLowerCase()] || 'text-gray-300';
            return (
              <div key={i} className={`${color}`}>
                <span className="font-semibold">{entry.player_name}</span>
                {' '}
                {entry.action}
                {entry.amount > 0 && ` ${formatCurrency(entry.amount)}`}
              </div>
            );
          })
        )}
      </div>
    </div>
  );
};
