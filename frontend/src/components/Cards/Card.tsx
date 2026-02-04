import React from 'react';
import { parseCard } from '../../utils/cardMapping';

interface CardProps {
  cardString: string;
  className?: string;
}

export const Card: React.FC<CardProps> = ({ cardString, className = '' }) => {
  const card = parseCard(cardString);

  if (!card) {
    return null;
  }

  return (
    <div
      className={`
        bg-white rounded-lg shadow-lg border-2 border-gray-300
        w-16 h-24 flex flex-col items-center justify-between
        p-2 animate-deal-card
        ${className}
      `}
    >
      <div className={`text-2xl font-bold ${card.color}`}>
        {card.rank}
      </div>
      <div className={`text-4xl ${card.color}`}>
        {card.symbol}
      </div>
      <div className={`text-2xl font-bold ${card.color}`}>
        {card.rank}
      </div>
    </div>
  );
};
