import React from 'react';
import { Card } from '../Cards/Card';

interface CommunityCardsProps {
  cards: string[];
  className?: string;
}

export const CommunityCards: React.FC<CommunityCardsProps> = ({
  cards,
  className = '',
}) => {
  return (
    <div className={`flex gap-2 justify-center ${className}`}>
      {cards.length === 0 ? (
        <div className="text-white text-sm opacity-50">No community cards</div>
      ) : (
        cards.map((card, index) => (
          <Card key={`${card}-${index}`} cardString={card} />
        ))
      )}
    </div>
  );
};
