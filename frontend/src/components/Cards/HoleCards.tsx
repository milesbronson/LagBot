import React from 'react';
import { Card } from './Card';
import { CardBack } from './CardBack';

interface HoleCardsProps {
  cards: string[] | null;
  showCards: boolean;
  className?: string;
}

export const HoleCards: React.FC<HoleCardsProps> = ({
  cards,
  showCards,
  className = '',
}) => {
  if (!cards || cards.length !== 2) {
    return (
      <div className={`flex gap-1 ${className}`}>
        <CardBack />
        <CardBack />
      </div>
    );
  }

  if (!showCards) {
    return (
      <div className={`flex gap-1 ${className}`}>
        <CardBack />
        <CardBack />
      </div>
    );
  }

  return (
    <div className={`flex gap-1 ${className}`}>
      <Card cardString={cards[0]} />
      <Card cardString={cards[1]} />
    </div>
  );
};
