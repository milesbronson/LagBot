import React from 'react';
import { formatCurrency } from '../../utils/formatting';

interface PotDisplayProps {
  amount: number;
  className?: string;
}

export const PotDisplay: React.FC<PotDisplayProps> = ({ amount, className = '' }) => {
  return (
    <div
      className={`
        bg-yellow-500 text-black rounded-full px-6 py-3
        font-bold text-lg shadow-lg border-4 border-yellow-600
        ${className}
      `}
    >
      Pot: {formatCurrency(amount)}
    </div>
  );
};
