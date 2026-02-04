import React from 'react';

interface CardBackProps {
  className?: string;
}

export const CardBack: React.FC<CardBackProps> = ({ className = '' }) => {
  return (
    <div
      className={`
        bg-gradient-to-br from-blue-800 to-blue-600
        rounded-lg shadow-lg border-2 border-gray-300
        w-16 h-24 flex items-center justify-center
        ${className}
      `}
    >
      <div className="text-white text-sm font-bold opacity-50">
        🂠
      </div>
    </div>
  );
};
