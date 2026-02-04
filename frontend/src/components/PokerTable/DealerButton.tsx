import React from 'react';

interface DealerButtonProps {
  className?: string;
}

export const DealerButton: React.FC<DealerButtonProps> = ({ className = '' }) => {
  return (
    <div
      className={`
        bg-yellow-500 text-black rounded-full w-10 h-10
        flex items-center justify-center font-bold text-lg
        border-2 border-yellow-600 shadow-lg
        ${className}
      `}
    >
      D
    </div>
  );
};
