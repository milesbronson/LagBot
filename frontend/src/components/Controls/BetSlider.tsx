import React from 'react';
import { formatCurrency } from '../../utils/formatting';

interface BetSliderProps {
  min: number;
  max: number;
  value: number;
  onChange: (value: number) => void;
  step?: number;
  className?: string;
}

export const BetSlider: React.FC<BetSliderProps> = ({
  min,
  max,
  value,
  onChange,
  step = 1,
  className = '',
}) => {
  return (
    <div className={`flex flex-col gap-2 ${className}`}>
      <div className="flex justify-between text-sm text-gray-300">
        <span>{formatCurrency(min)}</span>
        <span className="font-bold text-yellow-400">{formatCurrency(value)}</span>
        <span>{formatCurrency(max)}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        value={value}
        step={step}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-yellow-500"
      />
    </div>
  );
};
