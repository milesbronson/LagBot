export interface Position {
  x: number;
  y: number;
}

export function calculateSeatPosition(
  seatIndex: number,
  totalSeats: number,
  tableWidth: number,
  tableHeight: number
): Position {
  const centerX = tableWidth / 2;
  const centerY = tableHeight / 2;

  if (totalSeats === 2) {
    const positions: Position[] = [
      { x: centerX, y: tableHeight - 30 },
      { x: centerX, y: 30 },
    ];
    return positions[seatIndex];
  }

  const radiusX = tableWidth * 0.42;
  const radiusY = tableHeight * 0.38;

  // Start from bottom (human seat) and go clockwise
  const angle = (seatIndex / totalSeats) * 2 * Math.PI + Math.PI / 2;

  return {
    x: centerX + radiusX * Math.cos(angle),
    y: centerY + radiusY * Math.sin(angle),
  };
}

export function calculateBetPosition(
  seatPosition: Position,
  tableWidth: number,
  tableHeight: number
): Position {
  const centerX = tableWidth / 2;
  const centerY = tableHeight / 2;

  return {
    x: seatPosition.x + (centerX - seatPosition.x) * 0.45,
    y: seatPosition.y + (centerY - seatPosition.y) * 0.45,
  };
}
