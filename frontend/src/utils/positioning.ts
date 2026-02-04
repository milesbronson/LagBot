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
  // Arrange seats in an ellipse around the table
  const centerX = tableWidth / 2;
  const centerY = tableHeight / 2;
  const radiusX = tableWidth * 0.4;
  const radiusY = tableHeight * 0.35;

  // Start from top and go clockwise
  const angle = (seatIndex / totalSeats) * 2 * Math.PI - Math.PI / 2;

  return {
    x: centerX + radiusX * Math.cos(angle),
    y: centerY + radiusY * Math.sin(angle),
  };
}
