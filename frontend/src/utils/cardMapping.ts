// Map card strings (e.g., "Ah", "Kd") to display values

export interface CardDisplay {
  rank: string;
  suit: string;
  color: string;
  symbol: string;
}

const rankMap: { [key: string]: string } = {
  'A': 'A',
  'K': 'K',
  'Q': 'Q',
  'J': 'J',
  'T': '10',
  '9': '9',
  '8': '8',
  '7': '7',
  '6': '6',
  '5': '5',
  '4': '4',
  '3': '3',
  '2': '2',
};

const suitMap: { [key: string]: { name: string; symbol: string; color: string } } = {
  'h': { name: 'hearts', symbol: '♥', color: 'text-red-600' },
  'd': { name: 'diamonds', symbol: '♦', color: 'text-red-600' },
  'c': { name: 'clubs', symbol: '♣', color: 'text-gray-800' },
  's': { name: 'spades', symbol: '♠', color: 'text-gray-800' },
};

export function parseCard(cardString: string): CardDisplay | null {
  if (!cardString || cardString.length < 2) return null;

  const rankChar = cardString[0];
  const suitChar = cardString[1].toLowerCase();

  const rank = rankMap[rankChar];
  const suitInfo = suitMap[suitChar];

  if (!rank || !suitInfo) return null;

  return {
    rank,
    suit: suitInfo.name,
    color: suitInfo.color,
    symbol: suitInfo.symbol,
  };
}

export function getCardImage(cardString: string): string {
  const card = parseCard(cardString);
  if (!card) return '/cards/back.svg';

  // For now, we'll use a text-based representation
  // In a real app, you'd have SVG card images
  return `/cards/${card.rank.toLowerCase()}_${card.suit}.svg`;
}
