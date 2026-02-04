export function formatCurrency(amount: number): string {
  return `$${amount.toLocaleString()}`;
}

export function formatPercentage(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

export function formatDecimal(value: number, decimals: number = 2): string {
  return value.toFixed(decimals);
}
