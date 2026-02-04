export interface GameState {
  hand_number: number;
  betting_round: string;
  pot: number;
  current_bet: number;
  min_raise: number;
  community_cards: string[];
  players: Player[];
  current_player_idx: number;
  is_human_turn: boolean;
  valid_actions: number[];
  hand_complete: boolean;
  winner_info: WinnerInfo | null;
  small_blind: number;
  big_blind: number;
}

export interface Player {
  player_id: number;
  name: string;
  stack: number;
  bet: number;
  is_active: boolean;
  is_all_in: boolean;
  is_folded: boolean;
  hole_cards: string[] | null;
  is_human: boolean;
  is_dealer: boolean;
  is_small_blind: boolean;
  is_big_blind: boolean;
}

export interface WinnerInfo {
  [playerId: string]: number;
}

export interface OpponentStats {
  vpip: number;
  pfr: number;
  af: number;
  hands: number;
  confidence: number;
}

export interface HandHistoryEntry {
  hand_number: number;
  actions: ActionHistoryEntry[];
  winner: number;
  pot: number;
}

export interface ActionHistoryEntry {
  player_id: number;
  player_name: string;
  action: string;
  amount: number;
  street: string;
}
