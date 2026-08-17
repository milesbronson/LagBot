export interface Action {
  action_type: number;
  raise_amount?: number;
}

export interface NewGameRequest {
  num_opponents: number;
  opponent_type: string;
  starting_stack: number;
  small_blind: number;
  big_blind: number;
}
