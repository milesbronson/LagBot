export interface Action {
  action_type: number;
  raise_amount?: number;
}

export enum ActionType {
  FOLD = 0,
  CALL = 1,
  RAISE = 2,
  ALL_IN = 99,
}

export interface NewGameRequest {
  num_opponents: number;
  opponent_type: string;
  starting_stack: number;
  small_blind: number;
  big_blind: number;
}
