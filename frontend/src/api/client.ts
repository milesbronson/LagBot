import axios from 'axios';
import { Action, NewGameRequest } from '../types/action';
import { GameState } from '../types/game';

const API_BASE = '/api';

const client = axios.create({
  baseURL: API_BASE,
  headers: {
    'Content-Type': 'application/json',
  },
});

export interface CreateGameResponse {
  session_id: string;
  state: GameState;
}

export async function createGame(request: NewGameRequest): Promise<CreateGameResponse> {
  const response = await client.post('/game/new', request);
  return response.data;
}

export async function submitAction(sessionId: string, action: Action): Promise<GameState> {
  const response = await client.post(`/game/${sessionId}/action`, action);
  return response.data;
}

export async function startNewHand(sessionId: string): Promise<GameState> {
  const response = await client.post(`/game/${sessionId}/new-hand`);
  return response.data;
}

export async function getGameState(sessionId: string): Promise<GameState> {
  const response = await client.get(`/game/${sessionId}/state`);
  return response.data;
}

export async function getOpponentStats(sessionId: string, playerId: number): Promise<any> {
  const response = await client.get(`/game/${sessionId}/opponent-stats/${playerId}`);
  return response.data;
}

export async function deleteGame(sessionId: string): Promise<void> {
  await client.delete(`/game/${sessionId}`);
}
