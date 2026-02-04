import { create } from 'zustand';
import { GameState } from '../types/game';
import { Action, NewGameRequest } from '../types/action';
import { createGame, submitAction, startNewHand } from '../api/client';

interface GameStore {
  gameState: GameState | null;
  sessionId: string | null;
  isConnected: boolean;
  isLoading: boolean;
  error: string | null;

  setGameState: (state: GameState) => void;
  setSessionId: (id: string) => void;
  setConnected: (connected: boolean) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;

  createNewGame: (request: NewGameRequest) => Promise<void>;
  submitPlayerAction: (action: Action) => Promise<void>;
  startNextHand: () => Promise<void>;
  resetGame: () => void;
}

export const useGameStore = create<GameStore>((set, get) => ({
  gameState: null,
  sessionId: null,
  isConnected: false,
  isLoading: false,
  error: null,

  setGameState: (state) => set({ gameState: state }),
  setSessionId: (id) => set({ sessionId: id }),
  setConnected: (connected) => set({ isConnected: connected }),
  setLoading: (loading) => set({ isLoading: loading }),
  setError: (error) => set({ error }),

  createNewGame: async (request: NewGameRequest) => {
    try {
      set({ isLoading: true, error: null });
      const response = await createGame(request);
      set({
        sessionId: response.session_id,
        gameState: response.state,
        isLoading: false,
      });
    } catch (error: any) {
      set({
        error: error.message || 'Failed to create game',
        isLoading: false,
      });
    }
  },

  submitPlayerAction: async (action: Action) => {
    const { sessionId } = get();
    if (!sessionId) {
      set({ error: 'No active session' });
      return;
    }

    try {
      set({ isLoading: true, error: null });
      const state = await submitAction(sessionId, action);
      set({ gameState: state, isLoading: false });
    } catch (error: any) {
      set({
        error: error.message || 'Failed to submit action',
        isLoading: false,
      });
    }
  },

  startNextHand: async () => {
    const { sessionId } = get();
    if (!sessionId) {
      set({ error: 'No active session' });
      return;
    }

    try {
      set({ isLoading: true, error: null });
      const state = await startNewHand(sessionId);
      set({ gameState: state, isLoading: false });
    } catch (error: any) {
      set({
        error: error.message || 'Failed to start new hand',
        isLoading: false,
      });
    }
  },

  resetGame: () => {
    set({
      gameState: null,
      sessionId: null,
      isConnected: false,
      isLoading: false,
      error: null,
    });
  },
}));
