import { create } from 'zustand';
import { GameState, ActionHistoryEntry, OpponentStats } from '../types/game';
import { Action, NewGameRequest } from '../types/action';
import { createGame, submitAction, startNewHand, getOpponentStats } from '../api/client';

interface LastAction {
  player_id: number;
  player_name: string;
  action: string;
  amount: number;
}

interface GameStore {
  gameState: GameState | null;
  sessionId: string | null;
  isConnected: boolean;
  isLoading: boolean;
  error: string | null;
  currentHandActions: ActionHistoryEntry[];
  lastAction: LastAction | null;
  opponentStats: Record<number, OpponentStats>;

  setGameState: (state: GameState) => void;
  setSessionId: (id: string) => void;
  setConnected: (connected: boolean) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  addHandAction: (action: ActionHistoryEntry) => void;
  clearHandActions: () => void;
  setLastAction: (action: LastAction | null) => void;

  createNewGame: (request: NewGameRequest) => Promise<void>;
  submitPlayerAction: (action: Action) => Promise<void>;
  startNextHand: () => Promise<void>;
  fetchOpponentStats: () => Promise<void>;
  resetGame: () => void;
}

export const useGameStore = create<GameStore>((set, get) => ({
  gameState: null,
  sessionId: null,
  isConnected: false,
  isLoading: false,
  error: null,
  currentHandActions: [],
  lastAction: null,
  opponentStats: {},

  setGameState: (state) => set({ gameState: state }),
  setSessionId: (id) => set({ sessionId: id }),
  setConnected: (connected) => set({ isConnected: connected }),
  setLoading: (loading) => set({ isLoading: loading }),
  setError: (error) => set({ error }),
  addHandAction: (action) =>
    set((state) => ({ currentHandActions: [...state.currentHandActions, action] })),
  clearHandActions: () => set({ currentHandActions: [] }),
  setLastAction: (action) => set({ lastAction: action }),

  createNewGame: async (request: NewGameRequest) => {
    try {
      set({ isLoading: true, error: null });
      const response = await createGame(request);
      set({
        sessionId: response.session_id,
        gameState: response.state,
        isLoading: false,
        currentHandActions: [],
        opponentStats: {},
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

      if (state.hand_complete) {
        get().fetchOpponentStats();
      }
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
      set({ isLoading: true, error: null, currentHandActions: [], lastAction: null });
      const state = await startNewHand(sessionId);
      set({ gameState: state, isLoading: false });
    } catch (error: any) {
      set({
        error: error.message || 'Failed to start new hand',
        isLoading: false,
      });
    }
  },

  fetchOpponentStats: async () => {
    const { sessionId, gameState } = get();
    if (!sessionId || !gameState) return;

    const stats: Record<number, OpponentStats> = {};
    for (const player of gameState.players) {
      if (player.is_human) continue;
      try {
        const data = await getOpponentStats(sessionId, player.player_id);
        if (data && data.hands_played > 0) {
          stats[player.player_id] = {
            vpip: data.vpip,
            pfr: data.pfr,
            af: data.af,
            hands: data.hands_played,
            confidence: data.confidence,
          };
        }
      } catch {
        // Stats not available yet
      }
    }
    set({ opponentStats: stats });
  },

  resetGame: () => {
    set({
      gameState: null,
      sessionId: null,
      isConnected: false,
      isLoading: false,
      error: null,
      currentHandActions: [],
      lastAction: null,
      opponentStats: {},
    });
  },
}));
