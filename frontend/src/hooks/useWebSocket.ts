import { useEffect, useRef, useCallback } from 'react';
import { useGameStore } from '../stores/gameStore';

const MAX_RETRIES = 5;
const BASE_DELAY = 1000;

export function useWebSocket(sessionId: string | null) {
  const ws = useRef<WebSocket | null>(null);
  const retryCount = useRef(0);
  const retryTimer = useRef<ReturnType<typeof setTimeout>>();
  const { setGameState, setConnected, setError, addHandAction, setLastAction } = useGameStore();

  const connect = useCallback(() => {
    if (!sessionId) return;

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/${sessionId}`;

    ws.current = new WebSocket(wsUrl);

    ws.current.onopen = () => {
      retryCount.current = 0;
      setConnected(true);
    };

    ws.current.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        if (data.type === 'connected') {
          setGameState(data.state);
        } else if (data.type === 'state_update') {
          const currentState = useGameStore.getState().gameState;
          if (currentState?.hand_complete && !data.state.hand_complete) {
            return;
          }
          setGameState(data.state);

          if (data.last_action) {
            const la = data.last_action;
            setLastAction(la);
            addHandAction({
              player_id: la.player_id,
              player_name: la.player_name,
              action: la.action,
              amount: la.amount,
              street: data.state.betting_round,
            });
            setTimeout(() => setLastAction(null), 2000);
          }
        } else if (data.type === 'error') {
          setError(data.message);
        }
      } catch {
        // ignore parse errors
      }
    };

    ws.current.onerror = () => {
      setConnected(false);
    };

    ws.current.onclose = () => {
      setConnected(false);
      if (retryCount.current < MAX_RETRIES) {
        const delay = BASE_DELAY * Math.pow(2, retryCount.current);
        retryCount.current++;
        retryTimer.current = setTimeout(connect, delay);
      }
    };
  }, [sessionId, setGameState, setConnected, setError, addHandAction, setLastAction]);

  useEffect(() => {
    connect();
    return () => {
      if (retryTimer.current) clearTimeout(retryTimer.current);
      if (ws.current) ws.current.close();
    };
  }, [connect]);

  return ws.current;
}
