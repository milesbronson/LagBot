import { useEffect, useRef } from 'react';
import { useGameStore } from '../stores/gameStore';

export function useWebSocket(sessionId: string | null) {
  const ws = useRef<WebSocket | null>(null);
  const { setGameState, setConnected, setError } = useGameStore();

  useEffect(() => {
    if (!sessionId) return;

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/${sessionId}`;

    ws.current = new WebSocket(wsUrl);

    ws.current.onopen = () => {
      console.log('WebSocket connected');
      setConnected(true);
    };

    ws.current.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        if (data.type === 'connected') {
          console.log('Initial state received');
          setGameState(data.state);
        } else if (data.type === 'state_update') {
          console.log('State update received');
          setGameState(data.state);
        } else if (data.type === 'error') {
          console.error('WebSocket error:', data.message);
          setError(data.message);
        }
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error);
      }
    };

    ws.current.onerror = (error) => {
      console.error('WebSocket error:', error);
      setConnected(false);
      setError('WebSocket connection error');
    };

    ws.current.onclose = () => {
      console.log('WebSocket disconnected');
      setConnected(false);
    };

    return () => {
      if (ws.current) {
        ws.current.close();
      }
    };
  }, [sessionId, setGameState, setConnected, setError]);

  return ws.current;
}
