# LagBot Web Interface Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         Browser (Client)                         │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                    React Frontend                           │ │
│  │                                                              │ │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │ │
│  │  │   App    │  │  Poker   │  │  Action  │  │  Modals  │   │ │
│  │  │Component │  │  Table   │  │  Panel   │  │          │   │ │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │ │
│  │         │             │             │             │         │ │
│  │         └─────────────┴─────────────┴─────────────┘         │ │
│  │                          │                                   │ │
│  │                  ┌───────▼────────┐                          │ │
│  │                  │ Zustand Store  │                          │ │
│  │                  └───────┬────────┘                          │ │
│  │                          │                                   │ │
│  │           ┌──────────────┴──────────────┐                   │ │
│  │           │                              │                   │ │
│  │    ┌──────▼──────┐              ┌───────▼────────┐          │ │
│  │    │ HTTP Client │              │ WebSocket Hook │          │ │
│  │    │   (Axios)   │              │                │          │ │
│  │    └──────┬──────┘              └───────┬────────┘          │ │
│  └───────────┼─────────────────────────────┼──────────────────┘ │
└──────────────┼─────────────────────────────┼────────────────────┘
               │                              │
        REST API Calls                 WebSocket Connection
               │                              │
┌──────────────▼──────────────────────────────▼────────────────────┐
│                      FastAPI Backend                              │
│                                                                    │
│  ┌──────────────┐                          ┌──────────────┐      │
│  │  REST Routes │                          │  WebSocket   │      │
│  │              │                          │   Handler    │      │
│  └──────┬───────┘                          └──────┬───────┘      │
│         │                                          │              │
│         └──────────────┬───────────────────────────┘              │
│                        │                                          │
│                 ┌──────▼──────┐                                   │
│                 │ Game Manager│                                   │
│                 │  (Singleton)│                                   │
│                 └──────┬──────┘                                   │
│                        │                                          │
│               ┌────────▼────────┐                                 │
│               │  Game Sessions  │                                 │
│               │  (Dictionary)   │                                 │
│               └────────┬────────┘                                 │
│                        │                                          │
│                 ┌──────▼──────┐                                   │
│                 │ GameSession │                                   │
│                 │   Instance  │                                   │
│                 └──────┬──────┘                                   │
│                        │                                          │
│         ┌──────────────┼──────────────┐                           │
│         │              │               │                          │
│  ┌──────▼──────┐ ┌────▼─────┐  ┌─────▼──────┐                   │
│  │TexasHoldem  │ │   Bot    │  │   State    │                   │
│  │     Env     │ │  Agents  │  │Serializer  │                   │
│  └─────────────┘ └──────────┘  └────────────┘                   │
└───────────────────────────────────────────────────────────────────┘
```

## Component Interaction Flow

### 1. Game Creation Flow

```
User clicks "New Game"
    │
    ▼
NewGameModal appears
    │
    ▼
User configures game settings
    │
    ▼
Submit to gameStore.createNewGame()
    │
    ▼
POST /api/game/new
    │
    ▼
GameManager.create_session()
    │
    ▼
GameSession initialized with TexasHoldemEnv
    │
    ▼
Load bot agents (PPO/Call/Random)
    │
    ▼
Start first hand (reset env)
    │
    ▼
Serialize game state to JSON
    │
    ▼
Return session_id + initial state
    │
    ▼
Frontend stores session_id
    │
    ▼
WebSocket connects to /ws/{session_id}
    │
    ▼
Game ready to play!
```

### 2. Action Execution Flow

```
User's turn → Action Panel shows
    │
    ▼
User clicks "Raise $100"
    │
    ▼
gameStore.submitPlayerAction({action_type: 2, raise_amount: 100})
    │
    ▼
POST /api/game/{session_id}/action
    │
    ▼
GameSession.execute_human_action()
    │
    ▼
env.step(action) → Execute human action
    │
    ▼
Broadcast state via WebSocket
    │
    ▼
While current_player != human:
    │
    ├─ Wait 500ms (bot "thinking")
    │
    ├─ Get bot action from agent
    │
    ├─ env.step(bot_action)
    │
    ├─ Broadcast state via WebSocket
    │
    └─ Check if betting round complete
    │
    ▼
Return final state
    │
    ▼
Frontend receives WebSocket update
    │
    ▼
Zustand store updates gameState
    │
    ▼
React components re-render
    │
    ▼
UI shows updated game state
```

### 3. WebSocket State Update Flow

```
Backend sends state update
    │
    ▼
WebSocket.onmessage event
    │
    ▼
Parse JSON data
    │
    ▼
Check message type:
    ├─ "connected" → Initial state
    ├─ "state_update" → Game state change
    └─ "error" → Error message
    │
    ▼
gameStore.setGameState(newState)
    │
    ▼
All components using gameState re-render:
    ├─ PokerTable updates player positions
    ├─ CommunityCards shows new cards
    ├─ PotDisplay updates pot amount
    ├─ PlayerInfo updates stacks/bets
    └─ ActionPanel enables/disables controls
```

## Data Flow Diagram

### Frontend State Management (Zustand)

```
┌────────────────────────────────────────┐
│          Zustand Store                 │
├────────────────────────────────────────┤
│ State:                                 │
│  - gameState: GameState | null         │
│  - sessionId: string | null            │
│  - isConnected: boolean                │
│  - isLoading: boolean                  │
│  - error: string | null                │
│                                        │
│ Actions:                               │
│  - setGameState()                      │
│  - createNewGame()                     │
│  - submitPlayerAction()                │
│  - startNextHand()                     │
│  - resetGame()                         │
└────────────────────────────────────────┘
         │
         │ Subscribe
         ▼
┌────────────────────┐
│  React Components  │
│  (Auto re-render)  │
└────────────────────┘
```

### Backend Session Management

```
┌────────────────────────────────────────┐
│         GameManager (Singleton)        │
├────────────────────────────────────────┤
│ sessions: Dict[session_id, GameSession]│
│                                        │
│ Methods:                               │
│  - create_session()                    │
│  - get_session()                       │
│  - delete_session()                    │
└────────────────────────────────────────┘
         │
         │ Manages
         ▼
┌────────────────────────────────────────┐
│           GameSession                  │
├────────────────────────────────────────┤
│ - env: TexasHoldemEnv                  │
│ - bot_agents: List[Agent]              │
│ - websocket_connections: List[WS]     │
│                                        │
│ Methods:                               │
│  - start_hand()                        │
│  - execute_human_action()              │
│  - _broadcast_current_state()          │
└────────────────────────────────────────┘
         │
         │ Wraps
         ▼
┌────────────────────────────────────────┐
│        TexasHoldemEnv                  │
│    (Existing poker environment)        │
└────────────────────────────────────────┘
```

## Technology Stack Details

### Backend Stack

```
FastAPI (Web Framework)
    │
    ├─ Uvicorn (ASGI Server)
    ├─ Pydantic (Data Validation)
    ├─ WebSockets (Real-time Communication)
    │
    └─ Integrates with existing:
        ├─ TexasHoldemEnv (Poker Engine)
        ├─ PPO Agents (AI Opponents)
        ├─ HandEvaluator (Card Logic)
        └─ OpponentTracker (Statistics)
```

### Frontend Stack

```
React 18 (UI Library)
    │
    ├─ TypeScript (Type Safety)
    ├─ Vite (Build Tool)
    ├─ Zustand (State Management)
    ├─ Axios (HTTP Client)
    ├─ Tailwind CSS (Styling)
    └─ React Toastify (Notifications)
```

## Network Communication

### REST API (Request/Response)

Used for:
- Creating new games
- Submitting player actions
- Starting new hands
- Getting game state snapshots
- Managing sessions

**Advantages**: Reliable, stateless, easy to test

### WebSocket (Bi-directional)

Used for:
- Real-time game state updates
- Bot action notifications
- Connection status

**Advantages**: Low latency, push notifications, real-time

### Why Hybrid Approach?

1. **REST for commands**: User actions are intentional and require acknowledgment
2. **WebSocket for updates**: Game state changes need immediate broadcast to all clients
3. **Best of both**: Reliability of REST + Real-time of WebSocket

## Deployment Architecture

### Development (Manual)

```
Terminal 1:                    Terminal 2:
┌──────────────────┐          ┌──────────────────┐
│ Backend          │          │ Frontend         │
│ Port: 8000       │◄────────►│ Port: 5173       │
│                  │  Proxy   │                  │
│ uvicorn --reload │          │ vite dev server  │
└──────────────────┘          └──────────────────┘
```

### Docker Compose

```
┌─────────────────────────────────────────┐
│        Docker Compose Network            │
│                                          │
│  ┌────────────────┐  ┌────────────────┐ │
│  │ backend:8000   │  │ frontend:5173  │ │
│  │ (Python)       │◄─┤ (Node.js)      │ │
│  └────────────────┘  └────────────────┘ │
│                                          │
│  Volumes:                                │
│  - ./backend → /app/backend              │
│  - ./src → /app/src                      │
│  - ./models → /app/models                │
│  - ./frontend → /app                     │
└─────────────────────────────────────────┘
```

## Security Considerations

### CORS Configuration
- Allows requests from localhost:5173 and localhost:3000
- Credentials enabled for WebSocket authentication
- All methods and headers allowed in development

### Session Management
- UUID-based session IDs
- Sessions stored in memory (ephemeral)
- No persistent storage of game data

### Input Validation
- Pydantic models validate all API inputs
- TypeScript provides compile-time type checking
- Action validation in GameState

## Performance Considerations

### Backend
- Async/await for non-blocking operations
- WebSocket broadcast to multiple clients
- Efficient game state serialization
- Bot action delays (500ms) for UX

### Frontend
- React 18 with concurrent features
- Zustand for minimal re-renders
- Component memoization opportunities
- Lazy loading for modals
- Optimized Tailwind CSS build

## Scalability Notes

Current implementation is single-server, in-memory. For scale:

1. **Session Storage**: Move to Redis/Database
2. **WebSocket**: Use Redis pub/sub for multi-server
3. **Load Balancing**: Sticky sessions for WebSocket
4. **Bot Execution**: Move to worker queue (Celery/RQ)
5. **State Persistence**: Database for game history
6. **Caching**: Cache opponent stats, hand history

## Error Handling Strategy

### Backend
- Try/catch blocks in all endpoints
- HTTPException with proper status codes
- WebSocket error messages
- Logging for debugging

### Frontend
- Error boundaries (can be added)
- Toast notifications for user errors
- Loading states during API calls
- WebSocket reconnection logic (can be enhanced)
- Graceful degradation when WebSocket fails

## Monitoring & Debugging

### Development Tools
- FastAPI automatic API docs at `/docs`
- React DevTools for component inspection
- Browser console for WebSocket messages
- Network tab for API requests
- Backend console for server logs

### Debugging Endpoints
- `GET /health` - Health check
- `GET /` - API info
- WebSocket connection status in UI

## Future Architecture Improvements

1. **Authentication**: Add JWT-based auth
2. **Database**: PostgreSQL for persistence
3. **Caching**: Redis for sessions and stats
4. **Message Queue**: RabbitMQ for bot actions
5. **API Gateway**: Kong or Nginx for routing
6. **Monitoring**: Prometheus + Grafana
7. **Logging**: ELK stack or CloudWatch
8. **CDN**: CloudFront for frontend assets
9. **Container Orchestration**: Kubernetes
10. **Microservices**: Separate game engine service
