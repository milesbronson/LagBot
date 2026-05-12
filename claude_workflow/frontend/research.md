# Frontend Research: LagBot Poker

Deep read of the entire codebase to understand how the frontend interacts with the backend, game engine, and trained models.

---

## Project Overview

LagBot is a Texas Hold'em poker bot trained with PPO (Proximal Policy Optimization). It has three layers:

1. **Core Engine** (`src/`) - Gym environment, game state machine, hand evaluator, opponent tracker, agents
2. **Backend** (`backend/`) - FastAPI REST + WebSocket server wrapping the engine
3. **Frontend** (`frontend/`) - React 18 + TypeScript + Vite + Zustand + TailwindCSS

The frontend communicates with the backend via REST (game actions) and WebSocket (real-time state updates). The backend instantiates the poker environment and runs bot agents server-side.

---

## Architecture: Data Flow

```
Frontend (React)
    │
    ├── REST: POST /api/game/new          → GameManager.create_session()
    ├── REST: POST /api/game/{id}/action  → GameSession.execute_human_action()
    ├── REST: POST /api/game/{id}/new-hand → GameSession.start_hand()
    ├── REST: GET  /api/game/{id}/state   → serialize_game_state()
    ├── REST: GET  /api/game/{id}/opponent-stats/{pid}
    ├── REST: DELETE /api/game/{id}
    │
    └── WS: /ws/{session_id}
            ← {"type": "connected", "state": {...}}
            ← {"type": "state_update", "state": {...}}
            ← {"type": "error", "message": "..."}
```

When a human submits an action via REST, the backend:
1. Executes the human action in the environment
2. Broadcasts state update via WebSocket
3. Loops through bot agents (with 0.5s delay between each)
4. Broadcasts after each bot action
5. Returns final state in the REST response

---

## Backend API (Complete)

### Endpoints

| Method | Path | Request Body | Response |
|--------|------|-------------|----------|
| POST | `/api/game/new` | `NewGameRequest` | `{session_id, state}` |
| POST | `/api/game/{id}/action` | `{action_type: int, raise_amount?: int}` | `GameState` |
| POST | `/api/game/{id}/new-hand` | none | `GameState` |
| GET | `/api/game/{id}/state` | none | `GameState` |
| GET | `/api/game/{id}/opponent-stats/{pid}` | none | `OpponentProfile` |
| DELETE | `/api/game/{id}` | none | `{message}` |

### NewGameRequest Shape
```json
{
  "num_opponents": 2,        // 1-9
  "opponent_type": "trained", // "trained"|"call"|"random"|"mixed"
  "starting_stack": 1000,
  "small_blind": 5,
  "big_blind": 10
}
```

### GameState Shape (returned by all game endpoints and WebSocket)
```json
{
  "hand_number": 1,
  "betting_round": "PREFLOP",   // PREFLOP|FLOP|TURN|RIVER|SHOWDOWN
  "pot": 15,
  "current_bet": 10,
  "min_raise": 10,
  "community_cards": [],         // e.g. ["Ah", "Kd", "Qs"]
  "players": [
    {
      "player_id": 0,
      "name": "You",
      "stack": 990,
      "bet": 0,
      "is_active": true,
      "is_all_in": false,
      "is_folded": false,
      "hole_cards": ["Ah", "Kd"],  // null for opponents (unless showdown)
      "is_human": true,
      "is_dealer": true,
      "is_small_blind": false,
      "is_big_blind": false
    }
  ],
  "current_player_idx": 0,
  "is_human_turn": true,
  "valid_actions": [0, 1, 2, 3, 4, 5],
  "hand_complete": false,
  "winner_info": null,            // {player_id: amount_won} when complete
  "small_blind": 5,
  "big_blind": 10
}
```

### Action Encoding
- `0` = Fold
- `1` = Check/Call
- `2` = Raise 50% pot (raise_bins[0])
- `3` = Raise 100% pot (raise_bins[1])
- `4` = Raise 200% pot (raise_bins[2])
- `5` = All-in

### Card String Format
- Rank + suit lowercase: `"Ah"` = Ace of hearts, `"Ts"` = Ten of spades
- Ranks: A, K, Q, J, T, 9, 8, 7, 6, 5, 4, 3, 2
- Suits: h (hearts), d (diamonds), c (clubs), s (spades)

### OpponentProfile Shape
```json
{
  "player_id": 1,
  "player_name": "Player 1",
  "hands_played": 42,
  "vpip": 0.45,
  "pfr": 0.28,
  "af": 1.8,
  "three_bet_percent": 0.12,
  "cbet_percent": 0.65,
  "fold_to_cbet_percent": 0.55,
  "went_to_showdown_percent": 0.25,
  "confidence": 0.42
}
```

---

## Game Engine Internals

### Environment (`src/poker_env/texas_holdem_env.py`)

- **Observation space**: 125 dimensions (53 base + 72 opponent tracking)
  - 42 dims: card encoding (7 cards x 6 features each)
  - 3 dims: hand_strength, pot_odds, SPR
  - 8 dims: stack, pot, bet, call, active_players, position, round, button
  - 72 dims: 9 opponent slots x 8 features (VPIP, PFR, AF, 3bet%, cbet%, fold_to_cbet%, wtsd%, confidence)

- **Action space**: Discrete(6) with default raise_bins=[0.5, 1.0, 2.0]
  - Raise amounts computed as `pot * raise_bin_percentage`
  - If player can't afford the raise, falls back to all-in or call

- **Reward**: Terminal profit/loss normalized by starting stack, plus minor fold-shaping

### Game State Machine (`src/poker_env/game_state.py`)

Hand lifecycle:
1. `start_new_hand()` - shuffle, deal, post blinds, set UTG
2. `execute_action()` - fold/call/raise logic, move to next player
3. `is_betting_round_complete()` - all players acted and matched bets
4. `advance_betting_round()` - burn, deal community cards, reset bets
5. `determine_winners()` - evaluate hands, distribute pots, apply rake

### Pot Manager (`src/poker_env/pot_manager.py`)

- Tracks current_bet, min_raise per round
- Handles side pots when players go all-in at different stack sizes
- Raise bins are pot-percentage based (0.5x, 1.0x, 2.0x pot)

### Opponent Tracker (`src/poker_env/opponent_tracker.py`)

- Per-opponent profile: VPIP, PFR, AF, 3-bet%, c-bet%, fold-to-cbet%, showdown%
- Confidence = min(hands_played / 100, 1.0)
- Stats fed into observation vector for the RL agent
- 72-dim fixed vector (9 slots x 8 features, zero-padded)

### Agents

| Agent | Description | Used For |
|-------|-------------|----------|
| `PPOAgent` | SB3 PPO, trains on GPU (CUDA/MPS) | Training |
| `OpponentPPO` | Loads trained .zip model | Bot opponents |
| `CallAgent` | Always checks/calls | Baseline opponent |
| `RandomAgent` | Random valid action | Baseline opponent |
| `HumanAgent` | CLI input | Interactive play |

Bot opponents are loaded in `GameSession._load_bot_agents()`:
- `"trained"`: Latest model from `models/ppo/ppo_poker_gen_*.zip`
- `"call"`: All CallAgent
- `"random"`: All RandomAgent
- `"mixed"`: Alternating Random and Call

---

## Current Frontend (Existing State)

### Tech Stack
- React 18.2.0, TypeScript 5.3.3, Vite 5.0.7
- Zustand 4.4.7 (state management)
- Axios 1.6.2 (HTTP), native WebSocket
- TailwindCSS 3.4.0, React Toastify 9.1.3
- Path alias: `@/*` → `./src/*`

### Component Tree
```
App.tsx
├── NewGameModal         - Game creation form (opponents, stacks, blinds)
├── HandResultModal      - Shows winners at hand end
├── Sidebar
│   ├── HandHistory      - Placeholder (not fully implemented)
│   └── OpponentStats    - Shows opponent list (stats hardcoded to null)
├── PokerTable
│   ├── CommunityCards   - Board cards
│   ├── PotDisplay       - Pot amount badge
│   ├── DealerButton     - "D" badge
│   └── PokerTableSeat (per player)
│       ├── HoleCards    - 2 cards or card backs
│       ├── PlayerInfo   - Name, stack, bet, status
│       └── PlayerActionIndicator - "Folded"/"All-In"/"Thinking..."
└── ActionPanel
    ├── QuickBetButtons  - Fold + Check/Call
    └── BetSlider        - Raise amount slider
```

### State Management (Zustand)
```typescript
gameState: GameState | null
sessionId: string | null
isConnected: boolean
isLoading: boolean
error: string | null

// Actions:
createNewGame(request) → POST /api/game/new
submitPlayerAction(action) → POST /api/game/{id}/action
startNextHand() → POST /api/game/{id}/new-hand
resetGame() → clear all state
```

### WebSocket Hook
- Connects to `ws://{host}/ws/{sessionId}`
- Handles: `connected`, `state_update`, `error` message types
- Updates gameStore on state changes
- No reconnection logic

### Vite Config
- Proxies `/api` → `http://localhost:8000`
- Proxies `/ws` → `ws://localhost:8000`
- Port 5173, host 0.0.0.0

### Styling
- Dark theme only (gray-800/gray-900 backgrounds)
- Custom colors: poker-felt (#0f5132), poker-rail (#8b4513), poker-chip (#ffd700)
- Custom animations: deal-card (0.3s slide-in), chip-move (0.5s bounce)
- Desktop-focused, no mobile optimization

---

## Known Issues & Gaps in Current Frontend

1. **OpponentStats hardcoded to null** - The sidebar component passes `stats: null` to PlayerStats, never fetches from API
2. **HandHistory is a placeholder** - Shows "Action history will be displayed here" with no actual data
3. **No hand history API integration** - `getHandHistory()` endpoint doesn't exist in backend routes (only `get_opponent_stats`)
4. **No reconnection logic** on WebSocket disconnect
5. **No mobile responsiveness** - Fixed pixel widths, desktop-only layout
6. **Action panel doesn't map raise bins** - BetSlider uses arbitrary min/max, not pot-percentage bins from valid_actions
7. **No loading states on table** during bot turns
8. **No animation for bot actions** - State jumps instantly after 0.5s server delay
9. **No sound effects or visual feedback** for actions
10. **Winner display is basic** - No hand ranking shown, no card highlighting
11. **No game settings persistence** - NewGameModal resets to defaults each time
12. **No error recovery** - Errors shown as toasts but no retry mechanism

---

## Training System (How Models Connect)

Training is completely separate from the web interface:

```
train.py → PPOAgent.train() → saves to models/{run_name}/final_model.zip
                                         ↓
GameSession._load_bot_agents("trained") → OpponentPPO(latest model)
                                         ↓
                              Bot plays via select_action(observation)
```

The frontend never touches training. It only interacts with trained models indirectly through the backend, which loads them as opponents.

Training configs live in `configs/` (YAML files) with architecture and hyperparameter variations. There are 29+ completed training runs in `metrics/`.

---

## File Index

### Backend (9 files)
- `backend/main.py` - FastAPI app, CORS, routers
- `backend/api/routes.py` - 6 REST endpoints
- `backend/api/websocket.py` - WebSocket handler
- `backend/services/game_manager.py` - Session factory (singleton)
- `backend/services/game_session.py` - Game session (wraps env + bots)
- `backend/models/requests.py` - Pydantic request DTOs
- `backend/models/responses.py` - Pydantic response DTOs
- `backend/utils/card_converter.py` - Treys int ↔ string
- `backend/utils/state_serializer.py` - GameState → JSON

### Frontend (29 files)
- Config: package.json, tsconfig.json, vite.config.ts, tailwind.config.js
- Entry: main.tsx, App.tsx
- API: api/client.ts
- Store: stores/gameStore.ts
- Hook: hooks/useWebSocket.ts
- Types: types/game.ts, types/action.ts
- Utils: utils/cardMapping.ts, utils/formatting.ts, utils/positioning.ts
- Components: 15 files across PokerTable/, Cards/, Controls/, Modals/, Player/, Sidebar/

### Engine (12 files)
- `src/poker_env/texas_holdem_env.py` - Main Gym environment (125-dim obs, 6 actions)
- `src/poker_env/game_state.py` - Game state machine
- `src/poker_env/hand_evaluator.py` - Treys wrapper
- `src/poker_env/opponent_tracker.py` - Per-opponent stats
- `src/poker_env/player.py` - Player state
- `src/poker_env/pot_manager.py` - Pot/side-pot/betting
- `src/agents/base_agent.py` - Abstract agent interface
- `src/agents/ppo_agent.py` - PPO training agent (SB3, GPU)
- `src/agents/opponent_ppo.py` - Trained model loader
- `src/agents/random_agent.py` - Call/Random agents
- `src/agents/human_agent.py` - CLI human agent
- `src/utils/model_manager.py` - Model save/load
