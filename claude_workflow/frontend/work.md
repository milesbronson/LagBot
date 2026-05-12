# Work Log: Multi-Player Frontend + Backend Fixes

Everything that was built, changed, configured, and how to run it.

---

## Quick Start

```bash
# 1. Start PostgreSQL (Docker must be running)
docker run -d --name lagbot-postgres \
  -e POSTGRES_USER=lagbot \
  -e POSTGRES_PASSWORD=lagbot \
  -e POSTGRES_DB=lagbot \
  -p 5432:5432 \
  postgres:16-alpine

# 2. Start Backend
cd /Users/mbb/Developer/Personal_Projects/LagBot
source venv/bin/activate
pip install asyncpg  # if not installed
PYTHONPATH=. uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# 3. Start Frontend
cd /Users/mbb/Developer/Personal_Projects/LagBot/frontend
npm install
npm run dev

# Open http://localhost:5173

# OR run everything with Docker Compose:
docker-compose up
```

---

## Credentials & Configuration

| What | Value | Where it's set |
|------|-------|---------------|
| PostgreSQL user | `lagbot` | docker-compose.yml, backend/db/database.py |
| PostgreSQL password | `lagbot` | docker-compose.yml, backend/db/database.py |
| PostgreSQL database | `lagbot` | docker-compose.yml, backend/db/database.py |
| PostgreSQL port | `5432` | docker-compose.yml |
| Database URL (local) | `postgresql://lagbot:lagbot@localhost:5432/lagbot` | backend/db/database.py (default) |
| Database URL (docker) | `postgresql://lagbot:lagbot@db:5432/lagbot` | docker-compose.yml env var |
| Database URL env var | `DATABASE_URL` | backend/db/database.py reads this |
| Backend port | `8000` | docker-compose.yml, vite.config.ts proxy |
| Frontend port | `5173` | Vite default, CORS allowed in main.py |
| CORS origins | `http://localhost:5173`, `http://localhost:3000` | backend/main.py |
| localStorage key | `lagbot-game-settings` | frontend NewGameModal.tsx |
| WebSocket max retries | `5` | frontend useWebSocket.ts |
| WebSocket retry base delay | `1000ms` (exponential backoff) | frontend useWebSocket.ts |
| Bot think delay | `0.5s` | backend game_session.py |
| Action badge display time | `2s` | frontend useWebSocket.ts |

---

## Phase 0: Fix Server Error

**Problem:** `POST /api/game/new` crashed because `_load_bot_agents("trained")` looked for `models/ppo/ppo_poker_gen_*.zip` — no such directory. Actual models live in `models/vs_v2_and_new_20260212_182022/model_*_steps.zip`.

**Also fixed:** `state_serializer.py` had attribute mismatches — Player class uses `.hand` not `.hole_cards`, `.current_bet` not `.bet`, `.is_active` not `.folded`, and `game_state` has no `dealer_idx`/`small_blind_idx`/`big_blind_idx` (computed from `button_position`).

### Files Changed

**`backend/services/game_session.py`** — `_find_latest_model()`
- Scans `models/` recursively for `model_*_steps.zip`
- Parses step count from filename, picks highest
- Falls back to any `.zip` file if no step-named files found
- Model found: `models/vs_v2_and_new_20260212_182022/model_6650000_steps.zip`

**`backend/utils/state_serializer.py`** — attribute fixes
- `player.hand` instead of `player.hole_cards`
- `player.current_bet` instead of `player.bet`
- `not player.is_active and not player.is_all_in` instead of `player.folded`
- Added `_get_position_indices()` to compute dealer/SB/BB from `button_position`
- Heads-up special case: SB = dealer

---

## Phase 1: Fix Backend Bot Loop

**Problem:** 3 bugs in `game_session.py` broke multi-player games:
1. `break` on folded/all-in instead of continuing
2. Fragile `bot_idx = current_player_idx - 1` index math
3. Manual `advance_betting_round()` call double-advancing (env already does this)

### Files Changed

**`backend/services/game_session.py`**
- `_load_bot_agents()` returns `Dict[int, object]` keyed by player_id instead of `List`
- New `_run_bot_loop()` method — simple while loop, looks up agents by `bot_agents.get(current_player_id)`, no folded/all-in check (env handles skip), no manual round advancement
- `start_hand()` made async — runs bot loop if human isn't first to act (e.g. human is dealer, UTG is a bot)

**`backend/api/routes.py`**
- `create_game()` and `start_new_hand()` now `await session.start_hand()`

---

## Phase 2: Backend Broadcasting

**Problem:** WebSocket broadcasts didn't include what action was just taken.

### Files Changed

**`backend/services/game_session.py`**
- `_broadcast_current_state()` accepts optional `last_action` dict
- Bot loop sends `last_action` with each broadcast: `{player_id, player_name, action, amount}`
- Human action also broadcast with `last_action`
- Message format: `{"type": "state_update", "state": {...}, "last_action": {...}}`

---

## Phase 3: PostgreSQL Hand History

### New Files Created

**`backend/db/__init__.py`** — empty package init

**`backend/db/schema.sql`** — 3 tables:
```
hands           — session_id, hand_number, community_cards, pot, winners
hand_actions    — hand_id FK, action_order, player_id, action_type, amount, betting_round
hand_players    — hand_id FK, player_id, hole_cards, starting/ending stack, is_human
```
All with CASCADE deletes and indexes on session_id and hand_id.

**`backend/db/database.py`** — asyncpg connection pool
- Reads `DATABASE_URL` env var, defaults to `postgresql://lagbot:lagbot@localhost:5432/lagbot`
- `get_pool()` — creates pool (1-5 connections), auto-runs schema.sql on first call
- `close_pool()` — graceful shutdown

**`backend/db/hand_history.py`** — persistence layer
- `save_hand()` — inserts into all 3 tables in a transaction
- `get_session_hands(session_id, limit=20)` — returns hands with nested actions and players
- `get_all_hands(limit=50)` — summary across all sessions

### Files Changed

**`backend/services/game_session.py`**
- Tracks `_hand_actions` list and `_hand_starting_stacks` dict per hand
- `_record_action()` appends action with player_id, action_type, betting_round, pot_after
- `_save_hand_to_db()` called when hand completes — graceful try/except (game works without DB)
- `start_hand()` resets action tracking and snapshots starting stacks
- `execute_human_action()` records human actions too

**`backend/main.py`**
- Added `lifespan` async context manager
- On startup: connects DB pool, prints status
- On shutdown: closes pool
- If DB unavailable, prints warning and continues (hand history disabled)

**`backend/api/routes.py`**
- `GET /api/game/{session_id}/hand-history?limit=20` — returns `{"hands": [...]}`
- `GET /api/hand-history?limit=50` — returns `{"hands": [...]}`

**`backend/requirements.txt`**
- Added `asyncpg>=0.29.0`

**`docker-compose.yml`**
- Added `db` service: `postgres:16-alpine`, port 5432, volume `pgdata`
- Backend gets `DATABASE_URL=postgresql://lagbot:lagbot@db:5432/lagbot` env var
- Backend depends_on: db

**`frontend/src/api/client.ts`**
- Added `getHandHistory(sessionId, limit)` function

---

## Phase 4: Frontend Layout for Multi-Player

### Files Changed

**`frontend/src/utils/positioning.ts`**
- `calculateSeatPosition()` — heads-up: top/bottom, 3+: ellipse (radiusX=0.42, radiusY=0.38), starts from bottom (human) clockwise
- New `calculateBetPosition()` — places chips 45% of the way from seat toward table center

**`frontend/src/components/PokerTable/PokerTable.tsx`**
- Responsive: 800x500 normal, 900x560 for 7+ players (`compact` mode)
- Renders bet chips between seats and pot center (gold circle + yellow amount text)
- Passes `compact` prop to seats

**`frontend/src/components/PokerTable/PokerTableSeat.tsx`**
- Accepts `compact` prop, passes to `PlayerInfo`

**`frontend/src/components/Player/PlayerInfo.tsx`**
- Two modes: normal (140px min, full labels) and compact (100px min, abbreviated)
- Removed inline bet display (now shown as chips on table)

---

## Phase 5: Frontend Features

### Files Changed

**`frontend/src/stores/gameStore.ts`**
- Added state: `currentHandActions`, `lastAction`, `opponentStats`
- Added actions: `addHandAction`, `clearHandActions`, `setLastAction`, `fetchOpponentStats`
- `fetchOpponentStats()` — called after hand completes, fetches stats for all opponents
- `startNextHand()` — clears hand actions and last action
- `createNewGame()` — resets opponent stats

**`frontend/src/hooks/useWebSocket.ts`**
- Parses `last_action` from state_update messages
- Adds to `currentHandActions` via `addHandAction()`
- Sets `lastAction` for 2 seconds (for action badge animation)
- Reconnection: exponential backoff, 5 retries, base 1000ms

**`frontend/src/components/Sidebar/OpponentStats.tsx`**
- Reads `opponentStats` from store instead of hardcoded null
- Shows real VPIP, PFR, AF, hands played per opponent

**`frontend/src/components/Sidebar/HandHistory.tsx`**
- Reads `currentHandActions` from store
- Color-coded action list: fold=red, check=gray, call=green, raise=yellow, all-in=red
- Auto-scrolls to latest action
- Shows "Waiting for actions..." when empty

**`frontend/src/components/Controls/ActionPanel.tsx`**
- Primary: slider for custom raise (min_raise to all-in)
- Quick buttons: ½ Pot, Pot, 2x Pot (set slider position, filtered by max)
- Always visible: Fold, Check/Call, All-In
- Shows "Waiting for other players..." with pulse animation when not human's turn

**`frontend/src/components/Player/PlayerActionIndicator.tsx`**
- Reads `lastAction` from store
- Shows animated badge for 2s after each action (bounce animation)
- Color-coded: fold=red-600, check=gray-600, call=green-600, raise=yellow-600, all-in=red-700
- Falls back to Folded/All-In/Thinking... static badges

---

## Phase 6: Polish

### Files Changed

**`frontend/src/hooks/useWebSocket.ts`**
- Reconnection with exponential backoff: `1000 * 2^retryCount` ms
- Max 5 retries before giving up
- Cleans up timer on unmount

**`frontend/src/components/Modals/HandResultModal.tsx`**
- Removed unused `useGameStore` import

**`frontend/src/components/Modals/NewGameModal.tsx`**
- Saves settings to `localStorage` key `lagbot-game-settings` on submit
- Loads saved settings on mount (with fallback defaults)

**`frontend/src/App.tsx`**
- Removed unused `React` import

---

## All Modified/Created Files

### Backend (11 files)
| File | Status |
|------|--------|
| `backend/main.py` | Modified (added lifespan for DB) |
| `backend/services/game_session.py` | Rewritten (model path, bot loop, action tracking, DB save) |
| `backend/api/routes.py` | Modified (await start_hand, added hand-history endpoints) |
| `backend/api/websocket.py` | Unchanged |
| `backend/utils/state_serializer.py` | Rewritten (attribute fixes, position computation) |
| `backend/db/__init__.py` | **New** |
| `backend/db/database.py` | **New** (asyncpg pool) |
| `backend/db/hand_history.py` | **New** (save/query hands) |
| `backend/db/schema.sql` | **New** (3 tables) |
| `backend/requirements.txt` | Modified (added asyncpg) |
| `docker-compose.yml` | Modified (added postgres service) |

### Frontend (14 files)
| File | Status |
|------|--------|
| `frontend/src/App.tsx` | Modified (removed unused import) |
| `frontend/src/api/client.ts` | Modified (added getHandHistory) |
| `frontend/src/stores/gameStore.ts` | Rewritten (added actions, stats, history state) |
| `frontend/src/hooks/useWebSocket.ts` | Rewritten (last_action parsing, reconnection) |
| `frontend/src/utils/positioning.ts` | Rewritten (heads-up, bet positions) |
| `frontend/src/components/PokerTable/PokerTable.tsx` | Rewritten (responsive, bet chips) |
| `frontend/src/components/PokerTable/PokerTableSeat.tsx` | Modified (compact prop) |
| `frontend/src/components/Player/PlayerInfo.tsx` | Rewritten (compact mode) |
| `frontend/src/components/Player/PlayerActionIndicator.tsx` | Rewritten (action badges) |
| `frontend/src/components/Controls/ActionPanel.tsx` | Rewritten (slider + quick buttons) |
| `frontend/src/components/Sidebar/OpponentStats.tsx` | Rewritten (real stats from store) |
| `frontend/src/components/Sidebar/HandHistory.tsx` | Rewritten (live action feed) |
| `frontend/src/components/Modals/HandResultModal.tsx` | Modified (removed unused import) |
| `frontend/src/components/Modals/NewGameModal.tsx` | Modified (localStorage persistence) |

---

## Verification

All tests pass:

```
Backend:
  - 2-player heads-up game: PASS
  - 6-player multi-player game: PASS
  - Opponent stats after 5 hands: PASS
  - Model discovery (6,650,000 steps): PASS
  - Bot agents keyed by player_id: PASS
  - Action tracking (12 actions recorded per hand): PASS
  - PostgreSQL integration (3 hands saved/queried): PASS

Frontend:
  - TypeScript: tsc --noEmit — 0 errors
  - Build: vite build — success (220KB JS, 26KB CSS)
```

---

## Architecture Summary

```
Browser (localhost:5173)
  │
  ├── REST (via Vite proxy to :8000)
  │   POST /api/game/new              → create session, start hand
  │   POST /api/game/{id}/action      → human acts, bots respond
  │   POST /api/game/{id}/new-hand    → deal next hand
  │   GET  /api/game/{id}/state       → poll current state
  │   GET  /api/game/{id}/opponent-stats/{pid}
  │   GET  /api/game/{id}/hand-history
  │   GET  /api/hand-history
  │
  └── WebSocket ws://localhost:8000/ws/{id}
      ← {"type": "state_update", "state": {...}, "last_action": {...}}

Backend (FastAPI :8000)
  │
  ├── GameSession (wraps TexasHoldemEnv)
  │   ├── Bot agents: Dict[player_id → OpponentPPO | CallAgent | RandomAgent]
  │   ├── Action tracking for DB persistence
  │   └── WebSocket broadcasting
  │
  └── PostgreSQL (:5432)
      ├── hands
      ├── hand_actions
      └── hand_players
```

---

## How Does This Work on a Network?

### The Short Answer

When you run `uvicorn backend.main:app --host 0.0.0.0 --port 8000`, the `--host 0.0.0.0` part is key. It tells the server to listen on **all network interfaces**, not just `localhost`. That means any device on your local network (WiFi/Ethernet) can connect to your machine's IP address on port 8000.

Same for the Vite dev server — `host: '0.0.0.0'` in `vite.config.ts` makes it reachable from other devices.

### Step by Step: What Happens When Your Friend Connects

1. **Your machine has an IP on your local network.** Something like `192.168.1.42`. You can find it with `ifconfig | grep "inet "` on macOS. Look for the one that's NOT `127.0.0.1`.

2. **The backend server binds to `0.0.0.0:8000`.** `0.0.0.0` is a special address meaning "listen on all interfaces" — your localhost (`127.0.0.1`), your WiFi IP (`192.168.1.42`), any VPN interface, etc. Without this, it would only listen on `127.0.0.1` and be invisible to other devices.

3. **The Vite dev server binds to `0.0.0.0:5173`.** Same thing — accessible from the network, not just your machine.

4. **Your friend opens `http://192.168.1.42:5173` on their phone/laptop.** Their browser loads the React frontend from your Vite dev server.

5. **The frontend makes API calls to `/api/...`.** These are relative URLs (no hostname), so the browser sends them back to `192.168.1.42:5173`.

6. **Vite's proxy forwards `/api/*` to `localhost:8000`.** This is configured in `vite.config.ts`:
   ```js
   proxy: {
     '/api': { target: 'http://localhost:8000', changeOrigin: true },
     '/ws':  { target: 'ws://localhost:8000', ws: true },
   }
   ```
   So your friend's browser talks to Vite, and Vite forwards to FastAPI. Your friend never directly hits port 8000.

7. **WebSocket connections work the same way.** The frontend connects to `ws://192.168.1.42:5173/ws/{session_id}`, and Vite proxies it to `ws://localhost:8000/ws/{session_id}`.

### Why This Is Cool

- **No deployment needed.** Your laptop IS the server. As long as everyone is on the same WiFi network, they can play.
- **The proxy hides the backend.** Only port 5173 needs to be reachable. The backend (port 8000) stays internal to your machine.
- **WebSockets give real-time updates.** Instead of everyone polling "has something changed?", the server pushes state updates instantly to all connected browsers the moment a bot or player acts.

### Network Diagram

```
Your Laptop (192.168.1.42)
├── Vite Dev Server (:5173) ← Friends connect here
│   ├── Serves React app (HTML/JS/CSS)
│   └── Proxies /api/* and /ws/* to backend
├── FastAPI Backend (:8000) ← Only Vite talks to this
│   ├── REST API (game creation, actions, stats)
│   ├── WebSocket server (real-time state updates)
│   └── Game engine + bot agents
└── PostgreSQL (:5432) ← Only backend talks to this

Friend's Phone (192.168.1.50)
└── Browser → http://192.168.1.42:5173
    ├── Loads React app
    ├── REST calls → /api/* → proxied to :8000
    └── WebSocket → /ws/* → proxied to :8000
```

### Why `npm run dev` Was Insanely Slow for Friends

When you run `npm run dev`, Vite starts a **development server**. This server does NOT bundle your code. Instead, it serves every single file in your project individually — each React component, each hook, each utility, each library dependency is a separate HTTP request. For our app, that's **~120 separate files** the browser has to download one by one.

On your own machine this is fine because those requests go to `localhost` — they never touch the network, they just bounce around inside your computer in microseconds.

But when your friend connects over WiFi, **every single one of those 120 files is a separate network round trip**. Each request has to:
1. Travel from their phone → WiFi router → your laptop
2. Vite transforms the file on-the-fly (TypeScript → JavaScript, JSX → JS)
3. Send the response back: your laptop → router → their phone
4. Their browser parses it, discovers more imports, and starts the next request

A single WiFi round trip is ~5-20ms. Multiply that by 120 sequential requests and you're looking at **2-5 seconds just for network latency**, plus all the on-the-fly compilation time. Some of the library files (React, Zustand, axios) are big and take extra time to transform.

**The fix: `npm run build` + `npm run preview`**

`npm run build` does all that transformation ahead of time and bundles everything into **3 files**:
- `index.html` — 0.5 KB
- `index.css` — 26 KB
- `index.js` — 220 KB (all of React, your components, everything, minified)

Then `npm run preview` serves those 3 files as a simple static file server. Your friend's browser makes 3 requests instead of 120, downloads ~250KB total, and the app loads instantly.

```
npm run dev (development):     120 files × ~5-20ms each = 2-5 seconds minimum
npm run preview (production):  3 files × ~5-20ms each   = 15-60ms
```

### How to Run for Friends

```bash
# Build once (only need to rebuild when you change code)
cd /Users/mbb/Developer/Personal_Projects/LagBot/frontend
npm run build

# Start the preview server (serves the built files)
npm run preview

# Friends connect to:
# http://192.168.0.102:4173
```

Make sure the backend is also running in another terminal:
```bash
cd /Users/mbb/Developer/Personal_Projects/LagBot
source venv/bin/activate
PYTHONPATH=. uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

### Gotchas

- **Firewall:** macOS will pop up "Do you want to allow incoming connections?" the first time. Click Allow.
- **Same network only:** This only works on local WiFi/LAN. For friends not on your network, you'd need port forwarding on your router or a tool like `ngrok` or `tailscale`.
- **Rebuild after code changes:** If you edit the frontend code, you need to run `npm run build` again. The preview server serves the last build — it doesn't auto-update like `npm run dev` does.
- **Use `npm run dev` for developing, `npm run preview` for playing with friends.**
