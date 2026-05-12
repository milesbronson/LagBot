# Plan: Multi-Player Frontend + Backend Fixes

## Context

Single human player vs 1-9 bots. No multi-human multiplayer — one human connects, bots run server-side. The frontend currently doesn't work at all (server error on game creation, likely a broken model path), and even if it did, it only supports heads-up layout. The backend bot loop has bugs that break 3+ player games. Missing features: opponent stats never fetched, hand history is a placeholder, bet sizing is limited. -m: this should be humans vs bots the idea is to be able to get all my friends together to play the bot. -m: make an explination section at the bottom of these files where you explain parts that I dont understand first thing being how is this hosted on a network what did you have to do to do that? how does that even work that is so cool!

### Running Locally

```bash
# Terminal 1 — Backend
cd /Users/mbb/Developer/Personal_Projects/LagBot
source venv/bin/activate  # or create: python -m venv venv && source venv/bin/activate
pip install -r backend/requirements.txt
pip install -r requirements.txt
PYTHONPATH=. uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 — Frontend
cd /Users/mbb/Developer/Personal_Projects/LagBot/frontend
npm install
npm run dev

# Open http://localhost:5173
```

---

## Phase 0: Fix Server Error (Blocking)

The game won't start at all right now. The `POST /api/game/new` endpoint calls `GameSession.__init__` which calls `_load_bot_agents("trained", N)`. This looks for model files matching `models/ppo/ppo_poker_gen_*.zip` — but no such directory exists. Actual trained models are in `models/vs_v2_and_new_20260212_182022/model_*_steps.zip`.

### 0.1 Fix model discovery path

**File:** `backend/services/game_session.py` — `_load_bot_agents()`

**Fix:** Update the glob pattern to find models in the actual location, or add a config/env var for the model directory. The simplest fix: scan `models/` for subdirectories, pick the latest, glob for `model_*_steps.zip`, sort by step count, load the latest.

### 0.2 Verify full game creation flow

Start the backend, hit `POST /api/game/new` with `{"num_opponents": 1, "opponent_type": "call"}` (bypasses model loading). If that works, the model path was the only blocker. Then test with `"trained"` after fixing the path.

---

## Phase 1: Fix Backend Bot Loop (Critical)

The bot action loop in `game_session.py` has bugs that break multi-player games. These are **backend bugs** in the game session layer — not frontend issues and not bugs in the core poker engine (`game_state.py`). The engine itself handles player advancement correctly; the session wrapper misuses it.

### 1.1 Fix bot loop breaking on folded/all-in players

**File:** `backend/services/game_session.py` (~line 160)

**Bug:** When the loop encounters a folded or all-in bot, it `break`s out of the entire loop instead of skipping to the next player. In a 6-player game, if bot 2 already folded, bots 3-5 never get to act.

**Fix:** Remove the folded/all-in check entirely. The environment handles this internally — `env.step()` and `current_player_idx` already skip past players who can't act.

### 1.2 Fix stale observations for bots

**File:** `backend/services/game_session.py` (~line 175)

**Bug:** All bots in a round may use the observation from before the previous bot acted.

**Fix:** Verify the loop properly chains observations — each `env.step()` returns a fresh `obs` that must be passed to the next bot's `select_action()`.

### 1.3 Player-keyed agent mapping (environment + backend)

**File:** `backend/services/game_session.py` (~line 172) AND `src/poker_env/texas_holdem_env.py`

**Bug:** `bot_idx = current_player_idx - 1` assumes human is always at index 0. Fragile arithmetic.

**Fix — two layers:**
1. **Environment layer:** Add a `player_id → agent` mapping dict to `TexasHoldemEnv`. All internal operations (training, step, observation) should be keyed by player_id, not positional index. This eliminates ambiguity during both training and live play.
2. **Backend session:** Build `player_id → bot_agent` dict at session creation time. Use this dict in the bot loop instead of index arithmetic.

### 1.4 Add betting round advancement in the bot loop

**Bug:** When the betting round completes mid-bot-loop, the new round doesn't start automatically if it's a bot's turn first.

**Fix:** After each `env.step()`, check if the hand advanced to a new betting round or completed. Continue the bot loop if `current_player_idx != human_player_id`. Only break when it's the human's turn or the hand is done.

---

## Phase 2: Fix Backend State Broadcasting

### 2.1 Broadcast after each bot action with action info

Currently broadcasts raw state. Add the action that was just taken so the frontend can animate it and build hand history.

**File:** `backend/services/game_session.py`

Add to broadcast message:
```json
{
  "type": "state_update",
  "state": {...},
  "last_action": {
    "player_id": 2,
    "player_name": "Player 2",
    "action": "raise",
    "amount": 50
  }
}
```

### 2.2 Wire up opponent stats endpoint

The endpoint exists (`GET /api/game/{id}/opponent-stats/{pid}`) but the frontend never calls it. No backend change needed — just needs frontend integration. Stats are shown **only to the human player** (this is a single-player-vs-bots game, no other humans to show them to).

---

## Phase 3: Add PostgreSQL for Hand History

Hand history needs persistence across sessions. PostgreSQL stores completed hands with full action sequences.

### 3.1 Database schema

```sql
CREATE TABLE hands (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(64) NOT NULL,
    hand_number INTEGER NOT NULL,
    num_players INTEGER NOT NULL,
    small_blind INTEGER NOT NULL,
    big_blind INTEGER NOT NULL,
    community_cards TEXT[],           -- e.g. {'Ah','Kd','Qs','7c','2h'}
    pot INTEGER NOT NULL,
    winner_ids INTEGER[],
    winner_amounts INTEGER[],
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE hand_actions (
    id SERIAL PRIMARY KEY,
    hand_id INTEGER REFERENCES hands(id),
    action_order INTEGER NOT NULL,
    player_id INTEGER NOT NULL,
    player_name VARCHAR(64),
    betting_round VARCHAR(16),       -- PREFLOP, FLOP, TURN, RIVER
    action_type VARCHAR(16),         -- fold, check, call, raise, all_in
    amount INTEGER DEFAULT 0,
    pot_after INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE hand_players (
    id SERIAL PRIMARY KEY,
    hand_id INTEGER REFERENCES hands(id),
    player_id INTEGER NOT NULL,
    player_name VARCHAR(64),
    hole_cards TEXT[],               -- null if mucked
    starting_stack INTEGER,
    ending_stack INTEGER,
    is_human BOOLEAN DEFAULT FALSE,
    position VARCHAR(16)             -- dealer, sb, bb, utg, etc.
);
```

### 3.2 Backend integration

**New file:** `backend/db/database.py` — asyncpg connection pool, query helpers
**New file:** `backend/db/models.py` — SQLAlchemy or raw SQL for hand persistence

- After each hand completes (in `game_session.py`), write to `hands`, `hand_actions`, `hand_players`
- Add endpoint: `GET /api/game/{id}/hand-history` — returns last N hands for the session
- Add endpoint: `GET /api/hand-history` — returns hand history across all sessions

### 3.3 Docker Compose update

Add PostgreSQL service to `docker-compose.yml`. For local dev, require a local Postgres instance or provide a `docker-compose.dev.yml` that runs just the DB.

---

## Phase 4: Rework Frontend Layout for Multi-Player

### 4.1 Make table responsive to player count

**File:** `frontend/src/components/PokerTable/PokerTable.tsx`

- Remove hardcoded 800x500px dimensions
- Use relative/percentage-based sizing that fills the container
- Scale seat size based on player count:
  - 2-3 players: larger seats, more spacing
  - 4-6 players: medium seats
  - 7-10 players: compact seats

### 4.2 Update seat positioning

**File:** `frontend/src/utils/positioning.ts`

- Keep the ellipse algorithm (it works mathematically)
- Add padding/overlap detection
- For heads-up (2 players): position top and bottom, not on ellipse
- For 3-6 players: standard ellipse with generous spacing
- For 7-10: tighter ellipse, smaller player info boxes

### 4.3 Compact PlayerInfo for large tables

**File:** `frontend/src/components/Player/PlayerInfo.tsx`

- Add a `compact` prop triggered when player count > 6
- Compact mode: smaller font, less padding, abbreviated labels
- Always show: name, stack, bet. Conditionally show: dealer/blind badges

### 4.4 Show player bets visually in front of each seat

**File:** `frontend/src/components/Player/PlayerInfo.tsx` or new `BetChip` component

- Display each player's current bet as a chip/badge positioned between their seat and the center of the table
- Update on each state change so players can see how much each opponent has put in
- Animate chip movement toward pot when betting round completes

---

## Phase 5: Wire Up Missing Frontend Features

### 5.1 Fetch and display opponent stats (human-only HUD)

**Files:** `frontend/src/components/Sidebar/OpponentStats.tsx`, `frontend/src/api/client.ts`, `frontend/src/stores/gameStore.ts`

- Add `opponentStats: Record<number, OpponentStats>` to the store
- After each hand completes, fetch stats for all opponents
- Pass real stats to `PlayerStats` component instead of null
- Show VPIP, PFR, AF, hands played, confidence meter
- Only the human player sees this — it's a heads-up display for the human's benefit

### 5.2 Build hand history from WebSocket events + PostgreSQL

**Files:** `frontend/src/components/Sidebar/HandHistory.tsx`, `frontend/src/stores/gameStore.ts`

**Live (current hand):**
- Add `currentHandActions: ActionHistoryEntry[]` to the store
- Parse `last_action` from WebSocket state_update messages
- Display scrollable list: "Player 2 raises to $50", "Player 3 folds"
- Color-code by action type (fold=red, raise=yellow, call=green)
- Clear on new hand

**Historical (previous hands):**
- Fetch from `GET /api/game/{id}/hand-history`
- Show collapsible summaries of previous hands
- Click to expand full action sequence

### 5.3 Bet sizing: slider + quick buttons

**File:** `frontend/src/components/Controls/ActionPanel.tsx`

The human is **not** limited to the bot's raise bins. Primary input is a slider for any custom amount between min_raise and all-in. Quick bet buttons are convenience shortcuts on top:

- **Slider:** Primary control, continuous range from min_raise to all-in
- **Quick buttons:** "½ Pot", "Pot", "2x Pot" — set the slider position, not a separate action
- **"All-In" button:** Always visible when valid
- **Fold / Check / Call:** Standard action buttons
- Disable based on `valid_actions` from game state

### 5.4 Show bot action animations

When a `state_update` arrives with `last_action`:
- Show a brief action badge on the acting player ("Raises $50", "Folds", "Calls")
- Fade out after 1-2 seconds
- Add card deal animation when community cards appear

---

## Phase 6: Polish

### 6.1 WebSocket reconnection
- Add reconnect logic with exponential backoff in `useWebSocket.ts`
- Show "Reconnecting..." indicator in header

### 6.2 Hand result improvements
- Show hand ranking name ("Full House", "Two Pair") in HandResultModal
- Highlight winning cards on the table
- Show pot distribution for side pots

### 6.3 Game settings persistence
- Save last NewGameModal settings to localStorage
- Pre-fill on next visit

---

## Todo List

### Phase 0: Fix Server Error
- [x] Fix model discovery path in `_load_bot_agents()`
- [x] Verify game creation works with "call" opponent type
- [x] Verify game creation works with "trained" opponent type
- [x] Test full game flow: create → play hand → new hand
- [x] Fix state_serializer.py attribute mismatches (player.hand, player.current_bet, player.is_active, position indices)

### Phase 1: Backend Bot Loop Fixes
- [x] Remove broken folded/all-in break logic
- [x] Build player_id → bot_agent mapping dict in session
- [x] Handle betting round advancement within bot loop (env handles internally)
- [x] Verify observation chaining between bots
- [x] Make start_hand async to run bot loop if human isn't first to act
- [x] Test with 2, 3, 6 player games
- [ ] Add player_id → agent mapping to environment layer (deferred — session-level mapping works, env change is for training refactor)

### Phase 2: Backend Broadcasting
- [x] Add last_action to state_update WebSocket messages
- [x] Verify opponent stats endpoint returns correct data

### Phase 3: PostgreSQL Hand History
- [x] Set up PostgreSQL (Docker container + docker-compose.yml)
- [x] Create schema (hands, hand_actions, hand_players tables with indexes)
- [x] Write backend persistence layer (asyncpg, auto-init schema on startup)
- [x] Add hand history REST endpoints (GET /api/game/{id}/hand-history, GET /api/hand-history)
- [x] Test hand storage and retrieval (3 hands saved/queried successfully)
- [x] Graceful fallback when DB unavailable (game still works, just no persistence)

### Phase 4: Frontend Layout
- [x] Make PokerTable responsive to player count
- [x] Update seat positioning for different table sizes (heads-up special case, ellipse for 3+)
- [x] Add compact mode to PlayerInfo for 7+ players
- [x] Add bet chip display in front of each player (with calculateBetPosition)

### Phase 5: Frontend Features
- [x] Wire opponent stats fetching and display (human-only HUD)
- [x] Build live hand history from WebSocket action events
- [x] Fetch historical hand history from PostgreSQL endpoint
- [x] Add slider + quick bet buttons (slider is primary, buttons are shortcuts)
- [x] Add bot action animations/badges

### Phase 6: Polish
- [x] WebSocket reconnection with exponential backoff (5 retries)
- [x] Hand result modal cleaned up (removed unused import)
- [x] localStorage for game settings (save/load on NewGameModal)

---

## Files to Modify

### Backend
- `backend/services/game_session.py` — Model path fix (Phase 0), bot loop rewrite (Phase 1), broadcasting (Phase 2), hand persistence (Phase 3)
- `backend/api/routes.py` — Hand history endpoints (Phase 3)
- `backend/api/websocket.py` — Broadcast format update (Phase 2)
- `src/poker_env/texas_holdem_env.py` — Player-keyed agent mapping (Phase 1)

### Frontend
- `frontend/src/components/PokerTable/PokerTable.tsx` — Responsive layout (Phase 4)
- `frontend/src/utils/positioning.ts` — Seat positioning (Phase 4)
- `frontend/src/components/Player/PlayerInfo.tsx` — Compact mode + bet display (Phase 4)
- `frontend/src/stores/gameStore.ts` — Opponent stats + hand history state (Phase 5)
- `frontend/src/api/client.ts` — Hand history + opponent stats API calls (Phase 5)
- `frontend/src/components/Sidebar/OpponentStats.tsx` — Real stats (Phase 5)
- `frontend/src/components/Sidebar/HandHistory.tsx` — Live + historical actions (Phase 5)
- `frontend/src/components/Controls/ActionPanel.tsx` — Slider + quick bet buttons (Phase 5)
- `frontend/src/components/Player/PlayerActionIndicator.tsx` — Action badges (Phase 5)
- `frontend/src/hooks/useWebSocket.ts` — Reconnection (Phase 6)
- `frontend/src/components/Modals/HandResultModal.tsx` — Hand rankings (Phase 6)
- `frontend/src/components/Modals/NewGameModal.tsx` — Settings persistence (Phase 6)

### New Files
- `backend/db/database.py` — PostgreSQL connection pool (Phase 3)
- `backend/db/models.py` — Hand history persistence (Phase 3)
- `frontend/src/components/Player/BetChip.tsx` — Bet display component (Phase 4, if not inlined)

---

## Explanations

### How Is This Hosted on a Network?

The magic is one flag: `--host 0.0.0.0`.

When you start the backend with `uvicorn ... --host 0.0.0.0 --port 8000`, and the frontend Vite server has `host: '0.0.0.0'` in its config, both servers listen on **all network interfaces** — not just localhost. That means any device on the same WiFi can reach your machine.

**How it flows:**

1. Your laptop has a local IP like `192.168.1.42` (find it with `ifconfig | grep "inet "`)
2. Your friend opens `http://192.168.1.42:5173` in their browser
3. Vite serves the React app (HTML/JS/CSS)
4. The React app makes API calls to `/api/...` — these go back to Vite on :5173
5. Vite's proxy (configured in `vite.config.ts`) forwards `/api/*` and `/ws/*` to FastAPI on :8000
6. FastAPI processes the request and responds
7. WebSocket connections get proxied the same way — real-time updates flow from backend → Vite → browser

**Why it works:** `0.0.0.0` means "listen on all interfaces" (WiFi, Ethernet, localhost). The default `127.0.0.1` only accepts connections from the same machine. That one change makes the difference between "only I can use this" and "anyone on my network can play."

**Key pieces in the code:**
- `vite.config.ts` line 18: `host: '0.0.0.0'` — frontend accessible from network
- `vite.config.ts` lines 19-28: proxy config — forwards `/api` and `/ws` to backend
- Uvicorn startup command: `--host 0.0.0.0` — backend accessible from network
- `backend/main.py` lines 34-40: CORS middleware — allows cross-origin requests

**Limitations:** Only works on local network (same WiFi). For remote friends, you'd need `ngrok`, `tailscale`, or router port forwarding.
