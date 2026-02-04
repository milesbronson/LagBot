# LagBot Poker Web Interface

A React-based web interface for playing Texas Hold'em poker against trained AI bots.

## Architecture

- **Backend**: FastAPI (Python) with REST API + WebSocket for real-time updates
- **Frontend**: React 18 + TypeScript + Vite + Zustand (state management)
- **Communication**: Hybrid approach - REST for actions, WebSocket for state broadcasts
- **Styling**: Tailwind CSS + custom animations

## Project Structure

```
LagBot/
├── backend/                    # FastAPI backend
│   ├── main.py                # FastAPI app entry point
│   ├── api/
│   │   ├── routes.py          # REST endpoints
│   │   └── websocket.py       # WebSocket handler
│   ├── services/
│   │   ├── game_manager.py    # Session management
│   │   └── game_session.py    # Wraps TexasHoldemEnv
│   ├── models/
│   │   ├── requests.py        # Pydantic request models
│   │   └── responses.py       # Pydantic response models
│   └── utils/
│       ├── card_converter.py  # Card conversion
│       └── state_serializer.py # Game state → JSON
│
└── frontend/                   # React frontend
    ├── src/
    │   ├── api/               # API client
    │   ├── stores/            # Zustand state management
    │   ├── types/             # TypeScript types
    │   ├── components/        # React components
    │   ├── hooks/             # Custom React hooks
    │   └── utils/             # Utility functions
    └── package.json
```

## Setup Instructions

### Option 1: Manual Setup

#### Backend Setup

```bash
cd /Users/mbb/Developer/Personal_Projects/LagBot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r backend/requirements.txt

# Run backend server
uvicorn backend.main:app --reload --port 8000
```

The backend will be available at http://localhost:8000
API documentation at http://localhost:8000/docs

#### Frontend Setup

```bash
cd /Users/mbb/Developer/Personal_Projects/LagBot/frontend

# Install dependencies
npm install

# Run development server
npm run dev
```

The frontend will be available at http://localhost:5173

### Option 2: Docker Setup

```bash
cd /Users/mbb/Developer/Personal_Projects/LagBot

# Start both backend and frontend
docker-compose up

# Or run in background
docker-compose up -d

# Stop services
docker-compose down
```

## Usage

1. Open http://localhost:5173 in your browser
2. Click "New Game" to configure and start a game
3. Select number of opponents and opponent type:
   - **Trained AI**: Uses trained PPO models (best opponents)
   - **Call Bot**: Always calls
   - **Random Bot**: Takes random actions
   - **Mixed Bots**: Mix of different bot types
4. Configure blinds and starting stacks
5. Play poker against the bots!

## Game Flow

1. **New Game**: User creates game → Backend creates GameSession → Returns initial state
2. **WebSocket Connect**: Frontend connects to WebSocket for real-time updates
3. **Player Action**: User clicks action → REST API call → Backend executes action
4. **Bot Actions**: Backend automatically plays bot turns with 500ms delays
5. **State Updates**: Backend broadcasts game state via WebSocket
6. **Hand Complete**: Modal shows results → User clicks "Next Hand"

## API Endpoints

### REST API

- `POST /api/game/new` - Create new game session
- `POST /api/game/{session_id}/action` - Submit player action
- `POST /api/game/{session_id}/new-hand` - Start new hand
- `GET /api/game/{session_id}/state` - Get current game state
- `GET /api/game/{session_id}/opponent-stats/{player_id}` - Get opponent stats
- `DELETE /api/game/{session_id}` - Delete game session

### WebSocket

- `ws://localhost:8000/ws/{session_id}` - Real-time game state updates

## Features

- Interactive poker table with visual card display
- Real-time game state updates via WebSocket
- Betting controls: Fold, Check, Call, Raise (custom amount), All-In
- Opponent statistics tracking (VPIP, PFR, AF)
- Hand history display
- Responsive design
- Card dealing animations
- Support for 2-10 players

## Technical Details

### Action Types

- `0` - Fold
- `1` - Check/Call
- `2+` - Raise (with custom amount) or All-In

### Game State

The game state includes:
- Hand number and betting round
- Pot size and current bet
- Community cards
- Player information (stack, bet, hole cards, status)
- Valid actions for human player
- Winner information (when hand complete)

### Bot Execution

- Bots execute with 500ms delays for better UX
- Bot agents loaded from trained PPO models in `models/ppo/`
- Falls back to CallAgent if trained models unavailable

## Development

### Backend Development

```bash
# Run with auto-reload
uvicorn backend.main:app --reload --port 8000

# Run tests (if available)
pytest backend/tests/
```

### Frontend Development

```bash
cd frontend

# Run dev server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview

# Type checking
tsc --noEmit
```

## Troubleshooting

### Backend Issues

1. **Import errors**: Make sure you're running from the LagBot root directory and PYTHONPATH includes the project root
2. **No trained models**: The app will work with CallAgent fallback if no trained models exist
3. **Port already in use**: Change port with `--port 8001`

### Frontend Issues

1. **WebSocket connection failed**: Check backend is running on port 8000
2. **API calls failing**: Check CORS settings in backend/main.py
3. **Build errors**: Clear node_modules and reinstall: `rm -rf node_modules && npm install`

## Future Enhancements

- [ ] Implement hand history with full action log
- [ ] Add opponent stats visualization (charts)
- [ ] Implement multi-table support
- [ ] Add tournament mode
- [ ] Save/load game sessions
- [ ] Add chat functionality
- [ ] Implement replays
- [ ] Mobile responsive improvements
- [ ] Add sound effects
- [ ] Implement achievements/badges

## License

See main project LICENSE file.
