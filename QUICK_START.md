# LagBot Poker - Quick Start Guide

## 30-Second Setup

```bash
# From LagBot directory
./check_setup.sh           # Verify requirements
pip install -r backend/requirements.txt
cd frontend && npm install && cd ..
./start_web.sh             # Start both servers
```

Open http://localhost:5173

## Manual Start

### Terminal 1 - Backend
```bash
cd /Users/mbb/Developer/Personal_Projects/LagBot
source venv/bin/activate  # If using venv
PYTHONPATH=. uvicorn backend.main:app --reload --port 8000
```

### Terminal 2 - Frontend
```bash
cd /Users/mbb/Developer/Personal_Projects/LagBot/frontend
npm run dev
```

## First Game

1. Browser opens to http://localhost:5173
2. "New Game" modal appears automatically
3. Configure:
   - **Opponents**: 2 (recommended for first game)
   - **Type**: "Call Bot" (easiest to play against)
   - **Stack**: 1000 chips
   - **Blinds**: 5/10
4. Click "Start Game"
5. See poker table with your cards at bottom
6. When your turn, use action buttons to play

## Quick Commands

```bash
# Check if running
lsof -i :8000  # Backend
lsof -i :5173  # Frontend

# Stop servers
pkill -f uvicorn  # Backend
pkill -f vite     # Frontend

# View logs
# Backend: Terminal 1 output
# Frontend: Terminal 2 output

# Test API
curl http://localhost:8000/health

# Build frontend for production
cd frontend && npm run build
```

## Common Issues

**Port already in use:**
```bash
lsof -ti:8000 | xargs kill -9  # Kill process on port 8000
lsof -ti:5173 | xargs kill -9  # Kill process on port 5173
```

**Module not found:**
```bash
# Backend
pip install -r backend/requirements.txt

# Frontend
cd frontend && npm install
```

**WebSocket won't connect:**
- Ensure backend is running on port 8000
- Check browser console for errors
- Verify no firewall blocking WebSocket

## File Locations

```
📁 Backend API:        backend/api/routes.py
📁 WebSocket:          backend/api/websocket.py
📁 Game Logic:         backend/services/game_session.py
📁 Frontend App:       frontend/src/App.tsx
📁 Game State:         frontend/src/stores/gameStore.ts
📁 Components:         frontend/src/components/
```

## Key URLs

- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## Action Types

When playing:
- **Fold** = Give up hand
- **Check** = Pass (when no bet)
- **Call** = Match current bet
- **Raise** = Increase bet (use slider)
- **All-In** = Bet all chips

## Opponent Types

- **Trained AI**: Best opponents (requires trained models)
- **Call Bot**: Always calls (good for testing)
- **Random Bot**: Random actions (unpredictable)
- **Mixed Bots**: Combination of types

## Game Info

- **VPIP**: Voluntarily Put money In Pot (%)
- **PFR**: Pre-Flop Raise (%)
- **AF**: Aggression Factor (ratio)
- **Dealer Button (D)**: Rotates each hand
- **Small Blind/Big Blind**: Posted before cards

## Tips

1. Start with 2 Call Bots to learn interface
2. Watch the pot size - displayed in center
3. Your cards always show, opponents' are hidden
4. Bot actions happen automatically with delays
5. "Next Hand" button appears when hand complete
6. Can create multiple games (different sessions)

## Next Steps

- Read README_WEB.md for detailed documentation
- Check TESTING.md for test procedures
- See ARCHITECTURE.md for system design
- Review IMPLEMENTATION_SUMMARY.md for overview

## Need Help?

1. Check browser console (F12)
2. Check backend terminal output
3. Verify both servers running
4. Review error messages in UI
5. Check README_WEB.md troubleshooting section

## Development

```bash
# Backend changes auto-reload (--reload flag)
# Frontend changes auto-reload (Vite HMR)

# TypeScript type check
cd frontend && npx tsc --noEmit

# API documentation
open http://localhost:8000/docs
```

## Docker Alternative

```bash
docker-compose up       # Start both services
docker-compose down     # Stop services
docker-compose logs -f  # View logs
```

## Quick Test Sequence

1. ✅ Both servers start without errors
2. ✅ Browser opens to game interface
3. ✅ Create new game with 2 opponents
4. ✅ See poker table with cards
5. ✅ Click "Call" when your turn
6. ✅ Watch bots take actions
7. ✅ Hand completes, shows winner
8. ✅ Click "Next Hand", new hand starts

If all steps pass, system is working correctly!

---

**Ready to Play? Run: `./start_web.sh`**
