# LagBot Web Interface Testing Guide

## Quick Start Test

1. **Install Backend Dependencies**
   ```bash
   cd /Users/mbb/Developer/Personal_Projects/LagBot
   pip install -r backend/requirements.txt
   ```

2. **Install Frontend Dependencies**
   ```bash
   cd frontend
   npm install
   cd ..
   ```

3. **Start Backend**
   ```bash
   # From LagBot root directory
   PYTHONPATH=. uvicorn backend.main:app --reload --port 8000
   ```

4. **In a new terminal, start Frontend**
   ```bash
   cd frontend
   npm run dev
   ```

5. **Open Browser**
   - Navigate to http://localhost:5173
   - You should see the LagBot Poker interface

## Manual API Testing

### Test 1: Health Check

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{"status": "healthy"}
```

### Test 2: Create New Game

```bash
curl -X POST http://localhost:8000/api/game/new \
  -H "Content-Type: application/json" \
  -d '{
    "num_opponents": 2,
    "opponent_type": "call",
    "starting_stack": 1000,
    "small_blind": 5,
    "big_blind": 10
  }'
```

Expected response includes:
- `session_id`: UUID string
- `state`: Complete game state object with players, cards, pot, etc.

### Test 3: Submit Action

Replace `{session_id}` with the ID from Test 2:

```bash
curl -X POST http://localhost:8000/api/game/{session_id}/action \
  -H "Content-Type: application/json" \
  -d '{
    "action_type": 1
  }'
```

Expected response: Updated game state

### Test 4: WebSocket Connection

Install `wscat` if needed: `npm install -g wscat`

```bash
wscat -c ws://localhost:8000/ws/{session_id}
```

You should receive:
1. Initial connection message with game state
2. State updates as actions occur

## UI Testing Checklist

### Initial Load
- [ ] Page loads without errors
- [ ] "New Game" modal appears
- [ ] No console errors in browser dev tools

### Game Creation
- [ ] Can configure number of opponents (1-9)
- [ ] Can select opponent type (trained/call/random/mixed)
- [ ] Can set starting stack and blinds
- [ ] Click "Start Game" creates game successfully
- [ ] Poker table appears with correct number of players

### Visual Display
- [ ] Poker table renders with green felt background
- [ ] Players arranged in ellipse around table
- [ ] Human player's cards are visible
- [ ] Opponent cards are face-down
- [ ] Pot displays in center
- [ ] Community cards area visible
- [ ] Betting round displayed (PREFLOP/FLOP/TURN/RIVER)

### Player Information
- [ ] Each player shows name, stack, current bet
- [ ] Dealer button (D) shows on correct player
- [ ] Current player highlighted
- [ ] Folded players marked
- [ ] All-in players marked

### Action Controls (When Your Turn)
- [ ] Action panel appears at bottom
- [ ] "Fold" button works
- [ ] "Check" button shows when no bet
- [ ] "Call $X" button shows correct amount
- [ ] Raise slider appears and adjusts amount
- [ ] "Raise" button works with slider amount
- [ ] "All-In" button works

### Game Flow
- [ ] Bots take actions automatically with delays
- [ ] Game state updates in real-time
- [ ] Pot increases correctly
- [ ] Community cards appear at correct times:
  - Flop: 3 cards
  - Turn: 4th card
  - River: 5th card
- [ ] Hand completes when appropriate

### Hand Completion
- [ ] Hand result modal appears
- [ ] Shows winner(s) and amount won
- [ ] Shows winning cards at showdown
- [ ] "Next Hand" button works
- [ ] New hand starts correctly
- [ ] Stacks update from previous hand

### Sidebar Features
- [ ] Hand history displays current hand info
- [ ] Opponent stats area visible
- [ ] Game info shows hand number, blinds, player count

### Error Handling
- [ ] Network errors show toast notifications
- [ ] Invalid actions prevented by UI
- [ ] Disconnected WebSocket shows status

### Multiple Hands Test
Play 5 hands in a row and verify:
- [ ] Dealer button rotates correctly
- [ ] Blinds post correctly each hand
- [ ] Stacks persist across hands
- [ ] No memory leaks (check browser performance)

## Common Issues and Solutions

### Backend Won't Start

**Issue**: `ModuleNotFoundError: No module named 'src'`
**Solution**: Run from LagBot root with `PYTHONPATH=. uvicorn backend.main:app --reload`

**Issue**: `No module named 'fastapi'`
**Solution**: Install backend dependencies: `pip install -r backend/requirements.txt`

### Frontend Won't Start

**Issue**: `npm: command not found`
**Solution**: Install Node.js from https://nodejs.org/

**Issue**: Build errors
**Solution**:
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
```

### WebSocket Connection Failed

**Issue**: WebSocket connection refused
**Solution**:
1. Check backend is running on port 8000
2. Check browser console for CORS errors
3. Verify proxy settings in `vite.config.ts`

### Cards Not Displaying

**Issue**: Cards show as plain text or don't render
**Solution**: Card components use CSS and text symbols - check Tailwind CSS is loaded

### Bots Not Moving

**Issue**: Game freezes after player action
**Solution**:
1. Check backend console for errors
2. Verify bot agents loading correctly
3. Check if trained models exist in `models/ppo/` (will fallback to CallAgent)

## Performance Testing

### Load Test (Optional)

Test multiple concurrent games:

```bash
# Install artillery if needed
npm install -g artillery

# Create test-config.yml:
cat > test-config.yml << EOF
config:
  target: "http://localhost:8000"
  phases:
    - duration: 60
      arrivalRate: 5
scenarios:
  - name: "Create games"
    flow:
      - post:
          url: "/api/game/new"
          json:
            num_opponents: 2
            opponent_type: "call"
            starting_stack: 1000
            small_blind: 5
            big_blind: 10
EOF

# Run load test
artillery run test-config.yml
```

## Success Criteria

All tests should pass:
- ✅ Backend starts without errors
- ✅ Frontend starts without errors
- ✅ Can create new game
- ✅ WebSocket connects successfully
- ✅ Can play complete hand from start to finish
- ✅ Bots take actions automatically
- ✅ UI updates in real-time
- ✅ Hand results display correctly
- ✅ Can play multiple consecutive hands
- ✅ No console errors during normal gameplay

## Reporting Issues

If you encounter issues:

1. Check browser console for errors (F12 → Console tab)
2. Check backend terminal for error messages
3. Verify all dependencies installed
4. Check README_WEB.md for setup instructions
5. Try the Docker setup if manual setup fails
