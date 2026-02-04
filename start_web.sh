#!/bin/bash

# LagBot Web Interface Startup Script

echo "Starting LagBot Poker Web Interface..."
echo "======================================="
echo ""

# Check if we're in the right directory
if [ ! -f "backend/main.py" ]; then
    echo "Error: Please run this script from the LagBot root directory"
    exit 1
fi

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Shutting down servers..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    exit 0
}

trap cleanup SIGINT SIGTERM

# Start backend
echo "Starting backend server..."
cd "$(dirname "$0")"
source venv/bin/activate 2>/dev/null || true
PYTHONPATH=. uvicorn backend.main:app --reload --port 8000 &
BACKEND_PID=$!
echo "Backend started on http://localhost:8000 (PID: $BACKEND_PID)"
echo "API docs available at http://localhost:8000/docs"
echo ""

# Wait a bit for backend to start
sleep 2

# Start frontend
echo "Starting frontend server..."
cd frontend
npm run dev &
FRONTEND_PID=$!
echo "Frontend started on http://localhost:5173 (PID: $FRONTEND_PID)"
echo ""

echo "======================================="
echo "Both servers are running!"
echo "Open http://localhost:5173 in your browser"
echo "Press Ctrl+C to stop both servers"
echo "======================================="

# Wait for processes
wait
