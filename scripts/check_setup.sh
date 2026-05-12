#!/bin/bash

# LagBot Setup Verification Script

echo "LagBot Web Interface Setup Check"
echo "================================="
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track if everything is OK
ALL_OK=true

# Check Python
echo -n "Checking Python... "
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo -e "${GREEN}✓${NC} $PYTHON_VERSION"
else
    echo -e "${RED}✗${NC} Python3 not found"
    ALL_OK=false
fi

# Check pip
echo -n "Checking pip... "
if command -v pip3 &> /dev/null; then
    PIP_VERSION=$(pip3 --version | cut -d' ' -f2)
    echo -e "${GREEN}✓${NC} pip $PIP_VERSION"
else
    echo -e "${RED}✗${NC} pip3 not found"
    ALL_OK=false
fi

# Check Node.js
echo -n "Checking Node.js... "
if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version)
    echo -e "${GREEN}✓${NC} $NODE_VERSION"
else
    echo -e "${RED}✗${NC} Node.js not found"
    echo -e "   ${YELLOW}Install from https://nodejs.org/${NC}"
    ALL_OK=false
fi

# Check npm
echo -n "Checking npm... "
if command -v npm &> /dev/null; then
    NPM_VERSION=$(npm --version)
    echo -e "${GREEN}✓${NC} npm $NPM_VERSION"
else
    echo -e "${RED}✗${NC} npm not found"
    ALL_OK=false
fi

echo ""
echo "Checking project structure..."

# Check backend files
echo -n "Backend files... "
if [ -f "backend/main.py" ]; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC} backend/main.py not found"
    ALL_OK=false
fi

# Check frontend files
echo -n "Frontend files... "
if [ -f "frontend/package.json" ]; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC} frontend/package.json not found"
    ALL_OK=false
fi

# Check if backend dependencies installed
echo -n "Backend dependencies... "
if pip3 list 2>/dev/null | grep -q "fastapi"; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${YELLOW}?${NC} FastAPI not found"
    echo "   Run: pip install -r backend/requirements.txt"
fi

# Check if frontend dependencies installed
echo -n "Frontend dependencies... "
if [ -d "frontend/node_modules" ]; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${YELLOW}?${NC} node_modules not found"
    echo "   Run: cd frontend && npm install"
fi

echo ""
echo "Checking ports..."

# Check if port 8000 is available
echo -n "Port 8000 (backend)... "
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo -e "${YELLOW}!${NC} Port 8000 already in use"
else
    echo -e "${GREEN}✓${NC} Available"
fi

# Check if port 5173 is available
echo -n "Port 5173 (frontend)... "
if lsof -Pi :5173 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo -e "${YELLOW}!${NC} Port 5173 already in use"
else
    echo -e "${GREEN}✓${NC} Available"
fi

echo ""
echo "================================="

if [ "$ALL_OK" = true ]; then
    echo -e "${GREEN}All required tools are installed!${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Install backend dependencies: pip install -r backend/requirements.txt"
    echo "2. Install frontend dependencies: cd frontend && npm install"
    echo "3. Start servers: ./start_web.sh"
else
    echo -e "${RED}Some requirements are missing. Please install them first.${NC}"
    exit 1
fi
