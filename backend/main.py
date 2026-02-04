"""
FastAPI backend for LagBot poker web interface.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.api import routes, websocket

app = FastAPI(
    title="LagBot Poker API",
    description="REST API and WebSocket server for poker game interface",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(routes.router, prefix="/api")
app.include_router(websocket.router)

@app.get("/")
async def root():
    return {"message": "LagBot Poker API", "status": "running"}

@app.get("/health")
async def health():
    return {"status": "healthy"}
