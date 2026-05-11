# LagBot

A Texas Hold'em poker bot trained with PPO (Proximal Policy Optimization). Play against the bot through a web interface, or train new models from scratch.

---

## Quick Start

### Environment setup

Dependencies are managed with [uv](https://docs.astral.sh/uv/). Install it once (`curl -LsSf https://astral.sh/uv/install.sh | sh`), then:

```bash
# Install everything (core + backend + dev tools). uv auto-installs Python 3.11 if missing.
uv sync --all-groups

# Or install only what you need:
uv sync               # core + dev (training, tests)
uv sync --group backend  # core + dev + backend server
```

This creates `.venv/` and is fully reproducible from `uv.lock`. `uv run <cmd>` runs commands inside it without needing to manually activate.

### Play via Web

```bash
# Option 1: Docker Compose (recommended)
docker compose up
# Open http://localhost:5173

# Option 2: Manual
PYTHONPATH=. uv run uvicorn backend.main:app --host 0.0.0.0 --port 8000

# In a second terminal:
cd frontend && npm run dev
# Open http://localhost:5173
```

### Train a Model

```bash
uv run python train.py --config configs/default_config.yaml --name my_run
```

### Play from CLI

```bash
PYTHONPATH=. uv run python play.py
```

### Run tests

```bash
uv run pytest
```

---

## Project Structure

```
LagBot/
├── train.py                    # Main training entry point
├── train_from_checkpoint.py    # Resume training from checkpoint
├── train_diverse_opponents.py  # Train vs rule-based opponents
├── train_vs_two_bots.py        # Train vs two specific models
├── play.py                     # Interactive CLI game
│
├── src/                        # Core poker engine
│   └── poker_env/              # Environment, game state, hand evaluator, agents
│
├── backend/                    # FastAPI REST + WebSocket server
│   ├── api/                    # Routes + WebSocket handler
│   ├── services/               # GameSession + GameManager
│   ├── db/                     # PostgreSQL hand history
│   └── utils/                  # State serializer, card converter
│
├── frontend/                   # React 18 + TypeScript UI
│   └── src/
│       ├── components/         # PokerTable, Controls, Modals, Sidebar
│       ├── stores/             # Zustand game state store
│       └── hooks/              # useWebSocket
│
├── configs/                    # Training config YAML files
├── models/                     # Trained model checkpoints
│   └── archive/                # Old model backups
├── scripts/                    # Utility and debug scripts
├── tests/                      # All tests
├── docs/                       # Documentation
│   └── images/                 # Training charts and screenshots
├── logs/                       # TensorBoard logs
├── metrics/                    # Training metrics JSON
└── claude_workflow/            # Development history and plans
```

---

## Architecture

```
Browser
  │
  ├── REST /api/*   ──► FastAPI (port 8000)
  └── WebSocket /ws/*      │
                           ├── GameSession (wraps TexasHoldemEnv)
                           │   └── Bot agents (PPO / Call / Random)
                           └── PostgreSQL (hand history)
```

In development, Vite (port 5173) proxies `/api` and `/ws` to FastAPI. In production, Nginx handles this.

---

## Docs

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — System design and data flow
- [`docs/TRAINING_GUIDE.md`](docs/TRAINING_GUIDE.md) — How to train models
- [`docs/GPU_TRAINING_GUIDE.md`](docs/GPU_TRAINING_GUIDE.md) — Apple Silicon / CUDA setup
- [`docs/QUICK_START.md`](docs/QUICK_START.md) — Quick reference
- [`docs/TESTING.md`](docs/TESTING.md) — Testing procedures
- [`docs/CHANGELOG.md`](docs/CHANGELOG.md) — History of major changes
- [`claude_workflow/`](claude_workflow/) — Development research, plans, and work logs

---

## Tech Stack

| Layer | Tech |
|-------|------|
| Game Engine | Python, Gymnasium, Stable-Baselines3 PPO, Treys |
| Backend | FastAPI, Uvicorn, asyncpg, PostgreSQL |
| Frontend | React 18, TypeScript, Vite, Zustand, Tailwind CSS |
| Deployment | Docker Compose |
