# Plan: AWS Deployment for LagBot Poker

Deploy the poker app to a single EC2 instance so anyone on the internet can play against your trained bot.

---

## Architecture Decision: Single EC2 with Docker Compose

Based on the research, we're going with **Option A: Single EC2 Instance**. The reasoning is simple: low traffic (friends, not the public), cost-sensitive ($30/mo not $130/mo), and Docker Compose already works locally. We can always upgrade later if needed.

```
Internet
   │
   ▼
[Route 53 or Cloudflare DNS]
   │  lagbot.yourdomain.com → 54.x.x.x (Elastic IP)
   │
   ▼
[EC2 t3.medium — Ubuntu 22.04 — 2 vCPU, 4GB RAM]
   │
   └── Docker Compose (docker-compose.prod.yml)
       │
       ├── nginx (ports 80 + 443 exposed to internet)
       │   ├── HTTPS termination (Let's Encrypt certs)
       │   ├── Serves frontend/dist/* (static React build)
       │   ├── Proxies /api/* → backend:8000
       │   ├── Proxies /ws/* → backend:8000 (WebSocket upgrade)
       │   └── Security headers
       │
       ├── backend (port 8000, internal only — NOT exposed to internet)
       │   ├── Gunicorn + 1 Uvicorn worker
       │   ├── FastAPI REST API (game creation, actions, stats)
       │   ├── FastAPI WebSocket server (real-time state updates)
       │   ├── PyTorch model inference (CPU-only, ~200MB model)
       │   └── Connects to db:5432
       │
       └── db (port 5432, internal only)
           └── PostgreSQL 16 (hand history)
```

---

## Where Our Existing Code Lives in This Architecture

This section maps every piece of our current codebase to where it ends up in production.

### What Goes Into the Backend Docker Image

The production Dockerfile builds a **single image** that contains the backend, game engine, trained model, AND the pre-built frontend static files:

```
Docker image: lagbot-backend
│
├── /app/backend/                 ← Our backend/ directory
│   ├── main.py                   ← FastAPI app entry point (starts here)
│   ├── api/routes.py             ← REST endpoints (/api/game/new, /action, etc.)
│   ├── api/websocket.py          ← WebSocket handler (/ws/{session_id})
│   ├── services/game_session.py  ← Wraps TexasHoldemEnv, manages bot agents
│   ├── services/game_manager.py  ← Session factory (singleton)
│   ├── db/database.py            ← asyncpg connection pool → connects to db container
│   ├── db/hand_history.py        ← Saves/queries completed hands
│   ├── db/schema.sql             ← Auto-creates tables on first connect
│   ├── utils/state_serializer.py ← Converts game state → JSON for frontend
│   └── utils/card_converter.py   ← Treys int ↔ "Ah" string format
│
├── /app/src/                     ← Our src/ directory (the poker engine)
│   └── poker_env/
│       ├── texas_holdem_env.py   ← The Gym environment (125-dim obs, 6 actions)
│       ├── game_state.py         ← Hand lifecycle state machine
│       ├── hand_evaluator.py     ← Treys wrapper for hand ranking
│       ├── opponent_tracker.py   ← Per-opponent VPIP/PFR/AF stats
│       ├── player.py             ← Player state (stack, cards, bets)
│       └── pot_manager.py        ← Pot/side-pot/raise calculations
│   └── agents/
│       ├── opponent_ppo.py       ← Loads trained .zip model for bot play
│       ├── random_agent.py       ← CallAgent / RandomAgent (baselines)
│       └── base_agent.py         ← Abstract agent interface
│
├── /app/models/                  ← Our models/ directory
│   └── vs_v2_and_new_.../
│       └── model_6650000_steps.zip  ← The trained bot (~200MB)
│
└── /app/frontend/dist/           ← Built by npm run build in Stage 1
    ├── index.html                ← React SPA entry point
    └── assets/
        ├── index-abc123.js       ← All React code + dependencies (~220KB)
        └── index-xyz789.css      ← All Tailwind styles (~26KB)
```

### What Nginx Serves Directly

Nginx serves the static frontend files WITHOUT going through Python. This is 10-100x faster than having FastAPI serve them:

```
nginx container
│
├── /usr/share/nginx/html/        ← Mounted from backend's frontend/dist/
│   ├── index.html                ← The React SPA
│   └── assets/
│       ├── index-*.js            ← React bundle
│       └── index-*.css           ← Styles
│
├── /etc/nginx/conf.d/lagbot.conf ← Our nginx config
│   ├── location / { ... }        ← Serves static files
│   ├── location /api/ { ... }    ← Proxies to backend:8000
│   ├── location /ws/ { ... }     ← Proxies WebSocket to backend:8000
│   └── location /health { ... }  ← Proxies health check
│
└── /etc/letsencrypt/             ← SSL certificates (mounted volume)
    └── live/lagbot.yourdomain.com/
        ├── fullchain.pem
        └── privkey.pem
```

### What Stays on the EC2 Host (Not in Docker)

```
/opt/lagbot/                      ← Project root on EC2
├── docker-compose.prod.yml       ← Production compose file
├── .env                          ← Secrets (DB password, domain name)
├── nginx/
│   └── conf.d/lagbot.conf        ← Nginx server config
└── certbot/
    ├── conf/                     ← Let's Encrypt certificates
    └── www/                      ← ACME challenge files
```

### What Does NOT Get Deployed

These stay on your local machine only:

```
NOT deployed:
├── train.py, train_*.py          ← Training scripts (training runs locally)
├── play.py                       ← CLI game (replaced by web interface)
├── scripts/                      ← Debug/analysis utilities
├── tests/                        ← Test suite
├── configs/                      ← Training YAML configs
├── logs/                         ← TensorBoard logs
├── metrics/                      ← Training metrics
├── claude_workflow/              ← Development docs
├── venv/                         ← Local Python virtualenv
├── docker-compose.yml            ← Dev compose (separate from prod)
├── frontend/src/                 ← Source code (built in Docker, not deployed raw)
└── models/archive/               ← Old model backups
```

---

## How the Request Flow Changes (Local vs Production)

### Currently (Local Development)

```
Browser (localhost:5173)
   │
   ├── GET / → Vite dev server serves React source files (120 separate files)
   ├── POST /api/game/new → Vite proxies to localhost:8000 → FastAPI
   └── WS /ws/{id} → Vite proxies to ws://localhost:8000 → FastAPI
```

**Key file:** `frontend/vite.config.ts` lines 19-28 — the proxy config that forwards `/api` and `/ws` to the backend. This is Vite-specific and only exists in development.

### Production

```
Browser (lagbot.yourdomain.com)
   │
   ├── GET / → Nginx serves index.html from frontend/dist/ (3 files, instant)
   ├── POST /api/game/new → Nginx proxies to backend:8000 → FastAPI
   └── WSS /ws/{id} → Nginx upgrades to WebSocket, proxies to backend:8000
```

**Key difference:** Vite is gone. Nginx replaces it. The frontend is pre-built static files. Nginx handles HTTPS. The backend is unchanged — same FastAPI code, same endpoints, same WebSocket handler.

### Why This Works Without Frontend Code Changes

The frontend API client (`frontend/src/api/client.ts` line 5) uses a relative base URL:
```typescript
const API_BASE = '/api';
```
Not `http://localhost:8000/api`. Just `/api`. This means the browser sends API requests to **whatever host served the page**. Locally that's `localhost:5173` (Vite). In production that's `lagbot.yourdomain.com` (Nginx). Either way, the proxy forwards to FastAPI. Zero frontend changes needed.

The WebSocket hook (`frontend/src/hooks/useWebSocket.ts` line 16-17) does the same thing:
```typescript
const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
const wsUrl = `${protocol}//${window.location.host}/ws/${sessionId}`;
```
Auto-detects HTTPS and uses the current host. Works everywhere without changes.

---

## Decision Points

### Decision 1: Domain Strategy

| Option | Cost | HTTPS | Looks |
|--------|------|-------|-------|
| **Buy a domain** (lagbot.com, playlagbot.com) | ~$12/yr | Yes (Let's Encrypt) | Professional |
| **DuckDNS** (lagbot.duckdns.org) | Free | Yes (Let's Encrypt) | Fine for friends |
| **Bare IP** (http://54.x.x.x) | Free | No | Sketchy, browser warnings |

**Recommendation:** DuckDNS to start (free, HTTPS works). Buy a real domain later if you want.

### Decision 2: Gunicorn Workers

| Option | Behavior | Best For |
|--------|----------|----------|
| **1 worker** | All requests go to one process. Simple. WebSocket sessions and REST share state naturally. | LagBot (low traffic, needs shared GameManager singleton) |
| **2 workers** | Better throughput, but WebSocket on worker 1, REST on worker 2 = different GameManager instances = broken. | Would need Redis for shared state. Overkill. |

**Recommendation:** 1 worker. The GameManager singleton (`backend/services/game_manager.py`) holds all active game sessions in memory. Multiple workers = multiple singletons = a player's REST calls can't find their WebSocket session. For 5-10 friends, 1 worker is plenty.

### Decision 3: PyTorch Image Strategy

| Option | Image Size | Deploy Speed | Flexibility |
|--------|-----------|-------------|-------------|
| **CPU-only PyTorch** (`--index-url .../cpu`) | ~800MB | Fast | Need to rebuild for new PyTorch version |
| **Full PyTorch** (default pip install) | ~2.5GB | Slow | Works with GPU if you ever add one |

**Recommendation:** CPU-only. We're doing inference only (`model.predict()`), not training. CPU inference for a 9M-param model takes <10ms per action. No GPU needed.

### Decision 4: Model Delivery

| Option | Image Size | Hot-Swap | Complexity |
|--------|-----------|----------|-----------|
| **Baked into Docker image** | +200MB | Rebuild to update | Simplest |
| **Mounted from EC2 host** | Same | Copy new file, restart | Medium |
| **Pull from S3 at startup** | Same | Update S3, restart | Most flexible |

**Recommendation:** Bake into Docker image for now. The model rarely changes (you'd retrain locally and redeploy). If you want to hot-swap models later, switch to S3.

---

## Phases

### Phase 0: Production Dockerfile (Multi-Stage Build)

Create a root-level `Dockerfile.prod` that builds the entire app in one image.

**Stage 1 — Build frontend:**
- `FROM node:18-alpine`
- `npm ci` + `npm run build` → produces `dist/` with static files
- This stage is thrown away after the build — only `dist/` is kept

**Stage 2 — Production backend:**
- `FROM python:3.10-slim`
- Install CPU-only PyTorch + backend requirements
- Copy `backend/`, `src/`, `models/` (game engine + trained model)
- Copy `frontend/dist/` from Stage 1
- Entrypoint: `gunicorn` with 1 Uvicorn worker

**What this replaces:** The existing `backend/Dockerfile` (dev mode, `--reload`, missing `src/` and `models/`) and `frontend/Dockerfile` (runs `npm run dev`, unnecessary in production).

**Files to create:**
- `Dockerfile.prod` (root level)

### Phase 1: Nginx Configuration

Create the Nginx config that replaces Vite as the reverse proxy.

**What it does:**
- Listens on 80 (HTTP → redirect to HTTPS) and 443 (HTTPS)
- Serves `frontend/dist/` for all non-API paths (React SPA — all routes serve index.html)
- Proxies `/api/*` to `backend:8000` (REST)
- Proxies `/ws/*` to `backend:8000` with WebSocket upgrade headers
- Sets `proxy_read_timeout 86400` for WebSocket (24 hours, prevents disconnect while player thinks)
- Includes Let's Encrypt ACME challenge location for certificate renewal
- Security headers (X-Frame-Options, X-Content-Type-Options, etc.)

**Files to create:**
- `nginx/conf.d/lagbot.conf`

### Phase 2: Production Docker Compose

Create `docker-compose.prod.yml` — the production version of our existing `docker-compose.yml`.

**Differences from dev compose:**
| Dev | Prod |
|-----|------|
| Volume mounts (`./backend:/app/backend`) for hot reload | No mounts — code baked into image |
| `--reload` flag on uvicorn | Gunicorn, no reload |
| Frontend container (Vite dev server) | No frontend container — Nginx serves static files |
| DB port exposed (`5432:5432`) | DB port internal only (no `ports:` mapping) |
| Hardcoded passwords | Read from `.env` file |
| No restart policy | `restart: always` on all services |
| No health checks | Health checks on backend and db |

**Services:**
1. **nginx** — `nginx:alpine`, mounts our config + certbot certs + frontend/dist from backend
2. **backend** — Built from `Dockerfile.prod`, internal port 8000 only
3. **db** — `postgres:16-alpine`, internal port 5432 only, persistent volume

**Files to create:**
- `docker-compose.prod.yml`
- `.env.example` (template for production secrets)

### Phase 3: Backend Code Changes

Small changes to make the backend production-ready.

**3.1 CORS from environment variable**

Currently hardcoded in `backend/main.py` line 46:
```python
allow_origins=["http://localhost:5173", "http://localhost:3000"]
```

Change to read from env var, with localhost as fallback for dev:
```python
import os
origins = os.environ.get("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",")
allow_origins=origins
```

This way, `docker-compose.prod.yml` sets `CORS_ORIGINS=https://lagbot.yourdomain.com` and dev still works without any env var.

**3.2 Add gunicorn to backend/requirements.txt**

Add `gunicorn>=21.2.0` — the production WSGI/ASGI server that manages Uvicorn workers.

**Files to modify:**
- `backend/main.py` (CORS env var)
- `backend/requirements.txt` (add gunicorn)

### Phase 4: EC2 Setup Script

A script that runs on a fresh Ubuntu 22.04 EC2 instance to install everything needed.

**What it installs:**
- Docker Engine + Docker Compose plugin
- Certbot (for Let's Encrypt SSL)
- Creates `/opt/lagbot/` directory structure
- Sets up UFW firewall (allow 22, 80, 443 only)

**Files to create:**
- `scripts/setup-ec2.sh`

### Phase 5: Deploy Script

A script you run from your local machine to deploy to the EC2 instance.

**What it does:**
1. Builds the Docker image locally (or on EC2)
2. SSHs into the EC2 instance
3. Pulls latest code
4. Builds and restarts containers
5. Verifies the health check passes

**Two strategies:**

**Option A — Build on EC2 (simpler):**
```bash
ssh ec2 "cd /opt/lagbot && git pull && docker compose -f docker-compose.prod.yml up -d --build"
```
Builds the image on the EC2 instance. Slower (t3.medium has 2 vCPUs) but no need for a Docker registry.

**Option B — Build locally, push to ECR:**
```bash
docker build -f Dockerfile.prod -t lagbot .
docker tag lagbot 123456.dkr.ecr.us-east-1.amazonaws.com/lagbot:latest
docker push ...
ssh ec2 "cd /opt/lagbot && docker compose -f docker-compose.prod.yml pull && docker compose up -d"
```
Faster deploys (image is pre-built on your powerful Mac), but requires setting up ECR.

**Recommendation:** Start with Option A. Add ECR later if build times bother you.

**Files to create:**
- `scripts/deploy.sh`

### Phase 6: SSL Certificate Setup

Get HTTPS working with Let's Encrypt.

**The bootstrap problem:** Certbot needs to verify you own the domain by placing a file at `http://yourdomain.com/.well-known/acme-challenge/xxx`. But our Nginx config expects SSL certs to exist at startup. Chicken-and-egg.

**Solution:** Two-step process:
1. First run: start Nginx with HTTP only (no SSL), run Certbot to get the certificate
2. Second run: restart Nginx with the real config that includes HTTPS

The deploy script handles this automatically on first deploy.

**No new files** — handled within the Nginx config (HTTP-only fallback) and deploy script.

### Phase 7: DNS + Elastic IP

Manual AWS console steps (documented in the deploy script output):
1. Allocate an Elastic IP
2. Associate it with the EC2 instance
3. Create a DNS record pointing your domain to the Elastic IP

If using DuckDNS:
```bash
curl "https://www.duckdns.org/update?domains=lagbot&token=YOUR_TOKEN&ip="
```

If using Route 53:
- Create hosted zone for your domain
- Add A record: `lagbot.yourdomain.com` → Elastic IP

---

## Files Summary

### New Files to Create

| File | Purpose |
|------|---------|
| `Dockerfile.prod` | Multi-stage production build (frontend build → backend + engine + model) |
| `docker-compose.prod.yml` | Production compose (nginx + backend + db, restart policies, no volume mounts) |
| `nginx/conf.d/lagbot.conf` | Nginx reverse proxy config (HTTPS, static files, /api proxy, /ws WebSocket proxy) |
| `.env.example` | Template for production env vars (DB password, domain, CORS origins) |
| `scripts/setup-ec2.sh` | First-time EC2 instance setup (Docker, certbot, firewall) |
| `scripts/deploy.sh` | Deploy from local to EC2 (git pull, build, restart, verify) |

### Existing Files to Modify

| File | Change | Why |
|------|--------|-----|
| `backend/main.py` | CORS origins from `CORS_ORIGINS` env var | Hardcoded localhost won't work in production |
| `backend/requirements.txt` | Add `gunicorn>=21.2.0` | Production ASGI server |

### Files That Need NO Changes (Already Production-Ready)

| File | Why It Already Works |
|------|---------------------|
| `frontend/src/api/client.ts` | Uses relative URL `/api` — works on any host |
| `frontend/src/hooks/useWebSocket.ts` | Auto-detects `wss://` from `window.location.protocol` |
| `backend/db/database.py` | Reads `DATABASE_URL` env var with localhost fallback |
| `backend/services/game_session.py` | No host/port assumptions — connects via env + Docker networking |
| `backend/api/routes.py` | Standard REST — host-agnostic |
| `backend/api/websocket.py` | Standard WebSocket — host-agnostic |
| All `src/poker_env/*.py` | Pure game logic — no network awareness at all |

---

## AWS Resources to Create (Console / CLI)

| Resource | Type | Cost | Purpose |
|----------|------|------|---------|
| EC2 instance | t3.medium | ~$30/mo | The server |
| Elastic IP | EIP | Free (while attached) | Static public IP |
| Security Group | SG | Free | Firewall (22, 80, 443 only) |
| Key Pair | SSH key | Free | SSH access |
| Route 53 hosted zone (optional) | DNS | $0.50/mo | Domain management |
| **Total** | | **~$32/mo** | |

---

## Todo

- [ ] Create `Dockerfile.prod` (multi-stage: frontend build + backend + model)
- [ ] Create `nginx/conf.d/lagbot.conf`
- [ ] Create `docker-compose.prod.yml`
- [ ] Create `.env.example`
- [ ] Modify `backend/main.py` — CORS from env var
- [ ] Add gunicorn to `backend/requirements.txt`
- [ ] Create `scripts/setup-ec2.sh`
- [ ] Create `scripts/deploy.sh`
- [ ] Test production build locally (`docker compose -f docker-compose.prod.yml up`)
- [ ] Launch EC2 instance + Elastic IP
- [ ] Set up DNS (DuckDNS or Route 53)
- [ ] Run setup-ec2.sh on the instance
- [ ] Deploy and get SSL certificate
- [ ] Verify: HTTPS, REST API, WebSocket, game plays correctly

---

## Explanations

### Why Do We Need a Separate Production Docker Compose?

The dev `docker-compose.yml` is designed for fast iteration:
- It mounts your local code into the container (`volumes: ./backend:/app/backend`) so changes appear instantly
- It runs `--reload` to auto-restart when files change
- It runs the Vite dev server for the frontend (hot module replacement)
- It exposes the DB port so you can connect with pgAdmin or psql

None of this makes sense in production. You don't edit code on the server. You don't need hot reload. You don't want the DB exposed to the internet. The production compose is hardened:
- Code is baked into the image (immutable)
- Gunicorn runs without reload
- Frontend is pre-built static files served by Nginx
- Only ports 80 and 443 are exposed
- Containers auto-restart on crash

### Why Gunicorn + Uvicorn Instead of Just Uvicorn?

Uvicorn is the ASGI server that speaks HTTP and WebSocket. It's what actually runs your FastAPI code. But it's a single process.

Gunicorn is a process manager. It starts and monitors worker processes. If a worker crashes, Gunicorn restarts it. If the whole server crashes, Docker's `restart: always` restarts Gunicorn, which restarts the worker.

Think of it as: Gunicorn is the babysitter, Uvicorn is the kid doing the actual work. In development, the kid runs unsupervised and it's fine. In production, you want the babysitter watching.

For LagBot we use 1 worker (because of the GameManager singleton), so the value of Gunicorn is purely crash recovery and graceful shutdown — not parallelism.

### What Happens When the Model Changes?

When you train a better model locally:
1. Run training on your Mac: `python train.py --config configs/...`
2. New model saved to `models/new_run_*/model_*_steps.zip`
3. Commit (or just have it in the repo)
4. Redeploy: `./scripts/deploy.sh`
5. The Docker build copies the new model into the image
6. The backend loads the new model on startup via `_find_latest_model()` in `game_session.py`

Players will face the new bot on their next game after the deploy. Existing active games will finish with the old bot (they're in memory), then new games get the new model.

### What If the EC2 Instance Runs Out of Memory?

PyTorch inference for a 9M-param model uses ~200-400MB of RAM. PostgreSQL uses ~50-100MB. Nginx uses ~5MB. Total: ~300-500MB.

On a t3.medium (4GB RAM), that leaves ~3.5GB for game sessions. Each active game session (the `TexasHoldemEnv` + player states + observations) uses ~5-10MB. You could comfortably run **100+ concurrent games** before memory becomes an issue.

If you DO run out of memory:
1. Check `docker stats` to see which container is using the most
2. Restart the backend: `docker compose -f docker-compose.prod.yml restart backend`
3. If it keeps happening, upgrade to t3.large (8GB, ~$60/mo)

### Why Not Just Use FastAPI to Serve Static Files?

FastAPI CAN serve static files (it already does — `backend/main.py` lines 61-70). So why add Nginx?

1. **Speed:** Nginx serves static files directly from disk using kernel-level `sendfile()`. Python reads the file into memory, then sends it. Nginx is 10-100x faster for static files.
2. **HTTPS:** Nginx handles TLS encryption natively. Making FastAPI handle HTTPS requires certificate management in Python — messy.
3. **WebSocket timeout:** Nginx lets you set `proxy_read_timeout 86400` (24 hours) for WebSocket paths. FastAPI/Uvicorn doesn't have this granularity.
4. **Security:** Nginx is battle-tested for internet-facing traffic. It handles malformed requests, slowloris attacks, oversized headers, etc. FastAPI is an application framework, not a security perimeter.

In development we skip Nginx because Vite already handles all these jobs (except HTTPS). In production, Nginx fills the Vite-shaped hole.
