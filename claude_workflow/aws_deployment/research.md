# Research: AWS Deployment for LagBot Poker

Making the poker app accessible from the public internet so anyone can play, not just people on your local WiFi.
---

## 1. What We're Deploying

LagBot is a 3-service application:

| Service | Tech | Port | Role |
|---------|------|------|------|
| **Frontend** | React 18 + Vite (static build) | 5173 (dev) / 4173 (preview) | Serves the poker UI |
| **Backend** | FastAPI + Uvicorn | 8000 | REST API + WebSocket server + game engine + bot agents |
| **Database** | PostgreSQL 16 | 5432 | Hand history persistence (optional — app works without it) |

The backend is the heavy piece — it runs PyTorch models (~9M params) for the trained poker bots. The frontend is just static files after `npm run build` (~250KB total). PostgreSQL is lightweight for our use case.

### Current State

- Docker Compose exists (`docker-compose.yml`) with all 3 services
- Dockerfiles exist for backend and frontend
- Backend already serves the frontend static build from `frontend/dist/` (see `backend/main.py` lines 61-70) — so in production we only need ONE server process
- CORS is hardcoded to `localhost:5173` and `localhost:3000` — needs updating for production
- WebSocket proxy is configured in Vite dev server — in production the backend handles WebSockets directly
- Frontend API client uses relative URLs (`/api/...`) — works with any host
- Health check endpoint exists: `GET /health`

### Key Constraint: PyTorch + Trained Models

The backend needs PyTorch and a ~200MB trained model file (`models/vs_v2_and_new_20260212_182022/model_6650000_steps.zip`). This means:
- Docker image will be large (~2-3GB with PyTorch)
- Instance needs enough RAM for PyTorch inference (not training — just running `model.predict()`)
- No GPU needed for inference — CPU is fine for running 1-10 concurrent games
- The model file needs to be baked into the Docker image or pulled from S3 at startup

---

## 2. Deployment Options

### 2.1 Option A: Single EC2 Instance (Simplest)

**Concept:** One EC2 instance running Docker Compose. Nginx reverse proxy in front for HTTPS and domain routing. Cheapest, simplest, most control.

**Architecture:**
```
Internet
   │
   ▼
[Route 53 DNS] → lagbot.yourdomain.com
   │
   ▼
[EC2 Instance (t3.medium / t3.large)]
   ├── Nginx (port 80/443)
   │   ├── SSL termination (Let's Encrypt)
   │   ├── Serves frontend static files
   │   ├── Proxies /api/* → FastAPI :8000
   │   └── Proxies /ws/* → FastAPI :8000 (WebSocket upgrade)
   │
   ├── FastAPI + Uvicorn (port 8000)
   │   ├── REST API
   │   ├── WebSocket server
   │   └── PyTorch model inference
   │
   └── PostgreSQL (port 5432)
       └── Hand history
```

**Instance sizing:**
| Instance | vCPU | RAM | Cost (on-demand) | Cost (reserved 1yr) | Notes |
|----------|------|-----|-------------------|---------------------|-------|
| t3.small | 2 | 2GB | ~$15/mo | ~$10/mo | Tight — PyTorch might OOM |
| t3.medium | 2 | 4GB | ~$30/mo | ~$20/mo | Minimum viable for PyTorch inference |
| t3.large | 2 | 8GB | ~$60/mo | ~$40/mo | Comfortable — handles 5-10 concurrent games |
| t3.xlarge | 4 | 16GB | ~$120/mo | ~$80/mo | Overkill unless you expect heavy traffic |

**Pros:**
- Cheapest option (~$30/mo for t3.medium)
- Full control over everything
- Simple mental model — one box, SSH in, fix things
- Docker Compose already exists, minimal changes
- Easy to start/stop to save money (only pay when running)

**Cons:**
- No auto-scaling — if 50 people hit it, it dies
- Single point of failure — instance goes down, site goes down
- Manual SSL setup (Let's Encrypt + certbot)
- Manual updates (SSH in, pull, rebuild, restart)

**Estimated effort:** 3-4 hours to get running, half a day with domain + HTTPS.

---

### 2.2 Option B: ECS Fargate (Managed Containers)

**Concept:** AWS runs your Docker containers for you. No EC2 instances to manage. Auto-scales, auto-heals. Costs more, but zero server management.

**Architecture:**
```
Internet
   │
   ▼
[Route 53 DNS] → lagbot.yourdomain.com
   │
   ▼
[Application Load Balancer (ALB)]
   ├── HTTPS termination (ACM certificate — free)
   ├── Routes /api/* and /ws/* → Backend Service
   └── Routes /* → Frontend (S3 + CloudFront, or same backend)

[ECS Fargate Cluster]
   ├── Backend Service (1-3 tasks)
   │   └── FastAPI container (2 vCPU, 4GB RAM)
   │       ├── REST + WebSocket
   │       └── PyTorch inference
   │
   └── (Optional) Frontend served from backend or S3+CloudFront

[RDS PostgreSQL]
   └── db.t3.micro ($15/mo) or Aurora Serverless v2
```

**Cost breakdown:**
| Component | Monthly Cost |
|-----------|-------------|
| Fargate (1 task, 2 vCPU, 4GB) | ~$70-90 |
| ALB | ~$20-25 |
| RDS db.t3.micro | ~$15 |
| Route 53 | ~$1 |
| ECR (image storage) | ~$1 |
| **Total** | **~$107-132/mo** |

**Pros:**
- No server management — AWS handles OS patches, container restarts, health checks
- Auto-scaling — add more tasks under load
- ALB handles HTTPS automatically with free AWS Certificate Manager certs
- Built-in health checks and auto-restart if container crashes
- Good resume/portfolio piece — shows cloud-native skills

**Cons:**
- 3-4x more expensive than a single EC2 instance
- More complex to set up (ECS task definitions, ALB target groups, security groups, VPC)
- Fargate cold starts can take 30-60 seconds (though tasks stay warm once running)
- WebSocket stickiness needs ALB configuration
- PyTorch Docker image is large — slow to pull on cold start

**Estimated effort:** 1-2 days. More AWS configuration, but well-documented.

---

### 2.3 Option C: Lightsail (Easiest AWS Option)

**Concept:** AWS Lightsail is "simple EC2." Fixed monthly pricing, built-in firewall, easier console. Think of it as AWS's answer to DigitalOcean.

**Architecture:**
```
Internet
   │
   ▼
[Lightsail Instance ($20/mo)]
   ├── Nginx + Let's Encrypt
   ├── Docker Compose
   │   ├── FastAPI + PyTorch
   │   └── PostgreSQL
   └── Static frontend served by Nginx or FastAPI
```

**Pricing:**
| Plan | vCPU | RAM | Storage | Cost |
|------|------|-----|---------|------|
| $10/mo | 2 | 2GB | 60GB | Tight for PyTorch |
| $20/mo | 2 | 4GB | 80GB | Minimum viable |
| $40/mo | 2 | 8GB | 160GB | Comfortable |

**Pros:**
- Fixed, predictable pricing (no surprise bills)
- Simpler console than full AWS
- 3 months free on some plans
- Static IP included
- Snapshots for backup

**Cons:**
- Same limitations as single EC2 (no auto-scaling)
- Less flexible than EC2 (can't change instance types as freely)
- No native container orchestration
- Lightsail networking is isolated from main AWS VPC by default

**Estimated effort:** 2-3 hours. Very similar to EC2 but with a simpler interface.

---

### 2.4 Option D: EC2 + CloudFront + S3 (Production-Grade)

**Concept:** Static frontend on S3 + CloudFront CDN (globally cached, fast). Backend on EC2 or ECS. Best performance, more moving parts.

**Architecture:**
```
Internet
   │
   ├── lagbot.yourdomain.com (static assets)
   │   └── CloudFront CDN → S3 Bucket
   │       └── frontend/dist/ (index.html, JS, CSS)
   │
   └── api.lagbot.yourdomain.com (API + WebSocket)
       └── ALB → EC2 / ECS
           ├── FastAPI REST
           ├── WebSocket
           └── PyTorch inference
```

**Pros:**
- Frontend loads instantly from CDN edge locations worldwide
- Backend only handles API calls, not static file serving
- Can scale frontend and backend independently
- Professional setup

**Cons:**
- More complex (S3 bucket policy, CloudFront distribution, CORS configuration, separate domains)
- WebSocket connections still go directly to the backend — CDN doesn't help there
- For a poker game, latency isn't really an issue (it's turn-based, not real-time FPS)
- Overkill for the expected traffic level

**Estimated effort:** 1 day on top of EC2/ECS setup.

---

### 2.5 Non-AWS Alternatives (For Context)

| Provider | Option | Cost | Notes |
|----------|--------|------|-------|
| **DigitalOcean** | Droplet (4GB) | $24/mo | Simpler than AWS, good docs |
| **Railway** | Container hosting | ~$20-50/mo | Git push to deploy, very easy |
| **Fly.io** | Container hosting | ~$15-30/mo | Edge deployment, good WebSocket support |
| **Render** | Web service | ~$25/mo | Git push deploy, free PostgreSQL |
| **Hetzner** | VPS (8GB) | ~$7/mo | Cheapest by far, EU-based |

These are simpler than AWS but don't give you the AWS experience/resume value.

---

## 3. Recommended Approach: Single EC2 Instance

For LagBot's use case, **Option A (single EC2)** is the best fit:

1. **Traffic is low** — You're sharing this with friends, not running a SaaS product. 5-10 concurrent users max.
2. **Cost matters** — $30/mo vs $130/mo is significant for a personal project.
3. **Simplicity** — One box, SSH in, everything is there. Easy to debug.
4. **Docker Compose already works** — Minimal changes needed.
5. **You can always upgrade later** — Start with EC2, move to ECS/Fargate if it takes off.

---c

## 4. What Needs to Change for Production Deployment

### 4.1 Docker Changes

**Backend Dockerfile** — Current one uses `--reload` (dev mode). Production needs:
- Multi-stage build to reduce image size
- No `--reload` flag
- Gunicorn with Uvicorn workers for production (handles multiple requests better)
- Copy `src/` and `models/` into the image (currently only `backend/` is copied)
- The trained model file needs to be in the image or fetched from S3

**Frontend Dockerfile** — Current one runs `npm run dev`. Production needs:
- Build step: `npm run build` to produce static files
- No need for a separate frontend container — FastAPI already serves `frontend/dist/`
- OR use Nginx to serve static files (more efficient)

**docker-compose.yml** — Production version needs:
- Remove volume mounts (no host code in production)
- Add Nginx service
- Add restart policies (`restart: always`)
- Environment variables for secrets (not hardcoded passwords)
- Network isolation (DB not exposed to internet)

### 4.2 Backend Changes

**CORS** (`backend/main.py` line 46):
```python
# Current (dev only):
allow_origins=["http://localhost:5173", "http://localhost:3000"]

# Production:
allow_origins=["https://lagbot.yourdomain.com", "http://localhost:5173"]
```

**Database URL**: Already reads `DATABASE_URL` env var — just needs the right value in production.

**Static file serving**: Already set up in `main.py` lines 61-70. If `frontend/dist/` exists, FastAPI serves it. This means in production we can skip a separate frontend container entirely.

### 4.3 Nginx Configuration

Nginx sits in front of everything, handling:
- HTTPS termination (SSL/TLS with Let's Encrypt)
- Serving static frontend files (faster than Python serving static files)
- Proxying `/api/*` and `/ws/*` to FastAPI
- WebSocket upgrade handling
- Basic rate limiting and security headers

### 4.4 Domain + DNS

Options:
- **Buy a domain** (~$10-15/year from Route 53, Namecheap, or Cloudflare)
- **Free subdomain** via services like `lagbot.duckdns.org` (free dynamic DNS)
- **No domain** — just use the EC2 public IP (works but no HTTPS, looks unprofessional)

For HTTPS, you need a domain. Let's Encrypt won't issue certificates for bare IP addresses.

### 4.5 SSL/HTTPS

**Let's Encrypt + Certbot** — Free, auto-renewing SSL certificates. Standard approach for personal projects. Certbot runs as a cron job to renew every 90 days.

---

## 5. Production Architecture (EC2 Path)

```
Internet
   │
   ▼
[Route 53] → lagbot.yourdomain.com → [Elastic IP]
   │
   ▼
[EC2 t3.medium (Ubuntu 22.04)]
   │
   ├── Docker Compose
   │   │
   │   ├── nginx (port 80 + 443)
   │   │   ├── SSL termination (Let's Encrypt certs mounted)
   │   │   ├── Serves /frontend/dist/* (static React build)
   │   │   ├── Proxy: /api/* → backend:8000
   │   │   ├── Proxy: /ws/* → backend:8000 (WebSocket upgrade)
   │   │   └── Security headers, rate limiting
   │   │
   │   ├── backend (port 8000, internal only)
   │   │   ├── Gunicorn + Uvicorn workers
   │   │   ├── FastAPI REST + WebSocket
   │   │   ├── PyTorch model inference
   │   │   └── Connects to db:5432
   │   │
   │   └── db (port 5432, internal only)
   │       └── PostgreSQL 16
   │
   ├── Certbot (Let's Encrypt renewal)
   │
   └── /opt/lagbot/
       ├── docker-compose.prod.yml
       ├── nginx/
       │   ├── nginx.conf
       │   └── conf.d/lagbot.conf
       ├── .env (secrets)
       └── certbot/
           ├── conf/   (certificates)
           └── www/    (ACME challenge)
```

### What Each Piece Does

**Nginx container**: The front door. Receives all traffic on ports 80 and 443. Serves the React build (static HTML/JS/CSS) directly — much faster than having Python serve files. For any request starting with `/api/` or `/ws/`, it passes the request through to the FastAPI backend. It also handles the HTTPS encryption/decryption so the backend doesn't have to.

**Backend container**: Runs the actual game. FastAPI handles REST requests (create game, take action, get stats) and WebSocket connections (real-time state updates). Loads the trained PyTorch model at startup for bot opponents. Only talks to Nginx and PostgreSQL — not directly exposed to the internet.

**PostgreSQL container**: Stores hand history. The backend connects to it internally. Not exposed to the internet. Data persists in a Docker volume.

**Certbot**: A separate process that obtains and renews free HTTPS certificates from Let's Encrypt. Runs once to get the initial certificate, then renews automatically every 60-90 days.

---

## 6. Networking Deep Dive

### How Traffic Flows from a Friend's Browser to Your Game

```
1. Friend types: https://lagbot.yourdomain.com
   │
2. DNS lookup: lagbot.yourdomain.com → 54.123.45.67 (your EC2 Elastic IP)
   │
3. TCP connection to 54.123.45.67:443 (HTTPS)
   │
4. Nginx receives the connection
   ├── TLS handshake (presents Let's Encrypt certificate)
   ├── Decrypts the request
   ├── Request is GET / (root path)
   ├── Serves /frontend/dist/index.html (React app)
   │
5. Browser loads index.html, finds <script> and <link> tags
   ├── GET /assets/index-abc123.js → Nginx serves from dist/assets/
   ├── GET /assets/index-xyz789.css → Nginx serves from dist/assets/
   │
6. React app boots, user clicks "New Game"
   ├── POST /api/game/new → Nginx proxies to backend:8000/api/game/new
   │   └── FastAPI creates game session, loads PyTorch model
   │   └── Returns {session_id, state}
   │
7. React opens WebSocket: wss://lagbot.yourdomain.com/ws/{session_id}
   ├── Nginx upgrades HTTP → WebSocket connection
   ├── Proxies to ws://backend:8000/ws/{session_id}
   │   └── FastAPI WebSocket handler registered
   │
8. Game plays out:
   ├── Human action: POST /api/game/{id}/action → Nginx → Backend
   ├── Bot responds: Backend → WebSocket → Nginx → Browser
   ├── State updates pushed in real-time via WebSocket
   │
9. Hand completes:
   ├── Backend saves to PostgreSQL (internal network)
   ├── Winner info broadcast via WebSocket
```

### Security Groups (EC2 Firewall)

| Rule | Port | Source | Why |
|------|------|--------|-----|
| SSH | 22 | Your IP only | Admin access |
| HTTP | 80 | 0.0.0.0/0 | Redirect to HTTPS + Let's Encrypt ACME challenge |
| HTTPS | 443 | 0.0.0.0/0 | All real traffic |

Ports 8000 (FastAPI) and 5432 (PostgreSQL) are NOT exposed to the internet — they only communicate within the Docker network.

### Why Elastic IP?

When you stop and restart an EC2 instance, it gets a new public IP. An Elastic IP is a static IP that stays the same. You point your domain at it once, and it always works. Free while the instance is running.

---

## 7. WebSocket Considerations

WebSockets are the trickiest part of deploying a web app. Some things to be aware of:

### Why WebSockets Are Special

Normal HTTP: browser sends request → server responds → connection closes. Done.
WebSocket: browser sends upgrade request → server agrees → connection stays open indefinitely. Either side can send messages at any time.

This means every proxy in the chain (Nginx, load balancers) needs to know about WebSockets and keep the connection alive instead of timing it out.

### Nginx WebSocket Config

```nginx
location /ws/ {
    proxy_pass http://backend:8000;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;       # Key: pass the Upgrade header
    proxy_set_header Connection "upgrade";         # Key: tell upstream to upgrade
    proxy_set_header Host $host;
    proxy_read_timeout 86400;                      # Keep alive for 24 hours
}
```

The `Upgrade` and `Connection` headers are what tell the server "this isn't a normal HTTP request, switch to WebSocket protocol." Without these, Nginx would strip them out and the WebSocket handshake would fail.

### Timeout Gotcha

Default Nginx `proxy_read_timeout` is 60 seconds. If no data flows for 60 seconds, Nginx kills the connection. For a poker game where a player might be thinking for a few minutes, this will cause disconnections. Setting it to 86400 (24 hours) avoids this. The frontend's reconnection logic (exponential backoff, 5 retries) is a safety net.

---

## 8. Docker Production Image

### The PyTorch Problem

PyTorch with CUDA support is ~2GB. But we don't need CUDA for inference — CPU-only PyTorch is ~200MB. This matters because:
- Smaller image = faster deploy
- Smaller image = less ECR storage cost
- Faster cold starts

```dockerfile
# Use CPU-only PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Multi-Stage Build

```dockerfile
# Stage 1: Build frontend
FROM node:18-alpine AS frontend-build
WORKDIR /app
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# Stage 2: Production backend
FROM python:3.10-slim
WORKDIR /app

# Install Python deps (CPU-only PyTorch)
COPY backend/requirements.txt ./backend/
RUN pip install --no-cache-dir -r backend/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Copy application code
COPY backend/ ./backend/
COPY src/ ./src/
COPY models/ ./models/

# Copy frontend build
COPY --from=frontend-build /app/dist ./frontend/dist

EXPOSE 8000
CMD ["gunicorn", "backend.main:app", "-w", "2", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]
```

This produces a single Docker image with everything: backend, game engine, trained models, and pre-built frontend static files. Nginx serves the static files, or the backend can serve them as a fallback (already coded in `main.py`).

### Gunicorn vs Raw Uvicorn

Current setup: `uvicorn backend.main:app` — single worker, single process.

Production: `gunicorn -w 2 -k uvicorn.workers.UvicornWorker` — Gunicorn manages 2 Uvicorn workers. If one crashes, Gunicorn restarts it. Two workers can handle concurrent requests better (one can serve a REST call while the other handles a WebSocket).

**Caveat with WebSockets:** Each WebSocket connection is pinned to one worker. If you have 2 workers, a player's REST calls might go to worker 1 while their WebSocket is on worker 2. This is fine because game state lives in the `GameManager` singleton — but only within a single process. With 2+ workers, you'd need shared state (Redis, or just use 1 worker). For LagBot's scale, **1-2 workers is fine**.

---

## 9. Deployment Workflow

### First-Time Setup (One-Time)

1. Launch EC2 instance (Ubuntu 22.04, t3.medium)
2. Install Docker + Docker Compose
3. Point domain to Elastic IP (Route 53 or your registrar)
4. Clone repo, copy production docker-compose and nginx config
5. Run Certbot to get SSL certificate
6. `docker compose -f docker-compose.prod.yml up -d`
7. Verify: `curl https://lagbot.yourdomain.com/health`

### Updating the App (Ongoing)

```bash
ssh ubuntu@lagbot.yourdomain.com
cd /opt/lagbot
git pull origin master
docker compose -f docker-compose.prod.yml build
docker compose -f docker-compose.prod.yml up -d
```

Or with a deployment script:
```bash
ssh ubuntu@lagbot.yourdomain.com 'cd /opt/lagbot && git pull && docker compose -f docker-compose.prod.yml up -d --build'
```

### Optional: CI/CD with GitHub Actions

Push to `master` → GitHub Actions builds Docker image → pushes to ECR → SSHs into EC2 and pulls new image. Fully automated deploys. Can be added later.

---

## 10. Cost Summary

### Minimum Viable Deployment (EC2)

| Item | Monthly Cost |
|------|-------------|
| EC2 t3.medium (on-demand) | ~$30 |
| Elastic IP (while instance running) | Free |
| Route 53 hosted zone | $0.50 |
| Domain name | ~$1/mo ($12/yr) |
| Data transfer (first 100GB free) | ~$0 |
| **Total** | **~$32/mo** |

### If You Stop the Instance When Not Playing

EC2 only charges while running. If you run it 4 hours/day for game nights:
- 4 hrs/day × 30 days × $0.042/hr = ~$5/mo for compute
- Plus $0.50 Route 53 + $1 domain = **~$7/mo**

### Savings Tips

- **Reserved Instance (1-year):** ~40% savings → ~$20/mo instead of $30
- **Spot Instance:** ~70% savings but can be interrupted (bad for a live game)
- **Turn it off when not using it:** Script to start/stop from your phone
- **Free tier:** If this is a new AWS account, t3.micro is free for 12 months (but 1GB RAM is too small for PyTorch)

---

## 11. Alternatives to Buying a Domain

If you don't want to buy a domain right away:

1. **DuckDNS** (free) — Get `lagbot.duckdns.org`, point it to your EC2 IP. Works with Let's Encrypt.
2. **nip.io** (free) — Use `54-123-45-67.nip.io` which auto-resolves to that IP. No setup needed but no HTTPS.
3. **Cloudflare Tunnel** (free) — Expose your EC2 without a public IP. Free HTTPS. More complex setup.
4. **Just use the IP** — `http://54.123.45.67` works but no HTTPS and looks sketchy to friends' browsers.

Recommendation: DuckDNS for free, or spend $12/year on a real `.com` domain from Namecheap/Cloudflare.

---

## 12. What About the Trained Model?

The trained model (`model_6650000_steps.zip`, ~200MB) needs to be accessible to the backend container. Options:

1. **Bake it into the Docker image** — Simplest. `COPY models/ ./models/` in the Dockerfile. Image is bigger but self-contained. Best for a single model.
2. **Mount from host** — `volumes: - ./models:/app/models` in docker-compose. Requires the model to exist on the EC2 instance.
3. **Pull from S3 at startup** — Add a startup script that downloads the latest model from S3 before starting Uvicorn. Most flexible, allows model updates without rebuilding the image.

For now, option 1 (bake it in) is simplest. If you train new models and want to hot-swap them, move to option 3 later.

---

## 13. Summary

| Decision | Choice | Why |
|----------|--------|-----|
| **Platform** | AWS EC2 (single instance) | Cheapest, simplest, enough for friends |
| **Instance** | t3.medium (2 vCPU, 4GB RAM) | Minimum for PyTorch inference |
| **OS** | Ubuntu 22.04 LTS | Standard, well-supported |
| **Containerization** | Docker Compose | Already have it working |
| **Reverse proxy** | Nginx | SSL, static files, WebSocket proxy |
| **SSL** | Let's Encrypt + Certbot | Free, auto-renewing |
| **Domain** | Buy one ($12/yr) or DuckDNS (free) | Need a domain for HTTPS |
| **Frontend** | Built into backend image | FastAPI already serves dist/, or Nginx serves it |
| **Database** | PostgreSQL in Docker | Same as local, just in a container |
| **Model** | Baked into Docker image | Simplest, self-contained |
| **Cost** | ~$30/mo (or ~$7/mo if started on-demand) | Cheap for a personal project |

### Files We'll Need to Create/Modify

**New files:**
- `docker-compose.prod.yml` — Production Docker Compose (no volume mounts, restart policies, proper networking)
- `Dockerfile` (root-level, multi-stage) — Single production image with frontend build + backend + models
- `nginx/nginx.conf` — Nginx configuration for SSL, static files, proxy
- `scripts/deploy.sh` — One-command deployment script
- `scripts/setup-ec2.sh` — First-time EC2 setup (install Docker, certbot, etc.)
- `.env.production` — Production environment variables (DB password, domain, etc.)

**Modified files:**
- `backend/main.py` — CORS origins from env var instead of hardcoded
- `backend/Dockerfile` — Production multi-stage build (or replaced by root Dockerfile)

---

## Explanations

### What Is a Reverse Proxy and Why Do We Need Nginx?

Right now when you run locally, your friends connect to Vite on port 5173, and Vite proxies API calls to FastAPI on port 8000. Vite is your reverse proxy in development.

In production, Vite doesn't exist — you have pre-built static files. Nginx replaces Vite as the reverse proxy:

```
Development:                          Production:
Browser → Vite (:5173) → FastAPI     Browser → Nginx (:443) → FastAPI
             ↑ serves React               ↑ serves React
             ↑ proxies /api/              ↑ proxies /api/
             ↑ proxies /ws/               ↑ proxies /ws/
                                          ↑ handles HTTPS
```

Nginx is just a very fast, battle-tested program that:
1. Accepts incoming connections from the internet
2. Looks at the URL path
3. Either serves a static file (React build) or passes the request to another program (FastAPI)
4. Handles HTTPS encryption so FastAPI doesn't have to

You COULD skip Nginx and just expose FastAPI directly, but Nginx is better at serving static files (10-100x faster than Python) and handling many simultaneous connections. It's also the standard way to do HTTPS in production.

### What Is an Elastic IP?

Every EC2 instance gets a public IP address, but it changes every time you stop and restart the instance. Imagine if your home address changed every time you went to sleep — your mail would never arrive.

An Elastic IP is a static public IP that you "own" in your AWS account. You attach it to your EC2 instance, and it stays the same no matter how many times you restart. Your domain always points to this IP.

It's free while attached to a running instance. AWS charges ~$3.65/mo if you have an Elastic IP that's NOT attached to anything (to discourage hoarding IPs).

### What Is Route 53?

AWS's DNS service. DNS is the system that translates human-readable names (lagbot.com) to IP addresses (54.123.45.67). When you type a URL in your browser, your computer asks a DNS server "what IP is lagbot.com?" and gets back the number.

Route 53 is where you create that mapping. You could also use Cloudflare or Namecheap DNS — they all do the same thing. Route 53 is convenient if you're already in AWS.

### What Is Let's Encrypt?

HTTPS requires an SSL certificate — a cryptographic proof that your server is who it says it is. Traditionally, these cost $50-200/year from companies like DigiCert.

Let's Encrypt is a free, automated certificate authority. You run a tool called Certbot on your server, it proves you own the domain (by placing a file at `http://yourdomain.com/.well-known/acme-challenge/xxxxx`), and Let's Encrypt issues a certificate. Certificates last 90 days, but Certbot auto-renews them with a cron job.

This is why you need a domain — Let's Encrypt verifies domain ownership, and you can't "own" a bare IP address.

### Docker Compose vs. Kubernetes — Why Not K8s?

Kubernetes (K8s) is a container orchestration system for running hundreds or thousands of containers across many servers. It handles scaling, load balancing, rolling updates, service discovery, and self-healing.

For LagBot — 3 containers on 1 server — Kubernetes is like hiring a fleet manager to park your bicycle. Docker Compose does exactly what we need: define 3 services, start them together, let them talk to each other on an internal network.

If LagBot somehow becomes a massive poker platform with thousands of concurrent games, THEN consider Kubernetes. For now, Docker Compose is the right tool.
