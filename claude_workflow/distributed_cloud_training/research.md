# Research: Distributed Cloud Training for LagBot

## 1. Current LagBot Training Architecture

### 1.1 Training Pipeline Overview

LagBot trains a PPO-based poker agent via iterative self-play in a 3-player Texas Hold'em environment. The pipeline flows through:

1. **Configuration Loading** — YAML configs from `configs/` define environment settings, hyperparameters, and network architecture (13 config variants exist).
2. **Opponent Selection** — Auto-detects the 2 most recent trained models from `models/` by modification time. Falls back to `CallAgent` + `RandomAgent` if no trained models exist.
3. **Environment Creation** — `TexasHoldemEnv` creates a 3-player game with a 125-dimensional observation space (53 base + 72 opponent tracking features).
4. **Agent Initialization** — `PPOAgent` wraps Stable Baselines3's PPO. Auto-detects device: CUDA → MPS (Apple Metal) → CPU.
5. **Environment Wrapping** — `OpponentAutoPlayWrapper` handles automatic opponent moves between the learning agent's turns.
6. **Training Execution** — `agent.model.learn()` runs for configured `total_timesteps` with callbacks for metrics, checkpoints, and opponent profit tracking.
7. **Model Saving** — Periodic checkpoints every N steps (default 50k) + `final_model.zip` saved for use as future opponent.

Entry point: `train.py`. Additional scripts: `train_from_checkpoint.py`, `train_diverse_opponents.py`, `resume_training.py`.

### 1.2 Environment Details

**Observation Space (125 dimensions):**
- 7 cards × 6 features = 42 dims (rank normalized + 4 suit one-hot + card present flag)
- Hand features = 3 dims (hand strength, pot odds, SPR)
- Game state = 8 dims (stack size, pot total, bet amount, call amount, player active, position, round, button)
- Opponent stats = 72 dims (9 opponent slots × 8 features: VPIP, PFR, AF, 3-bet%, C-bet%, fold-to-cbet%, showdown%, confidence)

**Action Space: Discrete(6)**
- 0: Fold | 1: Check/Call | 2-4: Raise (0.5x, 1x, 2x pot) | 5: All-in

**Key features:** Pot-based raise bins, hand strength via Treys library, all-in runout, configurable rake, stack reset every N timesteps.

### 1.3 Neural Network Architecture

Framework: Stable Baselines3 PPO with `MlpPolicy`.

Current architecture variants tested:
| Config | Architecture | Params |
|--------|-------------|--------|
| Default | 64-64 (SB3 default) | ~50K |
| Deep | Pi/Vf: 512→256→256→128 | ~2M |
| Wide | Pi/Vf: 1024→512→512→256 | ~9M |
| Shared Layers | Shared: 512→512→256, split heads | ~3M |
| Separate Actor-Critic | Independent 512→256 networks | ~2M |

Key hyperparameters (default):
- `learning_rate`: 0.001 | `n_steps`: 3000 | `batch_size`: 128
- `n_epochs`: 5 | `gamma`: 0.99 | `gae_lambda`: 0.95
- `clip_range`: 0.2 | `ent_coef`: 0.05 | `vf_coef`: 0.5

### 1.4 Self-Play / Generational Evolution

```
Gen 0:  Train from scratch vs CallAgent + RandomAgent
         ↓
Gen 1:  Auto-loads Gen0 model as opponent
         ↓
Gen 2:  Auto-loads Gen0 + Gen1 as opponents
         ↓
Gen N:  Auto-loads 2 most recent models as opponents
```

Models are discovered via `find_latest_models()` which scans `models/` for `final_model.zip` files sorted by modification time. Each generation learns to beat previous versions, creating progressively stronger agents.

### 1.5 Opponent Tracking System

`OpponentTracker` maintains per-opponent statistics (VPIP, PFR, AF, 3-bet%, fold-to-cbet%, showdown%, confidence) fed into the observation space (72 dims). This allows the learning agent to adapt its strategy based on opponent tendencies — a critical feature for poker.

### 1.6 Current Performance & Scale

- Training on Apple M4 Pro with MPS: ~1M steps in 2-3 hours (4-5x vs CPU)
- Largest config tested: `wide_architecture_9M.yaml` — 9M timesteps, estimated 24-48 hours on single GPU
- 20+ trained model variants in `models/` directory
- All training is currently **single-machine, single-GPU**

### 1.7 Monitoring & Metrics

- `TrainingMetrics`: episode rewards, win rates, action distribution, policy/value loss, entropy
- `OpponentProfitTracker`: per-opponent profit/loss, hands played, cumulative profit over time
- TensorBoard logging in `logs/`
- JSON metrics in `metrics/<run_name>/`

---

## 2. Limitations of the Current Setup

1. **Single-machine bottleneck** — Training is bound to one GPU/CPU. The 9M-step wide architecture run takes 24-48 hours.
2. **Sequential generational training** — Each generation must fully complete before the next begins. No parallel exploration of different strategies.
3. **No hyperparameter search at scale** — 13 configs exist but can only be tested one at a time.
4. **Limited opponent diversity** — Only trains against 2 most recent models. No population-based approach.
5. **SB3 does not support distributed training** — No native multi-GPU, no multi-node. Only `SubprocVecEnv` for CPU-parallel environment stepping.
6. **No cloud infrastructure** — Everything runs locally. No shared model storage, no experiment tracking at scale.

---

## 3. Distributed Training Approaches

### 3.1 Option A: Parallel Experiment Runs (Easiest — Stays on SB3)

**Concept:** Keep the current SB3/PPO stack unchanged. Run multiple independent training jobs in parallel on cloud GPUs, each with a different config or generation. Share a central model storage (S3/NFS) so jobs can auto-discover each other's trained models for opponent selection.

**Architecture:**
```
┌──────────────────────────────────────────────────┐
│              Shared Storage (S3 / NFS)           │
│   models/  metrics/  logs/  configs/             │
└──────┬──────────┬──────────┬──────────┬──────────┘
       │          │          │          │
  ┌────▼───┐ ┌───▼────┐ ┌───▼────┐ ┌───▼────┐
  │ Job 1  │ │ Job 2  │ │ Job 3  │ │ Job 4  │
  │ GPU    │ │ GPU    │ │ GPU    │ │ GPU    │
  │shared_ │ │wide_   │ │self_   │ │diverse_│
  │layers  │ │arch_9M │ │play_3M │ │opps    │
  └────────┘ └────────┘ └────────┘ └────────┘
```

**Pros:**
- Minimal code changes — modify `find_latest_models()` to read from shared storage
- Each job is identical to current `train.py`
- Easy to containerize with Docker (already have `docker-compose.yml`)
- Embarrassingly parallel — no coordination needed
- Can run 4-8 experiments simultaneously

**Cons:**
- Does not speed up a single training run
- No true distributed PPO (each job is still single-GPU)
- Models only interact via shared storage, not live self-play

**Estimated effort:** 1-2 days. Containerize, add S3 sync, deploy to cloud.

**Best for:** Hyperparameter sweeps, running many generational experiments concurrently.

---

### 3.2 Option B: Migrate to Ray/RLlib with DD-PPO (Medium — Full Distributed PPO)

**Concept:** Replace SB3 with Ray/RLlib to get native distributed PPO (DD-PPO). Multiple GPU workers each run environment rollouts and compute gradients in parallel, synchronized via PyTorch DDP. Near-linear scaling with GPU count.

**Architecture:**
```
┌─────────────────────────────────────────┐
│           Ray Head Node                  │
│   RLlib Algorithm (DD-PPO)              │
│   Policy Server + Gradient Aggregation  │
└──────┬──────────┬──────────┬────────────┘
       │          │          │
  ┌────▼───┐ ┌───▼────┐ ┌───▼────┐
  │Worker 1│ │Worker 2│ │Worker 3│
  │GPU + 4 │ │GPU + 4 │ │GPU + 4 │
  │envs    │ │envs    │ │envs    │
  │rollout │ │rollout │ │rollout │
  │+grads  │ │+grads  │ │+grads  │
  └────────┘ └────────┘ └────────┘
```

**What DD-PPO does:**
- Each worker independently collects experience in GPU-accelerated environments and computes gradients
- Gradients are synchronized via `DistributedDataParallel` AllReduce
- A preemption threshold allows stragglers to be cut off (e.g., once 80% of workers finish, start optimization)
- Demonstrated **107x speedup on 128 GPUs** over serial PPO

**Migration path:**
1. Register `TexasHoldemEnv` as a Gymnasium environment compatible with RLlib
2. Port opponent tracking / auto-play wrapper to RLlib's multi-agent API or custom environment wrapper
3. Translate YAML configs to RLlib `PPOConfig`
4. Deploy Ray cluster on cloud (AWS, GCP) with GPU workers

**Pros:**
- True distributed PPO — a single training run scales across GPUs
- Near-linear speedup (9M steps could go from 48 hours → ~6 hours on 8 GPUs)
- Built-in multi-agent support (MADDPG, QMIX, custom policies)
- Integrates with Ray Tune for population-based training and hyperparameter optimization
- Production-grade — used by OpenAI, Ant Group, many game AI labs

**Cons:**
- Significant code rewrite — must port environment, agent, and training loop to RLlib APIs
- RLlib has a steep learning curve and has undergone breaking API changes
- Opponent tracking system and auto-play wrapper need adaptation to RLlib's multi-agent paradigm
- Debugging distributed training is harder than single-machine

**Estimated effort:** 1-2 weeks. Environment registration, RLlib config, cluster setup, testing.

**Best for:** Speeding up individual training runs, scaling to much larger timestep counts (50M+).

---

### 3.3 Option C: Population-Based Training with MALib (Advanced — Best for Poker)

**Concept:** Use MALib or a custom PBT framework to maintain a population of agents training simultaneously. Agents play against each other, and periodically the worst performers inherit weights/hyperparameters from the best. This produces diverse, robust strategies — critical for poker where no single strategy dominates.

**Architecture:**
```
┌────────────────────────────────────────────────┐
│              Population Manager                 │
│  Agent_1 (Elo: 1450, lr=0.001, ent=0.03)      │
│  Agent_2 (Elo: 1520, lr=0.0005, ent=0.05)     │
│  Agent_3 (Elo: 1380, lr=0.003, ent=0.02)      │
│  ...                                            │
│  Agent_N (Elo: 1490, lr=0.0008, ent=0.04)     │
└──────┬──────────────────────────┬──────────────┘
       │                          │
  ┌────▼─────────────────┐  ┌────▼──────────────┐
  │   Match Scheduler    │  │  Evolutionary Step │
  │   Assigns matchups   │  │  Every K episodes: │
  │   between agents     │  │  - Rank by Elo     │
  │                      │  │  - Bottom 20% copy │
  │                      │  │    from top 20%    │
  │                      │  │  - Mutate HPs      │
  └──────┬───────────────┘  └────────────────────┘
         │
    ┌────▼───┐ ┌────────┐ ┌────────┐
    │Worker 1│ │Worker 2│ │Worker N│
    │Runs    │ │Runs    │ │Runs    │
    │matches │ │matches │ │matches │
    │Collects│ │Collects│ │Collects│
    │exp     │ │exp     │ │exp     │
    └────────┘ └────────┘ └────────┘
```

**MALib specifics:**
- Actor-Evaluator-Learner model with centralized task dispatching
- Supports PBT, Self-Play, Neural Fictitious Self-Play, PSRO
- Algorithms: PPO, DQN, SAC, MADDPG, QMIX
- >40K FPS on a 32-core machine; 5x faster than RLlib on multi-agent tasks
- Tested on Leduc Poker

**Pros:**
- Produces diverse, robust strategies (not just beating the last version)
- Automatic hyperparameter optimization via evolution
- Elo-based evaluation gives clear progress signal
- Prevents strategy cycling (A beats B, B beats C, C beats A)
- Most aligned with how top poker AIs (Pluribus population) train

**Cons:**
- Largest code rewrite — full migration from SB3 to MALib or custom PBT framework
- MALib is research-grade software, may have rough edges
- Requires more compute (N agents × training cost per agent)
- More complex to debug and monitor

**Estimated effort:** 2-4 weeks. Framework migration, PBT logic, Elo tracking, cluster setup.

**Best for:** Producing the strongest possible poker bot with diverse strategies.

---

### 3.4 Option D: Hybrid Approach (Recommended)

**Concept:** Combine Options A and B in phases. Start with parallel experiment runs (minimal changes), then migrate the training core to RLlib for distributed PPO, and optionally layer PBT on top.

**Phase 1 (Week 1):** Containerize current SB3 training. Deploy parallel jobs on cloud GPUs via shared storage. Run hyperparameter sweeps and concurrent generational training.

**Phase 2 (Weeks 2-3):** Migrate `TexasHoldemEnv` to RLlib-compatible format. Implement DD-PPO for distributed single-run training. Keep opponent tracking in custom env wrapper.

**Phase 3 (Optional, Week 4+):** Add population-based training layer. Maintain a population of agents with Elo tracking. Evolutionary selection of best strategies/hyperparameters.

---

## 4. Cloud Infrastructure Options

### 4.1 GPU Instance Pricing Comparison

| Provider | GPU | Instance | On-Demand $/hr/GPU | Spot $/hr/GPU | Notes |
|----------|-----|----------|-------------------|---------------|-------|
| **AWS** | H100 | P5 | ~$3.90 | ~$2.50 | 44% price cut June 2025 |
| **GCP** | H100 | A3-High | ~$3.00 | ~$2.25 | Cheapest hyperscaler on-demand |
| **Azure** | H100 | NCads H100 v5 | ~$6.98 | Varies | Most expensive |
| **AWS** | A100 | P4d | ~$2.50-3.00 | ~$1.00-1.50 | Best value for RL |
| **Lambda Labs** | A100 40GB | — | $1.29 | — | Budget option |
| **Lambda Labs** | H100 PCIe | — | $2.49 | — | Budget option |

### 4.2 Recommendations for LagBot

**For PPO-based deep RL training (our case):**
- **A100s are the sweet spot.** LagBot's networks (50K-9M params) are small compared to LLMs. H100s are overkill.
- **Spot instances are ideal.** RL training is checkpoint-friendly (we already save every 50K steps). Use spot for 60-90% savings.
- **AWS P4d spot (A100)** at ~$1.00-1.50/GPU-hour is the most cost-effective for our workload.
- **Lambda Labs A100** at $1.29/hr is competitive if spot isn't available.

**Cost estimates for a typical training campaign:**
| Scenario | GPUs | Hours | Spot Cost |
|----------|------|-------|-----------|
| 4 parallel 3M-step runs | 4× A100 | ~8 hrs each | ~$32-48 |
| 1 distributed 9M-step run (8 GPU) | 8× A100 | ~6 hrs | ~$48-72 |
| PBT with 8 agents, 3M steps each | 8× A100 | ~12 hrs | ~$96-144 |
| Full hyperparameter sweep (16 configs) | 16× A100 | ~8 hrs | ~$128-192 |

### 4.3 Infrastructure Stack

**Containerization:** Docker image with PyTorch + SB3 + Ray (for Phase 2). Mount shared volumes for models/metrics/logs.

**Orchestration options:**
- **AWS Batch** — Simple job scheduling, good for Phase 1 parallel runs
- **Kubernetes (EKS/GKE)** — Full orchestration for Phase 2+ distributed training
- **Ray Cluster on cloud** — Ray has native autoscaling cluster launchers for AWS/GCP

**Shared storage:**
- **S3 / GCS** — For model checkpoints and metrics (async sync)
- **EFS / NFS** — For real-time model sharing between distributed workers
- **FSx for Lustre** — High-performance parallel filesystem for intensive I/O

**Monitoring:**
- TensorBoard (already used) deployed as a service
- Weights & Biases for experiment tracking across distributed runs
- CloudWatch / Prometheus for infrastructure health

---

## 5. Relevant Frameworks & Tools

### 5.1 Framework Comparison

| Framework | Throughput | Multi-Machine | Multi-Agent | Poker Tested | Migration Effort |
|-----------|-----------|---------------|-------------|-------------|-----------------|
| **SB3 (current)** | Low | No | No | Yes (our env) | None |
| **Ray/RLlib** | High | Native | Excellent | Via OpenSpiel | Medium |
| **SampleFactory** | Very High (130K FPS) | Limited | Limited | No | Medium |
| **MALib** | High (40K FPS) | Native | Excellent | Leduc Poker | High |
| **TorchRL** | Medium | Via PyTorch DDP | Good | No | Medium |
| **CleanRL** | Medium | No | No | No | Low |

### 5.2 Poker-Specific Frameworks

- **PokerRL** — Multi-agent deep RL for poker. Supports distributed mode with ~4 lines of code change. Combines deep learning with CFR and Fictitious Play.
- **OpenSpiel** (Google DeepMind) — Supports CFR, extensive-form games, imperfect information games. C++ core with Python bindings.
- **RLCard** — RL toolkit for card games including Texas Hold'em.
- **pypoks** — Open-source deep RL + genetic algorithms for poker.

### 5.3 Key Algorithms Beyond PPO

- **CFR / MCCFRM** — What Pluribus used. CPU-bound game tree search, not neural-network based. Converges to Nash equilibrium. Pluribus trained in 8 days on 64 CPUs with no GPUs.
- **Deep CFR** — Replaces tabular regret tracking with neural networks for scaling to larger games.
- **Neural Fictitious Self-Play (NFSP)** — Combines deep learning with fictitious play for approximate Nash equilibria.
- **PSRO (Policy Space Response Oracle)** — Maintains a population of policies and computes best responses. Related to PBT.

**Important note:** The poker AI research community has largely moved toward CFR-based methods rather than pure deep RL (PPO) for solving poker. Pluribus achieved superhuman 6-player poker using MCCFRM on CPUs alone. However, our PPO-based approach with opponent modeling is a valid and interesting alternative — it produces agents that actively adapt to opponents rather than playing a fixed Nash equilibrium strategy.

---

## 6. Parallel Environment Rollout Options

### 6.1 Single-Machine Parallelism (Quick Wins)

**SB3 SubprocVecEnv (current capability):**
```python
from stable_baselines3.common.vec_env import SubprocVecEnv
env = SubprocVecEnv([make_env(i) for i in range(num_envs)])
```
Currently underutilized — LagBot runs 1 environment. Spinning up 8-16 parallel envs on a multi-core machine would give immediate throughput gains with zero architecture changes.

**EnvPool (20x faster than SubprocVecEnv):**
C++ threadpool-based engine. ~1M Atari FPS on DGX-A100. Compatible with SB3. However, requires porting `TexasHoldemEnv` to C++ or using their API — likely not worth the effort for our custom env.

### 6.2 Multi-Machine Parallelism

**RLlib Rollout Workers:**
```python
config = PPOConfig()
    .rollouts(num_rollout_workers=16, num_envs_per_worker=4)
    .training(train_batch_size=12800)
    .resources(num_gpus=1)
```
16 workers × 4 envs = 64 parallel environments across machines. Workers send experience to central GPU learner.

**Custom CPU-GPU Separation:**
```
CPU Workers (environment simulation) → Experience Buffer → GPU Learner (PPO updates)
```
Poker env stepping is CPU-cheap. The bottleneck is PPO gradient computation. Separating these allows many CPU workers to feed one GPU learner.

---

## 7. Key Considerations for LagBot Specifically

### 7.1 What Makes Poker Training Different

1. **Imperfect information** — Unlike chess/Go, agents can't see opponent hands. Opponent modeling (our 72-dim tracking) is essential.
2. **Stochastic outcomes** — Variance in poker is extremely high. Need many more samples for reliable learning signal.
3. **Multi-agent dynamics** — 3-player game means opponent strategies interact in complex ways.
4. **Non-transitive strategies** — Aggressive beats passive, traps beat aggressive, passive beats traps. PBT helps avoid cycling.
5. **Positional asymmetry** — Button vs blinds creates structural differences the agent must learn.

### 7.2 What to Preserve in Any Migration

- **Opponent tracking system** (72-dim observation features) — This is LagBot's key differentiator. Any framework migration must preserve this.
- **Generational self-play** — The model auto-discovery and opponent loading system works well.
- **Diverse opponent training** (`train_diverse_opponents.py`) — Training against CallAgent, RandomAgent, TightAgent, AggressiveAgent, PassiveAgent, ManiacAgent.
- **Checkpoint frequency** — 50K-step checkpoints for fault tolerance.
- **Profit tracking** — Per-opponent profit/loss metrics for evaluating training quality.

### 7.3 Immediate Quick Wins (Before Any Migration)

1. **Use SubprocVecEnv** — Run 8-16 parallel envs in current SB3 setup. Free speedup.
2. **Containerize with Docker** — The `docker-compose.yml` already exists. Finalize it for cloud deployment.
3. **Add S3 model sync** — Simple script to upload/download models from S3. Enables parallel cloud jobs.
4. **Weights & Biases integration** — Replace/augment TensorBoard for better distributed experiment tracking.

---

## 8. Summary of Options

| Option | Effort | Speedup | Best For |
|--------|--------|---------|----------|
| **A: Parallel Experiments (SB3)** | 1-2 days | N× jobs (not per-job) | HP sweeps, concurrent generations |
| **B: RLlib DD-PPO** | 1-2 weeks | Near-linear per-job | Faster individual training runs |
| **C: PBT with MALib** | 2-4 weeks | Population-level | Strongest, most diverse agents |
| **D: Hybrid (A → B → C)** | Phased | Incremental | Recommended path |

**Recommended approach:** Start with Option A (parallel experiments) for immediate value, then migrate to Option B (RLlib DD-PPO) for true distributed training, with Option C (PBT) as the long-term goal for producing the strongest possible poker bot.
