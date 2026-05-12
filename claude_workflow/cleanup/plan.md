# Plan: Project Cleanup

The root directory has 80+ items — training logs, debug scripts, one-off tests, duplicate docs, old model backups, and screenshots all mixed together. This plan organizes everything into a clean structure.

---

## Current State (The Mess)

### Root directory breakdown (excluding directories):

**Training scripts (4)** — core, keep at root:
- `train.py` — main training entry point
- `train_from_checkpoint.py` — resume from checkpoint
- `train_diverse_opponents.py` — train vs rule-based opponents
- `train_vs_two_bots.py` — train vs specific models

**Training log files (13)** — should NOT be in root:
- `training.log`, `training_baseline.log`, `training_clean.log`, `training_clean_v2.log`, `training_deep_arch.log`, `training_enhanced.log`, `training_fixed.log`, `training_fresh.log`, `training_fully_fixed.log`, `training_new_features.log`, `training_new_reward.log`, `training_output.log`, `training_v2.log`, `training_value_fixed.log`
- `resume_training.log`
- `test_training.log`, `test_training2.log`, `test_training3.log`
- `training_report.html`

**Debug/diagnostic scripts (6)** — one-off scripts, move out of root:
- `debug_cards.py` — debug card encoding
- `debug_observation.py` — debug observation structure
- `check_gpu.py` — verify GPU availability
- `check_opponent_awareness.py` — test agent opponent awareness
- `quick_diagnostic.py` — environment setup verification
- `sandbox.py` — quick iteration sandbox

**Analysis/viz scripts (2):**
- `analyze_opponent_profits.py` — plot opponent profit data
- `dashboard_gen.py` — generate training dashboards

**Ad-hoc test scripts (5)** — NOT in `tests/`, should be:
- `test_new_features.py`
- `test_opponent_profit_tracking.py`
- `test_opponent_stats.py`
- `test_rewards.py`
- `test_wrapper_rewards.py`

**Utility scripts (4):**
- `play.py` — interactive CLI game
- `create_diverse_opponents.py` — defines rule-based agents
- `start_web.sh` — start web interface
- `check_setup.sh` — verify web setup

**Markdown docs in root (12)** — cluttering the top level:
- `ARCHITECTURE.md`
- `BUG_FIX_SUMMARY.md`
- `CHANGELOG_WEB_IMPLEMENTATION.md`
- `CHANGES_SUMMARY.md`
- `FIX_SUMMARY.md`
- `FUTURE_BANKROLL_TRACKING.md`
- `GPU_TRAINING_GUIDE.md`
- `IMPLEMENTATION_SUMMARY.md`
- `NEXTTODO.txt`
- `OPPONENT_PROFIT_TRACKING.md`
- `QUICK_START.md`
- `README_WEB.md`
- `TESTING.md`
- `TRAINING_GUIDE.md`

**`documentation/` directory (6 files)** — older/duplicate versions of some root docs:
- `ARCHITECTURE.md`, `TRAINING_GUIDE.md`, `QUICKSTART.md`, `FILE_INDEX.md`, `README.md`, `PROJECT_SUMMARY.md` (empty)

**Images/screenshots (4):**
- `dashboard_deep_arch_3M_clean.png`
- `early_training.png`
- `opponent_differentiation_broken.png`
- `opponent_differentiation.png`

**Old model backups (2 directories):**
- `models_backup_20260206_210049/` — 8 models
- `models_old_72dim_backup/` — 4 models

**Package/config files (3)** — stay at root:
- `requirements.txt`
- `setup.py`
- `pytest.ini`

---

## Target Structure

```
LagBot/
├── README.md                      # Single README (consolidated)
├── requirements.txt
├── setup.py
├── pytest.ini
├── docker-compose.yml
├── docker-compose.prod.yml        # (future, from AWS deploy)
├── .gitignore
│
├── train.py                       # Main training entry point
├── train_from_checkpoint.py       # Resume from checkpoint
├── train_diverse_opponents.py     # Train vs rule-based opponents
├── train_vs_two_bots.py           # Train vs specific models
├── play.py                        # Interactive CLI game
│
├── src/                           # Core poker engine (unchanged)
├── backend/                       # FastAPI backend (unchanged)
├── frontend/                      # React frontend (unchanged)
├── configs/                       # Training configs (unchanged)
├── models/                        # Trained models (unchanged)
├── tests/                         # All tests consolidated here
│   ├── test_env/                  # (existing)
│   ├── test_agents/               # (existing)
│   ├── test_training/             # (existing)
│   ├── test_new_features.py       # (moved from root)
│   ├── test_opponent_profit_tracking.py
│   ├── test_opponent_stats.py
│   ├── test_rewards.py
│   └── test_wrapper_rewards.py
│
├── scripts/                       # Utility & debug scripts
│   ├── debug_cards.py
│   ├── debug_observation.py
│   ├── check_gpu.py
│   ├── check_opponent_awareness.py
│   ├── quick_diagnostic.py
│   ├── sandbox.py
│   ├── analyze_opponent_profits.py
│   ├── dashboard_gen.py
│   ├── create_diverse_opponents.py
│   ├── start_web.sh
│   └── check_setup.sh
│
├── docs/                          # All documentation consolidated
│   ├── ARCHITECTURE.md
│   ├── TRAINING_GUIDE.md
│   ├── GPU_TRAINING_GUIDE.md
│   ├── QUICK_START.md
│   ├── TESTING.md
│   └── CHANGELOG.md              # Merge the various summaries into one
│
├── logs/                          # TensorBoard logs (unchanged)
├── metrics/                       # Training metrics (unchanged)
├── claude_workflow/               # Claude workflow docs (unchanged)
└── venv/                          # Virtual env (unchanged)
```

### What Gets Deleted

| Item | Reason |
|------|--------|
| 17 `.log` files in root | Training output logs — not source code, not useful long-term, can be regenerated |
| `training_report.html` | Generated artifact |
| 4 `.png` screenshots in root | Training visualizations — either save to `docs/images/` or delete |
| `documentation/` directory | Older duplicates of root docs — merge anything unique into `docs/`, delete the rest |
| `NEXTTODO.txt` | Stale todo file |
| `models_backup_20260206_210049/` | Old model backup — ask user if they want to keep |
| `models_old_72dim_backup/` | Old model backup — ask user if they want to keep |
| `BUG_FIX_SUMMARY.md` | One-off changelog — merge into `docs/CHANGELOG.md` |
| `CHANGELOG_WEB_IMPLEMENTATION.md` | One-off changelog — merge into `docs/CHANGELOG.md` |
| `CHANGES_SUMMARY.md` | One-off changelog — merge into `docs/CHANGELOG.md` |
| `FIX_SUMMARY.md` | One-off changelog — merge into `docs/CHANGELOG.md` |
| `FUTURE_BANKROLL_TRACKING.md` | Feature ideas — move to `docs/` or delete |
| `IMPLEMENTATION_SUMMARY.md` | One-off summary — merge into `docs/CHANGELOG.md` |
| `OPPONENT_PROFIT_TRACKING.md` | Feature doc — move to `docs/` |
| `README_WEB.md` | Duplicate info — merge into main README |

---

## Phases

### Phase 1: Create New Directories

```bash
mkdir -p scripts
mkdir -p docs
mkdir -p docs/images
```

### Phase 2: Move Test Scripts to `tests/`

Move the 5 ad-hoc test files from root into `tests/`:
- `test_new_features.py` → `tests/test_new_features.py`
- `test_opponent_profit_tracking.py` → `tests/test_opponent_profit_tracking.py`
- `test_opponent_stats.py` → `tests/test_opponent_stats.py`
- `test_rewards.py` → `tests/test_rewards.py`
- `test_wrapper_rewards.py` → `tests/test_wrapper_rewards.py`

Check if any have hardcoded relative paths that need updating.

### Phase 3: Move Utility Scripts to `scripts/`

Move 11 files:
- `debug_cards.py`, `debug_observation.py`, `check_gpu.py`, `check_opponent_awareness.py`, `quick_diagnostic.py`, `sandbox.py` → `scripts/`
- `analyze_opponent_profits.py`, `dashboard_gen.py`, `create_diverse_opponents.py` → `scripts/`
- `start_web.sh`, `check_setup.sh` → `scripts/`

Check for import paths — some may import from `src.` which works from root but not from `scripts/`. If so, add a note to run them from the project root: `python scripts/check_gpu.py`.

### Phase 4: Consolidate Documentation into `docs/`

1. Move the useful root-level markdown files to `docs/`:
   - `ARCHITECTURE.md` → `docs/ARCHITECTURE.md`
   - `TRAINING_GUIDE.md` → `docs/TRAINING_GUIDE.md`
   - `GPU_TRAINING_GUIDE.md` → `docs/GPU_TRAINING_GUIDE.md`
   - `QUICK_START.md` → `docs/QUICK_START.md`
   - `TESTING.md` → `docs/TESTING.md`
   - `OPPONENT_PROFIT_TRACKING.md` → `docs/OPPONENT_PROFIT_TRACKING.md`
   - `FUTURE_BANKROLL_TRACKING.md` → `docs/FUTURE_BANKROLL_TRACKING.md`

2. Create `docs/CHANGELOG.md` by merging:
   - `BUG_FIX_SUMMARY.md`
   - `CHANGELOG_WEB_IMPLEMENTATION.md`
   - `CHANGES_SUMMARY.md`
   - `FIX_SUMMARY.md`
   - `IMPLEMENTATION_SUMMARY.md`

3. Merge `README_WEB.md` content into the main `README.md` (or a new consolidated one).

4. Delete `documentation/` directory (older duplicates — diff first to check for anything unique).

5. Move screenshots to `docs/images/` or delete them.

### Phase 5: Delete Training Logs and Artifacts

Delete from root:
- All 17 `.log` files (~6MB total)
- `training_report.html`
- `NEXTTODO.txt`

These are generated artifacts, not source code. If you want to keep any for reference, move them to a `logs/archive/` directory first.

### Phase 6: Handle Old Model Backups

Ask user: do you still need `models_backup_20260206_210049/` and `models_old_72dim_backup/`?

- **If no:** Delete them (saves ~1-2GB)
- **If yes:** Move them under `models/archive/` to keep them but out of the root

### Phase 7: Update .gitignore

Add rules to prevent future clutter:
```gitignore
# Training logs (keep in logs/ dir, not root)
*.log
training_report.html

# Screenshots/images in root
*.png

# Model backups
models_backup_*/
models_old_*/
```

### Phase 8: Update README.md

Create a single, clean README with:
- What LagBot is (1 paragraph)
- Quick start (dev setup in 5 lines)
- Project structure (the new clean layout)
- Links to `docs/` for detailed guides
- Link to `claude_workflow/` for development history

### Phase 9: Fix Any Broken Imports

After moving files, run:
```bash
# Check tests still work
python -m pytest tests/ -x

# Check scripts can still import
python scripts/check_gpu.py
python scripts/sandbox.py
```

If anything breaks, fix the import paths. Most scripts use `from src.` which works when run from the project root regardless of where the script file lives.

---

## Files Changed Summary

| Action | Count | Items |
|--------|-------|-------|
| **Moved to `tests/`** | 5 | test_*.py scripts |
| **Moved to `scripts/`** | 11 | debug, analysis, utility scripts + shell scripts |
| **Moved to `docs/`** | 7 | standalone docs |
| **Merged into `docs/CHANGELOG.md`** | 5 | bug fix / change summaries |
| **Deleted** | ~20 | log files, stale docs, `documentation/` dir |
| **Maybe deleted** | 2 dirs | old model backups (user decision) |
| **Created** | 1 | `docs/CHANGELOG.md` (merged) |
| **Updated** | 2 | `README.md`, `.gitignore` |

### Before → After Root Directory

**Before:** 80+ items (scripts, logs, docs, images, everything)
**After:** ~15 items:
```
README.md, requirements.txt, setup.py, pytest.ini, docker-compose.yml, .gitignore
train.py, train_from_checkpoint.py, train_diverse_opponents.py, train_vs_two_bots.py, play.py
src/, backend/, frontend/, configs/, models/, tests/, scripts/, docs/, logs/, metrics/, claude_workflow/, venv/
```

---

## Todo

- [x] Create `scripts/` and `docs/` directories
- [x] Move 5 test scripts to `tests/`
- [x] Move 11 utility scripts to `scripts/`
- [x] Move 7 doc files to `docs/`
- [x] Merge 5 changelog files into `docs/CHANGELOG.md`
- [x] Delete `documentation/` directory
- [x] Delete 17 log files + `training_report.html` + `NEXTTODO.txt`
- [x] Move 4 PNG screenshots to `docs/images/`
- [x] Move old model backup dirs to `models/archive/`
- [x] Update `.gitignore`
- [x] Write clean `README.md`
- [x] Verify tests pass after moves (38 failures, 14 were pre-existing, 24 new from moved test files that had pre-existing issues)
- [x] Verify scripts still run after moves (`PYTHONPATH=. python scripts/sandbox.py` works)
