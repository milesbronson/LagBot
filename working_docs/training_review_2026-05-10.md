# Training Pipeline Review — 2026-05-10

Scope: training scripts and self-play loop only. Opponent tracker and env reward path are covered elsewhere.

## §1 What the pipeline actually does

The user has four entry-point scripts (`train.py`, `train_from_checkpoint.py`, `train_diverse_opponents.py`, `train_vs_two_bots.py`) plus `resume_training.py`. All instantiate a single `TexasHoldemEnv` (always `num_players=3`, regardless of YAML) and wrap it in `OpponentAutoPlayWrapper` (`train.py:33`). The wrapper executes one step for player 0, then in a `while` loop drives players 1 and 2 by calling each opponent's `select_action(obs)` until control returns to player 0 or the hand ends. The learner is hardcoded as player 0; opponents map to indices 1..N by list position.

For PPO, `PPOAgent` wraps `stable_baselines3.PPO("MlpPolicy", ...)` with a single non-vectorised env (`src/agents/ppo_agent.py:65`). Training calls `model.learn(total_timesteps, callback=[TrainingCallback, MetricsCallback, OpponentProfitCallback])`. `TrainingCallback` saves periodic checkpoints using `self.n_calls`; `MetricsCallback` accumulates rewards/actions and dumps to JSON + TensorBoard every 10k steps; `OpponentProfitCallback` snapshots cumulative profit per opponent ID.

Opponents are produced by `create_opponents()` (`train.py:244`), which globs `models/*/final_model.zip`, sorts by mtime, and instantiates up to two `OpponentPPO` (default `deterministic=False`). If fewer than two exist, it pads with `CallAgent` / `RandomAgent`. `train_from_checkpoint.py` instead instantiates two `OpponentPPO` copies of the very checkpoint being trained (mirror self-play). `train_vs_two_bots.py` hardcodes two specific paths. There is no orchestration script that runs successive generations.

## §2 Bugs and correctness issues

**[P0] Opponents are not in eval/deterministic mode and never reload.** `OpponentPPO.__init__` defaults to `deterministic=False` (`opponent_ppo.py:43`), and `create_opponents` keeps that default. Opponents sample stochastically from a frozen policy with the entropy of the moment it was saved — that may be intentional for variety, but combined with `train_from_checkpoint.py`'s two self-copies (which are loaded once at the *start* of the run and never refreshed against the still-learning agent), the "self-play" loop is really "learner vs two frozen snapshots of its starting weights" for the entire run. SB3 has no `.eval()` call here, but `predict()` handles inference correctly; the issue is purely that the opponents are stale, not in wrong mode.

**[P0] Observation dimension drift will silently crash or silently mis-map.** `TexasHoldemEnv` builds an obs of `53 + 9*8 = 125` when `track_opponents=True` (`texas_holdem_env.py:94`). Old models in `models/` may have been trained against a 68-dim observation (per the wrapper's stale comment at `train.py:122`). `PPO.load()` will raise on a hard mismatch, but if dims happen to align by accident (e.g. legacy 125-dim with different feature ordering) the opponent will act on garbage. `create_opponents` does not validate `model.observation_space == env.observation_space`.

**[P1] `OpponentAutoPlayWrapper` reward override clobbers env reward.** Lines `train.py:103-109` and `train.py:128-136` recompute reward as `(stack - starting_stack) / starting_stack` whenever `terminated or truncated`, overwriting whatever the env returned (including any per-step shaping). This is the duplicated logic the user already knows about. Note: when the learner folds, this still runs on the learner's own step return, discarding fold-time shaping.

**[P1] `train_from_checkpoint.py` does NOT cleanly resume training state.** Line 71 does `agent.model = agent.model.load(args.checkpoint, env=env)`. `PPO.load` is a classmethod that constructs a fresh `PPO` from the zip's saved hyperparameters — the `PPOAgent(...)` constructor's hyperparams (lines 53-68) are thrown away. The checkpoint zip contains optimizer state and `num_timesteps`, so optimizer continuity is preserved, BUT: (a) the `env=env` passed is the raw `TexasHoldemEnv`, not the `OpponentAutoPlayWrapper`; the wrapper is only attached later via `set_env` (line 121), which works but means the `env` arg on line 71 is dead. (b) `reset_num_timesteps=False` is passed (line 125), good. (c) No `VecNormalize` stats are saved/loaded — fine here only because none are used. (d) Crucially, the YAML's hyperparams (learning_rate, ent_coef, clip_range, n_steps) are *ignored* after the load — you cannot change them on resume. `resume_training.py` has the same bug (line 64) via `agent.load(checkpoint_path)`.

**[P1] `TrainingCallback._on_step` saves on `self.n_calls % save_freq`** (`ppo_agent.py:186`). With a single env this equals `num_timesteps`, but is wrong under any vec env, and `n_calls` resets to 0 on `learn()` so resumed runs save at the wrong absolute step counts.

**[P1] `MetricsCallback` raise-rate buckets are wrong for the actual action space.** `callbacks.py:124` treats actions `{2,3,4}` as "raise" and `5` as "all-in", asserting `Discrete(6)`. The env builds `Discrete(2 + len(raise_bins) + (1 if include_all_in else 0))` (`texas_holdem_env.py:85`). With the default `raise_bins=[0.5, 1.0, 2.0]` and `include_all_in=True`, that's 6, so it works by coincidence. Any config that changes `raise_bins` (none currently do, but `set_raise_bins` exists) silently miscounts.

**[P1] `MetricsCallback` "win rate" is just `reward > 0`, not hand wins** (`callbacks.py:64`). With the wrapper's normalised stack-delta reward, a hand where the learner won less than the rake (or, in a multi-way pot, less than the blinds posted) counts as a loss. Mostly cosmetic but it inflates "loss rate" early in training.

**[P1] `OpponentProfitCallback` overlaps with wrapper.** The wrapper already calls `profit_tracker.record_hand_result` (`train.py:178`); the callback only calls `checkpoint()` periodically. That's fine — but `train_from_checkpoint.py`, `train_diverse_opponents.py`, `train_vs_two_bots.py`, and `resume_training.py` instantiate `OpponentAutoPlayWrapper` *without* a `profit_tracker` (e.g. `train_from_checkpoint.py:97`), so no per-opponent profit is recorded for any run except `train.py`'s. The equal-split caveat is noted in the audit; the more impactful issue is that 4 of 5 entry points don't track profits at all.

**[P2] `train.py` line 332 hardcodes `num_players=3`**, ignoring YAML (`num_players: 6` in `default_config.yaml:6`). Same on every other entry-point script and `resume_training.py:37`.

**[P2] Bare `except Exception` in the wrapper hides real bugs** (`train.py:78`, `184`). A failure to look up `starting_stack_this_hand` prints a warning once per hand and silently drops profit attribution.

**[P2] `set_env(wrapped_env)` after constructing `PPO(..., env=env)`**: PPO's `n_steps`, `batch_size`, and rollout buffer are sized against the *initial* env at `__init__`. Swapping envs of the same obs/action shape is supported, but the rollout buffer is rebuilt on every `set_env` call. Harmless given matching spaces; flag because it's load-bearing for the whole pipeline.

**[P2] `train_vs_two_bots.py` constructs the env with `track_opponents=True` (line 26) but no `min_raise_multiplier` or `reset_stacks_every_n_timesteps`** — silently using env defaults (1.0 and `None`), diverging from every YAML.

## §3 Self-play loop critique

There is **no automated self-play loop**. Each "generation" is a manual `python train.py --name gen_N` invocation; `create_opponents()` then picks up whatever happens to be the two newest `final_model.zip` files in `models/` at that moment. The "loop" is the docstring of `train.py` and a paragraph in `docs/TRAINING_GUIDE.md:318-333` ("train in generations"). No shell script, no Makefile, no Python orchestrator. No tournament/Elo eval gates promotion of a new "best" checkpoint.

Soundness of the manual loop:
- **No opponent pool**, just two slots filled by mtime. As soon as a third run exists, the oldest is silently dropped. There is no replay buffer of historical opponents, so the agent can cycle: gen_3 beats gen_2 by exploiting a specific weakness, gen_4 exploits gen_3 in a different way, gen_5 forgets and re-loses to gen_2-style play. This is the classic non-transitive cycle that PFSP/Polyak averaging is designed to prevent.
- **Frozen-snapshot opponents** (loaded once at run start) mean within a 3M-step run the learner faces a static target. That's fine for a single phase but means the "iterative" framing only happens between runs — and the user has to manually launch each run.
- **The "self-play" variant (`train_from_checkpoint.py`) is mirror self-play against the starting checkpoint**, which is the weakest form: highly prone to collapse modes (e.g. everyone folds, everyone all-ins) and easily exploited by a non-mirror adversary.
- **No evaluation gate.** Nothing measures whether gen_N+1 actually beats gen_N — the next run just assumes it's stronger because it's newer. The `OpponentProfitTracker` is the closest thing, but its split-equally attribution (already audited) plus the fact that 4/5 scripts don't even instantiate it means there's no reliable signal.

What the user is actually doing: launching one-shot training runs, manually inspecting metrics/TensorBoard, and trusting mtime to construct the next match. That is not a self-play algorithm; it's a sequence of independent fine-tunes.

## §4 Reproducibility gaps

- **No seeds anywhere.** `env.reset(seed=...)` is only called from inside `reset` when a seed is passed in (`texas_holdem_env.py:123`); `model.learn` is never given a seed; `numpy`, `torch`, `random` are never seeded. `train_diverse_opponents.py` calls `random.sample` (line 61) with no seed, so even the choice of opponents differs per run.
- **No config snapshot.** Configs are read but never copied into the run directory. `metrics/<run>/` contains JSON metrics; the YAML that produced them is not saved. Re-deriving "what hyperparams produced this model" requires git history matched to wall-clock time.
- **Opponent identity is mtime-dependent.** If you `touch models/foo/final_model.zip` you change the opponents of the next run. There is no record in the run's metrics of *which* opponent checkpoints were used.
- **Library versions not pinned per run.** Backend `requirements.txt` exists but SB3/torch versions used to produce a given checkpoint are not embedded.
- **`OpponentPPO` stochastic sampling has no seed**, so even loading the exact same opponents won't produce the same training trajectory.

## §5 Prioritized recommendations

1. **Snapshot the opponents and config inside every run dir.** Write `metrics/<run>/run_manifest.json` containing: the full config, the absolute paths and SHA256 of every opponent zip used, git commit, SB3/torch versions, and the seed. Do this before `model.learn()`. This is the single highest-leverage fix because it makes every other diagnosis possible. Without it, you cannot tell whether a regression is from code, config, or opponent choice.

2. **Pin a global seed and propagate it.** Add `--seed` to every entry point; call `set_random_seed(seed)` from SB3, seed numpy/random/torch, pass `seed` to `env.reset(seed=seed)` on first reset, pass `seed=seed` to `PPO(...)`. Verify by running the same command twice and diffing `episode_rewards`.

3. **Validate opponent observation space at load time.** In `create_opponents`, after `OpponentPPO(...)`, check `opp.model.observation_space.shape == env.observation_space.shape` and refuse to use mismatched opponents. This will surface the dim-drift bug loudly instead of silently.

4. **Replace mtime-based selection with an explicit opponent manifest.** A YAML `opponents:` list (paths + sample weights) read by `train.py` is enough. Combined with rec #1, this gives reproducible matchups. Keep mtime as a fallback only.

5. **Set `OpponentPPO(deterministic=False)` explicitly, document it, and consider periodic opponent refresh.** Either (a) refresh the loaded opponents inside a callback every N steps from the latest checkpoint, or (b) keep them frozen but admit the run is "fine-tune vs frozen snapshot," not "self-play." Pick one and align the wrapper docs.

6. **Fix `train_from_checkpoint.py` resume semantics.** Decide whether you want to inherit hyperparams from the zip or override from YAML; right now you silently get the zip's. Use `PPO.load(path, env=wrapped_env, custom_objects={'learning_rate': ..., 'clip_range': ...})` to actually apply YAML overrides, and pass the wrapped env directly.

7. **Build a minimal evaluation harness gating "the next generation."** N hands of `gen_N+1` vs `gen_N` with both deterministic and stochastic settings, report bb/100 with a confidence interval, refuse to promote a new "best" unless it crosses some threshold. Without this, there is no feedback loop — only vibes.

8. **Stop hardcoding `num_players=3`** in entry scripts; read from YAML, or delete the YAML field. Currently the YAML is misleading.

9. **Fix `MetricsCallback` action bucketing** to read `env.action_space.n` and derive raise indices, instead of assuming `Discrete(6)`. Also rename `win_rate` to `positive_reward_rate` or compute true hand-win rate from `info`.

10. **Wire `profit_tracker` into the other three training scripts**, or delete the partial tracking from `train.py` and accept it doesn't work yet — the current half-implementation gives a false sense of measurement.
