"""
train.py — single entrypoint for training PPO poker agents.

The training loop, in order:
1. Loads a YAML config.
2. Opens the AgentRegistry at models/registry.json.
3. Seeds rule-based fixtures (CallAgent, RandomAgent) into the registry
   on first run so the sampler always has a non-empty pool.
4. Samples opponents via OpponentSampler using the config-driven
   strategy (latest / random / weighted_recency / fixed).
5. Builds TexasHoldemEnv with num_players from config.
6. Optionally loads weights from a previously registered checkpoint
   (``continuation.resume_from``).
7. Trains the learner inside OpponentAutoPlayWrapper for
   ``training.total_timesteps`` steps.
8. Saves the final checkpoint.
9. If an eval gate is configured, plays a heads-up shootout against the
   resume parent. Skip registration if the gate fails.
10. On success, folds OpponentProfitTracker totals into the registry's
    matchup_history and registers a new AgentCard.

Repeat for ``continuation.generations`` cycles; each next generation
resumes from the just-registered card.
"""

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import yaml

from src.agents.opponent_ppo import OpponentPPO
from src.agents.ppo_agent import PPOAgent, TrainingCallback
from src.agents.random_agent import CallAgent, RandomAgent
from src.poker_env.texas_holdem_env import TexasHoldemEnv
from src.training.agent_card import AgentCard
from src.training.agent_registry import AgentRegistry
from src.training.callbacks import (
    BestCheckpointCallback,
    CriticCalibrationCallback,
    MetricsCallback,
    OpponentProfitCallback,
)
from src.training.eval_gate import EvalGate
from src.training.metrics import TrainingMetrics
from src.training.opponent_autoplay_wrapper import OpponentAutoPlayWrapper
from src.training.opponent_profit_tracker import OpponentProfitTracker
from src.training.opponent_sampler import OpponentSampler
from src.training.regression_eval import RegressionEval


CALL_FIXTURE_ID = "rule_call_v0"
RANDOM_FIXTURE_ID = "rule_random_v0"


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _instantiate_opponent(card: AgentCard):
    """Build a BaseAgent for use as an opponent from a registry card."""
    if card.kind == "ppo":
        agent = OpponentPPO(card.path, name=card.name)
        if not agent.is_loaded():
            raise RuntimeError(f"could not load PPO checkpoint from {card.path!r}")
        return agent
    if card.kind == "call":
        return CallAgent(name=card.name)
    if card.kind == "random":
        return RandomAgent(name=card.name)
    raise ValueError(f"unknown agent kind {card.kind!r}")


def _ensure_fixture_cards(registry: AgentRegistry) -> None:
    """First-run seeding: CallAgent + RandomAgent always available."""
    if registry.get(CALL_FIXTURE_ID) is None:
        registry.register(AgentCard(
            id=CALL_FIXTURE_ID, name="CallAgent", kind="call",
        ))
    if registry.get(RANDOM_FIXTURE_ID) is None:
        registry.register(AgentCard(
            id=RANDOM_FIXTURE_ID, name="RandomAgent", kind="random",
        ))


def _build_opponent_factory(
    registry: AgentRegistry,
    opponents_cfg: dict,
    non_learner_ids: List[int],
):
    """Returns ``(factory, initial_cards)``.

    The factory draws a fresh opponent card per seat on each call by
    sampling from the registry, instantiates the corresponding agent
    (cached so repeated draws of the same card don't reload PPO weights),
    and returns ``(kind, agent, card)`` for the wrapper.

    ``initial_cards`` is the audit trail of the first draw used by the
    run summary and ``AgentCard.trained_against_ids``."""
    sampler = OpponentSampler(registry)
    agent_cache: dict = {}

    def _instantiate(card: AgentCard):
        if card.id not in agent_cache:
            agent_cache[card.id] = _instantiate_opponent(card)
        return agent_cache[card.id]

    def factory(seat_id: int):
        cards = sampler.sample(
            n=1,
            strategy=opponents_cfg.get("strategy", "weighted_recency"),
            kind=opponents_cfg.get("kind"),
            ids=opponents_cfg.get("fixed_ids"),
        )
        if not cards:
            # Empty registry pool — fall back to the call fixture so the
            # wrapper always gets a valid (kind, agent, card) triple.
            card = registry.get(CALL_FIXTURE_ID)
        else:
            card = cards[0]
        return card.kind, _instantiate(card), card

    initial_cards = [factory(pid)[2] for pid in non_learner_ids]
    return factory, initial_cards


def select_opponent_cards(
    registry: AgentRegistry,
    opponents_cfg: dict,
    num_needed: int,
    exclude_self_id: Optional[str],
) -> List[AgentCard]:
    sampler = OpponentSampler(registry)
    exclude = [exclude_self_id] if exclude_self_id else None
    cards = sampler.sample(
        n=num_needed,
        strategy=opponents_cfg.get("strategy", "latest"),
        kind=opponents_cfg.get("kind"),
        exclude_ids=exclude,
        ids=opponents_cfg.get("fixed_ids"),
    )
    # Pad with rule fixtures if the registry pool was too small.
    while len(cards) < num_needed:
        fallback = registry.get(CALL_FIXTURE_ID) if len(cards) % 2 == 0 \
            else registry.get(RANDOM_FIXTURE_ID)
        cards.append(fallback)
    return cards


def _build_env(env_cfg: dict) -> TexasHoldemEnv:
    return TexasHoldemEnv(
        num_players=env_cfg["num_players"],
        starting_stack=env_cfg["starting_stack"],
        small_blind=env_cfg["small_blind"],
        big_blind=env_cfg["big_blind"],
        rake_percent=env_cfg["rake_percent"] if env_cfg.get("rake_enabled") else 0.0,
        rake_cap=env_cfg.get("rake_cap", 0),
        min_raise_multiplier=env_cfg.get("min_raise_multiplier", 1.0),
        reset_stacks_every_n_timesteps=env_cfg.get("reset_stacks_every_n_timesteps"),
        track_opponents=True,
    )


def _run_eval_gate(
    env_cfg: dict,
    eval_cfg: dict,
    final_model_path: str,
    parent_card: AgentCard,
) -> Tuple[bool, dict]:
    candidate = OpponentPPO(final_model_path + ".zip", name="candidate")
    predecessor = _instantiate_opponent(parent_card)
    gate = EvalGate(
        num_hands=eval_cfg.get("num_hands", 1000),
        threshold_mbb_per_100=eval_cfg.get("threshold_mbb_per_100", 0.0),
        starting_stack=env_cfg["starting_stack"],
        small_blind=env_cfg["small_blind"],
        big_blind=env_cfg["big_blind"],
        seed=eval_cfg.get("seed", 0),
    )
    result = gate.evaluate(
        candidate, predecessor,
        candidate_id="candidate", predecessor_id=parent_card.id,
    )
    return gate.passes(result), result.to_dict()


def _fold_behavior_stats_into_registry(
    registry: AgentRegistry,
    card_snapshots: dict,
) -> None:
    """Write each opponent's accumulated behavioural observations from
    this run back to its registry card. Rule fixtures are skipped — their
    behaviour is deterministic by construction and registry merging
    would only accumulate noise (matches the policy in
    ``_fold_profits_into_registry``).

    ``card_snapshots`` is keyed by card_id (not seat_id) so rotation mode
    can attribute hands to the right card even though multiple cards
    share a seat across episodes."""
    for card_id, entry in card_snapshots.items():
        card = registry.get(card_id)
        if card is None or card.kind != "ppo":
            continue
        if not entry:
            continue
        hands = int(entry.get("hands_observed", 0))
        if hands == 0:
            continue
        # update_behavior_stats expects rates only (without hands_observed) +
        # a separate hands count, so split the dict.
        rates = {k: v for k, v in entry.items() if k != "hands_observed"}
        registry.update_behavior_stats(card.id, rates, hands)


def _fold_profits_into_registry(
    registry: AgentRegistry,
    learner_card_id: str,
    profit_tracker: OpponentProfitTracker,
    seat_to_card: dict,
    timestep: int,
) -> None:
    for seat_id, opponent_card in seat_to_card.items():
        stats = profit_tracker.opponent_results.get(seat_id)
        if not stats or stats["hands_played"] == 0:
            continue
        # Don't dump matchup history onto rule fixtures' cards — they're
        # generic and would accumulate noise from every run forever. For
        # rule fixtures we log behaviour stats only if/when added later;
        # for ppo opponents we write per-observer matchup history.
        if opponent_card.kind != "ppo":
            continue
        registry.update_matchup(
            observer_id=learner_card_id,
            opponent_id=opponent_card.id,
            hands=stats["hands_played"],
            profit=stats["total_profit"],
            timestep=timestep,
        )


def train_one_generation(
    config: dict,
    run_name: str,
    registry: AgentRegistry,
) -> Optional[AgentCard]:
    env_cfg = config["environment"]
    train_cfg = config["training"]
    opp_cfg = config.get("opponents", {})
    cont_cfg = config.get("continuation", {})
    eval_cfg = config.get("eval_gate", {})
    regression_cfg = config.get("regression_eval", {})
    log_cfg = config["logging"]

    num_players = env_cfg["num_players"]
    if not 2 <= num_players <= 10:
        raise ValueError(f"num_players must be between 2 and 10, got {num_players}")

    _ensure_fixture_cards(registry)

    resume_from_id = cont_cfg.get("resume_from")
    parent_card = registry.get(resume_from_id) if resume_from_id else None
    if resume_from_id and parent_card is None:
        raise KeyError(f"continuation.resume_from {resume_from_id!r} not in registry")

    env = _build_env(env_cfg)

    log_dir = os.path.join(log_cfg["log_dir"], run_name)
    model_dir = os.path.join(log_cfg["model_dir"], run_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    metrics = TrainingMetrics(run_name, save_dir="metrics")
    profit_tracker = OpponentProfitTracker(run_name, save_dir="metrics")

    non_learner_ids = [
        p.player_id for p in env.game_state.players
        if p.player_id != env.learning_agent_id
    ]

    rotate_per_episode = bool(opp_cfg.get("rotate_per_episode", False))

    if rotate_per_episode:
        opponent_factory, factory_cards = _build_opponent_factory(
            registry, opp_cfg, non_learner_ids,
        )
        # In rotation mode, the factory picks fresh agents on every
        # reset() — opponent_cards is just used for the run summary and
        # the AgentCard.trained_against_ids audit field.
        opponent_cards = factory_cards
        seat_to_card: dict = {}
        opponents = None
    else:
        # Don't exclude the parent: in a chained self-play curriculum the
        # immediate predecessor is exactly who the new generation should
        # be learning to beat.
        opponent_cards = select_opponent_cards(
            registry, opp_cfg,
            num_needed=num_players - 1,
            exclude_self_id=None,
        )
        opponents = [(c.kind, _instantiate_opponent(c)) for c in opponent_cards]
        seat_to_card = dict(zip(non_learner_ids, opponent_cards))

        # Seed opponent stat priors from the registry so the obs vector
        # reflects accumulated knowledge of each opponent from prior runs.
        # In rotation mode the wrapper re-seeds per reset() from the
        # newly drawn card, so we skip seeding here.
        priors_by_pid = {
            pid: card.behavior_stats
            for pid, card in seat_to_card.items()
            if int((card.behavior_stats or {}).get("hands_observed", 0)) > 0
        }
        if priors_by_pid:
            env.opponent_tracker.seed_priors(priors_by_pid)

    policy_kwargs = train_cfg.get("policy_kwargs")
    # CPU is mandatory for MlpPolicy: SB3's own warning says MPS/CUDA are
    # slower than CPU for small dense nets, and our benchmarks confirmed
    # ~200 steps/sec on MPS vs much faster on CPU. Override via config if
    # ever switching to a CNN/image policy.
    device = train_cfg.get("device", "cpu")
    agent = PPOAgent(
        env=env,
        name=f"PPO_{run_name}",
        learning_rate=train_cfg["learning_rate"],
        n_steps=train_cfg["n_steps"],
        batch_size=train_cfg["batch_size"],
        n_epochs=train_cfg["n_epochs"],
        gamma=train_cfg["gamma"],
        gae_lambda=train_cfg["gae_lambda"],
        clip_range=train_cfg["clip_range"],
        ent_coef=train_cfg.get("ent_coef", 0.01),
        vf_coef=train_cfg.get("vf_coef", 0.5),
        max_grad_norm=train_cfg.get("max_grad_norm", 0.5),
        tensorboard_log=log_dir,
        policy_kwargs=policy_kwargs,
        device=device,
    )

    if parent_card and parent_card.path:
        agent.load(parent_card.path)

    if rotate_per_episode:
        wrapped_env = OpponentAutoPlayWrapper(
            env,
            opponent_factory=opponent_factory,
        )
    else:
        wrapped_env = OpponentAutoPlayWrapper(
            env,
            opponents_list=opponents,
            profit_tracker=profit_tracker,
            seat_to_card=seat_to_card,
        )
    agent.model.set_env(wrapped_env)

    save_callback = TrainingCallback(
        save_freq=log_cfg["save_frequency"], save_path=model_dir,
    )
    metrics_callback = MetricsCallback(metrics=metrics, log_freq=10000)
    profit_callback = OpponentProfitCallback(
        profit_tracker=profit_tracker, checkpoint_freq=10000,
    )
    critic_callback = CriticCalibrationCallback(
        save_dir=os.path.join("metrics", run_name), flush_freq=10000,
    )

    # Track the best checkpoint seen during training (by gate score vs the
    # immediate predecessor) so we can register that one instead of the
    # post-collapse final. Only enabled when there's a predecessor to
    # play against — fresh runs fall through to final_model.zip.
    best_callback: Optional[BestCheckpointCallback] = None
    if eval_cfg.get("enabled") and parent_card and parent_card.path:
        best_callback = BestCheckpointCallback(
            predecessor_path=parent_card.path,
            predecessor_id=parent_card.id,
            save_dir=model_dir,
            eval_freq=log_cfg["save_frequency"],
            num_hands=eval_cfg.get("num_hands", 1000),
            starting_stack=env_cfg["starting_stack"],
            small_blind=env_cfg["small_blind"],
            big_blind=env_cfg["big_blind"],
            seed=eval_cfg.get("seed", 0),
        )

    callbacks = [save_callback, metrics_callback, profit_callback, critic_callback]
    if best_callback is not None:
        callbacks.append(best_callback)

    print(f"\n{'='*70}\nTraining {run_name}  (gen {registry.next_generation()})\n{'='*70}")
    print(f"  env: {num_players} players, blinds {env_cfg['small_blind']}/{env_cfg['big_blind']}")
    print(f"  opponents: {[c.id for c in opponent_cards]}")
    print(f"  resume from: {resume_from_id or '(fresh)'}")
    print(f"  total timesteps: {train_cfg['total_timesteps']:,}")
    print()

    agent.model.learn(
        total_timesteps=train_cfg["total_timesteps"],
        callback=callbacks,
    )

    final_model_path = os.path.join(model_dir, "final_model")
    agent.save(final_model_path)

    register_path = final_model_path + ".zip"
    eval_stats: Optional[dict] = None
    threshold = eval_cfg.get("threshold_mbb_per_100", 0.0)

    if best_callback is not None and best_callback.best_stats is not None:
        # We tracked best vs predecessor throughout training — use it.
        if best_callback.best_mbb_per_100 < threshold:
            print(
                f"EvalGate FAILED for {run_name}: "
                f"best mbb/100={best_callback.best_mbb_per_100:+.2f} "
                f"@ {best_callback.best_steps:,} steps "
                f"< threshold {threshold}. Skipping registration."
            )
            return None
        register_path = os.path.join(model_dir, "best_model.zip")
        eval_stats = best_callback.best_stats
        print(
            f"\nUsing BEST checkpoint: {best_callback.best_steps:,} steps, "
            f"mbb/100={best_callback.best_mbb_per_100:+.0f}"
        )
    elif eval_cfg.get("enabled") and parent_card and parent_card.path:
        # Fallback path: callback never fired (e.g. training shorter than
        # one eval_freq window). Gate the final like before.
        passed, eval_stats = _run_eval_gate(
            env_cfg, eval_cfg, final_model_path, parent_card,
        )
        if not passed:
            print(
                f"EvalGate FAILED for {run_name}: "
                f"mbb/100={eval_stats['mbb_per_100']:.2f} "
                f"< threshold {threshold}. Skipping registration."
            )
            return None

    card = AgentCard(
        id=run_name,
        name=run_name,
        kind="ppo",
        path=register_path,
        generation=registry.next_generation(),
        parent_id=resume_from_id,
        trained_against_ids=[c.id for c in opponent_cards],
        training_config=train_cfg,
        total_timesteps=train_cfg["total_timesteps"],
        eval_stats=eval_stats,
    )
    registry.register(card)

    # Per-seat profit attribution only makes sense in static mode; under
    # rotation a seat hosts many cards over the course of a run, so the
    # bookkeeping in profit_tracker has no clean card->profit mapping.
    if not rotate_per_episode:
        _fold_profits_into_registry(
            registry,
            learner_card_id=card.id,
            profit_tracker=profit_tracker,
            seat_to_card=seat_to_card,
            timestep=train_cfg["total_timesteps"],
        )

    _fold_behavior_stats_into_registry(
        registry,
        card_snapshots=wrapped_env.snapshot_card_stats(),
    )

    print(f"\nRegistered {card.id} (gen {card.generation}, parent={resume_from_id})")

    if regression_cfg.get("enabled", True):
        reg = RegressionEval(
            num_hands=regression_cfg.get("num_hands", 500),
            threshold_mbb_per_100=regression_cfg.get("threshold_mbb_per_100", -100.0),
            starting_stack=env_cfg["starting_stack"],
            small_blind=env_cfg["small_blind"],
            big_blind=env_cfg["big_blind"],
            seed=regression_cfg.get("seed", 0),
            include_fixtures=regression_cfg.get("include_fixtures", True),
        )
        reg_result = reg.evaluate(card, registry)
        print(reg.report(reg_result))
        reg.save(reg_result, Path("metrics") / run_name / "regression.json")

    return card


def train(config_path: str, run_name: Optional[str] = None) -> None:
    config = load_config(config_path)
    registry = AgentRegistry()

    cont_cfg = config.get("continuation", {})
    generations = cont_cfg.get("generations", 1)

    base_name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    for gen in range(generations):
        gen_name = base_name if generations == 1 else f"{base_name}_gen{gen}"
        card = train_one_generation(config, gen_name, registry)
        if card and generations > 1:
            config.setdefault("continuation", {})["resume_from"] = card.id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO poker bot")
    parser.add_argument(
        "--config", default="configs/default_config.yaml",
        help="Path to YAML config",
    )
    parser.add_argument(
        "--name", default=None,
        help="Custom run name (default: timestamp)",
    )
    args = parser.parse_args()
    train(args.config, args.name)
