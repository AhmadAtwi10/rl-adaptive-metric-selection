"""
train.py
========
Unified training entry point for all environment and policy variants.

Selects environment and policy automatically from config + CLI flags:

  Default (soft constraints):
    Environment : MonitoringEnv  (soft capacity penalty via β·r_cap)
    Policy      : ActorCritic (flat MLP) or PerMetricActorCritic
                  — controlled by config policy.policy_type

  --masked (hard TopK masking):
    Environment : MonitoringEnvMasked (hard TopK per node, no overflow)
    Policy      : MaskedActorCritic (Gaussian, no β/η reward terms)

Data is saved under:
    data/<N>nodes-<M>metrics-seed<DATA_SEED>/train_episodes.pkl
    data/<N>nodes-<M>metrics-seed<DATA_SEED>/eval_episodes.pkl

keyed by (N, M) AND the data seed, so neither a different problem size nor a
different scenario seed can silently reuse another run's pool.

Seeds are split in two (see utils/seeding.py):
    --data-seed   scenario: episode generation
    --train-seed  training: network init, action sampling, minibatch shuffling
    --seed        legacy convenience, sets both

Usage:
    # Soft env — flat MLP (policy_type: flat in config)
    python scripts/train.py --run-name baseline

    # Soft env — PerMetricActorCritic (policy_type: per_metric in config)
    python scripts/train.py --run-name pm_run

    # Hard-masked env — MaskedActorCritic
    python scripts/train.py --run-name masked_v1 --masked

    # Common options
    python scripts/train.py --run-name run_seed2 --seed 2 --total-timesteps 6000000

    # Multi-seed protocol: same scenario, five different training seeds
    python scripts/train.py --run-name s1 --data-seed 42 --train-seed 101
    python scripts/train.py --run-name s2 --data-seed 42 --train-seed 102

    # Uses iot_cap_range / edge_cap_range / cloud_cap_range from config
    # Uses coverage_mode from config (window or per_timestep)
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
from datetime import datetime
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.synthetic_data import (build_episode_pool, save_dataset, load_dataset,
                                 topology_kwargs, pool_dir,
                                 ensure_pool)
from utils.seeding import (resolve_seeds, apply_seeds_to_cfg,
                           set_global_seeds, describe_seeds)
from utils.run_record import save_run_record
from envs.monitoring_env import MonitoringEnv
from agents.policy import ActorCritic, PerMetricActorCritic, MaskedActorCritic
from agents.ppo import PPOTrainer

# Masked env imported lazily to avoid hard dependency when not used
try:
    from envs.monitoring_env_masked import MonitoringEnvMasked
    _MASKED_AVAILABLE = True
except ImportError:
    _MASKED_AVAILABLE = False


# ------------------------------------------------------------------
# Policy factory
# ------------------------------------------------------------------

def build_policy(cfg: dict, obs_dim: int, action_dim: int) -> torch.nn.Module:
    """
    Instantiate the correct policy class based on config policy_type.

      "flat"       (default) — ActorCritic flat MLP
      "per_metric"           — PerMetricActorCritic shared encoder

    Falls back to "flat" if policy_type is absent so existing configs
    that do not specify it continue to work unchanged.
    """
    pcfg        = cfg["policy"]
    policy_type = pcfg.get("policy_type", "flat")
    ecfg        = cfg["environment"]
    N, M        = ecfg["n_nodes"], ecfg["n_metrics"]
    feature_dim = obs_dim // (N * M)

    if policy_type == "per_metric":
        policy = PerMetricActorCritic(
            n_nodes        = N,
            n_metrics      = M,
            feature_dim    = feature_dim,
            metric_hidden  = pcfg.get("metric_hidden",  128),
            global_hidden  = pcfg.get("global_hidden",  256),
            use_layer_norm = pcfg.get("use_layer_norm", True),
            global_pool    = pcfg.get("global_pool",    "mean"),
        )
        label = (f"PerMetricActorCritic  "
                 f"mh={pcfg.get('metric_hidden',128)}  "
                 f"gh={pcfg.get('global_hidden',256)}  "
                 f"pool={pcfg.get('global_pool','mean')}")
    else:
        policy = ActorCritic(
            obs_dim        = obs_dim,
            action_dim     = action_dim,
            hidden_sizes   = pcfg.get("hidden_sizes", [512, 256]),
            use_layer_norm = pcfg.get("use_layer_norm", True),
        )
        label = f"ActorCritic  hidden={pcfg.get('hidden_sizes',[512,256])}"

    n_params = sum(p.numel() for p in policy.parameters())
    print(f"[train] Policy: {label}  params={n_params:,}")
    return policy


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train RL-based monitoring agent")
    parser.add_argument("--config",           default="configs/config.yaml")
    parser.add_argument("--run-name",         default=None)
    parser.add_argument("--seed",             type=int, default=None,
                        help="Sets BOTH the data and training seed (legacy). "
                             "Use --data-seed / --train-seed to vary them "
                             "independently.")
    parser.add_argument("--data-seed",        type=int, default=None,
                        help="Scenario seed: episode generation. Selects the "
                             "pool directory, so different values give "
                             "different datasets.")
    parser.add_argument("--train-seed",       type=int, default=None,
                        help="Training seed: network init, action sampling, "
                             "minibatch shuffling. Vary this with --data-seed "
                             "held fixed to measure training stability.")
    parser.add_argument("--deterministic",    action="store_true",
                        help="Pin cuDNN and reject nondeterministic kernels. "
                             "Slower; some ops have no deterministic version.")
    parser.add_argument("--total-timesteps",  type=int, default=None)
    parser.add_argument("--device",           default=None)
    parser.add_argument("--no-cuda",          action="store_true")
    parser.add_argument(
        "--masked",
        action="store_true",
        help=(
            "Use MonitoringEnvMasked (hard TopK capacity masking) "
            "with MaskedActorCritic (Gaussian policy). "
            "Default: MonitoringEnv (soft penalty) with ActorCritic "
            "or PerMetricActorCritic depending on config policy_type."
        ),
    )
    return parser.parse_args()


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.total_timesteps is not None:
        cfg["ppo"]["total_timesteps"] = args.total_timesteps

    # Two independent seeds — see utils/seeding.py for why they must not be
    # the same knob.  Resolved seeds are written back into cfg so that the
    # config recorded next to the results reflects what actually ran.
    data_seed, train_seed = resolve_seeds(
        cfg, seed=args.seed, data_seed=args.data_seed, train_seed=args.train_seed
    )
    apply_seeds_to_cfg(cfg, data_seed, train_seed)
    set_global_seeds(train_seed, deterministic=args.deterministic)
    print(f"[train] Seeds: {describe_seeds(data_seed, train_seed, args.deterministic)}")

    # `seed` is kept as an alias for the data seed: everything downstream that
    # used it referred to episode generation.
    seed = data_seed

    if args.no_cuda:
        device = torch.device("cpu")
    elif args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Device: {device}")

    # Validate masked flag
    if args.masked and not _MASKED_AVAILABLE:
        print("[train] ERROR: --masked requires MonitoringEnvMasked "
              "but envs/monitoring_env_masked.py was not found.")
        return

    env_type = "masked" if args.masked else "soft"
    print(f"[train] Environment : {env_type}")

    # ── Run directory ────────────────────────────────────────────
    base_results = cfg["paths"]["results_dir"]
    if args.run_name:
        run_name = args.run_name
    elif cfg["paths"].get("run_name"):
        run_name = cfg["paths"]["run_name"]
    else:
        prefix   = "masked_" if args.masked else ""
        run_name = prefix + datetime.now().strftime("%Y%m%d_%H%M%S")

    results_dir = os.path.join(base_results, run_name)
    os.makedirs(results_dir, exist_ok=True)
    # Record the EFFECTIVE config (post CLI overrides) plus run metadata.
    # Copying args.config verbatim recorded the file's seed, not the one
    # actually used, which made per-seed results unattributable.
    save_run_record(results_dir, cfg, data_seed, train_seed,
                    device=device, deterministic=args.deterministic)

    print(f"[train] Run name    : {run_name}")
    print(f"[train] Results dir : {results_dir}")

    # ── Data ─────────────────────────────────────────────────────
    ecfg = cfg["environment"]
    dcfg = cfg["data"]
    N, M = ecfg["n_nodes"], ecfg["n_metrics"]

    # Single source of truth for load-or-build, shared with run_multiseed's
    # pre-generation step so the two can never produce different pools.
    train_eps, eval_eps, pool_path = ensure_pool(cfg, data_seed, "[train]")
    print(f"[train] Data dir    : {pool_path}")

    print(f"[train] Train episodes: {len(train_eps)}  |  Eval episodes: {len(eval_eps)}")

    # ── Reward weights ───────────────────────────────────────────
    rw = cfg["reward"]
    if args.masked:
        # Masked env has no capacity overflow and no idle penalty
        # — only importance, cost, and switching terms apply
        reward_weights = {
            "alpha":   rw["alpha"],
            "lambda_": rw["lambda_"],
            "delta":   rw["delta"],
        }
    else:
        reward_weights = {
            "alpha":   rw["alpha"],
            "beta":    rw["beta"],
            "lambda_": rw["lambda_"],
            "delta":   rw["delta"],
            "eta":     rw["eta"],
        }

    # ── Environments ─────────────────────────────────────────────
    n_envs        = cfg["ppo"]["n_envs"]
    coverage_mode = ecfg.get("coverage_mode", "window")

    if args.masked:
        def make_env(pool, randomize):
            # Pass coverage_mode if MonitoringEnvMasked accepts it.
            # Fall back gracefully if it does not (older version).
            try:
                return MonitoringEnvMasked(
                    episode_pool         = pool,
                    reward_weights       = reward_weights,
                    per_metric_cost      = ecfg["per_metric_cost"],
                    anomaly_baseline_pct = ecfg["anomaly_baseline_pct"],
                    causal_baseline_pct  = ecfg["causal_baseline_pct"],
                    randomize_episodes   = randomize,
                    window_size          = ecfg.get("window_size", 10),
                    window_agg           = ecfg.get("window_agg", "max"),
                    coverage_mode        = ecfg.get("coverage_mode", "window"),
                )
            except TypeError:
                # MonitoringEnvMasked does not yet accept coverage_mode
                return MonitoringEnvMasked(
                    episode_pool         = pool,
                    reward_weights       = reward_weights,
                    per_metric_cost      = ecfg["per_metric_cost"],
                    anomaly_baseline_pct = ecfg["anomaly_baseline_pct"],
                    causal_baseline_pct  = ecfg["causal_baseline_pct"],
                    randomize_episodes   = randomize,
                    window_size          = ecfg.get("window_size", 10),
                    window_agg           = ecfg.get("window_agg", "max"),
                )
    else:
        def make_env(pool, randomize):
            return MonitoringEnv(
                episode_pool         = pool,
                reward_weights       = reward_weights,
                per_metric_cost      = ecfg["per_metric_cost"],
                anomaly_baseline_pct = ecfg["anomaly_baseline_pct"],
                causal_baseline_pct  = ecfg["causal_baseline_pct"],
                randomize_episodes   = randomize,
                window_size          = ecfg.get("window_size", 1),
                window_agg           = ecfg.get("window_agg", "max"),
                coverage_mode        = coverage_mode,
            )

    train_envs  = [make_env(train_eps, randomize=True)  for _ in range(n_envs)]
    for i, env in enumerate(train_envs):
        env.reset(seed=seed + i)

    n_eval_envs = min(len(eval_eps), cfg["ppo"]["eval_episodes"])
    eval_envs   = [make_env(eval_eps,  randomize=False) for _ in range(n_eval_envs)]

    # ── Policy ───────────────────────────────────────────────────
    obs_dim    = train_envs[0].observation_space.shape[0]
    action_dim = (train_envs[0].action_space.n
                  if hasattr(train_envs[0].action_space, "n")
                  else train_envs[0].action_space.shape[0])

    if args.masked:
        # MaskedActorCritic is always used for the masked env — it is not
        # routed through build_policy() because:
        #   1. MonitoringEnvMasked has a continuous Box action space (not MultiBinary)
        #      → action_dim = action_space.shape[0], not action_space.n
        #   2. MaskedActorCritic uses a Gaussian policy over continuous logits;
        #      the environment then applies a hard TopK mask per node
        #   3. PerMetricActorCritic uses Bernoulli (binary) outputs — incompatible
        #      with the masked env's continuous action interface
        # build_policy() only handles flat/per_metric for the soft env.
        pcfg   = cfg["policy"]
        policy = MaskedActorCritic(
            obs_dim        = obs_dim,
            action_dim     = action_dim,
            hidden_sizes   = pcfg.get("hidden_sizes", [512, 256]),
            use_layer_norm = pcfg.get("use_layer_norm", True),
        )
        n_params = sum(p.numel() for p in policy.parameters())
        print(f"[train] Policy: MaskedActorCritic  "
              f"hidden={pcfg.get('hidden_sizes',[512,256])}  "
              f"params={n_params:,}")
    else:
        policy = build_policy(cfg, obs_dim, action_dim)

    print(f"[train] obs_dim={obs_dim}  action_dim={action_dim}")

    # ── Trainer ──────────────────────────────────────────────────
    trainer = PPOTrainer(
        envs        = train_envs,
        eval_envs   = eval_envs,
        policy      = policy,
        cfg         = cfg,
        device      = device,
        results_dir = results_dir,
    )

    trainer.train()

    print(f"\n[train] Run '{run_name}' complete.")
    print(f"[train] Environment : {env_type}")
    print(f"[train] Data        → {pool_path}")
    print(f"[train] Best model  → {trainer.model_path}")
    print(f"[train] Checkpoint  → {trainer.checkpoint_path}")
    print(f"\n[train] TensorBoard → tensorboard --logdir {base_results}")

    print("\n[train] Final evaluation:")
    final_stats = trainer.evaluate(n_episodes=cfg["ppo"]["eval_episodes"])
    for k, v in final_stats.items():
        print(f"  {k:<22}: {v:.4f}")


if __name__ == "__main__":
    main()
elif __name__ == "scripts.train":
    main()