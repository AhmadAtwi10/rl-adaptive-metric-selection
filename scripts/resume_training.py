"""
resume_training.py
==================
Continues training a PPO agent from a saved checkpoint.

Data is read from:
    data/<N>nodes-<M>metrics/train_episodes.pkl
    data/<N>nodes-<M>metrics/eval_episodes.pkl

matching the layout written by train.py.

Usage:
    python scripts/resume_training.py --run-name my_run --extra-timesteps 2000000
    python scripts/resume_training.py --run-name my_run --extra-timesteps 1000000 --lr 1e-4
    python scripts/resume_training.py --run-name masked_v1 --masked --extra-timesteps 2000000
    python scripts/resume_training.py --run-name pm_run --policy-type per_metric --extra-timesteps 5000000
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.synthetic_data import build_episode_pool, save_dataset, load_dataset, topology_kwargs, pool_dir, pool_config_key
from utils.seeding import (resolve_seeds, apply_seeds_to_cfg,
                           set_global_seeds, describe_seeds)
from utils.run_record import save_run_record
from envs.monitoring_env import MonitoringEnv
from agents.policy import ActorCritic, PerMetricActorCritic, MaskedActorCritic
from agents.ppo import PPOTrainer


# ------------------------------------------------------------------
# Helper — canonical data subdirectory for a given (N, M)
# ------------------------------------------------------------------



# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Resume PPO training from a checkpoint")
    p.add_argument("--run-name",        required=True,
                   help="Run subdirectory under results/ to resume")
    p.add_argument("--config",          default=None,
                   help="Config file. Default: results/<run-name>/config.yaml")
    p.add_argument("--checkpoint",      default=None,
                   help="Checkpoint path. Default: results/<run-name>/checkpoint.pt")
    p.add_argument("--extra-timesteps", type=int, default=2_000_000)
    p.add_argument("--lr",              type=float, default=None,
                   help="Override learning rate (fine-tuning)")
    p.add_argument("--masked",          action="store_true",
                   help="Use MonitoringEnvMasked + MaskedActorCritic")
    p.add_argument(
        "--policy-type",
        default=None,
        choices=["flat", "per_metric"],
        help=(
            "Policy architecture. "
            "'flat'       — ActorCritic flat MLP (default). "
            "'per_metric' — PerMetricActorCritic shared encoder. "
            "Default: reads from config policy.policy_type, fallback='flat'. "
            "Must match the architecture used during original training."
        ),
    )
    p.add_argument("--seed",       type=int, default=None,
                   help="Sets BOTH data and training seed (legacy).")
    p.add_argument("--data-seed",  type=int, default=None,
                   help="Scenario seed: episode generation / pool selection.")
    p.add_argument("--train-seed", type=int, default=None,
                   help="Training seed: init, action sampling, batching.")
    p.add_argument("--deterministic", action="store_true",
                   help="Pin cuDNN and reject nondeterministic kernels.")
    p.add_argument("--device",          default=None)
    p.add_argument("--no-cuda",         action="store_true")
    return p.parse_args()


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
    print(f"[resume] Policy: {label}  params={n_params:,}")
    return policy


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    args = parse_args()

    # ── Config ───────────────────────────────────────────────────
    if args.config:
        config_path = args.config
    else:
        config_path = os.path.join("results", args.run_name, "config.yaml")
        if not os.path.exists(config_path):
            config_path = "configs/config.yaml"
            print(f"[resume] Run config not found — using {config_path}")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    cfg["ppo"]["total_timesteps"] = args.extra_timesteps
    if args.lr is not None:
        cfg["ppo"]["learning_rate"] = args.lr
        print(f"[resume] Learning rate overridden → {args.lr}")

    # ── Paths ────────────────────────────────────────────────────
    results_dir = os.path.join(cfg["paths"]["results_dir"], args.run_name)
    model_path  = os.path.join(results_dir, cfg["paths"]["model_name"])
    ckpt_path   = args.checkpoint or os.path.join(results_dir, "checkpoint.pt")

    if not os.path.exists(ckpt_path):
        if os.path.exists(model_path):
            print(f"[resume] No checkpoint.pt — falling back to {model_path}")
            print(f"[resume] Optimizer state starts fresh (policy weights preserved).")
            ckpt_path = model_path
        else:
            print(f"[resume] ERROR: No checkpoint or model found in {results_dir}")
            print(f"[resume] Train first:  python scripts/train.py --run-name {args.run_name}")
            return

    # ── Device ───────────────────────────────────────────────────
    if args.no_cuda:
        device = torch.device("cpu")
    elif args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[resume] Device: {device}")

    data_seed, train_seed = resolve_seeds(
        cfg,
        seed=getattr(args, "seed", None),
        data_seed=getattr(args, "data_seed", None),
        train_seed=getattr(args, "train_seed", None),
    )
    apply_seeds_to_cfg(cfg, data_seed, train_seed)
    set_global_seeds(train_seed, deterministic=getattr(args, "deterministic", False))
    print(f"[seed] {describe_seeds(data_seed, train_seed)}")
    seed = data_seed

    # Resolve policy_type: CLI > config > default "flat"
    if not args.masked:
        policy_type = (
            args.policy_type
            or cfg["policy"].get("policy_type", "flat")
        )
        cfg["policy"]["policy_type"] = policy_type
        print(f"[resume] Policy type : {policy_type}")

    # ── Data ─────────────────────────────────────────────────────
    ecfg = cfg["environment"]
    dcfg = cfg["data"]
    N, M = ecfg["n_nodes"], ecfg["n_metrics"]

    pool_path   = pool_dir(cfg["paths"]["data_dir"], N, M, data_seed,
                           config_key=pool_config_key(cfg))
    train_path = os.path.join(pool_path, "train_episodes.pkl")
    eval_path  = os.path.join(pool_path, "eval_episodes.pkl")

    print(f"[resume] Data dir : {pool_path}")

    if os.path.exists(train_path) and os.path.exists(eval_path):
        print("[resume] Loading existing episode pools...")
        train_eps = load_dataset(train_path)
        eval_eps  = load_dataset(eval_path)
    else:
        print("[resume] Episode pools not found — regenerating...")
        train_eps = build_episode_pool(
            n_episodes           = dcfg["n_train_episodes"],
            n_nodes              = N,
            n_metrics            = M,
            episode_length       = ecfg["episode_length"],
            anomaly_density      = dcfg["anomaly_density"],
            causal_density       = dcfg["causal_density"],
            base_seed            = seed,
            anomaly_baseline_pct = ecfg["anomaly_baseline_pct"],
            causal_baseline_pct  = ecfg["causal_baseline_pct"],
            **topology_kwargs(ecfg),
        )
        eval_eps = build_episode_pool(
            n_episodes           = dcfg["n_eval_episodes"],
            n_nodes              = N,
            n_metrics            = M,
            episode_length       = ecfg["episode_length"],
            anomaly_density      = dcfg["anomaly_density"] * 1.2,
            causal_density       = dcfg["causal_density"],
            base_seed            = seed + 10000,
            anomaly_baseline_pct = ecfg["anomaly_baseline_pct"],
            causal_baseline_pct  = ecfg["causal_baseline_pct"],
            **topology_kwargs(ecfg),
        )
        save_dataset(train_eps, train_path)
        save_dataset(eval_eps,  eval_path)
        print(f"[resume] Saved → {train_path}")
        print(f"[resume] Saved → {eval_path}")

    print(f"[resume] Train: {len(train_eps)} eps  |  Eval: {len(eval_eps)} eps")

    # ── Reward weights ───────────────────────────────────────────
    rw = cfg["reward"]
    reward_weights = {
        "alpha":   rw["alpha"], "beta":    rw["beta"],
        "lambda_": rw["lambda_"], "delta": rw["delta"], "eta": rw["eta"],
    }

    # ── Environments ─────────────────────────────────────────────
    n_envs = cfg["ppo"]["n_envs"]

    if args.masked:
        from envs.monitoring_env_masked import MonitoringEnvMasked
        from agents.policy import MaskedActorCritic
        masked_rw = {k: v for k, v in reward_weights.items() if k not in ("beta", "eta")}
        def make_env(pool, randomize):
            return MonitoringEnvMasked(
                episode_pool=pool, reward_weights=masked_rw,
                per_metric_cost=ecfg["per_metric_cost"],
                anomaly_baseline_pct=ecfg["anomaly_baseline_pct"],
                causal_baseline_pct=ecfg["causal_baseline_pct"],
                randomize_episodes=randomize,
                window_size=ecfg.get("window_size", 10),
                window_agg=ecfg.get("window_agg", "max"),
            )
    else:
        def make_env(pool, randomize):
            return MonitoringEnv(
                episode_pool=pool, reward_weights=reward_weights,
                per_metric_cost=ecfg["per_metric_cost"],
                anomaly_baseline_pct=ecfg["anomaly_baseline_pct"],
                causal_baseline_pct=ecfg["causal_baseline_pct"],
                randomize_episodes=randomize,
                window_size=ecfg.get("window_size", 1),
                window_agg=ecfg.get("window_agg", "max"),
                coverage_mode=ecfg.get("coverage_mode", "window"),
            )

    train_envs  = [make_env(train_eps, randomize=True)  for _ in range(n_envs)]
    n_eval_envs = min(len(eval_eps), cfg["ppo"]["eval_episodes"])
    eval_envs   = [make_env(eval_eps,  randomize=False) for _ in range(n_eval_envs)]

    for i, env in enumerate(train_envs):
        env.reset(seed=seed + i)

    # ── Policy ───────────────────────────────────────────────────
    obs_dim    = train_envs[0].observation_space.shape[0]
    action_dim = (train_envs[0].action_space.n
                  if hasattr(train_envs[0].action_space, "n")
                  else train_envs[0].action_space.shape[0])

    if args.masked:
        policy = MaskedActorCritic(
            obs_dim=obs_dim, action_dim=action_dim,
            hidden_sizes=cfg["policy"].get("hidden_sizes", [512, 256]),
            use_layer_norm=cfg["policy"].get("use_layer_norm", True),
        )
        n_params = sum(p.numel() for p in policy.parameters())
        print(f"[resume] Policy: MaskedActorCritic  params={n_params:,}")
    else:
        policy = build_policy(cfg, obs_dim, action_dim)

    print(f"[resume] obs_dim={obs_dim}  action_dim={action_dim}")

    # ── Trainer + load checkpoint ─────────────────────────────────
    trainer = PPOTrainer(
        envs=train_envs, eval_envs=eval_envs,
        policy=policy, cfg=cfg, device=device, results_dir=results_dir,
    )

    print(f"\n[resume] Loading: {ckpt_path}")
    trainer.load_checkpoint(ckpt_path)

    # Apply LR override to already-loaded optimizer if requested
    if args.lr is not None:
        for param_group in trainer.optimizer.param_groups:
            param_group["lr"] = args.lr

    # ── Summary ──────────────────────────────────────────────────
    steps_done  = trainer._global_step
    steps_total = steps_done + args.extra_timesteps

    print(f"\n[resume] ─────────────────────────────────────────────")
    print(f"[resume] Run            : {args.run_name}")
    print(f"[resume] Data dir       : {pool_path}")
    print(f"[resume] Steps done     : {steps_done:,}")
    print(f"[resume] Steps to add   : {args.extra_timesteps:,}")
    print(f"[resume] Total steps    : {steps_total:,}")
    print(f"[resume] Best reward so : {trainer._best_eval_reward:.4f}")
    print(f"[resume] Checkpoint out : {trainer.checkpoint_path}")
    print(f"[resume] Best model out : {trainer.model_path}")
    print(f"[resume] ─────────────────────────────────────────────\n")

    # ── Train ────────────────────────────────────────────────────
    trainer.train()

    print(f"\n[resume] Done.")
    print(f"[resume] Total steps : {steps_done:,} + {args.extra_timesteps:,} = {steps_total:,}")
    print(f"[resume] Best reward : {trainer._best_eval_reward:.4f}")

    print("\n[resume] Final evaluation:")
    final = trainer.evaluate(n_episodes=cfg["ppo"]["eval_episodes"])
    for k, v in final.items():
        print(f"  {k:<22}: {v:.4f}")


if __name__ == "__main__":
    main()
elif __name__ == "scripts.resume_training":
    main()