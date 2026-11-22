"""
train_masked.py
===============
Training entry point for the hard-masked monitoring environment.

Data is saved under:
    data/<N>nodes-<M>metrics/train_episodes.pkl
    data/<N>nodes-<M>metrics/eval_episodes.pkl

Usage:
    PYTHONPATH=. python scripts/train_masked.py --run-name masked_v1
    PYTHONPATH=. python scripts/train_masked.py --run-name masked_v1 --total-timesteps 6000000
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

from data.synthetic_data import build_episode_pool, save_dataset, load_dataset, topology_kwargs, pool_dir, pool_config_key
from utils.seeding import (resolve_seeds, apply_seeds_to_cfg,
                           set_global_seeds, describe_seeds)
from utils.run_record import save_run_record
from envs.monitoring_env_masked import MonitoringEnvMasked
from agents.policy import MaskedActorCritic
from agents.ppo import PPOTrainer


# ------------------------------------------------------------------
# Helper — canonical data subdirectory for a given (N, M)
# ------------------------------------------------------------------



# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train RL agent with hard-masked environment")
    p.add_argument("--config",          default="configs/config.yaml")
    p.add_argument("--run-name",        default=None)
    p.add_argument("--seed",            type=int, default=None)
    p.add_argument("--data-seed",  type=int, default=None,
                   help="Scenario seed: episode generation / pool selection.")
    p.add_argument("--train-seed", type=int, default=None,
                   help="Training seed: init, action sampling, batching.")
    p.add_argument("--deterministic", action="store_true",
                   help="Pin cuDNN and reject nondeterministic kernels.")
    p.add_argument("--total-timesteps", type=int, default=None)
    p.add_argument("--device",          default=None)
    p.add_argument("--no-cuda",         action="store_true")
    return p.parse_args()


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.total_timesteps is not None:
        cfg["ppo"]["total_timesteps"] = args.total_timesteps

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

    if args.no_cuda:
        device = torch.device("cpu")
    elif args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_masked] Device: {device}")

    # ── Run directory ────────────────────────────────────────────
    base_results = cfg["paths"]["results_dir"]
    if args.run_name:
        run_name = args.run_name
    elif cfg["paths"].get("run_name"):
        run_name = cfg["paths"]["run_name"]
    else:
        run_name = "masked_" + datetime.now().strftime("%Y%m%d_%H%M%S")

    results_dir = os.path.join(base_results, run_name)
    os.makedirs(results_dir, exist_ok=True)
    save_run_record(results_dir, cfg, data_seed, train_seed,
                    device=str(device),
                    deterministic=getattr(args, "deterministic", False))

    print(f"[train_masked] Run name    : {run_name}")
    print(f"[train_masked] Results dir : {results_dir}")
    print(f"[train_masked] Environment : MonitoringEnvMasked (hard TopK masking)")

    # ── Data ─────────────────────────────────────────────────────
    ecfg = cfg["environment"]
    dcfg = cfg["data"]
    N, M = ecfg["n_nodes"], ecfg["n_metrics"]

    pool_path   = pool_dir(cfg["paths"]["data_dir"], N, M, data_seed,
                           config_key=pool_config_key(cfg))
    train_path = os.path.join(pool_path, "train_episodes.pkl")
    eval_path  = os.path.join(pool_path, "eval_episodes.pkl")

    print(f"[train_masked] Data dir    : {pool_path}")

    if os.path.exists(train_path) and os.path.exists(eval_path):
        print("[train_masked] Loading pre-generated datasets...")
        train_eps = load_dataset(train_path)
        eval_eps  = load_dataset(eval_path)
    else:
        print(f"[train_masked] Generating datasets for N={N}, M={M}...")
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
        print(f"[train_masked] Saved → {train_path}")
        print(f"[train_masked] Saved → {eval_path}")

    print(f"[train_masked] Train: {len(train_eps)} eps  Eval: {len(eval_eps)} eps")

    # ── Reward weights (no beta/eta for masked env) ──────────────
    rw = cfg["reward"]
    reward_weights = {
        "alpha":   rw["alpha"],
        "lambda_": rw["lambda_"],
        "delta":   rw["delta"],
    }

    # ── Environments ─────────────────────────────────────────────
    n_envs = cfg["ppo"]["n_envs"]

    def make_masked_env(pool, randomize):
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

    train_envs  = [make_masked_env(train_eps, randomize=True)  for _ in range(n_envs)]
    n_eval_envs = min(len(eval_eps), cfg["ppo"]["eval_episodes"])
    eval_envs   = [make_masked_env(eval_eps,  randomize=False) for _ in range(n_eval_envs)]

    for i, env in enumerate(train_envs):
        env.reset(seed=seed + i)

    # ── Policy ───────────────────────────────────────────────────
    obs_dim    = train_envs[0].observation_space.shape[0]
    action_dim = train_envs[0].action_space.shape[0]

    pcfg   = cfg["policy"]
    policy = MaskedActorCritic(
        obs_dim        = obs_dim,
        action_dim     = action_dim,
        hidden_sizes   = pcfg["hidden_sizes"],
        use_layer_norm = pcfg["use_layer_norm"],
    )
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"[train_masked] obs_dim={obs_dim}  action_dim={action_dim}  params={n_params:,}")

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

    print(f"\n[train_masked] Run '{run_name}' complete.")
    print(f"[train_masked] Data       → {pool_path}")
    print(f"[train_masked] Best model → {trainer.model_path}")
    print(f"[train_masked] Checkpoint → {trainer.checkpoint_path}")
    print(f"[train_masked] TensorBoard → tensorboard --logdir {base_results}")

    print("\n[train_masked] Final evaluation:")
    final = trainer.evaluate(n_episodes=cfg["ppo"]["eval_episodes"])
    for k, v in final.items():
        print(f"  {k:<22}: {v:.4f}")


if __name__ == "__main__":
    main()
elif __name__ == "scripts.train_masked":
    main()