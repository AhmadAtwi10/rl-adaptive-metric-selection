"""
train_per_metric.py
===================
Training script for the PerMetricActorCritic architecture.

Use this instead of train.py when N and M are large (e.g. N=28, M=38).
Everything is identical to train.py except the policy class and its
instantiation. All other components (PPOTrainer, environments, data
loading, checkpointing) are unchanged.

Why a separate script
---------------------
PerMetricActorCritic takes (n_nodes, n_metrics, feature_dim) as
constructor arguments instead of (obs_dim, action_dim). This is the
only difference from train.py. Keeping a separate entry point avoids
adding conditional logic to the main train.py.

Architecture parameters (set in config.yaml under policy:)
-----------------------------------------------------------
  metric_hidden  : hidden dim of the shared per-metric encoder  (default 128)
  global_hidden  : hidden dim of the global context encoder     (default 256)
  use_layer_norm : apply LayerNorm after each linear layer      (default true)

Usage
-----
  python scripts/train_per_metric.py --run-name n28m38_v1
  python scripts/train_per_metric.py --run-name n28m38_v1 --total-timesteps 50000000
  python scripts/train_per_metric.py --config configs/config_n28m38.yaml --run-name n28m38_v1
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
from envs.monitoring_env import MonitoringEnv
from agents.policy import PerMetricActorCritic
from agents.ppo import PPOTrainer


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------



def parse_args():
    p = argparse.ArgumentParser(
        description="Train PerMetricActorCritic for large N, M configurations"
    )
    p.add_argument("--config",           default="configs/config.yaml")
    p.add_argument("--run-name",         default=None)
    p.add_argument("--seed",             type=int,   default=None)
    p.add_argument("--data-seed",  type=int, default=None,
                   help="Scenario seed: episode generation / pool selection.")
    p.add_argument("--train-seed", type=int, default=None,
                   help="Training seed: init, action sampling, batching.")
    p.add_argument("--deterministic", action="store_true",
                   help="Pin cuDNN and reject nondeterministic kernels.")
    p.add_argument("--total-timesteps",  type=int,   default=None)
    p.add_argument("--metric-hidden",    type=int,   default=None,
                   help="Hidden dim of per-metric encoder. "
                        "Default: reads from config policy.metric_hidden, fallback=128")
    p.add_argument("--global-hidden",    type=int,   default=None,
                   help="Hidden dim of global context encoder. "
                        "Default: reads from config policy.global_hidden, fallback=256")
    p.add_argument("--global-pool",      default=None,
                   choices=["mean", "concat"],
                   help=(
                       "How to aggregate local embeddings for global context. "
                       "'mean'   — scalable, recommended for large N*M (N=28, M=38). "
                       "'concat' — more expressive, only for small N*M (N<=10, M<=15). "
                       "Default: reads from config policy.global_pool, fallback=mean."
                   ))
    p.add_argument("--device",           default=None)
    p.add_argument("--no-cuda",          action="store_true")
    return p.parse_args()


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # CLI overrides
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

    # Device
    if args.no_cuda:
        device = torch.device("cpu")
    elif args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_pm] Device: {device}")

    # ── Run directory ────────────────────────────────────────────
    base_results = cfg["paths"]["results_dir"]
    if args.run_name:
        run_name = args.run_name
    elif cfg["paths"].get("run_name"):
        run_name = cfg["paths"]["run_name"]
    else:
        run_name = "pm_" + datetime.now().strftime("%Y%m%d_%H%M%S")

    results_dir = os.path.join(base_results, run_name)
    os.makedirs(results_dir, exist_ok=True)
    save_run_record(results_dir, cfg, data_seed, train_seed,
                    device=str(device),
                    deterministic=getattr(args, "deterministic", False))

    print(f"[train_pm] Run name      : {run_name}")
    print(f"[train_pm] Results dir   : {results_dir}")
    print(f"[train_pm] Architecture  : PerMetricActorCritic")

    # ── Data ─────────────────────────────────────────────────────
    ecfg = cfg["environment"]
    dcfg = cfg["data"]
    N, M = ecfg["n_nodes"], ecfg["n_metrics"]

    pool_path   = pool_dir(cfg["paths"]["data_dir"], N, M, data_seed,
                           config_key=pool_config_key(cfg))
    train_path = os.path.join(pool_path, "train_episodes.pkl")
    eval_path  = os.path.join(pool_path, "eval_episodes.pkl")

    print(f"[train_pm] Data dir      : {pool_path}")

    if os.path.exists(train_path) and os.path.exists(eval_path):
        print("[train_pm] Loading pre-generated datasets...")
        train_eps = load_dataset(train_path)
        eval_eps  = load_dataset(eval_path)
    else:
        print(f"[train_pm] Generating datasets for N={N}, M={M}...")
        train_eps = build_episode_pool(
            n_episodes           = dcfg["n_train_episodes"],
            n_nodes=N, n_metrics=M,
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
            n_nodes=N, n_metrics=M,
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
        print(f"[train_pm] Saved → {train_path}")
        print(f"[train_pm] Saved → {eval_path}")

    print(f"[train_pm] Train: {len(train_eps)} eps  |  Eval: {len(eval_eps)} eps")

    # ── Reward weights ────────────────────────────────────────────
    rw = cfg["reward"]
    reward_weights = {
        "alpha":   rw["alpha"],   "beta":    rw["beta"],
        "lambda_": rw["lambda_"], "delta":   rw["delta"],
        "eta":     rw["eta"],
    }

    # ── Environments ─────────────────────────────────────────────
    n_envs = cfg["ppo"]["n_envs"]

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
            coverage_mode        = ecfg.get("coverage_mode", "window"),
        )

    train_envs  = [make_env(train_eps, randomize=True)  for _ in range(n_envs)]
    n_eval_envs = min(len(eval_eps), cfg["ppo"]["eval_episodes"])
    eval_envs   = [make_env(eval_eps,  randomize=False) for _ in range(n_eval_envs)]

    for i, env in enumerate(train_envs):
        env.reset(seed=seed + i)

    # ── Policy (PerMetricActorCritic) ─────────────────────────────
    # Feature dim: number of features per (node, metric) pair
    # obs_dim = N*M*feature_dim → feature_dim = obs_dim / (N*M)
    obs_dim     = train_envs[0].observation_space.shape[0]
    action_dim  = train_envs[0].action_space.n
    feature_dim = obs_dim // (N * M)   # = 6 with current monitoring_env

    pcfg          = cfg["policy"]
    metric_hidden = (args.metric_hidden
                     or pcfg.get("metric_hidden", 128))
    global_hidden = (args.global_hidden
                     or pcfg.get("global_hidden", 256))
    global_pool   = (args.global_pool
                     or pcfg.get("global_pool", "mean"))
    use_layer_norm = pcfg.get("use_layer_norm", True)

    policy = PerMetricActorCritic(
        n_nodes        = N,
        n_metrics      = M,
        feature_dim    = feature_dim,
        metric_hidden  = metric_hidden,
        global_hidden  = global_hidden,
        use_layer_norm = use_layer_norm,
        global_pool    = global_pool,
    )

    n_params = sum(p.numel() for p in policy.parameters())

    print(f"[train_pm] N={N}  M={M}  F={feature_dim}")
    print(f"[train_pm] obs_dim={obs_dim}  action_dim={action_dim}")
    print(f"[train_pm] metric_hidden={metric_hidden}  "
          f"global_hidden={global_hidden}  "
          f"global_pool={global_pool}")
    print(f"[train_pm] Parameters: {n_params:,}")
    print(f"[train_pm] Timesteps : {cfg['ppo']['total_timesteps']:,}")

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

    print(f"\n[train_pm] Run '{run_name}' complete.")
    print(f"[train_pm] Data       → {pool_path}")
    print(f"[train_pm] Best model → {trainer.model_path}")
    print(f"[train_pm] Checkpoint → {trainer.checkpoint_path}")
    print(f"\n[train_pm] TensorBoard → tensorboard --logdir {base_results}")

    print("\n[train_pm] Final evaluation:")
    final = trainer.evaluate(n_episodes=cfg["ppo"]["eval_episodes"])
    for k, v in final.items():
        print(f"  {k:<24}: {v:.4f}")


if __name__ == "__main__":
    main()
elif __name__ == "scripts.train_per_metric":
    main()