"""
train_ablation.py
=================
Trains one PPO agent per reward weight configuration (ablation study),
then evaluates all trained agents on the same eval pool and prints
a single comparison table.

Each configuration trains for the full total_timesteps from config.yaml
and saves independently to results/ablation/<config_name>/.

Ablation configurations (Table II of the paper; alpha, beta, lambda, delta, eta)
-----------------------
  importance_only      10,  0,  0,   0,   0
  importance_cost      10,  0,  3,   0,   0
  importance_capacity  10,  3,  0,   0,   0
  no_importance         0,  3,  3,   1,   0
  full_reward          10,  1,  0.1, 0.1, 2

Usage
-----
  # Run all configurations (sequential, uses config.yaml)
  python scripts/train_ablation.py

  # Run only specific configurations
  python scripts/train_ablation.py --configs importance_only full_reward

  # Use PerMetricActorCritic instead of flat MLP
  python scripts/train_ablation.py --policy-type per_metric

  # Custom timesteps (e.g. shorter for quick test)
  python scripts/train_ablation.py --timesteps 1000000

  # Skip training, only re-evaluate existing models
  python scripts/train_ablation.py --eval-only

  # Use a different config file
  python scripts/train_ablation.py --config configs/config_default.yaml
"""

import os
import sys
import argparse
import yaml
import copy
import csv
import numpy as np
import torch
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.synthetic_data import build_episode_pool, load_dataset, save_dataset, topology_kwargs, pool_dir, pool_config_key
from utils.seeding import (resolve_seeds, apply_seeds_to_cfg,
                           set_global_seeds, describe_seeds)
from utils.run_record import save_run_record
from envs.monitoring_env import MonitoringEnv
from agents.policy import ActorCritic, PerMetricActorCritic
from agents.ppo import PPOTrainer


# ══════════════════════════════════════════════════════════════════
# Ablation configurations
# ══════════════════════════════════════════════════════════════════

ABLATION_CONFIGS = {
    # The five reward-weight settings of Table II in the paper (alpha, beta, lambda, delta, eta).
    "importance_only": {
        "label":   "importance only",
        "weights": {"alpha": 10.0, "beta": 0.0, "lambda_": 0.0,
                    "delta": 0.0,  "eta":  0.0},
    },
    "importance_cost": {
        "label":   "importance + cost",
        "weights": {"alpha": 10.0, "beta": 0.0, "lambda_": 3.0,
                    "delta": 0.0,  "eta":  0.0},
    },
    "importance_capacity": {
        "label":   "importance + capacity",
        "weights": {"alpha": 10.0, "beta": 3.0, "lambda_": 0.0,
                    "delta": 0.0,  "eta":  0.0},
    },
    "no_importance": {
        "label":   "no importance (alpha = 0)",
        "weights": {"alpha": 0.0,  "beta": 3.0, "lambda_": 3.0,
                    "delta": 1.0,  "eta":  0.0},
    },
    "full_reward": {
        "label":   "full reward",
        "weights": {"alpha": 10.0, "beta": 1.0, "lambda_": 0.1,
                    "delta": 0.1,  "eta":  2.0},
    },
}

# ══════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Train one PPO agent per reward ablation configuration"
    )
    p.add_argument("--config",       default="configs/config.yaml",
                   help="Base config file")
    p.add_argument("--configs",      nargs="+", default=None,
                   choices=list(ABLATION_CONFIGS.keys()),
                   help="Which configurations to run. Default: all.")
    p.add_argument("--timesteps",    type=int, default=None,
                   help="Override total_timesteps per run")
    p.add_argument("--eval-only",    action="store_true",
                   help="Skip training, only evaluate existing models")
    p.add_argument("--seed",       type=int, default=None,
                   help="Sets BOTH data and training seed (legacy).")
    p.add_argument("--data-seed",  type=int, default=None,
                   help="Scenario seed: episode generation / pool selection.")
    p.add_argument("--train-seed", type=int, default=None,
                   help="Training seed: init, action sampling, batching.")
    p.add_argument("--deterministic", action="store_true",
                   help="Pin cuDNN and reject nondeterministic kernels.")
    p.add_argument("--device",       default=None)
    p.add_argument("--no-cuda",      action="store_true")
    p.add_argument(
        "--policy-type",
        default=None,
        choices=["flat", "per_metric"],
        help=(
            "Policy architecture to use for all ablation runs. "
            "'flat'       — ActorCritic flat MLP (default if not set in config). "
            "'per_metric' — PerMetricActorCritic shared encoder (recommended for large N,M). "
            "Default: reads from config policy.policy_type, fallback='flat'."
        ),
    )
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════



def make_envs(episode_pool, reward_weights, cfg, n, randomize, seed=0):
    ecfg = cfg["environment"]
    envs = []
    for i in range(n):
        env = MonitoringEnv(
            episode_pool         = episode_pool,
            reward_weights       = reward_weights,
            per_metric_cost      = ecfg["per_metric_cost"],
            anomaly_baseline_pct = ecfg["anomaly_baseline_pct"],
            causal_baseline_pct  = ecfg["causal_baseline_pct"],
            randomize_episodes   = randomize,
            window_size          = ecfg.get("window_size", 1),
            window_agg           = ecfg.get("window_agg", "max"),
            coverage_mode        = ecfg.get("coverage_mode", "window"),
        )
        env.reset(seed=seed + i)
        envs.append(env)
    return envs


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
        n_params = sum(p.numel() for p in policy.parameters())
        print(f"[policy] PerMetricActorCritic  "
              f"mh={pcfg.get('metric_hidden',128)}  "
              f"gh={pcfg.get('global_hidden',256)}  "
              f"pool={pcfg.get('global_pool','mean')}  "
              f"params={n_params:,}")
    else:
        policy = ActorCritic(
            obs_dim        = obs_dim,
            action_dim     = action_dim,
            hidden_sizes   = pcfg.get("hidden_sizes", [512, 256]),
            use_layer_norm = pcfg.get("use_layer_norm", True),
        )
        n_params = sum(p.numel() for p in policy.parameters())
        print(f"[policy] ActorCritic  "
              f"hidden={pcfg.get('hidden_sizes',[512,256])}  "
              f"params={n_params:,}")

    return policy


# ══════════════════════════════════════════════════════════════════
# Train one configuration
# ══════════════════════════════════════════════════════════════════

def train_one(
    config_name: str,
    config_info: dict,
    cfg: dict,
    train_eps: list,
    eval_eps: list,
    device: torch.device,
    ablation_base_dir: str,
) -> str:
    """
    Train one PPO agent for the given reward configuration.
    Returns the path to the saved best model.
    """
    label         = config_info["label"]
    reward_weights = config_info["weights"]

    results_dir = os.path.join(ablation_base_dir, config_name)
    os.makedirs(results_dir, exist_ok=True)

    # Save a copy of the config used for this run (with overridden weights)
    run_cfg = copy.deepcopy(cfg)
    run_cfg["reward"]["alpha"]   = reward_weights["alpha"]
    run_cfg["reward"]["beta"]    = reward_weights["beta"]
    run_cfg["reward"]["lambda_"] = reward_weights["lambda_"]
    run_cfg["reward"]["delta"]   = reward_weights["delta"]
    run_cfg["reward"]["eta"]     = reward_weights["eta"]
    with open(os.path.join(results_dir, "config.yaml"), "w") as f:
        yaml.dump(run_cfg, f, default_flow_style=False)

    print(f"\n{'━'*65}")
    print(f"  Training: {label}")
    print(f"  Weights : α={reward_weights['alpha']}  "
          f"β={reward_weights['beta']}  "
          f"λ={reward_weights['lambda_']}  "
          f"δ={reward_weights['delta']}  "
          f"η={reward_weights['eta']}")
    print(f"  Results : {results_dir}")
    print(f"{'━'*65}\n")

    seed   = cfg["data"]["seed"]
    pcfg   = cfg["ppo"]
    n_envs = pcfg["n_envs"]

    train_envs  = make_envs(train_eps, reward_weights, cfg,
                            n=n_envs, randomize=True, seed=seed)
    n_eval_envs = min(len(eval_eps), pcfg["eval_episodes"])
    eval_envs   = make_envs(eval_eps, reward_weights, cfg,
                            n=n_eval_envs, randomize=False, seed=seed)

    # Policy — type and dims resolved via build_policy
    obs_dim    = train_envs[0].observation_space.shape[0]
    action_dim = train_envs[0].action_space.n

    policy = build_policy(run_cfg, obs_dim, action_dim)

    trainer = PPOTrainer(
        envs        = train_envs,
        eval_envs   = eval_envs,
        policy      = policy,
        cfg         = run_cfg,
        device      = device,
        results_dir = results_dir,
    )

    trainer.train()

    print(f"\n  [{config_name}] done — "
          f"best reward={trainer._best_eval_reward:.3f}  "
          f"model → {trainer.model_path}")

    return trainer.model_path


# ══════════════════════════════════════════════════════════════════
# Evaluate one saved model
# ══════════════════════════════════════════════════════════════════

def evaluate_one(
    config_name: str,
    config_info: dict,
    cfg: dict,
    eval_eps: list,
    device: torch.device,
    ablation_base_dir: str,
) -> dict:
    """
    Load the saved best model for this config and run deterministic
    evaluation on the eval pool. Returns a dict of mean metrics.
    """
    model_path  = os.path.join(ablation_base_dir, config_name,
                               cfg["paths"]["model_name"])
    reward_weights = config_info["weights"]

    if not os.path.exists(model_path):
        print(f"  [{config_name}] No model at {model_path} — skipping eval")
        return {}

    # Build a temporary env to get obs_dim
    tmp_env    = make_envs(eval_eps[:1], reward_weights, cfg,
                           n=1, randomize=False)[0]
    obs_dim    = tmp_env.observation_space.shape[0]
    action_dim = tmp_env.action_space.n

    policy = build_policy(cfg, obs_dim, action_dim)
    policy.load_state_dict(torch.load(model_path, map_location=device))
    policy.to(device)
    policy.eval()

    # Eval environments with this config's reward weights
    n_eval_envs = min(len(eval_eps), cfg["ppo"]["eval_episodes"])
    eval_envs   = make_envs(eval_eps, reward_weights, cfg,
                            n=n_eval_envs, randomize=False)

    trainer = PPOTrainer(
        envs        = eval_envs,   # not used for training here
        eval_envs   = eval_envs,
        policy      = policy,
        cfg         = cfg,
        device      = device,
        results_dir = os.path.join(ablation_base_dir, config_name),
    )

    n_eval = min(len(eval_eps), cfg["ppo"]["eval_episodes"])
    stats  = trainer.evaluate(n_episodes=n_eval)

    print(f"  [{config_name}]  "
          f"reward={stats['mean_reward']:.3f}  "
          f"cov={stats.get('mean_coverage', float('nan')):.3f}  "
          f"sel={stats['mean_selected']:.1f}"
          f"({stats['mean_selection_rate']*100:.1f}%)  "
          f"cost={stats['mean_collection_cost']:.4f}")

    return stats


# ══════════════════════════════════════════════════════════════════
# Print comparison table
# ══════════════════════════════════════════════════════════════════

def print_ablation_table(all_results: dict):
    """Print a comparison table of all ablation configurations."""
    col_w = max(12, max(len(v["label"]) for v in ABLATION_CONFIGS.values()) + 2)

    metrics = [
        ("mean_reward",          "Reward",    ".3f"),
        ("mean_coverage",        "Coverage",  ".4f"),
        ("mean_selected",        "Selected",  ".1f"),
        ("mean_selection_rate",  "Sel.Rate",  ".3f"),
        ("mean_collection_cost", "CollCost",  ".5f"),
        ("mean_imp_score",       "Imp.Score", ".4f"),
    ]

    width = 28 + 12 * len(metrics)
    print(f"\n{'═'*width}")
    print(f"  Ablation Study — Trained PPO Agents")
    print(f"{'═'*width}")

    header = f"{'Configuration':<28}"
    for _, label, _ in metrics:
        header += f"{label:>12}"
    print(header)
    print("─" * width)

    for config_name, config_info in ABLATION_CONFIGS.items():
        stats = all_results.get(config_name, {})
        if not stats:
            continue
        row = f"{config_info['label']:<28}"
        for key, _, fmt in metrics:
            v = stats.get(key, float("nan"))
            if isinstance(v, float) and np.isnan(v):
                row += f"{'—':>12}"
            else:
                row += f"{format(v, fmt):>12}"
        print(row)

    print("═" * width)


# ══════════════════════════════════════════════════════════════════
# Save results CSV
# ══════════════════════════════════════════════════════════════════

def save_ablation_csv(all_results: dict, config_infos: dict, path: str):
    rows = []
    for config_name, stats in all_results.items():
        if not stats:
            continue
        rows.append({
            "config":  config_name,
            "label":   config_infos[config_name]["label"],
            **{f"alpha":   config_infos[config_name]["weights"]["alpha"],
               f"beta":    config_infos[config_name]["weights"]["beta"],
               f"lambda_": config_infos[config_name]["weights"]["lambda_"],
               f"delta":   config_infos[config_name]["weights"]["delta"],
               f"eta":     config_infos[config_name]["weights"]["eta"]},
            **stats,
        })
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[ablation] Results saved → {path}")


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.timesteps is not None:
        cfg["ppo"]["total_timesteps"] = args.timesteps

    # Device
    if args.no_cuda:
        device = torch.device("cpu")
    elif args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    policy_type = (
        args.policy_type
        or cfg["policy"].get("policy_type", "flat")
    )
    cfg["policy"]["policy_type"] = policy_type
    print(f"[ablation] Policy type : {policy_type}")

    # Which configurations to run
    configs_to_run = args.configs or list(ABLATION_CONFIGS.keys())
    configs_to_run = {k: ABLATION_CONFIGS[k] for k in configs_to_run}

    # Ablation output dir
    base_results      = cfg["paths"]["results_dir"]
    ablation_base_dir = os.path.join(base_results, "ablation")
    os.makedirs(ablation_base_dir, exist_ok=True)

    ecfg = cfg["environment"]
    dcfg = cfg["data"]
    N, M = ecfg["n_nodes"], ecfg["n_metrics"]

    # ── Load / generate data ──────────────────────────────────────
    pool_path   = pool_dir(cfg["paths"]["data_dir"], N, M, data_seed,
                           config_key=pool_config_key(cfg))
    train_path = os.path.join(pool_path, "train_episodes.pkl")
    eval_path  = os.path.join(pool_path, "eval_episodes.pkl")

    if os.path.exists(train_path) and os.path.exists(eval_path):
        print("[ablation] Loading existing episode pools...")
        train_eps = load_dataset(train_path)
        eval_eps  = load_dataset(eval_path)
    else:
        print(f"[ablation] Generating episode pools for N={N}, M={M}...")
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
        os.makedirs(pool_path, exist_ok=True)
        save_dataset(train_eps, train_path)
        save_dataset(eval_eps,  eval_path)

    print(f"\n[ablation] Device          : {device}")
    print(f"[ablation] N={N}  M={M}  T={ecfg['episode_length']}  "
          f"W={ecfg.get('window_size',1)}")
    print(f"[ablation] Total timesteps : {cfg['ppo']['total_timesteps']:,} per run")
    print(f"[ablation] Configurations  : {len(configs_to_run)}")
    print(f"[ablation] Output dir      : {ablation_base_dir}")
    print(f"[ablation] Configurations to run:")
    for k, v in configs_to_run.items():
        w = v["weights"]
        print(f"  {k:<22} α={w['alpha']}  β={w['beta']}  "
              f"λ={w['lambda_']}  δ={w['delta']}  η={w['eta']}")

    t_total = cfg["ppo"]["total_timesteps"] * len(configs_to_run)
    print(f"\n[ablation] Total env steps across all runs: {t_total:,}")
    print(f"[ablation] Runs will execute SEQUENTIALLY\n")

    # ── Training phase ────────────────────────────────────────────
    if not args.eval_only:
        t_start_all = __import__("time").time()
        for i, (config_name, config_info) in enumerate(configs_to_run.items()):
            print(f"\n[ablation] ── Run {i+1}/{len(configs_to_run)} ──────────────────────────")
            train_one(
                config_name     = config_name,
                config_info     = config_info,
                cfg             = cfg,
                train_eps       = train_eps,
                eval_eps        = eval_eps,
                device          = device,
                ablation_base_dir = ablation_base_dir,
            )
        elapsed = __import__("time").time() - t_start_all
        print(f"\n[ablation] All training runs complete — "
              f"total time: {elapsed/3600:.1f}h")
    else:
        print("[ablation] --eval-only: skipping training")

    # ── Evaluation phase ──────────────────────────────────────────
    print(f"\n[ablation] ── Evaluating all configurations ─────────────────")
    all_results = {}
    for config_name, config_info in configs_to_run.items():
        all_results[config_name] = evaluate_one(
            config_name       = config_name,
            config_info       = config_info,
            cfg               = cfg,
            eval_eps          = eval_eps,
            device            = device,
            ablation_base_dir = ablation_base_dir,
        )

    # ── Print & save ──────────────────────────────────────────────
    print_ablation_table(all_results)

    csv_path = os.path.join(ablation_base_dir, "ablation_results.csv")
    save_ablation_csv(all_results, configs_to_run, csv_path)

    print(f"\n[ablation] Done.")
    print(f"[ablation] Individual run logs: {ablation_base_dir}/<config>/training_log.csv")
    print(f"[ablation] Aggregate results  : {csv_path}")
    print(f"\n[ablation] TensorBoard (all runs):")
    print(f"  tensorboard --logdir {ablation_base_dir}")


if __name__ == "__main__":
    main()
elif __name__ == "scripts.train_ablation":
    main()