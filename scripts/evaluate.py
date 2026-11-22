"""
evaluate.py
===========
Evaluates a trained PPO model against all baselines.

Data is read from:
    data/<N>nodes-<M>metrics/eval_episodes.pkl

matching the layout written by train.py.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --model results/my_run/best_model.pt --episodes 20
    python scripts/evaluate.py --run-name my_run
"""

import os
import sys
import argparse
import yaml
import csv
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.synthetic_data import load_dataset, build_episode_pool, save_dataset, topology_kwargs, pool_dir, pool_config_key
from utils.seeding import resolve_seeds
from envs.monitoring_env import MonitoringEnv
from agents.policy import ActorCritic
from baselines.baselines import evaluate_baselines


# ------------------------------------------------------------------
# Helper — canonical data subdirectory for a given (N, M)
# ------------------------------------------------------------------



# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   default="configs/config.yaml")
    parser.add_argument("--model",    default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--device",   default=None)
    return parser.parse_args()


# ------------------------------------------------------------------
# RL agent evaluation — uses env directly for correct obs
# ------------------------------------------------------------------

def evaluate_rl_agent(policy, eval_eps, reward_weights, cfg, device, n_episodes):
    ecfg = cfg["environment"]
    env  = MonitoringEnv(
        episode_pool         = eval_eps,
        reward_weights       = reward_weights,
        per_metric_cost      = ecfg["per_metric_cost"],
        anomaly_baseline_pct = ecfg["anomaly_baseline_pct"],
        causal_baseline_pct  = ecfg["causal_baseline_pct"],
        randomize_episodes   = False,
        window_size          = ecfg.get("window_size", 1),
        window_agg           = ecfg.get("window_agg", "max"),
    )

    policy.eval()
    all_rewards, all_selected, all_coverage, all_imp = [], [], [], []

    for _ in range(min(n_episodes, len(eval_eps))):
        obs, _ = env.reset()
        done   = False
        ep_reward, ep_sel, ep_cov, ep_imp = 0.0, [], [], []

        with torch.no_grad():
            while not done:
                obs_t  = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                action = policy.get_deterministic_action(obs_t).squeeze(0)
                obs, rew, term, trunc, info = env.step(action.cpu().numpy())
                done = term or trunc
                ep_reward += rew
                ep_sel.append(info["selected_count"])
                ep_imp.append(info.get("r_imp", 0.0))
                if "anomaly_coverage" in info:
                    ep_cov.append(info["anomaly_coverage"])

        all_rewards.append(ep_reward)
        all_selected.append(float(np.mean(ep_sel)))
        all_imp.append(float(np.mean(ep_imp)))
        if ep_cov:
            all_coverage.append(float(np.mean(ep_cov)))

    result = {
        "total_reward":   float(np.mean(all_rewards)),
        "mean_selected":  float(np.mean(all_selected)),
        "mean_imp_score": float(np.mean(all_imp)),
        "mean_cost":      float(np.mean(all_selected)) * ecfg["per_metric_cost"],
    }
    if all_coverage:
        result["anomaly_coverage"] = float(np.mean(all_coverage))
    return result


# ------------------------------------------------------------------
# Printers / savers
# ------------------------------------------------------------------

def print_table(title, results):
    print(f"\n{'='*72}")
    print(f"  {title}")
    print(f"{'='*72}")
    print(f"{'Method':<25} {'Selected':>10} {'Cost':>8} {'Imp.Score':>12} {'Cov.':>8}")
    print("-" * 72)
    for name, stats in results.items():
        cov = stats.get("anomaly_coverage", float("nan"))
        print(f"{name:<25} {stats['mean_selected']:>10.1f} {stats['mean_cost']:>8.4f} "
              f"{stats['mean_imp_score']:>12.4f} {cov:>8.3f}")
    print("=" * 72)


def save_results(results, path):
    rows = [{"method": name, **stats} for name, stats in results.items()]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[eval] Results saved → {path}")


def plot_results(results, log_path, plots_dir):
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:
        print("[eval] matplotlib/pandas not available — skipping plots")
        return

    os.makedirs(plots_dir, exist_ok=True)
    methods  = list(results.keys())
    selected = [results[m]["mean_selected"]           for m in methods]
    imp      = [results[m]["mean_imp_score"]          for m in methods]
    coverage = [results[m].get("anomaly_coverage", 0) for m in methods]

    x, width = np.arange(len(methods)), 0.25
    fig, ax1 = plt.subplots(figsize=(max(8, len(methods) * 2), 5))
    ax2 = ax1.twinx()
    ax1.bar(x - width, selected, width, label="Mean Selected",    color="#4C72B0", alpha=0.85)
    ax1.bar(x,         imp,      width, label="Importance Score", color="#55A868", alpha=0.85)
    ax2.bar(x + width, coverage, width, label="Anomaly Coverage", color="#C44E52", alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=15, ha="right")
    ax1.set_ylabel("Count / Importance Score")
    ax2.set_ylabel("Anomaly Coverage")
    ax1.set_title("Baseline Comparison")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "baseline_comparison.png"), dpi=150)
    plt.close()

    if os.path.exists(log_path):
        try:
            df = pd.read_csv(log_path)
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            axes[0].plot(df["step"], df["eval_reward"],   color="#4C72B0", linewidth=2)
            axes[0].set_title("Reward"); axes[0].grid(True, alpha=0.3)
            axes[1].plot(df["step"], df["eval_selected"], color="#55A868", linewidth=2)
            axes[1].set_title("Selected metrics"); axes[1].grid(True, alpha=0.3)
            if "eval_coverage" in df.columns:
                axes[2].plot(df["step"], df["eval_coverage"], color="#C44E52", linewidth=2)
                axes[2].set_title("Coverage"); axes[2].grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "training_curves.png"), dpi=150)
            plt.close()
        except Exception as e:
            print(f"[eval] Could not plot training curves: {e}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(
        args.device if args.device else
        ("cuda" if torch.cuda.is_available() else "cpu")
    )

    base_results = cfg["paths"]["results_dir"]
    results_dir  = os.path.join(base_results, args.run_name) if args.run_name else base_results
    os.makedirs(results_dir, exist_ok=True)

    # ── Data ─────────────────────────────────────────────────────
    ecfg = cfg["environment"]
    dcfg = cfg["data"]
    N, M = ecfg["n_nodes"], ecfg["n_metrics"]

    pool_path  = pool_dir(cfg["paths"]["data_dir"], N, M, resolve_seeds(cfg)[0],
                          config_key=pool_config_key(cfg))
    eval_path = os.path.join(pool_path, "eval_episodes.pkl")

    if not os.path.exists(eval_path):
        print(f"[eval] Eval dataset not found at {eval_path} — generating...")
        eval_eps = build_episode_pool(
            n_episodes           = args.episodes,
            n_nodes              = N,
            n_metrics            = M,
            episode_length       = ecfg["episode_length"],
            anomaly_density      = dcfg["anomaly_density"] * 1.2,
            causal_density       = dcfg["causal_density"],
            base_seed            = dcfg["seed"] + 10000,
            anomaly_baseline_pct = ecfg["anomaly_baseline_pct"],
            causal_baseline_pct  = ecfg["causal_baseline_pct"],
            **topology_kwargs(ecfg),
        )
        save_dataset(eval_eps, eval_path)
        print(f"[eval] Saved → {eval_path}")
    else:
        eval_eps = load_dataset(eval_path)

    n_eval   = min(args.episodes, len(eval_eps))
    eval_eps = eval_eps[:n_eval]
    print(f"[eval] Data dir : {pool_path}")
    print(f"[eval] Episodes : {n_eval}")

    rw = cfg["reward"]
    reward_weights = {
        "alpha":   rw["alpha"], "beta":    rw["beta"],
        "lambda_": rw["lambda_"], "delta": rw["delta"], "eta": rw["eta"],
    }

    # ── Baselines ────────────────────────────────────────────────
    print("\n[eval] Running baselines...")
    baseline_results = evaluate_baselines(eval_eps, reward_weights, ecfg["per_metric_cost"],
                                      seed=resolve_seeds(cfg)[1])

    # ── RL Agent ─────────────────────────────────────────────────
    model_path = args.model or os.path.join(results_dir, cfg["paths"]["model_name"])
    all_results = dict(baseline_results)

    if os.path.exists(model_path):
        print(f"[eval] Loading model from {model_path}...")
        _tmp_env   = MonitoringEnv(
            episode_pool=eval_eps[:1], reward_weights=reward_weights,
            per_metric_cost=ecfg["per_metric_cost"],
            anomaly_baseline_pct=ecfg["anomaly_baseline_pct"],
            causal_baseline_pct=ecfg["causal_baseline_pct"],
            window_size=ecfg.get("window_size", 1),
            window_agg=ecfg.get("window_agg", "max"),
        )
        obs_dim    = _tmp_env.observation_space.shape[0]
        action_dim = _tmp_env.action_space.n
        pcfg       = cfg["policy"]
        policy     = ActorCritic(
            obs_dim=obs_dim, action_dim=action_dim,
            hidden_sizes=pcfg["hidden_sizes"], use_layer_norm=pcfg["use_layer_norm"],
        )
        policy.load_state_dict(torch.load(model_path, map_location=device))
        policy.to(device)
        print(f"[eval] obs_dim={obs_dim}  action_dim={action_dim}")

        rl_stats = evaluate_rl_agent(policy, eval_eps, reward_weights, cfg, device, n_eval)
        all_results["RL-PPO (Ours)"] = rl_stats
    else:
        print(f"[eval] No model at {model_path} — baselines only.")

    print_table("Evaluation Results", all_results)
    save_results(all_results, os.path.join(results_dir, "evaluation_results.csv"))
    plot_results(all_results,
                 os.path.join(results_dir, cfg["paths"]["log_name"]),
                 os.path.join(results_dir, "plots"))


if __name__ == "__main__":
    main()