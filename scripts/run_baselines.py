"""
run_baselines.py
================
Runs all baseline methods and the reward-weight ablation study.

FIX 1 (previous): baselines use per-node tiered costs from episode dict,
not a flat per_metric_cost scalar.

FIX 2 (this version): baselines now operate at WINDOW granularity, not
per-timestep. Previously run_episode_with_node_costs() called select(t)
at every single timestep t=0..T-1, giving baselines W times more decisions
than the RL agent (which decides once per window). This inflated baseline
coverage unfairly — TopK-Anomaly-Causal showed ~0.92 instead of ~0.88.

The correct approach mirrors exactly what the RL environment does:
  - One selection per window (W timesteps)
  - Scores aggregated over the window (max, same as env window_agg=max)
  - Coverage evaluated against window-max labels (any anomaly in window)
  - Reward computed as mean of per-timestep values held constant over window

This makes baselines and the RL agent operate at the same temporal
granularity, producing a fair comparison.

Usage:
    python scripts/run_baselines.py
    python scripts/run_baselines.py --config configs/config.yaml --episodes 30
"""

import os
import sys
import argparse
import yaml
import numpy as np
import csv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.synthetic_data import build_episode_pool, load_dataset, save_dataset, topology_kwargs, pool_dir, pool_config_key
from utils.seeding import resolve_seeds
from baselines.baselines import MonitorEverything, RandomFeasible, TopKAnomalyCausal


# ------------------------------------------------------------------
# Helper — canonical data subdirectory
# ------------------------------------------------------------------



# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   default="configs/config.yaml")
    parser.add_argument("--episodes", type=int, default=20)
    return parser.parse_args()


# ------------------------------------------------------------------
# Window-based baseline evaluation
# ------------------------------------------------------------------

def run_episode_window_based(
    baseline,
    ep: dict,
    reward_weights: dict,
    window_size: int,
    window_agg: str = "max",
) -> dict:
    """
    Run one baseline episode at WINDOW granularity, matching the RL env.

    The baseline makes ONE decision per window using aggregated scores,
    and that decision is held constant for all W timesteps in the window.
    Coverage is evaluated against window-level labels (any anomaly in window).

    Parameters
    ----------
    baseline    : BaseBaseline instance
    ep          : episode dict
    reward_weights : alpha, beta, lambda_, delta, eta
    window_size : W — timesteps per decision window
    window_agg  : how to aggregate scores over window ("max" or "mean")
    """
    N  = ep["n_nodes"]
    M  = ep["n_metrics"]
    T  = ep["T"]
    A  = ep["anomaly_scores"]    # [T, N, M]
    C  = ep["causal_scores"]     # [N, M]
    L  = ep["anomaly_labels"]    # [T, N, M]
    caps       = ep["node_capacities"]
    node_costs = ep.get("node_costs", np.full(N, 0.01, dtype=np.float32))

    b_a = ep.get("b_a", 0.0)
    b_c = ep.get("b_c", 0.0)

    alpha   = reward_weights.get("alpha",   5.0)
    beta    = reward_weights.get("beta",    3.0)
    lambda_ = reward_weights.get("lambda_", 0.1)
    delta   = reward_weights.get("delta",   0.1)
    eta     = reward_weights.get("eta",     2.0)

    n_windows = T // window_size

    # Pre-compute window-aggregated scores [n_windows, N, M]
    agg_scores = np.zeros((n_windows, N, M), dtype=np.float32)
    for w in range(n_windows):
        t_s = w * window_size
        t_e = min(t_s + window_size, T)
        if window_agg == "max":
            agg_scores[w] = A[t_s:t_e].max(axis=0)
        else:
            agg_scores[w] = A[t_s:t_e].mean(axis=0)

    # Build proxy episode so baseline.select(w) uses aggregated scores
    proxy_ep = {
        "n_nodes":         N,
        "n_metrics":       M,
        "T":               n_windows,
        "anomaly_scores":  agg_scores,
        "anomaly_labels":  np.zeros((n_windows, N, M), dtype=np.int32),
        "causal_scores":   C,
        "node_capacities": caps,
        "b_a": b_a,
        "b_c": b_c,
    }
    baseline.reset(proxy_ep)

    total_reward    = 0.0
    total_selected  = 0
    total_imp       = 0.0
    total_cost      = 0.0
    window_coverages = []
    prev_action     = np.zeros((N, M), dtype=np.float32)

    for w in range(n_windows):
        t_s = w * window_size
        t_e = min(t_s + window_size, T)
        W_actual = t_e - t_s

        # One decision per window using aggregated scores
        a   = baseline.select(w)   # [N, M]
        A_w = agg_scores[w]        # [N, M]

        # Reward terms
        imp_a = np.maximum(0.0, A_w - b_a)
        imp_c = np.maximum(0.0, C  - b_c)
        r_imp = float((a * (imp_a + imp_c)).sum())

        selected_per_node = a.sum(axis=1)
        overflow = np.maximum(0.0, selected_per_node - caps)
        r_cap    = float(overflow.sum())
        r_cost   = float((node_costs * selected_per_node).sum())

        is_unimp = ((A_w < b_a) & (C < b_c)).astype(np.float32)
        r_switch = float((np.abs(a - prev_action) * is_unimp).sum())
        r_idle   = 1.0 if a.sum() == 0 else 0.0

        reward   = (alpha * r_imp - beta * r_cap - lambda_ * r_cost
                    - delta * r_switch - eta * r_idle)
        reward  /= W_actual   # normalise by window length (same as env)

        total_reward   += reward
        total_selected += int(a.sum())
        total_imp      += r_imp
        total_cost     += r_cost

        # Coverage — window-level (any anomaly in window = must cover)
        L_w   = L[t_s:t_e].max(axis=0)   # [N, M]
        n_anom = L_w.sum()
        window_coverages.append(
            float((a * L_w).sum() / n_anom) if n_anom > 0 else 1.0
        )

        prev_action = a.copy()

    return {
        "total_reward":     total_reward,
        "mean_selected":    total_selected / n_windows,
        "mean_imp_score":   total_imp      / n_windows,
        "mean_cost":        total_cost     / n_windows,
        "anomaly_coverage": float(np.mean(window_coverages)),
    }


# ------------------------------------------------------------------
# Evaluate all baselines
# ------------------------------------------------------------------

def evaluate_all_baselines(eval_eps, reward_weights, window_size, window_agg,
                           baseline_seed=0):
    # baseline_seed is the TRAIN seed: Random-Feasible's randomness is
    # algorithmic, the counterpart of the RL agent's init/sampling noise, so
    # it must move with the training seed rather than being pinned.
    baselines = {
        "Monitor-Everything":  MonitorEverything(),
        "Random-Feasible":     RandomFeasible(seed=baseline_seed),
        "TopK-Anomaly-Causal": TopKAnomalyCausal(),
    }
    results = {name: [] for name in baselines}
    for ep in eval_eps:
        for name, bl in baselines.items():
            stats = run_episode_window_based(
                bl, ep, reward_weights, window_size, window_agg
            )
            results[name].append(stats)

    return {
        name: {k: float(np.mean([s[k] for s in ep_stats]))
               for k in ep_stats[0]}
        for name, ep_stats in results.items()
    }


# ------------------------------------------------------------------
# Ablation study
# ------------------------------------------------------------------

ABLATION_CONFIGS = [
    (10.0, 0.0, 0.0, 0.0, 0.0, "alpha only"),
    (10.0, 3.0, 0.0, 0.0, 0.0, "alpha + beta"),
    (10.0, 3.0, 0.1, 0.0, 0.0, "alpha + beta + lambda"),
    (10.0, 3.0, 0.1, 0.1, 2.0, "Full reward (paper)"),
    (0.0,  3.0, 0.1, 0.0, 0.0, "no importance (alpha=0)"),
]


def run_ablation(eval_eps, window_size, window_agg):
    bl = TopKAnomalyCausal()
    results = {}
    for alpha, beta, lambda_, delta, eta, label in ABLATION_CONFIGS:
        rw = {"alpha": alpha, "beta": beta, "lambda_": lambda_,
              "delta": delta, "eta": eta}
        ep_stats = [
            run_episode_window_based(bl, ep, rw, window_size, window_agg)
            for ep in eval_eps
        ]
        results[label] = {
            k: float(np.mean([s[k] for s in ep_stats]))
            for k in ep_stats[0]
        }
    return results


# ------------------------------------------------------------------
# Printers
# ------------------------------------------------------------------

def print_table(title, results):
    print(f"\n{'='*82}")
    print(f"  {title}")
    print(f"{'='*82}")
    print(f"{'Method':<28} {'Selected':>10} {'Cost':>10} "
          f"{'Imp.Score':>12} {'Coverage':>10}")
    print("-" * 82)
    for name, stats in results.items():
        cov = stats.get("anomaly_coverage", float("nan"))
        print(f"{name:<28} {stats['mean_selected']:>10.2f} "
              f"{stats['mean_cost']:>10.5f} "
              f"{stats['mean_imp_score']:>12.4f} {cov:>10.3f}")
    print("=" * 82)


def print_ablation(results):
    print(f"\n{'='*82}")
    print("  Ablation Study — Reward Weights (TopK-Anomaly-Causal proxy)")
    print(f"{'='*82}")
    print(f"{'Config':<30} {'Selected':>10} {'Imp.Score':>12} "
          f"{'Cost':>10} {'Coverage':>10}")
    print("-" * 82)
    for label, stats in results.items():
        cov = stats.get("anomaly_coverage", float("nan"))
        print(f"{label:<30} {stats['mean_selected']:>10.2f} "
              f"{stats['mean_imp_score']:>12.4f} "
              f"{stats['mean_cost']:>10.5f} {cov:>10.3f}")
    print("=" * 82)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    ecfg = cfg["environment"]
    dcfg = cfg["data"]
    N, M = ecfg["n_nodes"], ecfg["n_metrics"]
    W    = ecfg.get("window_size", 1)
    agg  = ecfg.get("window_agg", "max")

    results_dir = cfg["paths"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    # ── Data ─────────────────────────────────────────────────────
    pool_path  = pool_dir(cfg["paths"]["data_dir"], N, M, resolve_seeds(cfg)[0],
                          config_key=pool_config_key(cfg))
    eval_path = os.path.join(pool_path, "eval_episodes.pkl")

    if os.path.exists(eval_path):
        eval_eps = load_dataset(eval_path)
    else:
        print(f"[baselines] Eval dataset not found — generating...")
        eval_eps = build_episode_pool(
            n_episodes           = args.episodes,
            n_nodes              = N, n_metrics = M,
            episode_length       = ecfg["episode_length"],
            anomaly_density      = dcfg["anomaly_density"] * 1.2,
            causal_density       = dcfg["causal_density"],
            base_seed            = dcfg["seed"] + 10000,
            anomaly_baseline_pct = ecfg["anomaly_baseline_pct"],
            causal_baseline_pct  = ecfg["causal_baseline_pct"],
            **topology_kwargs(ecfg),
        )
        save_dataset(eval_eps, eval_path)
        print(f"[baselines] Saved → {eval_path}")

    eval_eps = eval_eps[:args.episodes]
    print(f"[baselines] Data dir  : {pool_path}")
    print(f"[baselines] Episodes  : {len(eval_eps)}")
    print(f"[baselines] Window W  : {W}  agg={agg}")
    print(f"[baselines] Note: baselines make 1 decision per {W} timesteps "
          f"— same granularity as RL agent")

    rw = cfg["reward"]
    reward_weights = {
        "alpha": rw["alpha"], "beta": rw["beta"],
        "lambda_": rw["lambda_"], "delta": rw["delta"], "eta": rw["eta"],
    }

    # ── Baseline comparison ───────────────────────────────────────
    print("\n[baselines] Running all baselines...")
    results = evaluate_all_baselines(eval_eps, reward_weights, W, agg,
                                     baseline_seed=resolve_seeds(cfg)[1])
    print_table(
        f"Baseline Comparison — window-based (W={W}, agg={agg})", results
    )

    # ── Ablation ─────────────────────────────────────────────────
    print("\n[ablation] Running reward weight ablation...")
    ablation_results = run_ablation(eval_eps, W, agg)
    print_ablation(ablation_results)

    # ── Save ─────────────────────────────────────────────────────
    rows = [{"method": k, **v} for k, v in results.items()]
    path = os.path.join(results_dir, "baseline_results.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    print(f"\n[baselines] Saved → {path}")

    abl_rows = [{"config": k, **v} for k, v in ablation_results.items()]
    path = os.path.join(results_dir, "ablation_results.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(abl_rows[0].keys()))
        writer.writeheader(); writer.writerows(abl_rows)
    print(f"[baselines] Saved → {path}")


if __name__ == "__main__":
    main()