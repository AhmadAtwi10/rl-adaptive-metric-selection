"""
diagnose_reward.py
==================
Runs the trained agent for one episode and prints every reward term
at each decision window, plus a coverage ceiling analysis showing
where coverage is being lost.

FIX: now uses MonitoringEnv.step() directly instead of rebuilding the
observation manually, so the policy always receives the correct 5-feature
observation it was trained on.

Usage:
    PYTHONPATH=. python scripts/diagnose_reward.py
    PYTHONPATH=. python scripts/diagnose_reward.py --model results/my_run/best_model.pt
    PYTHONPATH=. python scripts/diagnose_reward.py --model results/my_run/best_model.pt --windows 20
"""

import os
import sys
import argparse
import numpy as np
import torch
import yaml
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.synthetic_data import load_dataset
from envs.monitoring_env import MonitoringEnv
from agents.policy import ActorCritic


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",  default="configs/config.yaml")
    p.add_argument("--model",   default="results/best_model.pt")
    p.add_argument("--data",    default="data/eval_episodes.pkl")
    p.add_argument("--windows", type=int, default=None,
                   help="How many windows to print. Default: all")
    p.add_argument("--ep",      type=int, default=0,
                   help="Which episode index to use")
    return p.parse_args()


# ------------------------------------------------------------------
# Coverage ceiling analysis — shows the theoretical max coverage
# given capacity and window constraints, before any policy limitation
# ------------------------------------------------------------------

def compute_coverage_ceilings(ep: dict, window_size: int) -> dict:
    """
    Computes three coverage ceilings for a single episode:

    1. Oracle ceiling    — knows exactly which (n,m,t) are anomalous,
                           picks the best cap_n per node per timestep.
                           Hard limit from capacity alone.

    2. Window ceiling    — knows anomalies within each window,
                           selects best cap_n per node for the whole
                           window (held constant like the RL agent).
                           Additional loss from window-constant selection.

    3. TopK ceiling      — selects top-cap_n by anomaly score per node
                           per window (what a perfect policy should reach).
    """
    A      = ep["anomaly_scores"]    # [T, N, M]
    L      = ep["anomaly_labels"]    # [T, N, M]
    caps   = ep["node_capacities"]   # [N]
    T, N, M = A.shape
    n_windows = T // window_size

    # 1. Per-timestep oracle
    oracle_covs = []
    for t in range(T):
        for n in range(N):
            n_anom = int(L[t, n].sum())
            if n_anom == 0:
                oracle_covs.append(1.0)
                continue
            cap_n   = int(caps[n])
            covered = min(n_anom, cap_n)
            oracle_covs.append(covered / n_anom)

    # 2. Window-level oracle (knows labels inside window, holds selection constant)
    window_oracle_covs = []
    for w in range(n_windows):
        t_s, t_e = w * window_size, min((w + 1) * window_size, T)
        a = np.zeros((N, M), dtype=np.float32)
        for n in range(N):
            cap_n  = int(caps[n])
            anom_n = L[t_s:t_e, n, :].max(axis=0)   # any anomaly in window [M]
            idx    = np.where(anom_n > 0)[0]
            if len(idx) <= cap_n:
                a[n, idx] = 1.0
            else:
                scores = A[t_s:t_e, n, idx].max(axis=0)
                chosen = idx[np.argsort(scores)[::-1][:cap_n]]
                a[n, chosen] = 1.0
        for t in range(t_s, t_e):
            for n in range(N):
                n_anom = int(L[t, n].sum())
                if n_anom == 0:
                    window_oracle_covs.append(1.0)
                else:
                    covered = int((a[n] * L[t, n]).sum())
                    window_oracle_covs.append(covered / n_anom)

    # 3. Window-level TopK by score (what RL should learn)
    window_topk_covs = []
    for w in range(n_windows):
        t_s, t_e = w * window_size, min((w + 1) * window_size, T)
        window_max = A[t_s:t_e].max(axis=0)   # [N, M]
        a = np.zeros((N, M), dtype=np.float32)
        for n in range(N):
            cap_n = int(caps[n])
            topk  = np.argsort(window_max[n])[::-1][:cap_n]
            a[n, topk] = 1.0
        for t in range(t_s, t_e):
            for n in range(N):
                n_anom = int(L[t, n].sum())
                if n_anom == 0:
                    window_topk_covs.append(1.0)
                else:
                    covered = int((a[n] * L[t, n]).sum())
                    window_topk_covs.append(covered / n_anom)

    return {
        "oracle":       float(np.mean(oracle_covs)),
        "window_oracle":float(np.mean(window_oracle_covs)),
        "window_topk":  float(np.mean(window_topk_covs)),
    }


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load data ────────────────────────────────────────────────
    eps = load_dataset(args.data)
    ep  = eps[args.ep]

    rw = cfg["reward"]
    reward_weights = {
        "alpha":   rw["alpha"],
        "beta":    rw["beta"],
        "lambda_": rw["lambda_"],
        "delta":   rw["delta"],
        "eta":     rw["eta"],
    }

    ecfg = cfg["environment"]

    # ── Build environment (obs is constructed here, not manually) ──
    env = MonitoringEnv(
        episode_pool         = eps,
        reward_weights       = reward_weights,
        per_metric_cost      = ecfg["per_metric_cost"],
        anomaly_baseline_pct = ecfg["anomaly_baseline_pct"],
        causal_baseline_pct  = ecfg["causal_baseline_pct"],
        randomize_episodes   = False,
        window_size          = ecfg.get("window_size", 1),
        window_agg           = ecfg.get("window_agg", "max"),
    )

    # ── Load policy ──────────────────────────────────────────────
    # FIX: obs_dim queried from environment, not hardcoded.
    obs_dim    = env.observation_space.shape[0]   # N*M*5
    action_dim = env.action_space.n               # N*M

    N, M = ep["n_nodes"], ep["n_metrics"]
    pcfg = cfg["policy"]
    policy = ActorCritic(
        obs_dim        = obs_dim,
        action_dim     = action_dim,
        hidden_sizes   = pcfg["hidden_sizes"],
        use_layer_norm = pcfg["use_layer_norm"],
    )

    model_path = Path(args.model)
    if model_path.exists():
        policy.load_state_dict(torch.load(model_path, map_location=device))
        policy.to(device)
        policy.eval()
        agent_label = "Trained agent"
    else:
        print(f"[!] Model not found at {model_path} — using random policy")
        agent_label = "Random policy"

    # ── Coverage ceilings ────────────────────────────────────────
    W    = ecfg.get("window_size", 1)
    caps = ep["node_capacities"]
    total_cap = int(caps.sum())

    ceilings = compute_coverage_ceilings(ep, W)

    # ── Episode info ─────────────────────────────────────────────
    obs, _ = env.reset()

    print(f"\n{'═'*72}")
    print(f"  Reward Term Diagnostic  —  {agent_label}")
    print(f"{'═'*72}")
    print(f"  Episode         : {args.ep}")
    print(f"  Nodes (N)       : {N}")
    print(f"  Metrics/node (M): {M}")
    print(f"  Total metrics   : {N*M}")
    print(f"  Window size (W) : {W}")
    print(f"  Decisions/ep    : {env.n_windows}")
    print(f"  b_a (anom base) : {env._b_a:.4f}  (anomaly_baseline_pct={ecfg['anomaly_baseline_pct']})")
    print(f"  b_c (caus base) : {env._b_c:.4f}  (causal_baseline_pct={ecfg['causal_baseline_pct']})")

    print(f"\n  Node capacities:")
    for n in range(N):
        tier = "IoT  " if caps[n] <= 4 else ("Edge " if caps[n] <= 9 else "Cloud")
        print(f"    node_{n:02d} [{tier}]: cap={caps[n]:.0f}  ({100*caps[n]/M:.0f}% of {M} metrics)")
    print(f"  Total budget    : {total_cap} / {N*M}  ({100*total_cap/(N*M):.1f}%)")

    print(f"\n  Coverage ceilings (theoretical limits):")
    print(f"    Oracle (per-timestep, perfect knowledge) : {ceilings['oracle']:.4f}")
    print(f"    Window-oracle (W={W}, knows labels)      : {ceilings['window_oracle']:.4f}")
    print(f"    Window-TopK   (W={W}, ranks by score)    : {ceilings['window_topk']:.4f}  ← RL target")

    print(f"\n  Reward weights: "
          f"α={rw['alpha']} · r_imp  "
          f"−  β={rw['beta']} · r_cap  "
          f"−  λ={rw['lambda_']} · r_cost  "
          f"−  δ={rw['delta']} · r_switch  "
          f"−  η={rw['eta']} · r_idle")

    # ── Per-window rollout ────────────────────────────────────────
    print(f"\n{'─'*72}")
    print(f"  All values are per-step averages (window totals ÷ W={W})")
    print(f"{'─'*72}")
    print(f"  {'Win':>4} {'Sel':>4} {'α·imp':>8} {'β·cap':>7} "
          f"{'λ·cost':>7} {'δ·sw':>6} {'η·idle':>6} "
          f"{'reward':>8} {'cov':>6}")
    print(f"{'─'*72}")

    acc = {k: 0.0 for k in
           ["r_imp", "r_cap", "r_cost", "r_switch", "r_idle", "reward", "coverage", "sel"]}
    n_windows_run = 0
    n_print = args.windows or env.n_windows

    done = False
    while not done:
        with torch.no_grad():
            obs_t  = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            action = policy.get_deterministic_action(obs_t).squeeze(0).cpu().numpy()

        obs, reward, term, trunc, info = env.step(action)
        done = term or trunc

        w = info["window"] - 1   # 0-indexed
        w_imp    = rw["alpha"]   * info["r_imp"]
        w_cap    = rw["beta"]    * info["r_cap"]
        w_cost   = rw["lambda_"] * info["r_cost"]
        w_switch = rw["delta"]   * info["r_switch"]
        w_idle   = rw["eta"]     * info["r_idle"]
        cov      = info.get("anomaly_coverage", float("nan"))

        if w < n_print:
            print(f"  {w:>4d} "
                  f"{info['selected_count']:>4d} "
                  f"{w_imp:>8.3f} "
                  f"{w_cap:>7.3f} "
                  f"{w_cost:>7.3f} "
                  f"{w_switch:>6.3f} "
                  f"{w_idle:>6.2f} "
                  f"{reward:>8.3f} "
                  f"{cov:>6.3f}")

        acc["r_imp"]    += w_imp
        acc["r_cap"]    += w_cap
        acc["r_cost"]   += w_cost
        acc["r_switch"] += w_switch
        acc["r_idle"]   += w_idle
        acc["reward"]   += reward
        acc["coverage"] += cov if not np.isnan(cov) else 0.0
        acc["sel"]      += info["selected_count"]
        n_windows_run   += 1

    n = max(n_windows_run, 1)

    print(f"{'─'*72}")
    print(f"  {'MEAN':>4} "
          f"{acc['sel']/n:>4.1f} "
          f"{acc['r_imp']/n:>8.3f} "
          f"{acc['r_cap']/n:>7.3f} "
          f"{acc['r_cost']/n:>7.3f} "
          f"{acc['r_switch']/n:>6.3f} "
          f"{acc['r_idle']/n:>6.2f} "
          f"{acc['reward']/n:>8.3f} "
          f"{acc['coverage']/n:>6.3f}")
    print(f"{'═'*72}")

    agent_coverage = acc["coverage"] / n

    print(f"\n  Per-step mean reward : {acc['reward']/n:.3f}")
    print(f"  Episode total reward : {acc['reward']:.1f}")

    print(f"\n  Reward breakdown (mean per window):")
    total_abs = sum(abs(acc[k] / n) for k in ["r_imp", "r_cap", "r_cost", "r_switch", "r_idle"])
    if total_abs > 1e-8:
        for key, label in [
            ("r_imp",    "α·r_imp   "),
            ("r_cap",    "β·r_cap   "),
            ("r_cost",   "λ·r_cost  "),
            ("r_switch", "δ·r_switch"),
            ("r_idle",   "η·r_idle  "),
        ]:
            pct = 100 * abs(acc[key] / n) / total_abs
            print(f"    {label} = {acc[key]/n:>8.3f}  ({pct:>5.1f}%)")

    print(f"\n  Selection: {acc['sel']/n:.1f} metrics/window  "
          f"(budget={total_cap}, {100*acc['sel']/n/total_cap:.0f}%)")

    print(f"\n  Coverage summary:")
    print(f"    Agent coverage           : {agent_coverage:.4f}")
    print(f"    Window-TopK ceiling      : {ceilings['window_topk']:.4f}  (RL target)")
    print(f"    Window-oracle ceiling    : {ceilings['window_oracle']:.4f}")
    print(f"    Per-timestep oracle      : {ceilings['oracle']:.4f}")
    gap = ceilings["window_topk"] - agent_coverage
    print(f"    Gap to RL target         : {gap:.4f}  ({100*gap/ceilings['window_topk']:.1f}% of target)")

    # ── Diagnosis ────────────────────────────────────────────────
    print(f"\n  Diagnosis:")

    imp_frac = (acc["r_imp"] / n) / max(total_abs, 1e-8)
    cap_frac = abs(acc["r_cap"] / n) / max(total_abs, 1e-8)

    if imp_frac < 0.4:
        print(f"  ✗ Importance reward weak ({imp_frac*100:.1f}% of |terms|)")
        print(f"    → Lower anomaly_baseline_pct (currently {ecfg['anomaly_baseline_pct']})")
        print(f"    → Recommended: 30")
    else:
        print(f"  ✓ Importance reward dominates ({imp_frac*100:.1f}%)")

    if cap_frac > 0.3:
        print(f"  ✗ Capacity penalty large ({cap_frac*100:.1f}%) — agent over-selects")
        print(f"    → Consider increasing beta (currently {rw['beta']})")
    else:
        print(f"  ✓ Capacity penalty under control ({cap_frac*100:.1f}%)")

    sel_ratio = acc["sel"] / n / max(total_cap, 1)
    if sel_ratio > 1.1:
        print(f"  ✗ Agent selects {sel_ratio*100:.0f}% of budget — overflowing")
    elif sel_ratio < 0.5:
        print(f"  ✗ Agent uses only {sel_ratio*100:.0f}% of budget — under-selecting")
        print(f"    → Lower beta or increase alpha")
    else:
        print(f"  ✓ Selection near budget ({sel_ratio*100:.0f}%)")

    b_c_val = env._b_c
    if b_c_val >= 0.99:
        print(f"  ✗ b_c={b_c_val:.4f} ≈ 1.0 — causal reward is ZERO everywhere")
        print(f"    → Lower causal_baseline_pct (currently {ecfg['causal_baseline_pct']})")
        print(f"    → Recommended: 50")
    else:
        print(f"  ✓ Causal baseline b_c={b_c_val:.4f} is active")

    if agent_coverage < 0.7 * ceilings["window_topk"]:
        print(f"  ✗ Coverage {agent_coverage:.3f} is far below window-TopK ceiling {ceilings['window_topk']:.3f}")
        print(f"    → Check obs_dim matches between training and evaluation")
        print(f"    → Check b_a and b_c are correctly set")
    elif agent_coverage < 0.9 * ceilings["window_topk"]:
        print(f"  ~ Coverage {agent_coverage:.3f} is below window-TopK ceiling — room to improve")
    else:
        print(f"  ✓ Coverage {agent_coverage:.3f} is near the window-TopK ceiling")


if __name__ == "__main__":
    main()
elif __name__ == "scripts.diagnose_reward":
    main()