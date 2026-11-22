"""
visualize_decision_trace.py
===========================
Generates a qualitative multi-panel figure showing how the agent's
monitoring decisions evolve over one episode, correlating:
  Panel 1 — Anomaly score heatmap  (N×M × W windows)
  Panel 2 — Agent selections with decision quality overlay
             Green border = correctly selected (high anomaly, selected)
             Red border   = missed anomaly    (high anomaly, NOT selected)
             Grey fill    = wasted selection  (low anomaly, selected)
  Panel 3 — Collection cost per window (agent vs Monitor-Everything)
             Grey band = windows where cost is "wasted" (low anomaly, high cost)
  Panel 4 — Capacity utilization per node per window
             Red zone = capacity violated

All panels share the same x-axis (window index).

The figure makes three claims visible simultaneously:
  1. The agent selects metrics with high anomaly scores (correct)
  2. The agent avoids expensive selections when anomaly is low (cost-aware)
  3. The agent respects capacity constraints most of the time

Usage
-----
  # From a pkl episode pool — picks one representative episode
  python scripts/visualize_decision_trace.py \
      --pkl  data/10nodes-15metrics/eval_episodes.pkl \
      --model results/my_run/best_model.pt \
      --config configs/config.yaml

  # Pick a specific episode index
  python scripts/visualize_decision_trace.py \
      --pkl  data/10nodes-15metrics/eval_episodes.pkl \
      --model results/my_run/best_model.pt \
      --config configs/config.yaml \
      --episode-idx 3

  # Show only a subset of windows (cleaner figure)
  python scripts/visualize_decision_trace.py \
      --pkl data/10nodes-15metrics/eval_episodes.pkl \
      --model results/my_run/best_model.pt \
      --config configs/config.yaml \
      --window-range 0 30
"""

import os
import sys
import argparse
import pickle
import yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ══════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pkl",          required=True,
                   help="Path to eval_episodes.pkl")
    p.add_argument("--model",        required=True,
                   help="Path to best_model.pt")
    p.add_argument("--config",       default="configs/config.yaml")
    p.add_argument("--out-dir",      default=None)
    p.add_argument("--episode-idx",  type=int, default=0,
                   help="Which episode to visualise (default: 0).")
    p.add_argument("--window-range", nargs=2, type=int, default=None,
                   metavar=("START", "END"),
                   help="Show only windows [START, END). Default: all.")
    p.add_argument("--anomaly-thresh", type=float, default=None,
                   help="Threshold above which a metric is 'highly anomalous'. "
                        "Default: b_a from episode.")
    p.add_argument("--figsize",      nargs=2, type=float, default=[7.0, 5.5],
                   metavar=("W", "H"))
    p.add_argument("--dpi",          type=int, default=200)
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════
# Collect agent trajectory
# ══════════════════════════════════════════════════════════════════

def collect_trajectory(episode, model_path, cfg, device):
    """Run agent on episode, return list of [N,M] actions per window."""
    from envs.monitoring_env import MonitoringEnv

    ecfg = cfg["environment"]
    rw   = cfg["reward"]
    reward_weights = {
        "alpha": rw["alpha"], "beta": rw["beta"],
        "lambda_": rw["lambda_"], "delta": rw["delta"], "eta": rw["eta"],
    }

    env = MonitoringEnv(
        episode_pool         = [episode],
        reward_weights       = reward_weights,
        per_metric_cost      = ecfg["per_metric_cost"],
        anomaly_baseline_pct = ecfg["anomaly_baseline_pct"],
        causal_baseline_pct  = ecfg["causal_baseline_pct"],
        randomize_episodes   = False,
        window_size          = ecfg.get("window_size", 1),
        window_agg           = ecfg.get("window_agg", "max"),
        coverage_mode        = ecfg.get("coverage_mode", "window"),
    )

    # Build policy
    obs_dim    = env.observation_space.shape[0]
    action_dim = env.action_space.n
    pcfg       = cfg["policy"]

    from agents.policy import ActorCritic, PerMetricActorCritic

    if pcfg.get("policy_type", "flat") == "per_metric":
        N, M = ecfg["n_nodes"], ecfg["n_metrics"]
        policy = PerMetricActorCritic(
            n_nodes        = N,
            n_metrics      = M,
            feature_dim    = obs_dim // (N * M),
            metric_hidden  = pcfg.get("metric_hidden", 128),
            global_hidden  = pcfg.get("global_hidden", 256),
            use_layer_norm = pcfg.get("use_layer_norm", True),
            global_pool    = pcfg.get("global_pool", "mean"),
        )
    else:
        policy = ActorCritic(
            obs_dim        = obs_dim,
            action_dim     = action_dim,
            hidden_sizes   = pcfg.get("hidden_sizes", [512, 256]),
            use_layer_norm = pcfg.get("use_layer_norm", True),
        )

    policy.load_state_dict(torch.load(model_path, map_location=device))
    policy.to(device); policy.eval()

    obs, _ = env.reset()
    done   = False
    actions = []; infos = []
    N = episode["n_nodes"]; M = episode["n_metrics"]

    with torch.no_grad():
        while not done:
            obs_t  = torch.tensor(obs, dtype=torch.float32,
                                  device=device).unsqueeze(0)
            action = policy.get_deterministic_action(obs_t).squeeze(0).cpu().numpy()
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc
            actions.append(action.reshape(N, M).copy())
            infos.append(info)

    return actions, infos


# ══════════════════════════════════════════════════════════════════
# Build panel data
# ══════════════════════════════════════════════════════════════════

def build_data(episode, actions, window_range=None, anom_thresh=None):
    A     = episode["anomaly_scores"]    # [T, N, M]
    L     = episode["anomaly_labels"]    # [T, N, M]
    caps  = episode["node_capacities"]  # [N]
    costs = episode.get("node_costs",
                        np.full(episode["n_nodes"], 0.01))
    N, M  = episode["n_nodes"], episode["n_metrics"]
    T     = episode["T"]
    W     = 10
    n_w   = len(actions)

    b_a = episode.get("b_a", float(np.percentile(A, 30)))
    if anom_thresh is not None:
        b_a = anom_thresh

    # Window range
    w0 = window_range[0] if window_range else 0
    w1 = min(window_range[1], n_w) if window_range else n_w
    actions = actions[w0:w1]
    n_win   = w1 - w0

    # ── Panel 1: window-max anomaly scores [N*M, n_win] ──────────
    score_grid = np.zeros((N * M, n_win), dtype=np.float32)
    label_grid = np.zeros((N * M, n_win), dtype=np.int32)
    for wi, w in enumerate(range(w0, w1)):
        t_s, t_e = w * W, min((w + 1) * W, T)
        A_w = A[t_s:t_e].max(axis=0)   # [N, M]
        L_w = L[t_s:t_e].max(axis=0)   # [N, M]
        for n in range(N):
            for m in range(M):
                score_grid[n * M + m, wi] = A_w[n, m]
                label_grid[n * M + m, wi] = L_w[n, m]

    # ── Panel 2: selections [N*M, n_win] ─────────────────────────
    sel_grid = np.zeros((N * M, n_win), dtype=np.float32)
    for wi, a in enumerate(actions):
        for n in range(N):
            for m in range(M):
                sel_grid[n * M + m, wi] = a[n, m]

    # Decision quality masks
    high_anom = score_grid > b_a                     # high anomaly score
    caught    = (sel_grid == 1) & (label_grid == 1)  # selected & anomalous
    missed    = (sel_grid == 0) & (label_grid == 1)  # not selected & anomalous
    wasted    = (sel_grid == 1) & (label_grid == 0) & (~high_anom)  # selected & low score

    # ── Panel 3: cost per window ──────────────────────────────────
    agent_cost   = np.zeros(n_win)
    monitor_cost = np.zeros(n_win)
    low_anom_window = np.zeros(n_win, dtype=bool)

    for wi, w in enumerate(range(w0, w1)):
        t_s, t_e = w * W, min((w + 1) * W, T)
        A_w  = A[t_s:t_e].max(axis=0)
        a    = actions[wi]
        spn  = a.sum(axis=1)
        agent_cost[wi]   = float((costs * spn).sum())
        monitor_cost[wi] = float((costs * M).sum())
        low_anom_window[wi] = (A_w.mean() < b_a * 0.5)

    # ── Panel 4: capacity utilization per node [N, n_win] ────────
    cap_util = np.zeros((N, n_win), dtype=np.float32)
    violated = np.zeros((N, n_win), dtype=bool)
    for wi, a in enumerate(actions):
        for n in range(N):
            sel = a[n].sum()
            cap_util[n, wi] = sel / (caps[n] + 1e-8)
            violated[n, wi] = sel > caps[n]

    return {
        "score_grid":      score_grid,
        "sel_grid":        sel_grid,
        "label_grid":      label_grid,
        "caught":          caught,
        "missed":          missed,
        "wasted":          wasted,
        "agent_cost":      agent_cost,
        "monitor_cost":    monitor_cost,
        "low_anom_window": low_anom_window,
        "cap_util":        cap_util,
        "violated":        violated,
        "b_a":             b_a,
        "n_win":           n_win,
        "w0":              w0,
        "N": N, "M": M,
        "caps": caps,
        "costs": costs,
    }


# ══════════════════════════════════════════════════════════════════
# Plot
# ══════════════════════════════════════════════════════════════════

def plot(d, out_dir, figsize, dpi):

    plt.rcParams.update({
        "font.family":    "serif",
        "font.size":      7,
        "axes.titlesize": 7,
        "axes.labelsize": 6,
        "xtick.labelsize":5.5,
        "ytick.labelsize":5.5,
        "legend.fontsize":5.5,
        "pdf.fonttype":   42,
        "ps.fonttype":    42,
    })

    N, M   = d["N"], d["M"]
    n_win  = d["n_win"]
    w0     = d["w0"]
    wins   = np.arange(w0, w0 + n_win)

    # Custom colormaps
    score_cmap = LinearSegmentedColormap.from_list(
        "score", ["#F0F4FF", "#BFDBFE", "#3B82F6", "#1E3A8A"])
    sel_cmap   = LinearSegmentedColormap.from_list(
        "sel",   ["#F8FAFC", "#1D4ED8"])
    cap_cmap   = LinearSegmentedColormap.from_list(
        "cap",   ["#DCFCE7", "#FEF9C3", "#FEE2E2"])

    fig, axes = plt.subplots(
        4, 1, figsize=figsize,
        gridspec_kw={"height_ratios": [3, 3, 1.8, 2.0]},
        sharex=True
    )
    fig.subplots_adjust(hspace=0.10, left=0.12, right=0.96,
                        top=0.93, bottom=0.10)

    # ── Panel 1: Anomaly score heatmap ───────────────────────────
    ax1 = axes[0]
    im1 = ax1.imshow(d["score_grid"], aspect="auto", origin="lower",
                     cmap=score_cmap, vmin=0, vmax=1,
                     extent=[w0 - 0.5, w0 + n_win - 0.5, -0.5, N*M - 0.5])

    # Mark anomalous cells (label=1) with a subtle hatch
    for wi in range(n_win):
        for nm in range(N * M):
            if d["label_grid"][nm, wi] == 1:
                ax1.add_patch(plt.Rectangle(
                    (w0 + wi - 0.5, nm - 0.5), 1, 1,
                    fill=False, hatch="///", edgecolor="#DC2626",
                    linewidth=0, alpha=0.5, zorder=2))

    ax1.set_ylabel("(n,m) pair", fontsize=6)
    ax1.set_title("(a)  Anomaly Scores  [window-max]  — hatched = ground-truth anomaly",
                  fontsize=6.5, fontweight="bold", pad=3)
    cb1 = plt.colorbar(im1, ax=ax1, pad=0.01, fraction=0.015)
    cb1.ax.tick_params(labelsize=5)
    cb1.set_label("Score", fontsize=5)

    # Threshold line annotation
    ax1.text(w0 + n_win - 0.5, N*M * d["b_a"],
             f"b_a={d['b_a']:.2f}", ha="right", va="bottom",
             fontsize=5, color="#7C3AED")

    # ── Panel 2: Selection heatmap with decision quality ─────────
    ax2 = axes[1]
    im2 = ax2.imshow(d["sel_grid"], aspect="auto", origin="lower",
                     cmap=sel_cmap, vmin=0, vmax=1,
                     extent=[w0 - 0.5, w0 + n_win - 0.5, -0.5, N*M - 0.5])

    # Green border = caught (selected + anomalous)
    for wi in range(n_win):
        for nm in range(N * M):
            if d["caught"][nm, wi]:
                ax2.add_patch(plt.Rectangle(
                    (w0 + wi - 0.5, nm - 0.5), 1, 1,
                    fill=False, edgecolor="#16A34A",
                    linewidth=0.8, zorder=3))
            elif d["missed"][nm, wi]:
                # Red border = missed anomaly
                ax2.add_patch(plt.Rectangle(
                    (w0 + wi - 0.5, nm - 0.5), 1, 1,
                    fill=False, edgecolor="#DC2626",
                    linewidth=0.8, zorder=3))
            elif d["wasted"][nm, wi]:
                # Orange fill = wasted selection (low anomaly, selected)
                ax2.add_patch(plt.Rectangle(
                    (w0 + wi - 0.5, nm - 0.5), 1, 1,
                    facecolor="#FED7AA", edgecolor="#EA580C",
                    linewidth=0.4, alpha=0.7, zorder=2))

    ax2.set_ylabel("(n,m) pair", fontsize=6)
    ax2.set_title(
        "(b)  Agent Selections  — "
        r"$\bf{Green}$: caught  $\bf{Red}$: missed  $\bf{Orange}$: wasted",
        fontsize=6.5, fontweight="bold", pad=3)
    cb2 = plt.colorbar(im2, ax=ax2, pad=0.01, fraction=0.015)
    cb2.ax.tick_params(labelsize=5)
    cb2.set_label("Selected", fontsize=5)

    # ── Panel 3: Collection cost per window ──────────────────────
    ax3 = axes[2]
    ax3.plot(wins, d["agent_cost"],
             color="#1D4ED8", linewidth=1.2, label="RL agent", zorder=4)
    ax3.plot(wins, d["monitor_cost"],
             color="#9CA3AF", linewidth=0.8, linestyle="--",
             label="Monitor-All", zorder=3)

    # Shade windows where anomaly is low (cost is "wasted effort")
    for wi, w in enumerate(wins):
        if d["low_anom_window"][wi]:
            ax3.axvspan(w - 0.5, w + 0.5, alpha=0.18,
                        color="#FCA5A5", zorder=1)

    ax3.set_ylabel("Coll. Cost", fontsize=6)
    ax3.set_title(
        "(c)  Collection Cost per Window  — "
        r"$\bf{Red\ band}$: low-anomaly windows (cost not justified)",
        fontsize=6.5, fontweight="bold", pad=3)
    ax3.legend(loc="upper right", fontsize=5, framealpha=0.85)
    ax3.grid(True, alpha=0.25, linestyle="--")
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.set_ylim(bottom=0)

    # ── Panel 4: Capacity utilization per node ───────────────────
    ax4 = axes[3]
    im4 = ax4.imshow(d["cap_util"], aspect="auto", origin="lower",
                     cmap=cap_cmap, vmin=0, vmax=1.2,
                     extent=[w0 - 0.5, w0 + n_win - 0.5, -0.5, N - 0.5])

    # Red outline on violated cells
    for wi in range(n_win):
        for n in range(N):
            if d["violated"][n, wi]:
                ax4.add_patch(plt.Rectangle(
                    (w0 + wi - 0.5, n - 0.5), 1, 1,
                    fill=False, edgecolor="#DC2626",
                    linewidth=1.0, zorder=3))

    ax4.set_ylabel("Node", fontsize=6)
    ax4.set_xlabel("Window index", fontsize=6)
    ax4.set_title(
        r"(d)  Capacity Utilization  — $\bf{Red\ border}$: capacity violated",
        fontsize=6.5, fontweight="bold", pad=3)
    ax4.set_yticks(range(N))
    ax4.set_yticklabels([f"n{n}" for n in range(N)], fontsize=5)
    cb4 = plt.colorbar(im4, ax=ax4, pad=0.01, fraction=0.015)
    cb4.ax.tick_params(labelsize=5)
    cb4.set_label("κ_n used", fontsize=5)

    # ── Shared x-axis tick ────────────────────────────────────────
    step = max(1, n_win // 10)
    xticks = np.arange(w0, w0 + n_win, step)
    ax4.set_xticks(xticks)

    # ── Save ─────────────────────────────────────────────────────
    os.makedirs(out_dir, exist_ok=True)
    pdf = os.path.join(out_dir, "decision_trace.pdf")
    png = os.path.join(out_dir, "decision_trace.png")
    plt.savefig(pdf, dpi=dpi, bbox_inches="tight")
    plt.savefig(png, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[trace] Saved → {pdf}")
    print(f"[trace] Saved → {png}")


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load episode
    with open(args.pkl, "rb") as f:
        pool = pickle.load(f)
    episode = pool[args.episode_idx]
    N, M    = episode["n_nodes"], episode["n_metrics"]
    print(f"[trace] Episode {args.episode_idx}: N={N}  M={M}  T={episode['T']}")

    out_dir = args.out_dir or os.path.join(
        os.path.dirname(args.pkl), "figures")

    # Collect agent trajectory
    print(f"[trace] Running agent...")
    actions, infos = collect_trajectory(episode, args.model, cfg, device)
    print(f"[trace] {len(actions)} windows collected")

    # Build panel data
    d = build_data(
        episode, actions,
        window_range  = args.window_range,
        anom_thresh   = args.anomaly_thresh,
    )
    print(f"[trace] Showing windows {d['w0']} → {d['w0']+d['n_win']}  "
          f"b_a={d['b_a']:.4f}")
    print(f"[trace] Caught  : {d['caught'].sum()}")
    print(f"[trace] Missed  : {d['missed'].sum()}")
    print(f"[trace] Wasted  : {d['wasted'].sum()}")
    print(f"[trace] Violated: {d['violated'].sum()} (node,window) pairs")

    plot(d, out_dir, tuple(args.figsize), args.dpi)
    print(f"[trace] Done. Figures in {out_dir}/")


if __name__ == "__main__":
    main()