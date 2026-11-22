"""
visualize_selection_dynamics.py
================================
Shows that the RL agent metric selection is DYNAMIC — it adapts
window by window in response to anomaly patterns — unlike static
baselines that always select the same metrics.

Three-panel figure for one representative episode:
  Panel (a) — Anomaly score heatmap [NM x W windows]
  Panel (b) — Number of selected metrics per window [line per method]
  Panel (c) — Selection change rate per window [line per method]

Usage
-----
  # With real model
  python scripts/visualize_selection_dynamics.py \\
      --pkl   data/10nodes-15metrics/eval_episodes.pkl \\
      --models results/pm_run/best_model.pt \\
               results/flat_run/best_model.pt \\
      --model-labels "RL-PerMetric" "RL-Flat" \\
      --config configs/config.yaml \\
      --episode-idx 0

  # Baselines only (no model needed)
  python scripts/visualize_selection_dynamics.py \\
      --pkl data/10nodes-15metrics/eval_episodes.pkl \\
      --config configs/config.yaml
"""

import os, sys, argparse, pickle, yaml, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── CLI ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pkl",           required=True)
    p.add_argument("--models",        nargs="+", default=None)
    p.add_argument("--model-labels",  nargs="+", default=None)
    p.add_argument("--config",        default="configs/config.yaml")
    p.add_argument("--episode-idx",   type=int,  default=0)
    p.add_argument("--window-range",  nargs=2,   type=int, default=None)
    p.add_argument("--out-dir",       default=None)
    p.add_argument("--figsize",       nargs=2,   type=float, default=[7.2, 5.5])
    p.add_argument("--dpi",           type=int,  default=300)
    p.add_argument("--smooth",        type=int,  default=3)
    return p.parse_args()


# ── Policy loading ────────────────────────────────────────────────

def load_policy(model_path, cfg, obs_dim, action_dim, device):
    from agents.policy import ActorCritic, PerMetricActorCritic
    pcfg        = cfg["policy"]
    policy_type = pcfg.get("policy_type", "flat")
    ecfg        = cfg["environment"]
    N, M        = ecfg["n_nodes"], ecfg["n_metrics"]
    feature_dim = obs_dim // (N * M)
    if policy_type == "per_metric":
        policy = PerMetricActorCritic(
            n_nodes=N, n_metrics=M, feature_dim=feature_dim,
            metric_hidden=pcfg.get("metric_hidden", 128),
            global_hidden=pcfg.get("global_hidden", 256),
            use_layer_norm=pcfg.get("use_layer_norm", True),
            global_pool=pcfg.get("global_pool", "mean"),
        )
    else:
        policy = ActorCritic(
            obs_dim=obs_dim, action_dim=action_dim,
            hidden_sizes=pcfg.get("hidden_sizes", [512, 256]),
            use_layer_norm=pcfg.get("use_layer_norm", True),
        )
    policy.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=False))
    policy.to(device); policy.eval()
    return policy


def collect_rl(policy, episode, cfg, device):
    from envs.monitoring_env import MonitoringEnv
    ecfg = cfg["environment"]; rw = cfg["reward"]
    reward_weights = {
        "alpha": rw["alpha"], "beta": rw.get("beta", 0),
        "lambda_": rw["lambda_"], "delta": rw.get("delta", 0),
        "eta": rw.get("eta", 0),
    }
    env = MonitoringEnv(
        episode_pool=[episode], reward_weights=reward_weights,
        per_metric_cost=ecfg["per_metric_cost"],
        anomaly_baseline_pct=ecfg["anomaly_baseline_pct"],
        causal_baseline_pct=ecfg["causal_baseline_pct"],
        randomize_episodes=False,
        window_size=ecfg.get("window_size", 10),
        window_agg=ecfg.get("window_agg", "max"),
        coverage_mode=ecfg.get("coverage_mode", "window"),
    )
    obs, _ = env.reset(); done = False; actions = []
    N, M = episode["n_nodes"], episode["n_metrics"]
    with torch.no_grad():
        while not done:
            obs_t  = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            action = policy.get_deterministic_action(obs_t).squeeze(0).cpu().numpy()
            obs, _, term, trunc, _ = env.step(action)
            done = term or trunc
            actions.append(action.reshape(N, M).copy())
    return np.array(actions)


# ── Baselines ─────────────────────────────────────────────────────

def topk_actions(episode, W):
    A = episode["anomaly_scores"]; T, N, M = A.shape
    cap = episode["node_capacities"]
    acts = []
    for w in range(T // W):
        A_w = A[w*W:min((w+1)*W,T)].max(axis=0)
        a   = np.zeros((N, M), np.float32)
        for n in range(N):
            k = max(1, int(cap[n]))
            a[n, np.argsort(A_w[n])[::-1][:k]] = 1.0
        acts.append(a)
    return np.array(acts)

def random_actions(episode, W, seed=0):
    rng = np.random.default_rng(seed)
    A = episode["anomaly_scores"]; T, N, M = A.shape
    cap = episode["node_capacities"]
    acts = []
    for w in range(T // W):
        a = np.zeros((N, M), np.float32)
        for n in range(N):
            k = max(1, int(cap[n]))
            a[n, rng.choice(M, size=k, replace=False)] = 1.0
        acts.append(a)
    return np.array(acts)

def monitor_all(episode, W):
    T, N, M = episode["anomaly_scores"].shape
    return np.ones((T // W, N, M), np.float32)


# ── Metrics ───────────────────────────────────────────────────────

def n_selected(actions):
    return actions.reshape(len(actions), -1).sum(axis=1)

def change_rate(actions):
    diff = np.abs(np.diff(actions, axis=0))
    NM   = actions.shape[1] * actions.shape[2]
    return diff.reshape(len(diff), -1).sum(axis=1) / NM

def anom_load(episode, W):
    L = episode["anomaly_labels"]; T, N, M = L.shape; NM = N * M
    vals = []
    for w in range(T // W):
        L_w = L[w*W:min((w+1)*W,T)].max(axis=0)
        vals.append(float(L_w.sum()) / NM)
    return np.array(vals)

def smooth(arr, w):
    if w <= 1 or len(arr) < w: return arr
    k = np.ones(w) / w
    return np.convolve(np.pad(arr,(w//2,w//2),mode="edge"),k,mode="valid")[:len(arr)]


# ── Plot ──────────────────────────────────────────────────────────

RL_COLORS = ["#EE6677","#228833","#AA3377","#4477AA"]

def plot(episode, w0, w1, score_grid, label_grid,
         methods, out_dir, figsize, dpi, smooth_w):

    plt.rcParams.update({
        "font.family":    "serif",
        "font.size":      6,
        "axes.titlesize": 6,
        "axes.labelsize": 6,
        "xtick.labelsize":5.5,
        "ytick.labelsize":5.5,
        "legend.fontsize":5,
        "lines.linewidth":1.0,
        "pdf.fonttype":   42,
        "ps.fonttype":    42,
    })

    NM   = score_grid.shape[0]
    wins = np.arange(w0, w1)

    fig, axes = plt.subplots(3, 1, figsize=figsize)
    fig.subplots_adjust(hspace=0.52, left=0.12, right=0.96,
                        top=0.93, bottom=0.09)

    # ── Panel (a) anomaly scores ─────────────────────────────────
    ax1 = axes[0]
    cmap_a = LinearSegmentedColormap.from_list(
        "an", ["#F0F4FF","#93C5FD","#1D4ED8","#1E3A8A"])
    im = ax1.imshow(score_grid[:, w0:w1], aspect="auto", origin="lower",
                    cmap=cmap_a, vmin=0, vmax=1,
                    extent=[w0-0.5, w1-0.5, -0.5, NM-0.5])
    for wi in range(w1 - w0):
        for nm in range(NM):
            if label_grid[nm, w0+wi] == 1:
                ax1.add_patch(plt.Rectangle(
                    (w0+wi-0.5, nm-0.5), 1, 1,
                    fill=False, hatch="///", edgecolor="#DC2626",
                    linewidth=0, alpha=0.45, zorder=2))
    ax1.set_ylabel("(n,m) pair", fontsize=6)
    ax1.set_title(
        "(a)  Anomaly Scores per Window  —  hatched = ground-truth anomaly",
        fontsize=6, fontweight="bold", pad=3)
    cb = plt.colorbar(im, ax=ax1, pad=0.01, fraction=0.015)
    cb.ax.tick_params(labelsize=4.5); cb.set_label("Score", fontsize=5)
    ax1.tick_params(axis="x", labelbottom=False)

    # ── Panel (b) selected per window ────────────────────────────
    ax2 = axes[1]
    # Anomaly load background (stored as plain array, not method tuple)
    al = methods["__anom_load"][w0:w1]
    ax2.fill_between(wins, 0, al * NM,
                     color="#FEE2E2", alpha=0.45, zorder=0,
                     label="Anomaly load (×NM)")

    for name, val in methods.items():
        if name.startswith("__"): continue
        acts, color, ls, lw = val
        sel = n_selected(acts)[w0:w1]
        ax2.plot(wins, sel, color=color, linestyle=ls, linewidth=lw,
                 label=name, zorder=4 if color in RL_COLORS else 3,
                 marker="none")

    total_cap = float(episode["node_capacities"].sum())
    ax2.axhline(total_cap, color="#374151", lw=0.6,
                linestyle=":", alpha=0.7)
    ax2.text(w0+0.5, total_cap+0.3,
             f"Σκ_n={int(total_cap)}",
             fontsize=4.5, color="#374151", fontstyle="italic")

    ax2.set_ylabel("# Selected Metrics", fontsize=6)
    ax2.set_title(
        "(b)  Selected Metrics per Window",
        fontsize=6, fontweight="bold", pad=3)
    ax2.legend(loc="upper right", fontsize=4.8, framealpha=0.9,
               ncol=2, handlelength=1.5, handleheight=0.8,
               borderpad=0.4, columnspacing=0.6)
    ax2.set_ylim(bottom=0)
    ax2.grid(True, alpha=0.2, ls="--")
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
    ax2.tick_params(axis="x", labelbottom=False)

    # ── Panel (c) change rate ─────────────────────────────────────
    ax3 = axes[2]
    for name, val in methods.items():
        if name.startswith("__"): continue
        acts, color, ls, lw = val
        cr   = change_rate(acts)        # [W-1]
        cr_s = smooth(cr, smooth_w)
        w_cr = np.arange(1, len(cr)+1)
        mask = (w_cr >= w0) & (w_cr < w1)
        if mask.sum() == 0: continue
        ax3.plot(w_cr[mask], cr_s[mask],
                 color=color, linestyle=ls, linewidth=lw,
                 label=name, zorder=4 if color in RL_COLORS else 3,
                 marker="none")

    ax3.set_xlabel("Window index", fontsize=6)
    ax3.set_ylabel("Change Rate |Δa|/NM", fontsize=6)
    ax3.set_title(
        "(c)  Selection Change Rate",
        fontsize=6, fontweight="bold", pad=3)
    ax3.set_ylim(bottom=0)
    ax3.grid(True, alpha=0.2, ls="--")
    ax3.spines["top"].set_visible(False); ax3.spines["right"].set_visible(False)

    for ax in axes:
        ax.set_xlim(w0-0.5, w1-0.5)

    os.makedirs(out_dir, exist_ok=True)
    for ext in ["pdf","png"]:
        path = os.path.join(out_dir, f"selection_dynamics.{ext}")
        plt.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"[dyn] Saved → {path}")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    with open(args.pkl, "rb") as f:
        pool = pickle.load(f)
    episode = pool[args.episode_idx]
    N, M    = episode["n_nodes"], episode["n_metrics"]
    T       = episode["T"]
    W       = cfg["environment"].get("window_size", 10)
    n_wins  = T // W
    print(f"[dyn] Episode {args.episode_idx}: N={N} M={M} T={T} W={W} windows={n_wins}")

    w0 = args.window_range[0] if args.window_range else 0
    w1 = min(args.window_range[1], n_wins) if args.window_range else n_wins

    out_dir = args.out_dir or os.path.join(
        os.path.dirname(args.pkl), "figures")

    # Anomaly score/label grids
    A = episode["anomaly_scores"]; L = episode["anomaly_labels"]
    score_grid = np.zeros((N*M, n_wins), np.float32)
    label_grid = np.zeros((N*M, n_wins), np.int32)
    for w in range(n_wins):
        A_w = A[w*W:min((w+1)*W,T)].max(axis=0)
        L_w = L[w*W:min((w+1)*W,T)].max(axis=0)
        for n in range(N):
            for m in range(M):
                score_grid[n*M+m, w] = A_w[n,m]
                label_grid[n*M+m, w] = L_w[n,m]

    # Build methods dict: name → (actions, color, linestyle, linewidth)
    methods = {}
    methods["Monitor-All"] = (monitor_all(episode,W),   "#BBBBBB", "-",  0.9)
    methods["TopK"]        = (topk_actions(episode,W),  "#66CCEE", "--", 1.1)
    methods["Random"]      = (random_actions(episode,W), "#CCBB44", ":",  0.9)
    methods["__anom_load"] = anom_load(episode,W)

    # RL agents
    if args.models:
        from envs.monitoring_env import MonitoringEnv
        ecfg = cfg["environment"]; rw = cfg["reward"]
        tmp  = MonitoringEnv(
            episode_pool=[episode],
            reward_weights={"alpha":rw["alpha"],"beta":rw.get("beta",0),
                            "lambda_":rw["lambda_"],"delta":rw.get("delta",0),
                            "eta":rw.get("eta",0)},
            per_metric_cost=ecfg["per_metric_cost"],
            anomaly_baseline_pct=ecfg["anomaly_baseline_pct"],
            causal_baseline_pct=ecfg["causal_baseline_pct"],
            window_size=W, window_agg=ecfg.get("window_agg","max"),
            coverage_mode=ecfg.get("coverage_mode","window"),
        )
        obs_dim    = tmp.observation_space.shape[0]
        action_dim = tmp.action_space.n

        rl_labels = args.model_labels or []
        for i, mp in enumerate(args.models):
            label = rl_labels[i] if i < len(rl_labels) else f"RL-{i+1}"
            color = RL_COLORS[i % len(RL_COLORS)]
            print(f"[dyn] Loading {label}")
            pol  = load_policy(mp, cfg, obs_dim, action_dim, device)
            acts = collect_rl(pol, episode, cfg, device)
            methods[label] = (acts, color, "-", 1.4)
            print(f"[dyn]   {label}: {len(acts)} windows")

    plot(episode, w0, w1, score_grid, label_grid,
         methods, out_dir, tuple(args.figsize), args.dpi, args.smooth)
    print("[dyn] Done.")


if __name__ == "__main__":
    main()