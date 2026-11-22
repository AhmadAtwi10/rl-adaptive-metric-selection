"""
visualize_training_curves.py
=============================
Reads training_log.csv produced by ppo.py and generates a two-panel
training curve figure:
  Panel (a) — Eval reward vs training steps
  Panel (b) — Anomaly Coverage Rate (ACR) vs training steps

Optionally overlays multiple runs (e.g. different seeds or configs)
on the same axes for comparison.

A horizontal dashed reference line is drawn on panel (b) at the
TopK-Anomaly-Causal baseline ACR level, showing when the agent
surpasses the strongest heuristic baseline.

Usage
-----
  # Single run (default path)
  python scripts/visualize_training_curves.py \
      --log results/my_run/training_log.csv

  # Multiple runs overlaid (e.g. 5 seeds)
  python scripts/visualize_training_curves.py \
      --logs results/seed1/training_log.csv \
             results/seed2/training_log.csv \
             results/seed3/training_log.csv \
      --labels "Seed 1" "Seed 2" "Seed 3"

  # With TopK baseline reference line
  python scripts/visualize_training_curves.py \
      --log results/my_run/training_log.csv \
      --topk-acr 0.87

  # Compare flat MLP vs PerMetric
  python scripts/visualize_training_curves.py \
      --logs results/flat_run/training_log.csv \
             results/pm_run/training_log.csv \
      --labels "Flat MLP" "PerMetric (Ours)" \
      --topk-acr 0.87

  # Custom output
  python scripts/visualize_training_curves.py \
      --log results/my_run/training_log.csv \
      --out-dir results/my_run/figures \
      --smooth 5
"""

import os
import sys
import argparse
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ══════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()

    # Single log (convenience)
    p.add_argument("--log",      default=None,
                   help="Path to a single training_log.csv.")

    # Multiple logs
    p.add_argument("--logs",     nargs="+", default=None,
                   help="Paths to multiple training_log.csv files "
                        "(overlaid on the same axes).")
    p.add_argument("--labels",   nargs="+", default=None,
                   help="Display labels for each log file. "
                        "Default: filename stem.")

    p.add_argument("--out-dir",  default=None,
                   help="Output directory. "
                        "Default: same directory as the first log file.")
    p.add_argument("--topk-acr", type=float, default=None,
                   help="ACR value of TopK-Anomaly-Causal baseline. "
                        "Drawn as horizontal dashed line on panel (b).")
    p.add_argument("--smooth",   type=int, default=5,
                   help="Smoothing window (rolling mean). "
                        "Default 5. Set to 1 to disable.")
    p.add_argument("--figsize",  nargs=2, type=float, default=[3.5, 2.2],
                   metavar=("W", "H"),
                   help="Figure size in inches (default: 3.5 2.2 — one IEEE column).")
    p.add_argument("--dpi",      type=int, default=300)
    p.add_argument("--no-fill",  action="store_true",
                   help="Do not shade the area under the curve.")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════

def load_log(path):
    """
    Load training_log.csv into arrays.
    Returns dict of column_name -> np.array.
    """
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            for k, v in row.items():
                try:
                    parsed[k] = float(v)
                except (ValueError, TypeError):
                    parsed[k] = float("nan")
            rows.append(parsed)

    if not rows:
        return {}

    keys = list(rows[0].keys())
    return {k: np.array([r.get(k, float("nan")) for r in rows]) for k in keys}


def smooth(arr, window):
    """Rolling mean smoothing."""
    if window <= 1 or len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    # pad with edge values to avoid boundary artifacts
    padded = np.pad(arr, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[:len(arr)]


# ══════════════════════════════════════════════════════════════════
# Palette — IEEE-friendly (Paul Tol bright)
# ══════════════════════════════════════════════════════════════════

PALETTE = [
    "#4477AA",   # blue
    "#EE6677",   # red
    "#228833",   # green
    "#CCBB44",   # yellow
    "#66CCEE",   # cyan
    "#AA3377",   # purple
    "#BBBBBB",   # grey
    "#000000",   # black
]

SUBFIG_LABELS = ["(a)", "(b)"]


# ══════════════════════════════════════════════════════════════════
# Plot
# ══════════════════════════════════════════════════════════════════

def plot(logs_data, labels, out_dir, topk_acr,
         smooth_w, figsize, dpi, fill):

    # ── rcParams ─────────────────────────────────────────────────
    plt.rcParams.update({
        "font.family":    "serif",
        "font.size":      6,
        "axes.titlesize": 6,
        "axes.labelsize": 6,
        "xtick.labelsize":5,
        "ytick.labelsize":5,
        "legend.fontsize":5,
        "pdf.fonttype":   42,
        "ps.fonttype":    42,
    })

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.subplots_adjust(wspace=0.42, left=0.11, right=0.98,
                        top=0.88, bottom=0.18)

    panels = [
        {
            "ax":     axes[0],
            "col":    "eval_reward",
            "title":  "Eval Reward",
            "ylabel": "Reward",
            "label":  SUBFIG_LABELS[0],
            "ref":    None,
            "reflabel": None,
        },
        {
            "ax":     axes[1],
            "col":    "eval_coverage",
            "title":  "Anomaly Coverage Rate",
            "ylabel": "ACR",
            "label":  SUBFIG_LABELS[1],
            "ref":    topk_acr,
            "reflabel": "TopK-Anomaly-Causal",
        },
    ]

    for panel in panels:
        ax     = panel["ax"]
        col    = panel["col"]
        has_data = False

        for i, (data, label) in enumerate(zip(logs_data, labels)):
            if col not in data:
                continue
            steps = data.get("step", np.arange(len(data[col])))
            vals  = data[col]

            # Remove NaN
            mask  = ~np.isnan(vals)
            steps = steps[mask]
            vals  = vals[mask]

            if len(vals) == 0:
                continue

            has_data = True
            color    = PALETTE[i % len(PALETTE)]

            # Smooth
            vals_s = smooth(vals, smooth_w)

            # Steps in millions
            steps_m = steps / 1e6

            # Plot smoothed line
            ax.plot(steps_m, vals_s,
                    color=color, linewidth=1.2,
                    label=label, zorder=4)

            # Raw values as light background
            if fill and smooth_w > 1:
                ax.fill_between(steps_m, vals, vals_s,
                                color=color, alpha=0.12, zorder=2)

        # Reference line (TopK baseline)
        if panel["ref"] is not None and has_data:
            ax.axhline(panel["ref"],
                       color="#374151", linewidth=0.9,
                       linestyle="--", zorder=3,
                       label=panel["reflabel"])
            # Annotate the line
            xlim = ax.get_xlim()
            ax.text(xlim[1] * 0.98, panel["ref"] + 0.005,
                    panel["reflabel"],
                    ha="right", va="bottom",
                    fontsize=4.5, color="#374151",
                    fontstyle="italic")

        # Axes formatting
        ax.set_xlabel("Training steps (×10⁶)", fontsize=6)
        ax.set_ylabel(panel["ylabel"], fontsize=6)
        ax.set_title(panel["title"], fontsize=6,
                     fontweight="bold", pad=4)
        ax.grid(True, alpha=0.3, linestyle="--", zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Subfigure label — top right
        ax.text(0.97, 0.03, panel["label"],
                transform=ax.transAxes,
                fontsize=7, fontweight="bold",
                va="bottom", ha="right", color="black")

        # ACR panel: set y limits sensibly
        if col == "eval_coverage":
            ax.set_ylim(bottom=0.0, top=1.05)

    # ── Shared legend ─────────────────────────────────────────────
    handles, lbls = axes[0].get_legend_handles_labels()
    # Deduplicate
    seen = set(); h_final = []; l_final = []
    for h, l in zip(handles, lbls):
        if l not in seen:
            seen.add(l); h_final.append(h); l_final.append(l)

    fig.legend(h_final, l_final,
               loc="lower center",
               ncol=min(len(h_final), 4),
               fontsize=5,
               framealpha=0.92,
               frameon=True,
               edgecolor="#cccccc",
               bbox_to_anchor=(0.5, -0.02),
               handlelength=1.2,
               handleheight=0.8,
               columnspacing=0.8)

    # ── Save ─────────────────────────────────────────────────────
    os.makedirs(out_dir, exist_ok=True)
    pdf_path = os.path.join(out_dir, "training_curves.pdf")
    png_path = os.path.join(out_dir, "training_curves.png")
    plt.savefig(pdf_path, dpi=dpi, bbox_inches="tight")
    plt.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[curves] Saved → {pdf_path}")
    print(f"[curves] Saved → {png_path}")


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # Resolve log paths
    if args.logs:
        log_paths = args.logs
    elif args.log:
        log_paths = [args.log]
    else:
        # Default: look in results/
        default = os.path.join("results", "training_log.csv")
        if os.path.exists(default):
            log_paths = [default]
        else:
            print("[curves] ERROR: no log file specified and default not found.")
            print("[curves] Pass --log <path> or --logs <path1> <path2> ...")
            sys.exit(1)

    # Resolve labels
    if args.labels:
        labels = args.labels
        if len(labels) < len(log_paths):
            # Fill missing labels with stem
            from pathlib import Path
            labels += [Path(p).parent.name for p in log_paths[len(labels):]]
    else:
        from pathlib import Path
        labels = [Path(p).parent.name for p in log_paths]

    # Output directory
    out_dir = args.out_dir or os.path.join(
        os.path.dirname(log_paths[0]), "figures"
    )

    # Load all logs
    logs_data = []
    for path in log_paths:
        if not os.path.exists(path):
            print(f"[curves] WARNING: {path} not found — skipping")
            logs_data.append({})
            continue
        data = load_log(path)
        logs_data.append(data)
        n = len(next(iter(data.values()), []))
        print(f"[curves] Loaded {n} eval checkpoints ← {path}")

    if not any(logs_data):
        print("[curves] No data loaded.")
        sys.exit(1)

    print(f"[curves] Labels  : {labels}")
    print(f"[curves] Smooth  : {args.smooth} steps")
    print(f"[curves] TopK ref: {args.topk_acr}")
    print(f"[curves] Out dir : {out_dir}")

    plot(
        logs_data = logs_data,
        labels    = labels,
        out_dir   = out_dir,
        topk_acr  = args.topk_acr,
        smooth_w  = args.smooth,
        figsize   = tuple(args.figsize),
        dpi       = args.dpi,
        fill      = not args.no_fill,
    )

    print("[curves] Done.")


if __name__ == "__main__":
    main()