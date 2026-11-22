"""
visualize_tier_analysis.py
==========================
Reads the aggregate metrics CSV produced by compute_metrics.py and
generates a two-panel figure showing tier-specific analysis:
  Panel (a) — ACR per tier (IoT / Edge / Cloud) per method
  Panel (b) — Collection Cost per tier per method

Reads directly from metrics_*_aggregate.csv — no re-evaluation needed.

Usage
-----
  # Default path
  python scripts/visualize_tier_analysis.py

  # Explicit CSV
  python scripts/visualize_tier_analysis.py \
      --csv results/my_run/metrics/metrics_window_aggregate.csv

  # Custom method order
  python scripts/visualize_tier_analysis.py \
      --method-order "Monitor-Everything" "TopK-Anomaly-Causal" "RL-PPO (Ours)"

  # Only show a subset of methods (cleaner figure)
  python scripts/visualize_tier_analysis.py \
      --methods "Monitor-Everything" "TopK-Anomaly-Causal" "RL-PPO (Ours)"
"""

import os
import sys
import argparse
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ══════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",          default=None,
                   help="Path to metrics_*_aggregate.csv. "
                        "Default: results/metrics/metrics_window_aggregate.csv")
    p.add_argument("--out-dir",      default=None,
                   help="Output directory. Default: same dir as CSV.")
    p.add_argument("--methods",      nargs="+", default=None,
                   help="Subset of methods to show (default: all).")
    p.add_argument("--method-order", nargs="+", default=None,
                   help="Explicit method order on x-axis.")
    p.add_argument("--no-std",       action="store_true",
                   help="Do not draw error bars.")
    p.add_argument("--figsize",      nargs=2, type=float, default=[3.5, 2.4],
                   metavar=("W", "H"),
                   help="Figure size in inches (default: 3.5 2.4 — one IEEE column).")
    p.add_argument("--dpi",          type=int, default=300)
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════
# CSV loader
# ══════════════════════════════════════════════════════════════════

def load_csv(path):
    data = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            method = row.pop("method")
            parsed = {}
            for k, v in row.items():
                try:
                    parsed[k] = float(v)
                except (ValueError, TypeError):
                    parsed[k] = float("nan")
            data[method] = parsed
    return data


# ══════════════════════════════════════════════════════════════════
# Palette — IEEE-friendly (Paul Tol bright)
# ══════════════════════════════════════════════════════════════════

PALETTE = [
    "#4477AA",   # blue
    "#BBBBBB",   # grey
    "#EE6677",   # red
    "#228833",   # green
    "#CCBB44",   # yellow
    "#66CCEE",   # cyan
    "#AA3377",   # purple
    "#000000",   # black
]

# Tier colours — fixed, consistent across both panels
TIER_COLORS = {
    "IoT":   "#EE6677",   # red   — expensive, constrained
    "Edge":  "#CCBB44",   # yellow
    "Cloud": "#4477AA",   # blue  — cheap, abundant
}

SUBFIG_LABELS = ["(a)", "(b)", "(c)"]


# ══════════════════════════════════════════════════════════════════
# Plot
# ══════════════════════════════════════════════════════════════════

def plot(data, methods, out_dir, show_std, figsize, dpi):

    # ── rcParams for IEEE ─────────────────────────────────────────
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

    # ── Data extraction ───────────────────────────────────────────
    tiers = ["IoT", "Edge", "Cloud"]
    acr_keys  = ["acr_iot_mean",  "acr_edge_mean",  "acr_cloud_mean"]
    cost_keys = ["cost_iot_mean", "cost_edge_mean",  "cost_cloud_mean"]
    acr_std_keys  = ["acr_iot_std",  "acr_edge_std",  "acr_cloud_std"]
    cost_std_keys = ["cost_iot_std", "cost_edge_std",  "cost_cloud_std"]

    def _val(method, key):
        v = data[method].get(key, float("nan"))
        return 0.0 if (isinstance(v, float) and np.isnan(v)) else v

    def _std(method, key):
        v = data[method].get(key, float("nan"))
        return 0.0 if (isinstance(v, float) and np.isnan(v)) else v

    # Check whether tier data exists
    has_data = any(
        not np.isnan(data[m].get(k, float("nan")))
        for m in methods for k in acr_keys
    )
    if not has_data:
        print("[tier_viz] No tier data found in CSV.")
        print("[tier_viz] Make sure compute_metrics.py ran with a config")
        print("[tier_viz] that has multiple tiers (N≥3 nodes across IoT/Edge/Cloud).")
        return

    # ── Layout ────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.subplots_adjust(wspace=0.48, left=0.10, right=0.98,
                        top=0.88, bottom=0.10)

    n_methods = len(methods)
    n_tiers   = len(tiers)
    # Group width = 1.0, bar width within group
    bw        = 0.22
    offsets   = np.linspace(
        -(n_tiers - 1) * bw / 2,
         (n_tiers - 1) * bw / 2,
        n_tiers
    )
    x = np.arange(n_methods)

    panels = [
        {
            "ax":       axes[0],
            "keys":     acr_keys,
            "std_keys": acr_std_keys,
            "title":    "Anomaly Coverage Rate (ACR)",
            "ylabel":   "ACR",
            "ylim":     (0.0, 1.08),
            "fmt":      ".3f",
            "label":    SUBFIG_LABELS[0],
        },
        {
            "ax":       axes[1],
            "keys":     cost_keys,
            "std_keys": cost_std_keys,
            "title":    "Collection Cost per Tier",
            "ylabel":   "Collection Cost",
            "ylim":     None,
            "fmt":      ".4f",
            "label":    SUBFIG_LABELS[1],
        },
    ]

    # ── Draw panels ───────────────────────────────────────────────
    for panel in panels:
        ax = panel["ax"]

        for ti, (tier, key, std_key, offset) in enumerate(
            zip(tiers, panel["keys"], panel["std_keys"], offsets)
        ):
            vals = np.array([_val(m, key) for m in methods])
            stds = np.array([_std(m, std_key) for m in methods])

            bars = ax.bar(
                x + offset, vals,
                width     = bw,
                color     = TIER_COLORS[tier],
                edgecolor = "white",
                linewidth = 0.5,
                label     = tier,
                zorder    = 3,
                alpha     = 0.88,
            )

            # Error bars
            if show_std and not np.all(stds == 0):
                ax.errorbar(
                    x + offset, vals, yerr=stds,
                    fmt        = "none",
                    ecolor     = "black",
                    elinewidth = 0.8,
                    capsize    = 2,
                    capthick   = 0.8,
                    zorder     = 4,
                )

            # Value labels on bars
            for bar, v in zip(bars, vals):
                if v == 0 or np.isnan(v):
                    continue
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + bar.get_height() * 0.04 + 1e-6,
                    format(v, panel["fmt"]),
                    ha="center", va="bottom",
                    fontsize=4, color="#111111",
                    rotation=90,
                )

        # Axes formatting
        ax.set_xticks(x)
        ax.set_xticklabels([""] * n_methods)   # no x labels — legend only
        ax.tick_params(axis="x", length=0)
        ax.set_ylabel(panel["ylabel"], fontsize=6)
        ax.set_title(panel["title"], fontsize=6, fontweight="bold", pad=4)

        if panel["ylim"]:
            ax.set_ylim(*panel["ylim"])
        else:
            all_vals = [_val(m, k) for m in methods for k in panel["keys"]]
            valid    = [v for v in all_vals if v > 0]
            if valid:
                ax.set_ylim(0, max(valid) * 1.45)

        ax.grid(True, axis="y", alpha=0.3, linestyle="--", zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Subfigure label — top right
        ax.text(0.97, 0.97, panel["label"],
                transform=ax.transAxes,
                fontsize=7, fontweight="bold",
                va="top", ha="right", color="black")

    # ── Two legends ───────────────────────────────────────────────
    # Legend 1: tiers (colour = tier)
    tier_handles = [
        mpatches.Patch(color=TIER_COLORS[t], label=t) for t in tiers
    ]
    # Legend 2: methods (hatch pattern per method using text)
    method_colors = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(methods)}

    # Place tier legend inside panel (a), top left
    axes[0].legend(
        handles=tier_handles,
        loc="upper left",
        fontsize=5,
        framealpha=0.9,
        frameon=True,
        edgecolor="#cccccc",
        handlelength=0.8,
        handleheight=0.7,
        borderpad=0.4,
    )

    # Place method legend below both panels
    method_handles = [
        mpatches.Patch(color="white", edgecolor="#444444",
                       label=m, linewidth=0.5)
        for i, m in enumerate(methods)
    ]
    # Show method names as x-tick annotations instead of legend
    # Write method names below each group
    for ax in axes:
        for xi, m in enumerate(methods):
            short = m.replace("Monitor-Everything", "Monitor\nAll")\
                     .replace("Random-Feasible", "Random")\
                     .replace("TopK-Anomaly-Causal", "TopK-AC")\
                     .replace("TopK-Anomaly-Only", "TopK-A")\
                     .replace("Cost-Aware TopK", "CostTopK")\
                     .replace("Behavioral Cloning", "BC")\
                     .replace("RL-PPO (Ours)", "Ours")\
                     .replace("RL-Flat MLP", "Flat MLP")\
                     .replace("RL-PerMetric", "PerMetric")
            ax.text(xi, -0.04, short,
                    transform=ax.get_xaxis_transform(),
                    ha="center", va="top",
                    fontsize=4.5, color="#333333")

    # ── Save ─────────────────────────────────────────────────────
    os.makedirs(out_dir, exist_ok=True)
    pdf_path = os.path.join(out_dir, "tier_analysis.pdf")
    png_path = os.path.join(out_dir, "tier_analysis.png")
    plt.savefig(pdf_path, dpi=dpi, bbox_inches="tight")
    plt.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[tier_viz] Saved → {pdf_path}")
    print(f"[tier_viz] Saved → {png_path}")


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # Resolve CSV
    if args.csv:
        csv_path = args.csv
    else:
        csv_path = os.path.join(
            "results", "metrics", "metrics_window_aggregate.csv"
        )

    if not os.path.exists(csv_path):
        print(f"[tier_viz] ERROR: CSV not found at {csv_path}")
        print("[tier_viz] Run compute_metrics.py first, or pass --csv <path>")
        sys.exit(1)

    out_dir = args.out_dir or os.path.join(
        os.path.dirname(csv_path), "figures"
    )

    # Load
    data = load_csv(csv_path)
    if not data:
        print("[tier_viz] No data in CSV.")
        sys.exit(1)

    # Resolve methods
    if args.methods:
        methods = [m for m in args.methods if m in data]
    elif args.method_order:
        methods = [m for m in args.method_order if m in data]
    else:
        methods = list(data.keys())

    print(f"[tier_viz] Methods : {methods}")
    print(f"[tier_viz] CSV     : {csv_path}")
    print(f"[tier_viz] Out dir : {out_dir}")

    plot(
        data      = data,
        methods   = methods,
        out_dir   = out_dir,
        show_std  = not args.no_std,
        figsize   = tuple(args.figsize),
        dpi       = args.dpi,
    )

    print(f"[tier_viz] Done.")


if __name__ == "__main__":
    main()