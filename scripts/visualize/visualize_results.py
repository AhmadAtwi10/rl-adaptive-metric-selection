"""
visualize_results.py
====================
Reads the aggregate metrics CSV produced by compute_metrics.py and
generates a clean paper-ready figure showing three metrics:
  - Anomaly Coverage Rate (ACR)
  - Mean Selected metrics per window
  - Mean Collection Cost

One grouped bar chart with three panels, one bar per method.
Saved as PDF (vector) and PNG (raster).

Usage
-----
  # Multi-seed grid: error bars are the between-seed std, matching Table I
  python scripts/visualize/visualize_results.py \
      --summary results/<dataset>/summary.csv \
      --out-dir results/<dataset>/figures

  # Default: reads metrics_window_aggregate.csv from results/metrics/
  python scripts/visualize_results.py

  # Explicit CSV path
  python scripts/visualize_results.py \
      --csv results/my_run/metrics/metrics_window_aggregate.csv

  # Explicit output directory
  python scripts/visualize_results.py \
      --csv results/my_run/metrics/metrics_window_aggregate.csv \
      --out-dir results/my_run/figures

  # Custom method order and colors
  python scripts/visualize_results.py \
      --csv results/my_run/metrics/metrics_window_aggregate.csv \
      --method-order "Monitor-Everything" "Random-Feasible" \
                     "TopK-Anomaly-Causal" "RL-Flat MLP" "RL-PerMetric"

  # per_timestep coverage variant
  python scripts/visualize_results.py \
      --csv results/my_run/metrics/metrics_per_timestep_aggregate.csv
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
from pathlib import Path


# ══════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--csv",
        default=None,
        help="Path to metrics_*_aggregate.csv. "
             "Default: results/metrics/metrics_window_aggregate.csv",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for figures. "
             "Default: same directory as the CSV file.",
    )
    p.add_argument(
        "--method-order",
        nargs="+",
        default=None,
        help="Explicit method order for the x-axis. "
             "Default: order as they appear in the CSV.",
    )
    p.add_argument(
        "--summary",
        default=None,
        help="aggregate_results.py's summary.csv for a multi-seed grid. "
             "Error bars are then the BETWEEN-SEED std (the same +/- as "
             "Table I), instead of the across-episode spread inside one run "
             "that --csv carries. Mutually exclusive with --csv.",
    )
    p.add_argument(
        "--no-std",
        action="store_true",
        help="Do not draw error bars even if std columns are present.",
    )
    p.add_argument(
        "--figsize",
        nargs=2,
        type=float,
        default=[3.5, 2.2],   # one-column IEEE: 3.5" wide, compact height
        metavar=("W", "H"),
        help="Figure size in inches (default: 3.5 2.2 — one IEEE column).",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=300,          # 300 DPI — print quality
    )
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════
# CSV loader
# ══════════════════════════════════════════════════════════════════

def load_csv(path: str) -> dict:
    """
    Load aggregate CSV into a dict:
        { method_name: { column: value, ... }, ... }
    """
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
    print(f"[viz] Loaded {len(data)} methods from {path}")
    for m in data:
        print(f"  {m}")
    return data


# Panel key used below -> metric name in aggregate_results.py's summary.csv
SUMMARY_KEYS = {
    "acr_mean":                  "acr",
    "mean_selected_mean":        "mean_selected",
    "mean_collection_cost_mean": "mean_collection_cost",
}


def load_summary(path: str) -> dict:
    """
    Load aggregate_results.py's long-format summary.csv into the same wide dict
    load_csv returns, so the plotting path is identical.

    The `<key>_std` entries carry `std_between_seeds` — seed variability, which
    is what a multi-seed figure should show.  load_csv's `_std` columns instead
    hold the across-episode spread within a single run; the two are different
    quantities and must not be mixed in one figure.
    """
    raw = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                raw.setdefault(row["method"], {})[row["metric"]] = (
                    float(row["mean"]), float(row["std_between_seeds"]),
                    int(row["n_seeds"]))
            except (ValueError, TypeError, KeyError):
                continue

    data = {}
    for method, mets in raw.items():
        entry, n_seeds = {}, None
        for key, metric in SUMMARY_KEYS.items():
            if metric not in mets:
                continue
            mean, std, n = mets[metric]
            entry[key], entry[key + "_std"] = mean, std
            n_seeds = n
        if entry:
            data[method] = entry
            print(f"  {method:<22} n={n_seeds}  " + "  ".join(
                f"{k.replace('_mean','')}={entry[k]:.4g}+-{entry[k+'_std']:.3g}"
                for k in SUMMARY_KEYS if k in entry))

    print(f"[viz] Loaded {len(data)} methods from {path} "
          f"(error bars = between-seed std)")
    return data


# ══════════════════════════════════════════════════════════════════
# Colour palette
# ══════════════════════════════════════════════════════════════════

# IEEE-friendly palette (Paul Tol "bright" scheme).
# Distinct in greyscale and colourblind-safe — suitable for IEEE publications.
PALETTE = [
    "#4477AA",   # blue        — Monitor-Everything
    "#BBBBBB",   # grey        — Random-Feasible
    "#EE6677",   # red         — TopK-Anomaly-Causal
    "#228833",   # green       — RL-Flat MLP
    "#CCBB44",   # yellow      — RL-PerMetric
    "#66CCEE",   # cyan        — extra
    "#AA3377",   # purple      — extra
    "#000000",   # black       — extra
]

# Subfigure labels (LaTeX-style)
SUBFIG_LABELS = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]


# ══════════════════════════════════════════════════════════════════
# Main plot
# ══════════════════════════════════════════════════════════════════

def plot(data: dict, methods: list, out_dir: str,
         show_std: bool, figsize, dpi: int):
    """
    Three-panel grouped bar chart:
      Panel 1 — ACR (Anomaly Coverage Rate)
      Panel 2 — Mean Selected metrics per window
      Panel 3 — Mean Collection Cost
    """

    # ── IEEE-style global font settings ──────────────────────────
    # At 3.5" column width, fonts need to be larger than for screen viewing
    # so they remain legible when the PDF is included in a two-column paper.
    plt.rcParams.update({
        "font.family":       "serif",      # IEEE uses Times/serif
        "font.size":         6,
        "axes.titlesize":    6,
        "axes.labelsize":    6,
        "xtick.labelsize":   5,
        "ytick.labelsize":   5,
        "legend.fontsize":   5,
        "figure.dpi":        dpi,
        "pdf.fonttype":      42,           # embed fonts as Type 42 (TrueType)
        "ps.fonttype":       42,           # ensures IEEE PDF submission accepts it
    })

    colors = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(methods)}

    def _val(method, key):
        v = data[method].get(key, float("nan"))
        return float("nan") if np.isnan(v) else v

    def _std(method, key):
        v = data[method].get(key + "_std", float("nan"))
        return float("nan") if np.isnan(v) else v

    # ── Panel definitions ─────────────────────────────────────────
    panels = [
        {
            "key":    "acr_mean",
            "label":  "ACR",
            "ylabel": "ACR",
            "ylim":   (0.0, 1.22),   # headroom for the value label + (a) marker
            "fmt":    ".2f",      # 3 decimal places
        },
        {
            "key":    "mean_selected_mean",
            "label":  "Mean Selected Metrics",
            "ylabel": "Selected Metrics",
            "ylim":   None,
            "fmt":    ".0f",      # no decimals
        },
        {
            "key":    "mean_collection_cost_mean",
            "label":  "Mean Collection Cost",
            "ylabel": "Collection Cost",
            "ylim":   None,
            "fmt":    ".1f",      # 3 decimal places
        },
    ]

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.subplots_adjust(wspace=0.55, left=0.10, right=0.98,
                        top=0.88, bottom=0.10)

    x     = np.arange(len(methods))
    bar_w = 0.55   # narrower bars so labels have breathing room at small size

    for idx, (ax, panel) in enumerate(zip(axes, panels)):
        vals = np.array([_val(m, panel["key"]) for m in methods])
        stds = np.array([_std(m, panel["key"]) for m in methods])

        # Draw bars
        bars = ax.bar(
            x, vals,
            width     = bar_w,
            color     = [colors[m] for m in methods],
            edgecolor = "white",
            linewidth = 0.8,
            zorder    = 3,
        )

        # Error bars
        if show_std and not np.all(np.isnan(stds)):
            ax.errorbar(
                x, vals, yerr=stds,
                fmt        = "none",
                ecolor     = "black",
                elinewidth = 1.2,
                capsize    = 4,
                capthick   = 1.2,
                zorder     = 4,
            )

        # Value labels on top of each bar.  Anchored above the error bar cap,
        # not the bar top, otherwise the whisker is drawn through the text.
        for bar, v, s in zip(bars, vals, stds):
            if np.isnan(v):
                continue
            top = bar.get_height() + (0.0 if np.isnan(s) else s)
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                top + abs(bar.get_height()) * 0.03 + 1e-6,
                format(v, panel["fmt"]),
                ha="center", va="bottom",
                fontsize=4.2, fontweight="bold", color="#111111",
            )

        # ── Axes formatting ───────────────────────────────────────
        # No x-axis tick labels (method names go in legend only)
        ax.set_xticks(x)
        ax.set_xticklabels([""] * len(methods))
        ax.tick_params(axis="x", length=0)

        # y-axis label
        ax.set_ylabel(panel["ylabel"], fontsize=6)

        # Panel title (metric name only — no higher/lower text)
        ax.set_title(panel["label"], fontsize=6, fontweight="bold", pad=4)

        # Subfigure label bottom-left inside the axes
        ax.text(
            0.97, 0.97, SUBFIG_LABELS[idx],
            transform=ax.transAxes,
            fontsize=7, fontweight="bold",
            va="top", ha="right",
            color="black",
        )

        # y-axis limits
        if panel["ylim"]:
            ax.set_ylim(*panel["ylim"])
        else:
            valid = vals[~np.isnan(vals)]
            if len(valid) > 0:
                ax.set_ylim(0, valid.max() * 1.30)

        ax.grid(True, axis="y", alpha=0.35, linestyle="--", zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # ── Shared legend ─────────────────────────────────────────────
    handles = [
        mpatches.Patch(color=colors[m], label=m)
        for m in methods
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=min(len(methods), 3),   # max 3 per row to fit in 3.5"
        fontsize=5,
        framealpha=0.95,
        frameon=True,
        edgecolor="#cccccc",
        bbox_to_anchor=(0.5, - 0.04),
        handlelength=1.0,
        handleheight=0.8,
        columnspacing=0.8,
        handletextpad=0.4,
    )

    # ── Save ─────────────────────────────────────────────────────
    os.makedirs(out_dir, exist_ok=True)

    pdf_path = os.path.join(out_dir, "acr_selected_cost.pdf")
    png_path = os.path.join(out_dir, "acr_selected_cost.png")

    plt.savefig(pdf_path, dpi=dpi, bbox_inches="tight")
    print(f"[viz] Saved → {pdf_path}")

    plt.savefig(png_path, dpi=dpi, bbox_inches="tight")
    print(f"[viz] Saved → {png_path}")

    plt.close()


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # ── Resolve CSV path ──────────────────────────────────────────
    if args.summary and args.csv:
        print("[viz] ERROR: pass either --csv or --summary, not both.")
        sys.exit(1)

    if args.summary:
        csv_path = args.summary
    elif args.csv:
        csv_path = args.csv
    else:
        csv_path = os.path.join(
            "results", "metrics", "metrics_window_aggregate.csv"
        )

    if not os.path.exists(csv_path):
        print(f"[viz] ERROR: CSV not found at {csv_path}")
        print("[viz] Run compute_metrics.py first, or pass --csv <path>")
        sys.exit(1)

    # ── Output directory ──────────────────────────────────────────
    out_dir = args.out_dir or os.path.join(
        os.path.dirname(csv_path), "figures"
    )

    # ── Load ─────────────────────────────────────────────────────
    data = load_summary(csv_path) if args.summary else load_csv(csv_path)

    if not data:
        print("[viz] No data found in CSV — nothing to plot.")
        sys.exit(1)

    # ── Method order ──────────────────────────────────────────────
    if args.method_order:
        # Keep only methods that exist in data
        methods = [m for m in args.method_order if m in data]
        missing = [m for m in args.method_order if m not in data]
        if missing:
            print(f"[viz] Warning: methods not found in CSV: {missing}")
    else:
        # summary.csv is sorted alphabetically, which puts "Random-Feasible"
        # after the RL rows and scrambles the cheap->expensive reading of the
        # bars.  Fall back to the canonical order, then append anything else.
        canonical = ["Monitor-Everything", "Random-Feasible",
                     "TopK-Anomaly-Causal", "RL-Flat MLP",
                     "RL-PerMetric (Ours)"]
        methods = ([m for m in canonical if m in data]
                   + [m for m in data if m not in canonical])

    print(f"[viz] Methods in order: {methods}")

    # ── Validate required columns ─────────────────────────────────
    required = ["acr_mean", "mean_selected_mean", "mean_collection_cost_mean"]
    for m in methods:
        for col in required:
            if col not in data[m]:
                print(f"[viz] Warning: column '{col}' missing for method '{m}'")

    # ── Plot ──────────────────────────────────────────────────────
    plot(
        data         = data,
        methods      = methods,
        out_dir      = out_dir,
        show_std     = not args.no_std,
        figsize      = tuple(args.figsize),
        dpi          = args.dpi,
    )

    print(f"\n[viz] Done. Figures in {out_dir}/")


if __name__ == "__main__":
    main()