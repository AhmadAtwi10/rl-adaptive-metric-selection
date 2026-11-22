"""
visualize_scaling_curves.py
===========================
Plots training dynamics across multiple (N,M) configurations and policy types.

Three-panel figure sharing the same x-axis (training steps):
  Panel (a) — Anomaly Coverage Rate (ACR)
  Panel (b) — Mean Selected metrics per window
  Panel (c) — Mean Collection Cost per window

Each curve = one (configuration × policy) combination, e.g.:
  N=5,M=7   PerMetric
  N=5,M=7   Flat MLP
  N=10,M=15 PerMetric
  N=10,M=15 Flat MLP
  N=28,M=38 PerMetric

Reads from training_log.csv files — the same files produced by ppo.py.
Required columns: step, eval_coverage, eval_selected, eval_collection_cost
(eval_selection_rate and eval_collection_cost were added in our ppo.py fix)

Usage
-----
  # Minimal — pass log paths with labels
  python scripts/visualize_scaling_curves.py \\
      --logs  results/n5m7_flat/training_log.csv \\
              results/n5m7_pm/training_log.csv \\
              results/n10m15_flat/training_log.csv \\
              results/n10m15_pm/training_log.csv \\
              results/n28m38_pm/training_log.csv \\
      --labels "N=5,M=7 · Flat" \\
               "N=5,M=7 · PerMetric" \\
               "N=10,M=15 · Flat" \\
               "N=10,M=15 · PerMetric" \\
               "N=28,M=38 · PerMetric" \\
      --out-dir results/figures

  # Group by config (same config = same color, policy = line style)
  python scripts/visualize_scaling_curves.py \\
      --logs  results/n5m7_flat/training_log.csv \\
              results/n5m7_pm/training_log.csv \\
              results/n10m15_flat/training_log.csv \\
              results/n10m15_pm/training_log.csv \\
      --labels "N=5,M=7 · Flat" "N=5,M=7 · PerMetric" \\
               "N=10,M=15 · Flat" "N=10,M=15 · PerMetric" \\
      --group-by-config   # same config = same color, policy = linestyle
      --smooth 8
"""

import os
import sys
import argparse
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


# ══════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--logs",    nargs="+", default=[],
                   help="Paths to training_log.csv files (one per curve). "
                        "Mutually exclusive with --group.")

    # ── Seed-band mode ────────────────────────────────────────────
    p.add_argument("--seed-band", action="store_true",
                   help="Collapse each --group into one curve: the across-seed "
                        "mean, shaded with +/- 1 sample std (n-1), matching the "
                        "+/- reported by aggregate_results.py and Table I.")
    p.add_argument("--group", action="append", nargs="+", default=[],
                   metavar=("LABEL", "LOG"),
                   help="One method's seeds: --group LABEL log1 log2 ...  "
                        "Repeatable, one per method. Shell globs expand, e.g. "
                        "--group Ours results/<ds>/per_metric/seed_*/training_log.csv")
    p.add_argument("--baselines", default=None, metavar="SUMMARY_CSV",
                   help="aggregate_results.py summary.csv. Non-RL methods in it "
                        "are drawn as horizontal mean lines with a +/- std band, "
                        "since they do not train and so have no curve.")
    p.add_argument("--labels",  nargs="+", default=None,
                   help="Label for each log file. "
                        "Format 'N=5,M=7 · PerMetric' is parsed to assign "
                        "color by config and linestyle by policy automatically.")
    p.add_argument("--out-dir", default=None,
                   help="Output directory. Default: same dir as first log.")
    p.add_argument("--smooth",  type=int, default=5,
                   help="Rolling mean window for smoothing (default 5). "
                        "Set to 1 to disable.")
    p.add_argument("--group-by-config", action="store_true",
                   help="Assign color by config (N,M) and linestyle by policy. "
                        "Requires labels to contain '·' separator.")
    p.add_argument("--figsize", nargs=2, type=float, default=[7.2, 2.6],
                   metavar=("W", "H"),
                   help="Figure size in inches (default: 7.2 2.6 — full IEEE width). "
                        "With 4 panels consider 9.6 2.6.")
    p.add_argument("--no-reward", action="store_true",
                   help="Show only 3 panels (ACR, Selected, Cost) — omit reward.")
    p.add_argument("--dpi",     type=int, default=300)
    p.add_argument("--no-fill", action="store_true",
                   help="Do not shade the raw (unsmoothed) area.")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════

def load_log(path: str) -> dict:
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


def smooth(arr: np.ndarray, w: int) -> np.ndarray:
    if w <= 1 or len(arr) < w:
        return arr
    kernel = np.ones(w) / w
    padded = np.pad(arr, (w // 2, w // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[:len(arr)]


# ══════════════════════════════════════════════════════════════════
# Style assignment
# ══════════════════════════════════════════════════════════════════

# Paul Tol bright palette — IEEE friendly
CONFIG_COLORS = [
    "#4477AA",   # blue       — config 1
    "#EE6677",   # red        — config 2
    "#228833",   # green      — config 3
    "#CCBB44",   # yellow     — config 4
    "#AA3377",   # purple     — config 5
    "#66CCEE",   # cyan       — config 6
]

POLICY_STYLES = {
    "flat":       ("-",  "o"),    # solid, circle
    "permetric":  ("--", "s"),    # dashed, square
    "per_metric": ("--", "s"),
    "topk":       (":",  "^"),    # dotted, triangle
    "monitor":    ("-.", "D"),    # dash-dot, diamond
    "default":    ("-",  "o"),
}


def assign_styles(labels: list, group_by_config: bool):
    """
    Returns list of (color, linestyle, marker) per curve.

    If group_by_config=True and labels contain '·':
      - color  = by config part (text before '·')
      - style  = by policy part (text after '·')

    Otherwise:
      - each curve gets a unique color from the palette
      - all solid lines
    """
    styles = []

    if group_by_config:
        # Extract config and policy from label
        config_ids = {}
        color_idx  = 0
        for label in labels:
            if "·" in label:
                config_part  = label.split("·")[0].strip()
                policy_part  = label.split("·")[1].strip().lower()
            else:
                config_part  = label
                policy_part  = "default"

            if config_part not in config_ids:
                config_ids[config_part] = CONFIG_COLORS[color_idx % len(CONFIG_COLORS)]
                color_idx += 1

            color  = config_ids[config_part]
            # Match policy to linestyle
            ps = "default"
            for key in POLICY_STYLES:
                if key in policy_part.replace(" ", "").replace("-", ""):
                    ps = key
                    break
            ls, marker = POLICY_STYLES[ps]
            styles.append((color, ls, marker))
    else:
        # Each curve gets a unique color, all solid
        for i, _ in enumerate(labels):
            color  = CONFIG_COLORS[i % len(CONFIG_COLORS)]
            styles.append((color, "-", "o"))

    return styles


# ══════════════════════════════════════════════════════════════════
# Column resolution
# ══════════════════════════════════════════════════════════════════

# Fallback column names for each metric
COL_COVERAGE = ["eval_coverage", "mean_coverage", "acr"]
COL_SELECTED = ["eval_selected", "mean_selected", "selected"]
COL_COST     = ["eval_collection_cost", "mean_collection_cost", "collection_cost"]
COL_REWARD   = ["eval_reward", "mean_reward", "reward"]
COL_STEP     = ["step", "steps", "global_step"]


def find_col(data: dict, candidates: list):
    for c in candidates:
        if c in data and not np.all(np.isnan(data[c])):
            return c
    return None


# ══════════════════════════════════════════════════════════════════
# Across-seed statistics
# ══════════════════════════════════════════════════════════════════

def seed_stats(datas: list, cols: list):
    """
    Mean and +/- 1 std across seeds, evaluated on a common step grid.

    run_multiseed cells share an eval schedule, so the grids normally coincide
    exactly.  They are still resampled onto the grid of the seed that stops
    earliest, so a crashed or resumed cell shortens the band instead of
    silently dropping out of it or extrapolating past where it has data.

    std is the sample std (n-1), the same convention as aggregate_results.py,
    so the ribbon and the +/- in the tables mean the same thing.

    Returns (steps, mean, std, n_seeds) or None if no seed has the metric.
    """
    grids = []
    for d in datas:
        if not d:
            continue
        col = find_col(d, cols)
        if col is None:
            continue
        step_col = find_col(d, COL_STEP)
        vals = d[col]
        steps = d[step_col] if step_col else np.arange(len(vals), dtype=float)
        mask = ~np.isnan(vals) & ~np.isnan(steps)
        if mask.sum() < 2:
            continue
        grids.append((steps[mask], vals[mask]))

    if not grids:
        return None

    ref = min(grids, key=lambda g: g[0][-1])[0]
    stack = np.stack([np.interp(ref, s, v) for s, v in grids])
    std = stack.std(axis=0, ddof=1) if len(stack) > 1 else np.zeros(len(ref))
    return ref, stack.mean(axis=0), std, len(stack)


def load_baselines(path: str) -> dict:
    """
    Read aggregate_results.py's long-format summary.csv into
        {method: {metric: (mean, std)}}
    keeping only the non-learning methods — the RL rows already have curves.
    """
    out = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row["method"].startswith("RL"):
                continue
            try:
                out.setdefault(row["method"], {})[row["metric"]] = (
                    float(row["mean"]), float(row["std_between_seeds"]))
            except (ValueError, TypeError, KeyError):
                continue
    return out


# Colours for the non-learning methods, kept consistent with
# visualize_results.py so the same method is the same colour in both figures.
BASELINE_COLORS = {
    "Monitor-Everything":  "#4477AA",
    "Random-Feasible":     "#BBBBBB",
    "TopK-Anomaly-Causal": "#EE6677",
}

# Same, for the two canonical learned methods.  Without this the default
# palette hands blue to the first curve and red to the second — exactly the
# colours Monitor-Everything and TopK already own, so the RL mean line would
# be indistinguishable from a baseline reference line.
RL_LABEL_COLORS = {
    "RL-Flat MLP":         "#228833",
    "RL-PerMetric (Ours)": "#CCBB44",
}


def recolor_against_baselines(labels, styles, baselines):
    """
    Give each learned curve a colour no drawn baseline is using: the canonical
    one when the label is recognised, otherwise the next free palette entry.
    """
    taken = {BASELINE_COLORS.get(n) for n in (baselines or {})}
    free = [c for c in CONFIG_COLORS
            if c not in taken and c not in RL_LABEL_COLORS.values()]
    out, i = [], 0
    for label, (color, ls, marker) in zip(labels, styles):
        if label in RL_LABEL_COLORS:
            color = RL_LABEL_COLORS[label]
        elif color in taken:
            color = free[i % len(free)] if free else color
            i += 1
        out.append((color, ls, marker))
    return out


# ══════════════════════════════════════════════════════════════════
# Plot
# ══════════════════════════════════════════════════════════════════

def plot(logs_data, labels, styles, out_dir,
         smooth_w, fill, figsize, dpi, show_reward=True,
         seed_band=False, baselines=None):

    plt.rcParams.update({
        "font.family":    "serif",
        "font.size":      6,
        "axes.titlesize": 6.5,
        "axes.labelsize": 6,
        "xtick.labelsize":5.5,
        "ytick.labelsize":5.5,
        "legend.fontsize":5,
        "pdf.fonttype":   42,
        "ps.fonttype":    42,
    })

    n_panels = 4 if show_reward else 3
    fig, axes = plt.subplots(1, n_panels, figsize=figsize)
    fig.subplots_adjust(wspace=0.38, left=0.08, right=0.97,
                        top=0.88, bottom=0.22)

    # "sum" names the matching metric in aggregate_results.py's summary.csv,
    # used to place the non-learning baselines.  Reward has no counterpart.
    panels = [
        {"ax": axes[0], "cols": COL_REWARD,   "title": "(a)  Eval Reward",
         "ylabel": "Reward", "ylim": None, "sum": None},
        {"ax": axes[1], "cols": COL_COVERAGE, "title": "(b)  Anomaly Coverage Rate",
         "ylabel": "ACR", "ylim": (0.0, 1.05), "sum": "acr"},
        {"ax": axes[2], "cols": COL_SELECTED,  "title": "(c)  Mean Selected / Window",
         "ylabel": "Selected Metrics", "ylim": None, "sum": "mean_selected"},
        {"ax": axes[3], "cols": COL_COST,       "title": "(d)  Mean Collection Cost",
         "ylabel": "Collection Cost", "ylim": None, "sum": "mean_collection_cost"},
    ] if show_reward else [
        {"ax": axes[0], "cols": COL_COVERAGE, "title": "(a)  Anomaly Coverage Rate",
         "ylabel": "ACR", "ylim": (0.0, 1.05), "sum": "acr"},
        {"ax": axes[1], "cols": COL_SELECTED,  "title": "(b)  Mean Selected / Window",
         "ylabel": "Selected Metrics", "ylim": None, "sum": "mean_selected"},
        {"ax": axes[2], "cols": COL_COST,       "title": "(c)  Mean Collection Cost",
         "ylabel": "Collection Cost", "ylim": None, "sum": "mean_collection_cost"},
    ]

    for panel in panels:
        ax = panel["ax"]

        # ── Seed-band mode: one mean curve + std ribbon per method ──
        if seed_band:
            for datas, label, (color, ls, marker) in zip(logs_data, labels, styles):
                stats = seed_stats(datas, panel["cols"])
                if stats is None:
                    continue
                steps, mu, sd, n_seeds = stats
                s_m = steps / 1e6            # steps in millions

                mu_s = smooth(mu, smooth_w)
                lo_s = smooth(mu - sd, smooth_w)
                hi_s = smooth(mu + sd, smooth_w)

                ax.fill_between(s_m, lo_s, hi_s,
                                color=color, alpha=0.18, linewidth=0, zorder=2)
                lw = 1.8 if "ours" in label.lower() else 1.2
                ax.plot(s_m, mu_s,
                        color=color, linestyle=ls, linewidth=lw,
                        label=f"{label} (n={n_seeds})", zorder=4,
                        marker=marker, markersize=2.5,
                        markevery=max(1, len(s_m) // 8))

        # Non-learning baselines: flat over training steps by definition.
        # Drawn in both modes; a no-op when --baselines was not passed.
        for name, mets in (baselines or {}).items():
            if panel["sum"] is None or panel["sum"] not in mets:
                continue
            mval, mstd = mets[panel["sum"]]
            color = BASELINE_COLORS.get(name, "#888888")
            ax.axhline(mval, color=color, linestyle=":", linewidth=1.0,
                       label=name, zorder=3)
            if mstd > 0:
                ax.axhspan(mval - mstd, mval + mstd,
                           color=color, alpha=0.12, linewidth=0, zorder=1)

        for data, label, (color, ls, marker) in ([] if seed_band else
                                                 zip(logs_data, labels, styles)):
            # Find step column
            step_col = find_col(data, COL_STEP)
            if step_col is None:
                steps = np.arange(len(next(iter(data.values()))))
            else:
                steps = data[step_col]

            # Find metric column
            col = find_col(data, panel["cols"])
            if col is None:
                continue

            vals = data[col]
            mask = ~np.isnan(vals)
            if mask.sum() == 0:
                continue

            s = steps[mask]
            v = vals[mask]
            s_m = s / 1e6    # convert to millions

            v_smooth = smooth(v, smooth_w)
            lw = 1.8 if "PerMetric" in label or "permetric" in label.lower() else 1.2

            # Raw values as light shaded region
            if fill and smooth_w > 1:
                ax.fill_between(s_m, v, v_smooth,
                                color=color, alpha=0.10, zorder=1)

            # Smoothed line
            ax.plot(s_m, v_smooth,
                    color=color, linestyle=ls, linewidth=lw,
                    label=label, zorder=4,
                    marker=marker, markersize=2.5,
                    markevery=max(1, len(s_m) // 8))

        # Axes formatting
        ax.set_xlabel("Training steps (×10⁶)", fontsize=6)
        ax.set_ylabel(panel["ylabel"], fontsize=6)
        ax.set_title(panel["title"], fontsize=6.5,
                     fontweight="bold", pad=3)
        if panel["ylim"]:
            ax.set_ylim(*panel["ylim"])
        else:
            ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.25, linestyle="--", zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # ── Shared legend below all panels ───────────────────────────
    handles, lbls = axes[0].get_legend_handles_labels()
    seen = set(lbls)
    for ax in axes[1:]:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen:
                handles.append(h); lbls.append(l); seen.add(l)

    ncol = min(len(handles), 3)
    fig.legend(handles, lbls,
               loc="lower center",
               ncol=ncol,
               fontsize=5,
               framealpha=0.92,
               frameon=True,
               edgecolor="#cccccc",
               bbox_to_anchor=(0.5, -0.04),
               handlelength=1.6,
               handleheight=0.8,
               columnspacing=0.8,
               handletextpad=0.4)

    # ── Linestyle legend (policy) — top right of panel (a) ───────
    # Only shown when group-by-config is active (mixed linestyles)
    unique_styles = {}
    for label, (color, ls, marker) in zip(labels, styles):
        if "·" in label:
            policy = label.split("·")[1].strip()
            if policy not in unique_styles:
                unique_styles[policy] = (ls, marker)
    if len(unique_styles) > 1:
        style_handles = [
            plt.Line2D([0], [0], color="black", linestyle=ls,
                       marker=mk, markersize=4, linewidth=1.2,
                       label=pol)
            for pol, (ls, mk) in unique_styles.items()
        ]
        axes[0].legend(handles=style_handles,
                       loc="lower right",
                       fontsize=5, framealpha=0.9,
                       handlelength=1.4, title="Policy",
                       title_fontsize=5)

    # ── Save ─────────────────────────────────────────────────────
    os.makedirs(out_dir, exist_ok=True)
    stem = "seed_band_curves" if seed_band else "scaling_curves"
    pdf_path = os.path.join(out_dir, stem + ".pdf")
    png_path = os.path.join(out_dir, stem + ".png")
    plt.savefig(pdf_path, dpi=dpi, bbox_inches="tight")
    plt.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[viz] Saved → {pdf_path}")
    print(f"[viz] Saved → {png_path}")


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    if args.group and args.logs:
        print("[viz] Use either --logs or --group, not both.")
        return
    if not args.group and not args.logs:
        print("[viz] No log files specified (pass --logs or --group).")
        return

    baselines = load_baselines(args.baselines) if args.baselines else None
    if baselines:
        print(f"[viz] Baselines from {args.baselines}: "
              f"{', '.join(sorted(baselines))}")

    # ── Seed-band mode: each --group collapses to one mean curve ──
    if args.group:
        if not args.seed_band:
            print("[viz] --group implies --seed-band; enabling it.")
        labels, groups = [], []
        for spec in args.group:
            if len(spec) < 2:
                print(f"[viz] --group needs a label and at least one log: {spec}")
                return
            label, paths = spec[0], spec[1:]
            datas = []
            for path in paths:
                if not os.path.exists(path):
                    print(f"[viz] WARNING: {path} not found — skipping")
                    continue
                datas.append(load_log(path))
            if not datas:
                print(f"[viz] WARNING: no readable logs for '{label}' — skipping")
                continue
            print(f"[viz] '{label}': {len(datas)} seed(s), "
                  f"{min(len(next(iter(d.values()), [])) for d in datas)}"
                  f"–{max(len(next(iter(d.values()), [])) for d in datas)} checkpoints")
            labels.append(label)
            groups.append(datas)

        if not groups:
            print("[viz] Nothing to plot.")
            return

        out_dir = args.out_dir or os.path.join(
            os.path.dirname(args.group[0][1]), "figures")
        styles = assign_styles(labels, args.group_by_config)
        if not args.group_by_config:
            styles = recolor_against_baselines(labels, styles, baselines)

        print(f"\n[viz] Seed-band mode: {len(groups)} method(s), "
              f"band = mean +/- 1 std across seeds, smooth={args.smooth}")
        print(f"[viz] Output → {out_dir}/seed_band_curves.pdf\n")

        plot(
            logs_data   = groups,
            labels      = labels,
            styles      = styles,
            out_dir     = out_dir,
            smooth_w    = args.smooth,
            fill        = not args.no_fill,
            figsize     = tuple(args.figsize),
            dpi         = args.dpi,
            show_reward = not args.no_reward,
            seed_band   = True,
            baselines   = baselines,
        )
        return

    # Resolve labels
    if args.labels:
        labels = args.labels
        if len(labels) < len(args.logs):
            labels += [Path(p).parent.name for p in args.logs[len(labels):]]
    else:
        labels = [Path(p).parent.name for p in args.logs]

    if len(labels) != len(args.logs):
        print(f"[viz] Warning: {len(args.logs)} logs but {len(labels)} labels.")
        labels = labels[:len(args.logs)]

    # Output directory
    out_dir = args.out_dir or os.path.join(
        os.path.dirname(args.logs[0]), "figures"
    )

    # Load all logs
    logs_data = []
    for i, (path, label) in enumerate(zip(args.logs, labels)):
        if not os.path.exists(path):
            print(f"[viz] WARNING: {path} not found — skipping '{label}'")
            logs_data.append({})
            continue
        data = load_log(path)
        n = len(next(iter(data.values()), []))
        # Report which columns were found
        cov  = find_col(data, COL_COVERAGE) or "MISSING"
        sel  = find_col(data, COL_SELECTED)  or "MISSING"
        cost = find_col(data, COL_COST)       or "MISSING"
        rew  = find_col(data, COL_REWARD)     or "MISSING"
        print(f"[viz] '{label}': {n} checkpoints  "
              f"Reward={rew}  ACR={cov}  Selected={sel}  Cost={cost}")
        logs_data.append(data)

    # Assign visual styles
    styles = assign_styles(labels, args.group_by_config)

    print(f"\n[viz] Plotting {len(args.logs)} curves  smooth={args.smooth}")
    print(f"[viz] Output → {out_dir}/scaling_curves.pdf\n")

    plot(
        logs_data  = logs_data,
        labels     = labels,
        styles     = styles,
        out_dir    = out_dir,
        smooth_w   = args.smooth,
        fill       = not args.no_fill,
        figsize    = tuple(args.figsize),
        dpi        = args.dpi,
        show_reward= not args.no_reward,
    )

    print("[viz] Done.")


if __name__ == "__main__":
    main()