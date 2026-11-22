"""
validate_scores.py
==================
Validates whether LSTM-AD anomaly scores are meaningful — i.e. whether
the algorithm assigned higher scores to anomalous timesteps than normal ones.

What this script checks
-----------------------
  For each metric of each machine:

  1. Score separation
     Mean score on anomalous timesteps vs mean score on normal timesteps.
     A good detector should have: mean_anomaly >> mean_normal.

  2. AUROC (Area Under ROC Curve)
     Threshold-free measure of ranking quality.
     AUROC = 1.0  → perfect separation
     AUROC = 0.5  → random (no signal)
     AUROC < 0.5  → worse than random (inverted scores)
     Target: AUROC > 0.7 for a useful detector.

  3. Average Precision (AP)
     Like AUROC but accounts for class imbalance (anomalies are rare).
     More meaningful than AUROC when anomaly rate is very low (<5%).

  4. Score distribution
     Min, max, mean, std of scores — to catch degenerate outputs
     (e.g. all-zero scores or all-one scores).

Usage
-----
  # Validate all machines found in results dir
  python scripts/validate_scores.py \\
      --results-dir data/timeeval_results/smd \\
      --input-dir   data/timeeval_input/smd \\
      --plot

  # Validate specific machine
  python scripts/validate_scores.py \\
      --results-dir data/timeeval_results/smd \\
      --input-dir   data/timeeval_input/smd \\
      --machines    machine-1-1 \\
      --plot

  # Use raw scores instead of normalized
  python scripts/validate_scores.py \\
      --results-dir data/timeeval_results/smd \\
      --input-dir   data/timeeval_input/smd \\
      --score-file  anomaly_scores_raw.ts
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ══════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Validate LSTM-AD anomaly scores against ground truth labels"
    )
    p.add_argument("--results-dir", default="data/timeeval_results/smd",
                   help="TimeEval results directory (contains LSTM-AD/<dataset>/1/*.ts)")
    p.add_argument("--input-dir",   default="data/timeeval_input/smd",
                   help="TimeEval input directory (contains per-metric train/test CSVs)")
    p.add_argument("--machines",    nargs="+", default=None,
                   help="Machines to validate. Default: all found")
    p.add_argument("--score-file",  default="anomaly_scores.ts",
                   help="Score filename to use (default: anomaly_scores.ts = normalized)")
    p.add_argument("--plot",        action="store_true",
                   help="Save validation plots per machine")
    p.add_argument("--out-dir",     default="results/score_validation",
                   help="Where to save plots and validation CSV")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════

def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute AUROC without sklearn."""
    if labels.sum() == 0 or labels.sum() == len(labels):
        return float("nan")
    # Sort by score descending
    order  = np.argsort(scores)[::-1]
    labels = labels[order]
    n_pos  = labels.sum()
    n_neg  = len(labels) - n_pos
    tp = np.cumsum(labels)
    fp = np.cumsum(1 - labels)
    tpr = tp / n_pos
    fpr = fp / n_neg
    # Trapezoidal AUC
    return float(np.trapz(tpr, fpr))


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute Average Precision (AP) without sklearn."""
    if labels.sum() == 0:
        return float("nan")
    order     = np.argsort(scores)[::-1]
    labels    = labels[order]
    n_pos     = labels.sum()
    tp        = np.cumsum(labels)
    precision = tp / np.arange(1, len(labels) + 1)
    recall    = tp / n_pos
    # AP = sum of precision * delta_recall
    recall_diff = np.diff(np.concatenate([[0], recall]))
    return float(np.sum(precision * recall_diff))


def score_separation(scores: np.ndarray, labels: np.ndarray):
    """Mean score on anomalous vs normal timesteps."""
    if labels.sum() == 0:
        return float("nan"), float(scores.mean())
    mean_anom   = float(scores[labels == 1].mean())
    mean_normal = float(scores[labels == 0].mean())
    return mean_anom, mean_normal


# ══════════════════════════════════════════════════════════════════
# Score and label loading
# ══════════════════════════════════════════════════════════════════

def find_score_file(results_dir: Path, machine: str, metric: int,
                    score_filename: str) -> Path:
    """
    Searches for the score file for a given machine/metric.
    TimeEval stores results under a hash-based subdirectory:
      results_dir/<timestamp>/LSTM-AD/<hash>/SMD/<dataset_name>/1/<score_filename>
    or:
      results_dir/LSTM-AD/<dataset_name>/1/<score_filename>
    """
    ds_name = f"{machine}_metric_{metric:02d}"

    # Search recursively — handles both path structures
    candidates = sorted(results_dir.rglob(f"*{ds_name}*/{score_filename}"))
    if candidates:
        # Use most recent (last alphabetically by timestamp folder)
        return candidates[-1]
    return None


def load_labels(input_dir: Path, machine: str, metric: int) -> np.ndarray:
    """Load per-metric labels from test CSV."""
    test_csv = input_dir / machine / f"metric_{metric:02d}" / "test.csv"
    if not test_csv.exists():
        return None
    df = pd.read_csv(test_csv)
    if "is_anomaly" not in df.columns:
        return None
    return df["is_anomaly"].values.astype(np.int32)


def load_scores(score_path: Path, T_expected: int,
                window_size: int = 30) -> np.ndarray:
    """
    Load and align scores from a .ts file.
    LSTM-AD may produce fewer scores than T_expected due to windowing.
    Pad front with zeros if needed.
    """
    scores = np.loadtxt(score_path, dtype=np.float32)
    if len(scores) < T_expected:
        pad    = T_expected - len(scores)
        scores = np.concatenate([np.zeros(pad, dtype=np.float32), scores])
    elif len(scores) > T_expected:
        scores = scores[:T_expected]
    return scores


# ══════════════════════════════════════════════════════════════════
# Per-machine validation
# ══════════════════════════════════════════════════════════════════

def validate_machine(
    machine:      str,
    results_dir:  Path,
    input_dir:    Path,
    score_filename: str,
    n_metrics:    int = 38,
) -> pd.DataFrame:
    """
    Validates scores for all metrics of one machine.
    Returns a DataFrame with one row per metric.
    """
    rows = []

    for m in range(n_metrics):
        # Load labels
        labels = load_labels(input_dir, machine, m)
        if labels is None:
            continue
        T = len(labels)

        # Find score file
        score_path = find_score_file(results_dir, machine, m, score_filename)
        if score_path is None:
            rows.append({
                "machine": machine, "metric": m,
                "status": "MISSING", "auroc": float("nan"),
                "ap": float("nan"), "mean_anom": float("nan"),
                "mean_normal": float("nan"), "gap": float("nan"),
                "anom_rate": float(labels.mean()),
                "score_min": float("nan"), "score_max": float("nan"),
                "score_std": float("nan"),
            })
            continue

        scores = load_scores(score_path, T)

        # Compute metrics
        auc        = auroc(scores, labels)
        ap         = average_precision(scores, labels)
        mean_anom, mean_normal = score_separation(scores, labels)
        gap        = (mean_anom - mean_normal
                      if not (np.isnan(mean_anom) or np.isnan(mean_normal))
                      else float("nan"))

        # Quality flag
        if np.isnan(auc):
            status = "NO_ANOMALY"   # metric has no anomalous timesteps
        elif auc >= 0.7:
            status = "GOOD"
        elif auc >= 0.5:
            status = "WEAK"
        else:
            status = "POOR"         # worse than random

        rows.append({
            "machine":     machine,
            "metric":      m,
            "status":      status,
            "auroc":       round(auc,  4),
            "ap":          round(ap,   4),
            "mean_anom":   round(mean_anom,   4) if not np.isnan(mean_anom)   else float("nan"),
            "mean_normal": round(mean_normal, 4) if not np.isnan(mean_normal) else float("nan"),
            "gap":         round(gap,  4) if not np.isnan(gap) else float("nan"),
            "anom_rate":   round(float(labels.mean()), 4),
            "score_min":   round(float(scores.min()), 4),
            "score_max":   round(float(scores.max()), 4),
            "score_std":   round(float(scores.std()), 4),
        })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════

def plot_machine(machine: str, df: pd.DataFrame,
                 results_dir: Path, input_dir: Path,
                 score_filename: str, out_dir: Path):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("    [plot] pip install matplotlib")
        return

    # ── Plot 1: AUROC bar chart per metric ───────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(16, 8))
    fig.suptitle(f"{machine} — Score Quality Validation", fontsize=13)

    ax = axes[0]
    metrics_with_anom = df[df["status"] != "NO_ANOMALY"]
    colors = {"GOOD": "#4CAF50", "WEAK": "#FF9800", "POOR": "#F44336",
              "MISSING": "#9E9E9E"}

    if not metrics_with_anom.empty:
        bar_colors = [colors.get(s, "#9E9E9E") for s in metrics_with_anom["status"]]
        ax.bar(metrics_with_anom["metric"], metrics_with_anom["auroc"],
               color=bar_colors, alpha=0.85)
        ax.axhline(0.7,  color="#4CAF50", linestyle="--", linewidth=1,
                   label="AUROC=0.7 (good threshold)")
        ax.axhline(0.5,  color="#F44336", linestyle="--", linewidth=1,
                   label="AUROC=0.5 (random)")
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Metric index")
        ax.set_ylabel("AUROC")
        ax.set_title("AUROC per metric (only metrics with anomalies shown)")
        patches = [mpatches.Patch(color=c, label=l)
                   for l, c in colors.items() if l != "MISSING"]
        ax.legend(handles=patches + ax.get_legend_handles_labels()[0][2:],
                  fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.3, axis="y")

    # ── Plot 2: Score gap bar chart ──────────────────────────────
    ax = axes[1]
    if not metrics_with_anom.empty:
        gap_colors = ["#4CAF50" if g > 0 else "#F44336"
                      for g in metrics_with_anom["gap"].fillna(0)]
        ax.bar(metrics_with_anom["metric"], metrics_with_anom["gap"],
               color=gap_colors, alpha=0.85)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Metric index")
        ax.set_ylabel("Score gap (anomaly mean − normal mean)")
        ax.set_title("Score gap per metric (positive = correct direction)")
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = out_dir / f"{machine}_auroc.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] {path}")

    # ── Plot 3: Score timeline for best and worst metric ─────────
    ranked = metrics_with_anom.dropna(subset=["auroc"]).sort_values("auroc")
    if len(ranked) < 1:
        return

    best_m  = int(ranked.iloc[-1]["metric"])
    worst_m = int(ranked.iloc[0]["metric"])

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    fig.suptitle(f"{machine} — Score Timeline: Best vs Worst Metric", fontsize=13)

    for ax, m, label in zip(axes,
                             [best_m, worst_m],
                             [f"metric_{best_m:02d} (AUROC={ranked.iloc[-1]['auroc']:.3f} — best)",
                              f"metric_{worst_m:02d} (AUROC={ranked.iloc[0]['auroc']:.3f} — worst)"]):
        labels_arr = load_labels(input_dir, machine, m)
        score_path = find_score_file(results_dir, machine, m, score_filename)
        if labels_arr is None or score_path is None:
            continue
        scores_arr = load_scores(score_path, len(labels_arr))
        T          = len(scores_arr)

        ax.plot(scores_arr, color="#2196F3", linewidth=0.8, label="score")

        # Shade anomaly regions
        changes = np.diff(np.concatenate([[0], labels_arr, [0]]))
        for s, e in zip(np.where(changes == 1)[0],
                        np.where(changes == -1)[0]):
            ax.axvspan(s, e, alpha=0.25, color="red", linewidth=0)

        ax.set_ylabel("Anomaly score")
        ax.set_title(label)
        ax.grid(True, alpha=0.3)
        anom_patch = mpatches.Patch(color="red", alpha=0.3, label="Anomaly")
        ax.legend(handles=[ax.get_legend_handles_labels()[0][0], anom_patch],
                  fontsize=8)

    axes[-1].set_xlabel("Timestep")
    plt.tight_layout()
    path = out_dir / f"{machine}_timeline.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] {path}")


# ══════════════════════════════════════════════════════════════════
# Summary printer
# ══════════════════════════════════════════════════════════════════

def print_summary(machine: str, df: pd.DataFrame):
    with_anom = df[df["status"] != "NO_ANOMALY"]
    n_total   = len(df)
    n_good    = (df["status"] == "GOOD").sum()
    n_weak    = (df["status"] == "WEAK").sum()
    n_poor    = (df["status"] == "POOR").sum()
    n_no_anom = (df["status"] == "NO_ANOMALY").sum()
    n_missing = (df["status"] == "MISSING").sum()

    mean_auroc = with_anom["auroc"].mean()
    mean_gap   = with_anom["gap"].mean()

    print(f"\n  {'─'*58}")
    print(f"  {machine}")
    print(f"  {'─'*58}")
    print(f"  Metrics total      : {n_total}")
    print(f"  With anomalies     : {len(with_anom)}  "
          f"(no-anomaly metrics: {n_no_anom})")
    print(f"  Missing scores     : {n_missing}")
    print(f"")
    print(f"  AUROC quality breakdown (metrics with anomalies):")
    print(f"    ✓ GOOD  (≥0.70) : {n_good}")
    print(f"    ~ WEAK  (≥0.50) : {n_weak}")
    print(f"    ✗ POOR  (<0.50) : {n_poor}")
    print(f"")
    print(f"  Mean AUROC         : {mean_auroc:.4f}")
    print(f"  Mean score gap     : {mean_gap:.4f}  "
          f"({'✓ positive' if mean_gap > 0 else '✗ negative — scores inverted'})")

    # Per-metric table for metrics with anomalies
    if not with_anom.empty:
        print(f"\n  {'Metric':>8} {'Status':>10} {'AUROC':>7} "
              f"{'AP':>7} {'Gap':>8} {'AномRate':>9}")
        print(f"  {'─'*56}")
        for _, row in with_anom.sort_values("auroc", ascending=False).iterrows():
            flag = ("✓" if row["status"] == "GOOD" else
                    "~" if row["status"] == "WEAK" else "✗")
            print(f"  metric_{int(row['metric']):02d}  "
                  f"{flag} {row['status']:>8}  "
                  f"{row['auroc']:>7.4f}  "
                  f"{row['ap']:>7.4f}  "
                  f"{row['gap']:>8.4f}  "
                  f"{row['anom_rate']:>8.4f}")


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

def main():
    args        = parse_args()
    results_dir = Path(args.results_dir)
    input_dir   = Path(args.input_dir)
    out_dir     = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Discover machines
    if args.machines:
        machines = args.machines
    else:
        machines = sorted(
            d.name for d in input_dir.iterdir()
            if d.is_dir() and d.name.startswith("machine")
        )

    if not machines:
        print(f"[ERROR] No machine folders found in {input_dir}")
        sys.exit(1)

    print(f"\n{'═'*60}")
    print(f"  Score Quality Validation")
    print(f"{'═'*60}")
    print(f"  Results dir  : {results_dir}")
    print(f"  Input dir    : {input_dir}")
    print(f"  Score file   : {args.score_file}")
    print(f"  Machines     : {machines}")

    all_dfs = []

    for machine in machines:
        # Auto-detect n_metrics from input dir
        metric_dirs = sorted((input_dir / machine).glob("metric_*"))
        n_metrics   = len(metric_dirs)
        if n_metrics == 0:
            print(f"\n  [!] {machine}: no metric folders found — skipping")
            continue

        print(f"\n  Validating {machine} ({n_metrics} metrics)...")

        df = validate_machine(
            machine       = machine,
            results_dir   = results_dir,
            input_dir     = input_dir,
            score_filename = args.score_file,
            n_metrics     = n_metrics,
        )
        print_summary(machine, df)
        all_dfs.append(df)

        if args.plot and not df.empty:
            plot_machine(machine, df, results_dir, input_dir,
                         args.score_file, out_dir)

    # Save combined CSV
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        csv_path = out_dir / "validation_results.csv"
        combined.to_csv(csv_path, index=False)
        print(f"\n{'═'*60}")
        print(f"  Overall summary across all machines")
        print(f"{'═'*60}")
        with_anom = combined[combined["status"] != "NO_ANOMALY"]
        print(f"  Total metrics validated : {len(combined)}")
        print(f"  Metrics with anomalies  : {len(with_anom)}")
        print(f"  Overall mean AUROC      : {with_anom['auroc'].mean():.4f}")
        print(f"  Overall mean score gap  : {with_anom['gap'].mean():.4f}")
        n_good = (combined["status"] == "GOOD").sum()
        n_weak = (combined["status"] == "WEAK").sum()
        n_poor = (combined["status"] == "POOR").sum()
        print(f"  GOOD (≥0.70) : {n_good}  "
              f"WEAK (≥0.50) : {n_weak}  "
              f"POOR (<0.50) : {n_poor}")
        print(f"\n  Results saved → {csv_path}")

        # Overall verdict
        mean_auroc = with_anom["auroc"].mean()
        print(f"\n  Verdict: ", end="")
        if mean_auroc >= 0.7:
            print(f"✓ Scores are reliable (mean AUROC={mean_auroc:.3f})")
            print(f"  Safe to use as anomaly_scores input to the RL agent.")
        elif mean_auroc >= 0.5:
            print(f"~ Scores have weak signal (mean AUROC={mean_auroc:.3f})")
            print(f"  Consider: more epochs, different window_size, or check data quality.")
        else:
            print(f"✗ Scores are unreliable (mean AUROC={mean_auroc:.3f})")
            print(f"  Do NOT use as RL input. Try: more epochs or different algorithm.")


if __name__ == "__main__":
    main()
elif __name__ == "scripts.validate_scores":
    main()