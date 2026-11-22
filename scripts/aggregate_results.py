"""
aggregate_results.py
====================
Aggregates the raw per-seed results produced by run_multiseed.py into
mean +/- std tables, and reports which cells failed.

Reads the per-seed layout:

    results/<dataset>/<method>/seed_<k>/metrics.csv

and writes, per dataset:

    results/<dataset>/summary.csv     one row per (method, metric): mean, std, n, ci95
    results/<dataset>/summary.json    same, machine-readable
    results/<dataset>/table.md        headline table, ready to read
    results/<dataset>/table.tex       LaTeX booktabs body for the paper

Two different standard deviations
---------------------------------
compute_metrics.py already emits `<metric>_std` — that is the spread ACROSS
EPISODES inside a single run, which is not what is reported.  This script ignores
those columns and computes the spread ACROSS SEEDS from the `<metric>_mean`
values, one per seed.  The output column `std` therefore always means
"between-seed", and `n` is the number of seeds that contributed.

Never reported as a seed count: a cell that failed.  Failed cells are read from
manifest.json and printed, so a table can never quietly rest on 3 seeds while
claiming 5.

Usage
-----
    python scripts/aggregate_results.py --input results/
    python scripts/aggregate_results.py --input results/ --dataset synthetic_1
    python scripts/aggregate_results.py --input results/ \\
        --metrics acr fmr f1_coverage tier_cost mean_selected \\
        --compare-to "RL-PPO (Ours)"
"""

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Headline metrics, in the order the paper reports them.  Anything else present
# in the CSVs is still aggregated into summary.csv; this list only controls the
# rendered tables.
DEFAULT_METRICS = ["acr", "fmr", "f1_coverage", "bur", "cvr",
                   "tier_cost", "mean_selected"]

# Lower-is-better metrics, for arrow annotations in the rendered tables.
LOWER_IS_BETTER = {"fmr", "cvr", "tier_cost", "mean_selected", "cost",
                   "latency_mean", "latency_median", "latency_p90",
                   "selection_rate", "bur"}

DIR_TO_LABEL = {
    "monitor_everything":  "Monitor-Everything",
    "random_feasible":     "Random-Feasible",
    "topk_anomaly_causal": "TopK-Anomaly-Causal",
    "flat":                "RL-Flat MLP",
    "per_metric":          "RL-PerMetric (Ours)",
}


def parse_args():
    p = argparse.ArgumentParser(description="Aggregate per-seed results")
    p.add_argument("--input", default="results",
                   help="Results root containing <dataset>/ directories.")
    p.add_argument("--dataset", default=None,
                   help="Aggregate only this dataset. Default: all found.")
    p.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS,
                   help="Metrics to render in the tables.")
    p.add_argument("--compare-to", default=None,
                   help="Method label to use as the reference in a paired "
                        "comparison (e.g. \"RL-PerMetric (Ours)\").")
    p.add_argument("--min-seeds", type=int, default=2,
                   help="Warn when a cell has fewer than this many seeds.")
    return p.parse_args()


# ------------------------------------------------------------------
# Statistics
# ------------------------------------------------------------------

def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def stdev(xs):
    """Sample standard deviation (n-1). Zero for a single observation."""
    n = len(xs)
    if n < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


def ci95(xs):
    """Normal-approximation half-width. With n<5 this is indicative only."""
    n = len(xs)
    if n < 2:
        return float("nan")
    return 1.96 * stdev(xs) / math.sqrt(n)


def paired_diff(a: dict, b: dict):
    """
    Paired per-seed difference a - b over the seeds both methods share.

    Paired because run_multiseed gives every method the same (data_seed,
    train_seed) at each seed index, so seed k is the same scenario for both.
    Returns (n_pairs, mean_diff, std_diff, cohens_dz) or None if <2 pairs.
    """
    shared = sorted(set(a) & set(b))
    if len(shared) < 2:
        return None
    diffs = [a[k] - b[k] for k in shared]
    sd = stdev(diffs)
    dz = mean(diffs) / sd if sd > 0 else float("nan")
    return len(shared), mean(diffs), sd, dz


# ------------------------------------------------------------------
# Loading
# ------------------------------------------------------------------

def load_dataset(dataset_dir: Path):
    """
    Returns values[method_label][metric][seed_k] = float

    Only `<metric>_mean` columns are read; `<metric>_std` is per-episode spread
    and is deliberately discarded (see module docstring).
    """
    values = defaultdict(lambda: defaultdict(dict))

    for method_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
        label = DIR_TO_LABEL.get(method_dir.name, method_dir.name)
        for seed_dir in sorted(method_dir.glob("seed_*")):
            csv_path = seed_dir / "metrics.csv"
            if not csv_path.exists():
                continue
            try:
                k = int(seed_dir.name.split("_")[1])
            except (IndexError, ValueError):
                continue
            with open(csv_path) as f:
                rows = list(csv.DictReader(f))
            if not rows:
                continue
            row = rows[0]
            for col, raw in row.items():
                if not col.endswith("_mean"):
                    continue
                metric = col[: -len("_mean")]
                try:
                    values[label][metric][k] = float(raw)
                except (TypeError, ValueError):
                    continue
    return values


def load_failures(dataset_dir: Path):
    mpath = dataset_dir / "manifest.json"
    if not mpath.exists():
        return [], None
    with open(mpath) as f:
        man = json.load(f)
    bad = [c for c in man.get("cells", [])
           if c.get("status") not in ("ok", "skipped")]
    return bad, man.get("protocol")


# ------------------------------------------------------------------
# Rendering
# ------------------------------------------------------------------

def fmt(m, s, n):
    if n == 0:
        return "—"
    if n == 1:
        return f"{m:.4f} (n=1)"
    return f"{m:.4f} ± {s:.4f}"


def render_markdown(dataset, values, metrics, protocol, failures, min_seeds):
    lines = [f"# {dataset} — mean ± std across seeds", ""]
    if protocol:
        lines += [f"Seed protocol: **{protocol}**.", ""]
        if protocol == "train-only":
            lines += ["> `data_seed` is fixed, so the non-learning baselines are "
                      "constant across seeds by construction; their ± is 0 by "
                      "design, not by measurement.", ""]
    lines += ["All ± values are the **between-seed** standard deviation. "
              "`n` is the number of seeds contributing.", ""]

    header = ["Method", "n"] + [
        m + (" ↓" if m in LOWER_IS_BETTER else " ↑") for m in metrics]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))

    for label in sorted(values):
        per_metric = values[label]
        ns = {len(per_metric.get(m, {})) for m in metrics if m in per_metric}
        n = max(ns) if ns else 0
        cells = []
        for m in metrics:
            xs = list(per_metric.get(m, {}).values())
            cells.append(fmt(mean(xs), stdev(xs), len(xs)) if xs else "—")
        lines.append("| " + " | ".join([label, str(n)] + cells) + " |")

    warn = [f"{lab} ({max((len(values[lab].get(m, {})) for m in metrics), default=0)} seeds)"
            for lab in values
            if max((len(values[lab].get(m, {})) for m in metrics), default=0) < min_seeds]
    if warn:
        lines += ["", f"> **Under {min_seeds} seeds:** " + ", ".join(warn)]
    if failures:
        lines += ["", "> **Failed cells (excluded from the table):**"]
        lines += [f"> - {c.get('method')}/seed_{c.get('k')} — {c.get('status')}"
                  for c in failures]
    return "\n".join(lines) + "\n"


def render_latex(values, metrics):
    out = []
    for label in sorted(values):
        cells = []
        for m in metrics:
            xs = list(values[label].get(m, {}).values())
            cells.append("--" if not xs
                         else f"${mean(xs):.3f} \\pm {stdev(xs):.3f}$")
        out.append(f"{label} & " + " & ".join(cells) + r" \\")
    return "\n".join(out) + "\n"


def main():
    args = parse_args()
    root = Path(args.input)
    if not root.is_absolute():
        root = Path(__file__).resolve().parent.parent / root

    datasets = ([root / args.dataset] if args.dataset
                else [p for p in sorted(root.iterdir())
                      if p.is_dir() and any(p.glob("*/seed_*/metrics.csv"))])
    if not datasets:
        print(f"[agg] No per-seed results found under {root}")
        print("[agg] Expected layout: <dataset>/<method>/seed_<k>/metrics.csv")
        return

    for ds_dir in datasets:
        values = load_dataset(ds_dir)
        if not values:
            print(f"[agg] {ds_dir.name}: nothing to aggregate")
            continue
        failures, protocol = load_failures(ds_dir)

        metrics = [m for m in args.metrics
                   if any(m in v for v in values.values())]
        missing = [m for m in args.metrics if m not in metrics]

        # summary.csv / summary.json — every metric found, not just headline ones
        all_metrics = sorted({m for v in values.values() for m in v})
        rows, blob = [], {}
        for label in sorted(values):
            blob[label] = {}
            for m in all_metrics:
                per_seed = values[label].get(m, {})
                xs = list(per_seed.values())
                if not xs:
                    continue
                rec = {"method": label, "metric": m, "n_seeds": len(xs),
                       "mean": mean(xs), "std_between_seeds": stdev(xs),
                       "ci95_halfwidth": ci95(xs),
                       "min": min(xs), "max": max(xs),
                       "per_seed": json.dumps(
                           {str(k): per_seed[k] for k in sorted(per_seed)})}
                rows.append(rec)
                blob[label][m] = {k: v for k, v in rec.items()
                                  if k not in ("method", "metric", "per_seed")}
                blob[label][m]["per_seed"] = {str(k): per_seed[k]
                                              for k in sorted(per_seed)}

        with open(ds_dir / "summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        with open(ds_dir / "summary.json", "w") as f:
            json.dump({"dataset": ds_dir.name, "protocol": protocol,
                       "failures": failures, "results": blob}, f, indent=2)

        md = render_markdown(ds_dir.name, values, metrics, protocol,
                             failures, args.min_seeds)
        (ds_dir / "table.md").write_text(md)
        (ds_dir / "table.tex").write_text(render_latex(values, metrics))

        print(md)
        if missing:
            print(f"[agg] NOTE: requested metrics not found in the CSVs: "
                  f"{', '.join(missing)}")

        # Paired comparison against a reference method
        if args.compare_to:
            ref = args.compare_to
            if ref not in values:
                print(f"[agg] --compare-to '{ref}' not found. "
                      f"Available: {', '.join(sorted(values))}")
            else:
                print(f"## Paired difference vs {ref}  (per-seed, same scenarios)\n")
                print(f"{'method':<26}{'metric':<16}{'n':>3}"
                      f"{'mean diff':>12}{'std':>10}{'dz':>8}")
                for label in sorted(values):
                    if label == ref:
                        continue
                    for m in metrics:
                        res = paired_diff(values[label].get(m, {}),
                                          values[ref].get(m, {}))
                        if res is None:
                            continue
                        n, md_, sd_, dz = res
                        print(f"{label:<26}{m:<16}{n:>3}"
                              f"{md_:>12.4f}{sd_:>10.4f}{dz:>8.2f}")
                print("\n  dz = Cohen's d for paired samples (mean diff / std of "
                      "diffs).\n  With n<5 treat it as indicative, not "
                      "inferential.\n")

        print(f"[agg] {ds_dir.name}: wrote summary.csv, summary.json, "
              f"table.md, table.tex")
        if failures:
            print(f"[agg] {ds_dir.name}: {len(failures)} FAILED cell(s) "
                  f"excluded — see manifest.json")


if __name__ == "__main__":
    main()
