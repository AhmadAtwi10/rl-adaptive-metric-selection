#!/usr/bin/env python
"""
evaluate_grid.py — score an already-trained multi-seed grid, then aggregate.

Everything `run_multiseed.py` does *after* training, and nothing it does before.
No model is ever retrained: this script reads `best_model.pt` from each cell,
re-runs `compute_metrics.py`, splits the resulting table into the per-method
layout, and calls `aggregate_results.py`.

Use it when the models are already on disk and only the scoring needs redoing —
a different `--episodes`, a metric newly added to `compute_metrics.py`, a cell
whose training succeeded but whose evaluation failed, or a baseline you want to
re-measure without touching the agents.

    # everything: agents + baselines + tables
    python scripts/evaluate_grid.py --dataset syn1_d025_10M_pct70 \
        --results-root results/r2 --episodes 20

    # baselines only, leaving the RL rows as they are
    python scripts/evaluate_grid.py --dataset syn1_d025_10M_pct70 \
        --results-root results/r2 --episodes 20 --only baselines

    # tables only, from metrics.csv files that already exist
    python scripts/evaluate_grid.py --dataset syn1_d025_10M_pct70 \
        --results-root results/r2 --only-aggregate

Each cell is scored with its OWN `config.yaml`, never the repository config.
That copy is the effective config: it carries the data seed and pool fingerprint
the model trained on. Scoring against the repo config silently evaluates a model
on whatever pool that config currently resolves to.

Evaluating one cell scores the three baselines in the same pass, on the same
episodes, through the same code path as the agent. Since every method shares a
seed's data seed under all three protocols, the baseline rows are identical
across the methods of one seed; by default they are therefore computed once per
seed rather than once per cell. Pass `--baselines every` to force the redundant
work (useful only if you suspect the cells disagree about their pool).
"""

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.run_multiseed import split_metrics_csv   # noqa: E402

PY = sys.executable

# compute_metrics writes "<stem>_aggregate.csv"; stem is "metrics_<coverage_mode>".
AGG_BASENAME = "metrics_{mode}_aggregate.csv"


# ------------------------------------------------------------------
# Cell discovery
# ------------------------------------------------------------------

def discover_cells(dataset_dir: Path, methods, seeds):
    """
    Return [{method, k, dir, config, model}] for every trained cell found.

    A cell needs config.yaml to be scored at all, and best_model.pt to have its
    agent scored. Missing pieces are reported, never skipped silently.
    """
    cells, problems = [], []
    for method in methods:
        mdir = dataset_dir / method
        if not mdir.is_dir():
            problems.append(f"no directory for method '{method}' in {dataset_dir}")
            continue
        for sdir in sorted(mdir.glob("seed_*")):
            try:
                k = int(sdir.name.split("_")[1])
            except (IndexError, ValueError):
                continue
            if seeds is not None and k not in seeds:
                continue
            cfg, model = sdir / "config.yaml", sdir / "best_model.pt"
            if not cfg.exists():
                problems.append(f"{method}/seed_{k}: no config.yaml — cannot score")
                continue
            if not model.exists():
                problems.append(f"{method}/seed_{k}: no best_model.pt — training "
                                f"never finished")
                continue
            cells.append({"method": method, "k": k, "dir": sdir,
                          "config": cfg, "model": model})
    return cells, problems


def check_seed(cell):
    """Effective config's data seed, or None if unreadable."""
    try:
        eff = yaml.safe_load(cell["config"].read_text())
        return int(eff["data"]["seed"])
    except Exception:
        return None


def coverage_mode_of(cell):
    try:
        eff = yaml.safe_load(cell["config"].read_text())
        return eff.get("environment", {}).get("coverage_mode", "window")
    except Exception:
        return "window"


# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------

def build_cmd(cell, args, with_baselines, with_agent):
    cmd = [PY, "scripts/compute_metrics.py",
           "--config",  str(cell["config"]),
           "--out-dir", str(cell["dir"])]
    if with_agent:
        cmd += ["--model", str(cell["model"])]
    else:
        cmd += ["--no-agent"]
    if not with_baselines:
        cmd += ["--no-baselines"]
    if args.episodes:
        cmd += ["--episodes", str(args.episodes)]
    if args.no_cuda:
        cmd += ["--device", "cpu"]
    elif args.device:
        cmd += ["--device", args.device]
    if args.save_detail:
        cmd += ["--save-detail"]
    return cmd


def score_cell(cell, args, with_baselines, with_agent, repo_root):
    tag = f"{cell['method']}/seed_{cell['k']}"
    log_path = cell["dir"] / "eval.log"
    t0 = time.time()
    with open(log_path, "w") as fh:
        rc = subprocess.call(build_cmd(cell, args, with_baselines, with_agent),
                             cwd=repo_root, stdout=fh, stderr=subprocess.STDOUT)
    dt = (time.time() - t0) / 60.0
    if rc != 0:
        return {"tag": tag, "ok": False,
                "msg": f"compute_metrics failed (rc={rc}) — see {log_path}"}
    src = cell["dir"] / AGG_BASENAME.format(mode=coverage_mode_of(cell))
    if not src.exists():
        return {"tag": tag, "ok": False,
                "msg": f"no {src.name} produced — see {log_path}"}
    return {"tag": tag, "ok": True, "src": src, "cell": cell,
            "msg": f"scored in {dt:.1f} min"}


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Evaluate a trained multi-seed grid and aggregate. Never trains.")
    p.add_argument("--dataset", required=True,
                   help="grid directory name under --results-root")
    p.add_argument("--results-root", default="results")
    p.add_argument("--methods", nargs="+", default=["flat", "per_metric"],
                   help="RL method directories to score")
    p.add_argument("--seeds", type=int, default=None,
                   help="score seeds 0..S-1 (default: every seed_* found)")
    p.add_argument("--episodes", type=int, default=None,
                   help="evaluation episodes per cell; omit to use the whole pool")
    p.add_argument("--only", choices=["all", "agents", "baselines"], default="all",
                   help="all (default) | agents: skip baselines | "
                        "baselines: skip agents, leaving RL rows untouched")
    p.add_argument("--baselines", choices=["once", "every"], default="once",
                   help="once (default): score baselines from one cell per seed; "
                        "every: score them in every cell")
    p.add_argument("--only-aggregate", action="store_true",
                   help="skip scoring; build tables from existing metrics.csv")
    p.add_argument("--compare-to", default=None,
                   help="reference method for paired diffs, e.g. "
                        "'RL-PerMetric (Ours)'")
    p.add_argument("--metrics", nargs="+", default=None,
                   help="metric columns for the rendered tables")
    p.add_argument("--max-parallel", type=int, default=1,
                   help="concurrent cells; evaluation is short, keep this low")
    p.add_argument("--device", default=None)
    p.add_argument("--no-cuda", action="store_true")
    p.add_argument("--save-detail", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    dataset_dir = (repo_root / args.results_root / args.dataset
                   if not os.path.isabs(args.results_root)
                   else Path(args.results_root) / args.dataset)
    if not dataset_dir.is_dir():
        sys.exit(f"[eval-grid] No such grid: {dataset_dir}")

    seeds = set(range(args.seeds)) if args.seeds else None
    cells, problems = discover_cells(dataset_dir, args.methods, seeds)

    print("=" * 72)
    print(f"  Evaluate grid — {args.dataset}")
    print(f"  Directory : {dataset_dir}")
    print(f"  Cells     : {len(cells)}   mode={args.only}  "
          f"baselines={args.baselines}")
    print(f"  Episodes  : {args.episodes or 'whole pool'}")
    print("=" * 72)

    for msg in problems:
        print(f"  ! {msg}")
    if not cells and not args.only_aggregate:
        sys.exit("[eval-grid] Nothing to score.")

    # Which cell carries the baselines for each seed.  Every method of a seed
    # shares that seed's data seed, so one cell per seed suffices.
    baseline_owner = {}
    for c in cells:
        baseline_owner.setdefault(c["k"], id(c))

    plan = []
    for c in cells:
        owns = (args.baselines == "every") or (baseline_owner[c["k"]] == id(c))
        with_baselines = args.only in ("all", "baselines") and owns
        with_agent = args.only in ("all", "agents")
        if not with_baselines and not with_agent:
            continue
        plan.append((c, with_baselines, with_agent))

    if args.dry_run:
        for c, wb, wa in plan:
            parts = [x for x, on in (("agent", wa), ("baselines", wb)) if on]
            print(f"   {c['method']}/seed_{c['k']:<3} data_seed={check_seed(c)}"
                  f"  -> {'+'.join(parts)}")
        print("\n  --dry-run: nothing executed.")
        return

    failures = []
    if not args.only_aggregate:
        results = []
        if args.max_parallel > 1:
            with ThreadPoolExecutor(max_workers=args.max_parallel) as ex:
                futs = {ex.submit(score_cell, c, args, wb, wa, repo_root): c
                        for c, wb, wa in plan}
                for fut in as_completed(futs):
                    results.append(fut.result())
        else:
            for c, wb, wa in plan:
                r = score_cell(c, args, wb, wa, repo_root)
                print(f"   {'ok  ' if r['ok'] else 'FAIL'} {r['tag']}  {r['msg']}")
                results.append(r)
        if args.max_parallel > 1:
            for r in sorted(results, key=lambda x: x["tag"]):
                print(f"   {'ok  ' if r['ok'] else 'FAIL'} {r['tag']}  {r['msg']}")

        # Split each aggregate table into <method>/seed_<k>/metrics.csv
        print("\n[eval-grid] Splitting into the per-method layout...")
        for r in results:
            if not r["ok"]:
                failures.append(r["tag"])
                continue
            written = split_metrics_csv(r["src"], dataset_dir,
                                        r["cell"]["k"], r["cell"]["method"])
            print(f"   {r['tag']}: {len(written)} method row(s)")
            if not written:
                failures.append(f"{r['tag']} (no rows written)")

    # ── Aggregate ────────────────────────────────────────────────
    print("\n[eval-grid] Aggregating...")
    agg = [PY, "scripts/aggregate_results.py",
           "--input", str(dataset_dir.parent), "--dataset", args.dataset]
    if args.compare_to:
        agg += ["--compare-to", args.compare_to]
    if args.metrics:
        agg += ["--metrics"] + args.metrics
    rc = subprocess.call(agg, cwd=repo_root)

    print("\n" + "=" * 72)
    if failures:
        print(f"  {len(failures)} cell(s) did not produce results:")
        for f in failures:
            print(f"    - {f}")
        print("  The table above is built from the cells that succeeded — check")
        print("  the n column before using it.")
    else:
        print("  All planned cells scored.")
    print(f"  Tables: {dataset_dir}/table.md, table.tex, summary.csv, summary.json")
    print("=" * 72)
    sys.exit(1 if (failures or rc != 0) else 0)


if __name__ == "__main__":
    main()
