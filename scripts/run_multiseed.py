"""
run_multiseed.py
================
Runs the (method x seed) grid for one dataset setting and lays the raw per-seed
results out in the following structure:

    results/<dataset>/<method>/seed_<k>/
        config.yaml              effective config for this cell
        run_metadata.json        both seeds, device, git commit, command
        best_model.pt            RL methods only
        training_log.csv         RL methods only
        metrics.csv              this method's row of the metrics table

Nothing is aggregated here.  Aggregation is a separate, re-runnable step
(aggregate_results.py) so raw results are never overwritten by a summary.

Seed protocols
--------------
Every method at seed index k receives the *same* (data_seed, train_seed) pair,
which is what makes the comparison paired.  How the pair is formed:

  paired      (default)  data_seed = base_data + k,  train_seed = base_train + k
              Both vary. Answers "are the reported differences stable across
              scenarios and training runs?" — the headline table.

  train-only            data_seed fixed,  train_seed = base_train + k
              Training stability on one fixed scenario. The non-learning
              baselines are then constant by construction; state that, do not
              present it as a measured zero variance.

  data-only             data_seed = base_data + k,  train_seed fixed
              Sensitivity to the scenario. Comparable across all methods.

Report which protocol produced a table.  Never mix protocols in one table: the
error bars mean different things.

Parallelism
-----------
Each cell is a separate process.  Scaling is sub-linear and GPU-bound, not
linear: measured on an RTX A6000 at N=5/M=7, a solo cell runs at 565 fps while
ten concurrent cells run at ~290 fps each — 5.1x aggregate speedup, with the
GPU pinned at 99% and CPU load only ~10 of 112 cores.  Aggregate throughput
saturates around 2900 fps, so raising --max-parallel past ~10 mostly slows each
cell without increasing total throughput.

Pools are pre-generated sequentially first: concurrent cells sharing a data
seed would otherwise race to create the same pool directory.

Failures
--------
A failed cell is recorded in manifest.json with status, return code and log
path, and the grid continues.  Failures are never silently dropped;
aggregate_results.py reports them next to the summary.

Usage
-----
    # Inspect the grid without running
    python scripts/run_multiseed.py --config configs/config.yaml \\
        --dataset synthetic_1_d025 --seeds 5 --dry-run

    # Run it, 10 cells at a time
    python scripts/run_multiseed.py --config configs/config.yaml \\
        --dataset synthetic_1_d025 --seeds 5 --max-parallel 10

    # Same grid at a different anomaly density
    python scripts/run_multiseed.py --config configs/config.yaml \\
        --dataset synthetic_1_d005 --seeds 5 --max-parallel 10 \\
        --set data.anomaly_density=0.05
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO = Path(__file__).resolve().parent.parent
PY = sys.executable

# Methods requiring training.  The value is the config override for the cell,
# so each cell's config fully describes what ran.
RL_METHODS = {
    "flat":       {"policy": {"policy_type": "flat"}},
    "per_metric": {"policy": {"policy_type": "per_metric"}},
}

# Row labels emitted by compute_metrics.py -> directory names.  Baselines need
# no training but still get per-seed directories: their results depend on the
# scenario, and Random-Feasible also on the train seed.
METHOD_ROW_TO_DIR = {
    "Monitor-Everything":  "monitor_everything",
    "Random-Feasible":     "random_feasible",
    "TopK-Anomaly-Causal": "topk_anomaly_causal",
}

_print_lock = threading.Lock()


def log(msg):
    with _print_lock:
        print(msg, flush=True)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Run the (method x seed) grid")
    p.add_argument("--config",   default="configs/config.yaml")
    p.add_argument("--dataset",  required=True,
                   help="Name for this setting, e.g. synthetic_1_d025, smd. "
                        "Becomes results/<dataset>/.")
    p.add_argument("--results-root", default="results")
    p.add_argument("--methods", nargs="+", default=["flat", "per_metric"],
                   choices=sorted(RL_METHODS))
    p.add_argument("--seeds", type=int, default=5,
                   help="Number of seed indices k = 0..S-1.")
    p.add_argument("--protocol", default="paired",
                   choices=["paired", "train-only", "data-only"])
    p.add_argument("--base-data-seed",  type=int, default=42)
    p.add_argument("--base-train-seed", type=int, default=101)
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--total-timesteps", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--no-cuda", action="store_true")
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--resume", action="store_true",
                   help="Skip cells that already have metrics.csv.")
    p.add_argument("--max-parallel", type=int, default=1,
                   help="Cells to run concurrently. Scaling is GPU-bound and "
                        "sub-linear: 10 cells gave 5.1x aggregate speedup on an "
                        "A6000 (565 fps solo -> 290 fps each). Past ~10 the GPU "
                        "saturates and each cell just gets slower.")
    p.add_argument("--skip-pregenerate", action="store_true",
                   help="Do not pre-generate pools. Only valid at "
                        "--max-parallel 1.")
    p.add_argument("--method-config", nargs="*", default=[], metavar="METHOD=PATH",
                   help="Per-method config, e.g. "
                        "--method-config flat=configs/config.yaml "
                        "per_metric=configs/config_per_metric.yaml. Methods not "
                        "listed fall back to --config. All configs MUST yield "
                        "the same pool fingerprint, otherwise the methods would "
                        "be compared on different data and the run is refused.")
    p.add_argument("--set", dest="overrides", nargs="*", default=[],
                   metavar="KEY=VALUE",
                   help="Dotted-path config overrides applied to every cell, "
                        "e.g. --set data.anomaly_density=0.05. Written into "
                        "each cell's config, so nothing is implicit.")
    return p.parse_args()


def apply_overrides(cfg: dict, overrides) -> dict:
    """Apply `a.b.c=value` overrides, values parsed as YAML scalars."""
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"--set expects KEY=VALUE, got {item!r}")
        key, raw = item.split("=", 1)
        node, parts = cfg, key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return cfg


def seed_pair(protocol, k, base_data, base_train):
    if protocol == "paired":
        return base_data + k, base_train + k
    if protocol == "train-only":
        return base_data, base_train + k
    if protocol == "data-only":
        return base_data + k, base_train
    raise ValueError(protocol)


def deep_update(base: dict, overrides: dict) -> dict:
    for key, val in overrides.items():
        if isinstance(val, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], val)
        else:
            base[key] = val
    return base


def run_cmd(cmd, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as fh:
        return subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                              cwd=str(REPO), text=True).returncode


def split_metrics_csv(src: Path, dataset_dir: Path, k: int, rl_method: str):
    """
    Split compute_metrics' aggregate table into one metrics.csv per method
    directory, so every method has raw per-seed output in the same layout.

    Note: `<metric>_std` in that file is the spread ACROSS EPISODES within this
    run.  It is not the across-seed spread, which aggregate_results.py computes
    from the `<metric>_mean` values.
    """
    if not src.exists():
        return []
    written = []
    with open(src) as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        label = row.get("method", "")
        subdir = METHOD_ROW_TO_DIR.get(label)
        if subdir is None:
            if label.startswith("RL"):
                subdir = rl_method          # the model trained in this cell
            else:
                continue
        out = dataset_dir / subdir / f"seed_{k}"
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "metrics.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(row.keys()))
            w.writeheader()
            w.writerow(row)
        written.append(str(out / "metrics.csv"))
    return written


# ------------------------------------------------------------------
# One cell
# ------------------------------------------------------------------

def run_cell(cell, idx, n_total, args, base_cfg, dataset_dir):
    """Run one grid cell in its own subprocess. Returns a manifest entry."""
    k, method = cell["k"], cell["method"]
    cell_dir = dataset_dir / method / f"seed_{k}"
    cell_dir.mkdir(parents=True, exist_ok=True)
    head = f"[{idx}/{n_total}] {args.dataset}/{method}/seed_{k}"

    if args.resume and (cell_dir / "metrics.csv").exists():
        log(f"{head}  SKIP (already has metrics.csv)")
        return {**cell, "status": "skipped"}

    cfg = yaml.safe_load(yaml.safe_dump(args._method_cfgs[method]))
    deep_update(cfg, RL_METHODS[method])
    if args.total_timesteps is not None:
        cfg["ppo"]["total_timesteps"] = args.total_timesteps
    cfg["paths"]["results_dir"] = str(dataset_dir)
    cell_cfg = cell_dir / "cell_config.yaml"
    with open(cell_cfg, "w") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)

    t0 = time.time()

    train_cmd = [PY, "scripts/train.py",
                 "--config", str(cell_cfg),
                 "--run-name", str(Path(method) / f"seed_{k}"),
                 "--data-seed", str(cell["data_seed"]),
                 "--train-seed", str(cell["train_seed"])]
    if args.no_cuda:
        train_cmd.append("--no-cuda")
    elif args.device:
        train_cmd += ["--device", args.device]
    if args.deterministic:
        train_cmd.append("--deterministic")

    log(f"{head}  training...")
    rc = run_cmd(train_cmd, cell_dir / "train.log")
    if rc != 0:
        log(f"{head}  TRAIN FAILED (rc={rc}) — see {cell_dir/'train.log'}")
        return {**cell, "status": "train_failed", "returncode": rc,
                "log": str(cell_dir / "train.log")}

    model = cell_dir / "best_model.pt"
    if not model.exists():
        log(f"{head}  NO MODEL produced")
        return {**cell, "status": "no_model",
                "log": str(cell_dir / "train.log")}

    # Evaluate against the run's OWN effective config, written by
    # save_run_record with the resolved seeds — never cell_config, which
    # carries the base seed.  Using the latter makes every seed evaluate on the
    # same pool while training varies: plausible-looking, and wrong.
    effective_cfg = cell_dir / "config.yaml"
    if not effective_cfg.exists():
        log(f"{head}  NO effective config — cannot evaluate reproducibly")
        return {**cell, "status": "no_effective_config"}

    eff = yaml.safe_load(effective_cfg.read_text())
    if int(eff["data"]["seed"]) != int(cell["data_seed"]):
        log(f"{head}  SEED MISMATCH: config says {eff['data']['seed']}, "
            f"cell expects {cell['data_seed']}")
        return {**cell, "status": "seed_mismatch",
                "effective_data_seed": eff["data"]["seed"]}

    eval_cmd = [PY, "scripts/compute_metrics.py",
                "--config", str(effective_cfg),
                "--model", str(model),
                "--out-dir", str(cell_dir)]
    if args.episodes:
        eval_cmd += ["--episodes", str(args.episodes)]
    if args.no_cuda:
        eval_cmd += ["--device", "cpu"]

    rc = run_cmd(eval_cmd, cell_dir / "eval.log")
    if rc != 0:
        log(f"{head}  EVAL FAILED (rc={rc}) — see {cell_dir/'eval.log'}")
        return {**cell, "status": "eval_failed", "returncode": rc,
                "log": str(cell_dir / "eval.log")}

    written = split_metrics_csv(
        cell_dir / "metrics_window_aggregate.csv", dataset_dir, k, method)
    dt = time.time() - t0
    log(f"{head}  ok ({dt/60:.1f} min, {len(written)} method rows)")
    return {**cell, "status": "ok", "seconds": round(dt, 1),
            "rows_written": written}


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    args = parse_args()

    with open(REPO / args.config) as fh:
        base_cfg = yaml.safe_load(fh)
    apply_overrides(base_cfg, args.overrides)

    # Per-method configs, each with the same --set overrides applied.
    method_cfgs, method_cfg_paths = {}, {}
    for item in args.method_config:
        if "=" not in item:
            raise ValueError(f"--method-config expects METHOD=PATH, got {item!r}")
        meth, path = item.split("=", 1)
        with open(REPO / path) as fh:
            mc = yaml.safe_load(fh)
        apply_overrides(mc, args.overrides)
        method_cfgs[meth], method_cfg_paths[meth] = mc, path
    for meth in args.methods:
        method_cfgs.setdefault(meth, base_cfg)
        method_cfg_paths.setdefault(meth, args.config)

    # A paired comparison requires every method to see the SAME episodes.  The
    # pool fingerprint captures every generation-relevant parameter, so if two
    # method configs disagree on any of them the comparison is not paired and
    # the numbers are not comparable.  Refuse rather than produce a table that
    # looks fine and is not.
    from data.synthetic_data import pool_config_key
    keys = {m: pool_config_key(c) for m, c in method_cfgs.items()
            if m in args.methods}
    if len(set(keys.values())) > 1:
        print("\n  REFUSING: method configs would generate DIFFERENT pools, so "
              "the methods\n  would not be compared on the same data:")
        for m, k in sorted(keys.items()):
            print(f"    {m:12} {method_cfg_paths[m]:38} pool={k}")
        print("  Reconcile the generation parameters (N, M, episode_length, "
              "densities,\n  tier split, capacity ranges, costs, baseline "
              "percentiles, episode counts).\n")
        return

    args._method_cfgs = method_cfgs
    dataset_dir = Path(args.results_root) / args.dataset
    if not dataset_dir.is_absolute():
        dataset_dir = REPO / dataset_dir

    grid = []
    for k in range(args.seeds):
        d_seed, t_seed = seed_pair(args.protocol, k,
                                   args.base_data_seed, args.base_train_seed)
        for method in args.methods:
            grid.append({"k": k, "method": method,
                         "data_seed": d_seed, "train_seed": t_seed})

    print(f"\n{'='*72}")
    print(f"  Multi-seed grid — dataset='{args.dataset}'  protocol='{args.protocol}'")
    print(f"  {len(args.methods)} method(s) x {args.seeds} seed(s) = {len(grid)} training runs")
    print(f"  Parallelism : {args.max_parallel}")
    print(f"  Results root: {dataset_dir}")
    if args.overrides:
        print(f"  Overrides   : {', '.join(args.overrides)}")
    print(f"{'='*72}")
    print(f"  {'k':>2}  {'method':<12} {'data_seed':>10} {'train_seed':>11}")
    for c in grid:
        print(f"  {c['k']:>2}  {c['method']:<12} {c['data_seed']:>10} {c['train_seed']:>11}")
    if args.protocol == "train-only":
        print("\n  NOTE: data_seed is fixed, so the three baselines are constant\n"
              "        across seeds by construction. Report that explicitly.")
    print()

    if args.dry_run:
        print("  --dry-run: nothing executed.\n")
        return

    if args.skip_pregenerate and args.max_parallel > 1:
        print("  REFUSING: --skip-pregenerate with --max-parallel > 1 would "
              "race on pool creation.\n")
        return

    # Pre-generate pools sequentially so every cell is a pure reader.
    if not args.skip_pregenerate:
        from data.synthetic_data import ensure_pool
        seeds_needed = sorted({c["data_seed"] for c in grid})
        print(f"  Pre-generating {len(seeds_needed)} pool(s) for data seeds "
              f"{seeds_needed} ...")
        for ds in seeds_needed:
            ensure_pool(yaml.safe_load(yaml.safe_dump(base_cfg)), ds, "  [pool]")
        print("  Pools ready.\n")

    dataset_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = dataset_dir / "manifest.json"
    manifest = {"dataset": args.dataset, "protocol": args.protocol,
                "config": args.config, "overrides": args.overrides,
                "method_configs": method_cfg_paths,
                "max_parallel": args.max_parallel,
                "started": datetime.now().isoformat(timespec="seconds"),
                "cells": []}

    t_start = time.time()
    if args.max_parallel <= 1:
        for i, cell in enumerate(grid, 1):
            manifest["cells"].append(
                run_cell(cell, i, len(grid), args, base_cfg, dataset_dir))
            with open(manifest_path, "w") as fh:
                json.dump(manifest, fh, indent=2)
    else:
        with ThreadPoolExecutor(max_workers=args.max_parallel) as pool:
            futures = {pool.submit(run_cell, cell, i, len(grid), args,
                                   base_cfg, dataset_dir): cell
                       for i, cell in enumerate(grid, 1)}
            for fut in as_completed(futures):
                manifest["cells"].append(fut.result())
                with open(manifest_path, "w") as fh:
                    json.dump(manifest, fh, indent=2)

    manifest["finished"] = datetime.now().isoformat(timespec="seconds")
    manifest["wall_seconds"] = round(time.time() - t_start, 1)
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)

    ok = sum(c["status"] == "ok" for c in manifest["cells"])
    bad = [c for c in manifest["cells"]
           if c["status"] not in ("ok", "skipped")]
    print(f"\n{'='*72}")
    print(f"  Done: {ok}/{len(grid)} cells ok, {len(bad)} failed  "
          f"({manifest['wall_seconds']/3600:.2f} h wall)")
    for c in bad:
        print(f"    FAILED  {c['method']}/seed_{c['k']}  status={c['status']}")
    print(f"  Manifest: {manifest_path}")
    print(f"  Next: python scripts/aggregate_results.py "
          f"--input {dataset_dir.parent}")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    main()
