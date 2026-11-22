"""
run_lstm_ad.py
==============
STEP 2 of the TimeEval pipeline.

Runs LSTM-AD via TimeEval on all per-metric CSVs produced by
convert_to_timeeval.py and saves the raw anomaly scores.

Prerequisites
-------------
  1. convert_to_timeeval.py has been run successfully
     → data/timeeval_input/ exists with metric CSVs + datasets.csv

  2. Docker is running:
     docker ps

  3. LSTM-AD image is pulled:
     docker pull ghcr.io/timeeval/lstm_ad:0.3.0

  4. TimeEval Python package is installed:
     pip install timeeval

What LSTM-AD does
-----------------
  For each metric series:
    TRAIN phase : trains an LSTM on train.csv to learn normal patterns
    SCORE phase : runs on test.csv, outputs squared prediction error per timestep
                  → saved as anomaly_scores.ts (one float per line)

  The score is the raw MSE — NOT normalised. Higher = more anomalous.
  Normalisation to [0,1] happens in the next step (extract_scores.py).

Output
------
  data/timeeval_results/
    LSTM-AD/
      machine-1-1_metric_00/
        1/
          anomaly_scores.ts    ← T_test floats, one per line
          model.pkl            ← trained LSTM weights
      machine-1-1_metric_01/
        1/
          anomaly_scores.ts
      ...

Usage
-----
  # Run on all machines found in timeeval_input/
  python scripts/run_lstm_ad.py \\
      --input-dir   data/timeeval_input \\
      --results-dir data/timeeval_results

  # Run on specific machines only
  python scripts/run_lstm_ad.py \\
      --input-dir   data/timeeval_input \\
      --results-dir data/timeeval_results \\
      --machines    machine-1-1 machine-1-2

  # Quick test: run on first 2 metrics of one machine only
  python scripts/run_lstm_ad.py \\
      --input-dir   data/timeeval_input \\
      --results-dir data/timeeval_results \\
      --machines    machine-1-1 \\
      --max-metrics 2
"""

import os
import sys
import argparse
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ══════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Run LSTM-AD via TimeEval on converted SMD metric CSVs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--input-dir",   default="data/timeeval_input",
                   help="Output of convert_to_timeeval.py (contains datasets.csv)")
    p.add_argument("--results-dir", default="data/timeeval_results",
                   help="Where TimeEval writes anomaly_scores.ts files")
    p.add_argument("--machines",    nargs="+", default=None,
                   help="Only process these machines. Default: all in datasets.csv")
    p.add_argument("--max-metrics", type=int, default=None,
                   help="Limit metrics per machine (useful for quick testing)")

    # LSTM-AD hyperparameters
    p.add_argument("--epochs",      type=int, default=50,
                   help="LSTM training epochs (default: 50)")
    p.add_argument("--window-size", type=int, default=30,
                   help="LSTM lookback window in timesteps (default: 30)")
    p.add_argument("--lstm-units",  type=int, default=64,
                   help="Hidden units per LSTM layer (default: 64)")
    p.add_argument("--lstm-layers", type=int, default=2,
                   help="Number of LSTM layers (default: 2)")
    p.add_argument("--batch-size",  type=int, default=32,
                   help="Training batch size (default: 32)")

    # Docker
    p.add_argument("--image",       default="ghcr.io/timeeval/lstm_ad:0.3.0",
                   help="Docker image to use (default: ghcr.io/timeeval/lstm_ad:0.3.0)")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════
# Pre-flight checks
# ══════════════════════════════════════════════════════════════════

def check_prerequisites(input_dir: Path, image: str):
    """Verify Docker is running, image exists, and datasets.csv is present."""
    import subprocess

    print("  Checking prerequisites...")

    # 1. Docker daemon
    result = subprocess.run(
        ["docker", "ps"], capture_output=True, text=True
    )
    if result.returncode != 0:
        print("\n  [ERROR] Docker is not running or not accessible.")
        print("          Start Docker and try again.")
        print("          On Linux: sudo systemctl start docker")
        sys.exit(1)
    print("  ✓ Docker is running")

    # 2. LSTM-AD image
    result = subprocess.run(
        ["docker", "images", "-q", image],
        capture_output=True, text=True
    )
    if not result.stdout.strip():
        print(f"\n  [ERROR] Docker image not found: {image}")
        print(f"          Pull it with:")
        print(f"            docker pull {image}")
        sys.exit(1)
    print(f"  ✓ Image found: {image}")

    # 3. datasets.csv
    idx = input_dir / "datasets.csv"
    if not idx.exists():
        print(f"\n  [ERROR] datasets.csv not found at {idx}")
        print("          Run convert_to_timeeval.py first.")
        sys.exit(1)
    print(f"  ✓ datasets.csv found: {idx}")


# ══════════════════════════════════════════════════════════════════
# TimeEval runner
# ══════════════════════════════════════════════════════════════════

def run_lstm_ad(
    input_dir:   Path,
    results_dir: Path,
    machines:    list,
    max_metrics: int,
    params:      dict,
    image:       str,
):
    """
    Runs LSTM-AD via the TimeEval Python API.
    Uses DockerAdapter directly (most robust approach).
    """
    # ── Import checks with precise error messages ────────────────
    try:
        import timeeval
    except ImportError:
        print("\n  [ERROR] TimeEval framework not installed.")
        print("          Run: pip install TimeEval")
        print(f"          Requires Python 3.9–3.12 (you have: {sys.version})")
        sys.exit(1)

    try:
        from timeeval import TimeEval, DatasetManager, Algorithm, TrainingType, InputDimensionality
        from timeeval.adapters import DockerAdapter
        from timeeval.params import FixedParameters
    except ImportError as e:
        print(f"\n  [ERROR] TimeEval import failed: {e}")
        print(f"          Installed version: {timeeval.__version__}")
        print(f"          Try reinstalling: pip install --upgrade TimeEval")
        sys.exit(1)

    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Load all datasets from the manager ───────────────────────
    # dm.select() returns List[Tuple[str,str]] = [(collection, dataset_name), ...]
    dm = DatasetManager(str(input_dir), create_if_missing=False)
    all_datasets = dm.select()   # all 1064 tuples

    print(f"  DatasetManager loaded {len(all_datasets)} datasets")

    # ── Filter by machine and max_metrics ────────────────────────
    if machines or max_metrics:
        machine_counts = {}
        filtered = []
        for ds in all_datasets:
            collection, ds_name = ds  # each ds is (collection, dataset_name)

            # Filter by machine prefix
            if machines:
                matched = any(ds_name.startswith(m) for m in machines)
                if not matched:
                    continue
                machine_key = next(m for m in machines if ds_name.startswith(m))
            else:
                machine_key = ds_name.rsplit("_metric_", 1)[0] \
                    if "_metric_" in ds_name else ds_name

            # Apply max_metrics limit per machine
            if max_metrics:
                count = machine_counts.get(machine_key, 0)
                if count >= max_metrics:
                    continue
                machine_counts[machine_key] = count + 1

            filtered.append(ds)
        datasets = filtered
    else:
        datasets = all_datasets

    if not datasets:
        print(f"  [ERROR] No datasets found for machines: {machines}")
        sys.exit(1)

    n = len(datasets)
    print(f"\n  Running LSTM-AD on {n} series")
    print(f"  Estimated time: ~{n * 2} minutes on CPU\n")
    for ds in datasets[:5]:
        print(f"    {ds[0]} / {ds[1]}")
    if n > 5:
        print(f"    ... and {n-5} more")
    print()

    # ── Build algorithm ───────────────────────────────────────────
    # Note: DockerAdapter tag parameter must be set separately from image_name
    # to avoid "image:tag:latest" invalid reference format
    image_name, image_tag = (image.rsplit(":", 1) if ":" in image
                             else (image, "latest"))

    algorithm = Algorithm(
        name="LSTM-AD",
        main=DockerAdapter(
            image_name=image_name,
            tag=image_tag,
            skip_pull=True,
        ),
        training_type=TrainingType.SEMI_SUPERVISED,
        input_dimensionality=InputDimensionality.UNIVARIATE,
        data_as_file=True,
        param_config=FixedParameters(params),
    )

    te = TimeEval(
        dm,
        datasets,
        algorithms=[algorithm],
        results_path=results_dir,      # Path object, not str
        metrics=[],
        distributed=False,
    )

    t0 = time.time()
    te.run()
    elapsed = time.time() - t0

    print(f"\n  ✓ LSTM-AD finished in {int(elapsed//60)}m {int(elapsed%60)}s")
    print(f"  Results → {results_dir}")


# ══════════════════════════════════════════════════════════════════
# Post-run summary
# ══════════════════════════════════════════════════════════════════

def summarise_results(results_dir: Path, machines: list):
    """
    Scans results directory and reports which score files were created
    and their sizes, so you can verify everything ran correctly.
    """
    import numpy as np

    print(f"\n{'═'*60}")
    print(f"  Results Summary")
    print(f"{'═'*60}")

    score_files = sorted(results_dir.rglob("anomaly_scores.ts"))

    if not score_files:
        print(f"  [!] No anomaly_scores.ts files found in {results_dir}")
        print(f"      Something went wrong — check TimeEval logs above.")
        return

    print(f"  Found {len(score_files)} score files\n")
    print(f"  {'Dataset':<40} {'Length':>8} {'Min':>8} {'Max':>8}")
    print(f"  {'─'*64}")

    for f in score_files[:20]:   # show first 20
        dataset_name = f.parent.parent.name
        try:
            scores = np.loadtxt(f)
            print(f"  {dataset_name:<40} {len(scores):>8} "
                  f"{scores.min():>8.4f} {scores.max():>8.4f}")
        except Exception as e:
            print(f"  {dataset_name:<40} [ERROR reading: {e}]")

    if len(score_files) > 20:
        print(f"  ... and {len(score_files)-20} more")

    print(f"\n{'═'*60}")
    print(f"  Score file location pattern:")
    print(f"    {results_dir}/")
    print(f"      LSTM-AD/")
    print(f"        machine-1-1_metric_00/1/anomaly_scores.ts")
    print(f"        machine-1-1_metric_01/1/anomaly_scores.ts")
    print(f"        ...")
    print(f"")
    print(f"  Note: dataset_name uses underscore (machine-1-1_metric_00)")
    print(f"        file path uses slash   (machine-1-1/metric_00/train.csv)")
    print(f"        These are two different things — both are correct.")
    print(f"")
    print(f"  Next step: run extract_scores.py to normalise and assemble")
    print(f"  all per-metric scores into [T, M] arrays per machine.")
    print(f"{'═'*60}")
    print(f"\n  Each file contains T_test floats (one per line).")
    print(f"  These are RAW prediction errors, not yet normalised to [0,1].")
    print(f"\n  Next step: run extract_scores.py to normalise and assemble")
    print(f"  all per-metric scores into [T, M] arrays per machine.")
    print(f"{'═'*60}")


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

def main():
    args        = parse_args()
    input_dir   = Path(args.input_dir)
    results_dir = Path(args.results_dir)

    params = {
        "epochs":           args.epochs,
        "window_size":      args.window_size,
        "lstm_units":       args.lstm_units,
        "lstm_layers":      args.lstm_layers,
        "batch_size":       args.batch_size,
        "split":            0.8,
        "prediction_length": 1,        # required: steps ahead to predict
        "early_stopping_delta":    0.05,
        "early_stopping_patience": 10,
    }

    print(f"\n{'═'*60}")
    print(f"  LSTM-AD via TimeEval")
    print(f"{'═'*60}")
    print(f"  Input dir    : {input_dir.resolve()}")
    print(f"  Results dir  : {results_dir.resolve()}")
    print(f"  Machines     : {args.machines or 'all'}")
    print(f"  Max metrics  : {args.max_metrics or 'all'}")
    print(f"  Docker image : {args.image}")
    print(f"  LSTM params  : {params}")

    # Pre-flight
    check_prerequisites(input_dir, args.image)

    # Run
    run_lstm_ad(
        input_dir   = input_dir,
        results_dir = results_dir,
        machines    = args.machines,
        max_metrics = args.max_metrics,
        params      = params,
        image       = args.image,
    )

    # Summary
    summarise_results(results_dir, args.machines)


if __name__ == "__main__":
    main()
elif __name__ == "scripts.run_lstm_ad":
    main()