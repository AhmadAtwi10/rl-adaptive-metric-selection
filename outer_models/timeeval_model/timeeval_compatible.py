"""
convert_to_timeeval.py
======================
STEP 1 of the TimeEval pipeline.

Converts your SMD-format dataset into the per-metric CSV structure
that TimeEval expects, and produces the datasets.csv index file.

What this script does
---------------------
  For each machine (e.g. machine-1-1):
    - Reads machine-1-1.train.csv  and  machine-1-1.test.csv
    - Splits the 38 multivariate columns into 38 individual series
    - Writes one train.csv + test.csv per metric in TimeEval format
    - Adds an entry to datasets.csv for each series

Input format (your SMD files)
------------------------------
  machine-1-1.train.csv
    timestamp, value-0, value-1, ..., value-37
    0, 0.032, 0.039, ...

  machine-1-1.test.csv
    timestamp, value-0, value-1, ..., value-37, is_anomaly
    0, 0.032, 0.039, ..., 0

Output format (TimeEval canonical)
------------------------------------
  timeeval_input/
    machine-1-1/
      metric_00/
        train.csv    →   timestamp, value
        test.csv     →   timestamp, value, is_anomaly
      metric_01/
        train.csv
        test.csv
      ...
      metric_37/
        ...
    machine-1-2/
      ...
    datasets.csv     ←  TimeEval index (one row per series)

Usage
-----
  # Convert all machines found in dataset_dir
  python scripts/convert_to_timeeval.py \\
      --dataset-dir datasets/SMD \\
      --out-dir     data/timeeval_input

  # Convert specific machines only
  python scripts/convert_to_timeeval.py \\
      --dataset-dir datasets/SMD \\
      --out-dir     data/timeeval_input \\
      --machines    machine-1-1 machine-1-2 machine-2-1

  # Preview what would be converted without writing anything
  python scripts/convert_to_timeeval.py \\
      --dataset-dir datasets/SMD \\
      --out-dir     data/timeeval_input \\
      --dry-run
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
        description="Convert SMD-format dataset to TimeEval per-metric CSVs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--dataset-dir",          required=True,
                   help="Folder containing machine-*.train.csv and machine-*.test.csv")
    p.add_argument("--out-dir",              default="data/timeeval_input",
                   help="Where to write the converted files (default: data/timeeval_input)")
    p.add_argument("--interpretation-dir",   default=None,
                   help="Folder containing interpretation_label .txt files "
                        "(e.g. machine-1-1.txt). If provided, per-metric labels "
                        "are used instead of machine-level labels.")
    p.add_argument("--machines",             nargs="+", default=None,
                   help="Only convert these machines. Default: all found in dataset-dir")
    p.add_argument("--dry-run",              action="store_true",
                   help="Print what would be done without writing any files")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════
# Format detection
# ══════════════════════════════════════════════════════════════════

def find_machines(dataset_dir: Path) -> list:
    """
    Auto-discovers all machine names by looking for *.train.csv files.
    Returns a sorted list of machine names (e.g. ['machine-1-1', 'machine-1-2', ...])
    """
    files = sorted(dataset_dir.glob("*.train.csv"))
    if not files:
        print(f"[ERROR] No *.train.csv files found in {dataset_dir}")
        print(f"        Expected files like: machine-1-1.train.csv")
        sys.exit(1)
    return [f.stem.replace(".train", "") for f in files]


# ══════════════════════════════════════════════════════════════════
# Interpretation label parser
# ══════════════════════════════════════════════════════════════════

def parse_interpretation_label(txt_path: Path, T_test: int, M: int) -> np.ndarray:
    """
    Parses an SMD interpretation_label .txt file and builds a
    per-metric binary label matrix of shape [T_test, M].

    File format (one anomaly segment per line):
        start_idx-end_idx:metric_idx,metric_idx,...

    Example:
        15849-16368:1,9,10,12,13,14,15
        16963-17517:1,2,3,4,6,7,9,10,11,12,13,14,15,16,19,20,21,22,24,25,26,27,28,29,30,31,32,33,34,35,36

    Returns:
        labels_per_metric : np.ndarray  [T_test, M]  int32
            labels_per_metric[t, m] = 1  iff  metric m was anomalous at timestep t
    """
    labels = np.zeros((T_test, M), dtype=np.int32)

    with open(txt_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Split on ':' to get range and metrics
            try:
                range_part, metrics_part = line.split(":")
            except ValueError:
                print(f"  [!] Could not parse line: '{line}' — skipping")
                continue

            # Parse start-end
            try:
                start, end = map(int, range_part.split("-"))
            except ValueError:
                print(f"  [!] Could not parse range '{range_part}' — skipping")
                continue

            # Parse metric indices
            try:
                metric_indices = list(map(int, metrics_part.split(",")))
            except ValueError:
                print(f"  [!] Could not parse metrics '{metrics_part}' — skipping")
                continue

            # Clip to valid range (interpretation labels use test-split indices)
            start = max(0, start)
            end   = min(T_test - 1, end)

            # Mark valid metric indices
            for m_idx in metric_indices:
                if 0 <= m_idx < M:
                    labels[start:end + 1, m_idx] = 1
                else:
                    print(f"  [!] Metric index {m_idx} out of range [0,{M-1}] — skipping")

    return labels


def load_machine(dataset_dir: Path, machine: str):
    """
    Loads train and test CSVs for one machine.

    Returns:
      train_values    : np.ndarray  [T_train, M]  float32
      test_values     : np.ndarray  [T_test,  M]  float32
      machine_labels  : np.ndarray  [T_test]       int32   machine-level (0/1)
      metric_names    : list of str  ['value-0', ..., 'value-37']
    """
    train_path = dataset_dir / f"{machine}.train.csv"
    test_path  = dataset_dir / f"{machine}.test.csv"

    if not train_path.exists():
        raise FileNotFoundError(f"Train file not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")

    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)

    # ── Drop timestamp column ─────────────────────────────────────
    def drop_ts(df):
        col = df.columns[0]
        if col.lower() in ("timestamp", "time", "datetime", "date", "index"):
            return df.drop(columns=[col])
        try:
            vals = pd.to_numeric(df[col]).values
            if np.allclose(vals, np.arange(len(vals))):
                return df.drop(columns=[col])
        except (ValueError, TypeError):
            pass
        return df

    train_df = drop_ts(train_df)
    test_df  = drop_ts(test_df)

    # ── Extract machine-level labels from test ────────────────────
    if "is_anomaly" in test_df.columns:
        machine_labels = test_df["is_anomaly"].values.astype(np.int32)
        test_df        = test_df.drop(columns=["is_anomaly"])
    else:
        machine_labels = test_df.iloc[:, -1].values.astype(np.int32)
        test_df        = test_df.iloc[:, :-1]

    if "is_anomaly" in train_df.columns:
        train_df = train_df.drop(columns=["is_anomaly"])

    metric_names = list(train_df.columns)
    train_values = train_df.values.astype(np.float32)
    test_values  = test_df.values.astype(np.float32)

    return train_values, test_values, machine_labels, metric_names


# ══════════════════════════════════════════════════════════════════
# Conversion
# ══════════════════════════════════════════════════════════════════

def convert_machine(
    machine:        str,
    train_values:   np.ndarray,    # [T_train, M]
    test_values:    np.ndarray,    # [T_test,  M]
    labels:         np.ndarray,    # [T_test, M] per-metric  OR  [T_test] machine-level
    metric_names:   list,
    out_dir:        Path,
    dry_run:        bool,
) -> list:
    """
    Writes one train.csv + test.csv per metric for a single machine.

    labels can be:
      [T_test, M]  →  per-metric labels from interpretation_label file
      [T_test]     →  machine-level label broadcast to all metrics (fallback)

    Returns a list of datasets.csv index rows.
    """
    T_train, M = train_values.shape
    T_test      = test_values.shape[0]
    index_rows  = []

    # Ensure labels is always [T_test, M]
    if labels.ndim == 1:
        # Broadcast machine-level label to all metrics
        labels_2d = np.tile(labels[:, np.newaxis], (1, M))   # [T_test, M]
    else:
        labels_2d = labels   # already [T_test, M]

    for m in range(M):
        metric_dir = out_dir / machine / f"metric_{m:02d}"

        train_csv_path = metric_dir / "train.csv"
        test_csv_path  = metric_dir / "test.csv"

        if not dry_run:
            metric_dir.mkdir(parents=True, exist_ok=True)

            # Training CSV: timestamp + value + is_anomaly (all 0 — normal data)
            pd.DataFrame({
                "timestamp":  np.arange(T_train),
                "value":      train_values[:, m],
                "is_anomaly": np.zeros(T_train, dtype=np.int32),
            }).to_csv(train_csv_path, index=False)

            # Test CSV: timestamp + value + is_anomaly (per-metric labels)
            pd.DataFrame({
                "timestamp":  np.arange(T_test),
                "value":      test_values[:, m],
                "is_anomaly": labels_2d[:, m],
            }).to_csv(test_csv_path, index=False)

        index_rows.append({
            "collection_name":      "SMD",
            "dataset_name":         f"{machine}_metric_{m:02d}",
            "train_path":           str(train_csv_path.resolve()),
            "test_path":            str(test_csv_path.resolve()),
            # Required by TimeEval Dataset constructor
            "dataset_type":         "real",
            "datetime_index":       False,
            "split_at":             0.8,
            "train_type":           "semi-supervised",
            "train_is_normal":      True,
            "input_type":           "univariate",
            "length":               T_test,
            "dimensions":           1,
            "contamination":        float(labels_2d[:, m].mean()),
            "num_anomalies":        int((labels_2d[:, m].sum() > 0)),
            "min_anomaly_length":   1,
            "median_anomaly_length":1,
            "max_anomaly_length":   T_test,
        })

    return index_rows


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

def main():
    args        = parse_args()
    dataset_dir = Path(args.dataset_dir)
    out_dir     = Path(args.out_dir)
    interp_dir  = Path(args.interpretation_dir) if args.interpretation_dir else None

    # Discover machines
    if args.machines:
        machines = args.machines
    else:
        machines = find_machines(dataset_dir)

    # Check interpretation label directory
    using_per_metric_labels = False
    if interp_dir:
        if interp_dir.exists():
            using_per_metric_labels = True
            print(f"  Interpretation labels : {interp_dir}  ✓")
        else:
            print(f"  [!] Interpretation dir not found: {interp_dir}")
            print(f"      Falling back to machine-level labels")

    print(f"\n{'═'*62}")
    print(f"  SMD → TimeEval Conversion")
    print(f"{'═'*62}")
    print(f"  Dataset dir    : {dataset_dir.resolve()}")
    print(f"  Output dir     : {out_dir.resolve()}")
    print(f"  Machines       : {len(machines)}")
    print(f"  Label mode     : {'per-metric (interpretation_label)' if using_per_metric_labels else 'machine-level (broadcast)'}")
    print(f"  Dry run        : {args.dry_run}")
    if args.dry_run:
        print(f"\n  [DRY RUN] No files will be written\n")

    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    all_index_rows = []
    total_series   = 0

    print(f"\n  {'Machine':<18} {'T_train':>8} {'T_test':>8} "
          f"{'M':>5} {'Mach.Anom%':>11} {'Metric.Anom%':>13} {'Label src':>12}")
    print(f"  {'─'*78}")

    for machine in machines:
        try:
            train_values, test_values, machine_labels, metric_names = \
                load_machine(dataset_dir, machine)
        except FileNotFoundError as e:
            print(f"  [!] {e} — skipping")
            continue

        T_train, M = train_values.shape
        T_test      = test_values.shape[0]

        # ── Decide which labels to use ───────────────────────────
        interp_path = interp_dir / f"{machine}.txt" if interp_dir else None

        if using_per_metric_labels and interp_path and interp_path.exists():
            # Parse interpretation label → [T_test, M]
            labels_2d  = parse_interpretation_label(interp_path, T_test, M)
            label_src  = "interp"
            metric_anom_pct = 100.0 * labels_2d.mean()
        else:
            if using_per_metric_labels:
                print(f"  [!] {machine}: interpretation label not found "
                      f"→ falling back to machine-level")
            # Broadcast machine-level label → [T_test, M]
            labels_2d  = np.tile(machine_labels[:, np.newaxis], (1, M))
            label_src  = "machine"
            metric_anom_pct = 100.0 * machine_labels.mean()

        mach_anom_pct = 100.0 * machine_labels.mean()
        print(f"  {machine:<18} {T_train:>8} {T_test:>8} "
              f"{M:>5} {mach_anom_pct:>10.2f}% {metric_anom_pct:>12.2f}% {label_src:>12}")

        rows = convert_machine(
            machine      = machine,
            train_values = train_values,
            test_values  = test_values,
            labels       = labels_2d,
            metric_names = metric_names,
            out_dir      = out_dir,
            dry_run      = args.dry_run,
        )
        all_index_rows.extend(rows)
        total_series += len(rows)

    # Write datasets.csv
    # IMPORTANT: TimeEval's DatasetManager loads this with index_col=[0,1]
    # making collection_name and dataset_name a MultiIndex.
    # We must write them as the first two columns WITHOUT a separate row index.
    datasets_csv_path = out_dir / "datasets.csv"
    if not args.dry_run:
        df = pd.DataFrame(all_index_rows)
        # Ensure collection_name and dataset_name are first two columns
        cols = ["collection_name", "dataset_name"] + \
               [c for c in df.columns if c not in ("collection_name", "dataset_name")]
        df = df[cols]
        df.to_csv(datasets_csv_path, index=False)

    print(f"\n{'═'*62}")
    print(f"  Summary")
    print(f"{'═'*62}")
    print(f"  Machines processed : {len(machines)}")
    print(f"  Total series       : {total_series}  "
          f"({len(machines)} machines × {total_series // max(len(machines),1)} metrics)")
    print(f"  Label mode         : {'per-metric from interpretation_label' if using_per_metric_labels else 'machine-level broadcast'}")
    print(f"\n  Output structure:")
    print(f"    {out_dir}/")
    print(f"      <machine>/metric_00/train.csv  test.csv")
    print(f"      <machine>/metric_01/train.csv  test.csv")
    print(f"      ...  ({total_series // max(len(machines),1)} metric folders per machine)")
    print(f"      datasets.csv   ← TimeEval index ({total_series} rows)")

    if not args.dry_run:
        print(f"\n  ✓ datasets.csv → {datasets_csv_path}")
    else:
        print(f"\n  [Dry run — re-run without --dry-run to write files]")

    _print_next_steps(out_dir, datasets_csv_path)


def _print_next_steps(out_dir: Path, datasets_csv_path: Path):
    print(f"""
{'═'*60}
  NEXT STEP — Running TimeEval with LSTM-AD
{'═'*60}

  Prerequisites:
    pip install TimeEval
    docker --version          # Docker must be running
    docker ps                 # verify Docker daemon is up

  Option A — Python API (recommended, most control):

    from timeeval import TimeEval, DatasetManager
    from timeeval.params import FixedParameters
    from timeeval.algorithms import lstm_ad

    dm       = DatasetManager("{out_dir}", create_if_missing=False)
    datasets = dm.select()   # or dm.select(dataset_names=["machine-1-1_metric_00"])

    te = TimeEval(
        dm,
        datasets,
        algorithms=[lstm_ad(params=FixedParameters({{
            "epochs":       50,
            "lstm_layers":  2,
            "lstm_units":   64,
            "window_size":  30,
            "batch_size":   32,
        }}))],
        results_path="data/timeeval_results",
        metrics=[],
        distributed=False,
    )
    te.run()

  Option B — TimeEval CLI (if installed):

    timeeval-run \\
        --data-dir  {out_dir} \\
        --results   data/timeeval_results \\
        --algorithm lstm_ad

  Where to find results after running:
    data/timeeval_results/
      LSTM-AD/
        machine-1-1_metric_00/1/anomaly_scores.ts   ← one float per line
        machine-1-1_metric_01/1/anomaly_scores.ts
        ...

  Each anomaly_scores.ts file contains T_test floats (one per timestep).
  These are raw prediction errors — NOT yet normalised to [0,1].
  The next script (extract_scores.py) will normalise and assemble them.
{'═'*60}
""")


if __name__ == "__main__":
    main()
elif __name__ == "scripts.convert_to_timeeval":
    main()