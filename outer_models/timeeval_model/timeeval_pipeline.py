"""
timeeval_pipeline.py
====================
End-to-end pipeline: Dataset → TimeEval LSTM-AD → Anomaly Scores → RL Evaluation

This single script runs all steps in order:

  STEP 1  Convert raw dataset to TimeEval's per-metric CSV format
  STEP 2  Run LSTM-AD via TimeEval to get raw anomaly scores
  STEP 3  Extract, align, and normalise scores into [T, N, M] arrays
  STEP 4  Verify scores look correct (sanity checks + optional plot)
  STEP 5  Run trained RL agent on the scores and report results

Each step can be run independently or skipped if already done.

Supported datasets
------------------
  smd         Server Machine Dataset (28 machines, 38 metrics each)
              Raw format: space-separated .txt files
              Download:   https://github.com/NetManAIOps/OmniAnomaly

  custom      Any dataset in the same format as SMD:
              train/<name>.txt  [T_train, M]  space-separated, no header
              test/<name>.txt   [T_test,  M]
              test_label/<name>.txt  [T_test] binary labels

Usage examples
--------------
  # Full pipeline on SMD (all machines)
  python scripts/timeeval_pipeline.py \\
      --dataset-dir /path/to/ServerMachineDataset \\
      --work-dir    data/timeeval_workspace \\
      --model       results/best_model.pt

  # Full pipeline on specific machines only
  python scripts/timeeval_pipeline.py \\
      --dataset-dir /path/to/ServerMachineDataset \\
      --work-dir    data/timeeval_workspace \\
      --model       results/best_model.pt \\
      --machines    machine-1-1 machine-1-2 machine-2-1

  # Skip steps already done (e.g. TimeEval already ran)
  python scripts/timeeval_pipeline.py \\
      --dataset-dir /path/to/ServerMachineDataset \\
      --work-dir    data/timeeval_workspace \\
      --model       results/best_model.pt \\
      --skip-convert \\
      --skip-timeeval

  # Only run steps 1-3 (no RL evaluation — just get the scores)
  python scripts/timeeval_pipeline.py \\
      --dataset-dir /path/to/ServerMachineDataset \\
      --work-dir    data/timeeval_workspace \\
      --skip-eval

  # With verification plots
  python scripts/timeeval_pipeline.py \\
      --dataset-dir /path/to/ServerMachineDataset \\
      --work-dir    data/timeeval_workspace \\
      --model       results/best_model.pt \\
      --plot

Output structure
----------------
  data/timeeval_workspace/
    timeeval_input/              ← step 1 output (per-metric CSVs)
      machine-1-1/
        metric_00/train.csv  test.csv
        metric_01/train.csv  test.csv
        ...
        labels_per_metric.npy  [T, M]  per-metric ground truth
        labels_node.npy        [T]     node-level union label
      datasets.csv             ← TimeEval index file (full 22-column schema)
    timeeval_results/          ← step 2 output (TimeEval's own tree)
      2026_01_31_12_00_00/     ← TimeEval creates one dir per run
        LSTM-AD/<params_id>/<collection>/<machine>_metric_00/1/
          anomaly_scores.ts      post-processed AND min-max scaled to [0,1]
          raw_anomaly_scores.ts  post-processed only
          model.pkl  execution.log  metrics.csv  hyper_params.json
    anomaly_scores/            ← step 3 output (final arrays)
      machine-1-1_scores.npy            [T, M]  float32  [0,1]
      machine-1-1_labels.npy            [T]     int32    node-level
      machine-1-1_labels_per_metric.npy [T, M]  int32
      per_metric_auroc.csv              one row per (machine, metric)
      ...
    plots/                     ← step 4 output (optional)
      machine-1-1_heatmap.png
      ...
    rl_evaluation/             ← step 5 output
      evaluation_results.csv
      evaluation_comparison.png
      machine-1-1_timestep_records.csv
      ...
"""

import os
import sys
import argparse
import csv
import time
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ══════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="TimeEval LSTM-AD → Anomaly Scores → RL Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Paths
    p.add_argument("--dataset-dir", required=True,
                   help="Root directory of the dataset "
                        "(must contain train/, test/, test_label/)")
    p.add_argument("--work-dir",    default="data/timeeval_workspace",
                   help="Working directory for all intermediate and output files")
    p.add_argument("--model",       default="results/best_model.pt",
                   help="Trained RL model checkpoint")
    p.add_argument("--config",      default="configs/config.yaml",
                   help="RL config file")

    # Dataset
    p.add_argument("--machines",    nargs="+", default=None,
                   help="Process only these machines/entities. "
                        "Default: auto-discover all from dataset-dir/train/")
    p.add_argument("--n-metrics",   type=int, default=None,
                   help="Number of metrics per machine. "
                        "Default: auto-detect from first training file")

    # Which detector
    p.add_argument("--algorithm", default="lstm_ad", choices=sorted(ALGORITHMS),
                   help="Detector used to produce anomaly scores. See the "
                        "ALGORITHMS registry for each one's training type and "
                        "scoring principle — prediction-based detectors fade on "
                        "sustained anomalies, reconstruction/distance ones do "
                        "not, and unsupervised ones assume anomalies are rare.")
    p.add_argument("--algo-param", action="append", default=[], metavar="K=V",
                   help="Override one registry parameter, e.g. "
                        "--algo-param epochs=10. Repeatable. Values are parsed "
                        "as int, then float, then left as a string.")

    # LSTM-AD hyperparameters (defaults = TimeEval-algorithms/lstm_ad 0.3.0)
    p.add_argument("--lstm-epochs",      type=int,   default=50)
    p.add_argument("--lstm-layers",      type=int,   default=2)
    p.add_argument("--lstm-units",       type=int,   default=64,
                   help="IGNORED — LSTM-AD has no such parameter; its hidden "
                        "size is fixed to the input dimensionality. Kept so "
                        "existing command lines keep working.")
    p.add_argument("--window-size",      type=int,   default=30)
    p.add_argument("--prediction-window-size", type=int, default=1,
                   help="Points predicted per window. Must match what the "
                        "postprocess step assumes, so leave at the default "
                        "unless you know why.")
    p.add_argument("--lstm-batch-size",  type=int,   default=32)
    p.add_argument("--lstm-split",       type=float, default=0.9,
                   help="Train/validation split. The validation part is what "
                        "the Gaussian error model is fitted on.")
    p.add_argument("--lstm-seed",        type=int,   default=42,
                   help="random_state passed to the container")

    # Pipeline control
    p.add_argument("--skip-convert",  action="store_true",
                   help="Skip step 1 (CSVs already created). Step 1 is skipped "
                        "automatically when the shared conversion already "
                        "exists, so this is rarely needed.")
    p.add_argument("--force-convert", action="store_true",
                   help="Rebuild the shared conversion even if it exists. Needed "
                        "only after changing the raw dataset or step 1 itself — "
                        "and note it invalidates every detector's scores in this "
                        "work dir, since they were computed against the old "
                        "labels_per_metric.npy.")
    p.add_argument("--skip-timeeval", action="store_true",
                   help="Skip step 2 (TimeEval already ran)")
    p.add_argument("--skip-eval",     action="store_true",
                   help="Skip step 5 (only produce score files)")
    p.add_argument("--allow-missing-scores", action="store_true",
                   help="Continue with zero-filled scores when a series has no "
                        "result file. Off by default: a missing score file is a "
                        "pipeline failure, not a data property.")
    p.add_argument("--plot",          action="store_true",
                   help="Save score heatmap plots per machine (step 4)")

    # Other
    p.add_argument("--device",  default=None,
                   help="cuda / cpu for RL evaluation")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════

def banner(title: str):
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print(f"{'═'*60}")


def elapsed(t0: float) -> str:
    s = time.time() - t0
    return f"{int(s//60)}m {int(s%60)}s"


# ══════════════════════════════════════════════════════════════════
# STEP 1 — Convert dataset to TimeEval per-metric CSV format
# ══════════════════════════════════════════════════════════════════

def _detect_format(dataset_dir: Path, machine: str):
    """
    Auto-detects which file format is used and returns
    (train_path, test_path, label_path, fmt) where fmt is one of:
      'csv_flat'   — datasets/SMD/machine-1-1.train.csv  /  .test.csv
                     (multivariate CSV, all metrics in columns)
      'txt_split'  — train/<machine>.txt  /  test/<machine>.txt
                     /  test_label/<machine>.txt
                     (original OmniAnomaly raw format)
    Returns None if no matching files found.
    """
    # Format 1: flat CSV  (your actual format)
    # datasets/SMD/machine-1-1.train.csv
    # datasets/SMD/machine-1-1.test.csv
    train_csv = dataset_dir / f"{machine}.train.csv"
    test_csv  = dataset_dir / f"{machine}.test.csv"
    if train_csv.exists() and test_csv.exists():
        return train_csv, test_csv, None, "csv_flat"

    # Format 2: original txt split into subfolders
    train_txt = dataset_dir / "train"      / f"{machine}.txt"
    test_txt  = dataset_dir / "test"       / f"{machine}.txt"
    label_txt = dataset_dir / "test_label" / f"{machine}.txt"
    if train_txt.exists() and test_txt.exists():
        return train_txt, test_txt, label_txt, "txt_split"

    return None, None, None, None


def _load_machine_data(train_path, test_path, label_path, fmt: str):
    """
    Loads train/test arrays and labels for one machine.

    Returns:
      train_raw : np.ndarray [T_train, M]
      test_raw  : np.ndarray [T_test,  M]
      labels    : np.ndarray [T_test]   binary int

    For csv_flat format the test CSV already contains all metric columns
    plus an is_anomaly column as the last column.
    """
    if fmt == "csv_flat":
        # Multivariate CSV — all metrics as columns
        # Expected columns: timestamp, metric_0, metric_1, ..., is_anomaly
        # OR:               metric_0, metric_1, ..., is_anomaly  (no timestamp)
        train_df = pd.read_csv(train_path)
        test_df  = pd.read_csv(test_path)

        # Drop timestamp column if present (non-numeric first col or named 'timestamp')
        def drop_timestamp(df):
            first_col = df.columns[0]
            # Explicit name match
            if first_col.lower() in ("timestamp", "time", "datetime", "date", "index"):
                return df.iloc[:, 1:]
            # Non-numeric → definitely a datetime string
            try:
                pd.to_numeric(df[first_col])
            except (ValueError, TypeError):
                return df.iloc[:, 1:]
            # Numeric first column: only drop if it is a PERFECT integer
            # sequential index 0,1,2,...,N-1 AND all other columns are floats
            # (i.e. it carries no information beyond row number)
            vals = df[first_col].values
            is_seq_index = (
                np.issubdtype(vals.dtype, np.integer) or
                (np.allclose(vals, np.round(vals)) and
                 np.allclose(vals, np.arange(len(vals))))
            )
            if is_seq_index:
                return df.iloc[:, 1:]
            # Otherwise keep it — it is a real metric column
            return df

        train_df = drop_timestamp(train_df)
        test_df  = drop_timestamp(test_df)

        # Last column of test is is_anomaly
        if "is_anomaly" in test_df.columns:
            labels    = test_df["is_anomaly"].values.astype(np.int32)
            test_df   = test_df.drop(columns=["is_anomaly"])
        else:
            # Assume last column is the label
            labels    = test_df.iloc[:, -1].values.astype(np.int32)
            test_df   = test_df.iloc[:, :-1]

        # Train CSV may or may not have is_anomaly column
        if "is_anomaly" in train_df.columns:
            train_df = train_df.drop(columns=["is_anomaly"])

        train_raw = train_df.values.astype(np.float32)
        test_raw  = test_df.values.astype(np.float32)

    elif fmt == "txt_split":
        # Original OmniAnomaly format — space-separated .txt files
        train_raw = np.loadtxt(train_path, dtype=np.float32)
        test_raw  = np.loadtxt(test_path,  dtype=np.float32)
        labels    = np.loadtxt(label_path, dtype=np.int32) \
                    if label_path and Path(label_path).exists() \
                    else np.zeros(len(test_raw), dtype=np.int32)
    else:
        raise ValueError(f"Unknown format: {fmt}")

    # Ensure 2D
    if train_raw.ndim == 1:
        train_raw = train_raw[:, np.newaxis]
    if test_raw.ndim == 1:
        test_raw = test_raw[:, np.newaxis]

    return train_raw, test_raw, labels


def _parse_interp_label(line: str, T: int, M: int) -> np.ndarray:
    """
    Parse one SMD-style interpretation-label line into a [T, M] binary array.

    Format:  "start-end:m1,m2,...  start-end:m1,m2,..."
      - Events are space-separated, start/end are INCLUSIVE, metrics are 1-based.

    Duplicated (not imported) from `scripts.compute_metrics.parse_interp_label`
    on purpose: steps 1-4 must run inside the TimeEval venv, which has neither
    torch nor the RL package installed.  Keep the two in sync.
    """
    L = np.zeros((T, M), dtype=np.int32)
    for event in line.strip().split():
        if not event or ":" not in event:
            continue
        time_part, metric_part = event.split(":", 1)
        t_start, t_end = map(int, time_part.split("-"))
        metrics = [int(m) - 1 for m in metric_part.split(",") if m.strip()]
        t_end_clip = min(t_end + 1, T)          # inclusive end → exclusive slice
        for m in metrics:
            if 0 <= m < M:
                L[t_start:t_end_clip, m] = 1
    return L


def _load_interp_labels(dataset_dir: Path, machine: str, T: int, M: int):
    """
    Per-metric ground truth for one machine, or (None, None) when the dataset
    does not ship interpretation labels.

    Searched, in order:
      <dataset_dir>/<machine>.interp_label.txt           (data/synthetic_raw.py)
      <dataset_dir>/interpretation_label/<machine>.txt   (original SMD layout)

    Why this matters: without it every one of the M univariate series inherits
    the *node-level* label (anomalous if ANY metric is), so metric m is scored
    against anomalies that happened in other metrics.  On the 5x7 synthetic set
    that inflates the positive rate from ~0.08-0.45 per metric to ~0.64-0.79,
    which would make every per-metric AUROC meaningless.
    """
    for path in (dataset_dir / f"{machine}.interp_label.txt",
                 dataset_dir / "interpretation_label" / f"{machine}.txt"):
        if path.exists():
            return _parse_interp_label(path.read_text(), T, M), path
    return None, None


# TimeEval addresses every series by (collection_name, dataset_name).  The name
# is arbitrary but must match between the index file and the result paths.
COLLECTION_NAME = "custom"


# ══════════════════════════════════════════════════════════════════
# Algorithm registry
# ══════════════════════════════════════════════════════════════════
#
# Which detector produces the anomaly scores.  All entries are UNIVARIATE-
# compatible and emit one score per timestep, which is what the [T, N, M] state
# tensor needs — a multivariate detector would emit one score per timestep for
# the whole node and could not fill the M axis.
#
# `training` records what the algorithm does with our clean train split:
#   semi-supervised — fits normality on the train split.  Preferred here: the
#                     test split is ~25% anomalous per metric, and unsupervised
#                     methods fit on the test data while assuming anomalies are
#                     rare, which that contamination violates.
#   unsupervised    — ignores the train split entirely.
#
# `principle` is why the choice matters, measured on this dataset:
#   prediction     — scores |x(t) - x_hat(t)|.  Detects the ONSET of an event
#                    but fades once it persists, because the predictor's own
#                    history window fills with the shifted values and it starts
#                    predicting the anomalous level correctly.  Measured for
#                    LSTM-AD: AUROC 0.821 on cells at event position 0-2 versus
#                    0.731 at position >=10, and 50% of anomalous cells are in
#                    the latter group.
#   reconstruction — scores how badly a whole window is rebuilt against learned
#                    normality, so a sustained shift stays anomalous throughout.
#   distance       — scores a subsequence by its distance/density relative to
#                    other subsequences.  Same sustained-event advantage, but
#                    assumes anomalies are rare.
#
# Parameter dicts hold ONLY keys the container declares in its manifest;
# unknown keys are silently dropped by the container and would only perturb
# TimeEval's hyper-parameter hash.
ALGORITHMS = {
    "lstm_ad": {
        "name":      "LSTM-AD",
        "training":  "semi-supervised",
        "principle": "prediction",
        "params": {
            "epochs": 50, "lstm_layers": 2, "window_size": 30,
            "prediction_window_size": 1, "batch_size": 32, "split": 0.9,
            "early_stopping_delta": 0.05, "early_stopping_patience": 10,
            "random_state": 42,
        },
    },
    "encdec_ad": {
        "name":      "EncDec-AD",
        "training":  "semi-supervised",
        "principle": "reconstruction",
        # Same LSTM family as LSTM-AD but reconstruction-scored, so a
        # head-to-head isolates the scoring principle rather than confounding
        # it with a change of architecture.
        # NOTE the container declares BOTH `window_size` and
        # `anomaly_window_size`, and its registered postprocess is
        # ReverseWindowing(2 * anomaly_window_size) — keep the two equal or the
        # score length will not match T.
        "params": {
            "epochs": 50, "lstm_layers": 1, "window_size": 30,
            "anomaly_window_size": 30, "latent_size": 40, "batch_size": 32,
            "split": 0.9, "early_stopping_delta": 0.05,
            "early_stopping_patience": 10, "random_state": 42,
        },
    },
    "donut": {
        "name":      "Donut",
        "training":  "semi-supervised",
        "principle": "reconstruction",
        # VAE built for seasonal univariate KPIs; our per-metric distinct
        # seasonal periods are exactly its assumption.  Different model family
        # from both LSTM methods, so agreement between them is informative.
        "params": {
            "epochs": 256, "latent_size": 5, "linear_hidden_size": 100,
            "regularization": 0.001, "window_size": 120, "random_state": 42,
        },
    },
    "subsequence_lof": {
        "name":      "Sub-LOF",
        "training":  "unsupervised",
        "principle": "distance",
        # Top performer in the TimeEval benchmark (Schmidl et al. 2022) and
        # distance-based, so sustained events suit it — but it is unsupervised
        # against a 25%-contaminated test split, which violates its rarity
        # assumption.  Underperformance here is a finding about contamination,
        # not a wasted run.
        "params": {
            "window_size": 100, "n_neighbors": 20, "leaf_size": 30,
            "distance_metric_order": 2, "random_state": 42,
        },
    },
}

# Full column set of TimeEval's dataset index.  DatasetManager.get() reads
# dataset_type, contamination, num_anomalies and the three anomaly-length
# columns unconditionally; a missing column raises KeyError, which TimeEval
# catches and reports as the misleading "Some datasets could not be found in
# DatasetManager!".  See timeeval/datasets/datasets.py::Datasets.get.
INDEX_COLUMNS = [
    "collection_name", "dataset_name", "train_path", "test_path",
    "dataset_type", "datetime_index", "split_at", "train_type",
    "train_is_normal", "input_type", "length", "dimensions", "contamination",
    "num_anomalies", "min_anomaly_length", "median_anomaly_length",
    "max_anomaly_length", "mean", "stddev", "trend", "stationarity",
    "period_size",
]


def algo_dir(work_dir: Path, algorithm: str) -> Path:
    """
    Per-detector output directory inside a single per-dataset work dir.

    Layout — one work dir per DATASET, one subdirectory per detector:

        <work_dir>/
          timeeval_input/            converted ONCE, shared by every detector
            <machine>/metric_NN/{train,test}.csv
            <machine>/labels_per_metric.npy      <- the ground truth
            datasets.csv
          <algorithm>/
            timeeval_results/        TimeEval's own tree
            anomaly_scores/          [T, M] arrays + per_metric_auroc.csv
            plots/

    The shared conversion is the point.  `timeeval_input/` is a pure function of
    the raw dataset — byte-identical for every detector — and it holds
    `labels_per_metric.npy`, which every AUROC is computed against.  Giving each
    detector its own copy wastes ~3.9 MB per run, but far worse, it lets two
    copies drift if step 1 ever changes between runs: two detectors would then
    be scored against different ground truth while still looking comparable.
    One copy makes that impossible by construction.
    """
    return work_dir / algorithm


def _anomaly_runs(labels: np.ndarray) -> np.ndarray:
    """Lengths of contiguous runs of 1s in a binary series."""
    padded = np.concatenate([[0], np.asarray(labels).astype(np.int8), [0]])
    d      = np.diff(padded)
    starts = np.where(d == 1)[0]
    ends   = np.where(d == -1)[0]
    return ends - starts


def _index_row(collection: str, dataset_name: str, train_path: Path,
               test_path: Path, values: np.ndarray,
               metric_labels: np.ndarray) -> dict:
    """One fully-populated row of TimeEval's datasets.csv."""
    runs = _anomaly_runs(metric_labels)
    return {
        "collection_name":       collection,
        "dataset_name":          dataset_name,
        "train_path":            str(train_path),
        "test_path":             str(test_path),
        "dataset_type":          "synthetic",
        "datetime_index":        False,
        "split_at":              0,          # unused: train/test are separate files
        "train_type":            "semi-supervised",
        "train_is_normal":       True,
        "input_type":            "univariate",
        "length":                int(len(metric_labels)),
        "dimensions":            1,
        "contamination":         float(np.mean(metric_labels)),
        "num_anomalies":         int(len(runs)),
        "min_anomaly_length":    int(runs.min())            if len(runs) else 0,
        "median_anomaly_length": int(np.median(runs))       if len(runs) else 0,
        "max_anomaly_length":    int(runs.max())            if len(runs) else 0,
        "mean":                  float(np.mean(values)),
        "stddev":                float(np.std(values)),
        "trend":                 "no trend",
        "stationarity":          "not_stationary",
        "period_size":           np.nan,
    }


def step1_convert(dataset_dir: Path, work_dir: Path, machines: list, n_metrics: int):
    """
    Reads raw dataset files (auto-detects format) and writes one pair of
    per-metric CSVs per machine in TimeEval's canonical univariate format.

    Supported input formats (auto-detected):

      FORMAT A — Flat multivariate CSV (your format):
        datasets/SMD/machine-1-1.train.csv   columns: [metric_0, metric_1, ..., metric_M-1]
        datasets/SMD/machine-1-1.test.csv    columns: [metric_0, ..., metric_M-1, is_anomaly]
        (timestamp column optional, auto-detected and dropped)

      FORMAT B — Original OmniAnomaly .txt split:
        train/<machine>.txt        [T_train, M]  space-separated
        test/<machine>.txt         [T_test,  M]  space-separated
        test_label/<machine>.txt   [T_test]      binary labels

    Output (TimeEval canonical univariate format):
      work_dir/timeeval_input/
        <machine>/
          metric_00/
            train.csv   columns: timestamp, value
            test.csv    columns: timestamp, value, is_anomaly
          metric_01/
            ...
        datasets.csv    ← TimeEval DatasetManager index
    """
    banner("STEP 1 — Converting dataset to TimeEval per-metric CSV format")
    t0 = time.time()

    out_dir = work_dir / "timeeval_input"
    out_dir.mkdir(parents=True, exist_ok=True)

    collection = COLLECTION_NAME
    index_rows = []

    for machine in machines:
        train_path, test_path, label_path, fmt = _detect_format(dataset_dir, machine)

        if fmt is None:
            print(f"  [!] {machine}: no matching files found — skipping")
            print(f"      Searched for:")
            print(f"        {dataset_dir}/{machine}.train.csv  (format A)")
            print(f"        {dataset_dir}/train/{machine}.txt  (format B)")
            continue

        print(f"  {machine}: detected format '{fmt}'")

        # Load data
        train_raw, test_raw, labels = _load_machine_data(
            train_path, test_path, label_path, fmt
        )

        T_train, M_actual = train_raw.shape
        T_test            = test_raw.shape[0]

        # Per-metric ground truth, when the dataset ships it.  Falls back to
        # broadcasting the node-level column across all metrics.
        per_metric, label_src = _load_interp_labels(
            dataset_dir, machine, T_test, M_actual
        )
        if per_metric is None:
            per_metric = np.repeat(labels[:, None], M_actual, axis=1)
            label_note = "node-level labels broadcast (no interpretation file)"
        else:
            label_note = f"per-metric labels from {label_src.name}"

        print(f"    train={T_train}  test={T_test}  "
              f"metrics={M_actual}  "
              f"anomaly_rate={labels.mean():.3f}")
        print(f"    {label_note}  "
              f"per-metric rate={per_metric.mean():.3f}")

        # Write one train.csv + test.csv per metric
        for m in range(M_actual):
            metric_dir = out_dir / machine / f"metric_{m:02d}"
            metric_dir.mkdir(parents=True, exist_ok=True)

            # Training CSV.  The is_anomaly column is REQUIRED even though it is
            # all zeros: the LSTM-AD container reads its input as
            # `df.iloc[:, 1:-1]`, dropping the first and last column.  Without a
            # trailing label column that slice is empty and the model is built
            # with input_size=0.
            pd.DataFrame({
                "timestamp":  np.arange(T_train),
                "value":      train_raw[:, m],
                "is_anomaly": np.zeros(T_train, dtype=np.int32),
            }).to_csv(metric_dir / "train.csv", index=False)

            # Test CSV — timestamp + value + is_anomaly (this metric's own label)
            pd.DataFrame({
                "timestamp":  np.arange(T_test),
                "value":      test_raw[:, m],
                "is_anomaly": per_metric[:, m].astype(np.int32),
            }).to_csv(metric_dir / "test.csv", index=False)

            # Paths are stored RELATIVE to the index file's directory.
            # DatasetManager resolves them as `datasets.csv's parent / path`
            # (dataset_manager.py::_get_dataset_path_internal), so relative
            # paths keep the whole tree relocatable — absolute ones silently
            # break the moment the directory is moved or renamed.
            index_rows.append(_index_row(
                collection   = collection,
                dataset_name = f"{machine}_metric_{m:02d}",
                train_path   = Path(machine) / f"metric_{m:02d}" / "train.csv",
                test_path    = Path(machine) / f"metric_{m:02d}" / "test.csv",
                values       = test_raw[:, m],
                metric_labels= per_metric[:, m],
            ))

        # Keep the labels beside the CSVs so step 3 can score per metric without
        # re-parsing the source dataset.
        np.save(out_dir / machine / "labels_per_metric.npy",
                per_metric.astype(np.int32))
        np.save(out_dir / machine / "labels_node.npy", labels.astype(np.int32))

    # Write datasets.csv index required by TimeEval DatasetManager.  It must be
    # indexed by (collection_name, dataset_name) — DatasetManager._load_df reads
    # it with index_col=["collection_name", "dataset_name"].
    idx_path = out_dir / "datasets.csv"
    (pd.DataFrame(index_rows)
       .set_index(["collection_name", "dataset_name"])
       .to_csv(idx_path))

    total_series = len(index_rows)
    print(f"\n  ✓ {total_series} series written → {out_dir}")
    print(f"  ✓ Index → {idx_path}")
    print(f"  Done in {elapsed(t0)}")
    return out_dir


# ══════════════════════════════════════════════════════════════════
# STEP 2 — Run LSTM-AD via TimeEval
# ══════════════════════════════════════════════════════════════════

def step2_run_timeeval(
    input_dir:  Path,
    work_dir:   Path,
    machines:   list,
    n_metrics:  int,
    algo_params: dict,
    algorithm:  str = "lstm_ad",
):
    """
    Runs the selected detector on every (machine, metric) series via TimeEval.

    Which detector is chosen matters more than it looks — see ALGORITHMS above
    for the prediction / reconstruction / distance distinction and the measured
    consequence for sustained anomalies.

    What LSTM-AD computes (verified against TimeEval 1.5.0 and
    TimeEval-algorithms/lstm_ad 0.3.0):
      - TRAIN: a stacked LSTM is fitted on train.csv; the last `1 - split` of
        that series is held out, and a per-dimension Gaussian (mean, var) is
        fitted to the absolute prediction errors on it.
      - EXECUTE: errors e = |y - y_hat| are averaged over the prediction
        dimension, then scored as  (e - mu) * var^-1 * (e - mu)  — the
        variance-normalised squared deviation the paper (SIII-B) describes.
      - The container emits one score per window, i.e. T - window_size -
        prediction_window_size + 1 values.  TimeEval's registered postprocess
        for this algorithm, ReverseWindowing(window_size + prediction_window_size),
        expands that back to exactly T, then min-max scales it to [0, 1].

    Every registered algorithm has its own postprocess that restores length T,
    so step 3 needs no per-algorithm special cases.

    Result path (NOTE the timestamp and hyper-parameter directories):
      <results_dir>/<YYYY_mm_dd_HH_MM_SS>/<AlgoName>/<params_id>/
          <collection>/<dataset_name>/<repetition>/anomaly_scores.ts

    Prerequisites: pip install timeeval, Docker running
    """
    spec = ALGORITHMS[algorithm]
    banner(f"STEP 2 — Running {spec['name']} via TimeEval")
    t0 = time.time()

    try:
        import importlib
        from timeeval import TimeEval, DatasetManager
        from timeeval.params import FixedParameters
        algo_factory = getattr(
            importlib.import_module("timeeval.algorithms"), algorithm)
    except ImportError as exc:
        print(f"\n  [ERROR] Could not import TimeEval: {exc}")
        print("          Run: pip install timeeval==1.5.0")
        print("          (it pins dask==2022.12.1 and numpy<2 — install it in a")
        print("           venv separate from the RL environment)")
        print("          Also verify Docker is running: docker ps\n")
        sys.exit(1)

    results_dir = algo_dir(work_dir, algorithm) / "timeeval_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    idx_path = input_dir / "datasets.csv"
    if not idx_path.exists():
        print(f"  [ERROR] datasets.csv not found at {idx_path}")
        print("          Run step 1 first (remove --skip-convert)")
        sys.exit(1)

    dm = DatasetManager(input_dir, create_if_missing=False)

    # Select datasets for requested machines only.
    #
    # Datasets.select() has no `dataset_names` list argument — it filters by a
    # single `collection`/`dataset` name plus metadata predicates — and returns
    # (collection_name, dataset_name) tuples. Those tuples are the dataset IDs
    # TimeEval wants, so build them directly and keep only the ones the index
    # actually contains.
    available = set(dm.select(collection=COLLECTION_NAME))
    if machines:
        wanted   = [(COLLECTION_NAME, f"{m}_metric_{i:02d}")
                    for m in machines for i in range(n_metrics)]
        datasets = [d for d in wanted if d in available]
        skipped  = [d for d in wanted if d not in available]
        if skipped:
            print(f"  [!] {len(skipped)} requested series are not in the index, "
                  f"e.g. {skipped[0][1]}")
    else:
        datasets = sorted(available)

    if not datasets:
        print(f"  [ERROR] No datasets selected from {idx_path}.")
        print("          Check that dataset_name values match "
              "'<machine>_metric_NN' and that --n-metrics is right.")
        sys.exit(1)

    n_series = len(datasets)
    print(f"  Running {spec['name']} on {n_series} series")
    print(f"  ({len(machines)} machines × {n_metrics} metrics)")
    print(f"  training={spec['training']}  principle={spec['principle']}")
    print(f"  params: {algo_params}")
    print(f"  Estimated time: ~{n_series * 2} minutes on CPU\n")

    te = TimeEval(
        dm,
        datasets,
        algorithms=[algo_factory(params=FixedParameters(algo_params))],
        results_path=results_dir,
        metrics=[],          # skip metric computation — raw scores only
        distributed=False,
        # An unsupervised algorithm against our semi-supervised dataset index is
        # a deliberate, valid combination (it simply ignores the train split);
        # without this TimeEval's compatibility filter silently drops every
        # experiment and then asserts that none were configured.
        force_training_type_match=False,
    )
    te.run()

    # TimeEval nests every run under a start-timestamp directory; that is the
    # path step 3 has to search, not the base results_dir.
    run_dir = Path(te.results_path)
    print(f"\n  ✓ TimeEval finished. Scores → {run_dir}")
    print(f"  Done in {elapsed(t0)}")
    return run_dir


# ══════════════════════════════════════════════════════════════════
# STEP 3 — Extract, align, normalise scores
# ══════════════════════════════════════════════════════════════════

def _find_score_file(results_dir: Path, ds_name: str, algo_name: str = None):
    """
    Locate one series' anomaly_scores.ts under a TimeEval results tree.

    The layout is
      <run_dir>/<algorithm>/<params_id>/<collection>/<dataset_name>/<rep>/anomaly_scores.ts
    (timeeval/utils/results_path.py::generate_experiment_path), and <run_dir>
    is itself a start-timestamp directory created by TimeEval.  Matching on the
    dataset-name *directory* is what makes this robust to all of those levels.

    A glob like `*{ds_name}*anomaly_scores.ts` cannot work here: `*` never
    matches across a path separator, and the name and the file sit in different
    components.
    """
    matches = sorted(results_dir.glob(f"**/{ds_name}/*/anomaly_scores.ts"))
    if not matches:
        matches = sorted(p for p in results_dir.rglob("anomaly_scores.ts")
                         if ds_name in p.parts)
    # Keep only this algorithm's results.  Without this, pointing step 3 at a
    # work dir where several detectors have run would silently mix them.
    if algo_name:
        filtered = [p for p in matches if algo_name in p.parts]
        if filtered:
            matches = filtered
    if not matches:
        return None
    # Newest wins if the same series was run more than once.
    return max(matches, key=lambda p: p.stat().st_mtime)


def _auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """
    Rank-based AUROC with tie handling, computed from the Mann-Whitney U
    statistic.  Implemented locally so step 3 does not require scikit-learn in
    the TimeEval venv.  Returns NaN when only one class is present.
    """
    y = np.asarray(y_true).astype(bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    s_sorted = np.asarray(scores)[order]
    i = 0
    while i < len(s_sorted):                     # average ranks within ties
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def step3_extract_scores(
    results_dir:    Path,
    input_dir:      Path,
    work_dir:       Path,
    machines:       list,
    n_metrics:      int,
    allow_missing:  bool = False,
    algorithm:      str = "lstm_ad",
) -> dict:
    """
    Reads TimeEval's anomaly_scores.ts files and assembles them into
    per-machine numpy arrays of shape [T_test, M].

    What TimeEval has already done to these files
    ---------------------------------------------
    `anomaly_scores.ts` is NOT the container's raw output.  TimeEval writes two
    files (timeeval/_core/experiments.py::Experiment.evaluate):

        raw_anomaly_scores.ts  container output after `postprocess`
        anomaly_scores.ts      the above, min-max scaled to [0, 1]

    Because the registered postprocess for LSTM-AD is
    ReverseWindowing(window_size + prediction_window_size), the file is already
    exactly T_test long and already in [0, 1].  So there is nothing to pad and
    nothing to rescale here; a length mismatch means something went wrong
    upstream and is treated as an error rather than papered over with zeros.

    The one thing still worth doing is replacing NaN/Inf: TimeEval's scaler
    masks non-finite values out of the min-max fit and leaves them in place, so
    they can reach us intact and would silently poison every downstream metric.

    Per-metric AUROC is computed here against the metric's OWN label column,
    which step 1 takes from the interpretation-label file when one exists.
    """
    spec = ALGORITHMS.get(algorithm, {"name": algorithm})
    banner(f"STEP 3 — Extracting and validating {spec['name']} scores")
    t0 = time.time()

    scores_dir = algo_dir(work_dir, algorithm) / "anomaly_scores"
    scores_dir.mkdir(parents=True, exist_ok=True)

    all_scores = {}
    summaries  = []
    failures   = []

    for machine in machines:
        print(f"\n  [{machine}]")

        # Reference test CSV to get T_test
        ref_csv = input_dir / machine / "metric_00" / "test.csv"
        if not ref_csv.exists():
            print(f"    [!] Reference CSV not found — skipping")
            continue

        T_test = len(pd.read_csv(ref_csv, usecols=["timestamp"]))

        # Per-metric labels written by step 1; node-level column for reporting.
        pm_path = input_dir / machine / "labels_per_metric.npy"
        nd_path = input_dir / machine / "labels_node.npy"
        per_metric = (np.load(pm_path) if pm_path.exists()
                      else np.zeros((T_test, n_metrics), dtype=np.int32))
        labels = (np.load(nd_path) if nd_path.exists()
                  else per_metric.max(axis=1).astype(np.int32))

        score_matrix = np.zeros((T_test, n_metrics), dtype=np.float32)
        aurocs       = []

        for m in range(n_metrics):
            ds_name    = f"{machine}_metric_{m:02d}"
            score_file = _find_score_file(results_dir, ds_name, spec["name"])

            if score_file is None:
                failures.append(f"{ds_name}: no anomaly_scores.ts found under "
                                f"{results_dir}")
                print(f"    metric_{m:02d}: [FAIL] score file not found")
                aurocs.append(float("nan"))
                continue

            raw = np.loadtxt(score_file, dtype=np.float64, ndmin=1)

            if len(raw) != T_test:
                failures.append(
                    f"{ds_name}: got {len(raw)} scores, expected {T_test}. "
                    f"Expected ReverseWindowing(window_size + "
                    f"prediction_window_size) to restore full length."
                )
                print(f"    metric_{m:02d}: [FAIL] length {len(raw)} != {T_test}")
                aurocs.append(float("nan"))
                continue

            n_bad = int((~np.isfinite(raw)).sum())
            if n_bad:
                finite = raw[np.isfinite(raw)]
                fill   = float(finite.min()) if finite.size else 0.0
                raw    = np.where(np.isfinite(raw), raw, fill)
                print(f"    [note] metric_{m:02d}: {n_bad} non-finite scores "
                      f"→ filled with {fill:.4f}")

            # Idempotent when TimeEval already scaled (it does); kept so the
            # step also works on raw_anomaly_scores.ts.
            s_min, s_max = raw.min(), raw.max()
            if s_max - s_min > 1e-12:
                scores = ((raw - s_min) / (s_max - s_min)).astype(np.float32)
            else:
                scores = np.zeros(T_test, dtype=np.float32)
                failures.append(f"{ds_name}: constant score — {spec['name']} "
                                f"found no variation")

            score_matrix[:, m] = scores

            auroc = _auroc(per_metric[:, m], scores)
            aurocs.append(auroc)
            y = per_metric[:, m]
            if y.sum() and y.sum() < len(y):
                gap = float(scores[y == 1].mean() - scores[y == 0].mean())
                flag = "✓" if auroc >= 0.7 else ("~" if auroc >= 0.5 else "✗")
                print(f"    metric_{m:02d}: [{flag}]  AUROC={auroc:.3f}  "
                      f"pos_rate={y.mean():.3f}  gap={gap:+.4f}")
            else:
                print(f"    metric_{m:02d}: degenerate labels "
                      f"(pos_rate={y.mean():.3f}) — AUROC undefined")

        # ── Save ────────────────────────────────────────────────────────
        np.save(scores_dir / f"{machine}_scores.npy", score_matrix)
        np.save(scores_dir / f"{machine}_labels.npy", labels)
        np.save(scores_dir / f"{machine}_labels_per_metric.npy", per_metric)

        all_scores[machine] = {
            "scores":            score_matrix,
            "labels":            labels,
            "labels_per_metric": per_metric,
            "auroc":             aurocs,
        }
        finite_auroc = [a for a in aurocs if np.isfinite(a)]
        summaries.append({
            "machine":     machine,
            "T":           T_test,
            "M":           n_metrics,
            "mean_auroc":  float(np.mean(finite_auroc)) if finite_auroc else float("nan"),
            "min_auroc":   float(np.min(finite_auroc)) if finite_auroc else float("nan"),
            "anom_rate":   float(labels.mean()),
        })
        print(f"\n    Saved: {machine}_scores.npy  shape={score_matrix.shape}")

    # Per-metric AUROC table, so a dead metric is visible rather than silent.
    auroc_rows = [
        {"machine": mach, "metric": m, "auroc": d["auroc"][m],
         "pos_rate": float(d["labels_per_metric"][:, m].mean())}
        for mach, d in all_scores.items() for m in range(n_metrics)
    ]
    auroc_csv = scores_dir / "per_metric_auroc.csv"
    pd.DataFrame(auroc_rows).to_csv(auroc_csv, index=False)

    print(f"\n{'─'*66}")
    print(f"  {'Machine':<16} {'T':>7} {'M':>5} {'meanAUROC':>10} {'minAUROC':>9} {'Anom%':>7}")
    print(f"  {'─'*62}")
    for s in summaries:
        print(f"  {s['machine']:<16} {s['T']:>7} {s['M']:>5} "
              f"{s['mean_auroc']:>10.3f} {s['min_auroc']:>9.3f} "
              f"{s['anom_rate']*100:>6.1f}%")
    print(f"{'─'*66}")
    print(f"  Per-metric AUROC → {auroc_csv}")

    if failures:
        print(f"\n  [ERROR] {len(failures)} series did not produce usable scores:")
        for f in failures[:20]:
            print(f"    - {f}")
        if len(failures) > 20:
            print(f"    ... and {len(failures) - 20} more")
        print("\n  Refusing to continue with placeholder scores. Re-run step 2,")
        print("  or pass --allow-missing-scores to accept zeros for the failures.")
        if not allow_missing:
            sys.exit(1)

    print(f"  Done in {elapsed(t0)}")
    return all_scores


# ══════════════════════════════════════════════════════════════════
# STEP 4 — Verify and visualise scores
# ══════════════════════════════════════════════════════════════════

def step4_verify(work_dir: Path, machines: list, plot: bool,
                 algorithm: str = "lstm_ad"):
    """
    Sanity checks on extracted scores and optional heatmap plots.

    Checks:
      1. Values in [0, 1]
      2. Mean score on anomalous timesteps > mean on normal
      3. No all-zero metrics (would mean the detector failed)
    """
    banner("STEP 4 — Verifying scores")

    scores_dir = algo_dir(work_dir, algorithm) / "anomaly_scores"
    plots_dir  = algo_dir(work_dir, algorithm) / "plots"
    if plot:
        plots_dir.mkdir(parents=True, exist_ok=True)

    all_ok = True

    for machine in machines:
        score_path = scores_dir / f"{machine}_scores.npy"
        label_path = scores_dir / f"{machine}_labels.npy"

        if not score_path.exists():
            print(f"  [!] {machine}: score file not found — skipping verify")
            continue

        scores = np.load(score_path)   # [T, M]
        labels = np.load(label_path)   # [T]
        T, M   = scores.shape

        # Prefer per-metric ground truth: comparing metric m's scores against
        # the node-level union label mixes in anomalies from other metrics.
        pm_path = scores_dir / f"{machine}_labels_per_metric.npy"
        per_metric = (np.load(pm_path) if pm_path.exists()
                      else np.repeat(labels[:, None], M, axis=1))

        # Check 1: range
        c1 = scores.min() >= 0.0 and scores.max() <= 1.0

        # Check 2: signal quality, averaged over metrics with both classes
        gaps = [float(scores[per_metric[:, m] == 1, m].mean()
                      - scores[per_metric[:, m] == 0, m].mean())
                for m in range(M)
                if 0 < per_metric[:, m].sum() < T]
        if gaps:
            mean_anom   = float(np.mean([scores[per_metric[:, m] == 1, m].mean()
                                         for m in range(M)
                                         if 0 < per_metric[:, m].sum() < T]))
            mean_normal = float(np.mean([scores[per_metric[:, m] == 0, m].mean()
                                         for m in range(M)
                                         if 0 < per_metric[:, m].sum() < T]))
            c2 = float(np.mean(gaps)) > 0
        else:
            mean_anom = mean_normal = float("nan")
            c2 = None

        # Check 3: non-zero metrics
        zero_metrics = int((scores.max(axis=0) < 1e-6).sum())
        c3 = zero_metrics == 0

        status = "OK" if (c1 and c2 and c3) else "WARN"
        all_ok = all_ok and (c1 and c2 and c3)

        print(f"\n  {machine}  [{status}]")
        print(f"    Shape:           [{T}, {M}]")
        print(f"    Values in [0,1]: {'✓' if c1 else '✗'}")
        if c2 is not None:
            print(f"    Signal quality:  {'✓' if c2 else '✗'}  "
                  f"anom={mean_anom:.4f}  normal={mean_normal:.4f}")
        print(f"    Zero metrics:    {'✓ none' if c3 else f'✗ {zero_metrics} metrics have all-zero scores'}")

        # Optional heatmap
        if plot:
            _plot_heatmap(machine, scores, labels, plots_dir)

    if all_ok:
        print("\n  ✓ All checks passed — scores look correct")
    else:
        print("\n  ✗ Some checks failed — review metrics marked ✗ above")
        print("    Common causes:")
        print("      - LSTM-AD training failed (Docker OOM, too few timesteps)")
        print("      - Signal quality low: try increasing --lstm-epochs")


def _plot_heatmap(machine: str, scores: np.ndarray, labels: np.ndarray, plots_dir: Path):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("    [plot] matplotlib not available — pip install matplotlib")
        return

    T, M = scores.shape
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 7),
        gridspec_kw={"height_ratios": [5, 1]}
    )

    # Heatmap: metrics on y-axis, time on x-axis
    im = ax1.imshow(
        scores.T, aspect="auto", cmap="YlOrRd",
        vmin=0, vmax=1, interpolation="nearest"
    )
    plt.colorbar(im, ax=ax1, label="Anomaly score [0,1]")

    # Shade anomaly regions
    changes = np.diff(np.concatenate([[0], labels, [0]]))
    for s, e in zip(np.where(changes == 1)[0], np.where(changes == -1)[0]):
        ax1.axvspan(s, e, alpha=0.2, color="blue", linewidth=0)

    ax1.set_ylabel("Metric index")
    ax1.set_title(f"{machine} — LSTM-AD anomaly scores (per-metric normalised to [0,1])")

    anom_patch = mpatches.Patch(color="blue", alpha=0.3, label="Anomaly period")
    ax1.legend(handles=[anom_patch], fontsize=8, loc="upper right")

    # Label row
    ax2.fill_between(np.arange(T), labels, color="red", alpha=0.6, step="pre")
    ax2.set_xlim(0, T)
    ax2.set_ylim(0, 1.2)
    ax2.set_ylabel("Label")
    ax2.set_xlabel("Timestep")
    ax2.set_yticks([0, 1])

    plt.tight_layout()
    out_path = plots_dir / f"{machine}_heatmap.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    [plot] Saved → {out_path}")


# ══════════════════════════════════════════════════════════════════
# STEP 5 — Run RL agent on extracted scores
# ══════════════════════════════════════════════════════════════════

def step5_evaluate_rl(
    work_dir:  Path,
    machines:  list,
    model_path: str,
    config_path: str,
    device_str: str,
):
    """
    Loads the trained RL agent and runs it on the extracted anomaly scores.

    For each machine and each timestep:
      - Builds the state vector [M*4] from:
          [anomaly_scores_t, causal_scores, prev_action, available_capacity]
      - Agent selects a binary action (monitor / don't monitor per metric)
      - Records which metrics were selected vs which had highest anomaly scores
      - Computes: TopK precision, score gap, anomaly coverage

    Key question answered:
      "Does the agent select the metrics with the highest anomaly scores?"

    Note on causal scores:
      If no causal score file is found for a machine, uniform 0.5 is used.
      To use real causal scores (from Tigramite PCMCI+), place them at:
        data/causal_scores/<machine>_causal.npy   shape [M]
    """
    banner("STEP 5 — Evaluating RL agent on extracted scores")

    import torch
    import yaml
    from agents.policy import ActorCritic

    # Load config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(
        device_str if device_str else
        ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"  Device: {device}")

    scores_dir  = work_dir / "anomaly_scores"
    causal_dir  = Path("data/causal_scores")
    out_dir     = work_dir / "rl_evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load policy
    if not Path(model_path).exists():
        print(f"  [ERROR] Model not found at {model_path}")
        print("          Train first: PYTHONPATH=. python scripts/train.py")
        return

    # Detect M from first available score file
    first_scores = list(scores_dir.glob("*_scores.npy"))
    if not first_scores:
        print("  [ERROR] No score files found in", scores_dir)
        return
    M = np.load(first_scores[0]).shape[1]
    N = 1   # one node per machine

    pcfg   = cfg["policy"]
    policy = ActorCritic(
        obs_dim        = N * M * 4,
        action_dim     = N * M,
        hidden_sizes   = pcfg["hidden_sizes"],
        use_layer_norm = pcfg["use_layer_norm"],
    )
    policy.load_state_dict(torch.load(model_path, map_location=device))
    policy.to(device)
    policy.eval()
    print(f"  Model loaded from {model_path}  (obs={N*M*4}, action={N*M})")

    ecfg     = cfg["environment"]
    capacity = float(int(M * 0.5))    # default: 50% of metrics

    all_agg      = []
    all_records  = {}

    for machine in machines:
        score_path  = scores_dir / f"{machine}_scores.npy"
        label_path  = scores_dir / f"{machine}_labels.npy"
        causal_path = causal_dir  / f"{machine}_causal.npy"

        if not score_path.exists():
            print(f"  [!] {machine}: scores not found — skipping")
            continue

        anomaly_scores = np.load(score_path)   # [T, M]
        labels         = np.load(label_path)   # [T]
        causal_scores  = (np.load(causal_path).astype(np.float32)
                          if causal_path.exists()
                          else np.full(M, 0.5, dtype=np.float32))

        T = anomaly_scores.shape[0]

        # Baselines from this machine's distribution
        b_a = float(np.percentile(anomaly_scores, ecfg["anomaly_baseline_pct"]))
        b_c = float(np.percentile(causal_scores,  ecfg["causal_baseline_pct"]))

        print(f"\n  {machine}: T={T}  M={M}  cap={capacity:.0f}  "
              f"b_a={b_a:.4f}  anom_rate={labels.mean():.3f}")

        prev_action = np.zeros(M, dtype=np.float32)
        records     = []

        with torch.no_grad():
            for t in range(T):
                A_t = anomaly_scores[t]   # [M]

                # Build state [M*4]
                selected   = prev_action.sum()
                avail_cap  = float(np.clip(
                    (capacity - selected) / (capacity + 1e-8), 0.0, 1.0
                ))
                cap_feat   = np.full(M, avail_cap, dtype=np.float32)
                obs        = np.stack([
                    A_t, causal_scores, prev_action, cap_feat
                ], axis=-1).reshape(-1)   # [M*4]

                obs_t  = torch.tensor(obs, dtype=torch.float32,
                                      device=device).unsqueeze(0)
                action = policy.get_deterministic_action(obs_t) \
                               .squeeze(0).cpu().numpy()   # [M]

                n_selected = int(action.sum())

                # TopK precision: do agent's selections match top-n by anomaly score?
                k = max(n_selected, 1)
                topk_idx  = set(np.argsort(A_t)[::-1][:k])
                agent_idx = set(np.where(action == 1)[0])
                precision = len(topk_idx & agent_idx) / k

                # Score gap: mean score of selected vs unselected
                if n_selected > 0 and n_selected < M:
                    gap = float(A_t[action == 1].mean()) - \
                          float(A_t[action == 0].mean())
                else:
                    gap = 0.0

                # Anomaly coverage
                label_t = int(labels[t])
                covered = int(label_t > 0 and n_selected > 0)

                records.append({
                    "t":              t,
                    "n_selected":     n_selected,
                    "topk_precision": precision,
                    "score_gap":      gap,
                    "is_anomaly":     label_t,
                    "anomaly_covered":covered,
                    "max_score":      float(A_t.max()),
                    "mean_selected_score": float(A_t[action==1].mean())
                                           if n_selected > 0 else 0.0,
                })

                prev_action = action.copy()

        # Aggregate
        n_anom = sum(r["is_anomaly"] for r in records)
        agg = {
            "machine":          machine,
            "T":                T,
            "M":                M,
            "capacity":         capacity,
            "anomaly_rate":     float(labels.mean()),
            "mean_selected":    float(np.mean([r["n_selected"] for r in records])),
            "topk_precision":   float(np.mean([r["topk_precision"] for r in records])),
            "score_gap":        float(np.mean([r["score_gap"] for r in records])),
            "anomaly_coverage": float(np.mean([r["anomaly_covered"] for r in records]))
                                if n_anom > 0 else float("nan"),
        }
        all_agg.append(agg)
        all_records[machine] = records

        print(f"    TopK Precision : {agg['topk_precision']:.3f}  "
              f"(fraction of selections in top-K anomaly metrics)")
        print(f"    Score Gap      : {agg['score_gap']:+.4f}  "
              f"(selected score − unselected score, should be positive)")
        print(f"    Coverage       : {agg['anomaly_coverage']:.3f}  "
              f"(anomalous timesteps where agent selected ≥1 metric)")
        print(f"    Mean Selected  : {agg['mean_selected']:.1f} / {M}")

    # Print final summary table
    _print_rl_summary(all_agg)

    # Save results
    _save_rl_results(all_agg, all_records, out_dir)

    # Plot per-machine selection behaviour
    if all_records:
        _plot_rl_results(all_records, out_dir)

    print(f"\n  Results saved → {out_dir}/")


def _print_rl_summary(all_agg: list):
    print(f"\n{'═'*72}")
    print(f"  RL Agent Evaluation Summary")
    print(f"{'═'*72}")
    h = (f"  {'Machine':<16} {'Selected':>9} {'TopK-P':>8} "
         f"{'ScoreGap':>10} {'Coverage':>10}")
    print(h)
    print(f"  {'─'*60}")
    for r in all_agg:
        print(
            f"  {r['machine']:<16} "
            f"{r['mean_selected']:>9.1f} "
            f"{r['topk_precision']:>8.3f} "
            f"{r['score_gap']:>+10.4f} "
            f"{r['anomaly_coverage']:>10.3f}"
        )

    if all_agg:
        prec = np.mean([r["topk_precision"] for r in all_agg])
        gap  = np.mean([r["score_gap"]      for r in all_agg])
        cov  = np.nanmean([r["anomaly_coverage"] for r in all_agg])
        print(f"  {'─'*60}")
        print(f"  {'MEAN':<16} "
              f"{'':>9} "
              f"{prec:>8.3f} "
              f"{gap:>+10.4f} "
              f"{cov:>10.3f}")

        print(f"\n  Interpretation:")
        print(f"    TopK Precision {prec:.3f}: "
              + ("✓ agent selects high-anomaly metrics"
                 if prec > 0.6 else "✗ selections don't align with top scores"))
        print(f"    Score Gap {gap:+.4f}: "
              + ("✓ selected metrics have higher scores than unselected"
                 if gap > 0 else "✗ agent not preferring high-score metrics"))
        print(f"    Coverage {cov:.3f}: "
              + ("✓ most anomaly periods have at least one metric selected"
                 if cov > 0.7 else "✗ agent missing anomaly periods"))
    print(f"{'═'*72}")


def _save_rl_results(all_agg: list, all_records: dict, out_dir: Path):
    # Summary CSV
    if all_agg:
        keys = list(all_agg[0].keys())
        with open(out_dir / "evaluation_results.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_agg)

    # Per-timestep records per machine
    for machine, records in all_records.items():
        if not records:
            continue
        keys = list(records[0].keys())
        with open(out_dir / f"{machine}_records.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(records)


def _plot_rl_results(all_records: dict, out_dir: Path):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        return

    for machine, records in all_records.items():
        if not records:
            continue

        t_axis   = [r["t"]             for r in records]
        selected = [r["n_selected"]    for r in records]
        topk_p   = [r["topk_precision"] for r in records]
        gap      = [r["score_gap"]     for r in records]
        anom     = [r["is_anomaly"]    for r in records]
        sel_score= [r["mean_selected_score"] for r in records]

        fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
        fig.suptitle(f"{machine} — RL Agent on LSTM-AD scores", fontsize=13)

        # Panel 1: selected metrics count + anomaly shading
        ax = axes[0]
        ax.plot(t_axis, selected, color="#2196F3", linewidth=1, label="# selected")
        ax.plot(t_axis, sel_score, color="#FF9800", linewidth=1,
                alpha=0.8, label="mean score of selected")
        for t, flag in zip(t_axis, anom):
            if flag:
                ax.axvspan(t, t+1, alpha=0.1, color="red", linewidth=0)
        ax.set_ylabel("Count / Score")
        anom_patch = mpatches.Patch(color="red", alpha=0.2, label="Anomaly")
        ax.legend(handles=ax.get_legend_handles_labels()[0]+[anom_patch],
                  fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)

        # Panel 2: TopK precision
        ax = axes[1]
        ax.plot(t_axis, topk_p, color="#4CAF50", linewidth=1)
        ax.axhline(np.mean(topk_p), color="#4CAF50", linestyle="--",
                   linewidth=1, label=f"mean={np.mean(topk_p):.3f}")
        ax.set_ylabel("TopK Precision")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Panel 3: Score gap
        ax = axes[2]
        ax.plot(t_axis, gap, color="#9C27B0", linewidth=1)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.axhline(np.mean(gap), color="#9C27B0", linestyle="--",
                   linewidth=1, label=f"mean={np.mean(gap):+.4f}")
        ax.set_ylabel("Score Gap (sel − unsel)")
        ax.set_xlabel("Timestep")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = out_dir / f"{machine}_agent_behaviour.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"    [plot] Saved → {path}")


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    dataset_dir = Path(args.dataset_dir)
    work_dir    = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # ── Auto-discover machines ───────────────────────────────────
    if args.machines:
        machines = args.machines
    else:
        dataset_dir = Path(args.dataset_dir)

        # Format A: flat CSV files  (machine-1-1.train.csv)
        csv_machines = sorted(
            p.stem.replace(".train", "")
            for p in dataset_dir.glob("*.train.csv")
        )

        # Format B: txt split into subfolders  (train/machine-1-1.txt)
        txt_machines = []
        train_subdir = dataset_dir / "train"
        if train_subdir.exists():
            txt_machines = sorted(p.stem for p in train_subdir.glob("*.txt"))

        machines = csv_machines if csv_machines else txt_machines

        if not machines:
            print(f"[ERROR] Could not find any machine files in {dataset_dir}")
            print(f"        Expected either:")
            print(f"          {dataset_dir}/machine-1-1.train.csv  (format A)")
            print(f"          {dataset_dir}/train/machine-1-1.txt  (format B)")
            sys.exit(1)

    # ── Auto-detect n_metrics ────────────────────────────────────
    if args.n_metrics:
        n_metrics = args.n_metrics
    else:
        # Try format A first
        first_csv = dataset_dir / f"{machines[0]}.train.csv"
        if first_csv.exists():
            df = pd.read_csv(first_csv, nrows=1)
            # Drop timestamp if present, drop is_anomaly if present
            cols = [c for c in df.columns
                    if c not in ("timestamp", "is_anomaly")
                    and not c.lower().startswith("time")]
            n_metrics = len(cols)
        else:
            # Format B: count columns in .txt
            first_txt = dataset_dir / "train" / f"{machines[0]}.txt"
            sample    = np.loadtxt(first_txt, max_rows=1)
            n_metrics = 1 if sample.ndim == 0 else len(sample)

    print(f"\n{'═'*60}")
    print(f"  TimeEval LSTM-AD → RL Evaluation Pipeline")
    print(f"{'═'*60}")
    print(f"  Dataset dir : {dataset_dir}")
    print(f"  Work dir    : {work_dir}")
    print(f"  Machines    : {len(machines)} → {machines}")
    print(f"  Metrics/machine: {n_metrics}")
    print(f"  Algorithm   : {ALGORITHMS[args.algorithm]['name']} "
          f"({ALGORITHMS[args.algorithm]['training']}, "
          f"{ALGORITHMS[args.algorithm]['principle']}-based)")
    print(f"  Model       : {args.model}")

    # Parameters come from the ALGORITHMS registry, which holds ONLY keys the
    # container declares in its manifest — unknown keys are silently discarded
    # by the container and would only perturb TimeEval's hyper-parameter hash.
    spec        = ALGORITHMS[args.algorithm]
    algo_params = dict(spec["params"])

    # The --lstm-* / --window-size flags predate --algorithm and only apply to
    # lstm_ad. Applying them to another detector would inject keys it does not
    # declare, so they are restricted rather than silently ignored.
    if args.algorithm == "lstm_ad":
        algo_params.update({
            "epochs":                 args.lstm_epochs,
            "lstm_layers":            args.lstm_layers,
            "window_size":            args.window_size,
            "prediction_window_size": args.prediction_window_size,
            "batch_size":             args.lstm_batch_size,
            "split":                  args.lstm_split,
            "random_state":           args.lstm_seed,
        })

    # Generic per-algorithm overrides
    for kv in args.algo_param:
        if "=" not in kv:
            print(f"[ERROR] --algo-param needs K=V, got {kv!r}")
            sys.exit(1)
        k, v = kv.split("=", 1)
        if k not in algo_params:
            print(f"[ERROR] {args.algorithm} has no registry parameter {k!r}. "
                  f"Known: {', '.join(sorted(algo_params))}")
            sys.exit(1)
        for cast in (int, float):
            try:
                v = cast(v); break
            except ValueError:
                continue
        algo_params[k] = v

    # ── Step 1: Convert ─────────────────────────────────────────
    # The conversion is shared by every detector in this work dir, so it is
    # reused when already present rather than rebuilt per detector. Rebuilding
    # it would be wasted work at best; at worst two detectors end up scored
    # against different copies of labels_per_metric.npy.
    input_dir = work_dir / "timeeval_input"
    already   = (input_dir / "datasets.csv").exists()
    if args.skip_convert or (already and not args.force_convert):
        why = "--skip-convert" if args.skip_convert else "already present"
        print(f"\n  [SKIP] Step 1 — reusing shared conversion ({why})")
        print(f"         {input_dir}")
        if not already:
            print(f"  [ERROR] ...but {input_dir / 'datasets.csv'} does not "
                  f"exist. Drop --skip-convert.")
            sys.exit(1)
    else:
        input_dir = step1_convert(dataset_dir, work_dir, machines, n_metrics)

    # ── Step 2: Run TimeEval ─────────────────────────────────────
    results_dir = algo_dir(work_dir, args.algorithm) / "timeeval_results"
    if not args.skip_timeeval:
        results_dir = step2_run_timeeval(
            input_dir, work_dir, machines, n_metrics, algo_params,
            algorithm=args.algorithm,
        )
    else:
        # TimeEval nests each run under a start-timestamp directory. Searching
        # from the base directory would work but could mix two runs, so pin to
        # the most recent one.
        runs = sorted(d for d in results_dir.glob("*") if d.is_dir())
        if runs:
            results_dir = runs[-1]
        print(f"\n  [SKIP] Step 2 — using existing TimeEval results in {results_dir}")

    # ── Step 3: Extract scores ───────────────────────────────────
    all_scores = step3_extract_scores(
        results_dir   = results_dir,
        input_dir     = input_dir,
        work_dir      = work_dir,
        machines      = machines,
        n_metrics     = n_metrics,
        allow_missing = args.allow_missing_scores,
        algorithm     = args.algorithm,
    )

    # ── Step 4: Verify ───────────────────────────────────────────
    step4_verify(work_dir, machines, plot=args.plot, algorithm=args.algorithm)

    # ── Step 5: RL Evaluation ────────────────────────────────────
    if not args.skip_eval:
        step5_evaluate_rl(
            work_dir    = work_dir,
            machines    = machines,
            model_path  = args.model,
            config_path = args.config,
            device_str  = args.device,
        )
    else:
        print("\n  [SKIP] Step 5 — RL evaluation skipped")
        print(f"  Score files ready in: "
              f"{algo_dir(work_dir, args.algorithm) / 'anomaly_scores'}/")
        print(f"  To evaluate later (with causal scores from pcmci_pipeline.py):")
        print(f"    python scripts/compute_metrics.py --model {args.model} \\")
        print(f"        --source derived --raw-dataset-dir {args.dataset_dir} \\")
        print(f"        --derived-anomaly-dir {algo_dir(work_dir, args.algorithm) / 'anomaly_scores'} \\")
        print(f"        --derived-causal <causal_scores.npy>")

    banner("Pipeline complete")


if __name__ == "__main__":
    main()
elif __name__ == "scripts.timeeval_pipeline":
    main()