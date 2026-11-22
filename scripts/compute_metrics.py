"""
compute_metrics.py
==================
Computes all performance metrics for every baseline and the trained RL agent,
over the full evaluation episode pool.

FIX: collect_baseline_trajectory() now ranks metrics using window-MAX
aggregated scores (same as the RL env observation) instead of scores at
the window midpoint t_mid. Using t_mid deflated baseline coverage by ~0.20
because anomaly peaks occurring outside t_mid were missed by the ranking.

The rule for fair comparison is simple:
  - RL agent:  obs = max(A[t_s:t_e]),  action held constant for [t_s, t_e)
  - Baselines: rank by max(A[t_s:t_e]), action held constant for [t_s, t_e)
  - Coverage:  labels[t_s:t_e].max()   (any anomaly in window = must cover)

All three methods now use identical information and identical granularity.

Metrics computed
----------------
Coverage Quality:
  acr              Anomaly Coverage Rate
  weighted_acr     ACR weighted by anomaly score magnitude
  fmr              False Monitoring Rate
  f1_coverage      Harmonic mean of ACR and (1 - FMR)

Detection Speed:
  latency_mean     Mean detection latency (windows)
  latency_median   Median detection latency
  latency_p90      90th-percentile detection latency
  ttfd             Time-to-First-Detection within 1 window (fraction of events)

Resource Efficiency:
  bur              Budget Utilization Rate
  cuc              Coverage per Unit Cost
  cvr              Capacity Violation Rate
  tier_cost        Mean tier-weighted monitoring cost per window

Selection Quality:
  topk_precision   Fraction of selections matching TopK-oracle ranking
  score_gap        Mean score difference selected vs unselected
  causal_align     Spearman ρ between selection frequency and causal scores

Stability:
  stability        1 - mean normalised action change between windows
  reactivity       Mean selection increase after anomaly score spike

Tier-specific (per tier):
  acr_iot / acr_edge / acr_cloud
  cvr_iot / cvr_edge / cvr_cloud
  cuc_iot / cuc_edge / cuc_cloud

Usage
-----
  # Default: loads eval_episodes.pkl from data/<N>nodes-<M>metrics/
  python scripts/compute_metrics.py
  python scripts/compute_metrics.py --run-name my_run --episodes 20

  # Compare flat MLP vs PerMetricActorCritic
  python scripts/compute_metrics.py \
      --model-flat       results/flat_run/best_model.pt \
      --model-per-metric results/pm_run/best_model.pt

  # With custom labels in the output table
  python scripts/compute_metrics.py \
      --model-flat       results/flat_run/best_model.pt  --label-flat "RL-Flat" \
      --model-per-metric results/pm_run/best_model.pt    --label-per-metric "RL-PerMetric"

  # Explicit pkl path
  python scripts/compute_metrics.py --source pkl --pkl-path data/my_pool.pkl

  # Raw numpy arrays
  python scripts/compute_metrics.py --source npy \\
      --anomaly-scores data/scores.npy \\
      --causal-scores  data/causal.npy \\
      --labels         data/labels.npy \\
      --dataset-name   my_dataset

  # Other options
  python scripts/compute_metrics.py --no-agent
  python scripts/compute_metrics.py --run-name my_run --save-detail
  python scripts/compute_metrics.py --coverage-mode per_timestep
"""

import os
import sys
import csv
import copy
import json
import argparse
import yaml
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.synthetic_data import load_dataset, build_episode_pool, save_dataset, topology_kwargs, assign_tiers, pool_dir, pool_config_key
from utils.seeding import resolve_seeds
from envs.monitoring_env import MonitoringEnv
from agents.policy import ActorCritic, PerMetricActorCritic
from baselines.baselines import MonitorEverything, RandomFeasible, TopKAnomalyCausal


# ══════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",         default="configs/config.yaml")
    p.add_argument("--run-name",       default=None,
                   help="Run under results/. Model loaded from results/<run>/best_model.pt")
    p.add_argument("--episodes",       type=int, default=None)
    p.add_argument("--out-dir",        default=None)
    p.add_argument("--no-agent",       action="store_true")
    p.add_argument("--no-baselines",  action="store_true")
    p.add_argument("--save-detail",    action="store_true")
    p.add_argument("--device",         default=None)
    p.add_argument(
        "--coverage-mode",
        default=None,
        choices=["window", "per_timestep"],
        help=(
            "'window' (correct, for paper results) or "
            "'per_timestep' (legacy, inflated by anomaly-free steps). "
            "Default: reads from config environment.coverage_mode, fallback=window."
        ),
    )

    # ── Single model (original behaviour) ────────────────────────
    p.add_argument("--model", default=None,
                   help="Path to a single model checkpoint. "
                        "Added to results as 'RL-PPO (Ours)'. "
                        "Use --model-flat / --model-per-metric to compare two models.")

    # ── Two-model comparison ──────────────────────────────────────
    p.add_argument("--model-flat", default=None,
                   help="Path to flat MLP (ActorCritic) checkpoint.")
    p.add_argument("--label-flat", default="RL-Flat MLP",
                   help="Display label for the flat model in tables and figures.")
    p.add_argument("--config-flat", default=None,
                   help="Config file for the flat model. "
                        "Default: same as --config. "
                        "Use if the flat model was trained with a different config.")

    p.add_argument("--model-per-metric", default=None,
                   help="Path to PerMetricActorCritic checkpoint.")
    p.add_argument("--label-per-metric", default="RL-PerMetric",
                   help="Display label for the per-metric model in tables and figures.")
    p.add_argument("--config-per-metric", default=None,
                   help="Config file for the per-metric model. "
                        "Default: same as --config.")

    # ── Data source ───────────────────────────────────────────────
    p.add_argument("--source", default="pkl",
                   choices=["pkl", "npy", "smd", "derived"])
    p.add_argument("--pkl-path", default=None)
    p.add_argument("--anomaly-scores", default=None)
    p.add_argument("--causal-scores",  default=None)
    p.add_argument("--labels",         default=None)
    p.add_argument("--dataset-name",   default="custom")
    # SMD-specific
    p.add_argument("--anomaly-score-dir", default=None,
                   help="Directory with <machine>.npy anomaly score files "
                        "(required for --source smd)")
    # --source derived: LSTM-AD + PCMCI+ scores computed from raw synthetic telemetry
    p.add_argument("--raw-dataset-dir", default=None,
                   help="derived: data/raw-synthetic-... (read for meta.json — "
                        "capacities, costs, tier labels)")
    p.add_argument("--derived-anomaly-dir", default=None,
                   help="derived: data/derived-<dataset>/<algorithm>/anomaly_scores")
    p.add_argument("--derived-causal", default=None,
                   help="derived: data/causal-derived-.../causal_scores.npy")
    p.add_argument("--derived-score-transform", default="none",
                   choices=["none", "rank", "match"],
                   help="derived: remap the anomaly-score marginal. 'none' uses "
                        "TimeEval's min-max output as-is. 'rank' maps each "
                        "series to its percentile rank (deployable). 'match' "
                        "quantile-maps onto the training pool's marginal "
                        "(diagnostic — isolates calibration from detector "
                        "quality by preserving every within-series ordering).")

    p.add_argument("--interp-label-dir",  default=None,
                   help="Directory with <machine>.txt interpretation label files "
                        "(required for --source smd)")
    p.add_argument("--machines", nargs="+", default=None,
                   help="Machine names to evaluate e.g. machine-1-1 machine-1-2. "
                        "Default: all .npy files in --anomaly-score-dir")
    p.add_argument("--n-metrics", type=int, default=38,
                   help="Number of metrics per SMD machine (default: 38)")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════



# ------------------------------------------------------------------
# Data loaders
# ------------------------------------------------------------------

def load_pkl(pkl_path: str) -> list:
    """Load a pre-built episode pool from a .pkl file."""
    import pickle
    from pathlib import Path
    with open(pkl_path, "rb") as f:
        episodes = pickle.load(f)
    for ep in episodes:
        ep.setdefault("name", Path(pkl_path).stem)
    print(f"[metrics] Loaded {len(episodes)} episodes ← {pkl_path}")
    return episodes


def load_npy(
    anomaly_path: str,
    causal_path: str,
    labels_path: str,
    dataset_name: str,
    cfg: dict,
) -> list:
    """
    Load raw numpy arrays and wrap them into a single episode dict.

    Supported label shapes
    ----------------------
    [T, N, M]  per-metric labels — used as-is
    [T, M]     per-metric, N=1  — expanded to [T, 1, M]
    [T]        machine-level    — broadcast to [T, N, M]
               (all metrics share the label; use load_smd() for per-metric
                ground truth from SMD interpretation labels)
    """
    A = np.load(anomaly_path).astype(np.float32)
    C = np.load(causal_path).astype(np.float32)
    L = np.load(labels_path).astype(np.int32) if labels_path else None

    if A.ndim == 2:
        A = A[:, np.newaxis, :]
    T, N, M = A.shape
    if C.ndim == 1:
        C = C[np.newaxis, :]

    if L is None:
        L = np.zeros((T, N, M), dtype=np.int32)
    elif L.ndim == 1:
        L = np.broadcast_to(
            L[:, np.newaxis, np.newaxis], (T, N, M)
        ).copy().astype(np.int32)
    elif L.ndim == 2:
        L = L[:, np.newaxis, :]

    if L.shape != (T, N, M):
        raise ValueError(f"Label shape {L.shape} != score shape {A.shape}")

    ecfg       = cfg["environment"]
    caps       = np.array([max(2, int(M * 0.5))] * N, dtype=np.float32)
    node_costs = np.full(N, ecfg["per_metric_cost"], dtype=np.float32)
    b_a        = float(np.percentile(A, ecfg["anomaly_baseline_pct"]))
    b_c        = float(np.percentile(C, ecfg["causal_baseline_pct"]))

    print(f"[metrics] Loaded npy : shape=[{T},{N},{M}]  "
          f"b_a={b_a:.4f}  anomaly_rate={L.mean():.4f}")
    return [{
        "anomaly_scores": A, "anomaly_labels": L, "causal_scores": C,
        "node_capacities": caps, "node_costs": node_costs,
        "n_nodes": N, "n_metrics": M, "T": T,
        "name": dataset_name, "b_a": b_a, "b_c": b_c,
    }]


def _rank_transform(A: np.ndarray) -> np.ndarray:
    """
    Map each (node, metric) score series to its within-series percentile rank.

    Detector-agnostic and deployable: it needs no reference distribution, only
    the series itself.  Ties get average ranks, so a long run of identical
    scores does not become an arbitrary ordering.

    Why this exists: the policy learns a decision boundary in raw score space
    (measured at A ~ 0.60 for the 10M models, matching their training
    b_a = 0.543) and that boundary is fixed in the weights — neither b_a nor any
    normalised form of the score is part of the observation, so the policy
    cannot tell that the score distribution moved underneath it.  A rank
    transform makes the marginal distribution the same regardless of the
    detector's scale, which is what the boundary implicitly assumes.
    """
    from scipy.stats import rankdata
    T, N, M = A.shape
    out = np.empty_like(A, dtype=np.float32)
    for n in range(N):
        for m in range(M):
            out[:, n, m] = (rankdata(A[:, n, m], method="average") - 1) / (T - 1)
    return out


def _match_transform(A: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Quantile-map each series onto the marginal distribution of `reference`.

    Diagnostic rather than deployable — it consumes the training score
    distribution — but it is the clean attribution test: it changes ONLY the
    marginal distribution, preserving every within-series ordering.  If
    performance recovers under it, the transfer gap is calibration; if it does
    not, the gap is the detector's ranking quality (AUROC 0.987 -> 0.768) and no
    rescaling can fix it.
    """
    ranks = _rank_transform(A)
    ref_sorted = np.sort(np.asarray(reference).ravel())
    idx = np.clip((ranks * (len(ref_sorted) - 1)).astype(np.int64),
                  0, len(ref_sorted) - 1)
    return ref_sorted[idx].astype(np.float32)


def load_derived(
    raw_dataset_dir: str,
    anomaly_dir: str,
    causal_path: str,
    cfg: dict,
    dataset_name: str = None,
    score_transform: str = "none",
    reference_scores: np.ndarray = None,
) -> list:
    """
    Assemble one episode from scores produced by the REAL outer models.

    Loader for derived scores: anomaly scores from LSTM-AD via TimeEval
    (`outer_models/timeeval_model/timeeval_pipeline.py`) and causal scores from
    PCMCI+ via tigramite (`outer_models/causal_discovery/pcmci_pipeline.py`),
    over raw telemetry from `data/synthetic_raw.py`.  It closes the loop the
    paper describes but the synthetic pipeline never actually ran: elsewhere the
    scores are *generated* from Beta distributions rather than *detected*.

    Inputs
    ------
    raw_dataset_dir : data/raw-synthetic-...   — read for meta.json only
                      (capacities, costs, tier labels, node names)
    anomaly_dir     : data/derived-<dataset>/<algorithm>/anomaly_scores
                      <node>_scores.npy [T, M] and
                      <node>_labels_per_metric.npy [T, M]
    causal_path     : data/causal-derived-.../causal_scores.npy  [N, M]

    Unlike `load_smd`, which makes each machine a separate N=1 episode, this
    returns ONE episode spanning all N nodes.  That is the structure the agent
    was trained on — per-node capacities only mean something when the nodes are
    axes of a single decision problem, not separate episodes.

    Capacities, costs and tier labels come from `meta.json`, i.e. from
    `generate_capacities()` at generation time.  They are NOT re-derived here:
    `load_npy` and `load_smd` both invent `cap = max(2, M//2)` because they have
    no better source, and that silently discards the tier structure.
    """
    from pathlib import Path

    raw_dir  = Path(raw_dataset_dir)
    score_dir = Path(anomaly_dir)
    ecfg     = cfg["environment"]

    with open(raw_dir / "meta.json") as f:
        meta = json.load(f)

    N     = int(meta["n_nodes"])
    M     = int(meta["n_metrics"])
    nodes = [n["name"] for n in meta["nodes"]]

    A_list, L_list = [], []
    for node in nodes:
        s_path = score_dir / f"{node}_scores.npy"
        l_path = score_dir / f"{node}_labels_per_metric.npy"
        if not s_path.exists():
            raise FileNotFoundError(
                f"{s_path} not found. Run the TimeEval pipeline on "
                f"{raw_dir} first."
            )
        if not l_path.exists():
            raise FileNotFoundError(
                f"{l_path} not found. It is written by step 1 of the TimeEval "
                f"pipeline; a stale workspace from before per-metric labels "
                f"existed will not have it."
            )
        A_list.append(np.load(s_path).astype(np.float32))     # [T, M]
        L_list.append(np.load(l_path).astype(np.int32))       # [T, M]

    A = np.stack(A_list, axis=1)     # [T, N, M]
    L = np.stack(L_list, axis=1)     # [T, N, M]
    T = A.shape[0]

    if A.shape != (T, N, M):
        raise ValueError(f"Assembled anomaly scores {A.shape} != ({T},{N},{M})")
    if L.shape != A.shape:
        raise ValueError(f"Label shape {L.shape} != score shape {A.shape}")

    # Optional score transform.  Applied BEFORE b_a is computed, so the
    # threshold still comes from the distribution the agent actually sees.
    if score_transform == "rank":
        A = _rank_transform(A)
    elif score_transform == "match":
        if reference_scores is None:
            raise ValueError("score_transform='match' needs reference_scores")
        A = _match_transform(A, reference_scores)
    elif score_transform != "none":
        raise ValueError(f"unknown score_transform: {score_transform}")

    C = np.load(causal_path).astype(np.float32)               # [N, M]
    if C.shape != (N, M):
        raise ValueError(
            f"Causal scores {C.shape} != ({N},{M}). The causal and anomaly "
            f"score directories must come from the SAME raw dataset."
        )

    caps       = np.array(meta["node_capacities"], dtype=np.float32)
    node_costs = np.array(meta["node_costs"],      dtype=np.float32)
    tiers      = meta["node_tier_labels"]

    # Bake the thresholds in at build time, from the DERIVED distributions.
    # Fair comparison: agent, baselines and metrics must
    # all score against identical thresholds, so they are computed once here and
    # never recomputed per consumer.
    b_a = float(np.percentile(A, ecfg["anomaly_baseline_pct"]))
    b_c = float(np.percentile(C, ecfg["causal_baseline_pct"]))

    name = dataset_name or raw_dir.name
    print(f"[metrics] Loaded derived: [{T},{N},{M}]  ← {raw_dir.name}")
    print(f"[metrics]   score_transform={score_transform}  "
          f"score mean={A.mean():.4f} median={np.median(A):.4f}")
    print(f"[metrics]   anomaly ← {score_dir}")
    print(f"[metrics]   causal  ← {causal_path}")
    print(f"[metrics]   graph_layout={meta.get('graph_layout','per_node')}  "
          f"cross_node_density={meta.get('cross_node_density', 0)}")
    print(f"[metrics]   caps={caps.astype(int).tolist()}  tiers={tiers}")
    print(f"[metrics]   b_a={b_a:.4f} (pct {ecfg['anomaly_baseline_pct']})  "
          f"b_c={b_c:.4f} (pct {ecfg['causal_baseline_pct']})")
    print(f"[metrics]   anomaly_rate={L.mean():.4f}  "
          f"causal cells above b_c: {int((C > b_c).sum())}/{C.size}")

    if int((C > b_c).sum()) == 0:
        print(f"[metrics]   WARNING: no causal cell exceeds b_c, so "
              f"max(0, C - b_c) = 0 everywhere and the causal term earns "
              f"exactly zero reward. Lower causal_baseline_pct "
              f"(<= 80 for per-node-normalised scores at M={M}).")

    return [{
        "anomaly_scores":   A,
        "anomaly_labels":   L,
        "causal_scores":    C,
        "node_capacities":  caps,
        "node_costs":       node_costs,
        "node_tier_labels": tiers,
        "n_nodes": N, "n_metrics": M, "T": T,
        "name": name, "b_a": b_a, "b_c": b_c,
    }]


# ------------------------------------------------------------------
# SMD interpretation label parser
# ------------------------------------------------------------------

def parse_interp_label(line: str, T: int, M: int) -> np.ndarray:
    """
    Parse one SMD interpretation label line into a [T, M] binary array.

    Format:  "start-end:m1,m2,...  start-end:m1,m2,..."
      - Events are space-separated
      - start and end are INCLUSIVE timestep indices
      - Metric indices are 1-based (converted to 0-based internally)

    Example:
      "15849-16368:1,9,10,12  16963-17517:1,2,3,4,6"
    """
    L = np.zeros((T, M), dtype=np.int32)
    for event in line.strip().split():
        if not event or ":" not in event:
            continue
        time_part, metric_part = event.split(":", 1)
        t_start, t_end = map(int, time_part.split("-"))
        metrics = [int(m) - 1 for m in metric_part.split(",") if m.strip()]
        t_end_clip = min(t_end + 1, T)   # inclusive end → exclusive slice
        for m in metrics:
            if 0 <= m < M:
                L[t_start:t_end_clip, m] = 1
    return L


# ------------------------------------------------------------------
# SMD loader — per-metric ground truth from interpretation labels
# ------------------------------------------------------------------

def load_smd(
    anomaly_score_dir: str,
    interp_label_dir: str,
    machines: list,
    cfg: dict,
    n_metrics: int = 38,
) -> list:
    """
    Load SMD dataset as evaluation episodes with per-metric ground truth.

    Each machine becomes one episode: N=1 node, M=38 metrics, T=varies.

    Parameters
    ----------
    anomaly_score_dir : directory containing <machine>.npy files
                        Each file shape: [T, M] — pre-computed anomaly scores
    interp_label_dir  : directory containing <machine>.txt interpretation labels
                        Format: "start-end:m1,m2,...  start-end:m1,m2,..."
                        Metrics 1-based, end index inclusive.
    machines          : list of machine names e.g. ["machine-1-1", "machine-1-2"]
                        If None, auto-discovers all .npy files in score_dir.
    cfg               : config dict
    n_metrics         : metrics per machine (default 38 for SMD)
    """
    from pathlib import Path

    score_dir = Path(anomaly_score_dir)
    label_dir = Path(interp_label_dir)
    ecfg      = cfg["environment"]
    M         = n_metrics

    if machines is None:
        machines = sorted([f.stem for f in score_dir.glob("*.npy")])
        print(f"[metrics] Auto-discovered {len(machines)} machines")

    episodes = []
    for machine in machines:

        # ── Anomaly scores ────────────────────────────────────────
        score_path = score_dir / f"{machine}.npy"
        if not score_path.exists():
            print(f"[metrics] {machine}: score file not found — skipping")
            continue

        A = np.load(score_path).astype(np.float32)   # [T, M]
        if A.ndim == 1:
            A = A[:, np.newaxis].repeat(M, axis=1)
        if A.shape[1] != M:
            print(f"[metrics] {machine}: score has {A.shape[1]} metrics "
                  f"≠ {M} — skipping")
            continue
        T = A.shape[0]

        # ── Interpretation labels → per-metric [T, M] ────────────
        label_path = label_dir / f"{machine}.txt"
        if label_path.exists():
            with open(label_path) as f:
                line = f.read().strip()
            L_2d = parse_interp_label(line, T, M)   # [T, M]
        else:
            print(f"[metrics] {machine}: no interpretation label — "
                  f"using zero labels")
            L_2d = np.zeros((T, M), dtype=np.int32)

        # Expand to [T, N=1, M]
        A_3d = A[:, np.newaxis, :]      # [T, 1, M]
        L_3d = L_2d[:, np.newaxis, :]   # [T, 1, M]

        # ── Causal scores — uniform (no PCMCI+ for SMD) ──────────
        C = np.full((1, M), 0.5, dtype=np.float32)

        # ── Capacity — 50% of M ───────────────────────────────────
        cap       = max(2, int(M * 0.5))
        caps      = np.array([float(cap)], dtype=np.float32)
        node_costs = np.array([ecfg["per_metric_cost"]], dtype=np.float32)

        b_a = float(np.percentile(A, ecfg["anomaly_baseline_pct"]))
        b_c = float(np.percentile(C, ecfg["causal_baseline_pct"]))

        print(f"[metrics] {machine}: T={T}  cap={cap}  "
              f"anomaly_rate={L_2d.mean():.4f}  "
              f"anomalous_metrics={int(L_2d.max(axis=0).sum())}/{M}  "
              f"b_a={b_a:.4f}")

        episodes.append({
            "anomaly_scores":  A_3d,
            "anomaly_labels":  L_3d,
            "causal_scores":   C,
            "node_capacities": caps,
            "node_costs":      node_costs,
            "n_nodes":  1, "n_metrics": M, "T": T,
            "name":     machine, "b_a": b_a, "b_c": b_c,
        })

    print(f"[metrics] Loaded {len(episodes)} SMD episodes")
    return episodes


# assign_tiers is imported from data.synthetic_data — it reads the authoritative
# node_tier_labels recorded at generation time, and only falls back to cost
# thresholds for legacy pools.  Do not reintroduce a local threshold version:
# it silently misattributes every per-tier metric once costs are varied.


# ------------------------------------------------------------------
# Policy factory
# ------------------------------------------------------------------

def build_policy(cfg: dict, obs_dim: int, action_dim: int) -> torch.nn.Module:
    """
    Instantiate the correct policy class based on config.

    Reads cfg["policy"]["policy_type"]:
      "flat"       (default) — ActorCritic flat MLP
      "per_metric"           — PerMetricActorCritic shared encoder

    Falls back to "flat" if policy_type is not set, so existing configs
    that do not specify policy_type continue to work unchanged.
    """
    pcfg        = cfg["policy"]
    policy_type = pcfg.get("policy_type", "flat")
    ecfg        = cfg["environment"]
    N, M        = ecfg["n_nodes"], ecfg["n_metrics"]
    feature_dim = obs_dim // (N * M)   # = 6 with current monitoring_env

    if policy_type == "per_metric":
        policy = PerMetricActorCritic(
            n_nodes        = N,
            n_metrics      = M,
            feature_dim    = feature_dim,
            metric_hidden  = pcfg.get("metric_hidden",  128),
            global_hidden  = pcfg.get("global_hidden",  256),
            use_layer_norm = pcfg.get("use_layer_norm", True),
            global_pool    = pcfg.get("global_pool",    "mean"),
        )
        print(f"[policy] PerMetricActorCritic  "
              f"mh={pcfg.get('metric_hidden',128)}  "
              f"gh={pcfg.get('global_hidden',256)}  "
              f"pool={pcfg.get('global_pool','mean')}")
    else:
        policy = ActorCritic(
            obs_dim        = obs_dim,
            action_dim     = action_dim,
            hidden_sizes   = pcfg.get("hidden_sizes", [512, 256]),
            use_layer_norm = pcfg.get("use_layer_norm", True),
        )
        print(f"[policy] ActorCritic  "
              f"hidden={pcfg.get('hidden_sizes',[512,256])}")

    n_params = sum(p.numel() for p in policy.parameters())
    print(f"[policy] Parameters: {n_params:,}")
    return policy


# ══════════════════════════════════════════════════════════════════
# Core metric computation — operates on one episode trajectory
# ══════════════════════════════════════════════════════════════════

def compute_episode_metrics(
    actions:       List[np.ndarray],   # list of [N,M] binary arrays, one per window
    episode:       dict,
    window_size:   int,
    coverage_mode: str = "window",     # "window" (correct) | "per_timestep" (legacy)
) -> dict:
    """
    Compute all metrics for one episode given the per-window action trajectory.
    actions[w] is the [N,M] binary selection for window w, covering
    timesteps [w*W, (w+1)*W).

    coverage_mode controls ACR, weighted-ACR, FMR, F1, and tier ACR:
      "window"       — window-max labels. One score per window. Use for paper.
      "per_timestep" — per-timestep average, inflated by anomaly-free steps.
                       Use only to compare with old training logs.
    """
    A    = episode["anomaly_scores"]    # [T, N, M]
    L    = episode["anomaly_labels"]    # [T, N, M]
    C    = episode["causal_scores"]     # [N, M]
    caps = episode["node_capacities"]  # [N]
    costs = episode.get(
        "node_costs", np.full(episode["n_nodes"], 0.01, dtype=np.float32)
    )
    N, M = episode["n_nodes"], episode["n_metrics"]
    T    = episode["T"]
    W    = window_size
    n_windows     = len(actions)
    tiers         = assign_tiers(episode)   # authoritative labels, not cost thresholds
    total_budget  = float(caps.sum())
    total_metrics = N * M
    b_a = episode.get("b_a", 0.0)

    # Selection frequency per metric across all windows
    sel_freq = np.stack(actions).mean(axis=0)   # [N, M]

    # ── Coverage helper — respects coverage_mode ────────────────

    def _window_cov(w, a):
        """Returns (acr_value, L_w) for one window."""
        t_s, t_e = w * W, min((w + 1) * W, T)
        if coverage_mode == "window":
            L_w  = L[t_s:t_e].max(axis=0)
            n_an = int(L_w.sum())
            return (float((a * L_w).sum() / n_an) if n_an > 0 else 1.0), L_w
        else:
            # per_timestep: average over each t in window
            covs = []
            for t in range(t_s, t_e):
                L_t  = L[t]; n_an = int(L_t.sum())
                covs.append(float((a * L_t).sum() / n_an) if n_an > 0 else 1.0)
            L_w = L[t_s:t_e].max(axis=0)   # still used for FMR / wACR
            return float(np.mean(covs)), L_w

    # ── Group 1: Coverage Quality ──────────────────────────────

    acr_vals, wacr_n, wacr_d, fmr_vals = [], 0.0, 0.0, []
    for w, a in enumerate(actions):
        t_s, t_e = w * W, min((w + 1) * W, T)
        cov, L_w = _window_cov(w, a)
        A_w      = A[t_s:t_e].max(axis=0)          # [N,M] window-max scores

        # ACR
        acr_vals.append(cov)

        # Weighted ACR — always window-max (per-timestep would double-count bursts)
        wacr_n += float((a * L_w * A_w).sum())
        wacr_d += float((L_w * A_w).sum())

        # FMR — always window-max (same reasoning)
        sel = float(a.sum())
        if sel > 0:
            fmr_vals.append(float((a * (1 - L_w)).sum()) / sel)

    acr         = float(np.mean(acr_vals))
    weighted_acr = wacr_n / wacr_d if wacr_d > 1e-8 else 1.0
    fmr         = float(np.mean(fmr_vals)) if fmr_vals else 0.0
    precision   = 1.0 - fmr
    f1          = 2 * acr * precision / (acr + precision + 1e-8)

    # ── Group 2: Detection Speed ────────────────────────────────

    latencies, ttfd_events = [], []

    for n in range(N):
        for m in range(M):
            lbl = L[:, n, m]
            if lbl.sum() == 0:
                continue
            in_event, event_start_t = False, 0
            for t in range(T):
                if lbl[t] == 1 and not in_event:
                    in_event, event_start_t = True, t
                elif lbl[t] == 0 and in_event:
                    in_event = False
                    esw = event_start_t // W
                    dw  = next((w for w in range(esw, n_windows)
                                if actions[w][n, m] == 1), None)
                    lat = (dw - esw) if dw is not None else n_windows
                    latencies.append(lat)
                    ttfd_events.append(lat <= 1)
            if in_event:
                esw = event_start_t // W
                dw  = next((w for w in range(esw, n_windows)
                            if actions[w][n, m] == 1), None)
                lat = (dw - esw) if dw is not None else n_windows
                latencies.append(lat)
                ttfd_events.append(lat <= 1)

    latency_mean   = float(np.mean(latencies))         if latencies else 0.0
    latency_median = float(np.median(latencies))       if latencies else 0.0
    latency_p90    = float(np.percentile(latencies, 90)) if latencies else 0.0
    ttfd           = float(np.mean(ttfd_events))       if ttfd_events else 1.0

    # ── Group 3: Selection, Cost & Resource Efficiency ──────────

    sel_counts = []   # absolute selected metrics per window
    cost_per_w = []   # tier-weighted cost per window  Σ_n cost_n × sel_n
    bur_vals   = []
    cvr_vals   = []

    for a in actions:
        sel_per_node = a.sum(axis=1)
        n_sel = int(a.sum())
        sel_counts.append(n_sel)
        cost_per_w.append(float((costs * sel_per_node).sum()))
        bur_vals.append(float(n_sel) / (total_budget + 1e-8))
        cvr_vals.append(float((sel_per_node > caps).mean()))

    mean_selected        = float(np.mean(sel_counts))
    selection_rate       = mean_selected / (total_metrics + 1e-8)
    mean_collection_cost = float(np.mean(cost_per_w))
    bur                  = float(np.mean(bur_vals))
    cvr                  = float(np.mean(cvr_vals))
    cuc                  = acr / (mean_collection_cost + 1e-8)

    # ── Group 4: Selection Quality ──────────────────────────────

    topk_prec_vals, score_gap_vals = [], []
    for w, a in enumerate(actions):
        t_s, t_e = w * W, min((w + 1) * W, T)
        A_w = A[t_s:t_e].max(axis=0)
        for n in range(N):
            k = int(a[n].sum())
            if k == 0:
                continue
            topk_oracle = set(np.argsort(A_w[n])[::-1][:k])
            selected    = set(np.where(a[n] == 1)[0])
            topk_prec_vals.append(len(topk_oracle & selected) / k)

        n_sel = a.sum(); n_unsel = N * M - n_sel
        if n_sel > 0 and n_unsel > 0:
            score_gap_vals.append(
                float(A_w[a == 1].mean()) - float(A_w[a == 0].mean())
            )

    topk_precision = float(np.mean(topk_prec_vals)) if topk_prec_vals else 0.0
    score_gap      = float(np.mean(score_gap_vals))  if score_gap_vals else 0.0

    try:
        from scipy.stats import spearmanr
        rhos = []
        for n in range(N):
            if sel_freq[n].std() > 1e-6 and C[n].std() > 1e-6:
                rho, _ = spearmanr(sel_freq[n], C[n])
                rhos.append(rho)
        causal_align = float(np.mean(rhos)) if rhos else float("nan")
    except ImportError:
        causal_align = float("nan")

    # ── Group 5: Stability ──────────────────────────────────────

    stab_vals, react_vals = [], []
    for w in range(1, n_windows):
        diff = np.abs(actions[w] - actions[w - 1]).sum()
        stab_vals.append(1.0 - diff / (N * M))
    stability = float(np.mean(stab_vals)) if stab_vals else 1.0

    for n in range(N):
        for m in range(M):
            for w in range(1, n_windows - 1):
                t_s = w * W
                t_e = min(t_s + W, T)
                A_w = A[t_s:t_e, n, m].max()
                if A_w > b_a:
                    react_vals.append(
                        float(actions[w][n, m]) - float(actions[w - 1][n, m])
                    )
    reactivity = float(np.mean(react_vals)) if react_vals else 0.0

    # ── Group 6: Tier-Specific ──────────────────────────────────

    tier_metrics = {}
    for tier_name in ("iot", "edge", "cloud"):
        node_idx = [i for i, t in enumerate(tiers) if t == tier_name]
        if not node_idx:
            for k in (f"acr_{tier_name}", f"cvr_{tier_name}", f"cuc_{tier_name}"):
                tier_metrics[k] = float("nan")
            continue

        t_acr_vals, t_cvr_vals, t_cost_vals = [], [], []
        for w, a in enumerate(actions):
            t_s, t_e = w * W, min((w + 1) * W, T)
            for n in node_idx:
                # Tier ACR respects coverage_mode
                if coverage_mode == "window":
                    L_wn = L[t_s:t_e, n, :].max(axis=0)
                    n_an = int(L_wn.sum())
                    t_acr_vals.append(
                        float((a[n] * L_wn).sum() / n_an) if n_an > 0 else 1.0
                    )
                else:
                    node_covs = []
                    for t in range(t_s, t_e):
                        L_tn = L[t, n, :]; n_an = int(L_tn.sum())
                        node_covs.append(
                            float((a[n] * L_tn).sum() / n_an) if n_an > 0 else 1.0
                        )
                    t_acr_vals.append(float(np.mean(node_covs)))
                t_cvr_vals.append(float(a[n].sum() > caps[n]))
                t_cost_vals.append(float(a[n].sum() * costs[n]))

        t_acr = float(np.mean(t_acr_vals))
        t_cvr = float(np.mean(t_cvr_vals))
        t_cost = float(np.mean(t_cost_vals))
        tier_metrics[f"acr_{tier_name}"] = t_acr
        tier_metrics[f"cvr_{tier_name}"] = t_cvr
        tier_metrics[f"cuc_{tier_name}"] = t_acr / (t_cost + 1e-8)

    return {
        # Coverage
        "acr": acr, "weighted_acr": weighted_acr, "fmr": fmr,
        "f1_coverage": f1,
        # Detection
        "latency_mean": latency_mean, "latency_median": latency_median,
        "latency_p90": latency_p90,   "ttfd": ttfd,
        # Selection & cost (new)
        "mean_selected":        mean_selected,
        "selection_rate":       selection_rate,
        "mean_collection_cost": mean_collection_cost,
        # Resource efficiency
        "bur": bur, "cuc": cuc, "cvr": cvr,
        "tier_cost": mean_collection_cost,   # alias for backward compat
        # Selection quality
        "topk_precision": topk_precision, "score_gap": score_gap,
        "causal_align": causal_align,
        # Stability
        "stability": stability, "reactivity": reactivity,
        # Meta
        "coverage_mode": coverage_mode,
        **tier_metrics,
    }


# ══════════════════════════════════════════════════════════════════
# Trajectory collectors
# ══════════════════════════════════════════════════════════════════

def collect_baseline_trajectory(
    baseline,
    episode: dict,
    window_size: int,
    window_agg: str = "max",
) -> List[np.ndarray]:
    """
    Run a baseline over one episode and return per-window action list.

    FIX: uses window-MAX aggregated scores for ranking, same as the RL
    environment observation. Previously used scores at t_mid only, which
    missed anomaly peaks and deflated coverage by ~0.20.

    The proxy episode maps window index w → aggregated score, so
    baseline.select(w) ranks using the same information the agent has.
    """
    N = episode["n_nodes"]
    M = episode["n_metrics"]
    T = episode["T"]
    A = episode["anomaly_scores"]    # [T, N, M]
    C = episode["causal_scores"]     # [N, M]
    W = window_size
    n_windows = T // W

    # Pre-aggregate scores [n_windows, N, M]
    agg_scores = np.zeros((n_windows, N, M), dtype=np.float32)
    for w in range(n_windows):
        t_s = w * W
        t_e = min(t_s + W, T)
        if window_agg == "max":
            agg_scores[w] = A[t_s:t_e].max(axis=0)
        else:
            agg_scores[w] = A[t_s:t_e].mean(axis=0)

    # Proxy episode so select(w) sees window-aggregated scores
    proxy_ep = {
        "n_nodes":        N,
        "n_metrics":      M,
        "T":              n_windows,
        "anomaly_scores": agg_scores,
        "anomaly_labels": np.zeros((n_windows, N, M), dtype=np.int32),
        "causal_scores":  C,
        "node_capacities":episode["node_capacities"],
        "b_a": episode.get("b_a", 0.0),
        "b_c": episode.get("b_c", 0.0),
    }
    baseline.reset(proxy_ep)

    return [baseline.select(w).copy() for w in range(n_windows)]


def collect_agent_trajectory(
    policy,
    episode: dict,
    reward_weights: dict,
    cfg: dict,
    device: torch.device,
    coverage_mode: str = "window",
) -> List[np.ndarray]:
    """Run the RL agent over one episode and return per-window action list."""
    ecfg = cfg["environment"]
    env  = MonitoringEnv(
        episode_pool         = [episode],
        reward_weights       = reward_weights,
        per_metric_cost      = ecfg["per_metric_cost"],
        anomaly_baseline_pct = ecfg["anomaly_baseline_pct"],
        causal_baseline_pct  = ecfg["causal_baseline_pct"],
        randomize_episodes   = False,
        window_size          = ecfg.get("window_size", 1),
        window_agg           = ecfg.get("window_agg", "max"),
        coverage_mode        = coverage_mode,
    )
    obs, _ = env.reset()
    done   = False
    actions = []
    N = episode["n_nodes"]
    M = episode["n_metrics"]

    policy.eval()
    with torch.no_grad():
        while not done:
            obs_t  = torch.tensor(obs, dtype=torch.float32,
                                  device=device).unsqueeze(0)
            action = policy.get_deterministic_action(obs_t).squeeze(0).cpu().numpy()
            obs, _, term, trunc, _ = env.step(action)
            done = term or trunc
            actions.append(action.reshape(N, M).copy())

    return actions


# ══════════════════════════════════════════════════════════════════
# Aggregation
# ══════════════════════════════════════════════════════════════════

def aggregate(ep_metrics: List[dict]) -> dict:
    agg  = {}
    skip = {"coverage_mode"}
    for k in ep_metrics[0].keys():
        if k in skip:
            continue
        vals = [m[k] for m in ep_metrics
                if not (isinstance(m[k], float) and np.isnan(m[k]))]
        if vals:
            agg[f"{k}_mean"] = float(np.mean(vals))
            agg[f"{k}_std"]  = float(np.std(vals))
        else:
            agg[f"{k}_mean"] = float("nan")
            agg[f"{k}_std"]  = float("nan")
    return agg


# ══════════════════════════════════════════════════════════════════
# Printing
# ══════════════════════════════════════════════════════════════════

PRIMARY_METRICS = [
    ("acr_mean",                  "ACR ↑",          ".4f"),
    ("weighted_acr_mean",         "wACR ↑",         ".4f"),
    ("fmr_mean",                  "FMR ↓",          ".4f"),
    ("f1_coverage_mean",          "F1-Cov ↑",       ".4f"),
    ("latency_mean_mean",         "Latency ↓",      ".2f"),
    ("ttfd_mean",                 "TTFD@1 ↑",       ".4f"),
    # Selection & cost (new)
    ("mean_selected_mean",        "Selected ↓",     ".1f"),
    ("selection_rate_mean",       "Sel.Rate ↓",     ".3f"),
    ("mean_collection_cost_mean", "CollCost ↓",     ".5f"),
    # Efficiency
    ("bur_mean",                  "BUR",            ".4f"),
    ("cuc_mean",                  "CUC ↑",          ".4f"),
    ("cvr_mean",                  "CVR ↓",          ".4f"),
]

SECONDARY_METRICS = [
    ("topk_precision_mean","TopK-P ↑",     ".4f"),
    ("score_gap_mean",     "ScoreGap ↑",   ".4f"),
    ("causal_align_mean",  "CausalAlign ↑",".4f"),
    ("stability_mean",     "Stability ↑",  ".4f"),
    ("reactivity_mean",    "Reactivity ↑", ".4f"),
]

TIER_METRICS = [
    ("acr_iot_mean",       "ACR-IoT ↑",   ".4f"),
    ("acr_edge_mean",      "ACR-Edge ↑",  ".4f"),
    ("acr_cloud_mean",     "ACR-Cloud ↑", ".4f"),
    ("cvr_iot_mean",       "CVR-IoT ↓",   ".4f"),
    ("cvr_edge_mean",      "CVR-Edge ↓",  ".4f"),
    ("cvr_cloud_mean",     "CVR-Cloud ↓", ".4f"),
    ("cuc_iot_mean",       "CUC-IoT ↑",   ".4f"),
    ("cuc_edge_mean",      "CUC-Edge ↑",  ".4f"),
    ("cuc_cloud_mean",     "CUC-Cloud ↑", ".4f"),
]


def _val(d, key, fmt):
    v = d.get(key, float("nan"))
    if isinstance(v, float) and np.isnan(v):
        return "  —   "
    return format(v, fmt)


def print_metric_table(title, results, metric_defs):
    methods = list(results.keys())
    col_w   = max(14, max(len(m) for m in methods) + 2)
    width   = 22 + col_w * len(methods)
    print(f"\n{'═'*width}")
    print(f"  {title}")
    print(f"{'═'*width}")
    header = f"{'Metric':<22}"
    for m in methods:
        header += f"{m:>{col_w}}"
    print(header)
    print("─" * width)
    for key, label, fmt in metric_defs:
        row = f"{label:<22}"
        for m in methods:
            row += f"{_val(results[m], key, fmt):>{col_w}}"
        print(row)
    print("═" * width)


# ══════════════════════════════════════════════════════════════════
# CSV saving
# ══════════════════════════════════════════════════════════════════

def save_aggregate_csv(results, path):
    all_keys = sorted({k for v in results.values() for k in v})
    rows = [{"method": name,
             **{k: v.get(k, float("nan")) for k in all_keys}}
            for name, v in results.items()]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method"] + all_keys)
        writer.writeheader(); writer.writerows(rows)
    print(f"[metrics] Aggregate → {path}")


def save_detail_csv(detail, path):
    all_keys = sorted({k for rows in detail.values() for r in rows for k in r})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "episode"] + all_keys)
        writer.writeheader()
        for method, rows in detail.items():
            for ep_i, row in enumerate(rows):
                writer.writerow({"method": method, "episode": ep_i,
                                 **{k: row.get(k, float("nan"))
                                    for k in all_keys}})
    print(f"[metrics] Detail    → {path}")


# ------------------------------------------------------------------
# Agent loader helper
# ------------------------------------------------------------------

def load_agent(model_path, cfg, eval_eps, reward_weights,
               W, agg, coverage_mode, device):
    """
    Build and load a policy from a checkpoint.
    Uses build_policy() which reads policy_type from cfg.
    Returns (policy, collect_fn) or (None, None) if path not found.
    """
    if not model_path or not os.path.exists(model_path):
        print(f"[metrics] Model not found: {model_path}")
        return None, None

    ecfg = cfg["environment"]
    _tmp = MonitoringEnv(
        episode_pool=eval_eps[:1], reward_weights=reward_weights,
        per_metric_cost=ecfg["per_metric_cost"],
        anomaly_baseline_pct=ecfg["anomaly_baseline_pct"],
        causal_baseline_pct=ecfg["causal_baseline_pct"],
        window_size=W, window_agg=agg, coverage_mode=coverage_mode,
    )
    obs_dim    = _tmp.observation_space.shape[0]
    action_dim = _tmp.action_space.n

    policy = build_policy(cfg, obs_dim, action_dim)
    policy.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=False)
    )
    policy.to(device)
    print(f"[metrics] Loaded: {model_path}")

    collect_fn = lambda ep, p=policy: collect_agent_trajectory(
        p, ep, reward_weights, cfg, device, coverage_mode
    )
    return policy, collect_fn


# ------------------------------------------------------------------
# Paper figures
# ------------------------------------------------------------------

def plot_paper_figures(aggregate_results: dict, out_dir: str,
                       coverage_mode: str):
    """
    Generate paper-ready figures from aggregate results.

    Figures produced
    ----------------
    fig1_primary_metrics.pdf      Bar chart: ACR, F1-Coverage, Latency per method
    fig2_efficiency.pdf           Bar chart: Selected, CollCost, CUC per method
    fig3_tier_acr.pdf             Grouped bar: ACR per tier (IoT/Edge/Cloud)
    fig4_stability_quality.pdf    Bar chart: Stability, TopK-P, CausalAlign
    fig5_radar.pdf                Radar chart: 6 key metrics, all methods
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        np_ = np
    except ImportError:
        print("[metrics] matplotlib not available — skipping figures")
        return

    os.makedirs(out_dir, exist_ok=True)
    methods = list(aggregate_results.keys())

    # Colour palette — distinct, print-friendly
    PALETTE = [
        "#2196F3",   # blue      — Monitor-Everything
        "#9E9E9E",   # grey      — Random-Feasible
        "#FF9800",   # orange    — TopK-Anomaly-Causal
        "#4CAF50",   # green     — RL-Flat
        "#E91E63",   # pink/red  — RL-PerMetric
        "#9C27B0",   # purple    — extra
        "#00BCD4",   # cyan      — extra
    ]
    colors = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(methods)}

    def _get(method, key):
        v = aggregate_results[method].get(key, float("nan"))
        return 0.0 if (isinstance(v, float) and np_.isnan(v)) else v

    def _bar_chart(ax, metric_keys, metric_labels, title, ylabel="",
                   lower_better=False):
        x     = np_.arange(len(metric_keys))
        width = 0.8 / len(methods)
        for i, m in enumerate(methods):
            vals = [_get(m, k) for k in metric_keys]
            bars = ax.bar(x + i * width, vals, width,
                          label=m, color=colors[m], alpha=0.87,
                          edgecolor="white", linewidth=0.5)
        ax.set_xticks(x + width * (len(methods) - 1) / 2)
        ax.set_xticklabels(metric_labels, fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if lower_better:
            ax.invert_yaxis()

    # ── Figure 1: Primary coverage & detection ────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    _bar_chart(axes[0],
               ["acr_mean", "weighted_acr_mean", "f1_coverage_mean"],
               ["ACR", "wACR", "F1-Cov"],
               "Coverage Quality", ylabel="Score [0-1]")
    _bar_chart(axes[1],
               ["fmr_mean"],
               ["FMR"],
               "False Monitoring Rate", ylabel="FMR [0-1] ↓", lower_better=False)
    _bar_chart(axes[2],
               ["latency_mean_mean", "latency_p90_mean"],
               ["Latency (mean)", "Latency (p90)"],
               "Detection Latency", ylabel="Windows ↓")
    handles = [mpatches.Patch(color=colors[m], label=m) for m in methods]
    fig.legend(handles=handles, loc="lower center", ncol=len(methods),
               fontsize=8, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Primary Metrics  [{coverage_mode} coverage]",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    p1 = os.path.join(out_dir, "fig1_primary_metrics.pdf")
    plt.savefig(p1, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[metrics] Figure → {p1}")

    # ── Figure 2: Efficiency ──────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    _bar_chart(axes[0],
               ["mean_selected_mean"],
               ["Mean Selected"],
               "Metrics Selected per Window", ylabel="Count ↓")
    _bar_chart(axes[1],
               ["selection_rate_mean"],
               ["Selection Rate"],
               "Selection Rate", ylabel="Fraction of N×M ↓")
    _bar_chart(axes[2],
               ["mean_collection_cost_mean", "cuc_mean"],
               ["CollCost", "CUC"],
               "Cost & Efficiency",
               ylabel="Value")
    fig.legend(handles=handles, loc="lower center", ncol=len(methods),
               fontsize=8, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Resource Efficiency", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    p2 = os.path.join(out_dir, "fig2_efficiency.pdf")
    plt.savefig(p2, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[metrics] Figure → {p2}")

    # ── Figure 3: Tier ACR ────────────────────────────────────────
    tier_keys   = ["acr_iot_mean", "acr_edge_mean", "acr_cloud_mean"]
    tier_labels = ["IoT", "Edge", "Cloud"]
    # Check at least one tier has data
    has_tier = any(
        not (isinstance(aggregate_results[m].get(k, float("nan")), float)
             and np_.isnan(aggregate_results[m].get(k, float("nan"))))
        for m in methods for k in tier_keys
    )
    if has_tier:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        _bar_chart(ax, tier_keys, tier_labels,
                   "Anomaly Coverage Rate by Node Tier", ylabel="ACR [0-1] ↑")
        ax.legend(handles=handles, fontsize=8, framealpha=0.9)
        fig.suptitle("Tier-Specific Coverage", fontsize=13,
                     fontweight="bold", y=1.01)
        plt.tight_layout()
        p3 = os.path.join(out_dir, "fig3_tier_acr.pdf")
        plt.savefig(p3, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[metrics] Figure → {p3}")

    # ── Figure 4: Stability & selection quality ───────────────────
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    _bar_chart(axes[0],
               ["stability_mean", "reactivity_mean"],
               ["Stability", "Reactivity"],
               "Temporal Behaviour", ylabel="Score [0-1] ↑")
    _bar_chart(axes[1],
               ["topk_precision_mean", "score_gap_mean"],
               ["TopK-P", "ScoreGap"],
               "Selection Quality", ylabel="Score ↑")
    _bar_chart(axes[2],
               ["causal_align_mean"],
               ["CausalAlign"],
               "Causal Alignment (ρ)", ylabel="Spearman ρ ↑")
    fig.legend(handles=handles, loc="lower center", ncol=len(methods),
               fontsize=8, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Stability & Selection Quality",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    p4 = os.path.join(out_dir, "fig4_stability_quality.pdf")
    plt.savefig(p4, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[metrics] Figure → {p4}")

    # ── Figure 5: Radar chart ─────────────────────────────────────
    radar_keys = [
        ("acr_mean",                  "ACR ↑"),
        ("f1_coverage_mean",          "F1-Cov ↑"),
        ("ttfd_mean",                 "TTFD ↑"),
        ("stability_mean",            "Stability ↑"),
        ("cuc_mean",                  "CUC ↑"),
        ("topk_precision_mean",       "TopK-P ↑"),
    ]
    rk    = [k for k, _ in radar_keys]
    rl    = [l for _, l in radar_keys]
    n_ax  = len(rk)
    angles = np_.linspace(0, 2 * np_.pi, n_ax, endpoint=False).tolist()
    angles += angles[:1]   # close the polygon

    # Normalise each metric to [0,1] across methods for radar
    def _norm(key):
        vals = [_get(m, key) for m in methods]
        vmin, vmax = min(vals), max(vals)
        if vmax - vmin < 1e-8:
            return [1.0] * len(methods)
        return [(v - vmin) / (vmax - vmin) for v in vals]

    norm_vals = {k: _norm(k) for k in rk}

    fig, ax = plt.subplots(figsize=(6, 6),
                           subplot_kw=dict(polar=True))
    for i, m in enumerate(methods):
        vals = [norm_vals[k][i] for k in rk]
        vals += vals[:1]
        ax.plot(angles, vals, "o-", linewidth=2,
                color=colors[m], label=m, alpha=0.85)
        ax.fill(angles, vals, alpha=0.08, color=colors[m])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(rl, fontsize=9)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"],
                       fontsize=7, color="grey")
    ax.set_ylim(0, 1)
    ax.spines["polar"].set_visible(False)
    ax.legend(handles=handles, loc="upper right",
              bbox_to_anchor=(1.35, 1.15), fontsize=8, framealpha=0.9)
    ax.set_title("Multi-Metric Comparison (normalised)",
                 fontsize=11, fontweight="bold", pad=20)
    plt.tight_layout()
    p5 = os.path.join(out_dir, "fig5_radar.pdf")
    plt.savefig(p5, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[metrics] Figure → {p5}")

    print(f"[metrics] All figures saved to {out_dir}/")


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

def main():
    args = parse_args()


    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(
        args.device if args.device else
        ("cuda" if torch.cuda.is_available() else "cpu")
    )

    ecfg = cfg["environment"]
    dcfg = cfg["data"]
    rw   = cfg["reward"]
    N, M = ecfg["n_nodes"], ecfg["n_metrics"]
    W    = ecfg.get("window_size", 1)
    agg  = ecfg.get("window_agg", "max")

    # Resolve coverage_mode: CLI > config > default "window"
    coverage_mode = args.coverage_mode or ecfg.get("coverage_mode", "window")

    reward_weights = {
        "alpha": rw["alpha"], "beta": rw["beta"],
        "lambda_": rw["lambda_"], "delta": rw["delta"], "eta": rw["eta"],
    }

    # ── Output directory ─────────────────────────────────────────
    base_results = cfg["paths"]["results_dir"]
    run_dir = (os.path.join(base_results, args.run_name)
               if args.run_name else base_results)
    out_dir = args.out_dir or os.path.join(run_dir, "metrics")
    os.makedirs(out_dir, exist_ok=True)

    # ── Load evaluation episodes ──────────────────────────────────
    if args.source == "npy":
        if not args.anomaly_scores or not args.causal_scores:
            print("[metrics] ERROR: --anomaly-scores and --causal-scores "
                  "are required with --source npy")
            return
        eval_eps = load_npy(
            args.anomaly_scores,
            args.causal_scores,
            args.labels,
            args.dataset_name,
            cfg,
        )
        pool_path = "npy"

    elif args.source == "derived":
        missing = [n for n, v in (("--raw-dataset-dir",     args.raw_dataset_dir),
                                  ("--derived-anomaly-dir", args.derived_anomaly_dir),
                                  ("--derived-causal",      args.derived_causal))
                   if not v]
        if missing:
            print(f"[metrics] ERROR: {', '.join(missing)} required with "
                  f"--source derived")
            return
        reference = None
        if args.derived_score_transform == "match":
            # Reference marginal = the training pool the agent actually saw.
            ref_eps = build_episode_pool(
                n_episodes=5, n_nodes=N, n_metrics=M,
                episode_length=ecfg["episode_length"],
                anomaly_density=dcfg["anomaly_density"],
                causal_density=dcfg["causal_density"],
                base_seed=dcfg["seed"],
                anomaly_baseline_pct=ecfg["anomaly_baseline_pct"],
                causal_baseline_pct=ecfg["causal_baseline_pct"],
                **topology_kwargs(ecfg),
            )
            reference = np.concatenate(
                [e["anomaly_scores"].ravel() for e in ref_eps])
            print(f"[metrics] match reference: {reference.size} training scores, "
                  f"mean={reference.mean():.4f}")

        eval_eps  = load_derived(
            raw_dataset_dir = args.raw_dataset_dir,
            anomaly_dir     = args.derived_anomaly_dir,
            causal_path     = args.derived_causal,
            cfg             = cfg,
            dataset_name    = args.dataset_name,
            score_transform = args.derived_score_transform,
            reference_scores= reference,
        )
        pool_path = args.raw_dataset_dir
        # The derived dataset defines its own N and M; the config's values
        # describe the training pool and would mislabel every printout.
        N = eval_eps[0]["n_nodes"]
        M = eval_eps[0]["n_metrics"]

    elif args.source == "smd":
        if not args.anomaly_score_dir or not args.interp_label_dir:
            print("[metrics] ERROR: --anomaly-score-dir and --interp-label-dir "
                  "are required with --source smd")
            return
        eval_eps = load_smd(
            anomaly_score_dir = args.anomaly_score_dir,
            interp_label_dir  = args.interp_label_dir,
            machines          = args.machines,
            cfg               = cfg,
            n_metrics         = args.n_metrics,
        )
        pool_path = args.anomaly_score_dir
        # SMD episodes may have N=1, M=38 — override N,M for print
        if eval_eps:
            N = eval_eps[0]["n_nodes"]
            M = eval_eps[0]["n_metrics"]

    else:
        # Default: pkl
        if args.pkl_path:
            pool_path = os.path.dirname(args.pkl_path) or "."
            eval_eps = load_pkl(args.pkl_path)
        else:
            pool_path  = pool_dir(cfg["paths"]["data_dir"], N, M, resolve_seeds(cfg)[0],
                          config_key=pool_config_key(cfg))
            eval_path = os.path.join(pool_path, "eval_episodes.pkl")
            if not os.path.exists(eval_path):
                print(f"[metrics] Eval pool not found — generating...")
                eval_eps = build_episode_pool(
                    n_episodes           = dcfg["n_eval_episodes"],
                    n_nodes=N, n_metrics=M,
                    episode_length       = ecfg["episode_length"],
                    anomaly_density      = dcfg["anomaly_density"] * 1.2,
                    causal_density       = dcfg["causal_density"],
                    base_seed            = dcfg["seed"] + 10000,
                    anomaly_baseline_pct = ecfg["anomaly_baseline_pct"],
                    causal_baseline_pct  = ecfg["causal_baseline_pct"],
                    **topology_kwargs(ecfg),
                )
                save_dataset(eval_eps, eval_path)
            else:
                eval_eps = load_dataset(eval_path)

    n_eval   = args.episodes or len(eval_eps)
    eval_eps = eval_eps[:n_eval]

    print(f"[metrics] Source    : {args.source}")
    print(f"[metrics] Data dir  : {pool_path}")
    print(f"[metrics] Episodes  : {n_eval}  |  N={N}  M={M}  W={W}  agg={agg}")
    print(f"[metrics] Output    : {out_dir}")
    print(f"[metrics] Coverage  : {coverage_mode}")
    print(f"[metrics] Note: all methods use window-max scores and window labels")

    # ── Define methods ────────────────────────────────────────────
    methods: Dict[str,callable]={}
    
    if not args.no_baselines:
        baselines = {
            "Monitor-Everything":  MonitorEverything(),
            "Random-Feasible":     RandomFeasible(seed=resolve_seeds(cfg)[1]),
            "TopK-Anomaly-Causal": TopKAnomalyCausal(),
        }
        methods = {
            name: (lambda ep, bl=bl:
                collect_baseline_trajectory(bl, ep, W, agg))
            for name, bl in baselines.items()
        }

    # ── Load RL agent(s) ──────────────────────────────────────────
    if not args.no_agent:

        # ── Case 1: two-model comparison ─────────────────────────
        if args.model_flat or args.model_per_metric:

            if args.model_flat:
                # Load config for flat model (may differ from main config)
                if args.config_flat:
                    with open(args.config_flat) as f:
                        cfg_flat = yaml.safe_load(f)
                else:
                    cfg_flat = copy.deepcopy(cfg)
                cfg_flat["policy"]["policy_type"] = "flat"

                _, fn_flat = load_agent(
                    args.model_flat, cfg_flat, eval_eps,
                    reward_weights, W, agg, coverage_mode, device
                )
                if fn_flat:
                    methods[args.label_flat] = fn_flat

            if args.model_per_metric:
                # Load config for per-metric model
                if args.config_per_metric:
                    with open(args.config_per_metric) as f:
                        cfg_pm = yaml.safe_load(f)
                else:
                    cfg_pm = copy.deepcopy(cfg)
                cfg_pm["policy"]["policy_type"] = "per_metric"

                _, fn_pm = load_agent(
                    args.model_per_metric, cfg_pm, eval_eps,
                    reward_weights, W, agg, coverage_mode, device
                )
                if fn_pm:
                    methods[args.label_per_metric] = fn_pm

        # ── Case 2: single model (original behaviour) ─────────────
        else:
            model_path = (
                args.model
                or (os.path.join(run_dir, cfg["paths"]["model_name"])
                    if args.run_name else None)
            )
            _, fn = load_agent(
                model_path, cfg, eval_eps,
                reward_weights, W, agg, coverage_mode, device
            )
            if fn:
                methods["RL-PPO (Ours)"] = fn
            elif not model_path:
                print("[metrics] No model specified — baselines only.")


    if not methods:
        print("[eval] Nothing to run."); return

    # ── Run all methods ───────────────────────────────────────────
    aggregate_results = {}
    detail_results    = {}

    for method_name, collect_fn in methods.items():
        print(f"\n[metrics] Running: {method_name} ...")
        ep_metrics = []
        for ep_i, ep in enumerate(eval_eps):
            actions = collect_fn(ep)
            m       = compute_episode_metrics(actions, ep, W, coverage_mode)
            ep_metrics.append(m)
            if (ep_i + 1) % 10 == 0 or ep_i == n_eval - 1:
                print(f"           ep {ep_i+1}/{n_eval}  "
                      f"ACR={m['acr']:.3f}  F1={m['f1_coverage']:.3f}  "
                      f"Sel={m['mean_selected']:.1f}({m['selection_rate']*100:.1f}%)  "
                      f"Cost={m['mean_collection_cost']:.4f}  "
                      f"Lat={m['latency_mean']:.1f}w  [{coverage_mode}]")
        aggregate_results[method_name] = aggregate(ep_metrics)
        detail_results[method_name]    = ep_metrics

    # ── Print ─────────────────────────────────────────────────────
    suffix = f"(coverage_mode={coverage_mode})"
    print_metric_table(f"Primary Metrics {suffix}",
                       aggregate_results, PRIMARY_METRICS)
    print_metric_table(f"Selection Quality & Stability {suffix}",
                       aggregate_results, SECONDARY_METRICS)
    print_metric_table(f"Tier-Specific Analysis {suffix}",
                       aggregate_results, TIER_METRICS)

    # ── Save ──────────────────────────────────────────────────────
    stem = f"metrics_{coverage_mode}"
    save_aggregate_csv(aggregate_results,
                       os.path.join(out_dir, f"{stem}_aggregate.csv"))
    if args.save_detail:
        save_detail_csv(detail_results,
                        os.path.join(out_dir, f"{stem}_detail.csv"))

    # Latency distribution for CDF plots
    lat_path = os.path.join(out_dir, f"{stem}_latency.csv")
    with open(lat_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "episode", "latency_mean",
                         "latency_median", "latency_p90", "ttfd"])
        for mname, ep_metrics in detail_results.items():
            for ep_i, m in enumerate(ep_metrics):
                writer.writerow([mname, ep_i,
                                 m["latency_mean"], m["latency_median"],
                                 m["latency_p90"],  m["ttfd"]])
    print(f"[metrics] Latency   → {lat_path}")

    # ── Paper figures ─────────────────────────────────────────────
    fig_dir = os.path.join(out_dir, "figures")
    plot_paper_figures(aggregate_results, fig_dir, coverage_mode)

    print(f"\n[metrics] Done. All results in {out_dir}/")


if __name__ == "__main__":
    main()