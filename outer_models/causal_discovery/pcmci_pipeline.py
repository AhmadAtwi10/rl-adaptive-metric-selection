"""
pcmci_pipeline.py
=================
Causal scores from PCMCI+ (tigramite).

Why this exists
---------------
The paper (SIII-B) states that the causal importance indicator comes from
PCMCI+.  Nothing in the repo ever ran it: `data/synthetic_data.py` draws causal
scores as independent Beta samples with no relationship to the anomaly dynamics
(`generate_causal_scores`, line 213), and the only mention of PCMCI+ was a
placeholder docstring.  This script closes that gap by running genuine PCMCI+
over the raw series produced by `data/synthetic_raw.py`.

What it computes
----------------
Per node, over that node's M metrics, on the TRAIN (anomaly-free) split —
matching the paper's "we compute these indicators from historical training
data":

    CausalScore(i) = sum_{j != i} sum_tau  1(i_tau --> j) * |val_matrix[i,j,tau]|

then normalised per node by its max, matching the `[N, M]` convention that
`generate_causal_scores()` establishes.

Two choices that change the numbers, made explicit
--------------------------------------------------
1. `tau_min=1` — contemporaneous (tau=0) links are excluded entirely.
   Two reasons.  Statistically, at tau=0 PCMCI+ also emits "o-o" (unoriented,
   inside a Markov equivalence class) and "x-x" (conflicting), neither of which
   is a directed causal claim, so only "-->" could be counted anyway.  For this
   dataset it is also simply correct: `data/synthetic_raw.py` forces
   `A[:, :, 0] = 0`, so there is no contemporaneous structure to find, and
   anything recovered at tau=0 would be a false positive by construction.

2. Only `"-->"` is counted as an edge.  It is the sole code that asserts a
   direction.

Self-loops (i == j) are excluded from the score.  A metric that predicts its own
future tells you nothing about which *other* metric to monitor, and the
ground-truth `out_strength` in `causal_graph.npz` excludes them too, so
including them would break the comparison.

Validation
----------
Because `data/synthetic_raw.py` embeds a known graph, recovery is measurable
rather than assumed.  Three checks, all reported:

  * edge-recovery precision / recall / F1 against `causal_graph.npz`
  * Spearman rho between the recovered CausalScore and the true summed
    outgoing strength — the score is what feeds the agent, so ranking it
    correctly matters more than getting every edge right
  * a time-shuffled control: each metric permuted independently, destroying all
    temporal structure.  F1 must collapse.  Without this, a high F1 could just
    mean the test is liberal.

`pc_alpha` acts as a regularisation parameter, not a significance level, so the
`--pc-alpha-sweep` flag reruns everything over {0.01, 0.05, 0.1} to show the
scores are not an artefact of one setting.

Usage
-----
  python outer_models/causal_discovery/pcmci_pipeline.py \\
      --dataset-dir data/raw-synthetic-5nodes-7metrics-d0.25-seed42 \\
      --out-dir     data/causal-derived-5nodes-7metrics-d0.25-seed42

  # regularisation sweep
  python outer_models/causal_discovery/pcmci_pipeline.py \\
      --dataset-dir data/raw-synthetic-5nodes-7metrics-d0.25-seed42 \\
      --out-dir     data/causal-derived-5nodes-7metrics-d0.25-seed42 \\
      --pc-alpha-sweep

Output
------
  <out-dir>/
    causal_scores.npy      [N, M]  float32, normalised per node to [0,1]
    causal_scores_raw.npy  [N, M]  float32, before normalisation
    graphs.npz             per-node graph codes and val_matrix
    validation.csv         per-node recovery metrics
    summary.json           aggregate metrics + the settings that produced them
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))


DEFAULT_TAU_MAX  = 3       # matches data/synthetic_raw.py DEFAULT_TAU_MAX
DEFAULT_PC_ALPHA = 0.05
SWEEP_ALPHAS     = (0.01, 0.05, 0.1)


# ══════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="PCMCI+ causal scores from raw synthetic telemetry",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset-dir", required=True,
                   help="Directory produced by data/synthetic_raw.py")
    p.add_argument("--out-dir", required=True,
                   help="Where to write scores and validation output")
    p.add_argument("--nodes", nargs="+", default=None,
                   help="Process only these nodes. Default: all found.")
    p.add_argument("--tau-max", type=int, default=DEFAULT_TAU_MAX,
                   help="Maximum lag searched. Should be >= the generator's "
                        "tau_max, or true edges become unrecoverable.")
    p.add_argument("--pc-alpha", type=float, default=DEFAULT_PC_ALPHA,
                   help="PCMCI+ regularisation parameter")
    p.add_argument("--pc-alpha-sweep", action="store_true",
                   help=f"Rerun over {SWEEP_ALPHAS} and report all")
    p.add_argument("--skip-shuffle-control", action="store_true",
                   help="Skip the time-shuffled null. Not recommended: it is "
                        "the check that recovery is real rather than an "
                        "artefact of a liberal test.")
    p.add_argument("--scope", choices=("auto", "node", "system"), default="auto",
                   help="'node': one PCMCI+ run per node (cross-node edges "
                        "unrepresentable). 'system': one run over all N*M "
                        "metrics (cross-node edges discoverable). 'auto' picks "
                        "from the dataset's graph layout, which is almost "
                        "always what you want — the other two exist to measure "
                        "the cost of getting it wrong.")
    p.add_argument("--seed", type=int, default=0,
                   help="Seed for the shuffled control")
    return p.parse_args()


def banner(title: str):
    print(f"\n{'═'*66}")
    print(f"  {title}")
    print(f"{'═'*66}")


# ══════════════════════════════════════════════════════════════════
# Data
# ══════════════════════════════════════════════════════════════════

def discover_nodes(dataset_dir: Path) -> list:
    return sorted(p.name.replace(".train.csv", "")
                  for p in dataset_dir.glob("*.train.csv"))


def load_ground_truth(dataset_dir: Path):
    """
    Load `causal_graph.npz` and normalise it to a single system-wide
    `[K, K, tau_max+1]` array, whatever layout it was written in.

    `data/synthetic_raw.py` emits two layouts:
      "per_node"  coefficients [N, M, M, tau+1] — N independent graphs, no
                  cross-node edges exist
      "system"    coefficients [K, K, tau+1] with K = N*M — one joint graph
    Older files predate the `layout` key; they are per-node by shape.
    """
    gt = np.load(dataset_dir / "causal_graph.npz")
    coef = gt["coefficients"]
    layout = str(gt["layout"]) if "layout" in gt.files else (
        "per_node" if coef.ndim == 4 else "system")

    if layout == "per_node":
        N, M = coef.shape[0], coef.shape[1]
        K, P = N * M, coef.shape[3]
        full = np.zeros((K, K, P), dtype=coef.dtype)
        for n in range(N):                    # embed each block on the diagonal
            s = slice(n * M, (n + 1) * M)
            full[s, s, :] = coef[n]
        coef = full
    else:
        N = int(gt["n_nodes"]); M = int(gt["n_metrics"])

    return coef, gt["out_strength"], layout, N, M


def load_system(dataset_dir: Path, nodes: list) -> np.ndarray:
    """All nodes' TRAIN splits concatenated column-wise into [T, K]."""
    return np.concatenate([load_train_split(dataset_dir, n) for n in nodes],
                          axis=1)


def load_train_split(dataset_dir: Path, node: str) -> np.ndarray:
    """
    The node's TRAIN series as [T, M].

    The train split is used because it is anomaly-free by construction.  Running
    discovery on the test split would let injected anomalies — which propagate
    along the very edges we are trying to recover — inflate the dependencies and
    make recovery look better than it is.
    """
    df = pd.read_csv(dataset_dir / f"{node}.train.csv")
    cols = [c for c in df.columns if c not in ("timestamp", "is_anomaly")]
    return df[cols].values.astype(np.float64)


# ══════════════════════════════════════════════════════════════════
# PCMCI+
# ══════════════════════════════════════════════════════════════════

def run_pcmci_node(values: np.ndarray, tau_max: int, pc_alpha: float) -> dict:
    """
    PCMCI+ on one node's [T, M] series.

    Returns the raw tigramite output: `graph` holds string codes and
    `val_matrix` holds partial correlations, both [M, M, tau_max+1] and both
    indexed [source, target, lag] — the same convention as the ground-truth
    `coefficients` array, which is what makes them directly comparable.
    """
    from tigramite import data_processing as pp
    from tigramite.pcmci import PCMCI
    from tigramite.independence_tests.parcorr import ParCorr

    pcmci = PCMCI(dataframe=pp.DataFrame(values),
                  cond_ind_test=ParCorr(significance="analytic"),
                  verbosity=0)
    res = pcmci.run_pcmciplus(tau_min=1, tau_max=tau_max, pc_alpha=pc_alpha)
    return {"graph": res["graph"], "val_matrix": res["val_matrix"]}


def causal_score(graph: np.ndarray, val_matrix: np.ndarray) -> np.ndarray:
    """
    CausalScore(i) = sum_{j != i} sum_tau 1(i_tau --> j) * |val[i,j,tau]|

    Self-loops excluded (see module docstring). tau=0 is never populated because
    the search starts at tau_min=1, but it is masked out explicitly so the
    function stays correct if that ever changes.
    """
    M = graph.shape[0]
    mask = (graph == "-->")
    mask[:, :, 0] = False                       # never count contemporaneous
    mask[np.arange(M), np.arange(M), :] = False  # never count self-loops
    return (np.abs(val_matrix) * mask).sum(axis=(1, 2))


def normalise_per_node(raw: np.ndarray) -> np.ndarray:
    """
    Scale each node's scores to [0, 1] by its max, matching the convention in
    generate_causal_scores().  A node with no recovered edges scores all-zero
    rather than becoming NaN.
    """
    out = np.zeros_like(raw)
    for n in range(raw.shape[0]):
        m = raw[n].max()
        if m > 1e-12:
            out[n] = raw[n] / m
    return out


# ══════════════════════════════════════════════════════════════════
# Validation against the embedded ground truth
# ══════════════════════════════════════════════════════════════════

def true_edge_set(A: np.ndarray, exclude_self: bool) -> set:
    """
    Ground-truth lagged edges from a coefficient array [D, D, tau_max+1].
    A[i, j, tau] != 0 means variable i at lag tau drives variable j.
    """
    edges = set()
    for tau in range(1, A.shape[2]):
        for i, j in zip(*np.nonzero(A[:, :, tau])):
            if exclude_self and i == j:
                continue
            edges.add((int(i), int(j), int(tau)))
    return edges


def recovered_edge_set(graph: np.ndarray, exclude_self: bool) -> set:
    edges = set()
    for i, j, tau in zip(*np.nonzero(graph == "-->")):
        if tau == 0 or (exclude_self and i == j):
            continue
        edges.add((int(i), int(j), int(tau)))
    return edges


def split_by_locality(edges: set, n_metrics: int) -> tuple:
    """Partition system-indexed edges into (within-node, cross-node)."""
    within = {e for e in edges if e[0] // n_metrics == e[1] // n_metrics}
    return within, edges - within


def edge_metrics(truth: set, recovered: set) -> dict:
    tp = len(truth & recovered)
    fp = len(recovered - truth)
    fn = len(truth - recovered)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": prec, "recall": rec, "f1": f1,
            "n_true": len(truth), "n_recovered": len(recovered)}


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rho, NaN when either side is constant (rho undefined)."""
    from scipy.stats import spearmanr
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(spearmanr(a, b).statistic)


def shuffle_series(values: np.ndarray, rng) -> np.ndarray:
    """
    Permute each metric independently.  This destroys autocorrelation and every
    cross-lag dependence while preserving each metric's marginal distribution,
    so any edge recovered from it is a false positive.
    """
    out = values.copy()
    for m in range(out.shape[1]):
        rng.shuffle(out[:, m])
    return out


# ══════════════════════════════════════════════════════════════════
# Driver
# ══════════════════════════════════════════════════════════════════

def run_one_alpha(dataset_dir: Path, nodes: list, truth: np.ndarray,
                  out_strength: np.ndarray, tau_max: int, pc_alpha: float,
                  shuffle_control: bool, seed: int, scope: str, n_metrics: int,
                  verbose: bool = True):
    """
    One full pass at a single pc_alpha.

    scope="node"    N separate PCMCI+ runs, one per node's M metrics.  Cannot
                    represent a cross-node edge at all.
    scope="system"  one run over all K = N*M metrics, so cross-node edges are
                    discoverable.  Correct when the data has them; harmful when
                    it does not, because the candidate space grows ~5.7x while
                    the true-edge count stays put.

    `truth` is always the system-wide [K, K, tau+1] array, so both scopes are
    scored against exactly the same ground truth and their numbers are
    comparable.
    """
    rng = np.random.default_rng(seed)
    rows, graphs = [], {}

    if scope == "system":
        values = load_system(dataset_dir, nodes)
        res    = run_pcmci_node(values, tau_max, pc_alpha)
        raw    = causal_score(res["graph"], res["val_matrix"])
        graphs["system_graph"] = res["graph"]
        graphs["system_val"]   = res["val_matrix"]
        recovered = recovered_edge_set(res["graph"], exclude_self=True)
    else:
        parts, blocks = [], []
        for node in nodes:
            v   = load_train_split(dataset_dir, node)
            r   = run_pcmci_node(v, tau_max, pc_alpha)
            parts.append(causal_score(r["graph"], r["val_matrix"]))
            graphs[f"{node}_graph"] = r["graph"]
            graphs[f"{node}_val"]   = r["val_matrix"]
            blocks.append(r["graph"])
        raw = np.concatenate(parts)
        # Lift each per-node graph into system indices so edge sets align.
        recovered = set()
        for n, g in enumerate(blocks):
            off = n * n_metrics
            for i, j, tau in recovered_edge_set(g, exclude_self=True):
                recovered.add((off + i, off + j, tau))

    truth_edges = true_edge_set(truth, exclude_self=True)
    m_all = edge_metrics(truth_edges, recovered)

    tw, tc = split_by_locality(truth_edges, n_metrics)
    rw, rc = split_by_locality(recovered,  n_metrics)
    m_within = edge_metrics(tw, rw)
    m_cross  = edge_metrics(tc, rc)

    rho = spearman(raw, out_strength.ravel())

    row = {"scope": scope, "pc_alpha": pc_alpha,
           "f1": m_all["f1"], "precision": m_all["precision"],
           "recall": m_all["recall"],
           "tp": m_all["tp"], "fp": m_all["fp"], "fn": m_all["fn"],
           "n_true": m_all["n_true"], "n_recovered": m_all["n_recovered"],
           "f1_within": m_within["f1"], "recall_within": m_within["recall"],
           "n_true_within": m_within["n_true"],
           "f1_crossnode": m_cross["f1"], "recall_crossnode": m_cross["recall"],
           "n_true_crossnode": m_cross["n_true"],
           "n_rec_crossnode": m_cross["n_recovered"],
           "spearman_rho": rho}

    if shuffle_control:
        if scope == "system":
            sh_rec = recovered_edge_set(
                run_pcmci_node(shuffle_series(load_system(dataset_dir, nodes), rng),
                               tau_max, pc_alpha)["graph"], exclude_self=True)
        else:
            sh_rec = set()
            for n, node in enumerate(nodes):
                g = run_pcmci_node(
                    shuffle_series(load_train_split(dataset_dir, node), rng),
                    tau_max, pc_alpha)["graph"]
                off = n * n_metrics
                for i, j, tau in recovered_edge_set(g, exclude_self=True):
                    sh_rec.add((off + i, off + j, tau))
        m_sh = edge_metrics(truth_edges, sh_rec)
        row["f1_shuffled"]    = m_sh["f1"]
        row["n_rec_shuffled"] = m_sh["n_recovered"]

    rows.append(row)
    if verbose:
        msg = (f"  F1={m_all['f1']:.3f} (P={m_all['precision']:.3f} "
               f"R={m_all['recall']:.3f}, {m_all['tp']}/{m_all['n_true']} "
               f"true edges)  rho={rho:+.3f}")
        print(msg)
        print(f"    within-node : F1={m_within['f1']:.3f} "
              f"R={m_within['recall']:.3f}  ({m_within['tp']}/{m_within['n_true']})")
        if m_cross["n_true"]:
            print(f"    CROSS-NODE  : F1={m_cross['f1']:.3f} "
                  f"R={m_cross['recall']:.3f}  ({m_cross['tp']}/{m_cross['n_true']})")
        elif m_cross["n_recovered"]:
            print(f"    CROSS-NODE  : {m_cross['n_recovered']} recovered, "
                  f"0 true → all false positives")
        if shuffle_control:
            print(f"    shuffled    : F1={row['f1_shuffled']:.3f} "
                  f"({row['n_rec_shuffled']} edges)")

    return raw.astype(np.float32), rows, graphs


def main():
    args        = parse_args()
    dataset_dir = Path(args.dataset_dir)
    out_dir     = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    nodes = args.nodes or discover_nodes(dataset_dir)
    if not nodes:
        print(f"[ERROR] No *.train.csv found in {dataset_dir}")
        sys.exit(1)

    gt_path = dataset_dir / "causal_graph.npz"
    if not gt_path.exists():
        print(f"[ERROR] {gt_path} not found — this dataset carries no ground "
              f"truth, so recovery cannot be validated.")
        sys.exit(1)
    truth, out_strength, layout, N, M = load_ground_truth(dataset_dir)

    # Scope must match what the data can contain.  Running per-node discovery on
    # cross-node data makes those edges unrecoverable by construction; running
    # system-wide on per-node data inflates the candidate space ~5.7x (at N=5,
    # M=7) without adding a single true edge, so precision falls and every
    # cross-node "discovery" is a false positive.
    scope = args.scope
    if scope == "auto":
        scope = "system" if layout == "system" else "node"

    banner("PCMCI+ causal discovery")
    print(f"  Dataset : {dataset_dir}")
    print(f"  Nodes   : {len(nodes)} → {nodes}")
    print(f"  Layout  : {layout}   (ground truth {truth.shape[0]} variables)")
    print(f"  Scope   : {scope}"
          + ("  (auto-selected from layout)" if args.scope == "auto" else ""))
    print(f"  tau_max : {args.tau_max}  (generator used {truth.shape[2] - 1})")
    print(f"  pc_alpha: {SWEEP_ALPHAS if args.pc_alpha_sweep else args.pc_alpha}")
    print(f"  Contemporaneous links: excluded (tau_min=1)")
    print(f"  Shuffled control: {'off' if args.skip_shuffle_control else 'on'}")

    if scope == "node" and layout == "system":
        print("\n  [WARN] Per-node scope on cross-node data: every cross-node "
              "edge is unrecoverable by construction, so recall will be "
              "capped well below 1.")

    if args.tau_max < truth.shape[2] - 1:
        print(f"\n  [WARN] tau_max={args.tau_max} is below the generator's "
              f"{truth.shape[2] - 1}; true edges at higher lags cannot be found.")

    t0       = time.time()
    alphas   = SWEEP_ALPHAS if args.pc_alpha_sweep else (args.pc_alpha,)
    all_rows = []
    primary  = None

    for alpha in alphas:
        print(f"\n  ── pc_alpha = {alpha} " + "─" * 40)
        raw, rows, graphs = run_one_alpha(
            dataset_dir, nodes, truth, out_strength, args.tau_max, alpha,
            shuffle_control=not args.skip_shuffle_control, seed=args.seed,
            scope=scope, n_metrics=M,
        )
        all_rows += rows
        if alpha == args.pc_alpha or primary is None:
            primary = (alpha, raw, graphs)

    alpha_used, raw_flat, graphs = primary
    raw_scores = raw_flat.reshape(N, M)
    scores = normalise_per_node(raw_scores)

    # ── Save ────────────────────────────────────────────────────────
    np.save(out_dir / "causal_scores.npy", scores)
    np.save(out_dir / "causal_scores_raw.npy", raw_scores)
    np.savez_compressed(out_dir / "graphs.npz", **graphs)
    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / "validation.csv", index=False)

    # ── Report ──────────────────────────────────────────────────────
    banner("Validation")
    for alpha in alphas:
        sub = df[df.pc_alpha == alpha]
        line = (f"  pc_alpha={alpha}:  F1={sub.f1.mean():.3f}  "
                f"P={sub.precision.mean():.3f}  "
                f"R={sub.recall.mean():.3f}  "
                f"rho={sub.spearman_rho.mean():+.3f}"
                f"  | within F1={sub.f1_within.mean():.3f}"
                f"  cross-node F1={sub.f1_crossnode.mean():.3f}")
        if "f1_shuffled" in sub:
            line += (f"  | shuffled F1={sub.f1_shuffled.mean():.3f}  "
                     f"edges={sub.n_rec_shuffled.mean():.1f}")
        print(line)

    sub = df[df.pc_alpha == alpha_used]
    print(f"\n  Causal scores (pc_alpha={alpha_used}, normalised per node):")
    for n, node in enumerate(nodes):
        print(f"    {node}: {np.round(scores[n], 3)}")

    # The causal_baseline_pct trap: b_c is a percentile of the causal array, and
    # every shipped config sets it to 100, making max(0, C - b_c) == 0 for every
    # cell.  Report what the data would actually support.
    flat = scores.ravel()
    print(f"\n  Score distribution: min={flat.min():.3f} "
          f"median={np.median(flat):.3f} max={flat.max():.3f}  "
          f"zeros={int((flat == 0).sum())}/{flat.size}")
    print("  Percentiles that would become b_c:")
    for pct in (50, 60, 70, 80, 90, 100):
        thr = np.percentile(flat, pct)
        above = int((flat > thr).sum())
        print(f"    {pct:3d}% → b_c={thr:.3f}  ({above}/{flat.size} cells "
              f"earn causal reward)")

    summary = {
        "dataset_dir":      str(dataset_dir),
        "nodes":            nodes,
        "tau_max":          args.tau_max,
        "pc_alpha_primary": alpha_used,
        "pc_alpha_sweep":   list(alphas),
        "tau_min":          1,
        "contemporaneous":  "excluded",
        "self_loops":       "excluded from score",
        "scope":            scope,
        "graph_layout":     layout,
        "mean_f1":          float(sub.f1.mean()),
        "mean_precision":   float(sub.precision.mean()),
        "mean_recall":      float(sub.recall.mean()),
        "mean_f1_within":   float(sub.f1_within.mean()),
        "mean_f1_crossnode":     float(sub.f1_crossnode.mean()),
        "mean_recall_crossnode": float(sub.recall_crossnode.mean()),
        "n_true_crossnode":      int(sub.n_true_crossnode.iloc[0]),
        "mean_spearman":    float(sub.spearman_rho.mean()),
        "runtime_seconds":  round(time.time() - t0, 1),
    }
    if "f1_shuffled" in sub:
        summary["mean_f1_shuffled"] = float(sub.f1_shuffled.mean())
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n  ✓ Scores      → {out_dir / 'causal_scores.npy'}  {scores.shape}")
    print(f"  ✓ Validation  → {out_dir / 'validation.csv'}")
    print(f"  ✓ Summary     → {out_dir / 'summary.json'}")
    print(f"  Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
