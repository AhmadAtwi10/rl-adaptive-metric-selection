"""
synthetic_raw.py
================
Raw-value synthetic telemetry with a known causal graph.

`synthetic_data.py` generates anomaly and causal *scores* directly (Beta-distributed
scores with injected anomaly bursts), which is fast and fully controllable.  This
module instead generates the raw metric values: multivariate telemetry per node,
driven by a sparse lagged causal graph, with anomalies injected into a held-out
test split.  From these series the full scoring pipeline can be run — LSTM-AD
anomaly scores (outer_models/timeeval_model) and PCMCI+ causal scores
(outer_models/causal_discovery) — and the recovered scores can be checked against
the known ground truth before evaluating a trained agent on them
(`scripts/compute_metrics.py --source derived`).

It reuses the capacity / cost / tier logic of `synthetic_data.py`, so the topology
semantics are identical.

Signal model (per node)
-----------------------
Each node carries M metrics generated as a stationary VAR process on the
*deviation* from a deterministic base signal:

    d(t) = Σ_τ B_τ · d(t−τ) + ε(t)          B_τ[j,i] = A[i,j,τ]
    x(t) = base(t) + d(t)

  * `A[i,j,τ]` is the causal coefficient from metric i at lag τ onto metric j.
    Only τ ≥ 1 is used — contemporaneous links are deliberately excluded, so
    PCMCI+ can be run with `tau_min=1` and never has to orient a τ=0 edge.
  * `base` gives each metric its own seasonal period, phase and slow trend, so
    the series look like telemetry and LSTM-AD has a learnable normal pattern.
    Periods differ per metric on purpose: a *shared* season would act as a
    confounder and manufacture causal links that are not in the graph.
  * `ε` is AR(1) so the noise is temporally correlated rather than white.

Coefficients are rescaled until the VAR companion matrix has spectral radius
below `STABILITY_TARGET`, which is what keeps the process stationary.

Ground-truth labels are COUNTERFACTUAL
--------------------------------------
The test split is simulated twice from identical innovations: once clean, once
with anomalies injected into ε.  A cell (t, m) is labelled anomalous when

    |x_injected(t,m) − x_clean(t,m)| > label_threshold · σ_m

This is the honest definition, and it is why anomalies are injected into ε
rather than added to x after the fact.  An anomaly on a metric with outgoing
causal edges genuinely propagates to its descendants through the VAR recursion,
and the counterfactual picks those descendants up automatically — no
reachability bookkeeping, no hand-declared label sets.  It is also what makes
the causal score operationally meaningful: a high out-degree metric really does
carry information about downstream failures.

Output layout
-------------
    <out_dir>/
      node_00.train.csv          timestamp, metric_0 … metric_{M-1}
      node_00.test.csv           timestamp, metric_0 … metric_{M-1}, is_anomaly
      node_00.interp_label.txt   SMD-style per-metric ground truth
      ...
      causal_graph.npz           coefficients + true out-strength
      meta.json                  seeds, parameters, realised topology

The CSV layout is the one `timeeval_pipeline.py` already auto-detects
(`_detect_format`, "Format A"), so STEPS 1–4 of that pipeline run unchanged.
`is_anomaly` is a single per-timestep column — that is what TimeEval consumes;
per-metric truth lives in the interpretation-label file, which round-trips
through the existing `parse_interp_label()` in `scripts/compute_metrics.py`.

Usage
-----
    python data/synthetic_raw.py --config configs/config.yaml
    python data/synthetic_raw.py --config configs/config.yaml \\
        --data-seed 43 --t-train 4000 --t-test 2000 --out-dir data/raw-eval
"""

import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.synthetic_data import (          # noqa: E402
    SyntheticMonitoringDataset,
    tier_fracs,
    topology_kwargs,
)


# ------------------------------------------------------------------
# Defaults — every one of these is exposed on the CLI
# ------------------------------------------------------------------

DEFAULT_T_TRAIN     = 4000     # anomaly-free; LSTM-AD is semi-supervised
DEFAULT_T_TEST      = 2000
DEFAULT_TAU_MAX     = 3        # max causal lag
DEFAULT_EDGE_DENSITY = 0.20    # P(edge) over ordered metric pairs
DEFAULT_COEF_RANGE  = (0.30, 0.80)
DEFAULT_AR_COEF     = 0.35     # AR(1) on the innovations
DEFAULT_NOISE_STD   = 1.0
DEFAULT_BURN_IN     = 500      # discarded, kills the VAR transient

STABILITY_TARGET    = 0.85     # spectral radius ceiling for the companion matrix

# Anomaly injection
DEFAULT_LABEL_THRESHOLD  = 1.0    # in units of σ of the clean series
DENSITY_DAMPING          = 0.6    # <1 → approach the target density from below
SPIKE_MAGNITUDE          = 6.0    # in units of σ of the innovations
LEVEL_SHIFT_MAGNITUDE    = 2.5
VARIANCE_BURST_FACTOR    = 5.0
SPIKE_DURATION           = (1, 4)      # [lo, hi)
LEVEL_SHIFT_DURATION     = (20, 60)
VARIANCE_BURST_DURATION  = (15, 45)

ANOMALY_TYPES = ("spike", "level_shift", "variance_burst", "propagated")

# ── Cross-node coupling ───────────────────────────────────────────
# Probability of a cross-node edge is `cross_node_density * TIER_COUPLING[
# (tier_of_source, tier_of_target)]`, so the continuum's topology enters the
# causal structure instead of only setting capacities and costs.
#
# Direction of the bias is physical, not arbitrary: telemetry and load
# propagate upward (IoT -> Edge -> Cloud), orchestration and configuration
# propagate back down more weakly, and same-tier nodes couple weakest of all
# (shared network segment, shared load balancer).
#
# The spread is deliberately MODERATE — 5x between the strongest and weakest
# pair, not 0-vs-1.  If tier determined influence outright, tier and causal
# score would be collinear and no experiment could separate "the agent used the
# causal score" from "the agent learned a tier prior from capacity and cost".
TIER_COUPLING = {
    ("iot",   "edge"):  1.0,     # sensor behaviour drives edge aggregation
    ("edge",  "cloud"): 1.0,     # edge load drives cloud processing
    ("iot",   "cloud"): 0.5,     # skip-level, mediated, rarer
    ("cloud", "edge"):  0.4,     # config / orchestration pushed down
    ("edge",  "iot"):   0.4,
    ("cloud", "iot"):   0.2,     # weakest: two hops of mediation
    ("iot",   "iot"):   0.3,
    ("edge",  "edge"):  0.3,
    ("cloud", "cloud"): 0.3,
}

DEFAULT_CROSS_NODE_DENSITY    = 0.04   # 0.0 → within-node only (legacy path)
DEFAULT_CROSS_NODE_COEF_SCALE = 0.6    # cross-node coupling is mediated,
                                       # so weaker than within-node


# ══════════════════════════════════════════════════════════════════
# Causal graph
# ══════════════════════════════════════════════════════════════════

def sample_causal_graph(
    n_metrics: int,
    tau_max: int,
    edge_density: float,
    coef_range: Tuple[float, float],
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample a sparse lagged causal graph over one node's metrics.

    Returns
    -------
    A : np.ndarray [M, M, tau_max + 1]  float64
        `A[i, j, tau]` is the coefficient from metric i at lag tau onto
        metric j.  `A[..., 0]` is always zero: contemporaneous links are
        excluded by construction (see module docstring).

    Self-loops (i == j) are allowed at lag 1 only — a metric depending on its
    own recent past is ordinary telemetry behaviour, not a causal claim about
    another metric, and it is excluded from the causal score anyway (the paper's
    formula sums over j ≠ i).
    """
    M = n_metrics
    A = np.zeros((M, M, tau_max + 1), dtype=np.float64)
    lo, hi = coef_range

    for i in range(M):
        for j in range(M):
            if i == j:
                continue
            if rng.random() >= edge_density:
                continue
            tau  = int(rng.integers(1, tau_max + 1))
            sign = 1.0 if rng.random() < 0.5 else -1.0
            A[i, j, tau] = sign * rng.uniform(lo, hi)

    # Mild self-persistence at lag 1, so each metric has its own dynamics
    for i in range(M):
        A[i, i, 1] = rng.uniform(0.15, 0.45)

    return _rescale_for_stability(A)


def sample_system_graph(
    n_nodes: int,
    n_metrics: int,
    tier_labels: List[str],
    tau_max: int,
    edge_density: float,
    cross_node_density: float,
    coef_range: Tuple[float, float],
    cross_coef_scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample one lagged causal graph over ALL N*M metrics in the system.

    Returns
    -------
    A : np.ndarray [K, K, tau_max + 1] where K = n_nodes * n_metrics
        `A[k, l, tau]` is the coefficient from system variable k at lag tau
        onto variable l.  Variable `k = n * n_metrics + m` is metric m of node
        n, so the within-node blocks are the diagonal [M, M] sub-blocks.

    Within-node edges are drawn exactly as `sample_causal_graph` draws them.
    Cross-node edges are drawn at `cross_node_density * TIER_COUPLING[...]` and
    scaled by `cross_coef_scale`, because coupling between nodes is mediated by
    the network and by aggregation and is therefore weaker than coupling inside
    a single host.

    With `cross_node_density = 0` this is block-diagonal and mathematically
    equivalent to N independent per-node graphs — but note it is NOT numerically
    identical to the per-node path, because `_rescale_for_stability` then scales
    every block by one shared factor rather than each block by its own.  That is
    why `build_dataset` keeps the per-node path for the zero case.
    """
    K = n_nodes * n_metrics
    A = np.zeros((K, K, tau_max + 1), dtype=np.float64)
    lo, hi = coef_range

    # ── Within-node edges ─────────────────────────────────────────
    for n in range(n_nodes):
        off = n * n_metrics
        for i in range(n_metrics):
            for j in range(n_metrics):
                if i == j:
                    continue
                if rng.random() >= edge_density:
                    continue
                tau  = int(rng.integers(1, tau_max + 1))
                sign = 1.0 if rng.random() < 0.5 else -1.0
                A[off + i, off + j, tau] = sign * rng.uniform(lo, hi)

    # ── Cross-node edges, biased by the tier pair ─────────────────
    for a in range(n_nodes):
        for b in range(n_nodes):
            if a == b:
                continue
            weight = TIER_COUPLING[(tier_labels[a], tier_labels[b])]
            p = cross_node_density * weight
            if p <= 0.0:
                continue
            for i in range(n_metrics):
                for j in range(n_metrics):
                    if rng.random() >= p:
                        continue
                    tau  = int(rng.integers(1, tau_max + 1))
                    sign = 1.0 if rng.random() < 0.5 else -1.0
                    A[a * n_metrics + i, b * n_metrics + j, tau] = (
                        sign * rng.uniform(lo, hi) * cross_coef_scale
                    )

    # Mild self-persistence at lag 1, so each metric has its own dynamics
    for k in range(K):
        A[k, k, 1] = rng.uniform(0.15, 0.45)

    return _rescale_for_stability(A)


def _companion_spectral_radius(A: np.ndarray) -> float:
    """
    Largest eigenvalue modulus of the VAR companion matrix.  Below 1.0 means
    the process is stationary; we aim well below to keep the series bounded
    over long episodes.
    """
    M = A.shape[0]
    p = A.shape[2] - 1
    C = np.zeros((M * p, M * p), dtype=np.float64)
    for tau in range(1, p + 1):
        # B_tau = A[:, :, tau].T  so that d(t) = Σ B_tau · d(t−tau)
        C[:M, (tau - 1) * M: tau * M] = A[:, :, tau].T
    if p > 1:
        C[M:, :-M] = np.eye(M * (p - 1))
    return float(np.max(np.abs(np.linalg.eigvals(C))))


def _rescale_for_stability(A: np.ndarray) -> np.ndarray:
    """
    Shrink coefficients until the companion spectral radius clears
    STABILITY_TARGET.

    Scaling is applied iteratively rather than in one analytic step: the
    eigenvalues of a VAR(p) companion matrix are not a linear function of the
    coefficient scale, so a single division by the radius does not land where
    you expect.
    """
    for _ in range(200):
        radius = _companion_spectral_radius(A)
        if radius < STABILITY_TARGET:
            return A
        A = A * 0.9
    raise RuntimeError(
        "Could not stabilise the sampled VAR — reduce edge_density or "
        "coef_range and retry."
    )


# ══════════════════════════════════════════════════════════════════
# Base signal and innovations
# ══════════════════════════════════════════════════════════════════

def build_base_signal(
    T: int,
    n_metrics: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Deterministic per-metric base: seasonality + slow linear trend.

    Each metric gets its OWN period and phase.  A shared seasonal component
    across metrics would be a common cause and would show up in PCMCI+ as
    dense spurious structure, so distinct periods are a correctness
    requirement, not decoration.
    """
    t = np.arange(T, dtype=np.float64)
    base = np.zeros((T, n_metrics), dtype=np.float64)

    for m in range(n_metrics):
        period = rng.uniform(60.0, 400.0)
        phase  = rng.uniform(0.0, 2.0 * np.pi)
        amp    = rng.uniform(0.5, 2.5)
        trend  = rng.uniform(-1.0, 1.0)
        level  = rng.uniform(-2.0, 2.0)
        base[:, m] = (
            level
            + amp * np.sin(2.0 * np.pi * t / period + phase)
            + trend * (t / max(T - 1, 1))
        )
    return base


def build_innovations(
    T: int,
    n_metrics: int,
    ar_coef: float,
    noise_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """AR(1) innovations — temporally correlated noise, not white."""
    eps = np.zeros((T, n_metrics), dtype=np.float64)
    white = rng.normal(0.0, noise_std, size=(T, n_metrics))
    eps[0] = white[0]
    for t in range(1, T):
        eps[t] = ar_coef * eps[t - 1] + white[t]
    return eps


def simulate_var(A: np.ndarray, base: np.ndarray, eps: np.ndarray) -> np.ndarray:
    """
    Run the VAR forward: `x(t) = base(t) + Σ_τ B_τ · d(t−τ) + ε(t)`.

    The VAR acts on the deviation `d`, not on `x`, so the deterministic base
    (which carries a trend) is never fed back through the recursion and cannot
    be amplified into a divergence.
    """
    T, M = base.shape
    p = A.shape[2] - 1
    B = [None] + [A[:, :, tau].T.copy() for tau in range(1, p + 1)]

    d = np.zeros((T, M), dtype=np.float64)
    for t in range(T):
        acc = eps[t].copy()
        for tau in range(1, p + 1):
            if t - tau >= 0:
                acc += B[tau] @ d[t - tau]
        d[t] = acc
    return base + d


# ══════════════════════════════════════════════════════════════════
# Anomaly injection
# ══════════════════════════════════════════════════════════════════

def _out_degree_strength(A: np.ndarray) -> np.ndarray:
    """True outgoing causal strength per metric — Σ_{j≠i} Σ_τ |A[i,j,τ]|.

    This is the ground truth that the PCMCI+ causal score is checked
    against, and it is also used to pick sources for `propagated` anomalies.
    """
    M = A.shape[0]
    mask = ~np.eye(M, dtype=bool)
    return (np.abs(A).sum(axis=2) * mask).sum(axis=1)


def inject_anomalies(
    eps: np.ndarray,
    A: np.ndarray,
    n_events: int,
    rng: np.random.Generator,
    noise_std: float,
) -> Tuple[np.ndarray, List[dict]]:
    """
    Perturb the innovations to create anomalies.

    Injection targets ε rather than x so that the perturbation propagates
    through the VAR recursion to causal descendants.  Returns the modified
    innovations and a record of what was injected (for meta.json).
    """
    eps = eps.copy()
    T, M = eps.shape
    strength = _out_degree_strength(A)
    events: List[dict] = []

    for _ in range(n_events):
        kind = ANOMALY_TYPES[int(rng.integers(0, len(ANOMALY_TYPES)))]

        if kind == "propagated" and strength.sum() > 0:
            # Bias towards hubs so propagation is actually exercised
            m = int(rng.choice(M, p=strength / strength.sum()))
        else:
            m = int(rng.integers(0, M))

        if kind == "spike":
            dur = int(rng.integers(*SPIKE_DURATION))
            mag = SPIKE_MAGNITUDE
        elif kind == "variance_burst":
            dur = int(rng.integers(*VARIANCE_BURST_DURATION))
            mag = 0.0
        else:                                   # level_shift, propagated
            dur = int(rng.integers(*LEVEL_SHIFT_DURATION))
            mag = LEVEL_SHIFT_MAGNITUDE

        t0 = int(rng.integers(0, max(1, T - dur)))
        t1 = min(t0 + dur, T)
        sign = 1.0 if rng.random() < 0.5 else -1.0

        if kind == "variance_burst":
            eps[t0:t1, m] *= VARIANCE_BURST_FACTOR
        else:
            eps[t0:t1, m] += sign * mag * noise_std

        events.append({"type": kind, "metric": m,
                       "t_start": t0, "t_end": t1 - 1})

    return eps, events


def counterfactual_labels(
    x_clean: np.ndarray,
    x_anom: np.ndarray,
    label_threshold: float,
) -> np.ndarray:
    """
    Label a cell anomalous when the injection actually moved it, by more than
    `label_threshold` standard deviations of that metric's clean series.

    σ is taken from the CLEAN series so the threshold is not inflated by the
    anomalies it is meant to detect.
    """
    sigma = x_clean.std(axis=0)
    sigma = np.where(sigma > 1e-8, sigma, 1.0)
    return (np.abs(x_anom - x_clean) > label_threshold * sigma).astype(np.int32)


# ══════════════════════════════════════════════════════════════════
# Per-node generation
# ══════════════════════════════════════════════════════════════════

def generate_node(
    n_metrics: int,
    t_train: int,
    t_test: int,
    anomaly_density: float,
    rng: np.random.Generator,
    tau_max: int = DEFAULT_TAU_MAX,
    edge_density: float = DEFAULT_EDGE_DENSITY,
    coef_range: Tuple[float, float] = DEFAULT_COEF_RANGE,
    ar_coef: float = DEFAULT_AR_COEF,
    noise_std: float = DEFAULT_NOISE_STD,
    burn_in: int = DEFAULT_BURN_IN,
    label_threshold: float = DEFAULT_LABEL_THRESHOLD,
    max_rounds: int = 40,
) -> dict:
    """
    Generate one node's train/test series, labels and causal graph.

    Train and test come from the SAME process — one continuous simulation,
    split in two — so the test split is not a distribution shift.  Only the
    test half receives injections, which is what LSTM-AD's semi-supervised
    setup requires.
    """
    M = n_metrics
    A = sample_causal_graph(M, tau_max, edge_density, coef_range, rng)

    T_total = burn_in + t_train + t_test
    base = build_base_signal(T_total, M, rng)
    eps  = build_innovations(T_total, M, ar_coef, noise_std, rng)

    x_clean_full = simulate_var(A, base, eps)

    train = x_clean_full[burn_in: burn_in + t_train]
    test_slice = slice(burn_in + t_train, T_total)

    # Density control: inject, measure the realised label density, top up.
    # The footprint of an event cannot be predicted in advance — it depends on
    # how far the perturbation propagates through the sampled graph — so the
    # batch size for the next round is estimated from the density actually
    # achieved so far.  Events overlap, so density grows sublinearly in the
    # event count and the estimate deliberately re-derives each round rather
    # than extrapolating once.
    x_clean_test = x_clean_full[test_slice]
    events: List[dict] = []
    eps_anom = eps
    total_events = 0
    n_events = max(1, int(round(anomaly_density * 40)))

    for _ in range(max_rounds):
        eps_anom, ev = inject_anomalies(eps_anom, A, n_events, rng, noise_std)
        events.extend(ev)
        total_events += n_events

        test = simulate_var(A, base, eps_anom)[test_slice]
        labels = counterfactual_labels(x_clean_test, test, label_threshold)
        density = float(labels.mean())
        if density >= anomaly_density:
            break

        per_event = density / max(total_events, 1)
        if per_event <= 1e-9:
            n_events = max(1, total_events)          # nothing landed — double
        else:
            # DENSITY_DAMPING < 1 makes the loop approach the target from
            # below over several rounds instead of overshooting on one jump.
            n_events = int(np.ceil(
                DENSITY_DAMPING * (anomaly_density - density) / per_event
            ))
        # Cap growth so a pathological graph cannot explode the event count
        # Cap growth at a doubling per round, with enough rounds to converge.
        #
        # The previous 10x cap let one round jump far past the target at low
        # densities: at anomaly_density=0.05 a node overshot to 0.117 (2.3x),
        # and the per-node spread reached 2.4x, which would confound any
        # per-tier analysis. A 2x cap approaches the target finely instead.
        #
        # This DOES change the generated data (the cap binds at 0.25 too), so it
        # was applied as a deliberate all-datasets regeneration on 2026-08-10 —
        # every raw dataset, and every LSTM-AD / PCMCI+ score derived from one,
        # was rebuilt together. Do not change it again without doing the same.
        n_events = max(1, min(n_events, total_events))

    return {
        "train":        train.astype(np.float32),
        "test":         test.astype(np.float32),
        "labels":       labels,                       # [t_test, M]
        "causal_graph": A,                            # [M, M, tau_max+1]
        "out_strength": _out_degree_strength(A),      # [M]
        "events":       events,
        "realised_density": float(labels.mean()),
    }


def generate_system(
    n_nodes: int,
    n_metrics: int,
    tier_labels: List[str],
    t_train: int,
    t_test: int,
    anomaly_density: float,
    rng: np.random.Generator,
    tau_max: int = DEFAULT_TAU_MAX,
    edge_density: float = DEFAULT_EDGE_DENSITY,
    cross_node_density: float = DEFAULT_CROSS_NODE_DENSITY,
    coef_range: Tuple[float, float] = DEFAULT_COEF_RANGE,
    cross_coef_scale: float = DEFAULT_CROSS_NODE_COEF_SCALE,
    ar_coef: float = DEFAULT_AR_COEF,
    noise_std: float = DEFAULT_NOISE_STD,
    burn_in: int = DEFAULT_BURN_IN,
    label_threshold: float = DEFAULT_LABEL_THRESHOLD,
    max_rounds: int = 40,
) -> dict:
    """
    Generate the whole system jointly: one VAR over all K = N*M metrics.

    This is the cross-node path.  It is a single simulation rather than N
    independent ones, which is what allows an anomaly injected on one node to
    propagate to another through the graph.  Counterfactual labelling then picks
    that propagation up for free — a metric on node B is labelled anomalous iff
    the injection on node A measurably moved it — with no reachability
    bookkeeping anywhere.

    Density control targets the GLOBAL label density over all [t_test, K] cells,
    not per node.  Per-node density is a consequence of where injections land
    and how far they travel, and is reported rather than forced; forcing it
    would fight the propagation this path exists to create.
    """
    K = n_nodes * n_metrics
    A = sample_system_graph(
        n_nodes, n_metrics, tier_labels, tau_max, edge_density,
        cross_node_density, coef_range, cross_coef_scale, rng,
    )

    T_total = burn_in + t_train + t_test
    base = build_base_signal(T_total, K, rng)
    eps  = build_innovations(T_total, K, ar_coef, noise_std, rng)

    x_clean_full = simulate_var(A, base, eps)

    train      = x_clean_full[burn_in: burn_in + t_train]
    test_slice = slice(burn_in + t_train, T_total)
    x_clean_test = x_clean_full[test_slice]

    events: List[dict] = []
    eps_anom = eps
    total_events = 0
    n_events = max(1, int(round(anomaly_density * 40 * n_nodes)))

    for _ in range(max_rounds):
        eps_anom, ev = inject_anomalies(eps_anom, A, n_events, rng, noise_std)
        events.extend(ev)
        total_events += n_events

        test   = simulate_var(A, base, eps_anom)[test_slice]
        labels = counterfactual_labels(x_clean_test, test, label_threshold)
        density = float(labels.mean())
        if density >= anomaly_density:
            break

        per_event = density / max(total_events, 1)
        if per_event <= 1e-9:
            n_events = max(1, total_events)
        else:
            n_events = int(np.ceil(
                DENSITY_DAMPING * (anomaly_density - density) / per_event
            ))
        # Cap growth at a doubling per round, with enough rounds to converge.
        #
        # The previous 10x cap let one round jump far past the target at low
        # densities: at anomaly_density=0.05 a node overshot to 0.117 (2.3x),
        # and the per-node spread reached 2.4x, which would confound any
        # per-tier analysis. A 2x cap approaches the target finely instead.
        #
        # This DOES change the generated data (the cap binds at 0.25 too), so it
        # was applied as a deliberate all-datasets regeneration on 2026-08-10 —
        # every raw dataset, and every LSTM-AD / PCMCI+ score derived from one,
        # was rebuilt together. Do not change it again without doing the same.
        n_events = max(1, min(n_events, total_events))

    return {
        "train":        train.astype(np.float32),      # [t_train, K]
        "test":         test.astype(np.float32),       # [t_test,  K]
        "labels":       labels,                        # [t_test,  K]
        "causal_graph": A,                             # [K, K, tau_max+1]
        "out_strength": _out_degree_strength(A),       # [K] — system-wide
        "events":       events,
        "realised_density": float(labels.mean()),
    }


def count_cross_node_edges(A: np.ndarray, n_nodes: int, n_metrics: int) -> dict:
    """Edge counts of a system graph, split into within-node and cross-node."""
    K = n_nodes * n_metrics
    idx = np.arange(K)
    node_of = idx // n_metrics
    nz_i, nz_j, nz_t = np.nonzero(A[:, :, 1:] != 0)
    same_node = node_of[nz_i] == node_of[nz_j]
    self_loop = nz_i == nz_j
    return {
        "within_node": int((same_node & ~self_loop).sum()),
        "cross_node":  int((~same_node).sum()),
        "self_loops":  int(self_loop.sum()),
    }


# ══════════════════════════════════════════════════════════════════
# Interpretation labels (SMD format)
# ══════════════════════════════════════════════════════════════════

def encode_interp_label(labels: np.ndarray) -> str:
    """
    Encode a [T, M] binary label array as an SMD interpretation-label line.

    Format: space-separated `start-end:m1,m2,...` events, INCLUSIVE end,
    1-based metric indices — exactly what `parse_interp_label()` in
    `scripts/compute_metrics.py` expects, so the encoding round-trips.

    One event is emitted per maximal time interval over which the set of
    anomalous metrics is constant and non-empty.
    """
    T = labels.shape[0]
    parts: List[str] = []
    prev: Tuple[int, ...] = ()
    start = 0

    for t in range(T):
        cur = tuple(int(m) for m in np.nonzero(labels[t])[0])
        if cur != prev:
            if prev:
                parts.append(
                    f"{start}-{t - 1}:" + ",".join(str(m + 1) for m in prev)
                )
            prev, start = cur, t
    if prev:
        parts.append(f"{start}-{T - 1}:" + ",".join(str(m + 1) for m in prev))

    return " ".join(parts)


# ══════════════════════════════════════════════════════════════════
# Writing
# ══════════════════════════════════════════════════════════════════

def _write_csv(path: str, values: np.ndarray, is_anomaly: np.ndarray = None):
    """
    Write one entity CSV.  The timestamp column is a plain integer range, which
    `timeeval_pipeline._load_machine_data` recognises as an index and drops.
    """
    T, M = values.shape
    header = ["timestamp"] + [f"metric_{m}" for m in range(M)]
    cols = [np.arange(T)] + [values[:, m] for m in range(M)]
    if is_anomaly is not None:
        header.append("is_anomaly")
        cols.append(is_anomaly)

    with open(path, "w") as f:
        f.write(",".join(header) + "\n")
        for t in range(T):
            row = [str(int(cols[0][t]))]
            row += [f"{cols[m + 1][t]:.6f}" for m in range(M)]
            if is_anomaly is not None:
                row.append(str(int(cols[-1][t])))
            f.write(",".join(row) + "\n")


def dataset_key(n_nodes: int, n_metrics: int, anomaly_density: float,
                cross_node_density: float, data_seed: int) -> str:
    """
    Canonical identifier for one generated dataset.

        <N>nodes-<M>metrics-d<density>[-x<cross_density>]-seed<S>

    Every parameter that is routinely SWEPT goes in the name, so a sweep cannot
    overwrite its own earlier cells.  `data/synthetic_data.py::pool_config_key`
    documents the same failure for episode pools: "an anomaly-density sweep at
    fixed seeds makes each cell overwrite the previous one's pool".  The raw
    generator had exactly that bug — `--anomaly-density 0.05` and 0.25 both
    resolved to `...5nodes-7metrics-seed42`, and the second run silently
    replaced the first while every derived score directory kept pointing at it.

    The cross-node term appears only when non-zero, so within-node dataset names
    stay short and the legacy "xnode" prefix is no longer needed.

    Parameters NOT in the name (tau_max, edge_density, coefficient range,
    T_train/T_test, label_threshold) are covered instead by the meta.json guard
    in `main()` — encoding all of them would make the path unreadable.
    """
    key = f"{n_nodes}nodes-{n_metrics}metrics-d{anomaly_density:g}"
    if cross_node_density > 0:
        key += f"-x{cross_node_density:g}"
    return key + f"-seed{data_seed}"


def check_existing(out_dir: str, requested: dict, overwrite: bool) -> None:
    """
    Refuse to silently replace a dataset generated with different parameters.

    `dataset_key` covers the swept parameters; this covers the rest. Comparing
    against the recorded meta.json is what makes a stale directory an error
    rather than a silent, invisible mismatch with the derived scores computed
    from it.
    """
    meta_path = os.path.join(out_dir, "meta.json")
    if not os.path.exists(meta_path):
        return
    with open(meta_path) as f:
        old = json.load(f)
    diff = {k: (old.get(k), v) for k, v in requested.items()
            if k in old and old.get(k) != v}
    if not diff:
        return
    if overwrite:
        print(f"[raw] WARNING: overwriting a dataset generated with different "
              f"parameters: {diff}")
        return
    print(f"[raw] ERROR: {out_dir} already holds a dataset generated with "
          f"different parameters.")
    for k, (was, now) in diff.items():
        print(f"         {k}: existing={was!r}  requested={now!r}")
    print("       These differences are not in the directory name, so writing "
          "here would replace it invisibly — and any derived anomaly/causal "
          "scores computed from it would silently become stale.")
    print("       Use --out-dir for a different path, or --overwrite if that "
          "is genuinely what you want.")
    sys.exit(1)


def build_dataset(
    n_nodes: int,
    n_metrics: int,
    out_dir: str,
    data_seed: int,
    anomaly_density: float,
    t_train: int = DEFAULT_T_TRAIN,
    t_test: int = DEFAULT_T_TEST,
    topology: Dict = None,
    cross_node_density: float = DEFAULT_CROSS_NODE_DENSITY,
    cross_coef_scale: float = DEFAULT_CROSS_NODE_COEF_SCALE,
    **node_kwargs,
) -> dict:
    """
    Generate the dataset, write it to `out_dir`, return the metadata.

    Two generation paths, selected by `cross_node_density`:

      == 0  per-node.  N independent VARs, one per node, each with its own RNG
            stream.  This is the original behaviour and is preserved EXACTLY —
            same RNG draw order, same per-node stability rescaling — so datasets
            generated before cross-node support existed still reproduce
            byte-identically from the same seed.

      >  0  system-wide.  One joint VAR over all N*M metrics, so anomalies
            propagate between nodes along the graph.

    The paths are kept separate rather than unified because unifying them would
    silently change the zero case: joint stability rescaling applies one shared
    factor to every block, whereas the per-node path rescales each block on its
    own, and the resulting series differ.

    Capacities, costs and tier labels are produced by
    `SyntheticMonitoringDataset.generate_capacities()` rather than being
    reimplemented here, so the config-driven tier semantics apply unchanged.
    """
    topology = topology or {}
    os.makedirs(out_dir, exist_ok=True)

    # Reuse the existing topology machinery for capacities / costs / tiers
    topo_gen = SyntheticMonitoringDataset(
        n_nodes   = n_nodes,
        n_metrics = n_metrics,
        seed      = data_seed,
        **topology,
    )
    caps, costs, tier_labels = topo_gen.generate_capacities()
    n_iot, n_edge, n_cloud   = topo_gen.tier_counts()

    system_mode = cross_node_density > 0.0
    node_meta   = []
    edge_counts = None

    if system_mode:
        # ── Joint path ────────────────────────────────────────────
        rng = np.random.default_rng(np.random.SeedSequence(data_seed))
        sysd = generate_system(
            n_nodes            = n_nodes,
            n_metrics          = n_metrics,
            tier_labels        = tier_labels,
            t_train            = t_train,
            t_test             = t_test,
            anomaly_density    = anomaly_density,
            rng                = rng,
            cross_node_density = cross_node_density,
            cross_coef_scale   = cross_coef_scale,
            **node_kwargs,
        )
        A_sys       = sysd["causal_graph"]                       # [K, K, tau+1]
        edge_counts = count_cross_node_edges(A_sys, n_nodes, n_metrics)
        out_strength = sysd["out_strength"].reshape(n_nodes, n_metrics)

        for n in range(n_nodes):
            name = f"node_{n:02d}"
            sl   = slice(n * n_metrics, (n + 1) * n_metrics)
            labels_n = sysd["labels"][:, sl]

            _write_csv(os.path.join(out_dir, f"{name}.train.csv"),
                       sysd["train"][:, sl])
            _write_csv(os.path.join(out_dir, f"{name}.test.csv"),
                       sysd["test"][:, sl],
                       is_anomaly=labels_n.max(axis=1))
            with open(os.path.join(out_dir, f"{name}.interp_label.txt"), "w") as f:
                f.write(encode_interp_label(labels_n) + "\n")

            node_meta.append({
                "name":             name,
                "tier":             tier_labels[n],
                "capacity":         float(caps[n]),
                "cost":             float(costs[n]),
                "realised_density": float(labels_n.mean()),
                "out_strength":     out_strength[n].tolist(),
            })
            print(f"  {name}: tier={tier_labels[n]:<5} cap={int(caps[n]):>2}/{n_metrics}  "
                  f"density={labels_n.mean():.3f}")

        np.savez_compressed(
            os.path.join(out_dir, "causal_graph.npz"),
            coefficients = A_sys,                 # [K, K, tau_max+1]
            out_strength = out_strength,          # [N, M]
            layout       = np.array("system"),
            n_nodes      = np.array(n_nodes),
            n_metrics    = np.array(n_metrics),
        )
        realised = sysd["realised_density"]
        n_events = len(sysd["events"])

    else:
        # ── Per-node path (unchanged) ─────────────────────────────
        node_seeds = np.random.SeedSequence(data_seed).spawn(n_nodes)
        graphs, strengths = [], []

        for n in range(n_nodes):
            name = f"node_{n:02d}"
            rng  = np.random.default_rng(node_seeds[n])

            node = generate_node(
                n_metrics       = n_metrics,
                t_train         = t_train,
                t_test          = t_test,
                anomaly_density = anomaly_density,
                rng             = rng,
                **node_kwargs,
            )

            _write_csv(os.path.join(out_dir, f"{name}.train.csv"), node["train"])
            _write_csv(
                os.path.join(out_dir, f"{name}.test.csv"),
                node["test"],
                is_anomaly=node["labels"].max(axis=1),
            )
            with open(os.path.join(out_dir, f"{name}.interp_label.txt"), "w") as f:
                f.write(encode_interp_label(node["labels"]) + "\n")

            graphs.append(node["causal_graph"])
            strengths.append(node["out_strength"])
            node_meta.append({
                "name":             name,
                "tier":             tier_labels[n],
                "capacity":         float(caps[n]),
                "cost":             float(costs[n]),
                "realised_density": node["realised_density"],
                "n_events":         len(node["events"]),
                "n_causal_edges":   int((np.abs(node["causal_graph"]) > 0).sum()),
            })
            print(f"  {name}: tier={tier_labels[n]:<5} cap={int(caps[n]):>2}/{n_metrics}  "
                  f"density={node['realised_density']:.3f}  "
                  f"events={len(node['events']):>3}  "
                  f"edges={node_meta[-1]['n_causal_edges']:>3}")

        np.savez_compressed(
            os.path.join(out_dir, "causal_graph.npz"),
            coefficients = np.stack(graphs),      # [N, M, M, tau_max+1]
            out_strength = np.stack(strengths),   # [N, M]
            layout       = np.array("per_node"),
            n_nodes      = np.array(n_nodes),
            n_metrics    = np.array(n_metrics),
        )
        realised = float(np.mean([m["realised_density"] for m in node_meta]))
        n_events = int(sum(m["n_events"] for m in node_meta))

    meta = {
        "data_seed":       data_seed,
        "n_nodes":         n_nodes,
        "n_metrics":       n_metrics,
        "t_train":         t_train,
        "t_test":          t_test,
        "anomaly_density_requested": anomaly_density,
        "anomaly_density_realised":  realised,
        "graph_layout":    "system" if system_mode else "per_node",
        "cross_node_density":    cross_node_density,
        "cross_node_coef_scale": cross_coef_scale if system_mode else None,
        "tier_coupling":   ({f"{a}->{b}": w for (a, b), w in TIER_COUPLING.items()}
                            if system_mode else None),
        "edge_counts":     edge_counts,
        "n_events_total":  n_events,
        "tau_max":         node_kwargs.get("tau_max", DEFAULT_TAU_MAX),
        "edge_density":    node_kwargs.get("edge_density", DEFAULT_EDGE_DENSITY),
        "label_threshold": node_kwargs.get("label_threshold",
                                           DEFAULT_LABEL_THRESHOLD),
        "tier_counts":     {"iot": n_iot, "edge": n_edge, "cloud": n_cloud},
        "node_capacities": caps.tolist(),
        "node_costs":      costs.tolist(),
        "node_tier_labels": tier_labels,
        "nodes":           node_meta,
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return meta


# ══════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════

def main():
    import argparse
    import yaml

    p = argparse.ArgumentParser(
        description="Generate raw-value synthetic telemetry with a "
                    "known causal graph, for the real outer models."
    )
    p.add_argument("--config",    default="configs/config.yaml")
    p.add_argument("--data-seed", type=int, default=None,
                   help="Overrides data.seed from the config")
    p.add_argument("--n-nodes",   type=int, default=None)
    p.add_argument("--n-metrics", type=int, default=None)
    p.add_argument("--out-dir",   default=None,
                   help="Default: data/raw-synthetic-<N>nodes-<M>metrics-"
                        "d<density>[-x<cross_density>]-seed<S>")
    p.add_argument("--overwrite", action="store_true",
                   help="Replace an existing dataset whose recorded generation "
                        "parameters differ from the requested ones. Off by "
                        "default: overwriting invalidates every derived "
                        "anomaly/causal score computed from it, with nothing "
                        "to indicate it happened.")
    p.add_argument("--anomaly-density", type=float, default=None)
    p.add_argument("--t-train",   type=int,   default=DEFAULT_T_TRAIN)
    p.add_argument("--t-test",    type=int,   default=DEFAULT_T_TEST)
    p.add_argument("--tau-max",   type=int,   default=DEFAULT_TAU_MAX)
    p.add_argument("--edge-density",    type=float, default=DEFAULT_EDGE_DENSITY,
                   help="P(edge) over ordered metric pairs WITHIN a node")
    p.add_argument("--cross-node-density", type=float,
                   default=DEFAULT_CROSS_NODE_DENSITY,
                   help="Base P(edge) for ordered metric pairs ACROSS nodes, "
                        "multiplied by the tier-coupling weight for the "
                        "(source tier, target tier) pair. Set to 0 for "
                        "within-node-only generation, which uses the original "
                        "per-node code path and reproduces pre-cross-node "
                        "datasets byte-identically.")
    p.add_argument("--cross-node-coef-scale", type=float,
                   default=DEFAULT_CROSS_NODE_COEF_SCALE,
                   help="Cross-node coefficients are scaled by this relative to "
                        "within-node ones, since coupling between hosts is "
                        "mediated by the network and by aggregation.")
    p.add_argument("--ar-coef",         type=float, default=DEFAULT_AR_COEF)
    p.add_argument("--noise-std",       type=float, default=DEFAULT_NOISE_STD)
    p.add_argument("--burn-in",         type=int,   default=DEFAULT_BURN_IN)
    p.add_argument("--label-threshold", type=float,
                   default=DEFAULT_LABEL_THRESHOLD)
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    ecfg, dcfg = cfg["environment"], cfg["data"]

    N = args.n_nodes   if args.n_nodes   is not None else ecfg["n_nodes"]
    M = args.n_metrics if args.n_metrics is not None else ecfg["n_metrics"]
    seed = args.data_seed if args.data_seed is not None else int(dcfg["seed"])
    density = (args.anomaly_density if args.anomaly_density is not None
               else dcfg["anomaly_density"])

    key = dataset_key(N, M, density, args.cross_node_density, seed)
    out_dir = args.out_dir or os.path.join("data", f"raw-synthetic-{key}")

    # Guard the generation parameters that are NOT encoded in the name.
    check_existing(out_dir, {
        "t_train":         args.t_train,
        "t_test":          args.t_test,
        "tau_max":         args.tau_max,
        "edge_density":    args.edge_density,
        "label_threshold": args.label_threshold,
    }, args.overwrite)

    iot_f, edge_f = tier_fracs(ecfg)
    print(f"[raw] key={key}")
    print(f"[raw] N={N} M={M} seed={seed} density={density} "
          f"cross_node={args.cross_node_density}")
    print(f"[raw] tier fractions: iot={iot_f} edge={edge_f}")
    print(f"[raw] T_train={args.t_train} T_test={args.t_test} "
          f"tau_max={args.tau_max}")
    print(f"[raw] → {out_dir}\n")

    meta = build_dataset(
        n_nodes         = N,
        n_metrics       = M,
        out_dir         = out_dir,
        data_seed       = seed,
        anomaly_density = density,
        t_train         = args.t_train,
        t_test          = args.t_test,
        topology        = topology_kwargs(ecfg),
        cross_node_density = args.cross_node_density,
        cross_coef_scale   = args.cross_node_coef_scale,
        tau_max         = args.tau_max,
        edge_density    = args.edge_density,
        ar_coef         = args.ar_coef,
        noise_std       = args.noise_std,
        burn_in         = args.burn_in,
        label_threshold = args.label_threshold,
    )

    print(f"\n[raw] Realised anomaly density: "
          f"{meta['anomaly_density_realised']:.3f} "
          f"(requested {density})")
    print(f"[raw] Tier counts: {meta['tier_counts']}")
    print(f"[raw] Written to {out_dir}")


if __name__ == "__main__":
    main()
