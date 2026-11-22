"""
synthetic_data.py
=================
Generates synthetic monitoring datasets for the adaptive metric selection problem.

Each dataset contains:
  - anomaly_scores : np.ndarray [T, N, M]   time-varying anomaly scores in [0,1]
  - causal_scores  : np.ndarray [N, M]       static causal influence scores in [0,1]
  - anomaly_labels : np.ndarray [T, N, M]   binary ground-truth anomaly labels
  - node_capacities: np.ndarray [N]          per-node monitoring budget

FIX: capacities are now defined as fractions of M (metrics per node) rather than
hardcoded absolute integers.  This ensures they scale correctly regardless of M:

  Old (broken):
    IoT capacity = randint(2, 5)     → nonsensical if M=7  (cap can exceed M)
                                     → extremely tight if M=38 (cap=3/38 = 8%)

  New (correct):
    IoT capacity = randint(lo, hi) * M, where lo/hi are tier fractions
    → always a sensible fraction of M regardless of dataset size

Default fractions (configurable):
    IoT   : 20%–40% of M   (constrained device)
    Edge  : 40%–70% of M   (moderate resources)
    Cloud : 70%–90% of M   (abundant resources)
"""

import hashlib
import json
import numpy as np
import os
import pickle
from typing import Optional, Tuple, Dict


# ------------------------------------------------------------------
# Core generator
# ------------------------------------------------------------------

class SyntheticMonitoringDataset:
    """
    Generates a synthetic multi-node, multi-metric monitoring dataset
    with controllable anomaly patterns and causal structure.

    Parameters
    ----------
    n_nodes      : int   – Number of nodes (heterogeneous tiers)
    n_metrics    : int   – Metrics exposed per node
    T            : int   – Number of timesteps
    anomaly_density : float – Approximate fraction of anomalous (t,n,m) cells
    causal_density  : float – Fraction of metrics with high causal influence
    iot_frac     : float – Fraction of nodes that are IoT tier
    edge_frac    : float – Fraction of nodes that are Edge tier
    seed         : int   – Random seed

    Capacity fractions (as fraction of M):
    iot_cap_range   : (lo, hi) – IoT capacity in [lo*M, hi*M)
    edge_cap_range  : (lo, hi) – Edge capacity in [lo*M, hi*M)
    cloud_cap_range : (lo, hi) – Cloud capacity in [lo*M, hi*M)
    """

    def __init__(
        self,
        n_nodes: int = 10,
        n_metrics: int = 15,
        T: int = 512,
        anomaly_density: float = 0.15,
        causal_density: float = 0.25,
        iot_frac: float = 0.3,
        edge_frac: float = 0.4,
        seed: int = 42,
        # Capacity fractions of M — configurable, scale with M
        iot_cap_range:   Tuple[float, float] = (0.20, 0.40),
        edge_cap_range:  Tuple[float, float] = (0.40, 0.70),
        cloud_cap_range: Tuple[float, float] = (0.70, 0.90),
        # Per-tier monitoring cost (per selected metric per timestep)
        iot_cost:   float = 0.05,
        edge_cost:  float = 0.02,
        cloud_cost: float = 0.005,
    ):
        self.n_nodes  = n_nodes
        self.n_metrics = n_metrics
        self.T        = T
        self.anomaly_density = anomaly_density
        self.causal_density  = causal_density
        self.iot_frac  = iot_frac
        self.edge_frac = edge_frac
        self.seed = seed
        self.rng  = np.random.default_rng(seed)

        # Capacity fraction ranges — stored so they appear in episode dict
        self.iot_cap_range   = iot_cap_range
        self.edge_cap_range  = edge_cap_range
        self.cloud_cap_range = cloud_cap_range

        # Per-tier costs — stored so they appear in the episode dict
        self.iot_cost   = iot_cost
        self.edge_cost  = edge_cost
        self.cloud_cost = cloud_cost

    # ----------------------------------------------------------
    # Node capacities and costs (heterogeneous tiers)
    # ----------------------------------------------------------

    def tier_counts(self) -> Tuple[int, int, int]:
        """
        Number of (IoT, edge, cloud) nodes for this topology.

        Cloud is the remainder, never an independent parameter.  Integer
        flooring means the realised split can differ from the requested
        fractions when N is small: N=5 with 0.3/0.4 yields 1/2/2, i.e. an
        actual 20/40/40 rather than the requested 30/40/30.
        """
        n = self.n_nodes
        n_iot   = max(1, int(n * self.iot_frac))
        n_edge  = max(1, int(n * self.edge_frac))
        n_cloud = max(1, n - n_iot - n_edge)
        return n_iot, n_edge, n_cloud

    def generate_capacities(self) -> Tuple[np.ndarray, np.ndarray, list]:
        """
        Assign monitoring budgets and per-metric monitoring costs
        based on node tier.

        Capacities are sampled as integer fractions of M so they
        always make sense regardless of how large or small M is.

        With default fractions and M=15:
            IoT   → [3,  6)    (20%–40% of 15)
            Edge  → [6, 11)    (40%–70% of 15)
            Cloud → [11, 14)   (70%–90% of 15)

        With M=7:
            IoT   → [2,  3)    (20%–40% of 7)
            Edge  → [3,  5)    (40%–70% of 7)
            Cloud → [5,  7)    (70%–90% of 7)

        With M=38:
            IoT   → [8, 16)    (20%–40% of 38)
            Edge  → [16, 27)   (40%–70% of 38)
            Cloud → [27, 35)   (70%–90% of 38)

        Capacities are clipped to [1, M] as a safety guarantee.

        Costs (per selected metric per timestep) come from self.iot_cost /
        self.edge_cost / self.cloud_cost, which default to:
            IoT   → 0.05   (constrained device — high cost)
            Edge  → 0.02   (moderate resources)
            Cloud → 0.005  (abundant resources — low cost)

        Returns
        -------
        capacities  : np.ndarray [N]  float32
        node_costs  : np.ndarray [N]  float32
        tier_labels : list[str]  [N]  "iot" | "edge" | "cloud"

        tier_labels is returned because the shuffle below destroys the
        positional tier grouping.  Downstream code must not re-derive the tier
        by thresholding the cost — that only works for the default cost values.
        """
        n = self.n_nodes
        M = self.n_metrics

        n_iot, n_edge, n_cloud = self.tier_counts()

        def sample_caps(lo_frac, hi_frac, size):
            """
            Sample `size` integer capacities uniformly in
            [round(lo_frac * M), round(hi_frac * M)).
            Always at least 1, always at most M.
            """
            lo = max(1, round(lo_frac * M))
            hi = max(lo + 1, round(hi_frac * M))   # hi > lo guaranteed
            hi = min(hi, M + 1)                     # randint upper is exclusive
            return self.rng.integers(lo, hi, size=size)

        caps_iot   = sample_caps(*self.iot_cap_range,   n_iot)
        caps_edge  = sample_caps(*self.edge_cap_range,  n_edge)
        caps_cloud = sample_caps(*self.cloud_cap_range, n_cloud)

        caps = np.concatenate([caps_iot, caps_edge, caps_cloud])

        costs = np.concatenate([
            np.full(n_iot,   self.iot_cost,   dtype=np.float32),   # IoT
            np.full(n_edge,  self.edge_cost,  dtype=np.float32),   # Edge
            np.full(n_cloud, self.cloud_cost, dtype=np.float32),   # Cloud
        ])

        # Tier label per node, captured here while the grouping is still
        # positional — the shuffle below discards that information.
        labels = np.array(
            ["iot"] * n_iot + ["edge"] * n_edge + ["cloud"] * n_cloud,
            dtype=object,
        )

        # Shuffle all three arrays with the same permutation so each node
        # keeps its correct (capacity, cost, tier) triple after shuffling
        perm   = self.rng.permutation(n)
        caps   = caps[perm][:n].astype(np.float32)
        costs  = costs[perm][:n].astype(np.float32)
        labels = labels[perm][:n]

        # Safety clip — cap must be in [1, M]
        caps = np.clip(caps, 1, M).astype(np.float32)

        return caps, costs, [str(x) for x in labels]

    # ----------------------------------------------------------
    # Causal scores (static per-metric structural prior)
    # ----------------------------------------------------------

    def generate_causal_scores(self) -> np.ndarray:
        """
        Static [N, M] causal scores in [0,1].
        A fraction `causal_density` of metrics per node have high influence.
        Scores are normalized per node.
        """
        scores = np.zeros((self.n_nodes, self.n_metrics), dtype=np.float32)
        for n in range(self.n_nodes):
            n_high   = max(1, int(self.n_metrics * self.causal_density))
            high_idx = self.rng.choice(self.n_metrics, size=n_high, replace=False)
            scores[n, high_idx] = self.rng.beta(5, 1, size=n_high).astype(np.float32)
            low_idx  = np.setdiff1d(np.arange(self.n_metrics), high_idx)
            scores[n, low_idx]  = self.rng.beta(1, 5, size=len(low_idx)).astype(np.float32)
            max_s = scores[n].max()
            if max_s > 0:
                scores[n] /= max_s
        return scores

    # ----------------------------------------------------------
    # Anomaly scores (time-varying) + binary labels
    # ----------------------------------------------------------

    def generate_anomaly_scores(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Time-varying anomaly scores [T, N, M] in [0,1] and binary labels [T, N, M].

        Strategy:
        - Background: Beta(1, 8) — low scores
        - Anomaly events injected as bursts:
            * Random onset timestep
            * Duration: geometric(p=0.12) → typically 5–20 steps
            * Score during anomaly: Beta(6, 2) — high scores
        - Light exponential smoothing to simulate LSTM-AD output
        - Per-(node, metric) normalization to [0,1]
        """
        N, M, T = self.n_nodes, self.n_metrics, self.T
        raw_scores = np.zeros((T, N, M), dtype=np.float32)
        labels     = np.zeros((T, N, M), dtype=np.int32)

        for n in range(N):
            for m in range(M):
                raw_scores[:, n, m] = self.rng.beta(1, 8, size=T).astype(np.float32)

        # Inject anomaly events
        target_cells = int(self.anomaly_density * T * N * M)
        cells_injected = 0
        max_attempts   = target_cells * 10
        attempt = 0

        while cells_injected < target_cells and attempt < max_attempts:
            attempt += 1
            n       = int(self.rng.integers(0, N))
            m       = int(self.rng.integers(0, M))
            t_start = int(self.rng.integers(0, T))
            duration = min(int(self.rng.geometric(p=0.12)), T - t_start)
            t_end    = t_start + duration

            anomaly_vals = self.rng.beta(6, 2, size=duration).astype(np.float32)
            raw_scores[t_start:t_end, n, m] = np.maximum(
                raw_scores[t_start:t_end, n, m], anomaly_vals
            )
            labels[t_start:t_end, n, m] = 1
            cells_injected += duration

        # Exponential smoothing
        alpha_smooth = 0.3
        smoothed = raw_scores.copy()
        for t in range(1, T):
            smoothed[t] = alpha_smooth * raw_scores[t] + (1 - alpha_smooth) * smoothed[t - 1]

        # Per-(node, metric) normalization to [0,1]
        for n in range(N):
            for m in range(M):
                s     = smoothed[:, n, m]
                rng_s = s.max() - s.min()
                if rng_s > 1e-6:
                    smoothed[:, n, m] = (s - s.min()) / rng_s
                else:
                    smoothed[:, n, m] = 0.0

        return smoothed.astype(np.float32), labels.astype(np.int32)

    # ----------------------------------------------------------
    # Build full dataset
    # ----------------------------------------------------------

    def generate(self) -> Dict[str, np.ndarray]:
        capacities, node_costs, tier_labels = self.generate_capacities()
        causal_scores                  = self.generate_causal_scores()
        anomaly_scores, anomaly_labels = self.generate_anomaly_scores()

        n_iot, n_edge, n_cloud = self.tier_counts()

        return {
            "anomaly_scores":  anomaly_scores,   # [T, N, M]
            "anomaly_labels":  anomaly_labels,   # [T, N, M]
            "causal_scores":   causal_scores,    # [N, M]
            "node_capacities": capacities,       # [N]  always in [1, M]
            "node_costs":      node_costs,       # [N]  per-metric cost per node
            "n_nodes":         self.n_nodes,
            "n_metrics":       self.n_metrics,
            "T":               self.T,
            # Store fractions so the episode is self-documenting
            "cap_fractions": {
                "iot":   self.iot_cap_range,
                "edge":  self.edge_cap_range,
                "cloud": self.cloud_cap_range,
            },
            # Requested tier fractions vs. the split actually realised after
            # integer flooring — these differ for small N, so record both
            # rather than leaving the realised topology to be inferred.
            "tier_fractions": {
                "iot":   self.iot_frac,
                "edge":  self.edge_frac,
                "cloud": max(0.0, 1.0 - self.iot_frac - self.edge_frac),
            },
            "tier_counts": {
                "iot":   n_iot,
                "edge":  n_edge,
                "cloud": n_cloud,
            },
            # Per-node tier label, aligned with node_capacities / node_costs.
            # Authoritative: downstream code must read this rather than
            # re-deriving the tier by thresholding node_costs, which only
            # works for the default 0.05 / 0.02 / 0.005 values.
            "node_tier_labels": tier_labels,   # list[str] length N
            "tier_costs": {
                "iot":   self.iot_cost,
                "edge":  self.edge_cost,
                "cloud": self.cloud_cost,
            },
            # Generation parameters, so a cached pool can be checked against the
            # config that wants to use it.  Without these a density sweep
            # silently reuses the first pool it generated.
            "gen_params": {
                "anomaly_density": self.anomaly_density,
                "causal_density":  self.causal_density,
                "episode_length":  self.T,
            },
        }


# ------------------------------------------------------------------
# Compute baselines b_a, b_c from dataset
# ------------------------------------------------------------------

def compute_baselines(
    dataset: Dict[str, np.ndarray],
    anomaly_pct: int = 50,
    causal_pct:  int = 50,
) -> Tuple[float, float]:
    """
    Compute anomaly baseline b_a and causal baseline b_c
    as percentiles of their respective score distributions.
    """
    b_a = float(np.percentile(dataset["anomaly_scores"], anomaly_pct))
    b_c = float(np.percentile(dataset["causal_scores"],  causal_pct))
    return b_a, b_c


# ------------------------------------------------------------------
# Config readers
# ------------------------------------------------------------------
#
# These exist so that the topology parameters are read from the YAML in exactly
# one place.  Previously every caller inlined its own `.get(..., default)` chain,
# which let the configs and the code drift apart silently: `node_tiers` was
# parsed by nobody, and the capacity ranges were supplied by a fallback that
# five of six config files never overrode.

DEFAULT_IOT_FRAC  = 0.3
DEFAULT_EDGE_FRAC = 0.4

DEFAULT_IOT_CAP_RANGE   = (0.20, 0.40)
DEFAULT_EDGE_CAP_RANGE  = (0.40, 0.70)
DEFAULT_CLOUD_CAP_RANGE = (0.70, 0.90)

DEFAULT_IOT_COST   = 0.05
DEFAULT_EDGE_COST  = 0.02
DEFAULT_CLOUD_COST = 0.005


def tier_fracs(ecfg: Dict) -> Tuple[float, float]:
    """
    Read the IoT / edge node-tier fractions from an `environment` config block.

    Returns (iot_frac, edge_frac).  Cloud is always the remainder and is never
    read as an independent value, because the generator derives it that way
    (see SyntheticMonitoringDataset.tier_counts).

    Falls back to (0.3, 0.4) when `node_tiers` is absent, matching the
    historical hardcoded defaults.
    """
    tiers = ecfg.get("node_tiers") or {}

    iot_frac  = float(tiers.get("iot",  DEFAULT_IOT_FRAC))
    edge_frac = float(tiers.get("edge", DEFAULT_EDGE_FRAC))

    for name, val in (("iot", iot_frac), ("edge", edge_frac)):
        if not 0.0 <= val <= 1.0:
            raise ValueError(
                f"node_tiers.{name} must be in [0, 1], got {val}"
            )
    if iot_frac + edge_frac > 1.0:
        raise ValueError(
            f"node_tiers.iot + node_tiers.edge must be <= 1.0 "
            f"(cloud takes the remainder), got {iot_frac} + {edge_frac} "
            f"= {iot_frac + edge_frac}"
        )

    if "cloud" in tiers:
        total = iot_frac + edge_frac + float(tiers["cloud"])
        if abs(total - 1.0) > 1e-6:
            print(f"[data] WARNING: node_tiers sum to {total:.3f}, not 1.0. "
                  f"cloud is derived as the remainder, so its configured value "
                  f"({tiers['cloud']}) is ignored.")

    return iot_frac, edge_frac


def cap_ranges(ecfg: Dict) -> Tuple[Tuple[float, float],
                                    Tuple[float, float],
                                    Tuple[float, float]]:
    """
    Read the per-tier capacity ranges from an `environment` config block.

    Returns (iot_cap_range, edge_cap_range, cloud_cap_range), each a
    (lo, hi) pair expressed as a fraction of M.
    """
    out = []
    for name, default in (("iot_cap_range",   DEFAULT_IOT_CAP_RANGE),
                          ("edge_cap_range",  DEFAULT_EDGE_CAP_RANGE),
                          ("cloud_cap_range", DEFAULT_CLOUD_CAP_RANGE)):
        lo, hi = tuple(ecfg.get(name, default))
        if not 0.0 <= lo < hi <= 1.0:
            raise ValueError(
                f"{name} must satisfy 0 <= lo < hi <= 1, got [{lo}, {hi}]"
            )
        out.append((float(lo), float(hi)))
    return tuple(out)


def tier_costs(ecfg: Dict) -> Tuple[float, float, float]:
    """
    Read the per-tier monitoring costs from an `environment` config block.

    Returns (iot_cost, edge_cost, cloud_cost), the cost charged per selected
    metric per timestep on a node of that tier.

    Falls back to (0.05, 0.02, 0.005) when the keys are absent, matching the
    values that were previously hardcoded in generate_capacities().
    """
    out = []
    for name, default in (("cost_iot",   DEFAULT_IOT_COST),
                          ("cost_edge",  DEFAULT_EDGE_COST),
                          ("cost_cloud", DEFAULT_CLOUD_COST)):
        val = float(ecfg.get(name, default))
        if val <= 0.0:
            raise ValueError(f"{name} must be > 0, got {val}")
        out.append(val)

    iot_c, edge_c, cloud_c = out
    if not (iot_c >= edge_c >= cloud_c):
        print(f"[data] WARNING: monitoring costs do not follow the expected "
              f"IoT >= edge >= cloud ordering "
              f"(iot={iot_c}, edge={edge_c}, cloud={cloud_c}). "
              f"The qualitative ordering is the one continuum assumption the "
              f"literature supports, so this is probably a mistake.")

    return iot_c, edge_c, cloud_c


def assign_tiers(episode_or_costs, node_costs=None) -> list:
    """
    Return the per-node tier label list for an episode.

    Prefers the authoritative `node_tier_labels` recorded at generation time.
    Falls back to inferring the tier from cost magnitude, which is only valid
    for the default 0.05 / 0.02 / 0.005 values — legacy pools pickled before
    node_tier_labels existed, and real-data episodes that never had tiers.

    Accepts either an episode dict or a raw node_costs array, so existing
    callers that pass costs directly keep working.
    """
    if isinstance(episode_or_costs, dict):
        labels = episode_or_costs.get("node_tier_labels")
        if labels:
            return list(labels)
        costs = episode_or_costs.get("node_costs")
        if costs is None:
            raise KeyError("episode has neither node_tier_labels nor node_costs")
    else:
        costs = episode_or_costs

    # Legacy fallback — thresholds tuned to the default cost values.
    return ["iot" if c >= 0.04 else "edge" if c >= 0.015 else "cloud"
            for c in costs]


def pool_config_key(cfg: Dict) -> str:
    """
    Short fingerprint of every parameter that changes what a pool contains.

    The pool path embeds this so two configs that differ in ANY generation
    parameter get different directories.  Without it, an anomaly-density sweep
    at fixed seeds makes each cell overwrite the previous one's pool: correct
    only if run strictly sequentially, and impossible to parallelise.
    """
    ecfg, dcfg = cfg["environment"], cfg["data"]
    iot_f, edge_f = tier_fracs(ecfg)
    payload = {
        "tier_fracs": [iot_f, edge_f],
        "cap_ranges": [list(r) for r in cap_ranges(ecfg)],
        "costs":      list(tier_costs(ecfg)),
        "anomaly_density": dcfg.get("anomaly_density"),
        "causal_density":  dcfg.get("causal_density"),
        "episode_length":  ecfg.get("episode_length"),
        "n_train": dcfg.get("n_train_episodes"),
        "n_eval":  dcfg.get("n_eval_episodes"),
        "anomaly_baseline_pct": ecfg.get("anomaly_baseline_pct"),
        "causal_baseline_pct":  ecfg.get("causal_baseline_pct"),
    }
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:8]


def pool_dir(base_data_dir: str,
             n_nodes: int,
             n_metrics: int,
             data_seed: Optional[int] = None,
             warn_legacy: bool = True,
             config_key: Optional[str] = None) -> str:
    """
    Canonical directory for a generated episode pool.

        data/<N>nodes-<M>metrics-seed<S>/

    The seed is part of the path because a pool IS a function of the data seed.
    Previously the path was keyed by (N, M) only, so a run launched with a
    different seed would silently load a cached pool generated under the old
    one — the seed never reached the data at all.

    Passing data_seed=None reproduces the old unkeyed layout, for readers that
    genuinely want "whatever pool is there" rather than a specific scenario.

    Creates the directory.  If a legacy unkeyed pool exists and the keyed one
    does not, warns — its provenance is unknown, which is precisely the problem
    this keying fixes.  Regeneration is deterministic, so nothing is lost.
    """
    stem = f"{n_nodes}nodes-{n_metrics}metrics"
    if data_seed is None:
        subdir = os.path.join(base_data_dir, stem)
        os.makedirs(subdir, exist_ok=True)
        return subdir

    subdir = os.path.join(base_data_dir, f"{stem}-seed{int(data_seed)}")
    if config_key:
        subdir += f"-{config_key}"

    if warn_legacy and not os.path.exists(subdir):
        legacy = os.path.join(base_data_dir, stem)
        if os.path.exists(os.path.join(legacy, "train_episodes.pkl")):
            print(f"[data] NOTE: found a legacy pool at {legacy} that is not "
                  f"keyed by seed, so the seed that produced it is unknown. "
                  f"Ignoring it and generating a pool for data_seed="
                  f"{data_seed} at {subdir}. The legacy directory can be "
                  f"deleted once no old run depends on it.")

    os.makedirs(subdir, exist_ok=True)
    return subdir


def ensure_pool(cfg: Dict, data_seed: int, log_prefix: str = "[data]"):
    """
    Load the episode pool for `data_seed`, generating it if absent or stale.

    Single source of truth for the load-or-build path.  Both train.py and
    run_multiseed.py's pre-generation step call this, so the two can never
    drift into producing different pools for the same seed.

    Regenerates when the cached pool disagrees with the config's topology —
    see validate_pool_config for why that check is not optional.

    Returns (train_eps, eval_eps, pool_path).
    """
    ecfg, dcfg = cfg["environment"], cfg["data"]
    N, M = ecfg["n_nodes"], ecfg["n_metrics"]

    key  = pool_config_key(cfg)
    path = pool_dir(cfg["paths"]["data_dir"], N, M, data_seed,
                    config_key=key)
    train_path = os.path.join(path, "train_episodes.pkl")
    eval_path  = os.path.join(path, "eval_episodes.pkl")

    cached = os.path.exists(train_path) and os.path.exists(eval_path)
    if cached:
        train_eps = load_dataset(train_path)
        eval_eps  = load_dataset(eval_path)
        mismatches = validate_pool_config(
            train_eps, ecfg, dcfg, dcfg.get("n_train_episodes"))
        if mismatches:
            print(f"{log_prefix} Cached pool does not match the requested topology:")
            for m in mismatches:
                print(f"{log_prefix}   - {m}")
            print(f"{log_prefix} Regenerating the pool for this config.")
            cached = False

    if not cached:
        print(f"{log_prefix} Generating datasets for N={N}, M={M}, "
              f"data_seed={data_seed}...")
        train_eps = build_episode_pool(
            n_episodes           = dcfg["n_train_episodes"],
            n_nodes              = N,
            n_metrics            = M,
            episode_length       = ecfg["episode_length"],
            anomaly_density      = dcfg["anomaly_density"],
            causal_density       = dcfg["causal_density"],
            base_seed            = data_seed,
            anomaly_baseline_pct = ecfg["anomaly_baseline_pct"],
            causal_baseline_pct  = ecfg["causal_baseline_pct"],
            **topology_kwargs(ecfg),
        )
        eval_eps = build_episode_pool(
            n_episodes           = dcfg["n_eval_episodes"],
            n_nodes              = N,
            n_metrics            = M,
            episode_length       = ecfg["episode_length"],
            anomaly_density      = dcfg["anomaly_density"] * 1.2,
            causal_density       = dcfg["causal_density"],
            base_seed            = data_seed + 10000,
            anomaly_baseline_pct = ecfg["anomaly_baseline_pct"],
            causal_baseline_pct  = ecfg["causal_baseline_pct"],
            **topology_kwargs(ecfg),
        )
        save_dataset(train_eps, train_path)
        save_dataset(eval_eps,  eval_path)

    return train_eps, eval_eps, path


def validate_pool_config(episodes: list, ecfg: Dict,
                         dcfg: Optional[Dict] = None,
                         n_expected: Optional[int] = None) -> list:
    """
    Check that a loaded pool matches the config that wants to use it.

    Covers both the topology (tier split, capacity ranges, per-tier costs) and
    the generation parameters (anomaly/causal density, episode length, episode
    count).  Both matter: the pool path is keyed by (N, M, data_seed) only, so
    ANY other parameter can disagree with a cached pool.

    The pool path is keyed by (N, M, data_seed) but NOT by the topology
    parameters, so without this check a sweep that varies tier fractions,
    capacity ranges or per-tier costs at a fixed seed would silently reuse the
    first pool it generated and report identical results for every cell — the
    exact failure mode a sensitivity analysis is meant to detect.

    Returns a list of human-readable mismatch descriptions; empty means the
    cached pool is safe to use.  Pools generated before these fields were
    recorded are skipped rather than reported as mismatched.
    """
    if not episodes:
        return []
    ep = episodes[0]
    mismatches = []

    want_iot, want_edge = tier_fracs(ecfg)
    got = ep.get("tier_fractions")
    if got is not None:
        if (abs(float(got.get("iot", -1)) - want_iot) > 1e-9 or
                abs(float(got.get("edge", -1)) - want_edge) > 1e-9):
            mismatches.append(
                f"node_tiers: pool has iot={got.get('iot')}, edge={got.get('edge')}; "
                f"config wants iot={want_iot}, edge={want_edge}")

    want_caps = cap_ranges(ecfg)
    got = ep.get("cap_fractions")
    if got is not None:
        for name, key, want in (("iot_cap_range", "iot", want_caps[0]),
                                ("edge_cap_range", "edge", want_caps[1]),
                                ("cloud_cap_range", "cloud", want_caps[2])):
            have = got.get(key)
            if have is not None and tuple(float(x) for x in have) != tuple(want):
                mismatches.append(
                    f"{name}: pool has {tuple(have)}, config wants {tuple(want)}")

    want_costs = tier_costs(ecfg)
    got = ep.get("tier_costs")
    if got is not None:
        for key, want in zip(("iot", "edge", "cloud"), want_costs):
            have = got.get(key)
            if have is not None and abs(float(have) - want) > 1e-12:
                mismatches.append(
                    f"cost_{key}: pool has {have}, config wants {want}")

    # Generation parameters — a density sweep changes these and nothing else,
    # so omitting this check makes every cell of the sweep identical.
    got = ep.get("gen_params")
    if got is None:
        # Pool predates provenance recording, so what it was built with cannot
        # be established.  Regenerate rather than trust it: generation is
        # deterministic given the seed, so nothing is lost.
        mismatches.append(
            "gen_params: pool predates provenance recording, so its anomaly/"
            "causal density cannot be verified — regenerating")
    elif dcfg is not None:
        for key, want, label in (
                ("anomaly_density", dcfg.get("anomaly_density"), "anomaly_density"),
                ("causal_density",  dcfg.get("causal_density"),  "causal_density"),
                ("episode_length",  ecfg.get("episode_length"),  "episode_length")):
            have = got.get(key)
            if have is not None and want is not None and \
                    abs(float(have) - float(want)) > 1e-12:
                mismatches.append(
                    f"{label}: pool has {have}, config wants {want}")

    if n_expected is not None and len(episodes) != n_expected:
        mismatches.append(
            f"episode count: pool has {len(episodes)}, config wants {n_expected}")

    return mismatches


def topology_kwargs(ecfg: Dict) -> Dict:
    """
    Every topology parameter for build_episode_pool(), read from an
    `environment` config block.  Call sites use:

        build_episode_pool(..., **topology_kwargs(ecfg))

    so that adding a topology parameter later means editing this function
    rather than ten scripts.
    """
    iot_frac, edge_frac        = tier_fracs(ecfg)
    iot_c, edge_c, cloud_c     = cap_ranges(ecfg)
    iot_k, edge_k, cloud_k     = tier_costs(ecfg)
    return {
        "iot_frac":        iot_frac,
        "edge_frac":       edge_frac,
        "iot_cap_range":   iot_c,
        "edge_cap_range":  edge_c,
        "cloud_cap_range": cloud_c,
        "iot_cost":        iot_k,
        "edge_cost":       edge_k,
        "cloud_cost":      cloud_k,
    }


# ------------------------------------------------------------------
# Multi-episode dataset builder
# ------------------------------------------------------------------

def build_episode_pool(
    n_episodes: int,
    n_nodes: int = 10,
    n_metrics: int = 15,
    episode_length: int = 512,
    anomaly_density: float = 0.15,
    causal_density: float = 0.25,
    base_seed: int = 0,
    anomaly_baseline_pct: int = 30,
    causal_baseline_pct:  int = 50,
    # Node-tier fractions — passed through to SyntheticMonitoringDataset
    iot_frac:  float = DEFAULT_IOT_FRAC,
    edge_frac: float = DEFAULT_EDGE_FRAC,
    # Capacity fractions — passed through to SyntheticMonitoringDataset
    iot_cap_range:   Tuple[float, float] = DEFAULT_IOT_CAP_RANGE,
    edge_cap_range:  Tuple[float, float] = DEFAULT_EDGE_CAP_RANGE,
    cloud_cap_range: Tuple[float, float] = DEFAULT_CLOUD_CAP_RANGE,
    # Per-tier monitoring costs — passed through to SyntheticMonitoringDataset
    iot_cost:   float = DEFAULT_IOT_COST,
    edge_cost:  float = DEFAULT_EDGE_COST,
    cloud_cost: float = DEFAULT_CLOUD_COST,
) -> list:
    """
    Build a pool of n_episodes independent synthetic episodes.

    Node-tier fractions determine how many nodes belong to each layer:
        IoT   : max(1, int(N * iot_frac))
        Edge  : max(1, int(N * edge_frac))
        Cloud : the remainder

    Capacity fractions scale automatically with n_metrics:
        IoT   : [iot_lo * M,   iot_hi * M)
        Edge  : [edge_lo * M,  edge_hi * M)
        Cloud : [cloud_lo * M, cloud_hi * M)

    b_a and b_c are stored per-episode using the configured percentiles.
    """
    episodes = []
    for i in range(n_episodes):
        gen = SyntheticMonitoringDataset(
            n_nodes         = n_nodes,
            n_metrics       = n_metrics,
            T               = episode_length,
            anomaly_density = anomaly_density,
            causal_density  = causal_density,
            seed            = base_seed + i,
            iot_frac        = iot_frac,
            edge_frac       = edge_frac,
            iot_cap_range   = iot_cap_range,
            edge_cap_range  = edge_cap_range,
            cloud_cap_range = cloud_cap_range,
            iot_cost        = iot_cost,
            edge_cost       = edge_cost,
            cloud_cost      = cloud_cost,
        )
        ep = gen.generate()
        b_a, b_c = compute_baselines(
            ep,
            anomaly_pct = anomaly_baseline_pct,
            causal_pct  = causal_baseline_pct,
        )
        ep["b_a"] = b_a
        ep["b_c"] = b_c
        episodes.append(ep)
    return episodes


# ------------------------------------------------------------------
# Save / load
# ------------------------------------------------------------------

def save_dataset(episodes: list, path: str):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(episodes, f)
    print(f"[data] Saved {len(episodes)} episodes → {path}")


def load_dataset(path: str) -> list:
    with open(path, "rb") as f:
        episodes = pickle.load(f)
    print(f"[data] Loaded {len(episodes)} episodes ← {path}")
    return episodes


# ------------------------------------------------------------------
# Quick verification — shows capacity ranges for any (N, M)
# ------------------------------------------------------------------

def print_capacity_summary(n_nodes: int, n_metrics: int,
                           iot_cap_range=DEFAULT_IOT_CAP_RANGE,
                           edge_cap_range=DEFAULT_EDGE_CAP_RANGE,
                           cloud_cap_range=DEFAULT_CLOUD_CAP_RANGE,
                           iot_frac=DEFAULT_IOT_FRAC,
                           edge_frac=DEFAULT_EDGE_FRAC):
    M = n_metrics
    print(f"\nTopology for N={n_nodes}, M={n_metrics}:")

    # Realised node counts — integer flooring means these can differ from the
    # requested fractions, so print both rather than only the request.
    gen = SyntheticMonitoringDataset(n_nodes=n_nodes, n_metrics=n_metrics,
                                     iot_frac=iot_frac, edge_frac=edge_frac)
    n_iot, n_edge, n_cloud = gen.tier_counts()
    cloud_frac = max(0.0, 1.0 - iot_frac - edge_frac)

    print(f"  requested split : IoT {iot_frac*100:.0f}%  "
          f"Edge {edge_frac*100:.0f}%  Cloud {cloud_frac*100:.0f}%")
    print(f"  realised split  : IoT {n_iot}  Edge {n_edge}  Cloud {n_cloud}  "
          f"= {n_iot/n_nodes*100:.0f}% / {n_edge/n_nodes*100:.0f}% / "
          f"{n_cloud/n_nodes*100:.0f}%")
    if (n_iot, n_edge, n_cloud) != (round(n_nodes * iot_frac),
                                    round(n_nodes * edge_frac),
                                    n_nodes - round(n_nodes * iot_frac)
                                            - round(n_nodes * edge_frac)):
        print(f"  NOTE: realised split differs from the requested fractions "
              f"because node counts are floored to integers.")

    print(f"\nCapacity ranges for N={n_nodes}, M={n_metrics}:")
    for tier, (lo, hi) in [("IoT",   iot_cap_range),
                            ("Edge",  edge_cap_range),
                            ("Cloud", cloud_cap_range)]:
        lo_abs = max(1, round(lo * M))
        hi_abs = max(lo_abs + 1, round(hi * M))
        hi_abs = min(hi_abs, M + 1)
        realised_lo = lo_abs / M
        realised_hi = (hi_abs - 1) / M
        note = ""
        if abs(realised_lo - lo) > 0.02 or abs(realised_hi - hi) > 0.02:
            note = (f"   <- realised {realised_lo*100:.0f}%–"
                    f"{realised_hi*100:.0f}%, differs from request")
        print(f"  {tier:<6}: [{lo_abs}, {hi_abs - 1}]  "
              f"({lo*100:.0f}%–{hi*100:.0f}% of M={M}){note}")
    print()


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    dcfg = cfg["data"]
    ecfg = cfg["environment"]
    N, M = ecfg["n_nodes"], ecfg["n_metrics"]

    iot_f, edge_f      = tier_fracs(ecfg)
    iot_c, edge_c, cl_c = cap_ranges(ecfg)

    # Show the realised topology and capacity ranges for this (N, M)
    print_capacity_summary(N, M, iot_c, edge_c, cl_c, iot_f, edge_f)

    print("[data] Generating training episodes...")
    train_eps = build_episode_pool(
        n_episodes           = dcfg["n_train_episodes"],
        n_nodes              = N,
        n_metrics            = M,
        episode_length       = ecfg["episode_length"],
        anomaly_density      = dcfg["anomaly_density"],
        causal_density       = dcfg["causal_density"],
        base_seed            = dcfg["seed"],
        anomaly_baseline_pct = ecfg["anomaly_baseline_pct"],
        causal_baseline_pct  = ecfg["causal_baseline_pct"],
        **topology_kwargs(ecfg),
    )

    print("[data] Generating evaluation episodes...")
    eval_eps = build_episode_pool(
        n_episodes           = dcfg["n_eval_episodes"],
        n_nodes              = N,
        n_metrics            = M,
        episode_length       = ecfg["episode_length"],
        anomaly_density      = dcfg["anomaly_density"] * 1.2,
        causal_density       = dcfg["causal_density"],
        base_seed            = dcfg["seed"] + 10000,
        anomaly_baseline_pct = ecfg["anomaly_baseline_pct"],
        causal_baseline_pct  = ecfg["causal_baseline_pct"],
        **topology_kwargs(ecfg),
    )

    # Save to N-nodes-M-metrics subdirectory
    data_dir = cfg["paths"]["data_dir"]
    save_dataset(train_eps, os.path.join(data_dir, "train_episodes.pkl"))
    save_dataset(eval_eps,  os.path.join(data_dir, "eval_episodes.pkl"))

    # Quick stats on first episode
    ep = train_eps[0]
    caps = ep["node_capacities"]
    print(f"\n[data] Sample episode stats (N={N}, M={M}):")
    print(f"  Timesteps         : {ep['T']}")
    print(f"  Anomaly density   : {ep['anomaly_labels'].mean():.3f}")
    print(f"  Anomaly baseline  : {ep['b_a']:.4f}")
    print(f"  Causal baseline   : {ep['b_c']:.4f}")
    print(f"  Node capacities   : {caps.astype(int).tolist()}")
    print(f"  Cap as % of M     : {[f'{100*c/M:.0f}%' for c in caps]}")
    print(f"  Node costs        : {ep['node_costs'].tolist()}")
    print(f"  Total budget      : {int(caps.sum())} / {N*M} "
          f"({100*caps.sum()/(N*M):.1f}% of all metrics)")