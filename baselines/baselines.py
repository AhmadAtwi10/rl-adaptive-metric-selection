"""
baselines.py
============
Baseline monitoring strategies for comparison with the RL agent.

  1. MonitorEverything     – selects all metrics at every timestep
  2. RandomFeasible        – random selection respecting per-node capacity
  3. TopKAnomalyCausal     – greedy ranking by combined anomaly+causal score
"""

import numpy as np
from typing import Dict, Any


class BaseBaseline:
    """Common interface for all baselines."""

    def reset(self, episode: dict):
        """Called at the start of each episode with the data dict."""
        self.N = episode["n_nodes"]
        self.M = episode["n_metrics"]
        self.T = episode["T"]
        self.anomaly_scores = episode["anomaly_scores"]   # [T, N, M]
        self.causal_scores  = episode["causal_scores"]    # [N, M]
        self.labels         = episode["anomaly_labels"]   # [T, N, M]
        self.capacities     = episode["node_capacities"]  # [N]
        self.b_a = episode.get("b_a", 0.0)
        self.b_c = episode.get("b_c", 0.0)
        self.t   = 0
        self.prev_action = np.zeros((self.N, self.M), dtype=np.float32)

    def select(self, t: int) -> np.ndarray:
        """Returns binary [N, M] selection matrix."""
        raise NotImplementedError

    def run_episode(
        self,
        episode: dict,
        reward_weights: dict,
        per_metric_cost: float = 0.01,
    ) -> Dict[str, Any]:
        """
        Run one full episode and return performance metrics.
        """
        self.reset(episode)
        alpha   = reward_weights.get("alpha",   5.0)
        beta    = reward_weights.get("beta",    1.0)
        lambda_ = reward_weights.get("lambda_", 1.0)
        delta   = reward_weights.get("delta",   0.1)
        eta     = reward_weights.get("eta",     2.0)

        total_reward    = 0.0
        total_selected  = 0
        total_imp       = 0.0
        total_cost      = 0.0
        coverages       = []

        for t in range(self.T):
            a = self.select(t)   # [N, M]
            A = self.anomaly_scores[t]
            C = self.causal_scores

            # Reward components
            imp_a = np.maximum(0.0, A - self.b_a)
            imp_c = np.maximum(0.0, C - self.b_c)
            r_imp = float((a * (imp_a + imp_c)).sum())

            overflow  = np.maximum(0.0, a.sum(axis=1) - self.capacities)
            r_cap     = float(overflow.sum())
            r_cost    = float(a.sum()) * per_metric_cost

            is_unimp   = ((A < self.b_a) & (C < self.b_c)).astype(np.float32)
            r_switch   = float((np.abs(a - self.prev_action) * is_unimp).sum())
            r_idle     = 1.0 if a.sum() == 0 else 0.0

            reward = (alpha * r_imp - beta * r_cap - lambda_ * r_cost
                      - delta * r_switch - eta * r_idle)

            total_reward   += reward
            total_selected += int(a.sum())
            total_imp      += r_imp
            total_cost     += float(a.sum()) * per_metric_cost

            # Anomaly coverage
            labels_t = self.labels[t]
            n_anom   = labels_t.sum()
            if n_anom > 0:
                coverages.append(float((a * labels_t).sum() / n_anom))
            else:
                coverages.append(1.0)

            self.prev_action = a.copy()

        return {
            "total_reward":    total_reward,
            "mean_selected":   total_selected / self.T,
            "mean_imp_score":  total_imp      / self.T,
            "mean_cost":       total_cost     / self.T,
            "anomaly_coverage":float(np.mean(coverages)),
        }


# ------------------------------------------------------------------
# 1. Monitor Everything
# ------------------------------------------------------------------

class MonitorEverything(BaseBaseline):
    """Selects all metrics on all nodes at every timestep."""

    def select(self, t: int) -> np.ndarray:
        return np.ones((self.N, self.M), dtype=np.float32)


# ------------------------------------------------------------------
# 2. Random Feasible
# ------------------------------------------------------------------

class RandomFeasible(BaseBaseline):
    """
    Randomly selects up to MC_n metrics per node (uniform, without replacement).
    """

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def select(self, t: int) -> np.ndarray:
        a = np.zeros((self.N, self.M), dtype=np.float32)
        for n in range(self.N):
            cap = int(self.capacities[n])
            k   = min(cap, self.M)
            chosen = self.rng.choice(self.M, size=k, replace=False)
            a[n, chosen] = 1.0
        return a


# ------------------------------------------------------------------
# 3. TopK Anomaly-Causal
# ------------------------------------------------------------------

class TopKAnomalyCausal(BaseBaseline):
    """
    At each timestep, ranks metrics on each node by:
        combined_score = 0.65*A_{n,m}(t) + 0.35*C_{n,m}
    Then selects the top MC_n metrics per node.
    This uses the same informative priors as the RL agent but without learning.
    """

    def select(self, t: int) -> np.ndarray:
        A = self.anomaly_scores[t]   # [N, M]
        C = self.causal_scores       # [N, M]
        combined = 0.60*A + 0.4*C           # [N, M]

        a = np.zeros((self.N, self.M), dtype=np.float32)
        for n in range(self.N):
            cap  = int(self.capacities[n])
            k    = min(cap, self.M)
            topk = np.argsort(combined[n])[::-1][:k]
            a[n, topk] = 1.0
        return a


# ------------------------------------------------------------------
# Batch evaluation helper
# ------------------------------------------------------------------

def evaluate_baselines(
    episodes: list,
    reward_weights: dict,
    per_metric_cost: float = 0.01,
    seed: int = 0,
) -> Dict[str, Dict[str, float]]:
    """
    Run all baselines over all episodes and return aggregate stats.

    Returns
    -------
    dict mapping baseline_name → {metric: mean_value}
    """
    baselines = {
        "Monitor-Everything":    MonitorEverything(),
        "Random-Feasible":       RandomFeasible(seed=seed),
        "TopK-Anomaly-Causal":   TopKAnomalyCausal(),
    }

    results = {name: [] for name in baselines}

    for ep in episodes:
        for name, bl in baselines.items():
            stats = bl.run_episode(ep, reward_weights, per_metric_cost)
            results[name].append(stats)

    # Aggregate
    agg = {}
    for name, ep_stats in results.items():
        keys = ep_stats[0].keys()
        agg[name] = {k: float(np.mean([s[k] for s in ep_stats])) for k in keys}

    return agg
