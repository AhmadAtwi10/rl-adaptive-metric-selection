"""
monitoring_env_masked.py
========================
Alternative Gymnasium environment with HARD capacity masking.

Key difference from monitoring_env.py:
  - Capacity is enforced HARD: the agent cannot select more than MC_n
    metrics per node. Selections exceeding the budget are masked to 0.
  - No capacity penalty term (r_cap = 0 always)
  - No idle penalty (masking guarantees at least 0 selections)
  - Simplified reward:
      r_t = α · r_imp(t) − λ · r_cost(t) − δ · r_switch(t)

Learning signal:
  The agent learns purely from importance vs cost:
    - High anomaly score → high r_imp → SELECT
    - Low anomaly score  → low r_imp, but still pays cost → DON'T SELECT
  This is a much simpler learning problem than the soft-constraint version.

Masking strategy (per node):
  At each window, the agent outputs a probability p_{n,m} for each metric.
  The top-MC_n metrics by probability are selected (hard TopK mask).
  This means the action is always feasible by construction.

State  : R^{N × M × 5}  per (node, metric):
           [anomaly_score, causal_score, placement_history,
            available_capacity (always 1.0 here), node_cost]

Action : logits ∈ R^{N × M}  — ranked and masked to top-MC_n per node
         Final binary selection always respects capacity

Reward : α·r_imp − λ·r_cost − δ·r_switch
         (no r_cap — masking makes overflow impossible)
         (no r_idle — masking always selects exactly MC_n metrics)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Dict, Any


class MonitoringEnvMasked(gym.Env):
    """
    Hard-masked monitoring environment.

    The agent outputs a score (logit) for each metric per node.
    The top-MC_n metrics by score are selected — hard capacity enforcement.

    Parameters
    ----------
    episode_pool         : list of episode dicts
    reward_weights       : dict with keys alpha, lambda_, delta
                           (beta and eta not used — masking makes them irrelevant)
    per_metric_cost      : float, fallback flat cost if node_costs not in episode
    anomaly_baseline_pct : int, percentile for b_a
    causal_baseline_pct  : int, percentile for b_c
    randomize_episodes   : bool
    window_size          : int W, timesteps per decision window
    window_agg           : 'max' or 'mean' for score aggregation
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        episode_pool: list,
        reward_weights: Optional[Dict[str, float]] = None,
        per_metric_cost: float = 0.01,
        anomaly_baseline_pct: int = 80,
        causal_baseline_pct: int = 100,
        randomize_episodes: bool = True,
        window_size: int = 10,
        window_agg: str = "max",
    ):
        super().__init__()

        self.episode_pool        = episode_pool
        self.per_metric_cost     = per_metric_cost
        self.anomaly_baseline_pct= anomaly_baseline_pct
        self.causal_baseline_pct = causal_baseline_pct
        self.randomize_episodes  = randomize_episodes
        self.window_size         = max(1, window_size)
        self.window_agg          = window_agg

        # Reward weights — no beta (capacity), no eta (idle)
        rw = reward_weights or {}
        self.alpha   = rw.get("alpha",   5.0)
        self.lambda_ = rw.get("lambda_", 0.1)
        self.delta   = rw.get("delta",   0.1)

        # Dims from first episode
        ep0    = episode_pool[0]
        self.N = ep0["n_nodes"]
        self.M = ep0["n_metrics"]
        self.T = ep0["T"]

        self.n_windows = self.T // self.window_size

        # Observation: N*M*5 (same features as monitoring_env.py)
        obs_dim = self.N * self.M * 5
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Action: continuous logits R^{N*M}
        # The agent outputs a score per metric; we apply hard TopK mask per node.
        # Using Box instead of MultiBinary so the policy outputs real-valued scores.
        self.action_space = spaces.Box(
            low=-10.0, high=10.0, shape=(self.N * self.M,), dtype=np.float32
        )

        # Internal state
        self._anomaly_scores = None
        self._causal_scores  = None
        self._labels         = None
        self._capacities     = None
        self._node_costs     = None
        self._b_a  = 0.0
        self._b_c  = 0.0
        self._w    = 0
        self._prev_action = None   # [N, M] binary — last masked selection
        self._ep_idx = 0
        self._rng    = np.random.default_rng(0)

    # ----------------------------------------------------------
    # Masking: convert logits → binary selection respecting MC_n
    # ----------------------------------------------------------
    def _apply_mask(self, logits: np.ndarray) -> np.ndarray:
        """
        For each node n, select the top-MC_n metrics by logit value.
        Returns binary [N, M] selection matrix with exactly MC_n ones per node.

        Parameters
        ----------
        logits : np.ndarray [N*M]  real-valued scores from the policy

        Returns
        -------
        action : np.ndarray [N, M]  binary, exactly cap[n] ones per row
        """
        logits_2d = logits.reshape(self.N, self.M)
        action    = np.zeros((self.N, self.M), dtype=np.float32)

        for n in range(self.N):
            cap   = int(self._capacities[n])
            k     = min(cap, self.M)
            # Select top-k indices by logit (highest scores → selected)
            top_k = np.argsort(logits_2d[n])[::-1][:k]
            action[n, top_k] = 1.0

        return action

    # ----------------------------------------------------------
    # Reset
    # ----------------------------------------------------------
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        if self.randomize_episodes:
            self._ep_idx = int(self._rng.integers(0, len(self.episode_pool)))
        else:
            self._ep_idx = (self._ep_idx + 1) % len(self.episode_pool)

        ep = self.episode_pool[self._ep_idx]
        self._anomaly_scores = ep["anomaly_scores"]
        self._causal_scores  = ep["causal_scores"]
        self._labels         = ep["anomaly_labels"]
        self._capacities     = ep["node_capacities"]

        if "node_costs" in ep:
            self._node_costs = ep["node_costs"]
        else:
            self._node_costs = np.full(self.N, self.per_metric_cost, dtype=np.float32)

        self._b_a = float(ep.get("b_a", np.percentile(
            self._anomaly_scores, self.anomaly_baseline_pct)))
        self._b_c = float(ep.get("b_c", np.percentile(
            self._causal_scores,  self.causal_baseline_pct)))

        self._w           = 0
        # prev_action: zeros initially (nothing was selected)
        self._prev_action = np.zeros((self.N, self.M), dtype=np.float32)

        return self._get_obs(), {}

    # ----------------------------------------------------------
    # Step
    # ----------------------------------------------------------
    def step(self, action: np.ndarray):
        """
        action : np.ndarray [N*M]  — real-valued logits from policy
        Applies hard TopK mask → binary selection respecting MC_n.
        Accumulates reward over the window.
        """
        # Apply hard capacity mask
        a = self._apply_mask(action)   # [N, M] binary, exactly cap[n] per row

        t_start  = self._w * self.window_size
        t_end    = min(t_start + self.window_size, self.T)
        W_actual = t_end - t_start

        total_reward = 0.0
        total_imp    = 0.0
        total_cost   = 0.0
        total_switch = 0.0
        coverages    = []

        C = self._causal_scores

        for t in range(t_start, t_end):
            A_t = self._anomaly_scores[t]
            rew, rinfo = self._compute_reward(a, A_t, C)
            total_reward += rew
            total_imp    += rinfo["r_imp"]
            total_cost   += rinfo["r_cost"]
            total_switch += rinfo["r_switch"]

            if self._labels is not None:
                L_t    = self._labels[t]
                n_anom = L_t.sum()
                if n_anom > 0:
                    coverages.append(float((a * L_t).sum() / n_anom))
                else:
                    coverages.append(1.0)

        self._prev_action = a.copy()
        self._w += 1

        terminated = self._w >= self.n_windows
        truncated  = False

        reward_per_step = total_reward / W_actual

        info = {
            "r_imp":          total_imp    / W_actual,
            "r_cost":         total_cost   / W_actual,
            "r_switch":       total_switch / W_actual,
            "r_cap":          0.0,   # always 0 — masking guarantees feasibility
            "reward":         reward_per_step,
            "selected_count": int(a.sum()),
            "window":         self._w,
            "t_start":        t_start,
            "t_end":          t_end,
            "anomaly_coverage": float(np.mean(coverages)) if coverages else 1.0,
        }

        return self._get_obs(), float(reward_per_step), terminated, truncated, info

    # ----------------------------------------------------------
    # Observation
    # ----------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        w_clamped = min(self._w, self.n_windows - 1)
        t_start   = w_clamped * self.window_size
        t_end     = max(min(t_start + self.window_size, self.T), t_start + 1)

        window_scores = self._anomaly_scores[t_start:t_end]

        if self.window_agg == "max":
            A_agg = window_scores.max(axis=0)
        else:
            A_agg = window_scores.mean(axis=0)

        C = self._causal_scores
        H = self._prev_action

        # Capacity feature: always 1.0 (hard mask guarantees feasibility)
        # Still included so policy knows how many slots are available
        selected_per_node = self._prev_action.sum(axis=1)
        avail = np.clip(
            (self._capacities - selected_per_node) / (self._capacities + 1e-8),
            0.0, 1.0
        )
        cap_feature  = np.tile(avail[:, np.newaxis], (1, self.M))

        max_cost     = self._node_costs.max() + 1e-8
        cost_norm    = self._node_costs / max_cost
        cost_feature = np.tile(cost_norm[:, np.newaxis], (1, self.M))

        obs = np.stack([A_agg, C, H, cap_feature, cost_feature], axis=-1)
        return obs.reshape(-1).astype(np.float32)

    # ----------------------------------------------------------
    # Reward (no capacity or idle penalty)
    # ----------------------------------------------------------
    def _compute_reward(
        self,
        a: np.ndarray,   # [N, M] binary — already masked
        A: np.ndarray,   # [N, M] anomaly scores at timestep t
        C: np.ndarray,   # [N, M] causal scores
    ):
        # Importance: reward for selecting anomalous/causal metrics
        imp_a = np.maximum(0.0, A - self._b_a)
        imp_c = np.maximum(0.0, C - self._b_c)
        r_imp = float((a * (imp_a + imp_c)).sum())

        # Cost: pay per selected metric, higher for IoT nodes
        selected_per_node = a.sum(axis=1)
        r_cost = float((self._node_costs * selected_per_node).sum())

        # Switching: penalise changing unimportant metrics
        is_unimportant = ((A < self._b_a) & (C < self._b_c)).astype(np.float32)
        changes  = np.abs(a - self._prev_action) * is_unimportant
        r_switch = float(changes.sum())

        # No capacity penalty — masking makes overflow impossible
        # No idle penalty — TopK always selects exactly MC_n metrics
        reward = (
              self.alpha   * r_imp
            - self.lambda_ * r_cost
            - self.delta   * r_switch
        )

        return reward, {
            "r_imp": r_imp, "r_cost": r_cost, "r_switch": r_switch,
        }

    # ----------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------
    def get_selection(self, logits: np.ndarray) -> np.ndarray:
        """Public helper: apply mask and return binary [N, M] selection."""
        return self._apply_mask(logits)

    def get_episode_info(self) -> Dict[str, Any]:
        return {
            "ep_idx":      self._ep_idx,
            "n_nodes":     self.N,
            "n_metrics":   self.M,
            "T":           self.T,
            "window_size": self.window_size,
            "n_windows":   self.n_windows,
            "capacities":  self._capacities.tolist(),
            "node_costs":  self._node_costs.tolist(),
            "b_a":         self._b_a,
            "b_c":         self._b_c,
        }

    def render(self):
        pass