"""
monitoring_env.py
=================
Gymnasium environment for the adaptive metric selection MDP.
 
State  : R^{N × M × 6}  per (node, metric):
           [anomaly_score, causal_score, placement_history,
            cap_remaining, node_cost, cap_absolute]
 
Action : {0,1}^{N × M}  — multi-binary selection matrix
         Held FIXED for the entire window of W timesteps
 
Reward : α·r_imp − β·r_cap − λ·r_cost − δ·r_switch − η·r_idle
         Accumulated per-timestep over the window, then normalised by W
 
Coverage mode (coverage_mode argument)
---------------------------------------
  "window"      Correct, consistent with compute_metrics.
                L_w = labels[t_s:t_e].max(axis=0) — any anomaly in window.
                Training eval cov (~0.77) matches evaluate_agent.py exactly.
                Use this for final evaluation and paper results.
 
  "per_timestep" Legacy / diagnostic.
                 Averages coverage over every individual timestep.
                 Anomaly-free timesteps contribute 1.0, inflating the number.
                 Training eval cov appears higher (~0.87).
                 Useful to track learning progress during long runs where you
                 want a smoother, faster-updating signal.
 
Set coverage_mode in config.yaml:
    environment:
      coverage_mode: window        # recommended
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Dict, Any, List


class MonitoringEnv(gym.Env):
    """
    Monitoring Metric Selection Environment with configurable window size.

    Parameters
    ----------
    episode_pool         : list of dataset dicts
    reward_weights       : dict with keys alpha, beta, lambda_, delta, eta
    per_metric_cost      : float, cost per selected metric per timestep
    anomaly_baseline_pct : int, percentile to compute b_a
    causal_baseline_pct  : int, percentile to compute b_c
    randomize_episodes   : bool
    window_size          : int, W — number of timesteps per decision window.
                           W=1 → per-timestep (original behaviour)
                           W>1 → agent selects once, held for W timesteps
    window_agg           : str, how to aggregate anomaly scores over window
                           'mean' — captures sustained anomalies
                           'max'  — captures brief spikes (recommended)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        episode_pool: list,
        reward_weights: Optional[Dict[str, float]] = None,
        per_metric_cost: float = 0.01,
        anomaly_baseline_pct: int = 30,
        causal_baseline_pct: int = 30,
        randomize_episodes: bool = True,
        window_size: int = 1,
        window_agg: str = "max",
        coverage_mode: str = "window", # "window" or "per_timestep"
    ):
        super().__init__()

        self.episode_pool        = episode_pool
        self.per_metric_cost     = per_metric_cost
        self.anomaly_baseline_pct= anomaly_baseline_pct
        self.causal_baseline_pct = causal_baseline_pct
        self.randomize_episodes  = randomize_episodes
        self.window_size         = max(1, window_size)
        self.window_agg          = window_agg   # 'mean' or 'max'
    
        if coverage_mode not in ("window", "per_timestep"):
            raise ValueError(
                f"coverage_mode must be 'window' or 'per_timestep', got '{coverage_mode}'"
            )
        self.coverage_mode = coverage_mode

        # Reward weights
        rw = reward_weights or {}
        self.alpha   = rw.get("alpha",   5.0)
        self.beta    = rw.get("beta",    3.0)
        self.lambda_ = rw.get("lambda_", 0.1)
        self.delta   = rw.get("delta",   0.1)
        self.eta     = rw.get("eta",     2.0)

        # Infer dims from first episode
        ep0      = episode_pool[0]
        self.N   = ep0["n_nodes"]
        self.M   = ep0["n_metrics"]
        self.T   = ep0["T"]

        # Number of decision windows per episode
        self.n_windows = self.T // self.window_size   # drop remainder

        # Spaces — 5/6 features per (node, metric) whether or not we include the absolute capacity feature
        # [anomaly_score, causal_score, placement_history,
        #  available_capacity, node_cost, (optional) absolute_capacity]
        obs_dim = self.N * self.M * 5  # if we are not including the absolute capacity feature
        # obs_dim = self.N * self.M * 6

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.MultiBinary(self.N * self.M)

        # Internal state
        self._anomaly_scores = None   # [T, N, M]
        self._causal_scores  = None   # [N, M]
        self._labels         = None   # [T, N, M]
        self._capacities     = None   # [N]
        self._b_a  = 0.0
        self._b_c  = 0.0
        self._w    = 0                # current window index
        self._prev_action = None      # [N, M]
        self._ep_idx = 0
        self._rng    = np.random.default_rng(0)

    # ----------------------------------------------------------
    # Reset
    # ----------------------------------------------------------
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
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
        self._node_costs     = ep.get(
            "node_costs",
            np.full(self.N, self.per_metric_cost, dtype=np.float32),
        )
        self._b_a = float(ep.get("b_a", np.percentile(
            self._anomaly_scores, self.anomaly_baseline_pct)))
        self._b_c = float(ep.get("b_c", np.percentile(
            self._causal_scores, self.causal_baseline_pct)))
 
        self._w           = 0
        self._prev_action = np.zeros((self.N, self.M), dtype=np.float32)
 
        return self._get_obs(), {}
 
    # ----------------------------------------------------------
    # Step
    # ----------------------------------------------------------
 
    def step(self, action: np.ndarray):
        """
        Apply the agent's window selection.
 
        Reward  — accumulated per-timestep, normalised by W.
        Coverage — controlled by self.coverage_mode:
 
          "window"       (recommended, matches compute_metrics)
            L_w = labels[t_s:t_e].max(axis=0)
            coverage = (a * L_w).sum() / L_w.sum()
            One score per window. A metric anomalous for 3/10 steps
            counts once — the agent either covered it or not.
 
          "per_timestep" (legacy, inflated)
            Averages coverage over each t in [t_s, t_e).
            Anomaly-free timesteps contribute 1.0, pushing the average up.
            Use only for diagnostic comparison with old checkpoints.
        """
        a = action.reshape(self.N, self.M).astype(np.float32)
 
        t_start  = self._w * self.window_size
        t_end    = min(t_start + self.window_size, self.T)
        W_actual = t_end - t_start
 
        C = self._causal_scores
 
        # ── Reward (always per-timestep) ──────────────────────────
        total_reward = total_imp = total_cap = 0.0
        total_cost   = total_switch = total_idle = 0.0
 
        for t in range(t_start, t_end):
            A_t = self._anomaly_scores[t]
            rew, rinfo = self._compute_reward(a, A_t, C)
            total_reward += rew
            total_imp    += rinfo["r_imp"]
            total_cap    += rinfo["r_cap"]
            total_cost   += rinfo["r_cost"]
            total_switch += rinfo["r_switch"]
            total_idle   += rinfo["r_idle"]
 
        # ── Coverage ──────────────────────────────────────────────
        if self._labels is not None:
            if self.coverage_mode == "window":
                # Correct: one binary decision per window
                L_w    = self._labels[t_start:t_end].max(axis=0)   # [N,M]
                n_anom = int(L_w.sum())
                window_coverage = (
                    float((a * L_w).sum() / n_anom) if n_anom > 0 else 1.0
                )
            else:
                # per_timestep: legacy, inflated by anomaly-free timesteps
                coverages = []
                for t in range(t_start, t_end):
                    L_t   = self._labels[t]
                    n_an  = int(L_t.sum())
                    coverages.append(
                        float((a * L_t).sum() / n_an) if n_an > 0 else 1.0
                    )
                window_coverage = float(np.mean(coverages))
        else:
            window_coverage = 1.0
 
        # ── Advance ───────────────────────────────────────────────
        self._prev_action = a.copy()
        self._w += 1
        terminated = self._w >= self.n_windows
 
        reward_per_step = total_reward / W_actual
 
        info = {
            "r_imp":            total_imp    / W_actual,
            "r_cap":            total_cap    / W_actual,
            "r_cost":           total_cost   / W_actual,
            "r_switch":         total_switch / W_actual,
            "r_idle":           total_idle   / W_actual,
            "reward":           reward_per_step,
            "selected_count":   int(a.sum()),
            "window":           self._w,
            "t_start":          t_start,
            "t_end":            t_end,
            "anomaly_coverage": window_coverage,
            "coverage_mode":    self.coverage_mode,
        }
 
        return self._get_obs(), float(reward_per_step), terminated, False, info
 
    # ----------------------------------------------------------
    # Observation builder
    # ----------------------------------------------------------
 
    def _get_obs(self) -> np.ndarray:
        """
        5 or 6-feature state for the UPCOMING window.
 
          0  anomaly_score    window-max (or mean) of A  [0,1]
          1  causal_score     static C[n,m]               [0,1]
          2  history          prev_action[n,m]  binary    {0,1}
          3  cap_remaining    (cap_n - selected) / cap_n  [0,1]  dynamic
          4  node_cost        normalised cost (1=IoT)      [0,1]  static
          5  cap_absolute     cap_n / M                   (0,1]  static either commented or kept
        """
        w_clamped = min(self._w, self.n_windows - 1)
        t_start   = w_clamped * self.window_size
        t_end     = max(min(t_start + self.window_size, self.T), t_start + 1)
 
        window_scores = self._anomaly_scores[t_start:t_end]
        A_agg = (window_scores.max(axis=0) if self.window_agg == "max"
                 else window_scores.mean(axis=0))
 
        C = self._causal_scores
        H = self._prev_action
 
        # Feature 3: remaining capacity (dynamic)
        sel_per_node = self._prev_action.sum(axis=1)
        avail = np.clip(
            (self._capacities - sel_per_node) / (self._capacities + 1e-8),
            0.0, 1.0,
        )
        cap_remaining = np.tile(avail[:, np.newaxis], (1, self.M))
 
        # Feature 4: normalised node cost (static)
        cost_norm    = self._node_costs / (self._node_costs.max() + 1e-8)
        cost_feature = np.tile(cost_norm[:, np.newaxis], (1, self.M))
 
        # Feature 5: absolute capacity fraction (static)
        #cap_abs         = np.clip(self._capacities / self.M, 0.0, 1.0)
        #cap_abs_feature = np.tile(cap_abs[:, np.newaxis], (1, self.M))
 
        obs = np.stack(
            [A_agg, C, H, cap_remaining, cost_feature],
            axis=-1,
        )   # [N, M, 5]
        
        # with feature 5 (absolute capacity fraction) version
        # obs = np.stack(
        #     [A_agg, C, H, cap_remaining, cost_feature, cap_abs_feature],
        #     axis=-1,
        # )   # [N, M, 6]
        return obs.reshape(-1).astype(np.float32)
 
    # ----------------------------------------------------------
    # Reward
    # ----------------------------------------------------------
 
    def _compute_reward(self, a, A, C):
        imp_a = np.maximum(0.0, A - self._b_a)
        imp_c = np.maximum(0.0, C - self._b_c)
        r_imp = float((a * (imp_a + imp_c)).sum())
 
        sel_per_node = a.sum(axis=1)
        overflow = np.maximum(0.0, sel_per_node - self._capacities)
        r_cap    = float(overflow.sum())
        r_cost   = float((self._node_costs * sel_per_node).sum())
 
        is_unimp = ((A < self._b_a) & (C < self._b_c)).astype(np.float32)
        r_switch = float((np.abs(a - self._prev_action) * is_unimp).sum())
        r_idle   = 1.0 if a.sum() == 0 else 0.0
 
        reward = (self.alpha * r_imp - self.beta * r_cap
                  - self.lambda_ * r_cost - self.delta * r_switch
                  - self.eta * r_idle)
        return reward, {
            "r_imp": r_imp, "r_cap": r_cap, "r_cost": r_cost,
            "r_switch": r_switch, "r_idle": r_idle,
        }
 
    # ----------------------------------------------------------
    # Properties
    # ----------------------------------------------------------
 
    @property
    def current_timestep(self) -> int:
        return self._w * self.window_size
 
    @property
    def n_decisions_per_episode(self) -> int:
        return self.n_windows
 
    def get_episode_info(self) -> dict:
        return {
            "ep_idx":        self._ep_idx,
            "n_nodes":       self.N,
            "n_metrics":     self.M,
            "T":             self.T,
            "window_size":   self.window_size,
            "n_windows":     self.n_windows,
            "capacities":    self._capacities.tolist(),
            "b_a":           self._b_a,
            "b_c":           self._b_c,
            "coverage_mode": self.coverage_mode,
        }
 
    def render(self):
        pass
 
 
# ------------------------------------------------------------------
# Helper
# ------------------------------------------------------------------
 
def make_env(episode_pool: list, reward_weights: dict, cfg: dict, seed: int = 0):
    def _init():
        env = MonitoringEnv(
            episode_pool         = episode_pool,
            reward_weights       = reward_weights,
            per_metric_cost      = cfg["environment"]["per_metric_cost"],
            anomaly_baseline_pct = cfg["environment"]["anomaly_baseline_pct"],
            causal_baseline_pct  = cfg["environment"]["causal_baseline_pct"],
            randomize_episodes   = True,
            window_size          = cfg["environment"].get("window_size", 1),
            window_agg           = cfg["environment"].get("window_agg", "max"),
            coverage_mode        = cfg["environment"].get("coverage_mode", "window"),
        )
        env.reset(seed=seed)
        return env
    return _init