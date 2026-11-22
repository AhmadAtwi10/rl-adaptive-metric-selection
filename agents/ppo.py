"""
ppo.py
======
Proximal Policy Optimization trainer for the adaptive monitoring environment.

Implements:
  - GAE (Generalized Advantage Estimation)
  - Clipped surrogate objective
  - Value function clipping
  - Entropy regularization
  - Gradient clipping
  - Periodic evaluation with model checkpointing

FIX: save_model() now saves a FULL checkpoint (policy weights + optimizer
state + global step + best eval reward + Welford running stats) so that
training can be resumed exactly with resume_training.py.
load_model() is updated to match, with graceful fallback for legacy
checkpoints that only contain policy weights.
"""

import os
import time
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Optional, Tuple

from agents.policy import ActorCritic
from envs.monitoring_env import MonitoringEnv

# TensorBoard — optional but highly recommended
try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_AVAILABLE = True
except ImportError:
    _TB_AVAILABLE = False
    print("[ppo] WARNING: tensorboard not installed. "
          "Run: pip install tensorboard\n"
          "         CSV logging will still work.")


class RolloutBuffer:
    """Stores rollout transitions for one PPO update."""

    def __init__(
        self,
        n_steps: int,
        n_envs: int,
        obs_dim: int,
        action_dim: int,
        device: torch.device,
    ):
        self.n_steps    = n_steps
        self.n_envs     = n_envs
        self.obs_dim    = obs_dim
        self.action_dim = action_dim
        self.device     = device
        self.ptr        = 0
        self._alloc()

    def _alloc(self):
        B = self.n_steps * self.n_envs
        self.obs        = torch.zeros(B, self.obs_dim,    device=self.device)
        self.actions    = torch.zeros(B, self.action_dim, device=self.device)
        self.log_probs  = torch.zeros(B,                  device=self.device)
        self.rewards    = torch.zeros(B,                  device=self.device)
        self.values     = torch.zeros(B,                  device=self.device)
        self.dones      = torch.zeros(B,                  device=self.device)
        self.advantages = torch.zeros(B,                  device=self.device)
        self.returns    = torch.zeros(B,                  device=self.device)

    def add(self, obs, actions, log_probs, rewards, values, dones):
        """Add one step from all envs."""
        idx = slice(self.ptr * self.n_envs, (self.ptr + 1) * self.n_envs)
        self.obs[idx]       = obs
        self.actions[idx]   = actions
        self.log_probs[idx] = log_probs
        self.rewards[idx]   = rewards
        self.values[idx]    = values
        self.dones[idx]     = dones
        self.ptr += 1

    def compute_returns_and_advantages(
        self,
        last_values: torch.Tensor,
        last_dones:  torch.Tensor,
        gamma: float,
        gae_lambda: float,
    ):
        """Compute GAE advantages and discounted returns in-place."""
        last_gae = torch.zeros(self.n_envs, device=self.device)

        for t in reversed(range(self.n_steps)):
            idx      = slice(t * self.n_envs, (t + 1) * self.n_envs)
            next_idx = slice((t + 1) * self.n_envs, (t + 2) * self.n_envs)

            if t == self.n_steps - 1:
                next_non_terminal = 1.0 - last_dones.float()
                next_values       = last_values
            else:
                next_non_terminal = 1.0 - self.dones[next_idx]
                next_values       = self.values[next_idx]

            delta    = (self.rewards[idx]
                        + gamma * next_values * next_non_terminal
                        - self.values[idx])
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae

            self.advantages[idx] = last_gae
            self.returns[idx]    = last_gae + self.values[idx]

    def reset(self):
        self.ptr = 0

    def get_dataloader(self, batch_size: int):
        ds = TensorDataset(
            self.obs, self.actions, self.log_probs,
            self.advantages, self.returns, self.values
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=True)


# ------------------------------------------------------------------
# PPO Trainer
# ------------------------------------------------------------------

class PPOTrainer:

    def __init__(
        self,
        envs: List[MonitoringEnv],
        eval_envs: List[MonitoringEnv],
        policy: ActorCritic,
        cfg: dict,
        device: torch.device,
        results_dir: str = "results",
    ):
        self.envs        = envs
        self.eval_envs   = eval_envs
        self.policy      = policy.to(device)
        self.device      = device
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

        pcfg = cfg["ppo"]
        self.total_timesteps = pcfg["total_timesteps"]
        self.n_envs          = pcfg["n_envs"]
        self.n_steps         = pcfg["n_steps"]
        self.batch_size      = pcfg["batch_size"]
        self.n_epochs        = pcfg["n_epochs"]
        self.gamma           = pcfg["gamma"]
        self.gae_lambda      = pcfg["gae_lambda"]
        self.clip_range      = pcfg["clip_range"]
        self.ent_coef        = pcfg["ent_coef"]
        self.vf_coef         = pcfg["vf_coef"]
        self.max_grad_norm   = pcfg["max_grad_norm"]
        self.eval_freq       = pcfg["eval_freq"]
        self.eval_episodes   = pcfg["eval_episodes"]
        self.lr              = pcfg["learning_rate"]


        # best_model.pt  — policy weights only (lightweight, for inference)
        # checkpoint.pt  — full trainer state (for resuming training)
        self.model_path      = os.path.join(results_dir, cfg["paths"]["model_name"])
        self.checkpoint_path = os.path.join(results_dir, "checkpoint.pt")
        self.log_path        = os.path.join(results_dir, cfg["paths"]["log_name"])

        self.optimizer = optim.Adam(policy.parameters(), lr=self.lr, eps=1e-5)
        self.obs_dim    = policy.obs_dim
        self.action_dim = policy.action_dim

        self.buffer = RolloutBuffer(
            self.n_steps, self.n_envs,
            self.obs_dim, self.action_dim, device
        )

        self._global_step      = 0
        self._best_eval_reward = -np.inf
        self._log_rows: List[dict] = []
        
        #added
        self.total_metrics = self.action_dim


        # Welford running reward normalizer — prevents value loss explosion.
        # Returns can reach ~50,000/episode without normalization, causing
        # value loss (MSE) of ~2,000,000 that drowns the policy gradient.
        self._reward_running_mean  = 0.0
        self._reward_running_var   = 1.0
        self._reward_running_count = 0

        # TensorBoard writer
        tb_dir = os.path.join(results_dir, "tensorboard")
        if _TB_AVAILABLE:
            self.writer = SummaryWriter(log_dir=tb_dir)
            print(f"[ppo] TensorBoard logs → {tb_dir}")
            print(f"[ppo] Launch with:  tensorboard --logdir {tb_dir}\n")
        else:
            self.writer = None

        # Initialize env states
        self._obs   = np.zeros((self.n_envs, self.obs_dim), dtype=np.float32)
        self._dones = np.zeros(self.n_envs, dtype=bool)
        for i, env in enumerate(self.envs):
            obs, _ = env.reset()
            self._obs[i] = obs

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self):
        print(f"\n{'='*60}")
        print(f"  PPO Training — {self.total_timesteps:,} timesteps")
        print(f"  Envs: {self.n_envs}  |  n_steps: {self.n_steps}")
        print(f"  Device: {self.device}")
        print(f"{'='*60}\n")

        n_updates = self.total_timesteps // (self.n_steps * self.n_envs)
        t_start   = time.time()

        for update in range(1, n_updates + 1):
            # ---------- Collect rollout ----------
            rollout_stats = self._collect_rollout()

            # ---------- PPO update ----------
            stats = self._ppo_update()
            self._global_step += self.n_steps * self.n_envs

            # ---------- Logging ----------
            if update % max(1, n_updates // 200) == 0:
                elapsed = time.time() - t_start
                fps = self._global_step / elapsed
                print(
                    f"[{update:5d}/{n_updates}] "
                    f"step={self._global_step:8,d}  "
                    f"fps={fps:5.0f}  "
                    f"loss={stats['loss']:.4f}  "
                    f"v_loss={stats['value_loss']:.4f}  "
                    f"ent={stats['entropy']:.3f}"
                )
                if self.writer:
                    self.writer.add_scalar("train/loss",        stats["loss"],        self._global_step)
                    self.writer.add_scalar("train/policy_loss",  stats["policy_loss"], self._global_step)
                    self.writer.add_scalar("train/value_loss",   stats["value_loss"],  self._global_step)
                    self.writer.add_scalar("train/entropy",      stats["entropy"],     self._global_step)
                    self.writer.add_scalar("train/clip_frac",    stats["clip_frac"],   self._global_step)
                    self.writer.add_scalar("train/fps",          fps,                  self._global_step)
                    # Rollout-level environment metrics (mirrors eval/ keys)
                    if "mean_selected" in rollout_stats:
                        self.writer.add_scalar("train/mean_selected",        rollout_stats["mean_selected"],        self._global_step)
                        self.writer.add_scalar("train/mean_selection_rate",  rollout_stats["mean_selection_rate"],  self._global_step)
                    if "mean_coverage" in rollout_stats:
                        self.writer.add_scalar("train/anomaly_coverage",     rollout_stats["mean_coverage"],        self._global_step)
                    if "mean_imp_score" in rollout_stats:
                        self.writer.add_scalar("train/mean_imp_score",       rollout_stats["mean_imp_score"],       self._global_step)
                    if "mean_collection_cost" in rollout_stats:
                        self.writer.add_scalar("train/mean_collection_cost", rollout_stats["mean_collection_cost"], self._global_step)
                    self.writer.add_scalar("train/mean_reward",              rollout_stats["mean_reward"],          self._global_step)

            # ---------- Evaluation ----------
            eval_interval = max(1, self.eval_freq // (self.n_steps * self.n_envs))
            if update % eval_interval == 0:
                eval_stats = self.evaluate(self.eval_episodes)
                self._log_rows.append({
                    "step":                  self._global_step,
                    "eval_reward":           eval_stats["mean_reward"],
                    "eval_selected":         eval_stats["mean_selected"],
                    "eval_selection_rate":   eval_stats["mean_selection_rate"],
                    "eval_collection_cost":  eval_stats["mean_collection_cost"],
                    "eval_coverage":         eval_stats.get("mean_coverage", float("nan")),
                    "eval_imp_score":        eval_stats["mean_imp_score"],
                    **stats,
                })
                print(
                    f"  >> EVAL  reward={eval_stats['mean_reward']:.3f}  "
                    f"selected={eval_stats['mean_selected']:.1f}"
                    f"({eval_stats['mean_selection_rate']*100:.1f}%)  "
                    f"cost={eval_stats['mean_collection_cost']:.4f}  "
                    f"cov={eval_stats.get('mean_coverage', float('nan')):.3f}  "
                    f"imp={eval_stats['mean_imp_score']:.3f}"
                )
                if self.writer:
                    self.writer.add_scalar("eval/mean_reward",          eval_stats["mean_reward"],          self._global_step)
                    self.writer.add_scalar("eval/std_reward",           eval_stats["std_reward"],           self._global_step)
                    self.writer.add_scalar("eval/mean_selected",        eval_stats["mean_selected"],        self._global_step)
                    self.writer.add_scalar("eval/mean_selection_rate",  eval_stats["mean_selection_rate"],  self._global_step)
                    self.writer.add_scalar("eval/mean_collection_cost", eval_stats["mean_collection_cost"], self._global_step)
                    self.writer.add_scalar("eval/mean_imp_score",       eval_stats["mean_imp_score"],       self._global_step)
                    if "mean_coverage" in eval_stats:
                        self.writer.add_scalar("eval/anomaly_coverage", eval_stats["mean_coverage"], self._global_step)

                if eval_stats["mean_reward"] > self._best_eval_reward:
                    self._best_eval_reward = eval_stats["mean_reward"]
                    self.save_model(self.model_path)           # lightweight — weights only
                    self.save_checkpoint(self.checkpoint_path) # full — for resuming
                    print(f"  >> Saved best model + checkpoint  (reward={self._best_eval_reward:.3f})")
                    if self.writer:
                        self.writer.add_scalar("eval/best_reward", self._best_eval_reward, self._global_step)

                self._flush_log()

        print(f"\nTraining complete. Best eval reward: {self._best_eval_reward:.3f}")
        self._flush_log()
        if self.writer:
            self.writer.close()
        return self._log_rows

    # ------------------------------------------------------------------
    # Reward normalization
    # ------------------------------------------------------------------

    def _normalize_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Normalize rewards using Welford online running statistics.
        Keeps returns in a reasonable scale (~1-10) so value loss stays
        manageable and doesn't drown the policy gradient signal.
        """
        r_np = rewards.cpu().numpy()
        for r in r_np:
            self._reward_running_count += 1
            delta = r - self._reward_running_mean
            self._reward_running_mean += delta / self._reward_running_count
            delta2 = r - self._reward_running_mean
            self._reward_running_var  += delta * delta2

        std = np.sqrt(self._reward_running_var / max(self._reward_running_count - 1, 1))
        std = max(std, 1e-8)
        normalized = (r_np - self._reward_running_mean) / std
        return torch.tensor(normalized, dtype=torch.float32, device=self.device)

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------

    def _collect_rollout(self) -> dict:
        """
        Collect one rollout of n_steps transitions from all envs.
        Returns train-time stats (selected, coverage, cost, reward)
        accumulated over the rollout — logged under train/ in TensorBoard.
        """
        self.buffer.reset()
        self.policy.eval()

        # Accumulators for train-time metrics
        rollout_selected  = []
        rollout_coverage  = []
        rollout_imp       = []
        rollout_cost      = []
        rollout_rewards   = []

        with torch.no_grad():
            for _ in range(self.n_steps):
                obs_t = torch.tensor(self._obs, dtype=torch.float32, device=self.device)
                actions, log_probs, _, values = self.policy.get_action_and_value(obs_t)

                actions_np   = actions.cpu().numpy()
                values_np    = values.cpu().numpy()

                rewards   = np.zeros(self.n_envs, dtype=np.float32)
                new_obs   = np.zeros_like(self._obs)
                new_dones = np.zeros(self.n_envs, dtype=bool)

                for i, env in enumerate(self.envs):
                    obs_, rew_, term_, trunc_, info = env.step(actions_np[i])
                    rewards[i]   = rew_
                    new_dones[i] = term_ or trunc_

                    # Accumulate info metrics from every step
                    if "selected_count" in info:
                        rollout_selected.append(info["selected_count"])
                    if "anomaly_coverage" in info:
                        rollout_coverage.append(info["anomaly_coverage"])
                    if "r_imp" in info:
                        rollout_imp.append(info["r_imp"])
                    if "r_cost" in info:
                        rollout_cost.append(info["r_cost"])
                    rollout_rewards.append(rew_)

                    if term_ or trunc_:
                        obs_, _ = env.reset()
                    new_obs[i] = obs_

                rewards_tensor     = torch.tensor(rewards, device=self.device)
                rewards_normalized = self._normalize_rewards(rewards_tensor)

                self.buffer.add(
                    obs_t,
                    actions,
                    log_probs,
                    rewards_normalized,
                    torch.tensor(values_np, device=self.device),
                    torch.tensor(self._dones.astype(np.float32), device=self.device),
                )
                self._obs   = new_obs
                self._dones = new_dones

            last_obs_t  = torch.tensor(self._obs, dtype=torch.float32, device=self.device)
            last_values = self.policy.get_value(last_obs_t)
            last_dones  = torch.tensor(self._dones.astype(np.float32), device=self.device)

        self.buffer.compute_returns_and_advantages(
            last_values, last_dones, self.gamma, self.gae_lambda
        )

        # Build rollout stats dict
        rollout_stats = {
            "mean_reward":    float(np.mean(rollout_rewards)) if rollout_rewards else float("nan"),
        }
        if rollout_selected:
            total_metrics = self.envs[0].observation_space.shape[0] // 5  # NM
            rollout_stats["mean_selected"]       = float(np.mean(rollout_selected))
            rollout_stats["mean_selection_rate"] = float(np.mean(rollout_selected)) / max(total_metrics, 1)
        if rollout_coverage:
            rollout_stats["mean_coverage"]       = float(np.mean(rollout_coverage))
        if rollout_imp:
            rollout_stats["mean_imp_score"]      = float(np.mean(rollout_imp))
        if rollout_cost:
            rollout_stats["mean_collection_cost"]= float(np.mean(rollout_cost))

        return rollout_stats

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def _ppo_update(self) -> dict:
        self.policy.train()

        adv = self.buffer.advantages
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        self.buffer.advantages = adv

        total_loss  = 0.0
        total_pg    = 0.0
        total_vf    = 0.0
        total_ent   = 0.0
        total_clips = 0
        n_batches   = 0

        loader = self.buffer.get_dataloader(self.batch_size)

        for epoch in range(self.n_epochs):
            for obs_b, act_b, old_lp_b, adv_b, ret_b, old_val_b in loader:
                _, new_lp, entropy, new_val = self.policy.get_action_and_value(
                    obs_b, action=act_b
                )

                ratio    = torch.exp(new_lp - old_lp_b)
                pg_loss1 = -adv_b * ratio
                pg_loss2 = -adv_b * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                pg_loss  = torch.max(pg_loss1, pg_loss2).mean()

                v_clipped = old_val_b + torch.clamp(
                    new_val - old_val_b, -self.clip_range, self.clip_range
                )
                vf_loss = torch.max(
                    F.mse_loss(new_val, ret_b),
                    F.mse_loss(v_clipped, ret_b)
                )

                ent_loss = entropy.mean()
                loss     = pg_loss + self.vf_coef * vf_loss - self.ent_coef * ent_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_loss  += loss.item()
                total_pg    += pg_loss.item()
                total_vf    += vf_loss.item()
                total_ent   += ent_loss.item()
                total_clips += ((ratio - 1).abs() > self.clip_range).float().mean().item()
                n_batches   += 1

        nb = max(n_batches, 1)
        return {
            "loss":        total_loss  / nb,
            "policy_loss": total_pg    / nb,
            "value_loss":  total_vf    / nb,
            "entropy":     total_ent   / nb,
            "clip_frac":   total_clips / nb,
        }

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, n_episodes: int) -> dict:
        self.policy.eval()
        all_rewards      = []
        all_selected     = []
        all_coverage     = []
        all_imp          = []
        all_sel_rate     = []
        all_coll_cost    = []

        with torch.no_grad():
            for ep_i in range(n_episodes):
                env = self.eval_envs[ep_i % len(self.eval_envs)]
                obs, _ = env.reset()
                done = False
                ep_reward   = 0.0
                ep_selected = []
                ep_coverage = []
                ep_imp      = []
                ep_cost     = []

                while not done:
                    obs_t  = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                    action = self.policy.get_deterministic_action(obs_t).squeeze(0)
                    obs, rew, term, trunc, info = env.step(action.cpu().numpy())
                    done = term or trunc
                    ep_reward   += rew
                    ep_selected.append(info["selected_count"])
                    if "anomaly_coverage" in info:
                        ep_coverage.append(info["anomaly_coverage"])
                    ep_imp.append(info.get("r_imp", 0.0))
                    ep_cost.append(info.get("r_cost", 0.0))

                all_rewards.append(ep_reward)
                all_selected.append(np.mean(ep_selected))
                if ep_coverage:
                    all_coverage.append(np.mean(ep_coverage))
                all_imp.append(np.mean(ep_imp))
                # selection_rate = mean_selected / total_metrics
                all_sel_rate.append(
                    float(np.mean(ep_selected)) / max(self.total_metrics, 1)
                )
                # collection cost = mean of r_cost info values per window
                all_coll_cost.append(float(np.mean(ep_cost)))

        stats = {
            "mean_reward":          float(np.mean(all_rewards)),
            "std_reward":           float(np.std(all_rewards)),
            "mean_selected":        float(np.mean(all_selected)),
            "mean_selection_rate":  float(np.mean(all_sel_rate)),
            "mean_collection_cost": float(np.mean(all_coll_cost)),
            "mean_imp_score":       float(np.mean(all_imp)),
        }
        if all_coverage:
            stats["mean_coverage"] = float(np.mean(all_coverage))
        return stats

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save_model(self, path: str):
        """
        Save policy weights only (lightweight file for inference/evaluation).
        Used as best_model.pt — load with load_model() or torch.load() directly.
        """
        torch.save(self.policy.state_dict(), path)

    def save_checkpoint(self, path: str):
        """
        Save full trainer state for resuming training.
        Includes: policy weights, optimizer state, global step,
        best eval reward, and Welford running stats.
        """
        torch.save({
            "policy_state":     self.policy.state_dict(),
            "optimizer_state":  self.optimizer.state_dict(),
            "global_step":      self._global_step,
            "best_eval_reward": self._best_eval_reward,
            "reward_mean":      self._reward_running_mean,
            "reward_var":       self._reward_running_var,
            "reward_count":     self._reward_running_count,
        }, path)

    def load_model(self, path: str):
        """
        Load policy weights only (for inference after training).
        For resuming training, use load_checkpoint() instead.
        """
        self.policy.load_state_dict(torch.load(path, map_location=self.device))

    def load_checkpoint(self, path: str):
        """
        Load full trainer state for resuming training.
        Falls back gracefully to policy-weights-only if the file is a
        legacy best_model.pt saved before this fix was applied.
        """
        ckpt = torch.load(path, map_location=self.device)

        if isinstance(ckpt, dict) and "policy_state" in ckpt:
            # Full checkpoint
            self.policy.load_state_dict(ckpt["policy_state"])
            self.optimizer.load_state_dict(ckpt["optimizer_state"])
            self._global_step          = ckpt["global_step"]
            self._best_eval_reward     = ckpt["best_eval_reward"]
            self._reward_running_mean  = ckpt["reward_mean"]
            self._reward_running_var   = ckpt["reward_var"]
            self._reward_running_count = ckpt["reward_count"]
            print(f"[ppo] Full checkpoint loaded from {path}")
            print(f"[ppo]   global_step      = {self._global_step:,}")
            print(f"[ppo]   best_eval_reward = {self._best_eval_reward:.4f}")
            print(f"[ppo]   reward_count     = {self._reward_running_count:,}")
        else:
            # Legacy: policy weights only
            self.policy.load_state_dict(ckpt)
            print(f"[ppo] Legacy checkpoint (policy weights only) loaded from {path}")
            print(f"[ppo] Optimizer state and step counters start fresh.")

    # ------------------------------------------------------------------
    # CSV log
    # ------------------------------------------------------------------

    def _flush_log(self):
        if not self._log_rows:
            return
        keys = list(self._log_rows[0].keys())
        with open(self.log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self._log_rows)