"""
policy.py
=========
Actor-Critic policy networks for multi-binary metric selection.

Three architectures are provided:

ActorCritic (existing)
    Flat MLP over the full flattened observation [N*M*F → h1 → h2 → N*M].
    Works well for small (N, M). Struggles for large action spaces (N=28,
    M=38) because the first linear layer must compress N*M*F features in
    one step with no structural inductive bias — logit (n,m) has no
    privileged connection to feature (n,m).

PerMetricActorCritic (new — recommended for large N, M)
    Processes each (node, metric) pair independently with a small shared
    MLP (same weights for all pairs), then combines local embeddings with
    a global context vector to produce one logit per pair.

    Key properties:
    - Parameter sharing across (n,m) pairs: the network learns "what makes
      a metric worth selecting" in feature space, not by memorising indices.
      A metric with high anomaly score is treated the same regardless of
      which (n,m) index it has.
    - The inductive bias explicitly connects feature [n,m] to logit [n,m]
      via the per-metric encoder — no need to discover this mapping from
      a large weight matrix.
    - Scales to any (N, M): parameter count is independent of N*M because
      the per-metric encoder weights are shared. Only the global encoder
      input size grows with N*M.
    - Better gradient signal: the policy gradient for logit (n,m) flows
      through the local embedding of features (n,m) directly.

MaskedActorCritic (existing)
    Gaussian policy for use with MonitoringEnvMasked (hard TopK masking).

Usage
-----
    # Small problem (N<=10, M<=15) — use existing flat MLP
    from agents.policy import ActorCritic
    policy = ActorCritic(obs_dim, action_dim, hidden_sizes=[512, 256])

    # Large problem (N=28, M=38) — use per-metric architecture
    from agents.policy import PerMetricActorCritic
    policy = PerMetricActorCritic(
        n_nodes=28, n_metrics=38, feature_dim=6,
        metric_hidden=128, global_hidden=256, context_hidden=128,
    )

    # Both have identical external interface:
    # get_action_and_value(), get_deterministic_action(), get_value()
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli
from typing import List, Tuple, Optional


# ══════════════════════════════════════════════════════════════════
# Shared building block
# ══════════════════════════════════════════════════════════════════

class SharedMLP(nn.Module):
    """Shared encoder used by both actor and critic."""

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: List[int],
        use_layer_norm: bool = True,
    ):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            if use_layer_norm:
                layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU())
            in_dim = h
        self.net     = nn.Sequential(*layers)
        self.out_dim = in_dim

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ══════════════════════════════════════════════════════════════════
# Existing flat MLP architecture (unchanged)
# ══════════════════════════════════════════════════════════════════

class ActorCritic(nn.Module):
    """
    Combined Actor-Critic for multi-binary action space.

    Flat MLP: obs [N*M*F] → shared encoder → actor logits [N*M] + value.
    Recommended for small problems (N<=10, M<=15).
    For larger problems use PerMetricActorCritic.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: List[int] = (512, 256),
        use_layer_norm: bool = True,
    ):
        super().__init__()
        self.obs_dim    = obs_dim
        self.action_dim = action_dim

        self.encoder = SharedMLP(obs_dim, list(hidden_sizes), use_layer_norm)

        self.actor_head = nn.Linear(self.encoder.out_dim, action_dim)
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.zeros_(self.actor_head.bias)

        self.critic_head = nn.Linear(self.encoder.out_dim, 1)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)
        nn.init.zeros_(self.critic_head.bias)

    def forward(self, obs: torch.Tensor):
        h      = self.encoder(obs)
        logits = self.actor_head(h)
        value  = self.critic_head(h).squeeze(-1)
        return logits, value

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        h = self.encoder(obs)
        return self.critic_head(h).squeeze(-1)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(obs)
        dist = Bernoulli(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy  = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy, value

    def get_deterministic_action(self, obs: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward(obs)
        return (torch.sigmoid(logits) > 0.5).float()


# ══════════════════════════════════════════════════════════════════
# New: Per-Metric Actor-Critic (recommended for large N, M)
# ══════════════════════════════════════════════════════════════════

class PerMetricActorCritic(nn.Module):
    """
    Actor-Critic with a shared per-(node,metric) encoder and global context.

    Architecture
    ------------
    Input obs: [B, N*M*F]  →  reshape to  [B, N*M, F]

    Step 1 — Local encoding (shared weights across all N*M pairs):
        local[b, k, :] = metric_encoder(obs[b, k, :])   k = 0..N*M-1
        Shape: [B, N*M, metric_hidden]

        The same small MLP is applied independently to each (node, metric)
        feature vector. Weight sharing means the network learns a universal
        "importance scoring function" over the 5 features rather than
        memorising per-index patterns.

    Step 2 — Global context (pool over all pairs, then encode):

        global_pool="mean"  (default, recommended for large N*M)
            pooled     = local.mean(dim=1)      # [B, metric_hidden]
            global_ctx = global_encoder(pooled) # [B, global_hidden]
            Global encoder input size = metric_hidden — independent of N*M.
            Scales to any (N, M) without parameter explosion.
            Loses: exact knowledge of which specific metrics are anomalous.
            Retains: overall system anomaly level.

        global_pool="concat"  (stronger but only for small N*M)
            flat       = local.reshape(B, N*M*metric_hidden)
            global_ctx = global_encoder(flat)   # [B, global_hidden]
            Global encoder input size = N*M*metric_hidden (grows with N*M).
            For N=28, M=38, mh=128 → input = 136K — very large linear layer.
            Retains: full joint information across all (node, metric) pairs.
            Use only when N*M <= ~200 (e.g. N=10, M=15 → 1500 input units).

    Step 3 — Per-metric logit (local + global → scalar):
        ctx_exp = global_ctx.unsqueeze(1).expand(-1, N*M, -1)
        combined = cat([local, ctx_exp], dim=-1)   [B, N*M, metric_hidden+global_hidden]
        logits = logit_head(combined).squeeze(-1)   [B, N*M]

        Each metric's selection logit depends on its own features (local)
        AND the global system state (ctx). This is the key inductive bias:
        logit (n,m) is a direct function of feature (n,m).

    Step 4 — Value (from global context only):
        value = critic_head(global_ctx)   [B]

    Parameters
    ----------
    n_nodes        : N — number of nodes
    n_metrics      : M — metrics per node
    feature_dim    : F — features per (node, metric) pair (default 5)
    metric_hidden  : hidden dim of the shared per-metric encoder
    global_hidden  : hidden dim of the global context encoder
    use_layer_norm : apply LayerNorm after each linear layer
    global_pool    : "mean" (default) or "concat"
                     "mean"   — scalable, use for large N*M (N=28, M=38)
                     "concat" — more expressive, use for small N*M (N<=10, M<=15)
    """

    def __init__(
        self,
        n_nodes:        int,
        n_metrics:      int,
        feature_dim:    int = 5,
        metric_hidden:  int = 128,
        global_hidden:  int = 256,
        use_layer_norm: bool = True,
        global_pool:    str  = "mean",   # "mean" | "concat"
    ):
        super().__init__()

        if global_pool not in ("mean", "concat"):
            raise ValueError(
                f"global_pool must be 'mean' or 'concat', got '{global_pool}'"
            )

        self.n_nodes       = n_nodes
        self.n_metrics     = n_metrics
        self.feature_dim   = feature_dim
        self.metric_hidden = metric_hidden
        self.global_hidden = global_hidden
        self.global_pool   = global_pool

        self.obs_dim    = n_nodes * n_metrics * feature_dim
        self.action_dim = n_nodes * n_metrics

        n_pairs = n_nodes * n_metrics

        # ── Step 1: Per-metric local encoder (shared weights) ────
        if use_layer_norm:
            self.metric_encoder = nn.Sequential(
                nn.Linear(feature_dim, metric_hidden),
                nn.LayerNorm(metric_hidden),
                nn.ReLU(),
                nn.Linear(metric_hidden, metric_hidden),
                nn.LayerNorm(metric_hidden),
                nn.ReLU(),
            )
        else:
            self.metric_encoder = nn.Sequential(
                nn.Linear(feature_dim, metric_hidden),
                nn.ReLU(),
                nn.Linear(metric_hidden, metric_hidden),
                nn.ReLU(),
            )

        # ── Step 2: Global context encoder ───────────────────────
        # Input size depends on global_pool:
        #   "mean"   → metric_hidden          (independent of N*M — scalable)
        #   "concat" → n_pairs * metric_hidden (grows with N*M — expressive)
        global_in = metric_hidden if global_pool == "mean" else n_pairs * metric_hidden

        if use_layer_norm:
            self.global_encoder = nn.Sequential(
                nn.Linear(global_in, global_hidden),
                nn.LayerNorm(global_hidden),
                nn.ReLU(),
            )
        else:
            self.global_encoder = nn.Sequential(
                nn.Linear(global_in, global_hidden),
                nn.ReLU(),
            )

        # ── Step 3: Per-metric logit head ─────────────────────────
        self.logit_head = nn.Linear(metric_hidden + global_hidden, 1)

        # ── Step 4: Critic head ───────────────────────────────────
        self.critic_head = nn.Linear(global_hidden, 1)

        self._init_weights()

    def _init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if "logit_head" in name:
                    # Small init → near-uniform initial policy (50/50)
                    nn.init.orthogonal_(m.weight, gain=0.01)
                elif "critic_head" in name:
                    nn.init.orthogonal_(m.weight, gain=1.0)
                else:
                    nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)

    def _encode(self, obs: torch.Tensor):
        """
        Core encoding pass. Returns (logits, value).

        obs : [B, N*M*F]
        """
        B = obs.shape[0]
        NM = self.n_nodes * self.n_metrics
        F  = self.feature_dim

        # Reshape to (node, metric) pairs
        x = obs.reshape(B, NM, F)                     # [B, NM, F]

        # Step 1: local encoding — apply shared metric_encoder to each pair
        # Flatten batch and pairs for efficient batched linear ops
        x_flat  = x.reshape(B * NM, F)                # [B*NM, F]
        local   = self.metric_encoder(x_flat)          # [B*NM, metric_hidden]
        local   = local.reshape(B, NM, self.metric_hidden)  # [B, NM, mh]

        # Step 2: global context — branches on self.global_pool
        if self.global_pool == "mean":
            # Mean pool → [B, metric_hidden], independent of NM
            pooled     = local.mean(dim=1)
            global_ctx = self.global_encoder(pooled)          # [B, global_hidden]
        else:
            # Concat → [B, NM*metric_hidden], captures full joint info
            flat       = local.reshape(B, NM * self.metric_hidden)
            global_ctx = self.global_encoder(flat)            # [B, global_hidden]

        # Step 3: per-metric logit
        ctx_exp  = global_ctx.unsqueeze(1).expand(-1, NM, -1)    # [B, NM, gh]
        combined = torch.cat([local, ctx_exp], dim=-1)            # [B, NM, mh+gh]
        logits   = self.logit_head(combined).squeeze(-1)          # [B, NM]

        # Step 4: value
        value = self.critic_head(global_ctx).squeeze(-1)          # [B]

        return logits, value

    def forward(self, obs: torch.Tensor):
        return self._encode(obs)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        _, value = self._encode(obs)
        return value

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample or evaluate action.

        Returns
        -------
        action   : [B, N*M]  binary selections
        log_prob : [B]       sum of per-metric log-probs
        entropy  : [B]       sum of per-metric entropies
        value    : [B]       critic value estimate
        """
        logits, value = self._encode(obs)
        dist = Bernoulli(logits=logits)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy  = dist.entropy().sum(dim=-1)

        return action, log_prob, entropy, value

    def get_deterministic_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Greedy action: select metric if sigmoid(logit) > 0.5."""
        logits, _ = self._encode(obs)
        return (torch.sigmoid(logits) > 0.5).float()


# ══════════════════════════════════════════════════════════════════
# Existing Masked Actor-Critic (unchanged)
# ══════════════════════════════════════════════════════════════════

class MaskedActorCritic(nn.Module):
    """
    Actor-Critic for the hard-masked environment (MonitoringEnvMasked).

    Gaussian policy over continuous logits R^{N*M}.
    The environment applies a hard TopK mask per node.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: List[int] = (512, 256),
        use_layer_norm: bool = True,
        log_std_init: float = 0.0,
    ):
        super().__init__()
        self.obs_dim    = obs_dim
        self.action_dim = action_dim

        self.encoder = SharedMLP(obs_dim, list(hidden_sizes), use_layer_norm)

        self.actor_mean = nn.Linear(self.encoder.out_dim, action_dim)
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.zeros_(self.actor_mean.bias)

        self.log_std = nn.Parameter(
            torch.ones(action_dim) * log_std_init
        )

        self.critic_head = nn.Linear(self.encoder.out_dim, 1)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)
        nn.init.zeros_(self.critic_head.bias)

    def forward(self, obs: torch.Tensor):
        h    = self.encoder(obs)
        mean = self.actor_mean(h)
        val  = self.critic_head(h).squeeze(-1)
        return mean, val

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        h = self.encoder(obs)
        return self.critic_head(h).squeeze(-1)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        from torch.distributions import Normal
        mean, value = self.forward(obs)
        std  = torch.exp(self.log_std.clamp(-4, 2))
        dist = Normal(mean, std)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy  = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy, value

    def get_deterministic_action(self, obs: torch.Tensor) -> torch.Tensor:
        mean, _ = self.forward(obs)
        return mean