"""
seeding.py
==========
Seed resolution and global RNG control.

Why two seeds
-------------
Randomness in this project comes from two independent places:

  data seed   — episode generation: anomaly bursts, causal scores, node
                capacities, tier assignment.  Defines the *scenario*.

  train seed  — network initialisation, action sampling during rollouts, and
                minibatch shuffling.  Defines the *training randomness*.

Until now a single `data.seed` drove both, which makes reported variance
uninterpretable.  The baselines (Monitor-Everything, TopK-Anomaly-Causal,
Random-Feasible) do not train at all, so their spread across runs measures
scenario variance only, while the RL methods' spread measures scenario +
initialisation + sampling variance.  Those two error bars are not comparable,
and a single knob cannot separate them.

With the split you can run either protocol deliberately:

  fixed data seed, varying train seed  → training stability on a fixed scenario
                                          (baselines are constant by definition)
  varying data seed, fixed train seed  → sensitivity to the scenario
                                          (comparable across all methods)

Backwards compatibility
-----------------------
`ppo.seed` defaults to `data.seed` when absent, and `--seed` still sets both,
so every existing config and command line reproduces its previous behaviour
exactly.
"""

import os
import random
from typing import Optional, Tuple

import numpy as np
import torch


# ------------------------------------------------------------------
# Seed resolution
# ------------------------------------------------------------------

def resolve_seeds(
    cfg: dict,
    seed: Optional[int] = None,
    data_seed: Optional[int] = None,
    train_seed: Optional[int] = None,
) -> Tuple[int, int]:
    """
    Work out the (data_seed, train_seed) pair actually in force.

    Precedence, lowest to highest:
        1. cfg["data"]["seed"]                  — scenario seed
           cfg["ppo"]["seed"]  (if present)     — training seed
        2. --seed          sets BOTH (legacy behaviour)
        3. --data-seed / --train-seed           set one each

    Returns the pair; the caller is responsible for writing it back into cfg
    so that the recorded config reflects what ran.
    """
    d = int(cfg["data"]["seed"])
    t = cfg.get("ppo", {}).get("seed")
    t = int(t) if t is not None else d          # legacy: train follows data

    if seed is not None:                        # --seed sets both
        d = t = int(seed)
    if data_seed is not None:
        d = int(data_seed)
    if train_seed is not None:
        t = int(train_seed)

    return d, t


def apply_seeds_to_cfg(cfg: dict, data_seed: int, train_seed: int) -> dict:
    """
    Write the resolved seeds back into the config dict, so that dumping cfg
    records what actually ran rather than what the file happened to say.
    """
    cfg["data"]["seed"] = data_seed
    cfg.setdefault("ppo", {})["seed"] = train_seed
    return cfg


# ------------------------------------------------------------------
# Global RNG control
# ------------------------------------------------------------------

def set_global_seeds(train_seed: int, deterministic: bool = False) -> None:
    """
    Seed every global RNG that affects training.

    Covers Python's `random`, NumPy's legacy global RNG, and PyTorch on both
    CPU and all CUDA devices.  Previously only NumPy and CPU-side PyTorch were
    seeded, leaving `random` and CUDA unseeded.

    Note this seeds the *training* RNGs only.  Episode generation uses its own
    `np.random.default_rng(data_seed + i)` per episode and is unaffected by
    global state, which is what makes the two seeds genuinely independent.

    deterministic=True additionally pins cuDNN and asks PyTorch to reject
    nondeterministic kernels.  It costs throughput and some ops have no
    deterministic implementation, so it is opt-in rather than the default.
    """
    random.seed(train_seed)
    np.random.seed(train_seed)
    torch.manual_seed(train_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Required by some CUDA kernels for reproducible reductions.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True)
        except Exception as exc:          # pragma: no cover - backend dependent
            print(f"[seed] WARNING: could not enable deterministic algorithms "
                  f"({exc}). Results may still vary slightly on GPU.")


def describe_seeds(data_seed: int, train_seed: int,
                   deterministic: bool = False) -> str:
    """One-line summary for run logs."""
    return (f"data_seed={data_seed} (scenario)  "
            f"train_seed={train_seed} (init/rollout/batching)  "
            f"deterministic={deterministic}")
