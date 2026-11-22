"""
run_record.py
=============
Records what a run actually did, next to its results.

Replaces `shutil.copy(args.config, results_dir)`, which copied the config file
verbatim and therefore recorded the file's seed rather than the one supplied on
the command line.  Any run launched with `--seed K` was silently recorded as
having used whatever the file said, making per-seed results unattributable.

Writes two files into the run directory:

  config.yaml        the *effective* config, i.e. cfg after all CLI overrides
  run_metadata.json  seeds, device, git commit, timestamp, command line
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from typing import Optional

import yaml


def _git_commit(repo_dir: str) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "-C", repo_dir, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _git_dirty(repo_dir: str) -> Optional[bool]:
    try:
        out = subprocess.run(
            ["git", "-C", repo_dir, "status", "--porcelain"],
            capture_output=True, text=True, timeout=5,
        )
        return bool(out.stdout.strip())
    except Exception:
        return None


def save_run_record(
    results_dir: str,
    cfg: dict,
    data_seed: int,
    train_seed: int,
    device: str = "",
    deterministic: bool = False,
    extra: Optional[dict] = None,
) -> None:
    """
    Write the effective config and run metadata into `results_dir`.

    `cfg` must already carry the resolved seeds (see
    utils.seeding.apply_seeds_to_cfg), so that config.yaml and
    run_metadata.json cannot disagree.
    """
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(results_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    meta = {
        "data_seed":     data_seed,
        "train_seed":    train_seed,
        "deterministic": deterministic,
        "device":        str(device),
        "timestamp":     datetime.now().isoformat(timespec="seconds"),
        "git_commit":    _git_commit(repo_dir),
        "git_dirty":     _git_dirty(repo_dir),
        "command":       " ".join(sys.argv),
        "n_nodes":       cfg.get("environment", {}).get("n_nodes"),
        "n_metrics":     cfg.get("environment", {}).get("n_metrics"),
        "reward":        cfg.get("reward"),
        "policy_type":   cfg.get("policy", {}).get("policy_type", "flat"),
    }
    if extra:
        meta.update(extra)

    with open(os.path.join(results_dir, "run_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
