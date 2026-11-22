"""
tensorboard_viz.py
==================
Launches TensorBoard and optionally logs extra per-metric statistics
(reward component breakdown, per-node capacity violations, etc.)
that are not tracked during training.

Usage:
    # Just launch TensorBoard
    python scripts/tensorboard_viz.py

    # Log extra analysis from a saved training log, then launch
    python scripts/tensorboard_viz.py --enrich --port 6007
"""

import os
import sys
import argparse
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir",  default="results/tensorboard",
                        help="TensorBoard log directory")
    parser.add_argument("--port",    type=int, default=6006)
    parser.add_argument("--enrich",  action="store_true",
                        help="Re-log CSV training data into TensorBoard before launching")
    parser.add_argument("--config",  default="configs/config.yaml")
    return parser.parse_args()


# ------------------------------------------------------------------
# Optional: re-log CSV → TensorBoard (fills gaps if writer crashed)
# ------------------------------------------------------------------

def enrich_from_csv(log_csv: str, tb_dir: str):
    """
    Re-reads training_log.csv and writes every column as a scalar
    into TensorBoard. Useful if you want to visualise a run that
    didn't have TensorBoard enabled.
    """
    try:
        from torch.utils.tensorboard import SummaryWriter
        import csv
    except ImportError:
        print("[tb] tensorboard not installed — run: pip install tensorboard")
        return

    if not os.path.exists(log_csv):
        print(f"[tb] No CSV found at {log_csv} — skipping enrichment")
        return

    writer = SummaryWriter(log_dir=tb_dir)
    with open(log_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            step = int(float(row["step"]))
            for key, val in row.items():
                if key == "step":
                    continue
                try:
                    writer.add_scalar(f"csv/{key}", float(val), step)
                except (ValueError, TypeError):
                    pass
    writer.close()
    print(f"[tb] Enriched TensorBoard logs from CSV → {tb_dir}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    args = parse_args()

    # Check tensorboard is installed
    try:
        import tensorboard
    except ImportError:
        print("\n[tb] TensorBoard is not installed.")
        print("     Install it with:  pip install tensorboard\n")
        sys.exit(1)

    tb_dir  = args.logdir
    os.makedirs(tb_dir, exist_ok=True)

    # Optionally enrich from CSV
    if args.enrich:
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        log_csv = os.path.join(cfg["paths"]["results_dir"], cfg["paths"]["log_name"])
        enrich_from_csv(log_csv, tb_dir)

    # Launch TensorBoard
    url = f"http://localhost:{args.port}"
    print(f"\n[tb] Starting TensorBoard...")
    print(f"[tb] Log directory : {os.path.abspath(tb_dir)}")
    print(f"[tb] Open browser  : {url}\n")

    cmd = ["tensorboard", f"--logdir={tb_dir}", f"--port={args.port}", "--bind_all"]
    subprocess.run(cmd)


if __name__ == "__main__":
    main()