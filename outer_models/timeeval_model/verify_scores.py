"""
step3_verify_scores.py
======================
Loads the extracted anomaly score arrays and verifies they look correct
before feeding them into the RL agent.

Checks performed:
  1. Shape is [T, 38] and values are in [0,1]
  2. Mean score on anomalous timesteps > mean score on normal timesteps
     (LSTM-AD should score anomalies higher)
  3. No all-zero metrics (would mean LSTM-AD failed for that metric)
  4. Saves a heatmap plot: anomaly scores over time per metric

Usage:
    python step3_verify_scores.py \
        --out-dir data/anomaly_scores \
        --machine machine-1-1
"""

import argparse
import numpy as np
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir",  default="data/anomaly_scores")
    p.add_argument("--machine",  default=None,
                   help="Single machine to inspect. Default: first found.")
    p.add_argument("--plot",     action="store_true",
                   help="Save heatmap plot")
    return p.parse_args()


def verify(out_dir: Path, machine: str, plot: bool):
    score_path = out_dir / f"{machine}_scores.npy"
    label_path = out_dir / f"{machine}_labels.npy"

    if not score_path.exists():
        print(f"[verify] ERROR: {score_path} not found")
        print("[verify] Run step2_run_lstm_ad.py first.")
        return

    scores = np.load(score_path)   # [T, M]
    labels = np.load(label_path)   # [T]
    T, M   = scores.shape

    print(f"\n{'='*60}")
    print(f"  Verification: {machine}")
    print(f"{'='*60}")
    print(f"  Score array shape : {scores.shape}  (T={T}, M={M})")
    print(f"  Score range       : [{scores.min():.6f}, {scores.max():.6f}]")
    print(f"  Label shape       : {labels.shape}")
    print(f"  Anomaly rate      : {labels.mean():.4f}")

    # Check 1: values in [0, 1]
    ok1 = scores.min() >= 0.0 and scores.max() <= 1.0
    print(f"\n  [{'✓' if ok1 else '✗'}] Scores in [0,1]")

    # Check 2: scores higher on anomalous timesteps
    if labels.sum() > 0:
        mean_anom   = scores[labels == 1].mean()
        mean_normal = scores[labels == 0].mean()
        ok2 = mean_anom > mean_normal
        print(f"  [{'✓' if ok2 else '✗'}] Mean score: "
              f"anomaly={mean_anom:.4f}  normal={mean_normal:.4f}  "
              f"(anomaly should be higher)")
    else:
        print("  [?] No anomalous timesteps in labels — cannot verify score quality")

    # Check 3: no all-zero metrics
    zero_metrics = [m for m in range(M) if scores[:, m].max() < 1e-6]
    if zero_metrics:
        print(f"  [✗] {len(zero_metrics)} metrics have all-zero scores: {zero_metrics}")
        print(f"      → LSTM-AD may have failed for these metrics")
    else:
        print(f"  [✓] All {M} metrics have non-zero scores")

    # Per-metric stats
    print(f"\n  Per-metric score summary (top 5 most anomalous):")
    mean_scores = scores[labels == 1].mean(axis=0) if labels.sum() > 0 else scores.mean(axis=0)
    top5 = np.argsort(mean_scores)[::-1][:5]
    for m in top5:
        print(f"    metric_{m:02d}: mean_on_anomaly={mean_scores[m]:.4f}  "
              f"overall_mean={scores[:,m].mean():.4f}  "
              f"overall_max={scores[:,m].max():.4f}")

    # Plot heatmap
    if plot:
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7),
                                            gridspec_kw={"height_ratios": [4, 1]})

            # Heatmap: scores[T, M] transposed → [M, T]
            im = ax1.imshow(
                scores.T,
                aspect="auto",
                cmap="YlOrRd",
                vmin=0, vmax=1,
                interpolation="nearest",
            )
            ax1.set_ylabel("Metric index")
            ax1.set_xlabel("Timestep")
            ax1.set_title(f"{machine} — LSTM-AD anomaly scores (normalized)")
            plt.colorbar(im, ax=ax1, label="Anomaly score [0,1]")

            # Shade anomaly regions on heatmap
            anom_changes = np.diff(np.concatenate([[0], labels, [0]]))
            starts = np.where(anom_changes == 1)[0]
            ends   = np.where(anom_changes == -1)[0]
            for s, e in zip(starts, ends):
                ax1.axvspan(s, e, alpha=0.25, color="blue", linewidth=0)

            # Binary label row
            ax2.fill_between(np.arange(T), labels, alpha=0.6, color="red")
            ax2.set_xlim(0, T)
            ax2.set_ylim(0, 1.2)
            ax2.set_ylabel("Anomaly")
            ax2.set_xlabel("Timestep")
            ax2.set_yticks([0, 1])

            anom_patch  = mpatches.Patch(color="blue",  alpha=0.25, label="Anomaly region (heatmap)")
            label_patch = mpatches.Patch(color="red",   alpha=0.6,  label="Ground truth label")
            fig.legend(handles=[anom_patch, label_patch], loc="upper right", fontsize=9)

            plt.tight_layout()
            save_path = Path(args.out_dir) / f"{machine}_score_heatmap.png"
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"\n  [plot] Saved → {save_path}")
        except ImportError:
            print("\n  [plot] matplotlib not available — install with: pip install matplotlib")

    print(f"\n{'='*60}\n")


def main():
    global args
    args    = parse_args()
    out_dir = Path(args.out_dir)

    if args.machine:
        machine = args.machine
    else:
        # Auto-detect first available machine
        found = sorted(out_dir.glob("*_scores.npy"))
        if not found:
            print(f"[verify] No score files found in {out_dir}")
            return
        machine = found[0].stem.replace("_scores", "")
        print(f"[verify] Auto-selected machine: {machine}")

    verify(out_dir, machine, args.plot)


if __name__ == "__main__":
    main()