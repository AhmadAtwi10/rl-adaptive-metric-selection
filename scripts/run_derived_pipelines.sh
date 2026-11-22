#!/usr/bin/env bash
# Rebuild every derived anomaly-score set from the raw datasets.
#
# Run from the repo root. Needs Docker and a separate venv with TimeEval (see requirements.txt).
# Work dirs must live under $HOME: docker-py resolves the `desktop-linux`
# context, whose VM only shares the home directory.
#
# One work dir per dataset; each detector writes into its own subdirectory and
# they share a single converted copy of the data (see algo_dir() in
# outer_models/timeeval_model/timeeval_pipeline.py).
set -euo pipefail
cd "$(dirname "$0")/.."

ALGO="${ALGO:-lstm_ad}"
TIMEEVAL_PY="${TIMEEVAL_PY:-./timeeval-venv/bin/python}"   # venv with timeeval installed
PY="${PY:-python}"                                         # main environment
KEYS=("${@:-5nodes-7metrics-d0.25-seed42 5nodes-7metrics-d0.25-x0.04-seed42 5nodes-7metrics-d0.05-seed42}")

for KEY in ${KEYS[@]}; do
    RAW="data/raw-synthetic-${KEY}"
    WORK="data/derived-${KEY}"
    LOG="results/derived_scores/${ALGO}_${KEY}.log"
    mkdir -p results/derived_scores
    echo "=== ${ALGO} on ${KEY} ==="
    "$TIMEEVAL_PY" outer_models/timeeval_model/timeeval_pipeline.py \
        --dataset-dir "$RAW" --work-dir "$WORK" \
        --algorithm "$ALGO" --skip-eval > "$LOG" 2>&1
    "$PY" -c "
import pandas as pd
d = pd.read_csv('${WORK}/${ALGO}/anomaly_scores/per_metric_auroc.csv')
print('  mean AUROC=%.4f  min=%.3f  max=%.3f  n=%d'
      % (d.auroc.mean(), d.auroc.min(), d.auroc.max(), len(d)))"
done
