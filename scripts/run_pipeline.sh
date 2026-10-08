#!/usr/bin/env bash
# Reproducible end-to-end pipeline: waits for SSL pretraining to finish (if
# still running), then runs the three finetune variants needed for the
# ablation study (full model, no-SSL-init, no-equivariance).
#
# Usage: bash scripts/run_pipeline.sh [SSL_PID]
set -euo pipefail
cd "$(dirname "$0")/.."

SSL_PID="${1:-}"
if [ -n "$SSL_PID" ]; then
    echo "Waiting for SSL pretraining (pid $SSL_PID) to finish..."
    while kill -0 "$SSL_PID" 2>/dev/null; do
        sleep 15
    done
    echo "SSL pretraining finished."
fi

export PYTHONPATH=src

if [ ! -f results/baselines.json ]; then
    echo "=== Baselines (logreg / random forest / plain MLP) ==="
    python3 -m tcga_bench.training.baselines --config configs/default.yaml || true
fi

echo "=== Finetune: full model (SSL init + equivariant encoder + Bayesian head) ==="
python3 -m tcga_bench.training.finetune --config configs/default.yaml --tag full

echo "=== Ablation: no SSL pretraining ==="
python3 -m tcga_bench.training.finetune --config configs/default.yaml --tag no_ssl --no_ssl

echo "=== Ablation: no equivariant encoder (plain MLP backbone) ==="
python3 -m tcga_bench.training.finetune --config configs/default.yaml --tag no_equivariance --no_equivariance

echo "ALL_EXPERIMENTS_DONE"
