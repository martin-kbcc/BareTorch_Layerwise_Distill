#!/usr/bin/env bash
# ==============================================================================
# 🚀 BareTorch Stage 1: Data Preparation & Sync Runner (Qwen3.5-0.8B)
# ==============================================================================
# Description: Syncs the 20M token multi-domain binary dataset from Cloudflare R2
#              required for Stage 1 layerwise distillation.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

DATA_TARGET_DIR="${PROJECT_ROOT}/data_20M/train"
TOTAL_TOKENS="20000000"

echo "=================================================================="
echo "🚀 Starting Stage 1 Dataset Sync (20M Tokens across 7 Domains)"
echo "=================================================================="
echo "Project Root       : ${PROJECT_ROOT}"
echo "Target Directory   : ${DATA_TARGET_DIR}"
echo "Total Tokens       : ${TOTAL_TOKENS}"
echo "=================================================================="

python3 "${PROJECT_ROOT}/sync_20m_dataset.py" \
    --target_dir "${DATA_TARGET_DIR}" \
    --total_tokens "${TOTAL_TOKENS}"

echo "=================================================================="
echo "✅ Stage 1 Dataset Sync Complete!"
echo "=================================================================="