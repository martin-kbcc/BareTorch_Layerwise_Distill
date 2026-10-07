#!/usr/bin/env bash
# ==============================================================================
# 🚀 BareTorch Stage 1: Data Preparation & Sync Runner (Qwen3.5-2B)
# ==============================================================================
# Description: Syncs the 100M token multi-domain binary dataset from Cloudflare R2
#              required for Stage 1 layerwise distillation on Qwen3.5-2B.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

DATA_TARGET_DIR="${PROJECT_ROOT}/data_100M/train"
TOTAL_TOKENS="100000000"

echo "=================================================================="
echo "🚀 Starting Stage 1 Dataset Sync (100M Tokens across 7 Domains)"
echo "=================================================================="
echo "Project Root       : ${PROJECT_ROOT}"
echo "Target Directory   : ${DATA_TARGET_DIR}"
echo "Total Tokens       : ${TOTAL_TOKENS} (100M)"
echo "=================================================================="

python3 "${PROJECT_ROOT}/sync_20m_dataset.py" \
    --target_dir "${DATA_TARGET_DIR}" \
    --total_tokens "${TOTAL_TOKENS}"

echo "=================================================================="
echo "✅ Stage 1 Dataset Sync Complete!"
echo "=================================================================="