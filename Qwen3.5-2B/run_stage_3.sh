#!/usr/bin/env bash
# ==============================================================================
# 🚀 BareTorch Stage 3: Data Preparation & Sync Runner (5B Tokens for Qwen3.5-2B)
# ==============================================================================
# Description: Syncs the 5B token multi-domain teacher predictions binary dataset
#              from Cloudflare R2 required for Stage 4 global logit distillation.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

DATA_TARGET_DIR="${PROJECT_ROOT}/data_5B/train"
TOTAL_TOKENS="5000000000"

echo "=================================================================="
echo "🚀 Starting Stage 3 Dataset Sync (5B Tokens across 7 Domains)"
echo "=================================================================="
echo "Project Root       : ${PROJECT_ROOT}"
echo "Target Directory   : ${DATA_TARGET_DIR}"
echo "Total Tokens       : ${TOTAL_TOKENS} (5B)"
echo "=================================================================="

python3 "${PROJECT_ROOT}/sync_1b_dataset.py" \
    --target_dir "${DATA_TARGET_DIR}" \
    --total_tokens "${TOTAL_TOKENS}"

echo "=================================================================="
echo "✅ Stage 3 Dataset Sync Complete!"
echo "=================================================================="