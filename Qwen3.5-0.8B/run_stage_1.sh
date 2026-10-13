#!/usr/bin/env bash
# ==============================================================================
# 🚀 BareTorch: Data Preparation & Sync Runner (20B Tokens for Full Scale Distillation)
# ==============================================================================
# Description: Syncs the 20B token multi-domain binary dataset shards from Cloudflare
#              R2 to local NVMe storage for the 2.5B model distillation run.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

# Target directory on NVMe storage (or project root)
DATA_TARGET_DIR="${PROJECT_ROOT}/data_20B/train"
TOTAL_TOKENS="20000000000"

# Prefer the generic sync script if present, otherwise fall back to sync_1b_dataset.py
SYNC_SCRIPT="${PROJECT_ROOT}/sync_dataset.py"
if [ ! -f "${SYNC_SCRIPT}" ]; then
    SYNC_SCRIPT="${PROJECT_ROOT}/sync_1b_dataset.py"
fi

mkdir -p "${DATA_TARGET_DIR}"

echo "=================================================================="
echo "🚀 Starting Dataset Sync: 20 Billion Tokens (7-Domain Mixture)"
echo "=================================================================="
echo "Project Root       : ${PROJECT_ROOT}"
echo "Sync Script        : ${SYNC_SCRIPT}"
echo "Target Directory   : ${DATA_TARGET_DIR}"
echo "Total Tokens       : ${TOTAL_TOKENS} (20 Billion Tokens)"
echo "=================================================================="

python3 "${SYNC_SCRIPT}" \
    --target_dir "${DATA_TARGET_DIR}" \
    --total_tokens "${TOTAL_TOKENS}"

echo "=================================================================="
echo "✅ 20BT Dataset Sync Complete!"
echo "   Verifying downloaded shards in: ${DATA_TARGET_DIR}"
echo "=================================================================="

# Print summary of downloaded shard files and total disk space
ls -lh "${DATA_TARGET_DIR}" | head -n 15
echo "Total size on disk:"
du -sh "${DATA_TARGET_DIR}"