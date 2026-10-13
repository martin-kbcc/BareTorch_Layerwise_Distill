#!/usr/bin/env bash
# ==============================================================================
# 🚀 BareTorch: Direct End-to-End Distillation Runner (20BT Budget on 8x H100)
# ==============================================================================
# Description: Launches 24-layer end-to-end logit distillation across 8 H100 GPUs.
#              Directly warm-starts from Qwen3.5-2B via weight surgery and executes:
#              - Phase 1 (4BT / 15,258 steps): Recurrent routing adaptation (Backbone frozen)
#              - Phase 2 (16BT / 61,034 steps): Global co-adaptation (Differential LRs)
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

# Prevent file descriptor exhaustion across DDP workers & memmaps
ulimit -n 65536

# ------------------------------------------------------------------------------
# 1. Configurable Variables
# ------------------------------------------------------------------------------
NUM_GPUS=8
PYTHON_SCRIPT="${PROJECT_ROOT}/train_stage_1.py"

# Path Directories
DATA_CACHE_DIR="${PROJECT_ROOT}/data_20B/train"
OUTPUT_DIR="${PROJECT_ROOT}/qwen3.5_2B_20BT_clm_checkpoints"
TEACHER_MODEL="Qwen/Qwen3.5-2B"
LAYER_SEQUENCE="cs_lrad,cs_lrad,cs_lrad,transformer"

# Structural Dimensions (Matching Qwen 3.5 2B with Deep Rank-16 State Tracking)
D_MODEL=2048
NUM_LAYERS=24
NUM_HEADS=16
CHUNK_SIZE=32
RANK=16
SEQ_LEN=2048

# Hyperparameters (20 Billion Unique Tokens @ Effective Batch Size 128)
# Global Step Size = 8 GPUs * 4 per-device batch * 4 grad accum * 2048 seq_len = 262,144 tokens/step
# Total Steps for 20B Tokens = 20,000,000,000 / 262,144 ≈ 76,292 steps
# Phase 1 Steps (20% = 4B Tokens) = 4,000,000,000 / 262,144 ≈ 15,258 steps
# Phase 2 Steps (80% = 16B Tokens) = 76,292 - 15,258 = 61,034 steps
MAX_STEPS=76292
PHASE1_STEPS=15258
WARMUP_STEPS=3800
WEIGHT_DECAY="0.01"

# Differential Learning Rates
LEARNING_RATE="2.5e-4"       # Base Peak LR for CS-LRAD mixer layers
BACKBONE_LR_RATIO="0.08"     # 0.08 * 2.5e-4 = 2.0e-5 for MLPs & Native Attention
EMB_LR_RATIO="0.02"          # 0.02 * 2.5e-4 = 5.0e-6 for Embeddings & LM Head

# Hardware Allocation (Effective Batch Size = 8 GPUs * 4 batch * 4 accum = 128)
BATCH_SIZE=4
GRAD_ACCUM=4

# Checkpoint & Eval Frequency (Every ~262M Tokens)
LOGGING_STEPS=50
SAVE_STEPS=1000
EVAL_STEPS=1000

# Cloudflare R2 Sync Settings
ENABLE_R2_SYNC=true
R2_BUCKET="baretorch-data"
R2_PREFIX="qwen3.5_2B_20BT_clm_checkpoints"

# ------------------------------------------------------------------------------
# 2. Environment Variables & Hardware Optimizations
# ------------------------------------------------------------------------------
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# ------------------------------------------------------------------------------
# 3. Pre-Flight Checks & Environment Validation
# ------------------------------------------------------------------------------
echo "=================================================================="
echo "🚀 Initializing BareTorch 20BT End-to-End Distillation (8x H100)"
echo "=================================================================="
echo "Target Python Script   : ${PYTHON_SCRIPT}"
echo "Number of GPUs         : ${NUM_GPUS}x H100 SXM5"
echo "Teacher Model          : ${TEACHER_MODEL}"
echo "Model Dimensions       : d_model=${D_MODEL}, num_layers=${NUM_LAYERS}, num_heads=${NUM_HEADS}"
echo "CS-LRAD Configuration  : rank=${RANK}, chunk_size=${CHUNK_SIZE}, seq_len=${SEQ_LEN}"
echo "Dataset Directory      : ${DATA_CACHE_DIR} (20BT Unique Stream)"
echo "Output Directory       : ${OUTPUT_DIR}"
echo "Tokens per Step        : $(( BATCH_SIZE * GRAD_ACCUM * NUM_GPUS * SEQ_LEN )) (262,144 tokens/step)"
echo "Total Training Steps   : ${MAX_STEPS} (20 Billion Unique Tokens)"
echo "Phase 1 Steps (CS-LRAD): 0 to ${PHASE1_STEPS} (4 Billion Tokens, Backbone Frozen)"
echo "Phase 2 Steps (Global) : ${PHASE1_STEPS} to ${MAX_STEPS} (16 Billion Tokens, Co-Adaptation)"
echo "Base Peak LR (CS-LRAD) : ${LEARNING_RATE}"
echo "Backbone Peak LR       : 2.0e-5 (Ratio: ${BACKBONE_LR_RATIO})"
echo "Embeddings Peak LR     : 5.0e-6 (Ratio: ${EMB_LR_RATIO})"
echo "Cloudflare R2 Sync     : ${ENABLE_R2_SYNC}"
echo "=================================================================="

if [ ! -d "${DATA_CACHE_DIR}" ]; then
    echo "❌ ERROR: Dataset cache directory '${DATA_CACHE_DIR}' does not exist!"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

sync
sudo sysctl -w vm.drop_caches=3 2>/dev/null || true

# ------------------------------------------------------------------------------
# 4. Launch Execution
# ------------------------------------------------------------------------------
echo -e "\n🔥 Launching torchrun across ${NUM_GPUS} GPUs...\n"

R2_FLAG=""
if [ "${ENABLE_R2_SYNC}" = true ]; then
    R2_FLAG="--r2_sync"
fi

torchrun --nproc_per_node="${NUM_GPUS}" "${PYTHON_SCRIPT}" \
    --data_cache_dir "${DATA_CACHE_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --teacher_model_name "${TEACHER_MODEL}" \
    --layer_sequence "${LAYER_SEQUENCE}" \
    --d_model "${D_MODEL}" \
    --num_layers "${NUM_LAYERS}" \
    --num_heads "${NUM_HEADS}" \
    --chunk_size "${CHUNK_SIZE}" \
    --rank "${RANK}" \
    --seq_len "${SEQ_LEN}" \
    --max_steps "${MAX_STEPS}" \
    --phase1_steps "${PHASE1_STEPS}" \
    --learning_rate "${LEARNING_RATE}" \
    --backbone_lr_ratio "${BACKBONE_LR_RATIO}" \
    --emb_lr_ratio "${EMB_LR_RATIO}" \
    --warmup_steps "${WARMUP_STEPS}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --batch_size "${BATCH_SIZE}" \
    --grad_accum "${GRAD_ACCUM}" \
    --logging_steps "${LOGGING_STEPS}" \
    --save_steps "${SAVE_STEPS}" \
    --eval_steps "${EVAL_STEPS}" \
    --r2_bucket "${R2_BUCKET}" \
    --r2_prefix "${R2_PREFIX}" \
    --compile \
    ${R2_FLAG}

echo "=================================================================="
echo "🎉 20BT END-TO-END DISTILLATION COMPLETE!"
echo "   Consolidated model weights and tokenizer saved to '${OUTPUT_DIR}'"
echo "=================================================================="