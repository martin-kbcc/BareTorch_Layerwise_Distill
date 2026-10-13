#!/usr/bin/env bash
# ==============================================================================
# 🚀 BareTorch: Direct End-to-End Distillation Runner (0.8B Model on 8x H100)
# ==============================================================================
# Description: Rapid 1BT calibration runner across 8 H100 SXM5 GPUs.
#              Directly warm-starts from Qwen3.5-0.8B via weight surgery:
#              - Phase 1 (200M tokens / 381 steps): Recurrent routing adaptation (Backbone frozen)
#              - Phase 2 (800M tokens / 1,527 steps): Global co-adaptation (Differential LRs)
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

# Prevent Linux open file descriptor exhaustion across DataLoader workers
ulimit -n 65536

# ------------------------------------------------------------------------------
# 1. Configurable Variables
# ------------------------------------------------------------------------------
NUM_GPUS=8
PYTHON_SCRIPT="${PROJECT_ROOT}/train_stage_1.py"

# Path Directories
DATA_CACHE_DIR="${PROJECT_ROOT}/data_1B/train"
OUTPUT_DIR="${PROJECT_ROOT}/qwen3.5_0.8B_clm_checkpoints"
TEACHER_MODEL="Qwen/Qwen3.5-0.8B"
LAYER_SEQUENCE="cs_lrad,cs_lrad,cs_lrad,transformer"

# Structural Dimensions (Matching Qwen 3.5 0.8B)
D_MODEL=1024
NUM_LAYERS=24
NUM_HEADS=16
CHUNK_SIZE=32
RANK=16
SEQ_LEN=2048

# Hardware Allocation (Effective Batch Size = 8 GPUs * 16 batch * 2 accum = 256 sequences)
# Tokens per step = 256 * 2048 = 524,288 tokens/step
BATCH_SIZE=16
GRAD_ACCUM=2

# Hyperparameters (1 Billion Tokens @ Effective Batch Size 256)
# Total Steps for 1B Tokens = 1,000,000,000 / 524,288 ≈ 1,908 steps
# Phase 1 Steps for 200M Tokens (20%) = 200,000,000 / 524,288 ≈ 381 steps
# Phase 2 Steps for 800M Tokens (80%) = 1,908 - 381 = 1,527 steps
MAX_STEPS=1908
PHASE1_STEPS=381
WARMUP_STEPS=150
WEIGHT_DECAY="0.01"

# Differential Learning Rates (Baseline Settings for Sweep)
LEARNING_RATE="2.5e-4"       # Base LR for CS-LRAD mixer layers
BACKBONE_LR_RATIO="0.08"     # 0.08 * 2.5e-4 = 2.0e-5 for MLPs & Attention
EMB_LR_RATIO="0.02"          # 0.02 * 2.5e-4 = 5.0e-6 for Embeddings & LM Head

# Checkpoint & Eval Frequency (Scaled for 1,908 total steps to monitor loss curve)
LOGGING_STEPS=25
SAVE_STEPS=250
EVAL_STEPS=250

# Cloudflare R2 Sync Settings
ENABLE_R2_SYNC=false
R2_BUCKET="baretorch-data"
R2_PREFIX="qwen3.5_0.8B_clm_checkpoints"

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
TOKENS_PER_STEP=$(( BATCH_SIZE * GRAD_ACCUM * NUM_GPUS * SEQ_LEN ))

echo "=================================================================="
echo "🚀 Initializing BareTorch 0.8B Calibration Run (8x H100)"
echo "=================================================================="
echo "Target Script          : ${PYTHON_SCRIPT}"
echo "Number of GPUs         : ${NUM_GPUS}x H100 SXM5"
echo "Teacher Model          : ${TEACHER_MODEL}"
echo "Model Dimensions       : d_model=${D_MODEL}, num_layers=${NUM_LAYERS}, num_heads=${NUM_HEADS}"
echo "CS-LRAD Configuration  : rank=${RANK}, chunk_size=${CHUNK_SIZE}, seq_len=${SEQ_LEN}"
echo "Dataset Directory      : ${DATA_CACHE_DIR}"
echo "Output Directory       : ${OUTPUT_DIR}"
echo "Batch Configuration    : ${NUM_GPUS} GPUs x ${BATCH_SIZE} batch x ${GRAD_ACCUM} accum"
echo "Tokens per Step        : ${TOKENS_PER_STEP} (524,288 tokens/step)"
echo "Total Training Steps   : ${MAX_STEPS} (~1 Billion Tokens)"
echo "Phase 1 Steps (CS-LRAD): 0 to ${PHASE1_STEPS} (~200 Million Tokens, Backbone Frozen)"
echo "Phase 2 Steps (Global) : ${PHASE1_STEPS} to ${MAX_STEPS} (~800 Million Tokens, Co-Adaptation)"
echo "Base LR (CS-LRAD)      : ${LEARNING_RATE}"
echo "Backbone Peak LR       : $(python3 -c "print(${LEARNING_RATE} * ${BACKBONE_LR_RATIO})") (Ratio: ${BACKBONE_LR_RATIO})"
echo "Embeddings Peak LR     : $(python3 -c "print(${LEARNING_RATE} * ${EMB_LR_RATIO})") (Ratio: ${EMB_LR_RATIO})"
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
echo "🎉 0.8B 1BT CALIBRATION DISTILLATION COMPLETE!"
echo "   Consolidated model weights and tokenizer saved to '${OUTPUT_DIR}'"
echo "=================================================================="