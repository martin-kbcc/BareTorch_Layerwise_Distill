#!/usr/bin/env bash
# ==============================================================================
# 🚀 BareTorch Stage 4: End-to-End Causal Language Modeling Alignment Runner (Qwen3.5-2B)
# ==============================================================================
# Description: Launches 24-layer global logit distillation alignment fine-tuning
#              across 8 H100 GPUs using torchrun DDP over the 5B dataset pool.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

# ------------------------------------------------------------------------------
# 1. Configurable Variables
# ------------------------------------------------------------------------------
NUM_GPUS=8
PYTHON_SCRIPT="${PROJECT_ROOT}/train_stage_2.py"

# Path Directories
DATA_CACHE_DIR="${PROJECT_ROOT}/data_5B/train"
CHECKPOINTS_DIR="${PROJECT_ROOT}/qwen3.5_2B_checkpoints_layers"
OUTPUT_DIR="${PROJECT_ROOT}/qwen3.5_2B_clm_checkpoints"
TEACHER_MODEL="Qwen/Qwen3.5-2B"
LAYER_SEQUENCE="cs_lrad,cs_lrad,cs_lrad,transformer"

# Structural Dimensions (Matching Qwen 3.5 2B)
D_MODEL=2048
NUM_LAYERS=24
NUM_HEADS=16

# Hyperparameters (5 Billion Tokens @ Effective Batch Size 128)
# 5B tokens / (8 GPUs * 16 batch_size * 1 grad_accum * 2048 seq_len = 262,144 tokens/step) ≈ 19,073 steps
MAX_STEPS=19073
LEARNING_RATE="1e-4"
SCHEDULER="cosine"
WARMUP_STEPS=500
WEIGHT_DECAY="0.01"

# Per-GPU Hardware Allocation (Total Effective Batch Size = 16 * 1 * 8 = 128)
BATCH_SIZE=16
GRAD_ACCUM=1

# Checkpoint & Eval Frequency
LOGGING_STEPS=250
SAVE_STEPS=1000
EVAL_STEPS=1000

# Cloudflare R2 Cloud Sync Settings
ENABLE_R2_SYNC=false
R2_BUCKET="baretorch-data"
R2_PREFIX="qwen3.5_2B_clm_checkpoints"

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
echo "🚀 Initializing BareTorch Stage 4 End-to-End Alignment Run (Qwen3.5-2B)"
echo "=================================================================="
echo "Number of GPUs         : ${NUM_GPUS}x H100"
echo "Teacher Model          : ${TEACHER_MODEL}"
echo "Model Dimensions       : d_model=${D_MODEL}, num_layers=${NUM_LAYERS}, num_heads=${NUM_HEADS}"
echo "Dataset Directory      : ${DATA_CACHE_DIR} (5B Tokens)"
echo "Stage 1 Layer Weights  : ${CHECKPOINTS_DIR}"
echo "Stage 2 Output Dir     : ${OUTPUT_DIR}"
echo "Effective Batch Size   : $(( BATCH_SIZE * GRAD_ACCUM * NUM_GPUS )) (${BATCH_SIZE} per GPU * ${GRAD_ACCUM} accum * ${NUM_GPUS} GPUs)"
echo "Tokens per Step        : $(( BATCH_SIZE * GRAD_ACCUM * NUM_GPUS * 2048 ))"
echo "Target Max Steps       : ${MAX_STEPS} (${SCHEDULER} decay)"
echo "Cloudflare R2 Sync     : ${ENABLE_R2_SYNC}"
echo "=================================================================="

if [ ! -d "${DATA_CACHE_DIR}" ]; then
    echo "❌ ERROR: Dataset cache directory '${DATA_CACHE_DIR}' does not exist!"
    echo "   Please ensure 'run_stage_3.sh' has finished downloading the binary shards."
    exit 1
fi

if [ ! -d "${CHECKPOINTS_DIR}" ]; then
    echo "❌ ERROR: Stage 1 checkpoints directory '${CHECKPOINTS_DIR}' does not exist!"
    echo "   Please run Stage 2 layerwise distillation first."
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

# Force memory cache flush before launching heavy model assembly
sync
sudo sysctl -w vm.drop_caches=3 2>/dev/null || true

# ------------------------------------------------------------------------------
# 4. Launch Stage 4 End-to-End Alignment Execution
# ------------------------------------------------------------------------------
echo -e "\n🔥 Launching torchrun across ${NUM_GPUS} GPUs...\n"

R2_FLAG=""
if [ "${ENABLE_R2_SYNC}" = true ]; then
    R2_FLAG="--r2_sync"
fi

torchrun --nproc_per_node="${NUM_GPUS}" "${PYTHON_SCRIPT}" \
    --data_cache_dir "${DATA_CACHE_DIR}" \
    --checkpoints_dir "${CHECKPOINTS_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --teacher_model_name "${TEACHER_MODEL}" \
    --layer_sequence "${LAYER_SEQUENCE}" \
    --d_model "${D_MODEL}" \
    --num_layers "${NUM_LAYERS}" \
    --num_heads "${NUM_HEADS}" \
    --max_steps "${MAX_STEPS}" \
    --learning_rate "${LEARNING_RATE}" \
    --scheduler "${SCHEDULER}" \
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
echo "🎉 STAGE 4 ALIGNMENT COMPLETE!"
echo "   Consolidated model weights and tokenizer saved to '${OUTPUT_DIR}'"
echo "=================================================================="