#!/usr/bin/env bash
# ==============================================================================
# 🚀 BareTorch: Layerwise Tripartite Distillation Runner (10BT Production Run)
# ==============================================================================
# Description: 10BT Layerwise supervision runner across 8 H100 SXM5 GPUs.
#              - Phase 1 (5.0BT / 6,358 steps): Layerwise cosine alignment (Backbone frozen)
#              - Phase 2 (5.0BT / 6,358 steps): Global co-adaptation (Differential LRs)
#              - Loss: 0.35 CE + 0.35 Top-4 KL + 0.30 Layerwise Cosine Distance
#              - Automated Cloudflare R2 checkpoint synchronization enabled
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

# Path Directories (Updated for Layerwise Run)
DATA_CACHE_DIR="${PROJECT_ROOT}/data_20B/train"
OUTPUT_DIR="${PROJECT_ROOT}/qwen3.5_0.8B_clm_layerwise_10BT_lr3e-4"
TEACHER_MODEL="Qwen/Qwen3.5-0.8B"
LAYER_SEQUENCE="cs_lrad,cs_lrad,cs_lrad,transformer"

# Structural Dimensions (Matching Qwen 3.5 0.8B)
D_MODEL=1024
NUM_LAYERS=24
NUM_HEADS=16
CHUNK_SIZE=32
RANK=16
SEQ_LEN=2048

# Hardware Allocation (Effective Batch Size = 8 GPUs * 12 batch * 4 accum = 384 sequences)
# Tokens per step = 384 * 2048 = 786,432 tokens/step
BATCH_SIZE=12
GRAD_ACCUM=4

# Hyperparameters (10 Billion Tokens @ Effective Batch Size 384)
# Total Steps for 10B Tokens = 10,000,000,000 / 786,432 ≈ 12,716 steps
# Phase 1 Steps for 5B Tokens (50%) = 5,000,000,000 / 786,432 ≈ 6,358 steps
# Phase 2 Steps for 5B Tokens (50%) = 12,716 - 6,358 = 6,358 steps
MAX_STEPS=12716
PHASE1_STEPS=6358
WARMUP_STEPS=500
WEIGHT_DECAY="0.01"

# Tripartite Loss Weights
ALPHA_CE="0.35"
ALPHA_KL="0.35"
ALPHA_COS="0.30"

# Differential Learning Rates
LEARNING_RATE="3e-4"         # Base LR for CS-LRAD mixer layers (strong plasticity)
BACKBONE_LR_RATIO="0.033"    # 0.033 * 3e-4 = ~1e-5 for MLPs & native Attention (rigid anchor)
EMB_LR_RATIO="0.005"         # 0.005 * 3e-4 = ~1.5e-6 for Embeddings & LM Head (semi-frozen)

# Checkpoint & Eval Frequency
LOGGING_STEPS=200
SAVE_STEPS=1000
EVAL_STEPS=1000

# Cloudflare R2 Sync Settings
ENABLE_R2_SYNC=true
R2_BUCKET="baretorch-data"
R2_PREFIX="qwen3.5_0.8B_clm_layerwise_10BT"

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
echo "🚀 Initializing BareTorch Layerwise Distillation (8x H100)"
echo "=================================================================="
echo "Target Script          : ${PYTHON_SCRIPT}"
echo "Number of GPUs         : ${NUM_GPUS}x H100 SXM5"
echo "Teacher Model          : ${TEACHER_MODEL}"
echo "Model Dimensions       : d_model=${D_MODEL}, num_layers=${NUM_LAYERS}, num_heads=${NUM_HEADS}"
echo "CS-LRAD Configuration  : rank=${RANK}, chunk_size=${CHUNK_SIZE}, seq_len=${SEQ_LEN}"
echo "Dataset Directory      : ${DATA_CACHE_DIR}"
echo "Output Directory       : ${OUTPUT_DIR}"
echo "Batch Configuration    : ${NUM_GPUS} GPUs x ${BATCH_SIZE} batch x ${GRAD_ACCUM} accum"
echo "Tokens per Step        : ${TOKENS_PER_STEP} (786,432 tokens/step)"
echo "Total Training Steps   : ${MAX_STEPS} (~10 Billion Tokens)"
echo "Phase 1 Steps (CS-LRAD): 0 to ${PHASE1_STEPS} (~5 Billion Tokens, Backbone Frozen)"
echo "Phase 2 Steps (Global) : ${PHASE1_STEPS} to ${MAX_STEPS} (~5 Billion Tokens, Co-Adaptation)"
echo "Loss Weights           : CE=${ALPHA_CE}, KL=${ALPHA_KL}, Layerwise-Cos=${ALPHA_COS}"
echo "Base LR (CS-LRAD)      : ${LEARNING_RATE}"
echo "Backbone Peak LR       : $(python3 -c "print(${LEARNING_RATE} * ${BACKBONE_LR_RATIO})") (Ratio: ${BACKBONE_LR_RATIO})"
echo "Embeddings Peak LR     : $(python3 -c "print(${LEARNING_RATE} * ${EMB_LR_RATIO})") (Ratio: ${EMB_LR_RATIO})"
echo "Cloudflare R2 Sync     : ${ENABLE_R2_SYNC} (Target: r2://${R2_BUCKET}/${R2_PREFIX})"
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
    --alpha_ce "${ALPHA_CE}" \
    --alpha_kl "${ALPHA_KL}" \
    --alpha_cos "${ALPHA_COS}" \
    --compile \
    --tie_embeddings \
    ${R2_FLAG}

echo "=================================================================="
echo "🎉 10BT LAYERWISE DISTILLATION RUN COMPLETE!"
echo "   Consolidated model weights and tokenizer saved to '${OUTPUT_DIR}'"
echo "=================================================================="