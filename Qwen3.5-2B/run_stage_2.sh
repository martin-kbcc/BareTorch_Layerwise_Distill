#!/usr/bin/env bash
# ==============================================================================
# 🚀 BareTorch Stage 2: 3:1 Hybrid Layerwise Distillation Runner (Qwen3.5-2B)
# ==============================================================================
# Description: Extracts layerwise hidden states and trains CS-LRAD single-layer
#              sub-modules across 24 layers using 8 GPUs.
# ==============================================================================

set -euo pipefail

# Layer range arguments (0 to 23 for Qwen 3.5 2B)
START_LAYER=${1:-0}
END_LAYER=${2:-23}

NUM_GPUS=8

# Project Paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

INPUT_DIR="${PROJECT_ROOT}/data_100M/train"
FEATURES_DIR="${PROJECT_ROOT}/qwen3.5_2B_features_cache"
CHECKPOINTS_DIR="${PROJECT_ROOT}/qwen3.5_2B_checkpoints_layers"
TEACHER_MODEL="Qwen/Qwen3.5-2B"
TOTAL_TOKENS="100000000"

# Model Dimensions (Qwen 3.5 2B)
D_MODEL=2048
NUM_HEADS=16

# Extraction Parameters
EXTRACT_BATCH_SIZE=32

# Training Parameters (100M Tokens)
TRAIN_BATCH_SIZE=64
GRAD_ACCUM=1
LEARNING_RATE="1e-3"
MAX_STEPS=763         # 1 full epoch over 100M tokens @ batch_size 64
WARMUP_STEPS=75       # ~10% warmup
SAVE_STEPS=380        # Mid-point checkpoint
LOGGING_STEPS=50      # Step logging interval

# Cloudflare R2 Sync Settings
ENABLE_R2_SYNC=${ENABLE_R2_SYNC:-true}
R2_BUCKET=${R2_BUCKET:-"baretorch-data"}
R2_PREFIX=${R2_PREFIX:-"qwen3.5_2B_checkpoints_layers"}

echo "=================================================================="
echo "🚀 Starting 3:1 Hybrid Layerwise Distillation Stage 2 (24 Layers)"
echo "=================================================================="
echo "Project Root       : ${PROJECT_ROOT}"
echo "Number of GPUs     : ${NUM_GPUS}x H100"
echo "Target Layers      : ${START_LAYER} through ${END_LAYER}"
echo "Pattern            : 3 CS-LRAD layers -> 1 Transformer layer"
echo "Teacher Model      : ${TEACHER_MODEL}"
echo "Input Dataset      : ${INPUT_DIR}"
echo "Features Cache     : ${FEATURES_DIR}"
echo "Checkpoints Dir    : ${CHECKPOINTS_DIR}"
echo "Cloudflare R2 Sync : ${ENABLE_R2_SYNC}"
echo "=================================================================="

for (( LAYER=START_LAYER; LAYER<=END_LAYER; LAYER++ )); do
    FINAL_WEIGHT_PATH="${CHECKPOINTS_DIR}/layer_${LAYER}/cs_lrad_layer_${LAYER}.pt"
    LAYER_OUTPUT_DIR="${CHECKPOINTS_DIR}/layer_${LAYER}"

    echo ""
    echo "=================================================================="
    echo "⚡ PROCESSING LAYER ${LAYER} / ${END_LAYER}"
    echo "=================================================================="

    # Check if Layer is already processed
    if [ -f "$FINAL_WEIGHT_PATH" ]; then
        echo "⏭️  Layer ${LAYER} weights already exist at '${FINAL_WEIGHT_PATH}'."
        echo "   Skipping Layer ${LAYER}."
        continue
    fi

    # Check if this layer is a standard Transformer layer (indices 3, 7, 11, 15, 19, 23)
    if [ $(( (LAYER + 1) % 4 )) -eq 0 ]; then
        echo "🔄 Layer ${LAYER} is a standard TRANSFORMER block in the 3:1 hybrid design."
        echo "   Copying Qwen Teacher Layer ${LAYER} weights directly (skipping distillation training)..."

        mkdir -p "$LAYER_OUTPUT_DIR"
        
        python3 -c "
import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained('${TEACHER_MODEL}', torch_dtype=torch.bfloat16, trust_remote_code=True)
layer_weights = model.model.layers[${LAYER}].state_dict()
torch.save(layer_weights, '${FINAL_WEIGHT_PATH}')
print('   ✅ Saved Qwen Transformer weights for Layer ${LAYER}')
"
        if [ "${ENABLE_R2_SYNC}" = true ]; then
            echo "📤 Syncing Layer ${LAYER} Transformer weights to Cloudflare R2..."
            rclone copy "${LAYER_OUTPUT_DIR}" "r2:${R2_BUCKET}/${R2_PREFIX}/layer_${LAYER}" --transfers 8 --s3-chunk-size 64M || true
        fi

        echo "🎉 LAYER ${LAYER} (TRANSFORMER) PROCESSED!"
        continue
    fi

    # ------------------------------------------------------------------
    # Step 1: Feature Extraction for CS-LRAD Layer
    # ------------------------------------------------------------------
    echo "📥 Step 1/2: Extracting hidden states for CS-LRAD Layer ${LAYER} across ${NUM_GPUS} GPUs..."
    
    torchrun --nproc_per_node="${NUM_GPUS}" "${PROJECT_ROOT}/hidden_states_extraction.py" \
        --input_dir "${INPUT_DIR}" \
        --output_dir "${FEATURES_DIR}" \
        --checkpoints_dir "${CHECKPOINTS_DIR}" \
        --target_layer "${LAYER}" \
        --teacher_model_name "${TEACHER_MODEL}" \
        --total_tokens "${TOTAL_TOKENS}" \
        --attn_implementation sdpa \
        --batch_size "${EXTRACT_BATCH_SIZE}"

    echo "✅ Feature extraction complete for Layer ${LAYER}."

    # ------------------------------------------------------------------
    # Step 2: Single-Layer Feature Matching Training
    # ------------------------------------------------------------------
    echo "🏋 Step 2/2: Training CS-LRAD Layer ${LAYER} across ${NUM_GPUS} GPUs..."

    R2_FLAG=""
    if [ "${ENABLE_R2_SYNC}" = true ]; then
        R2_FLAG="--r2_sync"
    fi

    torchrun --nproc_per_node="${NUM_GPUS}" "${PROJECT_ROOT}/train_stage_1.py" \
        --target_layer "${LAYER}" \
        --features_dir "${FEATURES_DIR}" \
        --output_dir "${CHECKPOINTS_DIR}" \
        --teacher_model_name "${TEACHER_MODEL}" \
        --d_model "${D_MODEL}" \
        --num_heads "${NUM_HEADS}" \
        --batch_size "${TRAIN_BATCH_SIZE}" \
        --grad_accum "${GRAD_ACCUM}" \
        --learning_rate "${LEARNING_RATE}" \
        --warmup_steps "${WARMUP_STEPS}" \
        --max_steps "${MAX_STEPS}" \
        --save_steps "${SAVE_STEPS}" \
        --logging_steps "${LOGGING_STEPS}" \
        --r2_bucket "${R2_BUCKET}" \
        --r2_prefix "${R2_PREFIX}" \
        ${R2_FLAG}

    echo "✅ Training complete for Layer ${LAYER}."

    # ------------------------------------------------------------------
    # Step 3: Disk Space & System RAM Cache Cleanup
    # ------------------------------------------------------------------
    echo "🧹 Reclaiming disk space and purging host RAM cache for Layer ${LAYER}..."
    rm -rf "${FEATURES_DIR}/layer_${LAYER}"
    sync  # Flush pending write operations to NVMe
    
    sudo sysctl -w vm.drop_caches=3 2>/dev/null || true
    
    echo "   Reclaimed disk space and freed host RAM for Layer ${LAYER}."
    echo "🎉 LAYER ${LAYER} (CS-LRAD) DISTILLATION COMPLETE!"
done

echo ""
echo "=================================================================="
echo "🏆 STAGE 2 LAYERWISE DISTILLATION COMPLETE FOR ALL 24 LAYERS (${START_LAYER}..${END_LAYER})!"
echo "=================================================================="