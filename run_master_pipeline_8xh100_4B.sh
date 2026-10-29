#!/usr/bin/env bash
# ==================================================================================================
# MASTER END-TO-END BARETORCH HYBRIDIZATION & UP-DISTILLATION ENGINE (8x NVIDIA H100 SXM5 - 80GB)
# - Scale Target                                  : BareTorch-Qwen3.5-4B-Hybrid (32 Layers, d=2560)
# - Schedule Profile                              : High-Precision Extended Horizon (+35-40% Steps, Gentle LRs)
# - Stage 0 + 0-Verify                            : Surgical Weight Transfer + 6-Point Verification
# - Stages 1, 2, 3 (2K Pretrain KD)               : Teacher = Qwen/Qwen3.5-4B (Native Subspace!)
# - Stage 3B (32K Long-Context Horizon Extension): Teacher = Qwen/Qwen3.5-4B (512-Chunk Calibration!)
# - Stage 4A (5-Pillar Focal SFT & KD)            : Teacher = Qwen/Qwen3.5-4B (Native Subspace!)
# - Stage 4B (DPO + Top-64 Skew-KD)               : Teacher = Qwen/Qwen3.5-9B -> Followed by lm_eval & 5-Task Gen!
# - Stage 4C (4-Pillar <think> Dr. GRPO + KD)     : Teacher = Qwen/Qwen3.5-9B (2100 Steps, G=12) -> Gen & lm_eval!
# ==================================================================================================
set -euo pipefail

export NCCL_NVLS_ENABLE=0
export NCCL_CUMEM_ENABLE=0

# --------------------------------------------------------------------------------------------------
# 1. Cluster Hardware & Model Topology Configuration (4B Student + 9B Up-Distill Teacher)
# --------------------------------------------------------------------------------------------------
NUM_GPUS=${NUM_GPUS:-8}
BASE_MODEL=${BASE_MODEL:-"Qwen/Qwen3.5-4B"}
POST_TEACHER=${POST_TEACHER:-"Qwen/Qwen3.5-9B"}
MODEL_TAG=${MODEL_TAG:-"4B_hybrid"}

# CS-LRAD Architecture Parameters for Qwen3.5-4B (32 layers, d_model=2560, 24 LRAD + 8 Full-Attn)
# rank=32 keeps low-rank recurrent state compact & fast while scaling capacity for d_model=2560
LRAD_CHUNK_SIZE=${LRAD_CHUNK_SIZE:-64}
LRAD_RANK=${LRAD_RANK:-32}

# High-speed NVMe storage root (defaults to /scratch)
STORAGE_ROOT=${STORAGE_ROOT:-"/scratch"}
DATA_ROOT="${STORAGE_ROOT}/data"
CKPT_ROOT="${STORAGE_ROOT}/checkpoints_${MODEL_TAG}"
LOG_DIR="${STORAGE_ROOT}/logs_${MODEL_TAG}"

mkdir -p "${DATA_ROOT}" "${CKPT_ROOT}" "${LOG_DIR}"

# H100 / NCCL / PyTorch Performance Environment Variables
export HF_HOME="${STORAGE_ROOT}/hf_cache"
export HF_DATASETS_CACHE="${STORAGE_ROOT}/hf_cache/datasets"
export TMPDIR="${STORAGE_ROOT}/tmp"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TMPDIR}"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM="false"
export OMP_NUM_THREADS=8
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Stage Checkpoint Paths (Isolated under /scratch/checkpoints_4B_hybrid)
CKPT_STAGE0="${CKPT_ROOT}/stage0_initialized"
CKPT_STAGE1="${CKPT_ROOT}/stage1_mixer_aligned"
CKPT_STAGE2="${CKPT_ROOT}/stage2_end2end_aligned"
CKPT_STAGE3="${CKPT_ROOT}/stage3_final_distilled"
CKPT_STAGE3B="${CKPT_ROOT}/stage3b_long_context_32k"
CKPT_STAGE4A="${CKPT_ROOT}/stage4a_5pillar_8k_4B"
CKPT_STAGE4B="${CKPT_ROOT}/stage4b_6pillar_dpo_9B"
CKPT_STAGE4C="${CKPT_ROOT}/stage4c_grpo_math_code_2100steps_9B"

# Shared Dataset Paths (Reuses existing tokenized FineWeb-Edu, 5-Pillar SFT & 6-Pillar DPO datasets!)
PRETRAIN_DATA_DIR="${DATA_ROOT}/fineweb_edu_packed_2048"
SFT_DATA_DIR="${DATA_ROOT}/stage4a_5pillar_8k_focal_2048"
DPO_DATA_DIR="${DATA_ROOT}/stage4b_6pillar_dpo_1024"

ACCEL_CMD="accelerate launch --multi_gpu --num_processes=${NUM_GPUS} --num_machines=1 --mixed_precision=bf16 --dynamo_backend=no"

echo "=================================================================================================="
echo " LAUNCHING BARETORCH 4B MASTER DISTILLATION ENGINE ON ${NUM_GPUS}x H100 (80GB) GPUs"
echo "  - Base Student & Stage 1-4A Teacher : ${BASE_MODEL} (CS-LRAD chunk=${LRAD_CHUNK_SIZE}, rank=${LRAD_RANK})"
echo "  - Stage 4B/4C Top-64 Skew-KD Teacher: ${POST_TEACHER}"
echo "  - Checkpoints Directory             : ${CKPT_ROOT}"
echo "  - Datasets Directory                : ${DATA_ROOT}"
echo "  - Logs Directory                    : ${LOG_DIR}"
echo "=================================================================================================="

PIPELINE_START=$(date +%s)

# --------------------------------------------------------------------------------------------------
# STAGE 0: Surgical Hybrid Initialization (init_student_00.py) + 6-Point Verification (verify_init.py)
# --------------------------------------------------------------------------------------------------
if [ ! -f "${CKPT_STAGE0}/pytorch_model.bin" ]; then
    echo -e "\n>>> [STAGE 0/8] Initializing BareTorch 4B Hybrid Student from ${BASE_MODEL}..."
    T0=$(date +%s)
    python init_student_00.py \
        --teacher_id "${BASE_MODEL}" \
        --save_dir "${CKPT_STAGE0}" \
        --chunk_size "${LRAD_CHUNK_SIZE}" \
        --rank "${LRAD_RANK}" \
        2>&1 | tee "${LOG_DIR}/00_init_student.log"
    echo ">>> [STAGE 0 DONE] Completed in $(($(date +%s) - T0))s."
else
    echo ">>> [STAGE 0 SKIP] Found existing ${CKPT_STAGE0}/pytorch_model.bin"
fi

if [ ! -f "${LOG_DIR}/00a_verify_init.done" ]; then
    echo -e "\n>>> [STAGE 0-VERIFY] Running 6-Point Surgical Initialization & Cache Verification..."
    python verify_init.py \
        --teacher_id "${BASE_MODEL}" \
        --ckpt_dir "${CKPT_STAGE0}" \
        --chunk_size "${LRAD_CHUNK_SIZE}" \
        --rank "${LRAD_RANK}" \
        2>&1 | tee "${LOG_DIR}/00a_verify_init.log"
    touch "${LOG_DIR}/00a_verify_init.done"
else
    echo ">>> [STAGE 0-VERIFY SKIP] Already verified ${CKPT_STAGE0}"
fi

# --------------------------------------------------------------------------------------------------
# STAGE 0B: Parallel Pretraining Dataset Preparation (prepare_dataset_00b.py)
# Expanded to 850M tokens so the +35-40% longer Stage 1-3B schedules see fresh tokens without over-cycling!
# --------------------------------------------------------------------------------------------------
if [ ! -d "${PRETRAIN_DATA_DIR}/train" ]; then
    echo -e "\n>>> [STAGE 0B/8] Streaming, Tokenizing & Packing 850M FineWeb-Edu Tokens (48 CPU workers)..."
    T0=$(date +%s)
    python prepare_dataset_00b.py \
        --tokenizer_path "${CKPT_STAGE0}" \
        --output_dir "${PRETRAIN_DATA_DIR}" \
        --cache_dir "${DATA_ROOT}/hf_cache_fineweb" \
        --seq_len 2048 \
        --target_tokens_millions 850.0 \
        --val_tokens_millions 5.0 \
        --num_proc 48 \
        2>&1 | tee "${LOG_DIR}/00b_prepare_dataset.log"
    echo ">>> [STAGE 0B DONE] Completed in $(($(date +%s) - T0))s."
else
    echo ">>> [STAGE 0B SKIP] Found existing packed dataset at ${PRETRAIN_DATA_DIR}"
fi

# --------------------------------------------------------------------------------------------------
# STAGE 1: Parallel Teacher-Forced Mixer Alignment (train_stage1_mixer_align_01.py)
# Teacher = Qwen/Qwen3.5-4B | 8 GPUs * 8 micro_batch * 1 accum * 2048 = 131,072 tokens/step * 1,400 steps (+40%)
# Lowered LR: 1.0e-3 -> 7.5e-4 for smoother subspace projection at d_model=2560
# --------------------------------------------------------------------------------------------------
if [ ! -f "${CKPT_STAGE1}/best/pytorch_model.bin" ]; then
    echo -e "\n>>> [STAGE 1/8] Running Parallel Teacher-Forced CS-LRAD Mixer Alignment (1400 steps, lr=7.5e-4)..."
    T0=$(date +%s)
    ${ACCEL_CMD} train_stage1_mixer_align_01.py \
        --teacher_id "${BASE_MODEL}" \
        --init_ckpt_dir "${CKPT_STAGE0}" \
        --data_dir "${PRETRAIN_DATA_DIR}" \
        --output_dir "${CKPT_STAGE1}" \
        --lr 7.5e-4 \
        --micro_batch_size 8 \
        --grad_accum_steps 1 \
        --max_steps 1400 \
        --log_steps 20 \
        --eval_steps 200 \
        --save_steps 700 \
        --num_workers 8 \
        2>&1 | tee "${LOG_DIR}/01_stage1_mixer_align.log"
    echo ">>> [STAGE 1 DONE] Completed in $(($(date +%s) - T0))s."
else
    echo ">>> [STAGE 1 SKIP] Found existing ${CKPT_STAGE1}/best/pytorch_model.bin"
fi

# --------------------------------------------------------------------------------------------------
# STAGE 2: End-to-End Hidden-State & Logit Distillation (train_stage2_end2end_02.py)
# Teacher = Qwen/Qwen3.5-4B | 8 GPUs * 4 micro_batch * 1 accum * 2048 = 65,536 tokens/step * 3,500 steps (+40%)
# Lowered LR: 2.5e-4 -> 1.8e-4 for tighter end-to-end inter-layer residual convergence
# --------------------------------------------------------------------------------------------------
if [ ! -f "${CKPT_STAGE2}/best/pytorch_model.bin" ]; then
    echo -e "\n>>> [STAGE 2/8] Running End-to-End Hidden-State & Logit KD (3500 steps, lr=1.8e-4)..."
    T0=$(date +%s)
    ${ACCEL_CMD} train_stage2_end2end_02.py \
        --teacher_id "${BASE_MODEL}" \
        --stage1_ckpt_dir "${CKPT_STAGE1}/best" \
        --data_dir "${PRETRAIN_DATA_DIR}" \
        --output_dir "${CKPT_STAGE2}" \
        --lr 1.8e-4 \
        --micro_batch_size 4 \
        --grad_accum_steps 1 \
        --max_steps 3500 \
        --logit_chunk_size 256 \
        --log_steps 25 \
        --eval_steps 250 \
        --save_steps 500 \
        --num_workers 8 \
        2>&1 | tee "${LOG_DIR}/02_stage2_end2end.log"
    echo ">>> [STAGE 2 DONE] Completed in $(($(date +%s) - T0))s."
else
    echo ">>> [STAGE 2 SKIP] Found existing ${CKPT_STAGE2}/best/pytorch_model.bin"
fi

# --------------------------------------------------------------------------------------------------
# STAGE 3: Full-Model Joint Knowledge Distillation (train_stage3_full_kd_03.py)
# Teacher = Qwen/Qwen3.5-4B | 8 GPUs * 2 micro_batch * 2 accum * 2048 = 65,536 tokens/step * 4,800 steps (+37%)
# Lowered Differential LRs: lrad_lr 1.2e-4 -> 9.0e-5 | backbone_lr 1.2e-5 -> 8.5e-6
# --------------------------------------------------------------------------------------------------
if [ ! -f "${CKPT_STAGE3}/best/pytorch_model.bin" ]; then
    echo -e "\n>>> [STAGE 3/8] Running Full-Model Joint KD with Differential LRs (4800 steps, lrad=9.0e-5, bb=8.5e-6)..."
    T0=$(date +%s)
    ${ACCEL_CMD} train_stage3_full_kd_03.py \
        --teacher_id "${BASE_MODEL}" \
        --stage2_ckpt_dir "${CKPT_STAGE2}/best" \
        --data_dir "${PRETRAIN_DATA_DIR}" \
        --output_dir "${CKPT_STAGE3}" \
        --lrad_lr 9.0e-5 \
        --backbone_lr 8.5e-6 \
        --micro_batch_size 2 \
        --grad_accum_steps 2 \
        --max_steps 4800 \
        --logit_chunk_size 128 \
        --log_steps 25 \
        --eval_steps 240 \
        --save_steps 600 \
        --num_workers 8 \
        --no_grad_checkpointing \
        2>&1 | tee "${LOG_DIR}/03_stage3_full_kd.log"
    echo ">>> [STAGE 3 DONE] Completed in $(($(date +%s) - T0))s."
else
    echo ">>> [STAGE 3 SKIP] Found existing ${CKPT_STAGE3}/best/pytorch_model.bin"
fi

# --------------------------------------------------------------------------------------------------
# STAGE 3B: Long-Context (32768-Token / 512-Chunk) Horizon Extension (train_stage3b_long_context_03b.py)
# Teacher = Qwen/Qwen3.5-4B | 8 GPUs * 1 micro_batch * 1 accum * 32768 = 262,144 tokens/step * 700 steps (+40%)
# Lowered Differential LRs: lrad_lr 5.0e-5 -> 3.8e-5 | backbone_lr 6.0e-6 -> 4.5e-6
# Note: Gradient checkpointing stays ENABLED here (no --no_grad_checkpointing flag) + logit_chunk_size=128!
# --------------------------------------------------------------------------------------------------
if [ ! -f "${CKPT_STAGE3B}/best/pytorch_model.bin" ]; then
    echo -e "\n>>> [STAGE 3B/8] Running 32768-Token (512-Chunk) Long-Context Horizon Extension (700 steps)..."
    T0=$(date +%s)
    ${ACCEL_CMD} train_stage3b_long_context_03b.py \
        --teacher_id "${BASE_MODEL}" \
        --stage3_ckpt_dir "${CKPT_STAGE3}/best" \
        --data_dir "${PRETRAIN_DATA_DIR}" \
        --output_dir "${CKPT_STAGE3B}" \
        --target_seq_len 32768 \
        --late_token_weight 1.5 \
        --lrad_lr 3.8e-5 \
        --backbone_lr 4.5e-6 \
        --micro_batch_size 1 \
        --grad_accum_steps 1 \
        --max_steps 700 \
        --logit_chunk_size 128 \
        --log_steps 10 \
        --eval_steps 100 \
        --save_steps 100 \
        --num_workers 8 \
        2>&1 | tee "${LOG_DIR}/03b_stage3b_long_context_32k.log"
    echo ">>> [STAGE 3B DONE] Completed in $(($(date +%s) - T0))s."
else
    echo ">>> [STAGE 3B SKIP] Found existing ${CKPT_STAGE3B}/best/pytorch_model.bin"
fi

# --------------------------------------------------------------------------------------------------
# STAGE 4A: 5-Pillar Focal SFT with NATIVE BASE_MODEL Teacher (train_stage4a_updistill_sft_05.py)
# Uses BASE_MODEL (Qwen/Qwen3.5-4B) so full-vocab 248K KL stays 100% aligned with tied LM head!
# 8 GPUs * 2 micro_batch * 2 accum * 2048 = 65,536 tokens/step * 5,500 steps (+37.5%) = 360.4M tokens
# Lowered Differential LRs: lrad_lr 6.0e-5 -> 4.5e-5 | backbone_lr 1.2e-5 -> 8.5e-6
# --------------------------------------------------------------------------------------------------
if [ ! -f "${CKPT_STAGE4A}/best/pytorch_model.bin" ]; then
    echo -e "\n>>> [STAGE 4A/8] Running 5-Pillar Focal SFT with Native ${BASE_MODEL} Teacher (5500 steps, lrad=4.5e-5, bb=8.5e-6)..."
    T0=$(date +%s)
    ${ACCEL_CMD} train_stage4a_updistill_sft_05.py \
        --teacher_id "${BASE_MODEL}" \
        --student_base_id "${BASE_MODEL}" \
        --stage3_ckpt_dir "${CKPT_STAGE3B}/best" \
        --pretrain_data_dir "${PRETRAIN_DATA_DIR}" \
        --sft_data_dir "${SFT_DATA_DIR}" \
        --cache_dir "${DATA_ROOT}/hf_cache_5pillar_8k_sft" \
        --output_dir "${CKPT_STAGE4A}" \
        --lrad_lr 4.5e-5 \
        --backbone_lr 8.5e-6 \
        --micro_batch_size 2 \
        --grad_accum_steps 2 \
        --max_steps 5500 \
        --logit_chunk_size 128 \
        --num_proc 32 \
        --num_workers 8 \
        --no_grad_checkpointing \
        2>&1 | tee "${LOG_DIR}/04a_stage4a_sft_4B.log"
    echo ">>> [STAGE 4A DONE] Completed in $(($(date +%s) - T0))s."
else
    echo ">>> [STAGE 4A SKIP] Found existing ${CKPT_STAGE4A}/best/pytorch_model.bin"
fi

# --------------------------------------------------------------------------------------------------
# STAGE 4B: 6-Pillar Contrastive DPO + DistiLLM-2 Top-64 Skew-KD from POST_TEACHER (Qwen/Qwen3.5-9B)
# Memory Footprint on 80GB H100: 9B Teacher (~18GB) + 4B Ref (~8GB) + 4B Student+Opt (~24GB) = ~50GB Static
# 8 GPUs * 1 micro_batch * 4 accum = 32 pairs/step * 1,750 steps (+40%) = 56,000 contrastive pairs!
# Lowered Differential LRs: lrad_lr 1.5e-5 -> 1.1e-5 | backbone_lr 4.0e-6 -> 2.8e-6
# --------------------------------------------------------------------------------------------------
if [ ! -f "${CKPT_STAGE4B}/best/pytorch_model.bin" ]; then
    echo -e "\n>>> [STAGE 4B/8] Running 6-Pillar Contrastive DPO + Top-64 Skew-KD from ${POST_TEACHER} (1750 steps, 32 pairs/step)..."
    T0=$(date +%s)
    ${ACCEL_CMD} train_stage4b_dpo_align_06.py \
        --teacher_id "${POST_TEACHER}" \
        --student_base_id "${BASE_MODEL}" \
        --stage4a_ckpt_dir "${CKPT_STAGE4A}/best" \
        --dpo_data_dir "${DPO_DATA_DIR}" \
        --cache_dir "${DATA_ROOT}/hf_cache_stage4b_6pillar_dpo" \
        --output_dir "${CKPT_STAGE4B}" \
        --lrad_lr 1.1e-5 \
        --backbone_lr 2.8e-6 \
        --micro_batch_size 1 \
        --grad_accum_steps 4 \
        --max_steps 1750 \
        --eval_steps 125 \
        --save_steps 250 \
        --logit_chunk_size 128 \
        --num_proc 32 \
        --num_workers 8 \
        2>&1 | tee "${LOG_DIR}/04b_stage4b_dpo_align_9B.log"
    echo ">>> [STAGE 4B DONE] Completed in $(($(date +%s) - T0))s."
else
    echo ">>> [STAGE 4B SKIP] Found existing ${CKPT_STAGE4B}/best/pytorch_model.bin"
fi

# --------------------------------------------------------------------------------------------------
# POST-STAGE 4B BENCHMARKS:
# 1. Full 10-Task Non-Generative lm_eval Benchmark Suite (Stage 4B vs. Base Qwen3.5-4B Teacher)
# 2. Pre-GRPO 5-Task Generative Baseline (GSM8K, MATH-500, GPQA, MBPP, HumanEval)
# --------------------------------------------------------------------------------------------------
if [ ! -f "${LOG_DIR}/lm_eval_stage4b_${MODEL_TAG}.json" ] && [ -f "run_lm_eval_benchmarks_04.py" ]; then
    echo -e "\n>>> [EVAL STAGE 4B - 1/2] Running 10-Task Non-Generative lm_eval Suite (Stage 4B vs. ${BASE_MODEL})..."
    python run_lm_eval_benchmarks_04.py \
        --teacher_id "${BASE_MODEL}" \
        --student_ckpt "${CKPT_STAGE4B}/best" \
        --tasks arc_easy,arc_challenge,piqa,winogrande,hellaswag,boolq,openbookqa,sciq,lambada_openai,mmlu \
        --num_fewshot 0 \
        --batch_size 32 \
        --device cuda:0 \
        --output_json "${LOG_DIR}/lm_eval_stage4b_${MODEL_TAG}.json" \
        2>&1 | tee "${LOG_DIR}/05a_eval_lm_eval_stage4b.log"
fi

if [ ! -f "${LOG_DIR}/generative_eval_stage4b_baseline_${MODEL_TAG}.json" ] && [ -f "run_generative_benchmarks_08.py" ]; then
    echo -e "\n>>> [EVAL STAGE 4B - 2/2] Running Pre-GRPO 5-Task Generative Suite (GSM8K, MATH-500, GPQA, MBPP, HumanEval)..."
    ${ACCEL_CMD} run_generative_benchmarks_08.py \
        --teacher_id "${BASE_MODEL}" \
        --student_base_id "${BASE_MODEL}" \
        --student_ckpt "${CKPT_STAGE4B}/best" \
        --tasks gsm8k,math500,gpqa,mbpp,humaneval \
        --batch_size 16 \
        --max_new_tokens 384 \
        --max_new_tokens_reasoning 512 \
        --max_samples 500 \
        --output_json "${LOG_DIR}/generative_eval_stage4b_baseline_${MODEL_TAG}.json" \
        2>&1 | tee "${LOG_DIR}/05b_eval_generative_stage4b_baseline.log"
fi

# --------------------------------------------------------------------------------------------------
# STAGE 4C: 4-Pillar <think>-Guided Dr. GRPO + Winner-Only Top-64 Skew-KD from POST_TEACHER (Qwen/Qwen3.5-9B)
# 8 GPUs * 1 accum * G=12 rollouts = 96 rollouts/step * 2,100 steps (+40%) = 201,600 verifiable on-policy rollouts!
# Lowered Differential LRs: lrad_lr 1.0e-5 -> 7.5e-6 | backbone_lr 2.5e-6 -> 1.8e-6
# (Gradient checkpointing enabled on 4B Student so 9B Teacher + 4B Ref + G=12 rollouts fit cleanly in 80GB)
# --------------------------------------------------------------------------------------------------
if [ ! -f "${CKPT_STAGE4C}/best/pytorch_model.bin" ]; then
    echo -e "\n>>> [STAGE 4C/8] Running 2100-Step 4-Pillar <think>-Guided Dr. GRPO + ${POST_TEACHER} Top-64 Skew-KD (G=12)..."
    T0=$(date +%s)
    ${ACCEL_CMD} train_stage4c_grpo_math_code_07.py \
        --teacher_id "${POST_TEACHER}" \
        --student_base_id "${BASE_MODEL}" \
        --input_ckpt_dir "${CKPT_STAGE4B}/best" \
        --output_dir "${CKPT_STAGE4C}" \
        --group_size 12 \
        --max_prompt_len 384 \
        --max_new_tokens 320 \
        --grad_accum_steps 1 \
        --lrad_lr 7.5e-6 \
        --backbone_lr 1.8e-6 \
        --grpo_weight 0.40 \
        --opd_skew_weight 0.40 \
        --gold_cot_weight 0.20 \
        --beta_ref_kl 0.05 \
        --max_steps 2100 \
        --logit_chunk_size 64 \
        --eval_steps 100 \
        --eval_prompts 100 \
        --save_steps 100 \
        2>&1 | tee "${LOG_DIR}/04c_stage4c_grpo_math_code_2100steps_9B.log"
    echo ">>> [STAGE 4C DONE] Completed in $(($(date +%s) - T0))s."
else
    echo ">>> [STAGE 4C SKIP] Found existing ${CKPT_STAGE4C}/best/pytorch_model.bin"
fi

# --------------------------------------------------------------------------------------------------
# POST-STAGE 4C BENCHMARKS:
# 1. 5-Task Generative Evaluation (GSM8K, MATH-500, GPQA, MBPP, HumanEval)
# 2. Final 10-Task Non-Generative lm_eval Verification on the Stage 4C Champion Checkpoint
# --------------------------------------------------------------------------------------------------
if [ -f "run_generative_benchmarks_08.py" ]; then
    echo -e "\n>>> [EVAL STAGE 4C - 1/2] Running Post-GRPO 5-Task Generative Suite (GSM8K, MATH-500, GPQA, MBPP, HumanEval)..."
    ${ACCEL_CMD} run_generative_benchmarks_08.py \
        --teacher_id "${BASE_MODEL}" \
        --student_base_id "${BASE_MODEL}" \
        --student_ckpt "${CKPT_STAGE4C}/best" \
        --tasks gsm8k,math500,gpqa,mbpp,humaneval \
        --batch_size 16 \
        --max_new_tokens 384 \
        --max_new_tokens_reasoning 512 \
        --max_samples 500 \
        --skip_teacher \
        --teacher_baseline_json "${LOG_DIR}/generative_eval_stage4b_baseline_${MODEL_TAG}.json" \
        --output_json "${LOG_DIR}/generative_eval_stage4c_2100steps_${MODEL_TAG}.json" \
        2>&1 | tee "${LOG_DIR}/06a_eval_generative_stage4c_2100steps.log"
fi

if [ -f "run_lm_eval_benchmarks_04.py" ]; then
    echo -e "\n>>> [EVAL STAGE 4C - 2/2] Running Final 10-Task Non-Generative lm_eval Suite (Stage 4C vs. ${BASE_MODEL})..."
    python run_lm_eval_benchmarks_04.py \
        --teacher_id "${BASE_MODEL}" \
        --student_ckpt "${CKPT_STAGE4C}/best" \
        --tasks arc_easy,arc_challenge,piqa,winogrande,hellaswag,boolq,openbookqa,sciq,lambada_openai,mmlu \
        --num_fewshot 0 \
        --batch_size 32 \
        --device cuda:0 \
        --output_json "${LOG_DIR}/lm_eval_stage4c_2100steps_${MODEL_TAG}.json" \
        2>&1 | tee "${LOG_DIR}/06b_eval_lm_eval_stage4c_2100steps.log"
fi

TOTAL_ELAPSED=$(($(date +%s) - PIPELINE_START))
echo "=================================================================================================="
echo " ALL BARETORCH 4B PIPELINE STAGES COMPLETED SUCCESSFULLY IN $((TOTAL_ELAPSED / 60))m $((TOTAL_ELAPSED % 60))s!"
echo "  - Stage 3B 32K Long-Context      : ${CKPT_STAGE3B}/best"
echo "  - Stage 4A SFT (4B Teacher)      : ${CKPT_STAGE4A}/best"
echo "  - Stage 4B DPO (9B Skew-KD)      : ${CKPT_STAGE4B}/best"
echo "  - Stage 4C GRPO (2100 Steps, G12): ${CKPT_STAGE4C}/best"
echo "  - Full Reports                   : ${LOG_DIR}/"
echo "=================================================================================================="