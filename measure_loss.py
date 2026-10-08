#!/usr/bin/env python3
# ==============================================================================
# 📊 BareTorch Model Loss Benchmarking & Distillation Gap Measurement Engine
# ==============================================================================

import os
import sys
import argparse
import glob
import logging
import math
import gc
import torch
import torch.nn.functional as F
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from baretorch.configuration_baretorch import BareTorchConfig
from baretorch.modeling_baretorch import BareTorchForCausalLM
from train_stage_2 import build_disjoint_train_val_datasets

torch.set_float32_matmul_precision("high")
torch.backends.cuda.enable_cudnn_sdp(False)

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)


def assemble_and_load_student(checkpoint_path: str, teacher_name: str, device: torch.device) -> torch.nn.Module:
    """
    Assembles 3:1 Hybrid BareTorch architecture (replacing layers 3, 7, 11, 15, 19, 23
    with native Qwen Transformer blocks) using the exact padded vocabulary size, then
    loads trained checkpoint weights.
    """
    logger.info("=" * 75)
    logger.info(f"📦 Assembling Hybrid BareTorch Architecture & Loading '{checkpoint_path}'...")
    logger.info("=" * 75)

    # 1. Load Teacher structure to extract native Transformer blocks & padded vocab size
    logger.info(f"📥 Loading base structure from '{teacher_name}'...")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        teacher_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    teacher_base = getattr(teacher_model, teacher_model.base_model_prefix, teacher_model)

    # Extract exact padded vocabulary size (248,320)
    padded_vocab_size = teacher_model.get_input_embeddings().weight.shape[0]
    logger.info(f"🎯 Detected padded vocab_size = {padded_vocab_size}")

    # 2. Define 3:1 Hybrid Config using padded_vocab_size
    config = BareTorchConfig(
        vocab_size=padded_vocab_size,
        d_model=1024,
        num_heads=16,
        num_layers=24,
        chunk_size=32,
        rank=8,
        max_seq_len=2048,
        layer_types=["cs_lrad", "cs_lrad", "cs_lrad", "transformer"] * 6,
    )

    # 3. Instantiate Student Model in BF16
    torch.set_default_dtype(torch.bfloat16)
    model = BareTorchForCausalLM(config)
    torch.set_default_dtype(torch.float32)

    # Attach rotary embedding module from teacher
    if hasattr(teacher_base, "rotary_emb"):
        model.model.rotary_emb = teacher_base.rotary_emb

    # 4. Swap in native Qwen Transformer blocks for indices 3, 7, 11, 15, 19, 23
    for l_idx in range(config.num_layers):
        if (l_idx + 1) % 4 == 0:
            model.model.layers[l_idx] = teacher_base.layers[l_idx]

    del teacher_model, teacher_base
    gc.collect()

    # 5. Load trained state dict from checkpoint
    weight_files = (
        glob.glob(os.path.join(checkpoint_path, "*.safetensors")) +
        glob.glob(os.path.join(checkpoint_path, "pytorch_model.bin")) +
        glob.glob(os.path.join(checkpoint_path, "model*.safetensors"))
    )

    if not weight_files:
        raise FileNotFoundError(f"❌ No weight tensors found in '{checkpoint_path}'")

    state_dict = {}
    for wf in sorted(weight_files):
        if wf.endswith(".safetensors"):
            from safetensors.torch import load_file
            state_dict.update(load_file(wf))
        else:
            state_dict.update(torch.load(wf, map_location="cpu"))

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    
    logger.info(f"✅ Hybrid Model Loaded Successfully! (Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)})")
    if len(missing) > 0 or len(unexpected) > 0:
        logger.info(f"  Missing: {missing[:3]}")
        logger.info(f"  Unexpected: {unexpected[:3]}")

    return model.to(device)


def compute_chunked_ce_loss(logits, labels, chunk_size=512):
    """Computes Cross-Entropy loss in sequential chunks to conserve GPU VRAM."""
    B, L, V = logits.shape
    total_loss = 0.0

    for c_start in range(0, L, chunk_size):
        c_end = min(c_start + chunk_size, L)
        c_logits = logits[:, c_start:c_end, :].contiguous().view(-1, V)
        c_labels = labels[:, c_start:c_end].contiguous().view(-1)

        ce_chunk = F.cross_entropy(c_logits, c_labels, ignore_index=-100, reduction="sum")
        total_loss += ce_chunk.item()

    return total_loss


def main():
    parser = argparse.ArgumentParser(description="BareTorch Loss Benchmarking Engine")
    parser.add_argument("--teacher_model_name", type=str, default="Qwen/Qwen3.5-0.8B")
    parser.add_argument(
        "--student_checkpoint",
        type=str,
        default=os.path.join(PROJECT_ROOT, "qwen3.5_0.8B_clm_checkpoints", "checkpoint-9000"),
    )
    parser.add_argument(
        "--data_cache_dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "data_1B", "train"),
    )
    parser.add_argument("--num_eval_seqs", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("=" * 75)
    logger.info("🧪 BARETORCH MODEL LOSS & DISTILLATION GAP EVALUATION")
    logger.info("=" * 75)
    logger.info(f"├─ Teacher Model       : {args.teacher_model_name}")
    logger.info(f"├─ Student Checkpoint  : {args.student_checkpoint}")
    logger.info(f"├─ Batch Size          : {args.batch_size}")
    logger.info(f"└─ Evaluation Seqs     : {args.num_eval_seqs:,} (~{(args.num_eval_seqs * args.seq_len)/1e6:.2f}M tokens)")
    logger.info("=" * 75)

    # 1. Load Evaluation Dataset
    teacher_config = AutoConfig.from_pretrained(args.teacher_model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model_name, trust_remote_code=True)
    eos_token_id = getattr(tokenizer, "eos_token_id", getattr(teacher_config, "eos_token_id", 248044))
    if isinstance(eos_token_id, (list, tuple)):
        eos_token_id = eos_token_id[0]

    _, val_dataset = build_disjoint_train_val_datasets(
        data_dir=args.data_cache_dir,
        seq_len=args.seq_len,
        max_val_seqs=args.num_eval_seqs,
        vocab_size=248320,
        eos_token_id=eos_token_id,
    )

    eval_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # 2. Load Teacher Model
    logger.info(f"📦 Loading Teacher Model ({args.teacher_model_name})...")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        args.teacher_model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    teacher_model.eval()

    # 3. Assemble & Load Hybrid Student Model
    student_model = assemble_and_load_student(
        checkpoint_path=args.student_checkpoint,
        teacher_name=args.teacher_model_name,
        device=device,
    )
    student_model.eval()

    total_teacher_ce = 0.0
    total_student_ce = 0.0
    total_kl_top4 = 0.0
    total_tokens = 0

    logger.info("\n🔥 Computing Cross-Entropy Loss & Top-4 KL Divergence across dataset...")

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating Batches"):
            input_ids = batch["input_ids"].to(device)
            teacher_indices = batch["teacher_indices"].to(device)
            teacher_values = batch["teacher_values"].to(device)

            labels = input_ids.clone()
            shift_labels = labels[:, 1:]
            shift_teacher_indices = teacher_indices[:, :-1, :]
            shift_teacher_values = teacher_values[:, :-1, :]

            # A. Teacher Loss
            teacher_out = teacher_model(input_ids=input_ids)
            shift_teacher_logits = teacher_out.logits[:, :-1, :]
            teacher_loss_ce_sum = compute_chunked_ce_loss(shift_teacher_logits, shift_labels)
            del teacher_out, shift_teacher_logits

            # B. Student Loss
            student_out = student_model(input_ids=input_ids)
            student_logits_raw = student_out.logits if hasattr(student_out, "logits") else student_out[0]
            shift_student_logits = student_logits_raw[:, :-1, :]
            student_loss_ce_sum = compute_chunked_ce_loss(shift_student_logits, shift_labels)

            # C. Top-4 KL Divergence
            p_teacher_top4 = F.softmax(shift_teacher_values / args.temperature, dim=-1)
            log_p_student = F.log_softmax(shift_student_logits.float() / args.temperature, dim=-1)
            student_top4_log_p = torch.gather(log_p_student, dim=-1, index=shift_teacher_indices)

            kl_div_batch = F.kl_div(student_top4_log_p, p_teacher_top4.float(), reduction="sum") * (args.temperature ** 2)

            num_batch_tokens = shift_labels.numel()
            total_teacher_ce += teacher_loss_ce_sum
            total_student_ce += student_loss_ce_sum
            total_kl_top4 += kl_div_batch.item()
            total_tokens += num_batch_tokens

            del student_out, student_logits_raw, shift_student_logits, log_p_student
            torch.cuda.empty_cache()

    avg_teacher_ce = total_teacher_ce / total_tokens
    avg_student_ce = total_student_ce / total_tokens
    avg_kl_top4 = total_kl_top4 / total_tokens

    teacher_ppl = math.exp(avg_teacher_ce)
    student_ppl = math.exp(avg_student_ce)
    ce_delta = avg_student_ce - avg_teacher_ce
    hybrid_loss = (0.5 * avg_student_ce) + (0.5 * avg_kl_top4)

    print("\n" + "=" * 75)
    print("📊 BARETORCH VS. QWEN TEACHER DISTILLATION GAP REPORT")
    print("=" * 75)
    print(f" Total Evaluated Tokens     : {total_tokens:,}")
    print(f" Vocabulary Size            : 248,320 tokens")
    print("-" * 75)
    print(f" Teacher Pure CE Loss       : {avg_teacher_ce:.4f}  (Perplexity: {teacher_ppl:.2f})")
    print(f" Student Pure CE Loss       : {avg_student_ce:.4f}  (Perplexity: {student_ppl:.2f})")
    print(f" Cross-Entropy Loss Gap Δ   : +{ce_delta:.4f} nats")
    print("-" * 75)
    print(f" Student Top-4 Teacher KL   : {avg_kl_top4:.4f}")
    print(f" Student Combined Hybrid    : {hybrid_loss:.4f}  (0.5*CE + 0.5*KL)")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()