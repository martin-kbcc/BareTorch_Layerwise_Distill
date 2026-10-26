# /home/martinkb/Desktop/BareTorch_Layerwise_Distill/train_stage3_full_kd_03.py
import os
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.utils.data import DataLoader
from accelerate import Accelerator
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from datasets import load_from_disk

from baretorch.cs_lrad import LRADDecoderBlock
from init_student_00 import load_qwen3_5_text_teacher, load_hybrid_student_checkpoint

# Enable TF32 Tensor Cores on RTX 4090
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 3: Full-Model Joint Knowledge Distillation with Differential LRs")
    # Paths
    parser.add_argument("--teacher_id", type=str, default="Qwen/Qwen3.5-0.8B", help="HuggingFace teacher model ID")
    parser.add_argument("--stage2_ckpt_dir", type=str, default="./checkpoints/stage2_end2end_aligned/best", help="Path to Stage 2 student checkpoint")
    parser.add_argument("--data_dir", type=str, default="./data/fineweb_edu_packed_2048", help="Path to pre-tokenized packed dataset")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/stage3_final_distilled", help="Directory to save Stage 3 final checkpoints")

    # Differential Learning Rates & Training Hyperparameters
    parser.add_argument("--lrad_lr", type=float, default=1.5e-4, help="Peak learning rate for CS-LRAD mixers & ln1")
    parser.add_argument("--backbone_lr", type=float, default=2.0e-5, help="Gentle peak learning rate for pretrained Qwen MLPs, Full-Attn, & Embeddings")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="AdamW weight decay")
    parser.add_argument("--micro_batch_size", type=int, default=2, help="Batch size per GPU per step")
    parser.add_argument("--grad_accum_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--max_steps", type=int, default=6000, help="Total optimizer steps (~196M tokens at 32k tok/step)")
    parser.add_argument("--warmup_ratio", type=float, default=0.05, help="Fraction of max_steps for linear warmup")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Gradient clipping norm")

    # Loss Weights
    parser.add_argument("--kl_weight", type=float, default=0.5, help="Weight for Teacher-Student Logit KL divergence")
    parser.add_argument("--ce_weight", type=float, default=0.5, help="Weight for ground-truth next-token Cross-Entropy loss")
    parser.add_argument("--temperature", type=float, default=1.0, help="Softmax temperature for Logit KL distillation")
    parser.add_argument("--logit_chunk_size", type=int, default=128, help="Sequence chunk size for memory-efficient 248k-vocab KL+CE")

    # Logging, Evaluation & Saving
    parser.add_argument("--log_steps", type=int, default=25, help="Print training metrics every N steps")
    parser.add_argument("--eval_steps", type=int, default=250, help="Run validation every N steps")
    parser.add_argument("--eval_batches", type=int, default=-1, help="Number of validation batches per GPU (-1 = entire validation dataset)")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every N steps")

    # Hardware & Compilation
    parser.add_argument("--num_workers", type=int, default=8, help="DataLoader CPU workers per GPU process")
    parser.add_argument("--compile_mode", type=str, default="default", choices=["default", "reduce-overhead", "max-autotune"])
    parser.add_argument("--no_compile", action="store_true", help="Disable torch.compile on CS-LRAD layers")
    parser.add_argument("--no_grad_checkpointing", action="store_true", help="Disable activation gradient checkpointing")
    parser.add_argument("--seed", type=int, default=44, help="Distinct random seed for Stage 3 data shuffling")
    return parser.parse_args()


def chunked_lm_head_kl_and_ce_loss(
    h_student: torch.Tensor,
    h_teacher: torch.Tensor,
    input_ids: torch.Tensor,
    student_lm_head: nn.Module,
    teacher_lm_head: nn.Module,
    temperature: float = 1.0,
    kl_weight: float = 0.5,
    ce_weight: float = 0.5,
    chunk_size: int = 128,
):
    s_dtype = student_lm_head.weight.dtype
    t_dtype = teacher_lm_head.weight.dtype

    h_s_shift = h_student[:, :-1, :].to(s_dtype)
    h_t_shift = h_teacher[:, :-1, :].detach().to(t_dtype)
    shift_labels = input_ids[:, 1:]

    B, L_minus_1, D = h_s_shift.shape

    def _chunk_loss_fn(h_s_c, h_t_c, labels_c):
        logits_s = student_lm_head(h_s_c.to(s_dtype)).float()
        with torch.no_grad():
            logits_t = teacher_lm_head(h_t_c.to(t_dtype)).float()
            probs_t = F.softmax(logits_t / temperature, dim=-1)
            ce_t = F.cross_entropy(logits_t.reshape(-1, logits_t.size(-1)), labels_c.reshape(-1))

        log_probs_s = F.log_softmax(logits_s / temperature, dim=-1)
        kl = F.kl_div(log_probs_s, probs_t, reduction="none").sum(dim=-1).mean() * (temperature ** 2)

        ce_s = F.cross_entropy(logits_s.reshape(-1, logits_s.size(-1)), labels_c.reshape(-1))
        return kl, ce_s, ce_t

    total_kl = 0.0
    total_ce_s = 0.0
    total_ce_t = 0.0

    for start in range(0, L_minus_1, chunk_size):
        end = min(start + chunk_size, L_minus_1)
        weight = (end - start) / float(L_minus_1)

        h_s_c = h_s_shift[:, start:end, :].contiguous()
        h_t_c = h_t_shift[:, start:end, :].contiguous()
        labels_c = shift_labels[:, start:end].contiguous()

        if h_student.requires_grad:
            kl_c, ce_s_c, ce_t_c = checkpoint.checkpoint(
                _chunk_loss_fn, h_s_c, h_t_c, labels_c, use_reentrant=False
            )
        else:
            kl_c, ce_s_c, ce_t_c = _chunk_loss_fn(h_s_c, h_t_c, labels_c)

        total_kl = total_kl + kl_c * weight
        total_ce_s = total_ce_s + ce_s_c * weight
        total_ce_t = total_ce_t + ce_t_c * weight

    combined_loss = kl_weight * total_kl + ce_weight * total_ce_s
    return combined_loss, total_kl.detach(), total_ce_s.detach(), total_ce_t.detach()


class Stage3FullDistillWrapper(nn.Module):
    def __init__(self, student_model):
        super().__init__()
        self.student = student_model

    def forward(self, input_ids):
        s_out = self.student.model(
            input_ids=input_ids,
            attention_mask=None,
            output_hidden_states=False,
            use_cache=False,
        )
        return s_out.last_hidden_state


def build_differential_optimizer(student_model, lrad_lr: float, backbone_lr: float, weight_decay: float):
    for p in student_model.parameters():
        p.requires_grad = True

    lrad_param_ids = set()
    lrad_params = []
    backbone_params = []

    for layer in student_model.model.layers:
        if isinstance(layer, LRADDecoderBlock):
            for p in list(layer.ln1.parameters()) + list(layer.attn.parameters()):
                lrad_param_ids.add(id(p))
                lrad_params.append(p)

    seen_ids = set(lrad_param_ids)
    for name, p in student_model.named_parameters():
        if id(p) not in seen_ids:
            seen_ids.add(id(p))
            backbone_params.append(p)

    optimizer = torch.optim.AdamW(
        [
            {"params": lrad_params, "lr": lrad_lr, "weight_decay": weight_decay},
            {"params": backbone_params, "lr": backbone_lr, "weight_decay": weight_decay},
        ],
        betas=(0.9, 0.95),
        fused=True,
    )
    return optimizer, lrad_params, backbone_params


@torch.no_grad()
def evaluate_stage3(accelerator, stage3_wrapper, teacher, t_backbone, val_loader, args):
    """
    Evaluates across the validation dataset (all batches when args.eval_batches <= 0)
    and reduces the exact mean across all GPUs.
    """
    stage3_wrapper.eval()
    unwrapped_student = accelerator.unwrap_model(stage3_wrapper).student

    local_sums = torch.zeros(4, dtype=torch.float64, device=accelerator.device)
    local_count = torch.zeros(1, dtype=torch.float64, device=accelerator.device)

    for batch_idx, batch in enumerate(val_loader):
        if args.eval_batches > 0 and batch_idx >= args.eval_batches:
            break
        input_ids = batch["input_ids"].to(accelerator.device, non_blocking=True)

        t_out = t_backbone(input_ids=input_ids, attention_mask=None, output_hidden_states=False, use_cache=False)
        s_last_h = stage3_wrapper(input_ids)

        loss, kl_val, ce_s_val, ce_t_val = chunked_lm_head_kl_and_ce_loss(
            h_student=s_last_h,
            h_teacher=t_out.last_hidden_state,
            input_ids=input_ids,
            student_lm_head=unwrapped_student.lm_head,
            teacher_lm_head=teacher.lm_head,
            temperature=args.temperature,
            kl_weight=args.kl_weight,
            ce_weight=args.ce_weight,
            chunk_size=args.logit_chunk_size,
        )

        local_sums[0] += loss.detach().to(torch.float64)
        local_sums[1] += kl_val.to(torch.float64)
        local_sums[2] += ce_s_val.to(torch.float64)
        local_sums[3] += ce_t_val.to(torch.float64)
        local_count[0] += 1.0

    # Sum across all GPUs and divide by total batches evaluated across all GPUs
    global_sums = accelerator.reduce(local_sums, reduction="sum")
    global_count = accelerator.reduce(local_count, reduction="sum").clamp(min=1.0)
    means = (global_sums / global_count).tolist()

    stage3_wrapper.train()
    return means[0], means[1], means[2], means[3]


def collate_packed(batch):
    input_ids = torch.tensor([item["input_ids"] for item in batch], dtype=torch.long)
    return {"input_ids": input_ids}


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    use_compile = not args.no_compile
    use_gc = not args.no_grad_checkpointing

    accelerator = Accelerator(mixed_precision="bf16", gradient_accumulation_steps=args.grad_accum_steps)

    ckpt_to_load = args.stage2_ckpt_dir
    if not os.path.exists(ckpt_to_load) and os.path.exists(os.path.dirname(ckpt_to_load)):
        ckpt_to_load = os.path.dirname(ckpt_to_load)

    tokenizer = AutoTokenizer.from_pretrained(ckpt_to_load, trust_remote_code=True)

    teacher, t_backbone, _ = load_qwen3_5_text_teacher(
        args.teacher_id, dtype=torch.bfloat16, device="cpu"
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    student = load_hybrid_student_checkpoint(
        ckpt_to_load, teacher_backbone=t_backbone, teacher_id=args.teacher_id, dtype=torch.bfloat16
    )

    if getattr(student.config, "tie_word_embeddings", True):
        student.lm_head.weight = student.model.token_embedding.weight

    if use_gc:
        student.gradient_checkpointing_enable()

    if use_compile:
        for layer in student.model.layers:
            if isinstance(layer, LRADDecoderBlock):
                layer.ln1.forward = torch.compile(layer.ln1.forward, mode=args.compile_mode, dynamic=False)
                layer.attn.chunk_forward = torch.compile(layer.attn.chunk_forward, mode=args.compile_mode, dynamic=False)
                layer.ln2.forward = torch.compile(layer.ln2.forward, mode=args.compile_mode, dynamic=False)

    t_backbone.to(accelerator.device)
    teacher.lm_head.to(accelerator.device)
    student.to(accelerator.device)

    optimizer, lrad_params, backbone_params = build_differential_optimizer(
        student, lrad_lr=args.lrad_lr, backbone_lr=args.backbone_lr, weight_decay=args.weight_decay
    )
    all_trainable_params = lrad_params + backbone_params

    stage3_wrapper = Stage3FullDistillWrapper(student)

    ds_dict = load_from_disk(args.data_dir)
    train_ds = ds_dict["train"]
    val_ds = ds_dict["validation"]
    seq_len = len(train_ds[0]["input_ids"])

    tokens_per_step = args.micro_batch_size * accelerator.num_processes * args.grad_accum_steps * seq_len

    if accelerator.is_main_process:
        n_lrad = sum(p.numel() for p in lrad_params) / 1e6
        n_back = sum(p.numel() for p in backbone_params) / 1e6
        val_scope = "ENTIRE validation set" if args.eval_batches <= 0 else f"{args.eval_batches * accelerator.num_processes} batches"
        print("=" * 80)
        print("STAGE 3: FULL-MODEL JOINT DISTILLATION (DIFFERENTIAL LEARNING RATES)")
        print(f"  - Loaded Stage 2 Ckpt  : {ckpt_to_load}")
        print(f"  - Group 1 (CS-LRAD)    : {n_lrad:.2f}M params @ Peak LR = {args.lrad_lr:.2e}")
        print(f"  - Group 2 (Backbone)   : {n_back:.2f}M params @ Peak LR = {args.backbone_lr:.2e}")
        print(f"  - Gradient Checkpoint  : {'ENABLED' if use_gc else 'DISABLED'}")
        print(f"  - Loss Formulation     : {args.kl_weight}*KL(T={args.temperature}) + {args.ce_weight}*CE")
        print(f"  - Global Batch Size    : {tokens_per_step:,} tokens/step ({args.micro_batch_size} micro * {accelerator.num_processes} GPUs * {args.grad_accum_steps} accum)")
        print(f"  - Train Log Averaging  : Global mean over {args.log_steps} steps ({args.log_steps * tokens_per_step:,} tokens)")
        print(f"  - Validation Scope     : {val_scope} ({len(val_ds) * seq_len / 1e6:.2f}M total val tokens)")
        print(f"  - Total Token Budget   : {tokens_per_step * args.max_steps / 1e6:.2f}M tokens ({args.max_steps} steps)")
        print("=" * 80)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.micro_batch_size,
        shuffle=True,
        collate_fn=collate_packed,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.micro_batch_size,
        shuffle=False,
        collate_fn=collate_packed,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_ratio * args.max_steps),
        num_training_steps=args.max_steps,
    )

    stage3_wrapper, optimizer, train_loader, val_loader = accelerator.prepare(
        stage3_wrapper, optimizer, train_loader, val_loader
    )
    stage3_wrapper.train()

    step = 0
    t0 = time.time()
    best_val_loss = float("inf")

    # On-device accumulators for exact multi-GPU reduction across all micro-batches in log_steps
    running_sums = torch.zeros(4, dtype=torch.float64, device=accelerator.device)
    running_count = torch.zeros(1, dtype=torch.float64, device=accelerator.device)

    while step < args.max_steps:
        for batch in train_loader:
            input_ids = batch["input_ids"].to(accelerator.device, non_blocking=True)

            with accelerator.accumulate(stage3_wrapper):
                with torch.no_grad():
                    t_out = t_backbone(
                        input_ids=input_ids,
                        attention_mask=None,
                        output_hidden_states=False,
                        use_cache=False,
                    )

                s_last_h = stage3_wrapper(input_ids)
                unwrapped_student = accelerator.unwrap_model(stage3_wrapper).student

                loss, kl_val, ce_s_val, ce_t_val = chunked_lm_head_kl_and_ce_loss(
                    h_student=s_last_h,
                    h_teacher=t_out.last_hidden_state,
                    input_ids=input_ids,
                    student_lm_head=unwrapped_student.lm_head,
                    teacher_lm_head=teacher.lm_head,
                    temperature=args.temperature,
                    kl_weight=args.kl_weight,
                    ce_weight=args.ce_weight,
                    chunk_size=args.logit_chunk_size,
                )

                running_sums[0] += loss.detach().to(torch.float64)
                running_sums[1] += kl_val.to(torch.float64)
                running_sums[2] += ce_s_val.to(torch.float64)
                running_sums[3] += ce_t_val.to(torch.float64)
                running_count[0] += 1.0

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(all_trainable_params, max_norm=args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                scheduler.step()
                step += 1

                if step == 1:
                    t0 = time.time()

                # 1. Logging (Exact global mean across ALL micro-batches and ALL GPUs in the window!)
                if step == 1 or step % args.log_steps == 0:
                    global_sums = accelerator.reduce(running_sums, reduction="sum")
                    global_count = accelerator.reduce(running_count, reduction="sum").clamp(min=1.0)
                    avg_loss, avg_kl, avg_ce_s, avg_ce_t = (global_sums / global_count).tolist()

                    running_sums.zero_()
                    running_count.zero_()

                    if accelerator.is_main_process:
                        elapsed = time.time() - t0
                        steps_logged = args.log_steps if step > 1 else 1
                        toks_per_sec = (steps_logged * tokens_per_step) / max(elapsed, 1e-5)
                        t0 = time.time()

                        lrs = scheduler.get_last_lr()
                        lr_lrad, lr_back = lrs[0], lrs[1]
                        s_ppl = min(torch.tensor(avg_ce_s).exp().item(), 9999.0)
                        t_ppl = min(torch.tensor(avg_ce_t).exp().item(), 9999.0)
                        tok_str = "Compiling DONE" if step == 1 else f"{toks_per_sec:,.0f} tok/s"

                        print(
                            f"[Stage 3] Step {step:04d}/{args.max_steps} | Total: {avg_loss:.4f} | "
                            f"KL: {avg_kl:.4f} | S_CE: {avg_ce_s:.4f} (PPL: {s_ppl:.2f} vs T_PPL: {t_ppl:.2f}) | "
                            f"LR_lrad: {lr_lrad:.2e} | LR_back: {lr_back:.2e} | {tok_str}"
                        )

                # 2. Validation Evaluation (Across the entire validation dataset and both GPUs)
                if step % args.eval_steps == 0 or step >= args.max_steps:
                    accelerator.wait_for_everyone()
                    val_loss, val_kl, val_ce_s, val_ce_t = evaluate_stage3(
                        accelerator, stage3_wrapper, teacher, t_backbone, val_loader, args
                    )
                    if accelerator.is_main_process:
                        val_s_ppl = min(torch.tensor(val_ce_s).exp().item(), 9999.0)
                        val_t_ppl = min(torch.tensor(val_ce_t).exp().item(), 9999.0)
                        print(
                            f"  >>> [VAL @ Step {step:04d}] Total: {val_loss:.4f} | KL: {val_kl:.4f} | "
                            f"Student CE: {val_ce_s:.4f} (Val PPL: {val_s_ppl:.2f}) | "
                            f"Teacher CE: {val_ce_t:.4f} (Teacher PPL: {val_t_ppl:.2f})"
                        )
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            best_dir = os.path.join(args.output_dir, "best")
                            os.makedirs(best_dir, exist_ok=True)
                            unwrapped = accelerator.unwrap_model(stage3_wrapper).student
                            unwrapped.config.save_pretrained(best_dir)
                            tokenizer.save_pretrained(best_dir)
                            torch.save(unwrapped.state_dict(), os.path.join(best_dir, "pytorch_model.bin"))
                            print(f"  >>> New Best Val Loss ({best_val_loss:.4f})! Saved to {best_dir}")
                    accelerator.wait_for_everyone()
                    t0 = time.time()

                # 3. Periodic Checkpointing
                if step % args.save_steps == 0 or step >= args.max_steps:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        os.makedirs(args.output_dir, exist_ok=True)
                        unwrapped = accelerator.unwrap_model(stage3_wrapper).student
                        unwrapped.config.save_pretrained(args.output_dir)
                        tokenizer.save_pretrained(args.output_dir)
                        torch.save(unwrapped.state_dict(), os.path.join(args.output_dir, "pytorch_model.bin"))
                        print(f"  --> Saved Latest Stage 3 checkpoint at step {step} to {args.output_dir}")
                    accelerator.wait_for_everyone()
                    t0 = time.time()

                if step >= args.max_steps:
                    break

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()