# /home/martinkb/Desktop/BareTorch_Layerwise_Distill/train_stage3b_long_context_03b.py
import os
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.utils.data import Dataset as TorchDataset, DataLoader
from accelerate import Accelerator
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from datasets import load_from_disk

from baretorch.cs_lrad import LRADDecoderBlock
from init_student_00 import load_qwen3_5_text_teacher, load_hybrid_student_checkpoint

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 3B: Long-Context (32768-Token / 512-Chunk) CS-LRAD Recurrent Horizon Extension KD"
    )
    # Paths
    parser.add_argument("--teacher_id", type=str, default="Qwen/Qwen3.5-2B", help="Native Teacher HF ID")
    parser.add_argument("--stage3_ckpt_dir", type=str, default="./checkpoints/stage3_final_distilled/best", help="Path to Stage 3 best checkpoint")
    parser.add_argument("--data_dir", type=str, default="./data/fineweb_edu_packed_2048", help="Path to existing 2048-packed FineWeb-Edu dataset")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/stage3b_long_context_32k", help="Directory to save Stage 3B long-context checkpoints")

    # Long-Context & Differential LR Hyperparameters
    parser.add_argument("--target_seq_len", type=int, default=32768, help="Target long-context sequence length (multiple of 2048)")
    parser.add_argument("--late_token_weight", type=float, default=1.5, help="Focal multiplier on tokens at positions >= 2048")
    parser.add_argument("--lrad_lr", type=float, default=6.0e-5, help="Peak LR for CS-LRAD mixers & ln1 during horizon extension")
    parser.add_argument("--backbone_lr", type=float, default=8.0e-6, help="Gentle peak LR for Backbone MLPs & Full-Attn layers")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--micro_batch_size", type=int, default=1, help="Long-context sequences per GPU per step")
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=500, help="500 steps @ 262.1k tok/step = ~131.1M long-context tokens")
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # Loss Weights
    parser.add_argument("--kl_weight", type=float, default=0.5)
    parser.add_argument("--ce_weight", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--logit_chunk_size", type=int, default=256, help="Chunk size along sequence to keep 248K vocab logits small")

    # Logging, Evaluation & Saving
    parser.add_argument("--log_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--eval_batches", type=int, default=-1, help="-1 = evaluate entire validation split")
    parser.add_argument("--save_steps", type=int, default=100)

    # Hardware & Compilation
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--compile_mode", type=str, default="default", choices=["default", "reduce-overhead", "max-autotune"])
    parser.add_argument("--no_compile", action="store_true")
    parser.add_argument("--no_grad_checkpointing", action="store_true", help="Keep grad checkpointing enabled by default for 32K seq_len")
    parser.add_argument("--seed", type=int, default=45)
    return parser.parse_args()


class LongContextConcatenatedDataset(TorchDataset):
    """
    Zero-copy wrapper that concatenates `concat_factor` (e.g., 16x 2048 = 32768 tokens)
    blocks from the existing `fineweb_edu_packed_2048` Arrow dataset on the fly.
    """
    def __init__(self, base_dataset, target_seq_len: int = 32768):
        self.base_ds = base_dataset
        self.base_seq_len = len(base_dataset[0]["input_ids"])
        self.concat_factor = max(1, target_seq_len // self.base_seq_len)
        self.target_seq_len = self.concat_factor * self.base_seq_len
        self.num_long_samples = len(base_dataset) // self.concat_factor

    def __len__(self):
        return self.num_long_samples

    def __getitem__(self, idx):
        start_idx = idx * self.concat_factor
        long_ids = []
        for offset in range(self.concat_factor):
            long_ids.extend(self.base_ds[start_idx + offset]["input_ids"])
        return {"input_ids": long_ids}


def collate_long_packed(batch):
    input_ids = torch.tensor([item["input_ids"] for item in batch], dtype=torch.long)
    return {"input_ids": input_ids}


def chunked_long_context_kl_and_ce_loss(
    h_student: torch.Tensor,
    h_teacher: torch.Tensor,
    input_ids: torch.Tensor,
    student_lm_head: nn.Module,
    teacher_lm_head: nn.Module,
    base_short_len: int = 2048,
    late_token_weight: float = 1.5,
    temperature: float = 1.0,
    kl_weight: float = 0.5,
    ce_weight: float = 0.5,
    chunk_size: int = 256,
):
    s_dtype = student_lm_head.weight.dtype
    t_dtype = teacher_lm_head.weight.dtype

    h_s_shift = h_student[:, :-1, :].to(s_dtype)
    h_t_shift = h_teacher[:, :-1, :].detach().to(t_dtype)
    shift_labels = input_ids[:, 1:]

    B, L_minus_1, _ = h_s_shift.shape

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
    total_Weight_sum = 0.0

    early_ce_s_sum, early_ce_t_sum, early_toks = 0.0, 0.0, 0.0
    late_ce_s_sum, late_ce_t_sum, late_toks = 0.0, 0.0, 0.0

    for start in range(0, L_minus_1, chunk_size):
        end = min(start + chunk_size, L_minus_1)
        n_toks = float(end - start)
        is_late = start >= base_short_len
        pos_w = late_token_weight if is_late else 1.0
        chunk_w = n_toks * pos_w

        h_s_c = h_s_shift[:, start:end, :].contiguous()
        h_t_c = h_t_shift[:, start:end, :].contiguous()
        labels_c = shift_labels[:, start:end].contiguous()

        if h_student.requires_grad:
            kl_c, ce_s_c, ce_t_c = checkpoint.checkpoint(
                _chunk_loss_fn, h_s_c, h_t_c, labels_c, use_reentrant=False
            )
        else:
            kl_c, ce_s_c, ce_t_c = _chunk_loss_fn(h_s_c, h_t_c, labels_c)

        total_kl = total_kl + kl_c * chunk_w
        total_ce_s = total_ce_s + ce_s_c * chunk_w
        total_Weight_sum += chunk_w

        if is_late:
            late_ce_s_sum = late_ce_s_sum + ce_s_c.detach() * n_toks
            late_ce_t_sum = late_ce_t_sum + ce_t_c.detach() * n_toks
            late_toks += n_toks
        else:
            early_ce_s_sum = early_ce_s_sum + ce_s_c.detach() * n_toks
            early_ce_t_sum = early_ce_t_sum + ce_t_c.detach() * n_toks
            early_toks += n_toks

    total_kl = total_kl / max(total_Weight_sum, 1.0)
    total_ce_s = total_ce_s / max(total_Weight_sum, 1.0)
    combined_loss = kl_weight * total_kl + ce_weight * total_ce_s

    early_ce_s = early_ce_s_sum / max(early_toks, 1.0)
    late_ce_s = late_ce_s_sum / max(late_toks, 1.0)
    late_ce_t = late_ce_t_sum / max(late_toks, 1.0)

    return combined_loss, total_kl.detach(), early_ce_s, late_ce_s, late_ce_t


class Stage3BLongContextWrapper(nn.Module):
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
    lrad_params, backbone_params = [], []

    for layer in student_model.model.layers:
        if isinstance(layer, LRADDecoderBlock):
            for p in list(layer.ln1.parameters()) + list(layer.attn.parameters()):
                lrad_param_ids.add(id(p))
                lrad_params.append(p)

    seen_ids = set(lrad_param_ids)
    for _, p in student_model.named_parameters():
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
def evaluate_stage3b(accelerator, wrapper, teacher, t_backbone, val_loader, args, base_short_len):
    wrapper.eval()
    unwrapped_student = accelerator.unwrap_model(wrapper).student

    local_sums = torch.zeros(5, dtype=torch.float64, device=accelerator.device)
    local_count = torch.zeros(1, dtype=torch.float64, device=accelerator.device)

    for batch_idx, batch in enumerate(val_loader):
        if args.eval_batches > 0 and batch_idx >= args.eval_batches:
            break
        input_ids = batch["input_ids"].to(accelerator.device, non_blocking=True)

        t_out = t_backbone(input_ids=input_ids, attention_mask=None, output_hidden_states=False, use_cache=False)
        s_last_h = wrapper(input_ids)

        loss, kl_val, early_ce_s, late_ce_s, late_ce_t = chunked_long_context_kl_and_ce_loss(
            h_student=s_last_h,
            h_teacher=t_out.last_hidden_state,
            input_ids=input_ids,
            student_lm_head=unwrapped_student.lm_head,
            teacher_lm_head=teacher.lm_head,
            base_short_len=base_short_len,
            late_token_weight=args.late_token_weight,
            temperature=args.temperature,
            kl_weight=args.kl_weight,
            ce_weight=args.ce_weight,
            chunk_size=args.logit_chunk_size,
        )

        local_sums[0] += loss.detach().to(torch.float64)
        local_sums[1] += kl_val.to(torch.float64)
        local_sums[2] += early_ce_s.to(torch.float64)
        local_sums[3] += late_ce_s.to(torch.float64)
        local_sums[4] += late_ce_t.to(torch.float64)
        local_count[0] += 1.0

    global_sums = accelerator.reduce(local_sums, reduction="sum")
    global_count = accelerator.reduce(local_count, reduction="sum").clamp(min=1.0)
    means = (global_sums / global_count).tolist()

    wrapper.train()
    return means[0], means[1], means[2], means[3], means[4]


def save_student_checkpoint_clean(unwrapped_student, tokenizer, save_dir: str):
    """
    Saves the student checkpoint while ensuring `gradient_checkpointing` is persisted
    as `False` in `config.json` so downstream stages (4A, 4B, 4C) do not inherit an
    active gradient checkpointing flag when running with `--no_grad_checkpointing` + `torch.compile`.
    """
    os.makedirs(save_dir, exist_ok=True)
    orig_gc = getattr(unwrapped_student.config, "gradient_checkpointing", False)
    unwrapped_student.config.gradient_checkpointing = False
    unwrapped_student.config.save_pretrained(save_dir)
    unwrapped_student.config.gradient_checkpointing = orig_gc
    tokenizer.save_pretrained(save_dir)
    torch.save(unwrapped_student.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    use_gc = not args.no_grad_checkpointing
    # CRITICAL FIX: Never combine torch.compile inside LRADDecoderBlock with gradient_checkpointing_enable(),
    # because torch.utils.checkpoint (use_reentrant=False) saved-tensor metadata validation conflicts with
    # TorchDynamo graph caching during backward recomputation!
    use_compile = (not args.no_compile) and (not use_gc)

    if hasattr(torch._C, "_dynamo") and hasattr(torch._C._dynamo.eval_frame, "_set_lru_cache"):
        torch._C._dynamo.eval_frame._set_lru_cache(False)

    accelerator = Accelerator(mixed_precision="bf16", gradient_accumulation_steps=args.grad_accum_steps)

    ckpt_to_load = args.stage3_ckpt_dir
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
    student.config.max_seq_len = max(getattr(student.config, "max_seq_len", 8192), args.target_seq_len)

    if getattr(student.config, "tie_word_embeddings", True):
        student.lm_head.weight = student.model.token_embedding.weight

    if use_gc:
        student.gradient_checkpointing_enable()
    else:
        if hasattr(student, "gradient_checkpointing_disable"):
            student.gradient_checkpointing_disable()
        student.config.gradient_checkpointing = False
        if hasattr(student, "model"):
            student.model.gradient_checkpointing = False

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

    wrapper = Stage3BLongContextWrapper(student)

    ds_dict = load_from_disk(args.data_dir)
    base_short_len = len(ds_dict["train"][0]["input_ids"])
    train_ds = LongContextConcatenatedDataset(ds_dict["train"], target_seq_len=args.target_seq_len)
    val_ds = LongContextConcatenatedDataset(ds_dict["validation"], target_seq_len=args.target_seq_len)
    seq_len = train_ds.target_seq_len
    num_chunks = seq_len // student.config.chunk_size

    tokens_per_step = args.micro_batch_size * accelerator.num_processes * args.grad_accum_steps * seq_len

    if accelerator.is_main_process:
        print("=" * 88)
        print(f"STAGE 3B: LONG-CONTEXT ({seq_len} TOKENS / {num_chunks} CHUNKS) RECURRENT HORIZON EXTENSION")
        print(f"  - Loaded Stage 3 Ckpt  : {ckpt_to_load}")
        print(f"  - Long-Context Dataset : {len(train_ds):,} train seqs ({seq_len} tok) | {len(val_ds):,} val seqs")
        print(f"  - Late-Horizon Weight  : {args.late_token_weight}x on positions [{base_short_len}..{seq_len}]")
        print(f"  - Differential LRs     : CS-LRAD = {args.lrad_lr:.2e} | Backbone = {args.backbone_lr:.2e}")
        print(f"  - Gradient Checkpoint  : {'ENABLED (Eager Recompute Safe)' if use_gc else 'DISABLED'}")
        print(f"  - Global Batch Size    : {tokens_per_step:,} tokens/step ({args.micro_batch_size} micro * {accelerator.num_processes} GPUs * {args.grad_accum_steps} accum)")
        print(f"  - Total Token Budget   : {tokens_per_step * args.max_steps / 1e6:.2f}M tokens ({args.max_steps} steps)")
        print("=" * 88)

    train_loader = DataLoader(
        train_ds, batch_size=args.micro_batch_size, shuffle=True,
        collate_fn=collate_long_packed, num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.micro_batch_size, shuffle=False,
        collate_fn=collate_long_packed, num_workers=4, pin_memory=True, drop_last=True
    )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=int(args.warmup_ratio * args.max_steps), num_training_steps=args.max_steps
    )

    wrapper, optimizer, train_loader, val_loader = accelerator.prepare(
        wrapper, optimizer, train_loader, val_loader
    )
    wrapper.train()

    step = 0
    t0 = time.time()
    best_val_loss = float("inf")

    running_sums = torch.zeros(5, dtype=torch.float64, device=accelerator.device)
    running_count = torch.zeros(1, dtype=torch.float64, device=accelerator.device)

    while step < args.max_steps:
        for batch in train_loader:
            input_ids = batch["input_ids"].to(accelerator.device, non_blocking=True)

            with accelerator.accumulate(wrapper):
                with torch.no_grad():
                    t_out = t_backbone(
                        input_ids=input_ids, attention_mask=None, output_hidden_states=False, use_cache=False
                    )

                s_last_h = wrapper(input_ids)
                unwrapped_student = accelerator.unwrap_model(wrapper).student

                loss, kl_val, early_ce_s, late_ce_s, late_ce_t = chunked_long_context_kl_and_ce_loss(
                    h_student=s_last_h,
                    h_teacher=t_out.last_hidden_state,
                    input_ids=input_ids,
                    student_lm_head=unwrapped_student.lm_head,
                    teacher_lm_head=teacher.lm_head,
                    base_short_len=base_short_len,
                    late_token_weight=args.late_token_weight,
                    temperature=args.temperature,
                    kl_weight=args.kl_weight,
                    ce_weight=args.ce_weight,
                    chunk_size=args.logit_chunk_size,
                )

                running_sums[0] += loss.detach().to(torch.float64)
                running_sums[1] += kl_val.to(torch.float64)
                running_sums[2] += early_ce_s.to(torch.float64)
                running_sums[3] += late_ce_s.to(torch.float64)
                running_sums[4] += late_ce_t.to(torch.float64)
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

                if step == 1 or step % args.log_steps == 0:
                    g_sums = accelerator.reduce(running_sums, reduction="sum")
                    g_cnt = accelerator.reduce(running_count, reduction="sum").clamp(min=1.0)
                    avg_l, avg_kl, avg_early_ce, avg_late_ce, avg_late_t_ce = (g_sums / g_cnt).tolist()
                    running_sums.zero_()
                    running_count.zero_()

                    if accelerator.is_main_process:
                        elapsed = time.time() - t0
                        steps_logged = args.log_steps if step > 1 else 1
                        toks_per_sec = (steps_logged * tokens_per_step) / max(elapsed, 1e-5)
                        t0 = time.time()

                        lrs = scheduler.get_last_lr()
                        early_ppl = min(torch.tensor(avg_early_ce).exp().item(), 9999.0)
                        late_ppl = min(torch.tensor(avg_late_ce).exp().item(), 9999.0)
                        t_late_ppl = min(torch.tensor(avg_late_t_ce).exp().item(), 9999.0)
                        tok_str = "Step 1 DONE" if step == 1 else f"{toks_per_sec:,.0f} tok/s"

                        print(
                            f"[Stage 3B 32K] Step {step:04d}/{args.max_steps} | Loss: {avg_l:.4f} | KL: {avg_kl:.4f} | "
                            f"PPL@0-2K: {early_ppl:.2f} | PPL@2K-32K: {late_ppl:.2f} (vs T: {t_late_ppl:.2f}) | "
                            f"LR: {lrs[0]:.2e}/{lrs[1]:.2e} | {tok_str}"
                        )

                if step % args.eval_steps == 0 or step >= args.max_steps:
                    accelerator.wait_for_everyone()
                    val_loss, val_kl, val_early_ce, val_late_ce, val_late_t_ce = evaluate_stage3b(
                        accelerator, wrapper, teacher, t_backbone, val_loader, args, base_short_len
                    )
                    if accelerator.is_main_process:
                        v_early_ppl = min(torch.tensor(val_early_ce).exp().item(), 9999.0)
                        v_late_ppl = min(torch.tensor(val_late_ce).exp().item(), 9999.0)
                        v_t_ppl = min(torch.tensor(val_late_t_ce).exp().item(), 9999.0)
                        print(
                            f"  >>> [VAL 32K @ Step {step:04d}] Total: {val_loss:.4f} | KL: {val_kl:.4f} | "
                            f"Val PPL@0-2K: {v_early_ppl:.2f} | Val PPL@2K-32K: {v_late_ppl:.2f} (Teacher 2K-32K PPL: {v_t_ppl:.2f})"
                        )
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            best_dir = os.path.join(args.output_dir, "best")
                            unwrapped = accelerator.unwrap_model(wrapper).student
                            save_student_checkpoint_clean(unwrapped, tokenizer, best_dir)
                            print(f"  >>> New Best 32K-Context Val Loss ({best_val_loss:.4f})! Saved to {best_dir}")
                    accelerator.wait_for_everyone()
                    t0 = time.time()

                if step % args.save_steps == 0 or step >= args.max_steps:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        unwrapped = accelerator.unwrap_model(wrapper).student
                        save_student_checkpoint_clean(unwrapped, tokenizer, args.output_dir)
                        print(f"  --> Saved Latest Stage 3B checkpoint at step {step} to {args.output_dir}")
                    accelerator.wait_for_everyone()
                    t0 = time.time()

                if step >= args.max_steps:
                    break

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()