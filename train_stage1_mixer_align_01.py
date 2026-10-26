# /home/martinkb/Desktop/BareTorch_Layerwise_Distill/train_stage1_mixer_align_01.py
import os
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from datasets import load_from_disk

from baretorch.cs_lrad import LRADDecoderBlock
from init_student_00 import load_qwen3_5_text_teacher, load_hybrid_student_checkpoint

# Enable TF32 Tensor Cores on RTX 4090 for all internal FP32 matmuls in CS-LRAD chunk_forward
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 1: Parallel Teacher-Forced CS-LRAD Mixer Alignment")
    # Paths
    parser.add_argument("--teacher_id", type=str, default="Qwen/Qwen3.5-0.8B", help="HuggingFace teacher model ID")
    parser.add_argument("--init_ckpt_dir", type=str, default="./checkpoints/stage0_initialized", help="Path to Stage 0 student checkpoint")
    parser.add_argument("--data_dir", type=str, default="./data/fineweb_edu_packed_2048", help="Path to pre-tokenized packed dataset")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/stage1_mixer_aligned", help="Directory to save Stage 1 checkpoints")

    # Training Hyperparameters
    parser.add_argument("--lr", type=float, default=1e-3, help="Peak learning rate for CS-LRAD mixers")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="AdamW weight decay")
    parser.add_argument("--micro_batch_size", type=int, default=4, help="Batch size per GPU per step")
    parser.add_argument("--grad_accum_steps", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--max_steps", type=int, default=2500, help="Total optimizer steps to train")
    parser.add_argument("--warmup_ratio", type=float, default=0.05, help="Fraction of max_steps for linear warmup")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--cos_weight", type=float, default=0.1, help="Weight for cosine distance term alongside Normalized MSE")

    # Logging, Evaluation & Saving
    parser.add_argument("--log_steps", type=int, default=25, help="Print training metrics every N steps")
    parser.add_argument("--eval_steps", type=int, default=250, help="Run validation every N steps")
    parser.add_argument("--eval_batches", type=int, default=-1, help="Number of validation batches per GPU (-1 = entire validation dataset)")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every N steps")

    # Hardware & Compilation (torch.compile ON by default for CS-LRAD!)
    parser.add_argument("--num_workers", type=int, default=8, help="DataLoader CPU workers per GPU process")
    parser.add_argument("--compile_mode", type=str, default="default", choices=["default", "reduce-overhead", "max-autotune"], help="torch.compile mode for CS-LRAD layers")
    parser.add_argument("--no_compile", action="store_true", help="Disable torch.compile on CS-LRAD layers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for Stage 1 data shuffling")
    return parser.parse_args()


class TeacherMixerExtractor:
    """
    Registers forward hooks on the Teacher's linear-attention DecoderLayers to capture:
      1. layer_inputs[idx]:  h_in entering the teacher DecoderLayer (before input_layernorm)
      2. mixer_outputs[idx]: y_mixer output by the teacher's linear_attn module (before residual connection)
    """
    def __init__(self, teacher_backbone, lrad_layer_indices):
        self.layer_inputs = {}
        self.mixer_outputs = {}
        self.hooks = []

        for idx in lrad_layer_indices:
            t_layer = teacher_backbone.layers[idx]
            t_mixer = getattr(t_layer, "linear_attn", getattr(t_layer, "self_attn", None))
            if t_mixer is None:
                raise ValueError(f"Could not find linear_attn or self_attn in teacher layer {idx}")

            self.hooks.append(t_layer.register_forward_pre_hook(self._make_input_hook(idx)))
            self.hooks.append(t_mixer.register_forward_hook(self._make_output_hook(idx)))

    def _make_input_hook(self, layer_idx):
        def hook(module, args):
            self.layer_inputs[layer_idx] = args[0].detach()
        return hook

    def _make_output_hook(self, layer_idx):
        def hook(module, args, output):
            out_tensor = output[0] if isinstance(output, (tuple, list)) else output
            self.mixer_outputs[layer_idx] = out_tensor.detach()
        return hook

    def clear(self):
        self.layer_inputs.clear()
        self.mixer_outputs.clear()

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()


def compute_mixer_alignment_loss(y_out_s: torch.Tensor, y_out_t: torch.Tensor, cos_weight: float = 0.1):
    """Fused loss helper for Normalized MSE + Cosine Distance."""
    y_s = y_out_s.float()
    y_t = y_out_t.float()

    mse_num = (y_s - y_t).pow(2).mean(dim=-1)
    mse_den = y_t.pow(2).mean(dim=-1).clamp(min=1e-5)
    norm_mse = (mse_num / mse_den).mean()

    cos_dist = (1.0 - F.cosine_similarity(y_s, y_t, dim=-1)).mean()
    layer_loss = norm_mse + cos_weight * cos_dist
    return layer_loss, norm_mse.detach()


class Stage1ParallelMixerWrapper(nn.Module):
    """
    Lightweight DDP-compatible wrapper holding ONLY the 18 CS-LRAD blocks (ln1 + attn).
    """
    def __init__(self, student_model, cos_weight: float = 0.1, use_compile: bool = True, compile_mode: str = "default"):
        super().__init__()
        self.cos_weight = cos_weight
        self.lrad_indices = [
            i for i, layer in enumerate(student_model.model.layers)
            if isinstance(layer, LRADDecoderBlock)
        ]
        self.blocks = nn.ModuleDict({
            str(i): student_model.model.layers[i] for i in self.lrad_indices
        })

        for p in self.parameters():
            p.requires_grad = False
        for i in self.lrad_indices:
            for p in self.blocks[str(i)].attn.parameters():
                p.requires_grad = True

        # Compile CS-LRAD chunk_forward, ln1.forward, and the loss function into Triton kernels
        # Compiling the method (rather than wrapping the Module) keeps state_dict keys 100% clean!
        if use_compile:
            for i in self.lrad_indices:
                block = self.blocks[str(i)]
                block.ln1.forward = torch.compile(block.ln1.forward, mode=compile_mode, dynamic=False)
                block.attn.chunk_forward = torch.compile(block.attn.chunk_forward, mode=compile_mode, dynamic=False)
            self.loss_fn = torch.compile(compute_mixer_alignment_loss, mode=compile_mode, dynamic=False)
        else:
            self.loss_fn = compute_mixer_alignment_loss

    def forward(self, teacher_inputs_dict, teacher_outputs_dict):
        per_layer_mse = {}
        total_loss = 0.0

        for i in self.lrad_indices:
            block = self.blocks[str(i)]
            h_in_t = teacher_inputs_dict[i]
            y_out_t = teacher_outputs_dict[i]

            normed_in = block.ln1(h_in_t)
            y_out_s, _ = block.attn(normed_in, use_cache=False, attention_mask=None)

            layer_loss, norm_mse = self.loss_fn(y_out_s, y_out_t, self.cos_weight)
            per_layer_mse[i] = norm_mse
            total_loss = total_loss + layer_loss

        total_loss = total_loss / len(self.lrad_indices)
        return total_loss, per_layer_mse


@torch.no_grad()
def evaluate_stage1(accelerator, stage1_wrapper, t_backbone, extractor, val_loader, max_batches: int):
    """
    Evaluates across the validation dataset (all batches when max_batches <= 0)
    and reduces the exact mean total loss and per-layer MSE across all GPUs.
    """
    stage1_wrapper.eval()
    unwrapped = accelerator.unwrap_model(stage1_wrapper)
    lrad_indices = unwrapped.lrad_indices
    num_lrad = len(lrad_indices)

    # Index 0 holds total_loss; indices 1..num_lrad hold per-layer MSEs
    local_sums = torch.zeros(1 + num_lrad, dtype=torch.float64, device=accelerator.device)
    local_count = torch.zeros(1, dtype=torch.float64, device=accelerator.device)

    for batch_idx, batch in enumerate(val_loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        input_ids = batch["input_ids"].to(accelerator.device, non_blocking=True)
        extractor.clear()
        _ = t_backbone(input_ids=input_ids, attention_mask=None, use_cache=False)

        loss, per_layer_mse = stage1_wrapper(extractor.layer_inputs, extractor.mixer_outputs)
        local_sums[0] += loss.detach().to(torch.float64)
        for idx_pos, layer_idx in enumerate(lrad_indices):
            local_sums[1 + idx_pos] += per_layer_mse[layer_idx].to(torch.float64)
        local_count[0] += 1.0

    global_sums = accelerator.reduce(local_sums, reduction="sum")
    global_count = accelerator.reduce(local_count, reduction="sum").clamp(min=1.0)
    means = (global_sums / global_count).tolist()

    avg_loss = means[0]
    avg_layer_mse = {layer_idx: means[1 + idx_pos] for idx_pos, layer_idx in enumerate(lrad_indices)}

    stage1_wrapper.train()
    return avg_loss, avg_layer_mse


def collate_packed(batch):
    input_ids = torch.tensor([item["input_ids"] for item in batch], dtype=torch.long)
    return {"input_ids": input_ids}


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    use_compile = not args.no_compile

    accelerator = Accelerator(mixed_precision="bf16", gradient_accumulation_steps=args.grad_accum_steps)

    tokenizer = AutoTokenizer.from_pretrained(args.init_ckpt_dir, trust_remote_code=True)

    # Load frozen Teacher text backbone
    teacher, t_backbone, _ = load_qwen3_5_text_teacher(
        args.teacher_id, dtype=torch.bfloat16, device="cpu"
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # Reconstruct Student from Stage 0 checkpoint
    student = load_hybrid_student_checkpoint(
        args.init_ckpt_dir, teacher_backbone=t_backbone, teacher_id=args.teacher_id, dtype=torch.bfloat16
    )

    t_backbone.to(accelerator.device)
    student.to(accelerator.device)

    stage1_wrapper = Stage1ParallelMixerWrapper(
        student,
        cos_weight=args.cos_weight,
        use_compile=use_compile,
        compile_mode=args.compile_mode,
    )
    lrad_indices = stage1_wrapper.lrad_indices
    num_lrad = len(lrad_indices)
    extractor = TeacherMixerExtractor(t_backbone, lrad_indices)

    trainable_params = [p for p in stage1_wrapper.parameters() if p.requires_grad]

    # Load Pre-Tokenized Packed Dataset from Disk
    ds_dict = load_from_disk(args.data_dir)
    train_ds = ds_dict["train"]
    val_ds = ds_dict["validation"]
    seq_len = len(train_ds[0]["input_ids"])

    tokens_per_step = args.micro_batch_size * accelerator.num_processes * args.grad_accum_steps * seq_len

    if accelerator.is_main_process:
        num_trainable = sum(p.numel() for p in trainable_params) / 1e6
        val_scope = "ENTIRE validation set" if args.eval_batches <= 0 else f"{args.eval_batches * accelerator.num_processes} batches"
        print("=" * 80)
        print("STAGE 1: PARALLEL TEACHER-FORCED MIXER ALIGNMENT")
        print(f"  - Packed Dataset       : {args.data_dir} (seq_len={seq_len}, Zero Padding)")
        print(f"  - Trainable Parameters : {num_trainable:.2f}M ({num_lrad} CS-LRAD Mixers only)")
        print(f"  - torch.compile CS-LRAD: {'ENABLED (' + args.compile_mode + ')' if use_compile else 'DISABLED'}")
        print(f"  - TF32 Tensor Cores    : ENABLED (high precision matmul)")
        print(f"  - Global Batch Size    : {tokens_per_step:,} tokens/step ({args.micro_batch_size} micro * {accelerator.num_processes} GPUs * {args.grad_accum_steps} accum)")
        print(f"  - Train Log Averaging  : Global mean over {args.log_steps} steps ({args.log_steps * tokens_per_step:,} tokens)")
        print(f"  - Validation Scope     : {val_scope} ({len(val_ds) * seq_len / 1e6:.2f}M total val tokens)")
        print(f"  - Total Token Budget   : {tokens_per_step * args.max_steps / 1e6:.2f}M tokens ({args.max_steps} steps)")
        print(f"  - Peak Learning Rate   : {args.lr:.2e}")
        print("=" * 80)
        if use_compile:
            print("Note: Step 1 will take ~20 seconds while TorchInductor compiles fused Triton kernels for CS-LRAD...")

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
        drop_last=True,  # Keep static batch size during eval so compiled static shapes never recompile!
    )

    optimizer = torch.optim.AdamW(
        trainable_params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95), fused=True
    )
    # Do NOT pass scheduler to accelerator.prepare() so it steps strictly once per global optimizer step
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_ratio * args.max_steps),
        num_training_steps=args.max_steps,
    )

    stage1_wrapper, optimizer, train_loader, val_loader = accelerator.prepare(
        stage1_wrapper, optimizer, train_loader, val_loader
    )
    stage1_wrapper.train()

    step = 0
    t0 = time.time()
    best_val_loss = float("inf")

    # On-device accumulators for exact multi-GPU reduction across all micro-batches in log_steps
    # Index 0 holds total_loss; indices 1..num_lrad hold per-layer MSEs
    running_sums = torch.zeros(1 + num_lrad, dtype=torch.float64, device=accelerator.device)
    running_count = torch.zeros(1, dtype=torch.float64, device=accelerator.device)

    while step < args.max_steps:
        for batch in train_loader:
            input_ids = batch["input_ids"].to(accelerator.device, non_blocking=True)

            with accelerator.accumulate(stage1_wrapper):
                extractor.clear()
                with torch.no_grad():
                    _ = t_backbone(input_ids=input_ids, attention_mask=None, use_cache=False)

                loss, per_layer_mse = stage1_wrapper(extractor.layer_inputs, extractor.mixer_outputs)

                running_sums[0] += loss.detach().to(torch.float64)
                for idx_pos, layer_idx in enumerate(lrad_indices):
                    running_sums[1 + idx_pos] += per_layer_mse[layer_idx].to(torch.float64)
                running_count[0] += 1.0

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, max_norm=args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                scheduler.step()
                step += 1

                # Reset timer right after Step 1 finishes so compilation time doesn't skew tok/s!
                if step == 1:
                    t0 = time.time()

                # 1. Logging (Exact global mean across ALL micro-batches and ALL GPUs in the window!)
                if step == 1 or step % args.log_steps == 0:
                    global_sums = accelerator.reduce(running_sums, reduction="sum")
                    global_count = accelerator.reduce(running_count, reduction="sum").clamp(min=1.0)
                    means = (global_sums / global_count).tolist()

                    running_sums.zero_()
                    running_count.zero_()

                    if accelerator.is_main_process:
                        elapsed = time.time() - t0
                        steps_logged = args.log_steps if step > 1 else 1
                        toks_per_sec = (steps_logged * tokens_per_step) / max(elapsed, 1e-5)
                        t0 = time.time()

                        avg_loss = means[0]
                        avg_layer_mse = {layer_idx: means[1 + idx_pos] for idx_pos, layer_idx in enumerate(lrad_indices)}
                        mean_mse = sum(avg_layer_mse.values()) / len(avg_layer_mse)

                        l0 = lrad_indices[0]
                        l_mid = lrad_indices[len(lrad_indices) // 2]
                        l_last = lrad_indices[-1]
                        cur_lr = scheduler.get_last_lr()[0]

                        tok_str = "Compiling DONE" if step == 1 else f"{toks_per_sec:,.0f} tok/s"
                        print(
                            f"[Stage 1] Step {step:04d}/{args.max_steps} | Loss: {avg_loss:.4f} | "
                            f"MeanMSE: {mean_mse:.4f} | L{l0:02d}: {avg_layer_mse[l0]:.4f} | "
                            f"L{l_mid:02d}: {avg_layer_mse[l_mid]:.4f} | L{l_last:02d}: {avg_layer_mse[l_last]:.4f} | "
                            f"LR: {cur_lr:.2e} | {tok_str}"
                        )

                # 2. Validation Evaluation (Across the entire validation dataset and both GPUs)
                if step % args.eval_steps == 0 or step >= args.max_steps:
                    accelerator.wait_for_everyone()
                    val_loss, val_layer_mse = evaluate_stage1(
                        accelerator, stage1_wrapper, t_backbone, extractor, val_loader, args.eval_batches
                    )
                    if accelerator.is_main_process:
                        mean_mse = sum(val_layer_mse.values()) / len(val_layer_mse)
                        l0 = lrad_indices[0]
                        l_mid = lrad_indices[len(lrad_indices) // 2]
                        l_last = lrad_indices[-1]
                        print(
                            f"  >>> [VAL @ Step {step:04d}] Val Loss: {val_loss:.4f} | "
                            f"MeanMSE: {mean_mse:.4f} | L{l0:02d}: {val_layer_mse[l0]:.4f} | "
                            f"L{l_mid:02d}: {val_layer_mse[l_mid]:.4f} | L{l_last:02d}: {val_layer_mse[l_last]:.4f}"
                        )
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            best_dir = os.path.join(args.output_dir, "best")
                            os.makedirs(best_dir, exist_ok=True)
                            student.config.save_pretrained(best_dir)
                            tokenizer.save_pretrained(best_dir)
                            torch.save(student.state_dict(), os.path.join(best_dir, "pytorch_model.bin"))
                            print(f"  >>> New Best Val Loss ({best_val_loss:.4f})! Saved to {best_dir}")
                    accelerator.wait_for_everyone()
                    t0 = time.time()

                # 3. Periodic Checkpointing
                if step % args.save_steps == 0 or step >= args.max_steps:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        os.makedirs(args.output_dir, exist_ok=True)
                        student.config.save_pretrained(args.output_dir)
                        tokenizer.save_pretrained(args.output_dir)
                        torch.save(student.state_dict(), os.path.join(args.output_dir, "pytorch_model.bin"))
                        print(f"  --> Saved Latest Stage 1 checkpoint at step {step} to {args.output_dir}")
                    accelerator.wait_for_everyone()
                    t0 = time.time()

                if step >= args.max_steps:
                    break

    extractor.remove_hooks()
    # Clean distributed shutdown to prevent rank 1 exit SIGSEGV
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()