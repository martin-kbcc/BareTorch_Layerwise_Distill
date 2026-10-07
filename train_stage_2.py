# /home/martinkb/Desktop/BareTorch_Layerwise_Distill/train_layer_alignment_ddp.py
import sys
import os

# ------------------------------------------------------------------
# 0. System Path Injection (Guarantees local 'baretorch' module resolution)
# ------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
import gc
import glob
import logging
import random
import subprocess
from datetime import timedelta
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.serialization
import torch.distributed as dist
from torch.utils.data import Dataset
from huggingface_hub import snapshot_download
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

# Enable TF32 Tensor Core Acceleration
torch.set_float32_matmul_precision("high")
torch.backends.cuda.enable_cudnn_sdp(False)

# ------------------------------------------------------------------
# Target Multi-Domain Proportional Ratios (7 Domains)
# ------------------------------------------------------------------
TARGET_RATIOS = {
    "fineweb_edu_100bt": 0.32,  # 32% Educational Web Prose
    "stack_dedup":       0.24,  # 24% Code Syntax & Structure
    "dclm_100bt":        0.16,  # 16% Broad Web Crawl
    "finemath_4plus":    0.10,  # 10% Mathematical Reasoning
    "cosmopedia_v2":     0.10,  # 10% Synthetic Textbook Prose
    "finepdfs_100bt":    0.07,  #  7% Technical & Structured PDFs
    "openr1_math":       0.01,  #  1% High-Signal Synthetic CoT Math
}

# ------------------------------------------------------------------
# 1. PyTorch 2.6+ NumPy Serialization Allowlist & Patched Loader
# ------------------------------------------------------------------
safe_numpy_types = [np.dtype, np.ndarray]
for mod_path in [
    "numpy._core.multiarray",
    "numpy.core.multiarray",
    "numpy._core.numerictypes",
]:
    try:
        mod = __import__(mod_path, fromlist=["scalar", "_reconstruct"])
        if hasattr(mod, "scalar"):
            safe_numpy_types.append(mod.scalar)
        if hasattr(mod, "_reconstruct"):
            safe_numpy_types.append(mod._reconstruct)
    except (ImportError, AttributeError):
        pass

try:
    torch.serialization.add_safe_globals(safe_numpy_types)
except Exception:
    pass

_orig_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)


torch.load = _patched_torch_load

# BareTorch Architecture Imports
from baretorch.configuration_baretorch import BareTorchConfig
from baretorch.cs_lrad import LRADDecoderBlock
from baretorch.modeling_baretorch import BareTorchForCausalLM

# Logging Setup
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("fsspec").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)


# ==================================================================
# 2. Cloudflare R2 Background Checkpoint Sync Callback
# ==================================================================


class R2CheckpointCallback(TrainerCallback):
    """Syncs saved checkpoints asynchronously to Cloudflare R2 using rclone in the background."""

    def __init__(
        self,
        bucket_name: str = "baretorch-data",
        remote_name: str = "r2",
        prefix: str = "checkpoints_stage2",
    ):
        self.bucket_name = bucket_name
        self.remote_name = remote_name
        self.prefix = prefix.strip("/")

    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            checkpoint_dir = f"checkpoint-{state.global_step}"
            local_ckpt_path = os.path.join(args.output_dir, checkpoint_dir)

            if os.path.exists(local_ckpt_path):
                rel_output_dir = os.path.basename(os.path.normpath(args.output_dir))
                target_r2_path = (
                    f"{self.remote_name}:{self.bucket_name}/{self.prefix}/{rel_output_dir}/{checkpoint_dir}"
                )

                logger.info(
                    f"\n☁️ [R2 Sync] Uploading {checkpoint_dir} to Cloudflare R2 ({target_r2_path}) in background..."
                )
                cmd = [
                    "rclone",
                    "copy",
                    local_ckpt_path,
                    target_r2_path,
                    "--transfers",
                    "8",
                    "--s3-chunk-size",
                    "64M",
                ]
                subprocess.Popen(
                    cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )


# ==================================================================
# 3. Teacher Predictions Triplet Memory-Mapped Dataset & Proportional Splitter
# ==================================================================


class DistillMemmapDataset(Dataset):
    """Zero-copy memory-mapped dataset reading raw uint32 packed tokens & teacher triplet binaries with OOV sanitization."""

    def __init__(
        self,
        data_dir: str,
        seq_len: int = 2048,
        samples: list = None,
        vocab_size: int = 248320,
        eos_token_id: int = 248044,
    ):
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.eos_token_id = eos_token_id
        self.samples = samples if samples is not None else []

    def _get_worker_mmap(self, path, dtype, shape):
        """Worker-isolated, read-only lazy memory-mapping cache."""
        current_pid = os.getpid()

        if (
            not hasattr(self, "_worker_mmap_cache")
            or getattr(self, "_worker_pid", None) != current_pid
        ):
            self._worker_mmap_cache = {}
            self._worker_pid = current_pid

        if path not in self._worker_mmap_cache:
            m = np.memmap(path, dtype=dtype, mode="r")
            m.flags.writeable = False
            self._worker_mmap_cache[path] = m.reshape(shape)

        return self._worker_mmap_cache[path]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if len(self.samples) == 0:
            raise RuntimeError(f"Dataset in directory '{self.data_dir}' is empty.")

        s_idx = idx % len(self.samples)
        t_path, idx_path, val_path, s_i = self.samples[s_idx]

        tok_mmap = self._get_worker_mmap(t_path, np.uint32, (-1, self.seq_len))
        idx_mmap = self._get_worker_mmap(idx_path, np.uint32, (-1, self.seq_len, 4))
        val_mmap = self._get_worker_mmap(val_path, np.float16, (-1, self.seq_len, 4))

        raw_tokens = tok_mmap[s_i].astype(np.int64)
        raw_teacher_indices = idx_mmap[s_i].astype(np.int64)
        teacher_values = torch.from_numpy(val_mmap[s_i].astype(np.float32))

        # Sanitize out-of-vocabulary input tokens
        oov_tok_mask = (raw_tokens < 0) | (raw_tokens >= self.vocab_size)
        if oov_tok_mask.any():
            raw_tokens[oov_tok_mask] = self.eos_token_id

        # Sanitize out-of-vocabulary teacher indices
        raw_teacher_indices = np.clip(raw_teacher_indices, 0, self.vocab_size - 1)

        tokens = torch.from_numpy(raw_tokens)
        teacher_indices = torch.from_numpy(raw_teacher_indices)

        # Convert NaN/Inf teacher logit values to 0.0
        teacher_values = torch.nan_to_num(
            teacher_values, nan=0.0, posinf=0.0, neginf=0.0
        )

        return {
            "input_ids": tokens,
            "labels": tokens.clone(),
            "teacher_indices": teacher_indices,
            "teacher_values": teacher_values,
        }


def build_disjoint_train_val_datasets(
    data_dir: str,
    seq_len: int = 2048,
    max_val_seqs: int = 2441,  # ~5M tokens evaluation set
    vocab_size: int = 248320,
    eos_token_id: int = 248044,
    seed: int = 42,
):
    """Creates strictly disjoint train and validation datasets proportionally sampled across TARGET_RATIOS."""
    all_collected_by_domain = {}

    for ds_name, ratio in TARGET_RATIOS.items():
        ds_dir = os.path.join(data_dir, ds_name)
        token_files = sorted(
            glob.glob(os.path.join(ds_dir, "**/*_packed_tokens.bin"), recursive=True)
        )
        token_files = [f for f in token_files if not f.endswith(".corrupt")]

        ds_samples = []
        for t_path in token_files:
            prefix = t_path.replace("_packed_tokens.bin", "")
            idx_path = f"{prefix}_teacher_indices.bin"
            val_path = f"{prefix}_teacher_values.bin"

            if os.path.exists(idx_path) and os.path.exists(val_path):
                num_tokens = os.path.getsize(t_path) // 4
                num_seqs = num_tokens // seq_len
                for s_i in range(num_seqs):
                    ds_samples.append((t_path, idx_path, val_path, s_i))

        all_collected_by_domain[ds_name] = ds_samples

    # Compute maximum total sequence capacity governed by domain availability
    max_possible_total = float("inf")
    for ds_name, ratio in TARGET_RATIOS.items():
        avail = len(all_collected_by_domain[ds_name])
        if ratio > 0 and avail > 0:
            max_possible_total = min(max_possible_total, avail / ratio)
        elif avail == 0:
            max_possible_total = 0

    proportional_samples = []
    if max_possible_total > 0 and max_possible_total != float("inf"):
        for ds_name, ratio in TARGET_RATIOS.items():
            target_count = int(max_possible_total * ratio)
            ds_samples = all_collected_by_domain[ds_name]
            rng_ds = random.Random(seed)
            rng_ds.shuffle(ds_samples)
            proportional_samples.extend(ds_samples[:target_count])
    else:
        # Fallback: load all available sequence triplets if folder layout is non-standard
        for ds_samples in all_collected_by_domain.values():
            proportional_samples.extend(ds_samples)

    rng = random.Random(seed)
    rng.shuffle(proportional_samples)

    total_seqs = len(proportional_samples)
    val_count = min(total_seqs, max_val_seqs)

    val_samples = proportional_samples[:val_count]
    train_samples = proportional_samples[val_count:]

    train_ds = DistillMemmapDataset(
        data_dir,
        seq_len=seq_len,
        samples=train_samples,
        vocab_size=vocab_size,
        eos_token_id=eos_token_id,
    )
    val_ds = DistillMemmapDataset(
        data_dir,
        seq_len=seq_len,
        samples=val_samples,
        vocab_size=vocab_size,
        eos_token_id=eos_token_id,
    )

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        logger.info(
            f"Loaded proportional dataset pool from '{data_dir}': {len(train_ds):,} train sequences (~{(len(train_ds)*seq_len)/1e6:.1f}M tokens) | "
            f"{len(val_ds):,} val sequences (~{(len(val_ds)*seq_len)/1e6:.1f}M tokens)"
        )

    return train_ds, val_ds


# ==================================================================
# 4. Stage 2 Distill Trainer (50% CE + 50% Top-4 Teacher KL Loss)
# ==================================================================


class DistillTrainer(Trainer):
    """Custom Trainer implementing 50% Cross-Entropy + 50% Top-4 Teacher KL Distillation Loss."""

    def __init__(
        self,
        *args,
        alpha_ce=0.5,
        alpha_kl=0.5,
        temperature=1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.alpha_ce = alpha_ce
        self.alpha_kl = alpha_kl
        self.temperature = temperature

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        import torch.utils.checkpoint as cp

        input_ids = inputs["input_ids"]
        labels = inputs["labels"]
        teacher_indices = inputs["teacher_indices"]
        teacher_values = inputs["teacher_values"]

        # Forward pass through student base model to compute hidden states
        base_model = getattr(model, "model", model)
        if hasattr(model, "module"):
            base_model = getattr(model.module, "model", base_model)

        lm_head = getattr(model, "lm_head", None)
        if lm_head is None and hasattr(model, "module"):
            lm_head = getattr(model.module, "lm_head", None)

        outputs = base_model(input_ids=input_ids, output_hidden_states=True)
        last_hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]

        shift_hidden = last_hidden[:, :-1, :]
        shift_labels = labels[:, 1:]
        shift_teacher_indices = teacher_indices[:, :-1, :]
        shift_teacher_values = teacher_values[:, :-1, :]

        p_teacher_full = F.softmax(
            shift_teacher_values / self.temperature, dim=-1
        )

        B, L_shift, _ = shift_hidden.shape
        num_tokens = B * L_shift
        chunk_size = 512

        total_loss_ce = 0.0
        total_loss_kl = 0.0

        def compute_chunk(c_hid, c_lbl, c_t_idx, c_t_p):
            target_dtype = (
                lm_head.weight.dtype
                if hasattr(lm_head, "weight")
                else torch.bfloat16
            )
            c_hid_input = (
                c_hid.to(target_dtype)
                if c_hid.dtype != target_dtype
                else c_hid
            )

            c_log = lm_head(c_hid_input)
            ce = F.cross_entropy(
                c_log.reshape(-1, c_log.size(-1)),
                c_lbl.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
            lps = F.log_softmax(c_log.float() / self.temperature, dim=-1)
            st_top4 = torch.gather(lps, dim=-1, index=c_t_idx)
            kl = F.kl_div(st_top4, c_t_p.float(), reduction="sum")
            return ce, kl

        for c_start in range(0, L_shift, chunk_size):
            c_end = min(c_start + chunk_size, L_shift)

            c_hidden = shift_hidden[:, c_start:c_end, :]
            c_labels = shift_labels[:, c_start:c_end]
            c_teacher_idx = shift_teacher_indices[:, c_start:c_end, :]
            c_p_teacher = p_teacher_full[:, c_start:c_end, :]

            if c_hidden.requires_grad:
                c_loss_ce, c_loss_kl = cp.checkpoint(
                    compute_chunk,
                    c_hidden,
                    c_labels,
                    c_teacher_idx,
                    c_p_teacher,
                    use_reentrant=False,
                )
            else:
                c_loss_ce, c_loss_kl = compute_chunk(
                    c_hidden, c_labels, c_teacher_idx, c_p_teacher
                )

            total_loss_ce += c_loss_ce
            total_loss_kl += c_loss_kl

        loss_ce = total_loss_ce / num_tokens
        loss_kl = (total_loss_kl / num_tokens) * (self.temperature ** 2)
        total_loss = (self.alpha_ce * loss_ce) + (self.alpha_kl * loss_kl)

        return (total_loss, outputs) if return_outputs else total_loss


# ==================================================================
# 5. Parameter Unfreezing Helper & Model Assembly Helper
# ==================================================================


def unfreeze_full_model_for_stage2(model: nn.Module) -> int:
    """
    Full Model Unfreezing (Stage 2 Global Logit Alignment):
      Unfreezes 100% of model parameters:
        - Token Embeddings & LM Head
        - Native Teacher Transformer Layers
        - Gated SwiGLU MLPs across all layers
        - RMSNorms across all layers
        - CS-LRAD Attention Engines
    """
    unfrozen_params = 0
    for name, param in model.named_parameters():
        param.requires_grad = True
        unfrozen_params += param.numel()

    return unfrozen_params


def assemble_full_baretorch_model(
    config: BareTorchConfig,
    checkpoints_dir: str,
    teacher_model_name: str,
    device: torch.device,
) -> BareTorchForCausalLM:
    """Instantiates a full BareTorchForCausalLM without GPU VRAM spikes."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))

    if local_rank == 0:
        logger.info("=" * 70)
        logger.info(f"📦 Assembling Full {config.num_layers}-Layer BareTorch Model from Stage 1 Checkpoints")
        logger.info("=" * 70)

    # 1. Resolve local snapshot path on Rank 0 and broadcast
    local_snapshot_path = teacher_model_name
    if rank == 0:
        logger.info(f"📥 Verifying Teacher Model snapshot '{teacher_model_name}' on Rank 0...")
        local_snapshot_path = snapshot_download(repo_id=teacher_model_name)

    if dist.is_initialized():
        path_list = [local_snapshot_path] if rank == 0 else [None]
        dist.broadcast_object_list(path_list, src=0)
        local_snapshot_path = path_list[0]
        dist.barrier()

    if local_rank == 0:
        logger.info(f"📥 Extracting teacher weights from CPU snapshot '{local_snapshot_path}'...")

    # 2. Load teacher model on CPU first
    teacher_model = AutoModelForCausalLM.from_pretrained(
        local_snapshot_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )

    teacher_emb = teacher_model.get_input_embeddings().weight.data.clone()
    padded_vocab_size = teacher_emb.shape[0]

    if config.vocab_size != padded_vocab_size:
        if local_rank == 0:
            logger.info(
                f"🎯 Adjusting student vocab_size from {config.vocab_size} -> {padded_vocab_size}"
            )
        config.vocab_size = padded_vocab_size

    lm_head_weight = None
    if hasattr(teacher_model, "lm_head"):
        lm_head_weight = teacher_model.lm_head.weight.data.clone()

    norm_weight = None
    teacher_base = getattr(teacher_model, teacher_model.base_model_prefix, teacher_model)
    if hasattr(teacher_base, "norm"):
        norm_weight = teacher_base.norm.weight.data.clone()

    # Extract native teacher rotary_emb module
    teacher_rotary_emb = None
    if hasattr(teacher_base, "rotary_emb"):
        teacher_rotary_emb = teacher_base.rotary_emb
    elif hasattr(teacher_model, "rotary_emb"):
        teacher_rotary_emb = teacher_model.rotary_emb

    # Extract native teacher Transformer layers (layers 3, 7, 11, 15, 19, 23...)
    teacher_transformer_layers = {}
    for l_idx in range(config.num_layers):
        if (l_idx + 1) % 4 == 0:
            layer_mod = teacher_base.layers[l_idx]
            pt_path = os.path.join(checkpoints_dir, f"layer_{l_idx}", f"cs_lrad_layer_{l_idx}.pt")
            if os.path.exists(pt_path):
                st = torch.load(pt_path, map_location="cpu")
                layer_mod.load_state_dict(st)
            teacher_transformer_layers[l_idx] = layer_mod

    # 3. Clean teacher wrapper from system memory
    del teacher_model, teacher_base
    gc.collect()
    torch.cuda.empty_cache()

    if local_rank == 0:
        logger.info("⚡ Creating BareTorch Student Model directly in BF16 on GPU...")

    # 4. Construct student model directly in BF16
    torch.set_default_dtype(torch.bfloat16)
    model = BareTorchForCausalLM(config).to(device)
    torch.set_default_dtype(torch.float32)

    # Attach teacher rotary embedding module to student model
    if teacher_rotary_emb is not None:
        model.model.rotary_emb = teacher_rotary_emb.to(device)

    with torch.no_grad():
        model.model.token_embedding.weight.data.copy_(teacher_emb.to(device))

        if lm_head_weight is not None and hasattr(model, "lm_head"):
            model.lm_head.weight.data.copy_(lm_head_weight.to(device))
        elif config.tie_word_embeddings:
            model.lm_head.weight = model.model.token_embedding.weight

        if norm_weight is not None and hasattr(model.model, "final_layernorm"):
            model.model.final_layernorm.weight.data.copy_(norm_weight.to(device))

    # 5. Populate student layers sequentially
    for l_idx in range(config.num_layers):
        pt_path = os.path.join(
            checkpoints_dir, f"layer_{l_idx}", f"cs_lrad_layer_{l_idx}.pt"
        )

        if (l_idx + 1) % 4 == 0:
            # Replace placeholder with native Teacher Transformer layer
            transformer_block = teacher_transformer_layers[l_idx].to(device)
            model.model.layers[l_idx] = transformer_block
            if local_rank == 0:
                logger.info(
                    f"   ├─ Layer {l_idx:02d}/{config.num_layers - 1:02d}: Native Teacher Transformer (Loaded successfully)"
                )
        else:
            if not os.path.exists(pt_path):
                raise FileNotFoundError(
                    f"❌ Missing Stage 1 distilled checkpoint for Layer {l_idx} at: {pt_path}"
                )

            state_dict = torch.load(pt_path, map_location=device)

            layer_module = model.model.layers[l_idx]
            for key, param in state_dict.items():
                parts = key.split(".")
                submod = layer_module
                for p in parts[:-1]:
                    submod = getattr(submod, p, None)
                    if submod is None:
                        break

                if submod is not None:
                    attr_name = parts[-1]
                    target_tensor = getattr(submod, attr_name, None)
                    if (
                        target_tensor is not None
                        and target_tensor.shape != param.shape
                    ):
                        if (
                            isinstance(submod, nn.Linear)
                            and attr_name == "weight"
                        ):
                            in_f = param.shape[1]
                            out_f = param.shape[0]
                            has_bias = (
                                f"{'.'.join(parts[:-1])}.bias" in state_dict
                            )
                            new_linear = nn.Linear(
                                in_f,
                                out_f,
                                bias=has_bias,
                                device=device,
                                dtype=torch.bfloat16,
                            )

                            parent = layer_module
                            for p in parts[:-2]:
                                parent = getattr(parent, p)
                            setattr(parent, parts[-2], new_linear)

            layer_module.load_state_dict(state_dict)
            if local_rank == 0:
                logger.info(
                    f"   ├─ Layer {l_idx:02d}/{config.num_layers - 1:02d}: CS-LRAD (Loaded from '{pt_path}')"
                )

    del teacher_transformer_layers, teacher_emb, lm_head_weight, norm_weight
    gc.collect()
    torch.cuda.empty_cache()

    if config.use_grad_checkpointing:
        model.gradient_checkpointing_enable()

    if local_rank == 0:
        logger.info("✅ Model Assembly Complete! VRAM usage fully optimized.")

    return model


# ==================================================================
# 6. Main Executable
# ==================================================================


def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    if "RANK" in os.environ:
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend="nccl", device_id=device, timeout=timedelta(minutes=60)
            )

    parser = argparse.ArgumentParser(
        description="BareTorch Stage 2 CS-LRAD Full Unfrozen Alignment Engine (Top-4 Teacher KL + CE + Cosine Scheduler + 8-Bit AdamW + BF16)"
    )

    # Architecture & Checkpoint Paths
    parser.add_argument(
        "--layer_sequence",
        type=str,
        default="cs_lrad,cs_lrad,cs_lrad,transformer",
    )
    parser.add_argument(
        "--checkpoints_dir",
        type=str,
        default="./checkpoints_layers",
        help="Directory containing Stage 1 single-layer checkpoints.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints_stage2_alignment",
        help="Output directory for full model Stage 2 checkpoints.",
    )
    parser.add_argument(
        "--teacher_model_name", type=str, default="Qwen/Qwen3.5-0.8B"
    )
    parser.add_argument(
        "--data_cache_dir",
        type=str,
        default="./data_1B/train",
        help="Directory containing the 1B tokenized teacher prediction binary dataset.",
    )
    parser.add_argument(
        "--tie_embeddings", action="store_true", default=False
    )

    # Distillation Parameters
    parser.add_argument("--alpha_ce", type=float, default=0.5, help="Weight for Cross-Entropy loss.")
    parser.add_argument("--alpha_kl", type=float, default=0.5, help="Weight for Top-4 Teacher KL loss.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature scaling for KL loss.")

    # Optimization Parameters (Default: 15,258 steps ≈ 1 epoch over 1B tokens @ Effective Batch Size 32)
    parser.add_argument("--max_steps", type=int, default=15258)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=["cosine", "linear"],
        help="Learning rate scheduler type (default: cosine).",
    )
    parser.add_argument("--warmup_steps", type=int, default=300)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)

    # Structural Dimensions (Default matching Qwen 3.5 0.8B)
    parser.add_argument("--d_model", type=int, default=1024)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--num_layers", type=int, default=24)
    parser.add_argument("--chunk_size", type=int, default=32)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=2048)

    # Performance & R2 Sync Flags
    parser.add_argument(
        "--r2_sync",
        action="store_true",
        default=True,
        help="Enable background checkpoint syncing to Cloudflare R2.",
    )
    parser.add_argument(
        "--r2_bucket", type=str, default="baretorch-data"
    )
    parser.add_argument(
        "--r2_prefix", type=str, default="checkpoints_stage2"
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        default=True,
        help="Apply targeted torch.compile to CS-LRAD recurrent layers.",
    )
    parser.add_argument(
        "--grad_checkpointing",
        action="store_true",
        default=True,
        help="Enable Gradient Checkpointing to conserve VRAM.",
    )
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=500)

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.teacher_model_name, trust_remote_code=True
    )

    teacher_config = AutoConfig.from_pretrained(args.teacher_model_name, trust_remote_code=True)
    vocab_size = getattr(teacher_config, "vocab_size", len(tokenizer))

    eos_token_id = getattr(tokenizer, "eos_token_id", getattr(teacher_config, "eos_token_id", 248044))
    if isinstance(eos_token_id, (list, tuple)):
        eos_token_id = eos_token_id[0]

    # Build 3:1 Hybrid Layer Sequence
    raw_sequence = [
        s.strip().lower()
        for s in args.layer_sequence.split(",")
        if s.strip()
    ]
    layer_types = [
        raw_sequence[i % len(raw_sequence)] for i in range(args.num_layers)
    ]

    config = BareTorchConfig(
        vocab_size=vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        chunk_size=args.chunk_size,
        rank=args.rank,
        max_seq_len=args.seq_len,
        layer_types=layer_types,
        use_grad_checkpointing=args.grad_checkpointing,
        tie_word_embeddings=args.tie_embeddings,
    )

    # 1. Assemble Student Model from Stage 1 Checkpoints
    model = assemble_full_baretorch_model(
        config=config,
        checkpoints_dir=args.checkpoints_dir,
        teacher_model_name=args.teacher_model_name,
        device=device,
    )

    # 2. Stage 2 Full Model Unfreezing (100% Parameters Trainable)
    unfrozen_params = unfreeze_full_model_for_stage2(model)

    # 3. Targeted torch.compile for student LRADDecoderBlock modules
    if args.compile:
        compile_mode = "reduce-overhead" if args.grad_accum == 1 else "default"
        if local_rank == 0:
            logger.info(
                f"🔥 Compiling student LRADDecoderBlocks with PyTorch Inductor (mode='{compile_mode}')..."
            )
        compiled_blocks = 0
        for module in model.modules():
            if module.__class__.__name__ == "LRADDecoderBlock":
                module.forward = torch.compile(module.forward, mode=compile_mode)
                compiled_blocks += 1
        if local_rank == 0:
            logger.info(
                f"Successfully compiled {compiled_blocks} LRADDecoderBlock sub-module(s)."
            )

    unique_params_dict = {p.data_ptr(): p for p in model.parameters()}
    total_params = sum(p.numel() for p in unique_params_dict.values())
    trainable_params = sum(
        p.numel() for p in unique_params_dict.values() if p.requires_grad
    )

    if local_rank == 0:
        logger.info("=" * 70)
        logger.info("🤖 CS-LRAD Stage 2 Full Unfrozen Logit Distillation Alignment Ready")
        logger.info(
            f"   ├─ Loss Objective       : {args.alpha_ce*100:.0f}% CE + {args.alpha_kl*100:.0f}% Top-4 KL (Temp={args.temperature})"
        )
        logger.info(
            f"   ├─ Total Parameters     : {total_params / 1e9:.3f} Billion ({total_params:,})"
        )
        logger.info(
            f"   ├─ Trainable Parameters : {trainable_params / 1e9:.3f} Billion ({trainable_params:,}) (100% Unfrozen)"
        )
        logger.info(
            f"   ├─ Frozen Percentage    : 0.00%"
        )
        logger.info(
            f"   ├─ Dataset Location     : {args.data_cache_dir}"
        )
        logger.info(
            f"   └─ LR Schedule          : {args.scheduler.upper()} (Warmup={args.warmup_steps} steps)"
        )
        logger.info("=" * 70)

    # 4. Load 1B Token Disjoint Training & Evaluation Datasets proportionally sampled across TARGET_RATIOS
    train_dataset, val_dataset = build_disjoint_train_val_datasets(
        data_dir=args.data_cache_dir,
        seq_len=args.seq_len,
        max_val_seqs=2441,
        vocab_size=vocab_size,
        eos_token_id=eos_token_id,
    )

    # 5. Training Arguments with Cosine Decay and 8-Bit AdamW
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        optim="adamw_bnb_8bit",  # 8-Bit AdamW Optimizer
        lr_scheduler_type=args.scheduler,  # Cosine Decay
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        bf16=True,  # Pure BF16 Execution
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_total_limit=3,
        torch_compile=False,
        gradient_checkpointing=args.grad_checkpointing,
        ddp_find_unused_parameters=False,
        dataloader_num_workers=4,  
        dataloader_prefetch_factor=2,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
    )

    # 6. Cloudflare R2 Background Sync Callback Setup
    enable_r2_sync = (
        args.r2_sync
        or os.environ.get("R2_SYNC", "0").lower() in ("1", "true", "yes")
    )
    r2_bucket = os.environ.get("R2_BUCKET", args.r2_bucket)
    r2_prefix = os.environ.get("R2_PREFIX", args.r2_prefix)

    callbacks = []
    if enable_r2_sync:
        if local_rank == 0:
            logger.info(
                f"☁️ Cloudflare R2 Sync ACTIVATED. Target Bucket: '{r2_bucket}' | Prefix: '{r2_prefix}'"
            )
        callbacks.append(
            R2CheckpointCallback(bucket_name=r2_bucket, prefix=r2_prefix)
        )

    trainer = DistillTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=callbacks,
        alpha_ce=args.alpha_ce,
        alpha_kl=args.alpha_kl,
        temperature=args.temperature,
    )

    # 7. Checkpoint Resumption Logic
    checkpoint_to_resume = None
    if os.path.exists(training_args.output_dir):
        existing_checkpoints = [
            d
            for d in os.listdir(training_args.output_dir)
            if d.startswith("checkpoint-")
        ]
        if existing_checkpoints:
            checkpoint_to_resume = True
            if local_rank == 0:
                logger.info(
                    f"Found existing checkpoint in '{training_args.output_dir}'. Resuming Stage 2 alignment..."
                )

    if local_rank == 0:
        logger.info("🔥 Starting Stage 2: CS-LRAD Full Unfrozen Logit Distillation Alignment in DDP Mode...")

    trainer.train(resume_from_checkpoint=checkpoint_to_resume)

    # 8. Consolidated Model & Tokenizer Save
    if local_rank == 0:
        logger.info(f"💾 Saving final Stage 2 aligned model & tokenizer to '{args.output_dir}'...")
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        logger.info("✅ Stage 2: CS-LRAD Alignment completed successfully!")

        if enable_r2_sync:
            rel_output_dir = os.path.basename(os.path.normpath(args.output_dir))
            target_r2_path = f"r2:{r2_bucket}/{r2_prefix.strip('/')}/{rel_output_dir}"
            logger.info(f"📤 Syncing final aligned weights to Cloudflare R2 ({target_r2_path})...")
            cmd = [
                "rclone",
                "copy",
                args.output_dir,
                target_r2_path,
                "--transfers",
                "8",
                "--s3-chunk-size",
                "64M",
            ]
            subprocess.run(cmd, check=False)
            logger.info("✅ Final aligned weights uploaded to Cloudflare R2!")

    # 9. Clean Process Group Shutdown
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

    os._exit(0)


if __name__ == "__main__":
    main()