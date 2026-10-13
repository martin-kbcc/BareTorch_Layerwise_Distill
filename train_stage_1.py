# /home/martinkb/Desktop/BareTorch_Layerwise_Distill/train_stage_2.py
import argparse
import gc
import glob
import logging
import os
import random
import subprocess
import sys
from datetime import timedelta

# ------------------------------------------------------------------
# 0. System Path Injection (Guarantees local 'baretorch' module resolution)
# ------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.serialization
from huggingface_hub import snapshot_download
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

# Prevent file descriptor exhaustion across DataLoader workers
try:
    mp.set_sharing_strategy("file_system")
except RuntimeError:
    pass

# Enable TF32 Tensor Core Acceleration
torch.set_float32_matmul_precision("high")
torch.backends.cuda.enable_cudnn_sdp(False)

# ------------------------------------------------------------------
# Target Multi-Domain Proportional Ratios (7 Domains)
# ------------------------------------------------------------------
TARGET_RATIOS = {
    "fineweb_edu_100bt": 0.32,  # 32% Educational Web Prose
    "stack_dedup": 0.24,        # 24% Code Syntax & Structure
    "dclm_100bt": 0.16,         # 16% Broad Web Crawl
    "finemath_4plus": 0.10,     # 10% Mathematical Reasoning
    "cosmopedia_v2": 0.10,      # 10% Synthetic Textbook Prose
    "finepdfs_100bt": 0.07,     #  7% Technical & Structured PDFs
    "openr1_math": 0.01,        #  1% High-Signal Synthetic CoT Math
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
# 3. Direct Model Surgery & Warm-Start Initializer (No Stage 1 Files)
# ==================================================================


def safe_copy_linear(target_module, target_attr, source_linear):
    """Safely copies linear layer weights and biases, handling shape alignment."""
    target_layer = getattr(target_module, target_attr, None)
    if target_layer is None or source_linear is None:
        return

    source_weight = source_linear.weight
    has_bias = source_linear.bias is not None

    if target_layer.weight.shape != source_weight.shape:
        new_linear = nn.Linear(
            source_linear.in_features,
            source_linear.out_features,
            bias=has_bias,
            device=source_weight.device,
            dtype=source_weight.dtype,
        )
        setattr(target_module, target_attr, new_linear)
        target_layer = new_linear

    with torch.no_grad():
        target_layer.weight.copy_(source_weight)
        if has_bias and target_layer.bias is not None:
            target_layer.bias.copy_(source_linear.bias)


def warm_start_cslrad_layer(teacher_layer, student_block: LRADDecoderBlock, rank=8):
    """
    Transfers matching projections from teacher layer to LRADDecoderBlock.
    Applies mathematically correct SVD projection (U * sqrt(S) and V * sqrt(S)).
    """
    with torch.no_grad():
        # 1. Copy RMSNorms
        if hasattr(student_block, "ln1") and hasattr(teacher_layer, "input_layernorm"):
            student_block.ln1.weight.copy_(teacher_layer.input_layernorm.weight)
        if hasattr(student_block, "ln2") and hasattr(teacher_layer, "post_attention_layernorm"):
            student_block.ln2.weight.copy_(teacher_layer.post_attention_layernorm.weight)

        # 2. Copy Gated SwiGLU MLP
        if hasattr(student_block, "mlp") and hasattr(teacher_layer, "mlp"):
            safe_copy_linear(student_block.mlp, "w1", teacher_layer.mlp.gate_proj)
            safe_copy_linear(student_block.mlp, "w2", teacher_layer.mlp.up_proj)
            safe_copy_linear(student_block.mlp, "w3", teacher_layer.mlp.down_proj)

        # 3. Copy Attention Projections
        if hasattr(student_block, "attn") and hasattr(teacher_layer, "self_attn"):
            safe_copy_linear(student_block.attn, "W_q", teacher_layer.self_attn.q_proj)
            safe_copy_linear(student_block.attn, "W_k", teacher_layer.self_attn.k_proj)
            safe_copy_linear(student_block.attn, "W_v", teacher_layer.self_attn.v_proj)
            safe_copy_linear(student_block.attn, "W_out", teacher_layer.self_attn.o_proj)

            if hasattr(student_block.attn, "W_swish_gate"):
                safe_copy_linear(student_block.attn, "W_swish_gate", teacher_layer.self_attn.q_proj)

            # 4. Truncated SVD for W_u and W_r factoring W_k^T * W_v
            W_k = teacher_layer.self_attn.k_proj.weight.float()
            W_v = teacher_layer.self_attn.v_proj.weight.float()
            prod = torch.matmul(W_k.T, W_v)

            U, S, V = torch.svd(prod)
            sqrt_s = torch.sqrt(torch.clamp(S[:rank], min=1e-8))

            svd_u = (U[:, :rank] * sqrt_s).T.to(torch.bfloat16)
            svd_r = (V[:, :rank] * sqrt_s).T.to(torch.bfloat16)

            d_model = student_block.attn.d_model
            num_heads = student_block.attn.num_heads

            svd_u_tiled = svd_u.repeat(num_heads, 1)[: num_heads * rank, :d_model]
            svd_r_tiled = svd_r.repeat(num_heads, 1)[: num_heads * rank, :d_model]

            if student_block.attn.W_u.weight.shape != svd_u_tiled.shape:
                student_block.attn.W_u = nn.Linear(
                    d_model, num_heads * rank, bias=False, device=W_k.device, dtype=torch.bfloat16
                )
                student_block.attn.W_r = nn.Linear(
                    d_model, num_heads * rank, bias=False, device=W_k.device, dtype=torch.bfloat16
                )

            student_block.attn.W_u.weight.copy_(svd_u_tiled)
            student_block.attn.W_r.weight.copy_(svd_r_tiled)

            # 5. Initialize Decay and Beta Gates to Neutral Zero Baseline
            if hasattr(student_block.attn, "W_gate"):
                torch.nn.init.zeros_(student_block.attn.W_gate.weight)
                if student_block.attn.W_gate.bias is not None:
                    torch.nn.init.zeros_(student_block.attn.W_gate.bias)

            if hasattr(student_block.attn, "W_beta_gate"):
                torch.nn.init.zeros_(student_block.attn.W_beta_gate.weight)
                if student_block.attn.W_beta_gate.bias is not None:
                    torch.nn.init.zeros_(student_block.attn.W_beta_gate.bias)


def assemble_baretorch_directly_from_teacher(
    config: BareTorchConfig,
    teacher_model_name: str,
    device: torch.device,
    rank: int = 8,
) -> BareTorchForCausalLM:
    """Directly synthesizes BareTorch model from Teacher weights with zero Stage 1 file reliance."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    proc_rank = int(os.environ.get("RANK", 0))

    if local_rank == 0:
        logger.info("=" * 70)
        logger.info(f"🧬 Performing Direct Weight Surgery from '{teacher_model_name}'")
        logger.info("=" * 70)

    local_snapshot_path = teacher_model_name
    if proc_rank == 0:
        local_snapshot_path = snapshot_download(repo_id=teacher_model_name)

    if dist.is_initialized():
        path_list = [local_snapshot_path] if proc_rank == 0 else [None]
        dist.broadcast_object_list(path_list, src=0)
        local_snapshot_path = path_list[0]
        dist.barrier()

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

    teacher_rotary_emb = None
    if hasattr(teacher_base, "rotary_emb"):
        teacher_rotary_emb = teacher_base.rotary_emb
    elif hasattr(teacher_model, "rotary_emb"):
        teacher_rotary_emb = teacher_model.rotary_emb

    torch.set_default_dtype(torch.bfloat16)
    student_model = BareTorchForCausalLM(config).to(device)
    torch.set_default_dtype(torch.float32)

    if teacher_rotary_emb is not None:
        student_model.model.rotary_emb = teacher_rotary_emb.to(device)

    with torch.no_grad():
        student_model.model.token_embedding.weight.data.copy_(teacher_emb.to(device))

        if lm_head_weight is not None and hasattr(student_model, "lm_head"):
            student_model.lm_head.weight.data.copy_(lm_head_weight.to(device))
        elif config.tie_word_embeddings:
            student_model.lm_head.weight = student_model.model.token_embedding.weight

        if norm_weight is not None and hasattr(student_model.model, "final_layernorm"):
            student_model.model.final_layernorm.weight.data.copy_(norm_weight.to(device))

    # Populate layers
    for l_idx in range(config.num_layers):
        if (l_idx + 1) % 4 == 0:
            # 1:3 pattern: keep pristine native Teacher Transformer layer
            student_model.model.layers[l_idx] = teacher_base.layers[l_idx].to(device)
            if local_rank == 0:
                logger.info(
                    f"   ├─ Layer {l_idx:02d}/{config.num_layers - 1:02d}: Native Teacher Transformer"
                )
        else:
            # Warm-start CS-LRAD layer
            warm_start_cslrad_layer(
                teacher_layer=teacher_base.layers[l_idx],
                student_block=student_model.model.layers[l_idx],
                rank=rank,
            )
            if local_rank == 0:
                logger.info(
                    f"   ├─ Layer {l_idx:02d}/{config.num_layers - 1:02d}: CS-LRAD (Warm-started via SVD & Projections)"
                )

    del teacher_model, teacher_base, teacher_emb, lm_head_weight, norm_weight
    gc.collect()
    torch.cuda.empty_cache()

    if local_rank == 0:
        logger.info("✅ Direct Weight Surgery Complete! Teacher backbone fully inherited.")

    return student_model


# ==================================================================
# 4. Teacher Predictions Triplet Memory-Mapped Dataset
# ==================================================================


class DistillMemmapDataset(Dataset):
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

        oov_tok_mask = (raw_tokens < 0) | (raw_tokens >= self.vocab_size)
        if oov_tok_mask.any():
            raw_tokens[oov_tok_mask] = self.eos_token_id

        raw_teacher_indices = np.clip(raw_teacher_indices, 0, self.vocab_size - 1)

        tokens = torch.from_numpy(raw_tokens)
        teacher_indices = torch.from_numpy(raw_teacher_indices)
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
    max_val_seqs: int = 2441,
    vocab_size: int = 248320,
    eos_token_id: int = 248044,
    seed: int = 42,
):
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
            f"Loaded dataset pool: {len(train_ds):,} train sequences (~{(len(train_ds)*seq_len)/1e6:.1f}M tokens) | "
            f"{len(val_ds):,} val sequences (~{(len(val_ds)*seq_len)/1e6:.1f}M tokens)"
        )

    return train_ds, val_ds


# ==================================================================
# 5. Two-Phase LR Scheduler Engine
# ==================================================================


def get_two_phase_schedule(
    optimizer,
    num_warmup_steps: int,
    phase1_steps: int,
    total_steps: int,
    backbone_lr_ratio: float = 0.08,
    emb_lr_ratio: float = 0.02,
    min_lr_ratio: float = 0.0,
):
    """
    Two-Phase Learning Rate Schedule:
      Group 0 & 1 (CS-LRAD Recurrent Layers):
        - Phase 1 (0 to phase1_steps): Warmup -> Stable at base LR.
        - Phase 2 (phase1_steps to total_steps): Cosine decay to min_lr_ratio.
      Group 2 & 3 (Inherited MLPs & Attention):
        - Phase 1: LR = 0.0 (Protected from distortion).
        - Phase 2: Gentle warmup -> Cosine decay scaled by backbone_lr_ratio.
      Group 4 (Embeddings & LM Head):
        - Phase 1: LR = 0.0.
        - Phase 2: Gentle warmup -> Cosine decay scaled by emb_lr_ratio.
    """
    phase2_steps = max(1, total_steps - phase1_steps)
    phase2_warmup = max(1, int(phase2_steps * 0.05))

    def lr_lambda_cslrad(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        elif current_step < phase1_steps:
            return 1.0
        else:
            progress = float(current_step - phase1_steps) / float(phase2_steps)
            progress = min(1.0, max(0.0, progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + np.cos(np.pi * progress))

    def lr_lambda_backbone(current_step):
        if current_step < phase1_steps:
            return 0.0
        p2_step = current_step - phase1_steps
        if p2_step < phase2_warmup:
            mult = float(p2_step) / float(phase2_warmup)
        else:
            progress = float(p2_step - phase2_warmup) / float(max(1, phase2_steps - phase2_warmup))
            progress = min(1.0, max(0.0, progress))
            mult = min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + np.cos(np.pi * progress))
        return backbone_lr_ratio * mult

    def lr_lambda_emb(current_step):
        if current_step < phase1_steps:
            return 0.0
        p2_step = current_step - phase1_steps
        if p2_step < phase2_warmup:
            mult = float(p2_step) / float(phase2_warmup)
        else:
            progress = float(p2_step - phase2_warmup) / float(max(1, phase2_steps - phase2_warmup))
            progress = min(1.0, max(0.0, progress))
            mult = min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + np.cos(np.pi * progress))
        return emb_lr_ratio * mult

    return LambdaLR(
        optimizer,
        [
            lr_lambda_cslrad,    # Group 0: cs_lrad decay
            lr_lambda_cslrad,    # Group 1: cs_lrad no-decay
            lr_lambda_backbone,  # Group 2: backbone decay
            lr_lambda_backbone,  # Group 3: backbone no-decay
            lr_lambda_emb,       # Group 4: emb & lm_head
        ],
    )


# ==================================================================
# 6. Two-Phase Distill Trainer (50% CE + 50% Top-4 KL)
# ==================================================================


class TwoPhaseDistillTrainer(Trainer):
    def __init__(
        self,
        *args,
        alpha_ce=0.5,
        alpha_kl=0.5,
        temperature=1.0,
        phase1_steps=2000,
        backbone_lr_ratio=0.08,
        emb_lr_ratio=0.02,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.alpha_ce = alpha_ce
        self.alpha_kl = alpha_kl
        self.temperature = temperature
        self.phase1_steps = phase1_steps
        self.backbone_lr_ratio = backbone_lr_ratio
        self.emb_lr_ratio = emb_lr_ratio
        self.non_cslrad_params = []

    def create_optimizer(self):
        """Constructs 5 distinct parameter groups for differential optimization."""
        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(self.model)

            cslrad_decay = []
            cslrad_no_decay = []
            backbone_decay = []
            backbone_no_decay = []
            emb_head = []

            for n, p in self.model.named_parameters():
                if not p.requires_grad:
                    continue

                if any(k in n for k in ["token_embedding", "lm_head"]):
                    emb_head.append(p)
                elif ".attn." in n and any(
                    k in n
                    for k in [
                        "W_u",
                        "W_r",
                        "W_gate",
                        "W_beta_gate",
                        "W_q",
                        "W_k",
                        "W_v",
                        "W_out",
                        "W_swish_gate",
                    ]
                ):
                    if n in decay_parameters:
                        cslrad_decay.append(p)
                    else:
                        cslrad_no_decay.append(p)
                else:
                    if n in decay_parameters:
                        backbone_decay.append(p)
                    else:
                        backbone_no_decay.append(p)

            self.non_cslrad_params = backbone_decay + backbone_no_decay + emb_head

            optimizer_grouped_parameters = [
                {"params": cslrad_decay, "weight_decay": self.args.weight_decay},
                {"params": cslrad_no_decay, "weight_decay": 0.0},
                {"params": backbone_decay, "weight_decay": self.args.weight_decay},
                {"params": backbone_no_decay, "weight_decay": 0.0},
                {"params": emb_head, "weight_decay": 0.0},
            ]

            import bitsandbytes as bnb

            self.optimizer = bnb.optim.AdamW8bit(
                optimizer_grouped_parameters,
                lr=self.args.learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
            )
        return self.optimizer

    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        if self.lr_scheduler is None:
            opt = self.optimizer if optimizer is None else optimizer
            self.lr_scheduler = get_two_phase_schedule(
                optimizer=opt,
                num_warmup_steps=self.args.warmup_steps,
                phase1_steps=self.phase1_steps,
                total_steps=num_training_steps,
                backbone_lr_ratio=self.backbone_lr_ratio,
                emb_lr_ratio=self.emb_lr_ratio,
            )
        return self.lr_scheduler

    def training_step(self, model, inputs, num_items_in_batch=None):
        """Zeroes non-CS-LRAD gradients during Phase 1 to prevent optimizer state pollution."""
        try:
            loss = super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)
        except TypeError:
            loss = super().training_step(model, inputs)

        # In Phase 1, purge non-CS-LRAD gradients after backward
        if self.state.global_step < self.phase1_steps:
            for p in self.non_cslrad_params:
                p.grad = None

        return loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        import torch.utils.checkpoint as cp

        input_ids = inputs["input_ids"]
        labels = inputs["labels"]
        teacher_indices = inputs["teacher_indices"]
        teacher_values = inputs["teacher_values"]

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

        p_teacher_full = F.softmax(shift_teacher_values / self.temperature, dim=-1)

        B, L_shift, _ = shift_hidden.shape
        num_tokens = B * L_shift
        chunk_size = 512

        total_loss_ce = 0.0
        total_loss_kl = 0.0

        def compute_chunk(c_hid, c_lbl, c_t_idx, c_t_p):
            target_dtype = lm_head.weight.dtype if hasattr(lm_head, "weight") else torch.bfloat16
            c_hid_input = c_hid.to(target_dtype) if c_hid.dtype != target_dtype else c_hid

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
        loss_kl = (total_loss_kl / num_tokens) * (self.temperature**2)
        total_loss = (self.alpha_ce * loss_ce) + (self.alpha_kl * loss_kl)

        return (total_loss, outputs) if return_outputs else total_loss


# ==================================================================
# 7. Main Executable
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
        description="BareTorch End-to-End Distillation (2-Phase Differential LR)"
    )

    # Architecture
    parser.add_argument(
        "--layer_sequence",
        type=str,
        default="cs_lrad,cs_lrad,cs_lrad,transformer",
    )
    parser.add_argument(
        "--teacher_model_name", type=str, default="Qwen/Qwen3.5-0.8B"
    )
    parser.add_argument(
        "--data_cache_dir",
        type=str,
        default="./data_1B/train",
        help="Directory containing tokenized teacher prediction binary dataset.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints_e2e_distill",
    )
    parser.add_argument(
        "--tie_embeddings", action="store_true", default=False
    )

    # Distillation Parameters
    parser.add_argument("--alpha_ce", type=float, default=0.5)
    parser.add_argument("--alpha_kl", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=1.0)

    # Two-Phase Optimization
    parser.add_argument(
        "--max_steps",
        type=int,
        default=15258,
        help="Total steps across both phases (~1B tokens @ EBS 32).",
    )
    parser.add_argument(
        "--phase1_steps",
        type=int,
        default=2000,
        help="Phase 1 steps: Only train CS-LRAD mixer layers (Backbone frozen).",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2.5e-4,
        help="Base peak learning rate for CS-LRAD recurrent layers.",
    )
    parser.add_argument(
        "--backbone_lr_ratio",
        type=float,
        default=0.08,
        help="Multiplier for MLPs and native Attention in Phase 2 (default: 0.08 => ~2e-5).",
    )
    parser.add_argument(
        "--emb_lr_ratio",
        type=float,
        default=0.02,
        help="Multiplier for Embeddings and LM Head in Phase 2 (default: 0.02 => ~5e-6).",
    )
    parser.add_argument("--warmup_steps", type=int, default=300)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)

    # Dimensions
    parser.add_argument("--d_model", type=int, default=1024)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--num_layers", type=int, default=24)
    parser.add_argument("--chunk_size", type=int, default=32)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=2048)

    # Performance & Sync
    parser.add_argument("--r2_sync", action="store_true", default=False)
    parser.add_argument("--r2_bucket", type=str, default="baretorch-data")
    parser.add_argument("--r2_prefix", type=str, default="checkpoints_e2e")
    parser.add_argument("--compile", action="store_true", default=False)
    parser.add_argument("--grad_checkpointing", action="store_true", default=False)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=500)

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.teacher_model_name, trust_remote_code=True
    )
    teacher_config = AutoConfig.from_pretrained(
        args.teacher_model_name, trust_remote_code=True
    )
    vocab_size = getattr(teacher_config, "vocab_size", len(tokenizer))
    eos_token_id = getattr(tokenizer, "eos_token_id", getattr(teacher_config, "eos_token_id", 248044))
    if isinstance(eos_token_id, (list, tuple)):
        eos_token_id = eos_token_id[0]

    raw_sequence = [s.strip().lower() for s in args.layer_sequence.split(",") if s.strip()]
    layer_types = [raw_sequence[i % len(raw_sequence)] for i in range(args.num_layers)]

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

    # 1. Direct Model Surgery from Teacher
    model = assemble_baretorch_directly_from_teacher(
        config=config,
        teacher_model_name=args.teacher_model_name,
        device=device,
        rank=args.rank,
    )

    # 2. Targeted torch.compile
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
            logger.info(f"Successfully compiled {compiled_blocks} LRADDecoderBlock sub-module(s).")

    unique_params_dict = {p.data_ptr(): p for p in model.parameters()}
    total_params = sum(p.numel() for p in unique_params_dict.values())

    if local_rank == 0:
        logger.info("=" * 70)
        logger.info("🚀 BareTorch Direct End-to-End Distillation Initialized")
        logger.info(f"   ├─ Total Parameters      : {total_params / 1e9:.3f} Billion ({total_params:,})")
        logger.info(f"   ├─ Phase 1 Steps (CS-LRAD): 0 to {args.phase1_steps:,} (Backbone Frozen)")
        logger.info(f"   ├─ Phase 2 Steps (Global) : {args.phase1_steps:,} to {args.max_steps:,} (Differential LR)")
        logger.info(f"   ├─ Peak LR (CS-LRAD)      : {args.learning_rate:.2e}")
        logger.info(f"   ├─ Peak LR (MLP Backbone) : {args.learning_rate * args.backbone_lr_ratio:.2e}")
        logger.info(f"   ├─ Peak LR (Embeddings)   : {args.learning_rate * args.emb_lr_ratio:.2e}")
        logger.info("=" * 70)

    # 3. Datasets
    train_dataset, val_dataset = build_disjoint_train_val_datasets(
        data_dir=args.data_cache_dir,
        seq_len=args.seq_len,
        max_val_seqs=2441,
        vocab_size=vocab_size,
        eos_token_id=eos_token_id,
    )

    # 4. Training Arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        bf16=True,
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

    # 5. Cloudflare R2 Sync
    enable_r2_sync = args.r2_sync or os.environ.get("R2_SYNC", "0").lower() in ("1", "true", "yes")
    r2_bucket = os.environ.get("R2_BUCKET", args.r2_bucket)
    r2_prefix = os.environ.get("R2_PREFIX", args.r2_prefix)

    callbacks = []
    if enable_r2_sync:
        if local_rank == 0:
            logger.info(
                f"☁️ Cloudflare R2 Sync ACTIVATED. Target Bucket: '{r2_bucket}' | Prefix: '{r2_prefix}'"
            )
        callbacks.append(R2CheckpointCallback(bucket_name=r2_bucket, prefix=r2_prefix))

    # 6. Trainer
    trainer = TwoPhaseDistillTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=callbacks,
        alpha_ce=args.alpha_ce,
        alpha_kl=args.alpha_kl,
        temperature=args.temperature,
        phase1_steps=args.phase1_steps,
        backbone_lr_ratio=args.backbone_lr_ratio,
        emb_lr_ratio=args.emb_lr_ratio,
    )

    # 7. Resumption
    checkpoint_to_resume = None
    if os.path.exists(training_args.output_dir):
        existing_checkpoints = [
            d for d in os.listdir(training_args.output_dir) if d.startswith("checkpoint-")
        ]
        if existing_checkpoints:
            checkpoint_to_resume = True
            if local_rank == 0:
                logger.info(
                    f"Found existing checkpoint in '{training_args.output_dir}'. Resuming..."
                )

    trainer.train(resume_from_checkpoint=checkpoint_to_resume)

    # 8. Save Final Model
    if local_rank == 0:
        logger.info(f"💾 Saving final model & tokenizer to '{args.output_dir}'...")
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        logger.info("✅ End-to-End Distillation completed successfully!")

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

    os._exit(0)


if __name__ == "__main__":
    main()