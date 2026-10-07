# /home/martinkb/Desktop/BareTorch_Layerwise_Distill/train_stage_1.py
import argparse
import glob
import inspect
import logging
import os
import gc
import subprocess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.serialization
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

# Enable TF32 for Tensor Core acceleration on RTX 4090
torch.set_float32_matmul_precision("high")
torch.backends.cuda.enable_cudnn_sdp(False)

# ------------------------------------------------------------------
# 1. PyTorch 2.6+ NumPy Serialization Allowlist
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

# ------------------------------------------------------------------
# 2. Universal Fail-Safe Checkpoint Loader
# ------------------------------------------------------------------
_orig_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)


torch.load = _patched_torch_load

# BareTorch imports
from baretorch.cs_lrad import LRADDecoderBlock

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
)
logger = logging.getLogger(__name__)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("fsspec").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)


# ==================================================================
# 3. Cloudflare R2 Background Checkpoint Sync Callback
# ==================================================================


class R2CheckpointCallback(TrainerCallback):
    """Syncs saved single-layer checkpoints asynchronously to Cloudflare R2 using rclone in the background."""

    def __init__(
        self,
        bucket_name: str = "baretorch-data",
        remote_name: str = "r2",
        prefix: str = "checkpoints_layers",
    ):
        self.bucket_name = bucket_name
        self.remote_name = remote_name
        self.prefix = prefix.strip("/")

    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            checkpoint_dir = f"checkpoint-{state.global_step}"
            local_ckpt_path = os.path.join(args.output_dir, checkpoint_dir)

            if os.path.exists(local_ckpt_path):
                layer_dir = os.path.basename(os.path.normpath(args.output_dir))
                target_r2_path = (
                    f"{self.remote_name}:{self.bucket_name}/{self.prefix}/{layer_dir}/{checkpoint_dir}"
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
# 4. Dynamic Weight Copier & Warm-Start Initializer
# ==================================================================


def safe_copy_linear(target_module, target_attr, source_linear):
    """
    Ensures target_module.target_attr matches source_linear's shape
    and copies weights & bias safely without tensor shape mismatches.
    """
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


def warm_start_cslrad_from_qwen(qwen_layer, cslrad_block: LRADDecoderBlock, rank=8):
    """
    Transfers matching weights from a Qwen 3.5 layer into a BareTorch LRADDecoderBlock.
    Unmapped low-rank and gating matrices are initialized via SVD and neutral baselines.
    """
    with torch.no_grad():
        # 1. Copy RMSNorms
        if hasattr(cslrad_block, "ln1") and hasattr(qwen_layer, "input_layernorm"):
            cslrad_block.ln1.weight.copy_(qwen_layer.input_layernorm.weight)
        if hasattr(cslrad_block, "ln2") and hasattr(qwen_layer, "post_attention_layernorm"):
            cslrad_block.ln2.weight.copy_(qwen_layer.post_attention_layernorm.weight)

        # 2. Dynamically Resize & Copy Gated MLP (SwiGLU)
        if hasattr(cslrad_block, "mlp") and hasattr(qwen_layer, "mlp"):
            safe_copy_linear(cslrad_block.mlp, "w1", qwen_layer.mlp.gate_proj)
            safe_copy_linear(cslrad_block.mlp, "w2", qwen_layer.mlp.up_proj)
            safe_copy_linear(cslrad_block.mlp, "w3", qwen_layer.mlp.down_proj)

        # 3. Dynamically Copy Attention Projections (W_q, W_k, W_v, W_out, W_swish_gate)
        if hasattr(cslrad_block, "attn") and hasattr(qwen_layer, "self_attn"):
            safe_copy_linear(cslrad_block.attn, "W_q", qwen_layer.self_attn.q_proj)
            safe_copy_linear(cslrad_block.attn, "W_k", qwen_layer.self_attn.k_proj)
            safe_copy_linear(cslrad_block.attn, "W_v", qwen_layer.self_attn.v_proj)
            safe_copy_linear(cslrad_block.attn, "W_out", qwen_layer.self_attn.o_proj)

            if hasattr(cslrad_block.attn, "W_swish_gate"):
                safe_copy_linear(cslrad_block.attn, "W_swish_gate", qwen_layer.self_attn.q_proj)

            # 4. Truncated SVD for W_u Subspace (rank r=8)
            W_k = qwen_layer.self_attn.k_proj.weight
            W_v = qwen_layer.self_attn.v_proj.weight
            prod = torch.matmul(W_k.T, W_v)
            U, S, V = torch.svd(prod)

            d_model = cslrad_block.attn.d_model
            num_heads = cslrad_block.attn.num_heads

            svd_u = U[:, :rank].T
            svd_u_tiled = svd_u.repeat(num_heads, 1)

            target_u_shape = cslrad_block.attn.W_u.weight.shape
            expected_u_shape = svd_u_tiled[: num_heads * rank, :d_model].shape

            if target_u_shape != expected_u_shape:
                cslrad_block.attn.W_u = nn.Linear(
                    d_model,
                    num_heads * rank,
                    bias=False,
                    device=W_k.device,
                    dtype=W_k.dtype,
                )
                cslrad_block.attn.W_r = nn.Linear(
                    d_model,
                    num_heads * rank,
                    bias=False,
                    device=W_k.device,
                    dtype=W_k.dtype,
                )

            cslrad_block.attn.W_u.weight.copy_(svd_u_tiled[: num_heads * rank, :d_model])
            cslrad_block.attn.W_r.weight.copy_(svd_u_tiled[: num_heads * rank, :d_model])

            # 5. Initialize Decay Gates to Neutral Baseline
            if hasattr(cslrad_block.attn, "W_gate"):
                torch.nn.init.zeros_(cslrad_block.attn.W_gate.weight)
                if cslrad_block.attn.W_gate.bias is not None:
                    torch.nn.init.zeros_(cslrad_block.attn.W_gate.bias)

            if hasattr(cslrad_block.attn, "W_beta_gate"):
                torch.nn.init.zeros_(cslrad_block.attn.W_beta_gate.weight)
                if cslrad_block.attn.W_beta_gate.bias is not None:
                    torch.nn.init.zeros_(cslrad_block.attn.W_beta_gate.bias)


# ==================================================================
# 5. Zero-Copy Memory-Mapped Dataset
# ==================================================================


class LayerFeatureDataset(Dataset):
    """
    Worker-isolated, zero-copy memory-mapping dataset for cached layer hidden states.
    Reads binary float16 files and yields PyTorch CPU tensors without single-threaded CPU casting stalls.
    """

    def __init__(
        self,
        features_dir: str,
        target_layer: int,
        seq_len: int = 2048,
        d_model: int = 2560,
    ):
        self.layer_dir = os.path.join(features_dir, f"layer_{target_layer}")
        self.seq_len = seq_len
        self.d_model = d_model
        self.samples = []

        hin_files = sorted(glob.glob(os.path.join(self.layer_dir, "*_hin.bin")))
        if not hin_files:
            raise FileNotFoundError(
                f"❌ No cached hidden state files found in '{self.layer_dir}'."
            )

        for hin_path in hin_files:
            hteacher_path = hin_path.replace("_hin.bin", "_hteacher.bin")
            if not os.path.exists(hteacher_path):
                continue

            num_floats = os.path.getsize(hin_path) // 2  # float16 = 2 bytes
            num_seqs = num_floats // (self.seq_len * self.d_model)

            for s_idx in range(num_seqs):
                self.samples.append((hin_path, hteacher_path, s_idx))

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if local_rank == 0:
            logger.info(
                f"Loaded {len(self.samples):,} sequences for Layer {target_layer} from '{self.layer_dir}'"
            )

    def _get_worker_mmap(self, path):
        current_pid = os.getpid()
        if (
            not hasattr(self, "_worker_mmap_cache")
            or getattr(self, "_worker_pid", None) != current_pid
        ):
            self._worker_mmap_cache = {}
            self._worker_pid = current_pid

        if path not in self._worker_mmap_cache:
            m = np.memmap(path, dtype=np.float16, mode="r")
            m.flags.writeable = False
            self._worker_mmap_cache[path] = m.reshape((-1, self.seq_len, self.d_model))

        return self._worker_mmap_cache[path]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        hin_path, hteacher_path, s_idx = self.samples[idx]

        hin_mmap = self._get_worker_mmap(hin_path)
        hteacher_mmap = self._get_worker_mmap(hteacher_path)

        # Zero-copy PyTorch tensor creation (leaves float16 -> bfloat16 casting to GPU Tensor Cores)
        h_in = torch.from_numpy(hin_mmap[s_idx])
        h_teacher = torch.from_numpy(hteacher_mmap[s_idx])

        return {"h_in": h_in, "h_teacher": h_teacher}


# ==================================================================
# 6. Single-Layer Feature Matching Trainer
# ==================================================================


class SingleLayerTrainer(Trainer):
    """Custom Hugging Face Trainer for optimizing a single LRADDecoderBlock using MSE Loss."""

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        # Move float16 inputs to bfloat16 directly on Tensor Cores
        h_in = inputs["h_in"].to(torch.bfloat16)
        h_teacher = inputs["h_teacher"].to(torch.bfloat16)

        h_pred, _ = model(h_in, use_cache=False)

        loss = F.mse_loss(h_pred.float(), h_teacher.float())

        return (loss, h_pred) if return_outputs else loss


# ==================================================================
# 7. Main Executable
# ==================================================================


def main():
    parser = argparse.ArgumentParser(
        description="BareTorch Single-Layer Feature Distillation Engine (DDP + 8-Bit AdamW + BF16 + Cloudflare R2 Sync)"
    )

    parser.add_argument(
        "--target_layer",
        type=int,
        required=True,
        help="Target layer index to train (0 to 31).",
    )
    parser.add_argument(
        "--features_dir",
        type=str,
        default="./qwen3.5_0.8B_features_cache",
        help="Path to directory containing cached layer features.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./qwen3.5_0.8B_checkpoints_layers",
        help="Base output directory for saved layer checkpoints.",
    )
    parser.add_argument(
        "--teacher_model_name",
        type=str,
        default="Qwen/Qwen3.5-0.8B",
        help="Teacher model repo for warm-start weight copying.",
    )
    parser.add_argument("--d_model", type=int, default=1024)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--chunk_size", type=int, default=32)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=2048)

    parser.add_argument("--max_steps", type=int, default=153)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--warmup_steps", type=int, default=15)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--grad_accum", type=int, default=1)

    parser.add_argument("--logging_steps", type=int, default=25)
    parser.add_argument("--save_steps", type=int, default=75)

    # Cloudflare R2 Sync Flags
    parser.add_argument(
        "--r2_sync",
        action="store_true",
        default=False,
        help="Enable background checkpoint syncing to Cloudflare R2.",
    )
    parser.add_argument(
        "--r2_bucket", type=str, default="baretorch-data"
    )
    parser.add_argument(
        "--r2_prefix", type=str, default="checkpoints_layers"
    )

    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    layer_output_dir = os.path.join(args.output_dir, f"layer_{args.target_layer}")

    if local_rank == 0:
        logger.info("=" * 70)
        logger.info(
            f"🚀 Initializing CS-LRAD Layer {args.target_layer} Distillation"
        )
        logger.info("   ├─ Precision         : BF16 Execution")
        logger.info("   ├─ Optimizer         : bitsandbytes 8-Bit AdamW")
        logger.info(
            f"   ├─ Features Directory: {args.features_dir}/layer_{args.target_layer}"
        )
        logger.info(f"   └─ Output Directory  : {layer_output_dir}")
        logger.info("=" * 70)

    # 1. Load Teacher Layer to extract exact shape attributes
    if local_rank == 0:
        logger.info(
            f"📦 Loading teacher model parameters from {args.teacher_model_name} (Layer {args.target_layer})..."
        )

    teacher_full = AutoModelForCausalLM.from_pretrained(
        args.teacher_model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    teacher_layer = teacher_full.model.layers[args.target_layer]

    # 2. Instantiate LRADDecoderBlock
    student_layer = LRADDecoderBlock(
        d_model=args.d_model,
        num_heads=args.num_heads,
        chunk_size=args.chunk_size,
        rank=args.rank,
        dropout=0.0,
    ).to(torch.bfloat16)

    # 3. Warm-Start student parameters directly from Qwen 3.5
    warm_start_cslrad_from_qwen(teacher_layer, student_layer, rank=args.rank)
    del teacher_full, teacher_layer

    # 4. Apply torch.compile to student LRADDecoderBlock
    compile_mode = "reduce-overhead" if args.grad_accum == 1 else "default"
    if local_rank == 0:
        logger.info(
            f"🔥 Compiling student LRADDecoderBlock with PyTorch Inductor (mode='{compile_mode}')..."
        )
    student_layer.forward = torch.compile(student_layer.forward, mode=compile_mode)

    # 5. Load Memmap Dataset
    train_dataset = LayerFeatureDataset(
        features_dir=args.features_dir,
        target_layer=args.target_layer,
        seq_len=args.seq_len,
        d_model=args.d_model,
    )

    # 6. Training Arguments with Optimized CPU Prefetching
    training_args = TrainingArguments(
        output_dir=layer_output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        optim="adamw_bnb_8bit",
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        bf16=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        ddp_find_unused_parameters=False,
        dataloader_num_workers=4,  
        dataloader_prefetch_factor=2,  
        dataloader_pin_memory=True,  
        remove_unused_columns=False,
    )

    # 7. Cloudflare R2 Sync Callbacks
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
                f"☁️ Cloudflare R2 Sync ACTIVATED for Layer {args.target_layer}. Target Bucket: '{r2_bucket}' | Prefix: '{r2_prefix}'"
            )
        callbacks.append(
            R2CheckpointCallback(bucket_name=r2_bucket, prefix=r2_prefix)
        )

    trainer = SingleLayerTrainer(
        model=student_layer,
        args=training_args,
        train_dataset=train_dataset,
        callbacks=callbacks,
    )

    # 8. Resume from existing checkpoint if present
    checkpoint_to_resume = None
    if os.path.exists(layer_output_dir):
        existing_checkpoints = [
            d for d in os.listdir(layer_output_dir) if d.startswith("checkpoint-")
        ]
        if existing_checkpoints:
            checkpoint_to_resume = True
            if local_rank == 0:
                logger.info(
                    f"Found existing checkpoint in '{layer_output_dir}'. Resuming layer training..."
                )

    trainer.train(resume_from_checkpoint=checkpoint_to_resume)

    # 9. Save final single-layer weights and sync to R2
    if local_rank == 0:
        final_weight_path = os.path.join(
            layer_output_dir, f"cs_lrad_layer_{args.target_layer}.pt"
        )
        torch.save(student_layer.state_dict(), final_weight_path)
        logger.info(
            f"✅ Layer {args.target_layer} training complete! Saved final layer weights to: {final_weight_path}"
        )

        if enable_r2_sync:
            layer_dir = os.path.basename(os.path.normpath(layer_output_dir))
            target_r2_path = f"r2:{r2_bucket}/{r2_prefix.strip('/')}/{layer_dir}"
            logger.info(f"📤 Syncing final Layer {args.target_layer} weights to Cloudflare R2 ({target_r2_path})...")
            cmd = [
                "rclone",
                "copy",
                layer_output_dir,
                target_r2_path,
                "--transfers",
                "8",
                "--s3-chunk-size",
                "64M",
            ]
            subprocess.run(cmd, check=False)
            logger.info(f"✅ Final Layer {args.target_layer} weights uploaded to Cloudflare R2!")

    # 10. Cleaning
    gc.collect()
    torch.cuda.empty_cache()

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

    os._exit(0)


if __name__ == "__main__":
    main()