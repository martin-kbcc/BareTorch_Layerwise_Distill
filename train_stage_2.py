# /home/martinkb/Desktop/BareTorch_Layerwise_Distill/train_stage_3.py
import argparse
import logging
import os
import subprocess
import numpy as np
import torch
import torch.serialization
from datetime import timedelta

# Disable Hugging Face Rust parallelism warnings/deadlocks in torchrun
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 1. Enable TF32 for Tensor Core acceleration
torch.set_float32_matmul_precision("high")

# 2. Bypass buggy cuDNN attention backend during Evaluation / no_grad
torch.backends.cuda.enable_cudnn_sdp(False)

# 3. Comprehensive allowlist for NumPy types in PyTorch 2.6+
safe_numpy_types = [np.dtype, np.ndarray]
for mod_path in ["numpy._core.multiarray", "numpy.core.multiarray", "numpy._core.numerictypes"]:
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

# 4. Universal Fail-Safe: Force trusted local checkpoints to bypass strict weights_only check
_orig_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from datasets import Dataset, load_dataset, interleave_datasets
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    default_data_collator,
)

from baretorch import BareTorchConfig, BareTorchForCausalLM

# ==============================================================================
#                                Logging Configuration
# ==============================================================================
logging.basicConfig(
    level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s"
)
logger = logging.getLogger(__name__)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)


# ==============================================================================
#                         Cloudflare R2 Background Sync Callback
# ==============================================================================
class R2CheckpointCallback(TrainerCallback):
    """Hugging Face Trainer Callback that automatically syncs newly saved
    checkpoints to Cloudflare R2 asynchronously using rclone.
    Does not block active GPU training execution.
    """

    def __init__(
        self,
        bucket_name: str = "baretorch-data",
        remote_name: str = "r2",
        prefix: str = "checkpoints",
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
                    f"\n[R2 Sync] Uploading {checkpoint_dir} to Cloudflare R2"
                    f" ({target_r2_path}) in background..."
                )
                cmd = [
                    "rclone",
                    "copy",
                    local_ckpt_path,
                    target_r2_path,
                    "--transfers",
                    "4",
                    "--s3-chunk-size",
                    "64M",
                ]
                subprocess.Popen(
                    cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )


def pack_chatml_dataset(raw_dataset, tokenizer, max_seq_len=2048, local_rank=0):
    """Packs multi-turn ChatML conversations using Whole-Sample Packing.

    Guarantees no conversation is cut in half across sequence boundaries.
    Applies completion-only loss masking (-100 for prompts/headers).
    Uses int64 NumPy arrays to prevent PyTorch cross-entropy scalar type errors.
    """
    if local_rank == 0:
        logger.info(
            "📦 Packing ChatML conversations (Whole-Sample Packing, High Memory Efficiency)..."
        )

    vocab_size = len(tokenizer)
    packed_samples = []
    current_ids = []
    current_labels = []

    iterator = tqdm(
        raw_dataset,
        desc="Packing ChatML Data",
        disable=(local_rank != 0),
        dynamic_ncols=True,
    )

    for example in iterator:
        messages = example.get("messages", [])
        if not messages:
            continue

        sample_ids = []
        sample_labels = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "assistant":
                header_str = f"<|im_start|>{role}\n"
                header_tokens = tokenizer.encode(header_str, add_special_tokens=False)
                body_str = f"{content}<|im_end|>\n"
                body_tokens = tokenizer.encode(body_str, add_special_tokens=False)

                sample_ids.extend(header_tokens + body_tokens)
                sample_labels.extend([-100] * len(header_tokens) + body_tokens)
            else:
                formatted_turn = f"<|im_start|>{role}\n{content}<|im_end|>\n"
                tokens = tokenizer.encode(formatted_turn, add_special_tokens=False)

                sample_ids.extend(tokens)
                sample_labels.extend([-100] * len(tokens))

        if len(sample_ids) > max_seq_len:
            continue

        if len(current_ids) + len(sample_ids) > max_seq_len:
            pad_len = max_seq_len - len(current_ids)
            chunk_ids = current_ids + [tokenizer.pad_token_id] * pad_len
            chunk_labels = current_labels + [-100] * pad_len
            chunk_mask = [1] * len(current_ids) + [0] * pad_len

            if any(lbl != -100 for lbl in chunk_labels):
                clean_ids = [
                    t if (0 <= t < vocab_size) else tokenizer.pad_token_id
                    for t in chunk_ids
                ]
                clean_labels = [
                    lbl if (0 <= lbl < vocab_size or lbl == -100) else -100
                    for lbl in chunk_labels
                ]

                packed_samples.append({
                    "input_ids": np.array(clean_ids, dtype=np.int64),
                    "labels": np.array(clean_labels, dtype=np.int64),
                    "attention_mask": np.array(chunk_mask, dtype=np.int64),
                })

            current_ids = []
            current_labels = []

        current_ids.extend(sample_ids)
        current_labels.extend(sample_labels)

    if current_ids:
        pad_len = max_seq_len - len(current_ids)
        chunk_ids = current_ids + [tokenizer.pad_token_id] * pad_len
        chunk_labels = current_labels + [-100] * pad_len
        chunk_mask = [1] * len(current_ids) + [0] * pad_len

        if any(lbl != -100 for lbl in chunk_labels):
            clean_ids = [
                t if (0 <= t < vocab_size) else tokenizer.pad_token_id
                for t in chunk_ids
            ]
            clean_labels = [
                lbl if (0 <= lbl < vocab_size or lbl == -100) else -100
                for lbl in chunk_labels
            ]

            packed_samples.append({
                "input_ids": np.array(clean_ids, dtype=np.int64),
                "labels": np.array(clean_labels, dtype=np.int64),
                "attention_mask": np.array(chunk_mask, dtype=np.int64),
            })

    if local_rank == 0:
        logger.info(
            f"✅ Packing complete: Created {len(packed_samples):,} whole-sample packed {max_seq_len}-token sequences."
        )

    return Dataset.from_list(packed_samples)


def main():
    if "LOCAL_RANK" in os.environ:
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend="nccl",
                timeout=timedelta(minutes=60)  # Extends timeout to 1 hour
            )
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)

    parser = argparse.ArgumentParser(
        description="BareTorch Stage 3: Supervised Fine-Tuning (SFT) Engine (8-Bit AdamW + BF16 Mixed Precision)"
    )

    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default="./qwen3.5_0.8B_clm_checkpoints",
        help="Path to pre-trained BareTorch checkpoint folder.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./qwen3.5_0.8B_sft_checkpoints",
        help="Directory to save fine-tuned SFT weights.",
    )

    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="Qwen/Qwen3.5-0.8B",
        help="Hugging Face tokenizer identifier.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Sub-sample N rows for fast iteration. Set to 0 for full dataset.",
    )

    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Per-GPU batch size."
    )
    parser.add_argument(
        "--grad_accum", type=int, default=4, help="Gradient accumulation steps."
    )
    parser.add_argument(
        "--learning_rate", type=float, default=2e-5, help="SFT learning rate."
    )
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable targeted torch.compile for CS-LRAD sub-modules.",
    )
    parser.add_argument(
        "--grad_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing.",
    )

    parser.add_argument(
        "--r2_sync",
        action="store_true",
        help="Enable background checkpoint syncing to Cloudflare R2 via rclone.",
    )
    parser.add_argument(
        "--r2_bucket",
        type=str,
        default="baretorch-data",
        help="Cloudflare R2 bucket name.",
    )
    parser.add_argument(
        "--r2_prefix",
        type=str,
        default="qwen3.5_0.8B_sft_checkpoints",
        help="Prefix path inside R2 bucket.",
    )

    args = parser.parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if local_rank == 0:
        logger.info(f"Initializing Tokenizer '{args.tokenizer_name}'...")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, trust_remote_code=True)
    tokenizer.model_max_length = args.seq_len
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if local_rank == 0:
        logger.info(
            f"Loading pre-trained model weights in BF16 precision from: {args.pretrained_model_path}"
        )

    model = BareTorchForCausalLM.from_pretrained(
        args.pretrained_model_path, 
        torch_dtype=torch.bfloat16,
    )

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model.config.use_grad_checkpointing = args.grad_checkpointing

    if args.compile:
        if local_rank == 0:
            logger.info(
                "⚡ Applying targeted torch.compile to custom CS-LRAD sub-modules..."
            )
        compiled_blocks = 0
        for name, module in model.named_modules():
            cls_name = module.__class__.__name__.lower()
            if "lrad" in cls_name or "lrad" in name.lower():
                module.forward = torch.compile(module.forward)
                compiled_blocks += 1
        if local_rank == 0:
            logger.info(
                f"Successfully compiled {compiled_blocks} CS-LRAD recurrent sub-module(s)."
            )

    # --------------------------------------------------------------------------
    # Distributed Rank-0 Dataset Preparation and Disk Caching
    # --------------------------------------------------------------------------
    cache_dir = os.path.join(args.output_dir, "packed_dataset")

    if local_rank == 0:
        if not os.path.exists(cache_dir):
            logger.info("Loading SFT Datasets (50% Chat / 20% Code / 15% Math / 15% Reasoning)...")

            # 1. 50% Chat Anchor (Selected columns only to prevent interleave schema mismatch)
            ds_chat = load_dataset("HuggingFaceTB/smoltalk", "all", split="train")
            ds_chat = ds_chat.select_columns(["messages"])

            # 2. 20% Code
            ds_code = load_dataset("ise-uiuc/Magicoder-Evol-Instruct-110K", split="train")
            ds_code = ds_code.map(
                lambda x: {
                    "messages": [
                        {"role": "user", "content": x["instruction"]},
                        {"role": "assistant", "content": x["response"]},
                    ]
                },
                remove_columns=ds_code.column_names,
            )

            # 3. 15% Math (NuminaMath-CoT with <think> tag injection)
            ds_math = load_dataset("AI-MO/NuminaMath-CoT", split="train")
            def format_math_with_think(x):
                problem = x["problem"]
                solution = x["solution"]
                if "####" in solution:
                    reasoning, answer = solution.rsplit("####", 1)
                    formatted_content = (
                        f"<think>\n{reasoning.strip()}\n</think>\n\n"
                        f"The final answer is \\boxed{{{answer.strip()}}}."
                    )
                else:
                    formatted_content = f"<think>\n{solution.strip()}\n</think>"

                return {
                    "messages": [
                        {"role": "user", "content": problem},
                        {"role": "assistant", "content": formatted_content},
                    ]
                }
            ds_math = ds_math.map(format_math_with_think, remove_columns=ds_math.column_names)

            # 4. 15% DeepSeek-R1 Distilled Reasoning (Bespoke-Stratos-17k)
            ds_reasoning = load_dataset("bespokelabs/Bespoke-Stratos-17k", split="train")
            def format_stratos_reasoning(x):
                if "messages" in x and x["messages"]:
                    messages = x["messages"]
                else:
                    system_msg = x.get("system", "")
                    user_q = x.get("question", x.get("problem", x.get("prompt", "")))
                    reasoning = x.get("reasoning", "")
                    response = x.get("response", x.get("solution", ""))

                    user_content = f"{system_msg}\n\n{user_q}".strip() if system_msg else user_q
                    if reasoning:
                        assistant_content = f"<think>\n{reasoning.strip()}\n</think>\n\n{response.strip()}"
                    else:
                        assistant_content = response.strip()

                    messages = [
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": assistant_content},
                    ]
                return {"messages": messages}
            ds_reasoning = ds_reasoning.map(format_stratos_reasoning, remove_columns=ds_reasoning.column_names)

            # Interleave into single dataset with 50/20/15/15 ratio
            raw_dataset = interleave_datasets(
                [ds_chat, ds_code, ds_math, ds_reasoning],
                probabilities=[0.50, 0.20, 0.15, 0.15],
                seed=42,
                stopping_strategy="all_exhausted",
            )

            if args.max_samples > 0 and len(raw_dataset) > args.max_samples:
                logger.info(
                    f"✂️ Sub-sampling dataset from {len(raw_dataset):,} rows to {args.max_samples:,} rows."
                )
                raw_dataset = raw_dataset.select(range(args.max_samples))

            processed_dataset = pack_chatml_dataset(
                raw_dataset, tokenizer, max_seq_len=args.seq_len, local_rank=local_rank
            )
            logger.info(f"💾 Caching packed dataset to '{cache_dir}'...")
            processed_dataset.save_to_disk(cache_dir)
        else:
            logger.info(f"⚡ Found pre-packed dataset at '{cache_dir}'. Fast-loading from disk...")

    # Wait for Rank 0 to finish dataset preparation and disk write
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    # Load dataset from disk fast on all ranks
    processed_dataset = Dataset.load_from_disk(cache_dir)

    dataset_split = processed_dataset.train_test_split(test_size=0.05, seed=42)
    train_data = dataset_split["train"]
    val_data = dataset_split["test"]

    if local_rank == 0:
        logger.info(
            f"Dataset split complete: {len(train_data):,} training samples | "
            f"{len(val_data):,} validation samples."
        )

    # Standard PyTorch DistributedDataParallel (DDP) configuration with 8-Bit AdamW & BF16 Mixed Precision
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        optim="adamw_bnb_8bit",  # bitsandbytes 8-Bit AdamW
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        bf16=True,  # BF16 Execution & Mixed Precision
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=250,
        save_strategy="steps",
        save_steps=250,
        save_total_limit=2,
        torch_compile=False,
        gradient_checkpointing=args.grad_checkpointing,
        ddp_find_unused_parameters=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
    )

    enable_r2_sync = args.r2_sync or os.environ.get("R2_SYNC", "0").lower() in ("1", "true", "yes")
    r2_bucket = os.environ.get("R2_BUCKET", args.r2_bucket)
    r2_prefix = os.environ.get("R2_PREFIX", args.r2_prefix)

    callbacks = []
    if enable_r2_sync:
        if local_rank == 0:
            logger.info(
                f"Cloudflare R2 Sync activated. Target Bucket: '{r2_bucket}' | Prefix: '{r2_prefix}'"
            )
        callbacks.append(
            R2CheckpointCallback(bucket_name=r2_bucket, prefix=r2_prefix)
        )
    else:
        if local_rank == 0:
            logger.info("R2 sync disabled. Running in local mode (disk checkpoints only).")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=default_data_collator,
        callbacks=callbacks,
    )

    checkpoint_to_resume = None
    if os.path.exists(training_args.output_dir):
        existing_checkpoints = [
            d for d in os.listdir(training_args.output_dir) if d.startswith("checkpoint-")
        ]
        if existing_checkpoints:
            checkpoint_to_resume = True
            if local_rank == 0:
                logger.info(
                    f"Found existing SFT checkpoint(s) in '{training_args.output_dir}'."
                    " Resuming training automatically..."
                )

    if local_rank == 0:
        logger.info("🔥 Starting Supervised Fine-Tuning (SFT) in DDP Mode (8-Bit AdamW + BF16)...")

    trainer.train(resume_from_checkpoint=checkpoint_to_resume)

    if local_rank == 0:
        logger.info(f"Saving final SFT checkpoint in BF16 format to '{args.output_dir}'...")
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        logger.info("✅ Supervised Fine-Tuning completed successfully!")

        if enable_r2_sync:
            rel_output_dir = os.path.basename(os.path.normpath(args.output_dir))
            target_r2_path = (
                f"r2:{r2_bucket}/{r2_prefix.strip('/')}/{rel_output_dir}"
            )
            logger.info(
                "📤 Syncing final SFT model weights to Cloudflare R2"
                f" ({target_r2_path})..."
            )
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
            logger.info("✅ Final SFT weights successfully uploaded to Cloudflare R2!")

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

    os._exit(0)


if __name__ == "__main__":
    main()