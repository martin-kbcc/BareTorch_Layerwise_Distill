import argparse
import glob
import logging
import os
import subprocess
import sys
import time
import warnings
from datetime import timedelta
import numpy as np
import torch
import torch.distributed as dist
from huggingface_hub import snapshot_download
from tqdm import tqdm
from transformers import AutoModelForCausalLM

import transformer_engine.pytorch as te
from transformer_engine.common.recipe import DelayedScaling, Format

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

active_r2_uploads = []


def parse_args():
    parser = argparse.ArgumentParser(
        description="BareTorch Ultra-Fast TransformerEngine FP8 Teacher Logit Extractor"
    )
    parser.add_argument("--input_dir", type=str, required=True, help="Path to tokenized input directory.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to output memmap directory.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3.5-9B")
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--logit_chunk_size", type=int, default=1024, help="Sequence chunk size for lm_head projection.")
    parser.add_argument("--dtype_input", type=str, default="uint32")
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="flash_attention_2",
        choices=["flash_attention_2", "sdpa", "eager"],
    )
    parser.add_argument(
        "--use_fp8",
        action="store_true",
        default=True,
        help="Enable NVIDIA TransformerEngine FP8 execution.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        default=True,
        help="Enable torch.compile Inductor fusion on the model backbone.",
    )
    parser.add_argument("--r2_sync", action="store_true", help="Sync completed bin files to Cloudflare R2 via rclone.")
    parser.add_argument("--r2_bucket", type=str, default="baretorch-data", help="R2 target bucket name.")
    parser.add_argument("--r2_remote", type=str, default="r2", help="rclone remote identifier.")
    parser.add_argument("--r2_prefix", type=str, default="teacher_predictions", help="Destination prefix inside R2 bucket.")
    return parser.parse_args()


def replace_linear_with_te(module, device):
    """Recursively replaces PyTorch Linear layers with TransformerEngine Linear layers."""
    for name, child in list(module.named_children()):
        if isinstance(child, torch.nn.Linear):
            has_bias = child.bias is not None
            te_layer = te.Linear(
                child.in_features,
                child.out_features,
                bias=has_bias,
                params_dtype=child.weight.dtype,
                device=device,
            )
            te_layer.weight.data = child.weight.data.to(device)
            if has_bias:
                te_layer.bias.data = child.bias.data.to(device)
            setattr(module, name, te_layer)
        else:
            replace_linear_with_te(child, device)


def process_shard(
    model,
    shard_path,
    output_dir,
    seq_len,
    batch_size,
    logit_chunk_size,
    dtype_input_str,
    device,
    rank,
    use_fp8,
    r2_sync,
    r2_remote,
    r2_bucket,
    r2_prefix,
):
    shard_name = os.path.basename(shard_path)
    base_name = os.path.splitext(shard_name)[0]

    packed_tokens_path = os.path.join(output_dir, f"{base_name}_packed_tokens.bin")
    indices_path = os.path.join(output_dir, f"{base_name}_teacher_indices.bin")
    values_path = os.path.join(output_dir, f"{base_name}_teacher_values.bin")

    # Return immediately if shard is fully converted
    if (
        os.path.exists(packed_tokens_path)
        and os.path.exists(indices_path)
        and os.path.exists(values_path)
        and os.path.getsize(packed_tokens_path) > 0
    ):
        return shard_name, True

    tmp_packed_path = packed_tokens_path + f".rank{rank}.tmp"
    tmp_indices_path = indices_path + f".rank{rank}.tmp"
    tmp_values_path = values_path + f".rank{rank}.tmp"

    dtype_input = np.dtype(dtype_input_str)
    raw_tokens = np.fromfile(shard_path, dtype=dtype_input)

    num_seqs = len(raw_tokens) // seq_len
    if num_seqs == 0:
        return shard_name, False

    expected_packed_bytes = num_seqs * seq_len * dtype_input.itemsize
    expected_indices_bytes = num_seqs * seq_len * 4 * 4  # uint32 = 4 bytes
    expected_values_bytes = num_seqs * seq_len * 4 * 2   # float16 = 2 bytes

    # Inspect existing .tmp files for partial extraction progress
    can_resume = (
        os.path.exists(tmp_packed_path)
        and os.path.exists(tmp_indices_path)
        and os.path.exists(tmp_values_path)
        and os.path.getsize(tmp_packed_path) == expected_packed_bytes
        and os.path.getsize(tmp_indices_path) == expected_indices_bytes
        and os.path.getsize(tmp_values_path) == expected_values_bytes
    )

    start_seq_idx = 0

    if can_resume:
        try:
            v_mem = np.memmap(tmp_values_path, dtype=np.float16, mode="r+", shape=(num_seqs, seq_len, 4))
            for b_idx in range(0, num_seqs, batch_size):
                check_seq = min(b_idx + batch_size - 1, num_seqs - 1)
                # Position index 1 in a sequence contains non-zero top-4 logprobs if processed
                if np.any(v_mem[check_seq, 1, :] != 0):
                    start_seq_idx = b_idx + batch_size
                else:
                    break
            del v_mem
            if start_seq_idx > 0:
                pct = (start_seq_idx / num_seqs) * 100
                print(f"🔄 [GPU {rank}] Resuming '{shard_name[:25]}' from sequence {start_seq_idx}/{num_seqs} ({pct:.1f}% saved!)")
        except Exception as e:
            print(f"⚠️ [GPU {rank}] Could not read temp progress ({e}). Restarting shard.")
            can_resume = False
            start_seq_idx = 0

    if not can_resume or start_seq_idx == 0:
        for tmp in [tmp_packed_path, tmp_indices_path, tmp_values_path]:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass

        packed_tokens = raw_tokens[: num_seqs * seq_len].reshape(num_seqs, seq_len)
        packed_tokens.tofile(tmp_packed_path)

        indices_memmap = np.memmap(
            tmp_indices_path, dtype=np.uint32, mode="w+", shape=(num_seqs, seq_len, 4)
        )
        values_memmap = np.memmap(
            tmp_values_path, dtype=np.float16, mode="w+", shape=(num_seqs, seq_len, 4)
        )
    else:
        indices_memmap = np.memmap(
            tmp_indices_path, dtype=np.uint32, mode="r+", shape=(num_seqs, seq_len, 4)
        )
        values_memmap = np.memmap(
            tmp_values_path, dtype=np.float16, mode="r+", shape=(num_seqs, seq_len, 4)
        )

    total_tokens = num_seqs * seq_len
    batch_pbar = tqdm(
        total=total_tokens,
        initial=start_seq_idx * seq_len,
        desc=f" ⚡ {shard_name[:25]:<25}",
        unit="tok",
        unit_scale=True,
        unit_divisor=1000,
        leave=False,
        disable=(rank != 0),
    )

    fp8_recipe = DelayedScaling(fp8_format=Format.E4M3, amax_history_len=16, amax_compute_algo="most_recent")
    base_model = getattr(model, model.base_model_prefix, model)
    lm_head = model.get_output_embeddings()

    packed_tokens_mmap = np.memmap(tmp_packed_path, dtype=dtype_input, mode="r", shape=(num_seqs, seq_len))

    for start_idx in range(start_seq_idx, num_seqs, batch_size):
        end_idx = min(start_idx + batch_size, num_seqs)
        curr_batch_size = end_idx - start_idx
        batch_np = packed_tokens_mmap[start_idx:end_idx]

        input_ids = torch.from_numpy(batch_np.astype(np.int64)).to(device, non_blocking=True)

        batch_indices_gpu = torch.empty((curr_batch_size, seq_len, 4), dtype=torch.int32, device=device)
        batch_values_gpu = torch.empty((curr_batch_size, seq_len, 4), dtype=torch.float16, device=device)

        with torch.inference_mode(), te.fp8_autocast(enabled=use_fp8, fp8_recipe=fp8_recipe):
            transformer_outputs = base_model(input_ids=input_ids, use_cache=False)
            hidden_states = transformer_outputs[0]

            for c_start in range(0, seq_len, logit_chunk_size):
                c_end = min(c_start + logit_chunk_size, seq_len)
                h_chunk = hidden_states[:, c_start:c_end, :]

                logits_chunk = lm_head(h_chunk)

                lse = torch.logsumexp(logits_chunk, dim=-1, keepdim=True)
                top_val, top_ind = torch.topk(logits_chunk, k=4, dim=-1)
                top_logprobs = top_val - lse

                if c_start == 0:
                    top_logprobs[:, 0, :] = 0.0
                    top_ind[:, 0, :] = 0

                batch_indices_gpu[:, c_start:c_end, :] = top_ind.to(torch.int32)
                batch_values_gpu[:, c_start:c_end, :] = top_logprobs.to(torch.float16)

                del logits_chunk, lse, top_val, top_ind, top_logprobs

            del input_ids, transformer_outputs, hidden_states

        indices_memmap[start_idx:end_idx] = batch_indices_gpu.cpu().numpy()
        values_memmap[start_idx:end_idx] = batch_values_gpu.cpu().numpy()

        del batch_indices_gpu, batch_values_gpu

        # Flush to NVMe disk after every batch
        indices_memmap.flush()
        values_memmap.flush()

        batch_pbar.update(curr_batch_size * seq_len)

    batch_pbar.close()
    indices_memmap.flush()
    values_memmap.flush()
    del indices_memmap, values_memmap, packed_tokens_mmap

    os.replace(tmp_packed_path, packed_tokens_path)
    os.replace(tmp_indices_path, indices_path)
    os.replace(tmp_values_path, values_path)

    if r2_sync:
        rel_output_dir = os.path.basename(os.path.normpath(output_dir))
        target_r2_dir = f"{r2_remote}:{r2_bucket}/{r2_prefix.strip('/')}/{rel_output_dir}"
        print(f"\n☁️ [R2 Sync] Uploading {base_name} binaries to Cloudflare R2 ({target_r2_dir}) in background...")

        for fpath in [packed_tokens_path, indices_path, values_path]:
            fname = os.path.basename(fpath)
            target_file_path = f"{target_r2_dir}/{fname}"
            cmd = [
                "rclone",
                "copyto",
                fpath,
                target_file_path,
                "--transfers",
                "4",
                "--s3-chunk-size",
                "64M",
            ]
            p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            active_r2_uploads.append(p)

    return shard_name, True


def main():
    args = parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    if "RANK" in os.environ:
        dist.init_process_group(
            backend="nccl",
            device_id=device,
            timeout=timedelta(seconds=18000),
        )
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        rank = 0
        world_size = 1

    if rank != 0:
        sys.stdout = open(os.devnull, "w")

    os.makedirs(args.output_dir, exist_ok=True)

    if rank == 0:
        print("=" * 70)
        print(f"🚀 Initializing TransformerEngine Engine: '{args.model_name}'")
        print(
            f"World Size: {world_size} GPUs | Device: cuda:{local_rank} | "
            f"Attention: {args.attn_implementation} | TE FP8: {args.use_fp8} | "
            f"Torch Compile: {args.compile} | Chunk Size: {args.logit_chunk_size}"
        )
        if args.r2_sync:
            print(f"☁️ R2 Sync Enabled -> Remote Target: {args.r2_remote}:{args.r2_bucket}/{args.r2_prefix}")
        print("=" * 70)

    if rank == 0:
        print("📥 Rank 0 pre-downloading/verifying model files in cache...")
        snapshot_download(repo_id=args.model_name)

    if dist.is_initialized():
        dist.barrier()

    if rank == 0:
        print("📦 Loading base model weights...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
    )

    if rank == 0:
        print(f"🚚 Moving base model to device cuda:{local_rank}...")
    model = model.to(device)

    if args.use_fp8:
        if rank == 0:
            print("⚡ Converting model Linear layers to NVIDIA TransformerEngine modules...")
        replace_linear_with_te(model, device)

    if args.compile:
        if rank == 0:
            print("🔥 Compiling model backbone with PyTorch Inductor (mode='reduce-overhead')...")
        base_model_obj = getattr(model, model.base_model_prefix, model)
        compiled_base = torch.compile(base_model_obj, mode="reduce-overhead")
        setattr(model, model.base_model_prefix, compiled_base)

    model.eval()

    shard_paths = sorted(glob.glob(f"{args.input_dir}/**/*.bin", recursive=True))
    shard_paths = [
        p
        for p in shard_paths
        if not p.endswith(
            (".tmp", "_packed_tokens.bin", "_teacher_indices.bin", "_teacher_values.bin")
        )
    ]

    rank_shards = shard_paths[rank::world_size]

    if rank == 0:
        print(
            f"\n📂 Total Shards: {len(shard_paths)} | Processing ~{len(rank_shards)} shards per GPU...\n"
        )

    start_time = time.time()
    for shard_path in rank_shards:
        process_shard(
            model=model,
            shard_path=shard_path,
            output_dir=args.output_dir,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            logit_chunk_size=args.logit_chunk_size,
            dtype_input_str=args.dtype_input,
            device=device,
            rank=rank,
            use_fp8=args.use_fp8,
            r2_sync=args.r2_sync,
            r2_remote=args.r2_remote,
            r2_bucket=args.r2_bucket,
            r2_prefix=args.r2_prefix,
        )

    if args.r2_sync and active_r2_uploads:
        if rank == 0:
            print("\n⏳ Waiting for remaining background R2 uploads to complete...")
        for p in active_r2_uploads:
            p.wait()
        if rank == 0:
            print("✅ All R2 background uploads finished successfully!")

    if dist.is_initialized():
        dist.barrier()

    elapsed_min = (time.time() - start_time) / 60
    if rank == 0:
        print("\n" + "=" * 70)
        print(
            f"🎉 SUCCESS: Logit extraction complete across {world_size} GPUs in {elapsed_min:.2f} minutes!"
        )
        print("=" * 70 + "\n")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()