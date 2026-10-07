# /home/martinkb/Desktop/BareTorch_Layerwise_Distill/hidden_states_extraction.py
import argparse
import glob
import gc
import logging
import os
import sys
import time
import warnings
from datetime import timedelta
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from huggingface_hub import snapshot_download
from tqdm import tqdm
from transformers import AutoModelForCausalLM

import transformer_engine.pytorch as te
from transformer_engine.common.recipe import DelayedScaling, Format

from baretorch.cs_lrad import LRADDecoderBlock

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# Multi-Domain Allocation Ratios
RATIOS = {
    "fineweb_edu_100bt": 0.25,
    "stack_dedup":       0.20,
    "dclm_100bt":        0.15,
    "cosmopedia_v2":     0.12,
    "finemath_4plus":    0.12,
    "finepdfs_100bt":    0.08,
    "openr1_math":       0.08,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="BareTorch Multi-GPU FP8 Exact-Quota Feature Extractor"
    )
    parser.add_argument("--input_dir", type=str, default="./data_100M/train")
    parser.add_argument("--output_dir", type=str, default="./features_cache")
    parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints_layers")
    parser.add_argument("--target_layer", type=int, required=True)
    parser.add_argument("--teacher_model_name", type=str, default="Qwen/Qwen3.5-2B")
    parser.add_argument("--total_tokens", type=float, default=100_000_000)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--dtype_input", type=str, default="uint32")
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="sdpa",
        choices=["flash_attention_2", "sdpa", "eager"],
    )
    parser.add_argument("--use_fp8", action="store_true", default=True)
    return parser.parse_args()


def replace_linear_with_te(module, device):
    """Replaces standard PyTorch linear layers with Transformer Engine FP8 Linear layers."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
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


def load_cslrad_block_checkpoint(pt_path, device, d_model=2048):
    """Loads a trained CS-LRAD block checkpoint with dynamic shape alignment."""
    state_dict = torch.load(pt_path, map_location=device)

    block = LRADDecoderBlock(
        d_model=d_model,
        num_heads=16,
        chunk_size=32,
        rank=8,
        dropout=0.0,
    ).to(device).to(torch.bfloat16)

    for key, param in state_dict.items():
        parts = key.split(".")
        submod = block
        for p in parts[:-1]:
            submod = getattr(submod, p, None)
            if submod is None:
                break

        if submod is not None:
            attr_name = parts[-1]
            target_tensor = getattr(submod, attr_name, None)
            if target_tensor is not None and target_tensor.shape != param.shape:
                if isinstance(submod, nn.Linear) and attr_name == "weight":
                    in_f = param.shape[1]
                    out_f = param.shape[0]
                    has_bias = f"{'.'.join(parts[:-1])}.bias" in state_dict
                    new_linear = nn.Linear(
                        in_f, out_f, bias=has_bias, device=device, dtype=torch.bfloat16
                    )

                    parent = block
                    for p in parts[:-2]:
                        parent = getattr(parent, p)
                    setattr(parent, parts[-2], new_linear)

    block.load_state_dict(state_dict)
    block.eval()
    return block


def process_domain_chunk(
    teacher_model,
    student_layers,
    target_layer,
    shard_path,
    ds_name,
    start_seq,
    end_seq,
    layer_output_dir,
    seq_len,
    batch_size,
    dtype_input_str,
    device,
    rank,
    use_fp8,
):
    """Processes a domain binary shard with double-buffered asynchronous GPU-to-CPU disk streaming."""
    dtype_input = np.dtype(dtype_input_str)

    raw_tokens = np.fromfile(shard_path, dtype=dtype_input)
    max_available_seqs = len(raw_tokens) // seq_len

    actual_end_seq = min(end_seq, max_available_seqs)
    num_seqs = actual_end_seq - start_seq

    if num_seqs <= 0:
        return

    start_tok = start_seq * seq_len
    end_tok = actual_end_seq * seq_len
    packed_tokens = raw_tokens[start_tok:end_tok].reshape(num_seqs, seq_len)
    del raw_tokens

    base_name = f"{ds_name}_quota"
    hin_path = os.path.join(layer_output_dir, f"{base_name}_rank{rank}_hin.bin")
    hteacher_path = os.path.join(
        layer_output_dir, f"{base_name}_rank{rank}_hteacher.bin"
    )

    if os.path.exists(hin_path) and os.path.getsize(hin_path) > 0:
        return

    tmp_hin_path = hin_path + ".tmp"
    tmp_hteacher_path = hteacher_path + ".tmp"

    d_model = getattr(
        teacher_model.config,
        "hidden_size",
        getattr(teacher_model.config, "d_model", 2048),
    )

    hin_memmap = np.memmap(
        tmp_hin_path, dtype=np.float16, mode="w+", shape=(num_seqs, seq_len, d_model)
    )
    hteacher_memmap = np.memmap(
        tmp_hteacher_path,
        dtype=np.float16,
        mode="w+",
        shape=(num_seqs, seq_len, d_model),
    )

    pbar = tqdm(
        total=num_seqs * seq_len,
        desc=f" ⚡ {ds_name} [GPU {rank}]",
        unit="tok",
        unit_scale=True,
        leave=False,
        disable=(rank != 0),
    )

    fp8_recipe = DelayedScaling(
        fp8_format=Format.E4M3, amax_history_len=16, amax_compute_algo="most_recent"
    )
    teacher_base = getattr(
        teacher_model, teacher_model.base_model_prefix, teacher_model
    )
    teacher_layer_module = teacher_base.layers[target_layer]

    prev_hin_cpu = None
    prev_hteacher_cpu = None
    prev_b_start = 0
    prev_b_end = 0

    for b_start in range(0, num_seqs, batch_size):
        b_end = min(b_start + batch_size, num_seqs)
        curr_batch = packed_tokens[b_start:b_end]
        curr_batch_size = b_end - b_start

        input_ids = (
            torch.from_numpy(curr_batch).to(device, non_blocking=True).to(torch.long)
        )

        with torch.inference_mode(), te.fp8_autocast(
            enabled=use_fp8, fp8_recipe=fp8_recipe
        ):
            h_in = teacher_model.get_input_embeddings()(input_ids)

            if target_layer > 0 and student_layers is not None:
                position_ids = (
                    torch.arange(seq_len, dtype=torch.long, device=device)
                    .unsqueeze(0)
                    .expand(curr_batch_size, -1)
                )
                position_embeddings = teacher_base.rotary_emb(h_in, position_ids)

                for l_idx, block in enumerate(student_layers):
                    if (l_idx + 1) % 4 == 0:
                        out = block(
                            h_in,
                            position_ids=position_ids,
                            position_embeddings=position_embeddings,
                        )
                        h_in = out[0] if isinstance(out, tuple) else out
                    else:
                        out = block(h_in, use_cache=False)
                        h_in = out[0] if isinstance(out, tuple) else out

            position_ids = (
                torch.arange(seq_len, dtype=torch.long, device=device)
                .unsqueeze(0)
                .expand(curr_batch_size, -1)
            )
            position_embeddings = teacher_base.rotary_emb(h_in, position_ids)

            teacher_out = teacher_layer_module(
                h_in,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )
            h_teacher = (
                teacher_out[0] if isinstance(teacher_out, tuple) else teacher_out
            )

        curr_hin_cpu = h_in.to(torch.float16).to("cpu", non_blocking=True)
        curr_hteacher_cpu = h_teacher.to(torch.float16).to("cpu", non_blocking=True)

        if prev_hin_cpu is not None:
            hin_memmap[prev_b_start:prev_b_end] = prev_hin_cpu.numpy()
            hteacher_memmap[prev_b_start:prev_b_end] = prev_hteacher_cpu.numpy()

        prev_hin_cpu = curr_hin_cpu
        prev_hteacher_cpu = curr_hteacher_cpu
        prev_b_start = b_start
        prev_b_end = b_end

        pbar.update(curr_batch_size * seq_len)

    if prev_hin_cpu is not None:
        hin_memmap[prev_b_start:prev_b_end] = prev_hin_cpu.numpy()
        hteacher_memmap[prev_b_start:prev_b_end] = prev_hteacher_cpu.numpy()

    pbar.close()
    hin_memmap.flush()
    hteacher_memmap.flush()

    del hin_memmap, hteacher_memmap, prev_hin_cpu, prev_hteacher_cpu, packed_tokens
    gc.collect()
    torch.cuda.empty_cache()

    os.replace(tmp_hin_path, hin_path)
    os.replace(tmp_hteacher_path, hteacher_path)


def main():
    args = parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    if "RANK" in os.environ:
        dist.init_process_group(
            backend="nccl", device_id=device, timeout=timedelta(seconds=18000)
        )
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        rank, world_size = 0, 1

    layer_output_dir = os.path.join(args.output_dir, f"layer_{args.target_layer}")
    os.makedirs(layer_output_dir, exist_ok=True)

    target_total_tokens = int(args.total_tokens)

    if rank == 0:
        print("=" * 70)
        print(
            f"🚀 Feature Extractor | Quota: {target_total_tokens/1e6:.1f}M Tokens | Layer: {args.target_layer} | Model: {args.teacher_model_name}"
        )
        print("=" * 70)

    # 1. Download/verify snapshot on Rank 0 and broadcast local path
    local_snapshot_path = args.teacher_model_name
    if rank == 0:
        print(f"📦 Verifying Teacher Model ({args.teacher_model_name}) on Rank 0...")
        local_snapshot_path = snapshot_download(repo_id=args.teacher_model_name)

    if dist.is_initialized():
        path_list = [local_snapshot_path] if rank == 0 else [None]
        dist.broadcast_object_list(path_list, src=0)
        local_snapshot_path = path_list[0]
        dist.barrier()

    if rank == 0:
        print(f"📦 Loading Teacher Model from local cache '{local_snapshot_path}'...")

    # 2. All ranks load directly from local snapshot directory
    teacher_model = AutoModelForCausalLM.from_pretrained(
        local_snapshot_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
        local_files_only=True,
    ).to(device)

    if args.use_fp8:
        replace_linear_with_te(teacher_model, device)
    teacher_model.eval()

    teacher_base = getattr(
        teacher_model, teacher_model.base_model_prefix, teacher_model
    )

    student_layers = nn.ModuleList() if args.target_layer > 0 else None
    if args.target_layer > 0:
        if rank == 0:
            print(
                f"📦 Assembling Student Stack (Layers 0..{args.target_layer - 1}) for Student-Fed extraction..."
            )

        d_model = getattr(
            teacher_model.config,
            "hidden_size",
            getattr(teacher_model.config, "d_model", 2048),
        )

        for l_idx in range(args.target_layer):
            if (l_idx + 1) % 4 == 0:
                if rank == 0:
                    print(
                        f"   ├─ Layer {l_idx}: Standard Transformer (teacher layer {l_idx})"
                    )
                student_layers.append(teacher_base.layers[l_idx])
            else:
                pt_path = os.path.join(
                    args.checkpoints_dir,
                    f"layer_{l_idx}",
                    f"cs_lrad_layer_{l_idx}.pt",
                )
                if not os.path.exists(pt_path):
                    raise FileNotFoundError(
                        f"❌ Missing trained student layer weights at: {pt_path}"
                    )

                block = load_cslrad_block_checkpoint(pt_path, device, d_model=d_model)
                student_layers.append(block)

                if rank == 0:
                    print(f"   ├─ Layer {l_idx}: CS-LRAD (loaded from {pt_path})")

    if dist.is_initialized():
        dist.barrier()

    start_time = time.time()

    for ds_name, ratio in RATIOS.items():
        ds_dir = os.path.join(args.input_dir, ds_name)
        if not os.path.exists(ds_dir):
            continue

        bin_files = sorted(glob.glob(os.path.join(ds_dir, "*.bin")))
        if not bin_files:
            continue

        target_tokens_domain = int(target_total_tokens * ratio)
        target_seqs_domain = target_tokens_domain // args.seq_len

        seqs_per_gpu = target_seqs_domain // world_size
        start_seq = rank * seqs_per_gpu
        end_seq = (
            (rank + 1) * seqs_per_gpu
            if rank != world_size - 1
            else target_seqs_domain
        )

        shard_path = bin_files[0]

        process_domain_chunk(
            teacher_model=teacher_model,
            student_layers=student_layers,
            target_layer=args.target_layer,
            shard_path=shard_path,
            ds_name=ds_name,
            start_seq=start_seq,
            end_seq=end_seq,
            layer_output_dir=layer_output_dir,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            dtype_input_str=args.dtype_input,
            device=device,
            rank=rank,
            use_fp8=args.use_fp8,
        )

    if dist.is_initialized():
        dist.barrier()

    if rank == 0:
        elapsed_min = (time.time() - start_time) / 60
        print("\n" + "=" * 70)
        print(
            f"🎉 SUCCESS: Extracted exactly {target_total_tokens/1e6:.1f}M tokens for Layer {args.target_layer} in {elapsed_min:.2f} mins!"
        )
        print("=" * 70 + "\n")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()