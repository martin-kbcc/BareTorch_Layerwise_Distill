# /home/martinkb/Desktop/BareTorch_Layerwise_Distill/run_hardware_benchmarks_09.py
import os
import gc
import csv
import copy
import json
import inspect
import logging
import argparse
import numpy as np
import torch

from baretorch import BareTorchConfig, BareTorchForCausalLM
from baretorch.cs_lrad import LRADDecoderBlock
from init_student_00 import (
    load_qwen3_5_text_teacher,
    load_hybrid_student_checkpoint,
    copy_qwen_rmsnorm_to_baretorch,
    warm_start_cs_lrad_from_qwen_deltanet,
)

# Enable high-performance TensorFloat-32 and Flash/Mem-Efficient SDPA kernels
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch.backends.cuda.enable_cudnn_sdp(False)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("hardware_benchmark")


def parse_args():
    parser = argparse.ArgumentParser(
        description="BareTorch CS-LRAD Hybrid vs. Native Qwen3.5 Multi-Scale Hardware & KV-Cache Benchmark"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=[],
        help="List of Qwen3.5 HF model IDs to benchmark on the fly (e.g., Qwen/Qwen3.5-0.8B Qwen/Qwen3.5-2B Qwen/Qwen3.5-4B)",
    )
    parser.add_argument(
        "--teacher_id",
        type=str,
        default="Qwen/Qwen3.5-0.8B",
        help="Single Hugging Face ID of the Qwen3.5 Teacher (used if --models is not provided)",
    )
    parser.add_argument(
        "--student_base_id",
        type=str,
        default="",
        help="Optional base model ID used when loading an explicit --student_ckpt",
    )
    parser.add_argument(
        "--student_ckpt",
        type=str,
        default="",
        help="Optional path to a trained BareTorch checkpoint. If empty (default), builds BareTorch on-the-fly in RAM!",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=64,
        help="CS-LRAD chunk size (default: 64)",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=0,
        help="CS-LRAD low-rank state factor r (0 = auto-scale: 16 for 0.8B, 32 for 2B, 40 for 4B)",
    )
    parser.add_argument(
        "--context_lengths",
        type=int,
        nargs="+",
        default=[512, 1024, 2048, 4096, 8192, 16384, 32768],
        help="List of prompt context lengths to benchmark",
    )
    parser.add_argument(
        "--gen_tokens",
        type=int,
        default=64,
        help="Number of tokens to decode autoregressively per trial",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Number of untimed warmup passes per context length",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=3,
        help="Number of timed measurement trials per context length",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Target CUDA GPU device (e.g., cuda:0 on RTX 4090)",
    )
    parser.add_argument(
        "--compile_mode",
        type=str,
        default="default",
        choices=["default", "reduce-overhead", "max-autotune"],
        help="torch.compile mode for CS-LRAD recurrent blocks",
    )
    parser.add_argument(
        "--no_compile",
        action="store_true",
        help="Disable torch.compile on BareTorch CS-LRAD blocks (compiled by default)",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="./hardware_benchmark_results.csv",
        help="Output CSV path for raw benchmark metrics",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="./hardware_benchmark_results.json",
        help="Output JSON path for full structured comparison report",
    )
    return parser.parse_args()


def clear_cuda(device: torch.device):
    """Flushes Python GC, clears TorchDynamo caches if needed, and resets CUDA peak memory stats."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)


def auto_select_lrad_rank(d_model: int, user_rank: int = 0) -> int:
    """
    Selects the canonical CS-LRAD low-rank factor `r` based on `d_model`:
      - 0.8B (d_model=1024) -> rank=16
      - 2B   (d_model=2048) -> rank=32
      - 4B   (d_model=2560) -> rank=40 (or 32)
    """
    if user_rank > 0:
        return user_rank
    return max(16, d_model // 64)


@torch.no_grad()
def build_in_memory_baretorch_from_loaded_teacher(
    teacher,
    t_backbone,
    t_cfg,
    chunk_size: int = 64,
    rank: int = 0,
    max_seq_len: int = 33000,
):
    """
    Constructs the exact BareTorch CS-LRAD Hybrid Student in CPU RAM directly from an
    already-loaded Qwen3.5 Teacher (`teacher`, `t_backbone`, `t_cfg`) without touching disk
    or downloading weights twice!
    """
    eff_rank = auto_select_lrad_rank(t_cfg.hidden_size, rank)

    layer_types = []
    for i, t_layer in enumerate(t_backbone.layers):
        if hasattr(t_cfg, "layer_types") and t_cfg.layer_types is not None:
            is_full = "full" in str(t_cfg.layer_types[i]).lower()
        else:
            is_full = hasattr(t_layer, "self_attn") and not hasattr(t_layer, "linear_attn")
        layer_types.append("transformer" if is_full else "cs_lrad")

    num_heads = getattr(t_cfg, "num_attention_heads", 16)
    num_kv_heads = getattr(t_cfg, "num_key_value_heads", 2)

    bt_config = BareTorchConfig(
        vocab_size=t_cfg.vocab_size,
        d_model=t_cfg.hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        num_layers=t_cfg.num_hidden_layers,
        intermediate_size=t_cfg.intermediate_size,
        chunk_size=chunk_size,
        rank=eff_rank,
        dropout=0.0,
        max_seq_len=max(8192, max_seq_len),
        use_grad_checkpointing=False,
        use_qk_norm=True,
        rms_norm_eps=getattr(t_cfg, "rms_norm_eps", 1e-6),
        rms_norm_unit_offset=True,
        logit_soft_cap=None,
        layer_types=layer_types,
        tie_word_embeddings=getattr(t_cfg, "tie_word_embeddings", True),
    )

    student = BareTorchForCausalLM(bt_config).to(torch.bfloat16)

    # Copy Embeddings, Final Norm, and LM Head
    student.model.token_embedding.weight.copy_(t_backbone.embed_tokens.weight)
    copy_qwen_rmsnorm_to_baretorch(student.model.final_norm, t_backbone.norm)
    if getattr(bt_config, "tie_word_embeddings", True):
        student.lm_head.weight = student.model.token_embedding.weight
    else:
        student.lm_head.weight.copy_(teacher.lm_head.weight)

    if hasattr(t_backbone, "rotary_emb"):
        student.model.rotary_emb = copy.deepcopy(t_backbone.rotary_emb)

    for i, (l_type, t_layer) in enumerate(zip(layer_types, t_backbone.layers)):
        if l_type == "transformer":
            student.model.layers[i] = copy.deepcopy(t_layer)
        else:
            s_layer: LRADDecoderBlock = student.model.layers[i]
            copy_qwen_rmsnorm_to_baretorch(s_layer.ln1, t_layer.input_layernorm)
            copy_qwen_rmsnorm_to_baretorch(s_layer.ln2, t_layer.post_attention_layernorm)
            s_layer.mlp.w1.weight.copy_(t_layer.mlp.gate_proj.weight)
            s_layer.mlp.w2.weight.copy_(t_layer.mlp.up_proj.weight)
            s_layer.mlp.w3.weight.copy_(t_layer.mlp.down_proj.weight)
            t_mixer = getattr(t_layer, "linear_attn", None)
            warm_start_cs_lrad_from_qwen_deltanet(s_layer.attn, t_mixer)

    return student, eff_rank


def enable_last_logit_only_and_bypass_nan_scan(model):
    """
    1. Patches `lm_head.forward` so that during prefill (`seq_len > 1`), only the final token's
       hidden state `hidden_states[:, -1:, :]` is projected into the 248,320 vocabulary.
       This prevents allocating multi-gigabyte `[1, L, 248320]` logit tensors at 16K-32K context
       (critical on a 24GB RTX 4090!) and ensures `torch.isnan(logits)` only scans `[1, 1, 248320]`.
    2. Disables any residual `gradient_checkpointing` flags on the model.
    """
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    if hasattr(model, "config"):
        model.config.gradient_checkpointing = False
    if hasattr(model, "model") and hasattr(model.model, "gradient_checkpointing"):
        model.model.gradient_checkpointing = False

    head = (
        model.get_output_embeddings()
        if hasattr(model, "get_output_embeddings") and model.get_output_embeddings() is not None
        else getattr(model, "lm_head", None)
    )

    if (
        head is not None
        and isinstance(head, torch.nn.Linear)
        and not getattr(head, "_is_efficient_patched", False)
    ):
        orig_forward = head.forward

        def efficient_head_forward(hidden_states, *args, **kwargs):
            if hidden_states.dim() == 3 and hidden_states.size(1) > 1:
                hidden_states = hidden_states[:, -1:, :]
            return orig_forward(hidden_states, *args, **kwargs)

        head.forward = efficient_head_forward
        head._is_efficient_patched = True


def compile_baretorch_lrad_blocks(student_model, compile_mode: str = "default"):
    """
    Compiles the CS-LRAD recurrent mixer (`chunk_forward` and `step_forward`) and RMSNorms
    using `torch.compile(..., dynamic=True)` so a single compiled graph handles all context
    lengths `[512 .. 32768]` and single-token autoregressive decoding without recompilation stalls.
    """
    if hasattr(torch._C, "_dynamo") and hasattr(torch._C._dynamo.eval_frame, "_set_lru_cache"):
        torch._C._dynamo.eval_frame._set_lru_cache(False)

    compiled_count = 0
    for layer in student_model.model.layers:
        if isinstance(layer, LRADDecoderBlock):
            layer.ln1.forward = torch.compile(layer.ln1.forward, mode=compile_mode, dynamic=True)
            layer.ln2.forward = torch.compile(layer.ln2.forward, mode=compile_mode, dynamic=True)
            if hasattr(layer.attn, "chunk_forward"):
                layer.attn.chunk_forward = torch.compile(
                    layer.attn.chunk_forward, mode=compile_mode, dynamic=True
                )
            if hasattr(layer.attn, "step_forward"):
                layer.attn.step_forward = torch.compile(
                    layer.attn.step_forward, mode=compile_mode, dynamic=True
                )
            else:
                layer.attn.forward = torch.compile(
                    layer.attn.forward, mode=compile_mode, dynamic=True
                )
            compiled_count += 1
    return compiled_count


def get_forward_kwargs(model, use_cache=True, num_logits_to_keep=1):
    kwargs = {"use_cache": use_cache}
    try:
        sig = inspect.signature(model.forward)
        if "num_logits_to_keep" in sig.parameters:
            kwargs["num_logits_to_keep"] = num_logits_to_keep
    except Exception:
        pass
    return kwargs


def measure_kv_cache_megabytes(past_key_values) -> float:
    """
    Recursively traverses `past_key_values` (supporting BareTorch hybrid dicts/tuples,
    Hugging Face `DynamicCache`, and Qwen3.5 hybrid linear/attention cache objects) and sums
    the exact GPU memory footprint in Megabytes (MB) of all cached state tensors.
    """
    if past_key_values is None:
        return 0.0

    seen_ptrs = set()

    def _accumulate_bytes(obj, depth=0) -> int:
        if obj is None or depth > 6:
            return 0
        if torch.is_tensor(obj):
            ptr = obj.data_ptr()
            if ptr in seen_ptrs:
                return 0
            seen_ptrs.add(ptr)
            return obj.numel() * obj.element_size()
        if isinstance(obj, dict):
            return sum(_accumulate_bytes(v, depth + 1) for v in obj.values())
        if isinstance(obj, (list, tuple, set)):
            return sum(_accumulate_bytes(v, depth + 1) for v in obj)
        if hasattr(obj, "__dict__"):
            return sum(
                _accumulate_bytes(v, depth + 1)
                for k, v in vars(obj).items()
                if not k.startswith("__")
            )
        return 0

    total_bytes = _accumulate_bytes(past_key_values)
    return float(total_bytes) / (1024.0 ** 2)


def benchmark_model_at_context(
    model,
    input_ids: torch.Tensor,
    base_weights_gb: float,
    gen_tokens: int = 64,
    warmup: int = 2,
    trials: int = 3,
):
    """
    Measures:
      1. Time to First Token (TTFT in ms)
      2. Prefill Throughput (tokens/sec)
      3. Autoregressive Decode Velocity (tokens/sec over `gen_tokens` steps)
      4. Exact Recurrent/KV-Cache Footprint (MB)
      5. Peak Total VRAM (GB) and Dynamic Activation/Cache Overhead (GB)
    """
    seq_len = input_ids.shape[1]
    device = input_ids.device
    prefill_kwargs = get_forward_kwargs(model, use_cache=True, num_logits_to_keep=1)
    decode_kwargs = get_forward_kwargs(model, use_cache=True, num_logits_to_keep=1)

    # 1. Per-Context Untimed Warmup
    with torch.no_grad():
        for _ in range(warmup):
            out = model(input_ids, **prefill_kwargs)
            past_kv = out.past_key_values
            next_token = out.logits[:, -1:].argmax(dim=-1)
            for _ in range(min(4, gen_tokens)):
                step_out = model(next_token, past_key_values=past_kv, **decode_kwargs)
                past_kv = step_out.past_key_values
                next_token = step_out.logits[:, -1:].argmax(dim=-1)
            del out, past_kv, step_out, next_token
    clear_cuda(device)

    prefill_latencies = []
    decode_latencies = []
    peak_memories_gb = []
    kv_cache_sizes_mb = []

    # 2. Hardware-Synchronized Timing Trials
    for _ in range(trials):
        clear_cuda(device)

        # --- A. Measure Prefill Phase (TTFT) ---
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        with torch.no_grad():
            out = model(input_ids, **prefill_kwargs)
        end_event.record()
        torch.cuda.synchronize(device)

        ttft_ms = start_event.elapsed_time(end_event)
        prefill_latencies.append(ttft_ms)

        past_kv = out.past_key_values
        next_token = out.logits[:, -1:].argmax(dim=-1)
        del out

        # --- B. Measure Autoregressive Decoding Phase ---
        dec_start_event = torch.cuda.Event(enable_timing=True)
        dec_end_event = torch.cuda.Event(enable_timing=True)

        dec_start_event.record()
        with torch.no_grad():
            for _ in range(gen_tokens):
                step_out = model(next_token, past_key_values=past_kv, **decode_kwargs)
                past_kv = step_out.past_key_values
                next_token = step_out.logits[:, -1:].argmax(dim=-1)
        dec_end_event.record()
        torch.cuda.synchronize(device)

        decode_ms = dec_start_event.elapsed_time(dec_end_event)
        decode_latencies.append(decode_ms)

        kv_mb = measure_kv_cache_megabytes(past_kv)
        kv_cache_sizes_mb.append(kv_mb)

        peak_vram_gb = torch.cuda.max_memory_allocated(device) / (1024.0 ** 3)
        peak_memories_gb.append(peak_vram_gb)

        del past_kv, step_out, next_token
        clear_cuda(device)

    avg_ttft_ms = float(np.median(prefill_latencies))
    avg_prefill_tok_s = (seq_len / (avg_ttft_ms / 1000.0)) if avg_ttft_ms > 0 else 0.0
    avg_decode_ms = float(np.median(decode_latencies))
    avg_decode_tok_s = (gen_tokens / (avg_decode_ms / 1000.0)) if avg_decode_ms > 0 else 0.0
    avg_peak_vram_gb = float(np.max(peak_memories_gb))
    avg_kv_cache_mb = float(np.median(kv_cache_sizes_mb))

    return {
        "context_len": seq_len,
        "ttft_ms": avg_ttft_ms,
        "prefill_tok_s": avg_prefill_tok_s,
        "decode_ms": avg_decode_ms,
        "decode_tok_s": avg_decode_tok_s,
        "kv_cache_mb": avg_kv_cache_mb,
        "base_weights_gb": base_weights_gb,
        "peak_vram_gb": avg_peak_vram_gb,
        "dynamic_overhead_gb": max(0.0, avg_peak_vram_gb - base_weights_gb),
    }


def main():
    args = parse_args()
    model_list = args.models if args.models else [args.teacher_id]
    use_compile = not args.no_compile

    device = torch.device(args.device)
    device_name = torch.cuda.get_device_name(device)
    max_ctx = max(args.context_lengths) + args.gen_tokens + 64

    mode_desc = (
        f"Checkpoint ({args.student_ckpt})"
        if (args.student_ckpt and os.path.exists(args.student_ckpt))
        else "On-The-Fly In-Memory Stage 0 Construction (No Checkpoint Required!)"
    )

    print("=" * 118)
    print("BARETORCH CS-LRAD HYBRID vs. NATIVE QWEN3.5 MULTI-SCALE HARDWARE & KV-CACHE BENCHMARK")
    print(f"  - GPU Device        : {device_name}")
    print(f"  - Models to Sweep   : {model_list}")
    print(f"  - Student Init Mode : {mode_desc}")
    print(f"  - Context Lengths   : {args.context_lengths}")
    print(f"  - Decode Steps      : {args.gen_tokens} tokens/trial ({args.warmup} warmup + {args.trials} timed trials)")
    print(f"  - torch.compile     : {'ENABLED (' + args.compile_mode + ')' if use_compile else 'DISABLED (Eager)'}")
    print("=" * 118)

    csv_rows = []
    all_reports = {}

    for model_id in model_list:
        short_tag = model_id.split("/")[-1]
        print(f"\n" + "#" * 118)
        print(f">>> BENCHMARKING SCALE PAIR: BareTorch-{short_tag} vs. Native {short_tag}")
        print("#" * 118)

        # 1. Load Qwen3.5 Teacher once on CPU
        logger.info(f"Loading '{model_id}' on CPU...")
        qwen_model, qwen_backbone, qwen_cfg = load_qwen3_5_text_teacher(
            model_id, dtype=torch.bfloat16, device="cpu"
        )

        # 2. Construct or Load BareTorch Student on CPU from the already-loaded Teacher
        ckpt_to_load = args.student_ckpt
        if ckpt_to_load and not os.path.exists(ckpt_to_load) and os.path.exists(os.path.dirname(ckpt_to_load)):
            ckpt_to_load = os.path.dirname(ckpt_to_load)

        if ckpt_to_load and os.path.exists(ckpt_to_load) and len(model_list) == 1:
            logger.info(f"Reconstructing BareTorch Student from checkpoint '{ckpt_to_load}'...")
            bt_model = load_hybrid_student_checkpoint(
                ckpt_to_load,
                teacher_backbone=qwen_backbone,
                teacher_id=model_id,
                dtype=torch.bfloat16,
            )
            eff_rank = getattr(bt_model.config, "rank", auto_select_lrad_rank(qwen_cfg.hidden_size, args.rank))
        else:
            logger.info(f"Building BareTorch Student on-the-fly in RAM from '{model_id}'...")
            bt_model, eff_rank = build_in_memory_baretorch_from_loaded_teacher(
                qwen_model,
                qwen_backbone,
                qwen_cfg,
                chunk_size=args.chunk_size,
                rank=args.rank,
                max_seq_len=max_ctx,
            )

        bt_model.config.max_seq_len = max(getattr(bt_model.config, "max_seq_len", 8192), max_ctx)
        if getattr(bt_model.config, "tie_word_embeddings", True):
            bt_model.lm_head.weight = bt_model.model.token_embedding.weight

        bt_label = f"BareTorch-{short_tag} (r={eff_rank}, {'Compiled' if use_compile else 'Eager'})"
        qw_label = f"{short_tag} (Native)"

        baretorch_results = {}
        qwen_results = {}

        # ----------------------------------------------------------------------
        # Phase A: Move BareTorch Student to GPU, Compile, Benchmark, and Unload
        # ----------------------------------------------------------------------
        torch._dynamo.reset()
        enable_last_logit_only_and_bypass_nan_scan(bt_model)
        bt_model.to(device)
        bt_model.eval()

        if use_compile:
            n_comp = compile_baretorch_lrad_blocks(bt_model, compile_mode=args.compile_mode)
            logger.info(f"Compiled {n_comp} CS-LRAD blocks with torch.compile(mode='{args.compile_mode}', dynamic=True).")

            logger.info("Running JIT warm-up across prefill & recurrent decode steps...")
            for warmup_len in (64, 512):
                dummy_inp = torch.randint(100, 5000, (1, warmup_len), dtype=torch.long, device=device)
                with torch.no_grad():
                    init_out = bt_model(dummy_inp, **get_forward_kwargs(bt_model))
                    p_kv = init_out.past_key_values
                    tok = init_out.logits[:, -1:].argmax(dim=-1)
                    for _ in range(16):
                        step_out = bt_model(tok, past_key_values=p_kv, use_cache=True)
                        p_kv = step_out.past_key_values
                        tok = step_out.logits[:, -1:].argmax(dim=-1)
                    del dummy_inp, init_out, p_kv, tok, step_out
            clear_cuda(device)

        clear_cuda(device)
        bt_weights_gb = torch.cuda.memory_allocated(device) / (1024.0 ** 3)
        bt_params_m = sum(p.numel() for p in bt_model.parameters()) / 1e6
        logger.info(f"{bt_label} Active Parameters: {bt_params_m:.1f}M | Static Weights VRAM: {bt_weights_gb:.2f} GB")

        for ctx in args.context_lengths:
            print(f"  --> Benchmarking {bt_label} @ {ctx:,} tokens...")
            synth_input = torch.randint(100, 10000, (1, ctx), dtype=torch.long, device=device)
            try:
                metrics = benchmark_model_at_context(
                    bt_model,
                    synth_input,
                    base_weights_gb=bt_weights_gb,
                    gen_tokens=args.gen_tokens,
                    warmup=args.warmup,
                    trials=args.trials,
                )
                metrics["params_m"] = round(bt_params_m, 1)
                baretorch_results[ctx] = metrics
                csv_rows.append({
                    "scale": short_tag,
                    "model": bt_label,
                    "params_m": round(bt_params_m, 1),
                    "compiled": use_compile,
                    "context_len": ctx,
                    "ttft_ms": round(metrics["ttft_ms"], 2),
                    "prefill_tok_s": round(metrics["prefill_tok_s"], 2),
                    "decode_tok_s": round(metrics["decode_tok_s"], 2),
                    "kv_cache_mb": round(metrics["kv_cache_mb"], 2),
                    "peak_vram_gb": round(metrics["peak_vram_gb"], 3),
                    "status": "SUCCESS",
                })
            except torch.cuda.OutOfMemoryError:
                print(f"  [OOM] {bt_label} Out-Of-Memory @ {ctx:,} tokens!")
                baretorch_results[ctx] = {"error": "OOM", "params_m": round(bt_params_m, 1)}
                csv_rows.append({
                    "scale": short_tag,
                    "model": bt_label,
                    "params_m": round(bt_params_m, 1),
                    "compiled": use_compile,
                    "context_len": ctx,
                    "ttft_ms": None,
                    "prefill_tok_s": None,
                    "decode_tok_s": None,
                    "kv_cache_mb": None,
                    "peak_vram_gb": None,
                    "status": "OOM",
                })
                clear_cuda(device)
            del synth_input

        del bt_model
        clear_cuda(device)
        logger.info(f"Unloaded {bt_label} from GPU memory.\n")

        # ----------------------------------------------------------------------
        # Phase B: Move Native Qwen3.5 Teacher to GPU, Benchmark, and Unload
        # ----------------------------------------------------------------------
        enable_last_logit_only_and_bypass_nan_scan(qwen_model)
        qwen_model.to(device)
        qwen_model.eval()

        clear_cuda(device)
        qw_weights_gb = torch.cuda.memory_allocated(device) / (1024.0 ** 3)
        qw_params_m = sum(p.numel() for p in qwen_model.parameters()) / 1e6
        logger.info(f"{qw_label} Active Parameters: {qw_params_m:.1f}M | Static Weights VRAM: {qw_weights_gb:.2f} GB")

        for ctx in args.context_lengths:
            print(f"  --> Benchmarking {qw_label} @ {ctx:,} tokens...")
            synth_input = torch.randint(100, 10000, (1, ctx), dtype=torch.long, device=device)
            try:
                metrics = benchmark_model_at_context(
                    qwen_model,
                    synth_input,
                    base_weights_gb=qw_weights_gb,
                    gen_tokens=args.gen_tokens,
                    warmup=args.warmup,
                    trials=args.trials,
                )
                metrics["params_m"] = round(qw_params_m, 1)
                qwen_results[ctx] = metrics
                csv_rows.append({
                    "scale": short_tag,
                    "model": qw_label,
                    "params_m": round(qw_params_m, 1),
                    "compiled": False,
                    "context_len": ctx,
                    "ttft_ms": round(metrics["ttft_ms"], 2),
                    "prefill_tok_s": round(metrics["prefill_tok_s"], 2),
                    "decode_tok_s": round(metrics["decode_tok_s"], 2),
                    "kv_cache_mb": round(metrics["kv_cache_mb"], 2),
                    "peak_vram_gb": round(metrics["peak_vram_gb"], 3),
                    "status": "SUCCESS",
                })
            except torch.cuda.OutOfMemoryError:
                print(f"  [OOM] {qw_label} Out-Of-Memory @ {ctx:,} tokens!")
                qwen_results[ctx] = {"error": "OOM", "params_m": round(qw_params_m, 1)}
                csv_rows.append({
                    "scale": short_tag,
                    "model": qw_label,
                    "params_m": round(qw_params_m, 1),
                    "compiled": False,
                    "context_len": ctx,
                    "ttft_ms": None,
                    "prefill_tok_s": None,
                    "decode_tok_s": None,
                    "kv_cache_mb": None,
                    "peak_vram_gb": None,
                    "status": "OOM",
                })
                clear_cuda(device)
            del synth_input

        del qwen_model, qwen_backbone
        clear_cuda(device)
        logger.info(f"Unloaded {qw_label} from GPU memory.\n")

        all_reports[short_tag] = {
            "bt_label": bt_label,
            "qw_label": qw_label,
            "bt_params_m": round(bt_params_m, 1),
            "qw_params_m": round(qw_params_m, 1),
            "bt_weights_gb": round(bt_weights_gb, 3),
            "qw_weights_gb": round(qw_weights_gb, 3),
            "baretorch": baretorch_results,
            "qwen": qwen_results,
        }

    # --------------------------------------------------------------------------
    # 3. Export CSV & JSON Reports
    # --------------------------------------------------------------------------
    fieldnames = [
        "scale",
        "model",
        "params_m",
        "compiled",
        "context_len",
        "ttft_ms",
        "prefill_tok_s",
        "decode_tok_s",
        "kv_cache_mb",
        "peak_vram_gb",
        "status",
    ]
    os.makedirs(os.path.dirname(os.path.abspath(args.csv_path)), exist_ok=True)
    with open(args.csv_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(
            {
                "device": device_name,
                "gen_tokens": args.gen_tokens,
                "context_lengths": args.context_lengths,
                "scales": all_reports,
            },
            f,
            indent=2,
        )

    logger.info(f"Saved CSV metrics to '{args.csv_path}' and JSON report to '{args.output_json}'.")

    # --------------------------------------------------------------------------
    # 4. Print Side-by-Side Markdown Summary Tables per Scale
    # --------------------------------------------------------------------------
    for short_tag, rep in all_reports.items():
        bt_label = rep["bt_label"]
        qw_label = rep["qw_label"]
        baretorch_results = rep["baretorch"]
        qwen_results = rep["qwen"]

        print("\n" + "=" * 124)
        print(
            f"COMPARATIVE HARDWARE REPORT [{short_tag}] ON {device_name} | "
            f"Student: {rep['bt_params_m']}M ({rep['bt_weights_gb']:.2f} GB) vs. "
            f"Teacher: {rep['qw_params_m']}M ({rep['qw_weights_gb']:.2f} GB)"
        )
        print("=" * 124)
        header = (
            "| Context | Model | TTFT (ms) | Prefill (tok/s) | Decode (tok/s) | "
            "KV-Cache (MB) | Peak VRAM (GB) | Speedup / Savings |"
        )
        sep = "| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |"
        print(header)
        print(sep)

        for ctx in args.context_lengths:
            bt = baretorch_results.get(ctx)
            qw = qwen_results.get(ctx)

            if bt is not None:
                if "error" in bt:
                    print(f"| {ctx:,} | **{bt_label}** | OOM | OOM | OOM | OOM | OOM | - |")
                else:
                    delta_str = "-"
                    if qw is not None and "error" not in qw:
                        pref_sp = bt["prefill_tok_s"] / max(qw["prefill_tok_s"], 1e-5)
                        dec_sp = bt["decode_tok_s"] / max(qw["decode_tok_s"], 1e-5)
                        kv_ratio = qw["kv_cache_mb"] / max(bt["kv_cache_mb"], 1e-5)
                        delta_str = f"**{pref_sp:.2f}x** Prefill, **{dec_sp:.2f}x** Decode, **{kv_ratio:.1f}x** Smaller Cache"
                    elif qw is not None and "error" in qw:
                        delta_str = "**Runs where Qwen OOMs!**"
                    print(
                        f"| {ctx:,} | **{bt_label}** | {bt['ttft_ms']:.1f} ms | "
                        f"{bt['prefill_tok_s']:,.0f} tok/s | {bt['decode_tok_s']:.1f} tok/s | "
                        f"{bt['kv_cache_mb']:.1f} MB | {bt['peak_vram_gb']:.2f} GB | {delta_str} |"
                    )

            if qw is not None:
                if "error" in qw:
                    print(f"| {ctx:,} | {qw_label} | OOM | OOM | OOM | OOM | OOM | Baseline |")
                else:
                    print(
                        f"| {ctx:,} | {qw_label} | {qw['ttft_ms']:.1f} ms | "
                        f"{qw['prefill_tok_s']:,.0f} tok/s | {qw['decode_tok_s']:.1f} tok/s | "
                        f"{qw['kv_cache_mb']:.1f} MB | {qw['peak_vram_gb']:.2f} GB | Baseline |"
                    )
            print("| --- | --- | --- | --- | --- | --- | --- | --- |")

        print("=" * 124)


if __name__ == "__main__":
    main()