# /home/martinkb/Desktop/BareTorch_Layerwise_Distill/init_student_00.py
import os
import copy
import argparse
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

from baretorch import BareTorchConfig, BareTorchForCausalLM
from baretorch.cs_lrad import LRADDecoderBlock


def load_qwen3_5_text_teacher(teacher_id: str = "Qwen/Qwen3.5-0.8B", dtype=torch.bfloat16, device="cpu"):
    """
    Loads Qwen3.5 (0.8B, 2B, 4B, 9B) as a pure causal text model (stripping the vision tower if present)
    and returns (teacher_causal_lm, text_backbone, text_config).
    """
    try:
        from transformers import Qwen3_5ForCausalLM
        teacher = Qwen3_5ForCausalLM.from_pretrained(
            teacher_id, dtype=dtype, device_map=device, trust_remote_code=True
        )
    except Exception:
        teacher = AutoModelForCausalLM.from_pretrained(
            teacher_id, torch_dtype=dtype, device_map=device, trust_remote_code=True
        )

    if hasattr(teacher, "model") and hasattr(teacher.model, "layers"):
        text_backbone = teacher.model
    elif hasattr(teacher, "model") and hasattr(teacher.model, "language_model"):
        text_backbone = teacher.model.language_model
    elif hasattr(teacher, "language_model") and hasattr(teacher.language_model, "model"):
        text_backbone = teacher.language_model.model
    else:
        raise RuntimeError(f"Could not locate .layers inside {type(teacher)}")

    t_cfg = getattr(teacher.config, "text_config", teacher.config)
    return teacher, text_backbone, t_cfg


def copy_qwen_rmsnorm_to_baretorch(bt_rmsnorm: nn.Module, qwen_rmsnorm: nn.Module):
    """
    Copies Qwen3_5RMSNorm weights directly into BareTorch RMSNorm with unit_offset=True
    so (1.0 + weight.float()) is computed in fp32 inside forward() with 0.0 rounding loss.
    """
    cls_name = qwen_rmsnorm.__class__.__name__
    is_zero_centered = ("Qwen3_5RMSNorm" in cls_name) or (qwen_rmsnorm.weight.abs().mean().item() < 0.5)
    bt_rmsnorm.unit_offset = is_zero_centered
    bt_rmsnorm.weight.copy_(qwen_rmsnorm.weight)
    bt_rmsnorm.eps = getattr(qwen_rmsnorm, "eps", getattr(qwen_rmsnorm, "variance_epsilon", 1e-6))


def downsample_rows(weight: torch.Tensor, target_rows: int) -> torch.Tensor:
    if weight.shape[0] == target_rows:
        return weight
    ratio = weight.shape[0] // target_rows
    return (weight.float().view(target_rows, ratio, weight.shape[1]).mean(dim=1) * (ratio ** 0.5)).to(weight.dtype)


def downsample_cols(weight: torch.Tensor, target_cols: int) -> torch.Tensor:
    if weight.shape[1] == target_cols:
        return weight
    ratio = weight.shape[1] // target_cols
    return (weight.float().view(weight.shape[0], target_cols, ratio).mean(dim=2) * (ratio ** 0.5)).to(weight.dtype)


def warm_start_cs_lrad_from_qwen_deltanet(s_attn, t_mixer):
    copied = []
    inner_dim = s_attn.inner_dim

    if hasattr(t_mixer, "in_proj_qkv"):
        qkv_w = t_mixer.in_proj_qkv.weight
        key_dim = getattr(t_mixer, "key_dim", qkv_w.shape[0] // 3)
        val_dim = getattr(t_mixer, "value_dim", qkv_w.shape[0] - 2 * key_dim)

        q_w = qkv_w[:key_dim, :]
        k_w = qkv_w[key_dim : 2 * key_dim, :]
        v_w = qkv_w[2 * key_dim : 2 * key_dim + val_dim, :]

        s_attn.W_q.weight.copy_(downsample_rows(q_w, inner_dim))
        s_attn.W_k.weight.copy_(downsample_rows(k_w, inner_dim))
        s_attn.W_v.weight.copy_(downsample_rows(v_w, inner_dim))
        copied.append("in_proj_qkv->(W_q,W_k,W_v)")

    if hasattr(t_mixer, "in_proj_z"):
        z_w = t_mixer.in_proj_z.weight
        s_attn.W_swish_gate.weight.copy_(downsample_rows(z_w, inner_dim))
        copied.append("in_proj_z->W_swish_gate")

    if hasattr(t_mixer, "out_proj"):
        out_w = t_mixer.out_proj.weight
        s_attn.W_out.weight.copy_(downsample_cols(out_w, inner_dim))
        copied.append("out_proj->W_out")

    s_attn._init_gate_parameters()
    return copied


@torch.no_grad()
def build_and_initialize_student(
    teacher_id: str = "Qwen/Qwen3.5-0.8B",
    save_dir: str = "./checkpoints/stage0_initialized",
    chunk_size: int = 64,
    rank: int = 16,
    splice_exact_qwen_full_attn: bool = True,
):
    print(f"1. Loading Teacher: {teacher_id}")
    tokenizer = AutoTokenizer.from_pretrained(teacher_id, trust_remote_code=True)
    teacher, t_backbone, t_cfg = load_qwen3_5_text_teacher(teacher_id, dtype=torch.bfloat16, device="cpu")

    layer_types = []
    for i, t_layer in enumerate(t_backbone.layers):
        if hasattr(t_cfg, "layer_types") and t_cfg.layer_types is not None:
            is_full = "full" in str(t_cfg.layer_types[i]).lower()
        else:
            is_full = hasattr(t_layer, "self_attn") and not hasattr(t_layer, "linear_attn")
        layer_types.append("transformer" if is_full else "cs_lrad")

    num_lrad = sum(1 for x in layer_types if x == "cs_lrad")
    num_full = sum(1 for x in layer_types if x == "transformer")
    num_heads = getattr(t_cfg, "num_attention_heads", 16)
    num_kv_heads = getattr(t_cfg, "num_key_value_heads", 2)
    print(
        f"2. Detected {teacher_id} Topology: {num_lrad} CS-LRAD layers + {num_full} Full-Attention layers "
        f"(d_model={t_cfg.hidden_size}, heads={num_heads}, kv_heads={num_kv_heads}, rank={rank})"
    )

    bt_config = BareTorchConfig(
        vocab_size=t_cfg.vocab_size,
        d_model=t_cfg.hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        num_layers=t_cfg.num_hidden_layers,
        intermediate_size=t_cfg.intermediate_size,
        chunk_size=chunk_size,
        rank=rank,
        dropout=0.0,
        max_seq_len=8192,
        use_grad_checkpointing=False,
        use_qk_norm=True,
        rms_norm_eps=getattr(t_cfg, "rms_norm_eps", 1e-6),
        rms_norm_unit_offset=True,
        logit_soft_cap=None,
        layer_types=layer_types,
        tie_word_embeddings=getattr(t_cfg, "tie_word_embeddings", True),
    )

    print("3. Instantiating Hybrid BareTorch Student...")
    student = BareTorchForCausalLM(bt_config).to(torch.bfloat16)

    # Copy Embeddings, Final Norm, and LM Head
    student.model.token_embedding.weight.copy_(t_backbone.embed_tokens.weight)
    copy_qwen_rmsnorm_to_baretorch(student.model.final_norm, t_backbone.norm)
    student.lm_head.weight.copy_(teacher.lm_head.weight)

    if hasattr(t_backbone, "rotary_emb"):
        student.model.rotary_emb = copy.deepcopy(t_backbone.rotary_emb)

    for i, (l_type, t_layer) in enumerate(zip(layer_types, t_backbone.layers)):
        if l_type == "transformer":
            if splice_exact_qwen_full_attn:
                student.model.layers[i] = copy.deepcopy(t_layer)
                print(f"  [Layer {i:02d}] Spliced intact Qwen3_5 Full-Attention Layer")
        else:
            s_layer: LRADDecoderBlock = student.model.layers[i]
            copy_qwen_rmsnorm_to_baretorch(s_layer.ln1, t_layer.input_layernorm)
            copy_qwen_rmsnorm_to_baretorch(s_layer.ln2, t_layer.post_attention_layernorm)

            s_layer.mlp.w1.weight.copy_(t_layer.mlp.gate_proj.weight)
            s_layer.mlp.w2.weight.copy_(t_layer.mlp.up_proj.weight)
            s_layer.mlp.w3.weight.copy_(t_layer.mlp.down_proj.weight)

            t_mixer = getattr(t_layer, "linear_attn", None)
            copied_info = warm_start_cs_lrad_from_qwen_deltanet(s_layer.attn, t_mixer)
            print(f"  [Layer {i:02d}] Initialized CS-LRAD Block (Copied Norms, MLP, Warm-Start: {copied_info})")

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        student.config.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        torch.save(student.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))
        print(f"\nSaved initialized hybrid student to: {save_dir}")

    return teacher, t_backbone, student, tokenizer


def load_hybrid_student_checkpoint(ckpt_dir: str, teacher_backbone=None, teacher_id: str = "Qwen/Qwen3.5-0.8B", dtype=torch.bfloat16):
    """
    Reconstructs the BareTorchForCausalLM hybrid architecture (including spliced Qwen3_5DecoderLayer
    full-attention blocks and rotary_emb) and loads the saved state_dict from ckpt_dir.
    """
    if teacher_backbone is None:
        _, teacher_backbone, _ = load_qwen3_5_text_teacher(teacher_id, dtype=dtype, device="cpu")

    bt_config = BareTorchConfig.from_pretrained(ckpt_dir)
    student = BareTorchForCausalLM(bt_config).to(dtype)

    if hasattr(teacher_backbone, "rotary_emb"):
        student.model.rotary_emb = copy.deepcopy(teacher_backbone.rotary_emb)

    for i, l_type in enumerate(bt_config.layer_types):
        if l_type == "transformer":
            student.model.layers[i] = copy.deepcopy(teacher_backbone.layers[i])

    state_dict = torch.load(os.path.join(ckpt_dir, "pytorch_model.bin"), map_location="cpu", weights_only=True)
    student.load_state_dict(state_dict, strict=True)
    return student


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 0: Initialize BareTorch Hybrid Student from Qwen3.5 Teacher")
    parser.add_argument("--teacher_id", type=str, default="Qwen/Qwen3.5-0.8B", help="Base Teacher HF ID")
    parser.add_argument("--save_dir", type=str, default="./checkpoints/stage0_initialized", help="Output checkpoint dir")
    parser.add_argument("--chunk_size", type=int, default=64, help="CS-LRAD chunk size")
    parser.add_argument("--rank", type=int, default=16, help="CS-LRAD low-rank state factor r (16 for 0.8B, 32 for 2B)")
    args = parser.parse_args()

    build_and_initialize_student(
        teacher_id=args.teacher_id,
        save_dir=args.save_dir,
        chunk_size=args.chunk_size,
        rank=args.rank,
    )