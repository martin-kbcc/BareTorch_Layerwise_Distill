# /home/martinkb/Desktop/BareTorch_Layerwise_Distill/verify_init.py
import os
import sys
import argparse
import torch
import torch.nn.functional as F
from baretorch.cs_lrad import LRADDecoderBlock
from init_student_00 import (
    load_qwen3_5_text_teacher,
    load_hybrid_student_checkpoint,
    build_and_initialize_student,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Universal BareTorch <-> Qwen3.5 (0.8B / 2B / 4B / 9B) Surgical Initialization & Cache Verifier"
    )
    parser.add_argument("--teacher_id", type=str, default="Qwen/Qwen3.5-2B", help="Base Qwen3.5 Teacher HF ID")
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="./checkpoints/stage0_initialized",
        help="Path to saved Stage 0 checkpoint to verify from disk (if missing, initializes in-memory)",
    )
    parser.add_argument("--chunk_size", type=int, default=64)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


@torch.no_grad()
def run_verification_suite(args):
    device = args.device
    print("=" * 90)
    print(f"RUNNING BARETORCH <-> {args.teacher_id} SURGICAL INITIALIZATION & CACHE VERIFICATION")
    print("=" * 90)

    if args.ckpt_dir and os.path.exists(os.path.join(args.ckpt_dir, "pytorch_model.bin")):
        print(f"Loading Teacher ({args.teacher_id}) and Saved Disk Checkpoint ({args.ckpt_dir})...")
        teacher, t_backbone, _ = load_qwen3_5_text_teacher(args.teacher_id, dtype=torch.bfloat16, device="cpu")
        student = load_hybrid_student_checkpoint(
            args.ckpt_dir, teacher_backbone=t_backbone, teacher_id=args.teacher_id, dtype=torch.bfloat16
        )
    else:
        print(f"Checkpoint not found at {args.ckpt_dir}; building in-memory from {args.teacher_id}...")
        teacher, t_backbone, student, _ = build_and_initialize_student(
            teacher_id=args.teacher_id,
            save_dir=None,
            chunk_size=args.chunk_size,
            rank=args.rank,
        )

    teacher.eval().to(device)
    student.eval().to(device)

    all_passed = True
    B, L = 2, 64
    torch.manual_seed(42)
    input_ids = torch.randint(10, 10000, (B, L), device=device)
    attention_mask = torch.ones((B, L), dtype=torch.long, device=device)
    position_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)

    # Dummy hidden state in bfloat16 matching student.config.d_model (1024 for 0.8B, 2048 for 2B, etc.)
    x_test = torch.randn(B, L, student.config.d_model, dtype=torch.bfloat16, device=device)

    # -------------------------------------------------------------------------
    # TEST 1: Token Embeddings, Final RMSNorm (+1.0 unit_offset), and LM Head
    # -------------------------------------------------------------------------
    emb_t = t_backbone.embed_tokens(input_ids)
    emb_s = student.model.token_embedding(input_ids)
    emb_diff = (emb_t - emb_s).abs().max().item()

    norm_t = t_backbone.norm(x_test)
    norm_s = student.model.final_norm(x_test)
    norm_diff = (norm_t - norm_s).abs().max().item()

    head_t = teacher.lm_head(norm_t)
    head_s = student.lm_head(norm_s)
    head_diff = (head_t - head_s).abs().max().item()

    t1_ok = (emb_diff == 0.0) and (norm_diff < 1e-3) and (head_diff < 1e-2)
    all_passed = all_passed and t1_ok
    print(f"\n[TEST 1] Global Components (d_model={student.config.d_model}, vocab={student.config.vocab_size}):")
    print(f"  - Token Embedding Max Abs Diff : {emb_diff:.8f}  {'[PASS]' if emb_diff == 0.0 else '[FAIL]'}")
    print(f"  - Final RMSNorm Max Abs Diff   : {norm_diff:.8f}  {'[PASS]' if norm_diff < 1e-3 else '[FAIL]'}")
    print(f"  - LM Head Logits Max Abs Diff  : {head_diff:.8f}  {'[PASS]' if head_diff < 1e-2 else '[FAIL]'}")

    # -------------------------------------------------------------------------
    # TEST 2: All CS-LRAD Blocks -> LN1, LN2, and SwiGLU MLP Equivalence
    # -------------------------------------------------------------------------
    max_ln1_diff = 0.0
    max_mlp_diff = 0.0
    lrad_indices = []

    for i, (t_layer, s_layer) in enumerate(zip(t_backbone.layers, student.model.layers)):
        if isinstance(s_layer, LRADDecoderBlock):
            lrad_indices.append(i)
            ln1_t = t_layer.input_layernorm(x_test)
            ln1_s = s_layer.ln1(x_test)
            max_ln1_diff = max(max_ln1_diff, (ln1_t - ln1_s).abs().max().item())

            mlp_t = t_layer.mlp(t_layer.post_attention_layernorm(x_test))
            mlp_s = s_layer.mlp(s_layer.ln2(x_test))
            max_mlp_diff = max(max_mlp_diff, (mlp_t - mlp_s).abs().max().item())

    t2_ok = (max_ln1_diff < 1e-3) and (max_mlp_diff < 1e-3) and (len(lrad_indices) > 0)
    all_passed = all_passed and t2_ok
    print(f"\n[TEST 2] Copied LN1, LN2, and SwiGLU MLP across {len(lrad_indices)} CS-LRAD Layers {lrad_indices}:")
    print(f"  - Max LN1 Abs Diff             : {max_ln1_diff:.8f}  {'[PASS]' if max_ln1_diff < 1e-3 else '[FAIL]'}")
    print(f"  - Max (LN2 + MLP) Abs Diff     : {max_mlp_diff:.8f}  {'[PASS]' if max_mlp_diff < 1e-3 else '[FAIL]'}")

    # -------------------------------------------------------------------------
    # TEST 3: All Full-Attention Anchor Layers Equivalence
    # -------------------------------------------------------------------------
    pos_embeds = student.model.rotary_emb(x_test, position_ids)
    max_full_attn_diff = 0.0
    full_indices = []

    for i, (t_layer, s_layer) in enumerate(zip(t_backbone.layers, student.model.layers)):
        if not isinstance(s_layer, LRADDecoderBlock):
            full_indices.append(i)
            out_t = t_layer(x_test, position_ids=position_ids, position_embeddings=pos_embeds)[0]
            out_s = s_layer(x_test, position_ids=position_ids, position_embeddings=pos_embeds)[0]
            max_full_attn_diff = max(max_full_attn_diff, (out_t - out_s).abs().max().item())

    t3_ok = (max_full_attn_diff == 0.0) and (len(full_indices) > 0)
    all_passed = all_passed and t3_ok
    print(f"\n[TEST 3] Spliced Full-Attention Anchor Layers ({len(full_indices)} layers: {full_indices}):")
    print(f"  - Max Full-Attention Abs Diff  : {max_full_attn_diff:.8f}  {'[PASS]' if t3_ok else '[FAIL]'}")

    # -------------------------------------------------------------------------
    # TEST 4: CS-LRAD Mixer Gate Initialization & Forward Pass Sanity Check
    # -------------------------------------------------------------------------
    all_gates_ok = True
    all_outputs_finite = True

    for i, s_layer in enumerate(student.model.layers):
        if isinstance(s_layer, LRADDecoderBlock):
            gate_bias = s_layer.attn.W_gate.bias
            beta_bias = s_layer.attn.W_beta_gate.bias
            if not (
                torch.allclose(gate_bias, torch.full_like(gate_bias, 2.5))
                and torch.allclose(beta_bias, torch.full_like(beta_bias, -1.0))
            ):
                all_gates_ok = False

            out_s, _ = s_layer(x_test, use_cache=False, attention_mask=attention_mask)
            if torch.isnan(out_s).any() or torch.isinf(out_s).any():
                all_outputs_finite = False

    t4_ok = all_gates_ok and all_outputs_finite
    all_passed = all_passed and t4_ok
    print(f"\n[TEST 4] CS-LRAD Gate Biases (rank={student.config.rank}, chunk={student.config.chunk_size}) & Finite Forward:")
    print(f"  - W_gate.bias == +2.5 & W_beta_gate.bias == -1.0 : {'[PASS]' if all_gates_ok else '[FAIL]'}")
    print(f"  - CS-LRAD chunk_forward finite (No NaN/Inf)      : {'[PASS]' if all_outputs_finite else '[FAIL]'}")

    # -------------------------------------------------------------------------
    # TEST 5: The "Bypass Mixer" End-to-End Scaffold Proof
    # -------------------------------------------------------------------------
    teacher_mixer_outs = {}
    hooks = []
    for i, t_layer in enumerate(t_backbone.layers):
        t_mixer = getattr(t_layer, "linear_attn", None)
        if t_mixer is not None:
            def make_hook(idx):
                def _hook(mod, inp, out):
                    teacher_mixer_outs[idx] = out[0] if isinstance(out, tuple) else out
                return _hook
            hooks.append(t_mixer.register_forward_hook(make_hook(i)))

    logits_t = teacher(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
    for h in hooks:
        h.remove()

    orig_forwards = {}
    for i, s_layer in enumerate(student.model.layers):
        if isinstance(s_layer, LRADDecoderBlock):
            orig_forwards[i] = s_layer.attn.forward
            def make_dummy_fwd(idx):
                def _dummy_fwd(x, past_state=None, use_cache=False, attention_mask=None):
                    return teacher_mixer_outs[idx], None
                return _dummy_fwd
            s_layer.attn.forward = make_dummy_fwd(i)

    logits_s_scaffold = student(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits

    for i, orig_fn in orig_forwards.items():
        student.model.layers[i].attn.forward = orig_fn

    cos_sim = F.cosine_similarity(logits_t.float(), logits_s_scaffold.float(), dim=-1).mean().item()
    max_logit_diff = (logits_t.float() - logits_s_scaffold.float()).abs().max().item()
    t5_ok = (cos_sim > 0.9999) and (max_logit_diff < 1e-2)
    all_passed = all_passed and t5_ok
    print(f"\n[TEST 5] End-to-End Scaffold Proof (Substituting Teacher Mixer outputs into Student):")
    print(f"  - Scaffold Logit Cosine Similarity : {cos_sim:.6f}  {'[PASS]' if cos_sim > 0.9999 else '[FAIL]'}")
    print(f"  - Scaffold Max Logit Abs Diff      : {max_logit_diff:.6f}  {'[PASS]' if max_logit_diff < 1e-2 else '[FAIL]'}")

    # -------------------------------------------------------------------------
    # TEST 6: Cross-Chunk Autoregressive Cache Equivalence (Prefill 70 + Step 5 vs Full 75)
    # -------------------------------------------------------------------------
    ids_75 = torch.randint(10, 10000, (1, 75), device=device)
    full_logits = student(ids_75, use_cache=False).logits[:, 69:, :]
    out_prefill = student(ids_75[:, :70], use_cache=True)
    pkv = out_prefill.past_key_values
    step_logits = [out_prefill.logits[:, -1:, :]]
    for t in range(70, 75):
        out_step = student(ids_75[:, t : t + 1], past_key_values=pkv, use_cache=True)
        pkv = out_step.past_key_values
        step_logits.append(out_step.logits)
    step_logits = torch.cat(step_logits, dim=1)

    cache_max_diff = (full_logits.float() - step_logits.float()).abs().max().item()
    top1_agree = (full_logits.argmax(-1) == step_logits.argmax(-1)).float().mean().item()
    t6_ok = top1_agree == 1.0 and cache_max_diff < 0.5
    all_passed = all_passed and t6_ok
    print(f"\n[TEST 6] Autoregressive Generation Cache Equivalence (Cross-Chunk 70->75 tokens):")
    print(f"  - Top-1 Token Agreement (Full vs Step) : {top1_agree*100:.1f}% (Max Diff: {cache_max_diff:.4f})  {'[PASS]' if t6_ok else '[FAIL]'}")
    print("=" * 90)

    if not all_passed:
        print("ERROR: One or more surgical initialization / cache verification tests FAILED!")
        sys.exit(1)
    print("ALL 6 SURGICAL INITIALIZATION & AUTOREGRESSIVE CACHE TESTS PASSED 100%!")


if __name__ == "__main__":
    args = parse_args()
    run_verification_suite(args)