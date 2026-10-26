# verify_init.py
import torch
import torch.nn.functional as F
from baretorch.cs_lrad import LRADDecoderBlock
from init_student_00 import build_and_initialize_student  # or import as module


@torch.no_grad()
def run_verification_suite(device="cuda" if torch.cuda.is_available() else "cpu"):
    print("=" * 80)
    print("RUNNING BARETORCH <-> QWEN3.5-0.8B SURGICAL INITIALIZATION VERIFICATION")
    print("=" * 80)

    teacher, t_backbone, student, tokenizer = build_and_initialize_student(save_dir=None)
    teacher.eval().to(device)
    student.eval().to(device)

    B, L = 2, 64
    torch.manual_seed(42)
    input_ids = torch.randint(0, 10000, (B, L), device=device)
    attention_mask = torch.ones((B, L), dtype=torch.long, device=device)
    position_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)

    # Dummy hidden state in bfloat16 for layerwise tests
    x_test = torch.randn(B, L, student.config.d_model, dtype=torch.bfloat16, device=device)

    # -------------------------------------------------------------------------
    # TEST 1: Token Embeddings, Final RMSNorm (+1.0 fix), and LM Head
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

    print(f"\n[TEST 1] Global Components (Embeddings, Final RMSNorm, LM Head):")
    print(f"  - Token Embedding Max Abs Diff : {emb_diff:.8f}  {'[PASS]' if emb_diff == 0.0 else '[FAIL]'}")
    print(f"  - Final RMSNorm Max Abs Diff   : {norm_diff:.8f}  {'[PASS]' if norm_diff < 1e-3 else '[FAIL]'}")
    print(f"  - LM Head Logits Max Abs Diff  : {head_diff:.8f}  {'[PASS]' if head_diff < 1e-2 else '[FAIL]'}")

    # -------------------------------------------------------------------------
    # TEST 2: All 18 CS-LRAD Blocks -> LN1, LN2, and SwiGLU MLP Equivalence
    # -------------------------------------------------------------------------
    print(f"\n[TEST 2] Checking Copied LN1, LN2, and SwiGLU MLP across all 18 CS-LRAD Layers:")
    max_ln1_diff = 0.0
    max_mlp_diff = 0.0
    lrad_count = 0

    for i, (t_layer, s_layer) in enumerate(zip(t_backbone.layers, student.model.layers)):
        if isinstance(s_layer, LRADDecoderBlock):
            lrad_count += 1
            # Compare LN1
            ln1_t = t_layer.input_layernorm(x_test)
            ln1_s = s_layer.ln1(x_test)
            d_ln1 = (ln1_t - ln1_s).abs().max().item()
            max_ln1_diff = max(max_ln1_diff, d_ln1)

            # Compare LN2 + SwiGLU MLP sub-block
            mlp_t = t_layer.mlp(t_layer.post_attention_layernorm(x_test))
            mlp_s = s_layer.mlp(s_layer.ln2(x_test))
            d_mlp = (mlp_t - mlp_s).abs().max().item()
            max_mlp_diff = max(max_mlp_diff, d_mlp)

    print(f"  - Checked {lrad_count} CS-LRAD layers:")
    print(f"  - Max LN1 Abs Diff across 18 layers       : {max_ln1_diff:.8f}  {'[PASS]' if max_ln1_diff < 1e-3 else '[FAIL]'}")
    print(f"  - Max (LN2 + MLP) Abs Diff across 18 layers: {max_mlp_diff:.8f}  {'[PASS]' if max_mlp_diff < 1e-3 else '[FAIL]'}")

    # -------------------------------------------------------------------------
    # TEST 3: All 6 Full-Attention Anchor Layers Equivalence
    # -------------------------------------------------------------------------
    print(f"\n[TEST 3] Checking 25% Full-Attention Anchor Layers (Layers 3, 7, 11, 15, 19, 23):")
    pos_embeds = student.model.rotary_emb(x_test, position_ids)
    max_full_attn_diff = 0.0
    full_count = 0

    for i, (t_layer, s_layer) in enumerate(zip(t_backbone.layers, student.model.layers)):
        if not isinstance(s_layer, LRADDecoderBlock):
            full_count += 1
            out_t = t_layer(x_test, position_ids=position_ids, position_embeddings=pos_embeds)[0]
            out_s = s_layer(x_test, position_ids=position_ids, position_embeddings=pos_embeds)[0]
            d_full = (out_t - out_s).abs().max().item()
            max_full_attn_diff = max(max_full_attn_diff, d_full)

    print(f"  - Checked {full_count} Full-Attention layers:")
    print(f"  - Max Full-Attention Layer Abs Diff       : {max_full_attn_diff:.8f}  {'[PASS]' if max_full_attn_diff == 0.0 else '[FAIL]'}")

    # -------------------------------------------------------------------------
    # TEST 4: CS-LRAD Mixer Gate Initialization & Forward Pass Sanity Check
    # -------------------------------------------------------------------------
    print(f"\n[TEST 4] Checking CS-LRAD Gate Biases & Warm-Started Mixer Forward Pass:")
    all_gates_ok = True
    all_outputs_finite = True

    for i, s_layer in enumerate(student.model.layers):
        if isinstance(s_layer, LRADDecoderBlock):
            gate_bias = s_layer.attn.W_gate.bias
            beta_bias = s_layer.attn.W_beta_gate.bias
            if not (torch.allclose(gate_bias, torch.full_like(gate_bias, 2.5)) and
                    torch.allclose(beta_bias, torch.full_like(beta_bias, -1.0))):
                all_gates_ok = False

            out_s, _ = s_layer(x_test, use_cache=False, attention_mask=attention_mask)
            if torch.isnan(out_s).any() or torch.isinf(out_s).any():
                all_outputs_finite = False

    print(f"  - W_gate.bias == +2.5 & W_beta_gate.bias == -1.0 : {'[PASS]' if all_gates_ok else '[FAIL]'}")
    print(f"  - CS-LRAD chunk_forward finite (No NaN/Inf)      : {'[PASS]' if all_outputs_finite else '[FAIL]'}")

    # -------------------------------------------------------------------------
    # TEST 5: The "Bypass Mixer" End-to-End Scaffold Proof
    # -------------------------------------------------------------------------
    print(f"\n[TEST 5] End-to-End Scaffold Proof (Substituting Teacher Mixer outputs into Student):")
    # Capture teacher linear_attn outputs during a teacher forward pass
    teacher_mixer_outs = {}
    hooks = []
    for i, t_layer in enumerate(t_backbone.layers):
        if hasattr(t_layer, "linear_attn"):
            def make_hook(idx):
                def _hook(mod, inp, out):
                    teacher_mixer_outs[idx] = out[0] if isinstance(out, tuple) else out
                return _hook
            hooks.append(t_layer.linear_attn.register_forward_hook(make_hook(i)))

    logits_t = teacher(input_ids=input_ids, attention_mask=attention_mask).logits
    for h in hooks:
        h.remove()

    #Temporarily patch Student CS-LRAD attn.forward to return teacher_mixer_outs[i]
    orig_forwards = {}
    for i, s_layer in enumerate(student.model.layers):
        if isinstance(s_layer, LRADDecoderBlock):
            orig_forwards[i] = s_layer.attn.forward
            def make_dummy_fwd(idx):
                def _dummy_fwd(x, past_state=None, use_cache=False, attention_mask=None):
                    return teacher_mixer_outs[idx], None
                return _dummy_fwd
            s_layer.attn.forward = make_dummy_fwd(i)

    logits_s_scaffold = student(input_ids=input_ids, attention_mask=attention_mask).logits

    # Restore student CS-LRAD attn.forward
    for i, orig_fn in orig_forwards.items():
        student.model.layers[i].attn.forward = orig_fn

    cos_sim = F.cosine_similarity(logits_t.float(), logits_s_scaffold.float(), dim=-1).mean().item()
    max_logit_diff = (logits_t.float() - logits_s_scaffold.float()).abs().max().item()
    print(f"  - End-to-End Scaffold Logit Cosine Similarity : {cos_sim:.6f}  {'[PASS]' if cos_sim > 0.9999 else '[FAIL]'}")
    print(f"  - End-to-End Scaffold Max Logit Abs Diff      : {max_logit_diff:.6f}  {'[PASS]' if max_logit_diff < 1e-2 else '[FAIL]'}")
    print("=" * 80)


if __name__ == "__main__":
    run_verification_suite()