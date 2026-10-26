# /home/martinkb/Desktop/BareTorch_Layerwise_Distill/train_stage4a_updistill_sft_05.py
import os
import re
import gc
import time
import random
import shutil
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.utils.data import DataLoader
from accelerate import Accelerator
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict, concatenate_datasets

from baretorch.cs_lrad import LRADDecoderBlock
from init_student_00 import load_qwen3_5_text_teacher, load_hybrid_student_checkpoint

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 4A: Full 5-Pillar 8K Focal Answer-Weighted + Cross-Chunk SFT & Up-Distillation")
    # Models & Paths
    parser.add_argument("--teacher_id", type=str, default="Qwen/Qwen3.5-0.8B", help="Teacher HF ID (Qwen/Qwen3.5-0.8B, Qwen/Qwen3.5-2B, or Qwen/Qwen3.5-4B)")
    parser.add_argument("--student_base_id", type=str, default="Qwen/Qwen3.5-0.8B", help="Base 0.8B model ID used to initialize Student structure")
    parser.add_argument("--stage3_ckpt_dir", type=str, default="./checkpoints/stage3_final_distilled/best", help="Path to Stage 3 student checkpoint")
    parser.add_argument("--pretrain_data_dir", type=str, default="./data/fineweb_edu_packed_2048", help="Path to existing FineWeb-Edu dataset for 22% replay")
    parser.add_argument("--sft_data_dir", type=str, default="./data/stage4a_5pillar_8k_focal_2048", help="Disk path for expanded 8K 5-Pillar focal-weighted SFT dataset")
    parser.add_argument("--cache_dir", type=str, default="./data/hf_cache_5pillar_8k_sft", help="Temporary Arrow disk cache directory")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/stage4a_5pillar_8k_0.8B", help="Directory to save Stage 4A 8K checkpoints")

    # Dataset Construction Parameters
    parser.add_argument("--num_raw_samples", type=int, default=560000, help="Base instruction pairs from SmolTalk, OpenHermes-2.5, and SlimOrca")
    parser.add_argument("--mmlu_aux_samples", type=int, default=100000, help="Unlock all 99,842 official MMLU auxiliary_train examples")
    parser.add_argument("--replay_ratio", type=float, default=0.28, help="FineWeb-Edu replay ratio relative to SFT blocks (0.28 => ~22% of combined pool)")
    parser.add_argument("--mc_Answer_weight", type=float, default=12.0, help="Focal gradient multiplier on single-token MMLU (A/B/C/D) decision tokens")
    parser.add_argument("--short_answer_weight", type=float, default=4.0, help="Focal gradient multiplier on concise ARC/QA/Cloze response tokens (<= 32 tokens)")
    parser.add_argument("--cross_chunk_weight", type=float, default=1.5, help="Extra multiplier when tokens cross a 64-token CS-LRAD chunk boundary from prompt start")
    parser.add_argument("--lrad_chunk_size", type=int, default=64, help="CS-LRAD chunk boundary interval for cross-chunk focal weighting")
    parser.add_argument("--val_blocks", type=int, default=500, help="Number of 2048-token blocks reserved for validation (~1.02M tokens)")
    parser.add_argument("--seq_len", type=int, default=2048, help="Bin-packed sequence length (matches Stages 1-3)")
    parser.add_argument("--num_proc", type=int, default=min(16, max(1, os.cpu_count() // 2)), help="Parallel CPU workers for dataset building")

    # Training Hyperparameters
    parser.add_argument("--lrad_lr", type=float, default=8e-5, help="Peak LR for CS-LRAD layers in Stage 4A")
    parser.add_argument("--backbone_lr", type=float, default=1.5e-5, help="Peak LR for Backbone MLPs & Full-Attn in Stage 4A (1.5e-5 preserves pretrain calibration)")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--micro_batch_size", type=int, default=2)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=8000, help="8000 steps of 100% unique observations (<1.0 epoch, ~262.1M tokens)")
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # Pure Logit Distillation + Focal Completion-Masked CE Weights
    parser.add_argument("--kl_weight", type=float, default=0.5, help="Weight for full-vocab Teacher Logit KL distillation")
    parser.add_argument("--ce_weight", type=float, default=0.5, help="Weight for completion-only response Cross-Entropy")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--logit_chunk_size", type=int, default=128)

    # Logging, Evaluation & Saving
    parser.add_argument("--log_steps", type=int, default=25)
    parser.add_argument("--eval_steps", type=int, default=250)
    parser.add_argument("--eval_batches", type=int, default=-1, help="-1 = evaluate entire validation split")
    parser.add_argument("--save_steps", type=int, default=500)

    # Hardware & Compilation
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--compile_mode", type=str, default="default", choices=["default", "reduce-overhead", "max-autotune"])
    parser.add_argument("--no_compile", action="store_true", help="Disable torch.compile on CS-LRAD layers")
    parser.add_argument("--no_grad_checkpointing", action="store_true", help="Disable gradient checkpointing on Student")
    parser.add_argument("--seed", type=int, default=101)
    return parser.parse_args()


# Regex patterns compiled once at module level for high-speed streaming extraction
OPTION_LINE_RE = re.compile(r"(?:^|\n)\s*(?:[\(\[]?([A-D])[\)\]\.\:\-]\s+)(.+?)(?=(?:\n\s*[\(\[]?[A-E][\)\]\.\:\-]\s+)|\Z)", re.DOTALL)
EXPLICIT_LETTER_RE = re.compile(
    r"(?:answer is|correct answer is|option is|choice is|answer\s*:|therefore\s*,?\s*the answer is|^)\s*\**[\(\[]?([A-D])[\)\]\.\:]?\**(?:\s|$|\.)",
    re.IGNORECASE | re.MULTILINE,
)
STEM_BOXED_OR_FINAL_RE = re.compile(
    r"(?:\\boxed\{\s*([A-D])\s*\}|(?:final answer|correct option|answer)\s*(?:is|:)\s*\**[\(\[]?([A-D])[\)\]]?\**|\*\*([A-D])\*\*\s*$)",
    re.IGNORECASE | re.MULTILINE,
)


def format_dual_mc_pair(stem: str, opt_map: dict, chosen_letter: str):
    """
    Given a clean question stem, a dictionary of options `{'A': ..., 'B': ...}`, and `chosen_letter`,
    returns both:
      1. `mmlu_raw`: `Question\nA. ...\nB. ...\nAnswer:` -> ` B` (when chosen_letter is in A..D)
      2. `arc_raw` : `Question: ...\nAnswer:` -> ` <Full Option Text>`
    """
    stem = re.sub(r"^(?:Q:|Question:)\s*", "", stem.strip(), flags=re.IGNORECASE).strip()
    if len(stem) < 6 or chosen_letter not in opt_map:
        return []

    results = []
    if chosen_letter in ("A", "B", "C", "D"):
        opts_formatted = "\n".join([f"{k}. {opt_map[k]}" for k in ("A", "B", "C", "D") if k in opt_map])
        mmlu_prompt = f"{stem}\n{opts_formatted}\nAnswer:"
        mmlu_target = f" {chosen_letter}"
        results.append((mmlu_prompt, mmlu_target, "mmlu_raw"))

    chosen_text = opt_map[chosen_letter].strip()
    if len(chosen_text) >= 2 and not chosen_text.lower().startswith(("all of the above", "none of the above", "both a and b")):
        arc_prompt = f"Question: {stem}\nAnswer:"
        arc_target = f" {chosen_text}"
        results.append((arc_prompt, arc_target, "arc_raw"))

    return results


def extract_mc_mmlu_and_arc_pairs(user_txt: str, asst_txt: str):
    """
    Broadly detects multiple-choice questions in general instruction datasets (OpenHermes-2.5,
    SlimOrca, SmolTalk) covering (A)/(B)/(C)/(D), A./B./C./D., A)/B)/C)/D), and STEM \\boxed{B}.
    """
    matches = OPTION_LINE_RE.findall(user_txt)
    if len(matches) < 3:
        return []

    opt_map = {}
    for letter, text in matches:
        clean_opt = " ".join(text.strip().split())
        if clean_opt:
            opt_map[letter.upper()] = clean_opt

    if not ("A" in opt_map and "B" in opt_map and "C" in opt_map):
        return []

    chosen_letter = None
    asst_clean = asst_txt.strip()

    # 1. Try explicit letter match
    m = EXPLICIT_LETTER_RE.search(asst_clean)
    if m:
        cand = m.group(1).upper()
        if cand in opt_map:
            chosen_letter = cand

    # 2. Try STEM boxed or end-of-CoT conclusion match
    if chosen_letter is None:
        m_stem = STEM_BOXED_OR_FINAL_RE.search(asst_clean)
        if m_stem:
            cand = (m_stem.group(1) or m_stem.group(2) or m_stem.group(3) or "").upper()
            if cand in opt_map:
                chosen_letter = cand

    # 3. Fallback: check if assistant response matches one of the option texts
    if chosen_letter is None and len(asst_clean) < 200:
        asst_lower = asst_clean.lower().rstrip(".")
        for let, o_txt in opt_map.items():
            o_lower = o_txt.lower().rstrip(".")
            if len(o_lower) >= 3 and (asst_lower == o_lower or asst_lower.endswith(" " + o_lower)):
                chosen_letter = let
                break

    if chosen_letter is None:
        return []

    first_opt_match = re.search(r"(?:^|\n)\s*[\(\[]?A[\)\]\.\:\-]\s+", user_txt)
    if not first_opt_match:
        return []
    stem = user_txt[:first_opt_match.start()].strip()
    return format_dual_mc_pair(stem, opt_map, chosen_letter)


def prepare_or_load_sft_dataset(args, tokenizer, accelerator):
    """
    Checks if `args.sft_data_dir` exists on disk AND contains enough blocks for a strict
    single-epoch run (`<= 1.0` epoch across `args.max_steps`).
    If not, Rank 0 builds the expanded 5-Pillar + Commonsense Focal SFT dataset in parallel on CPU.
    """
    blocks_per_step = args.micro_batch_size * accelerator.num_processes * args.grad_accum_steps
    min_required_train_blocks = args.max_steps * blocks_per_step

    need_build = not os.path.exists(args.sft_data_dir)
    if not need_build:
        try:
            existing_ds = load_from_disk(args.sft_data_dir)
            existing_train_len = len(existing_ds["train"])
            if existing_train_len < min_required_train_blocks:
                if accelerator.is_main_process:
                    print(
                        f"[Dataset Check] Existing dataset at `{args.sft_data_dir}` has {existing_train_len:,} blocks "
                        f"(< {min_required_train_blocks:,} needed for {args.max_steps} unique steps). Rebuilding!"
                    )
                need_build = True
        except Exception:
            need_build = True

    if need_build:
        if accelerator.is_main_process:
            if os.path.exists(args.sft_data_dir):
                shutil.rmtree(args.sft_data_dir)
            os.makedirs(args.cache_dir, exist_ok=True)

            n_smol = int(args.num_raw_samples * 0.35)
            n_hermes = int(args.num_raw_samples * 0.35)
            n_orca = int(args.num_raw_samples * 0.30)

            print("=" * 85)
            print(f"[Dataset Builder] Building Expanded 8K 5-Pillar + Commonsense Pool at `{args.sft_data_dir}` ({args.num_proc} CPU workers)...")
            print(f"  - General Stream  : {args.num_raw_samples:,} (SmolTalk: {n_smol:,} | OpenHermes-2.5: {n_hermes:,} | SlimOrca: {n_orca:,})")
            print(f"  - Dedicated MC/QA : Full MMLU aux_train (up to {args.mmlu_aux_samples:,}) + SciQ + ARC + OBQA + CSQA + Winogrande + PIQA")
            print(f"  - Focal Weights   : MMLU Letter={args.mc_Answer_weight}x | Concise ARC/QA/Cloze={args.short_answer_weight}x | Cross-Chunk={args.cross_chunk_weight}x")
            print("=" * 85)

            def raw_pairs_generator():
                mmlu_cnt, arc_cnt, comp_cnt = 0, 0, 0
                rng = random.Random(args.seed)
                letters = ["A", "B", "C", "D"]

                # -------------------------------------------------------------------------
                # PILLAR 1: Dedicated Decontaminated Academic, STEM & Commonsense Splits
                # -------------------------------------------------------------------------
                # 1A. Official MMLU auxiliary_train split (all 99,842 questions)
                try:
                    mmlu_aux = load_dataset("cais/mmlu", "all", split="auxiliary_train", streaming=True)
                    c_aux = 0
                    for item in mmlu_aux:
                        if c_aux >= args.mmlu_aux_samples:
                            break
                        row = item.get("train", item) if isinstance(item, dict) else item
                        q = row.get("question", "")
                        choices = row.get("choices", [])
                        ans_idx = row.get("answer", -1)
                        if q and len(choices) == 4 and 0 <= int(ans_idx) < 4:
                            opt_map = {letters[i]: str(choices[i]).strip() for i in range(4)}
                            chosen = letters[int(ans_idx)]
                            for p_mc, a_mc, m_mc in format_dual_mc_pair(q, opt_map, chosen):
                                yield {"user": p_mc, "assistant": a_mc, "mode": m_mc}
                                if m_mc == "mmlu_raw":
                                    mmlu_cnt += 1
                                else:
                                    arc_cnt += 1
                            c_aux += 1
                    print(f"  [Pillar 1A] Loaded {c_aux:,} official `cais/mmlu` auxiliary_train questions.")
                except Exception as e:
                    print(f"  [Warning] Could not stream `cais/mmlu` auxiliary_train: {e}")

                # 1B. AllenAI SciQ train split (11,679 STEM Physics/Chemistry/Biology MC questions)
                try:
                    sciq_tr = load_dataset("allenai/sciq", split="train", streaming=True)
                    c_sciq = 0
                    for item in sciq_tr:
                        q = item.get("question", "").strip()
                        corr = item.get("correct_answer", "").strip()
                        d1 = item.get("distractor1", "").strip()
                        d2 = item.get("distractor2", "").strip()
                        d3 = item.get("distractor3", "").strip()
                        supp = item.get("support", "").strip()
                        if q and corr and d1 and d2 and d3:
                            opts = [corr, d1, d2, d3]
                            rng.shuffle(opts)
                            chosen = letters[opts.index(corr)]
                            opt_map = {letters[i]: opts[i] for i in range(4)}
                            for p_mc, a_mc, m_mc in format_dual_mc_pair(q, opt_map, chosen):
                                yield {"user": p_mc, "assistant": a_mc, "mode": m_mc}
                                yield {"user": p_mc, "assistant": a_mc, "mode": m_mc}
                                if m_mc == "mmlu_raw":
                                    mmlu_cnt += 2
                                else:
                                    arc_cnt += 2
                            if supp:
                                q_ctx = f"{supp}\n\n{q}"
                                for p_mc, a_mc, m_mc in format_dual_mc_pair(q_ctx, opt_map, chosen):
                                    yield {"user": p_mc, "assistant": a_mc, "mode": m_mc}
                                    if m_mc == "mmlu_raw":
                                        mmlu_cnt += 1
                                    else:
                                        arc_cnt += 1
                            c_sciq += 1
                    print(f"  [Pillar 1B] Loaded {c_sciq:,} `allenai/sciq` STEM train questions (with context & no-context variants).")
                except Exception as e:
                    print(f"  [Warning] Could not stream `allenai/sciq`: {e}")

                # 1C. AllenAI AI2_ARC (ARC-Challenge + ARC-Easy), OpenBookQA, and tau/commonsense_qa train splits
                for ds_name, cfg_name in [
                    ("allenai/ai2_arc", "ARC-Challenge"),
                    ("allenai/ai2_arc", "ARC-Easy"),
                    ("allenai/openbookqa", "main"),
                    ("tau/commonsense_qa", "default"),
                ]:
                    try:
                        arc_ds = (
                            load_dataset(ds_name, split="train", streaming=True)
                            if cfg_name == "default"
                            else load_dataset(ds_name, cfg_name, split="train", streaming=True)
                        )
                        c_sub = 0
                        for item in arc_ds:
                            q = item.get("question", "") or item.get("question_stem", "")
                            choices_dict = item.get("choices", {})
                            c_texts = choices_dict.get("text", [])
                            c_labels = choices_dict.get("label", [])
                            ans_key = str(item.get("answerKey", "")).strip().upper()
                            num_to_let = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}
                            ans_key = num_to_let.get(ans_key, ans_key)
                            if q and len(c_texts) >= 3 and len(c_texts) == len(c_labels):
                                opt_map = {}
                                for lbl, txt in zip(c_labels, c_texts):
                                    mapped_lbl = num_to_let.get(str(lbl).strip().upper(), str(lbl).strip().upper())
                                    if mapped_lbl in ("A", "B", "C", "D", "E"):
                                        opt_map[mapped_lbl] = str(txt).strip()
                                if ans_key in opt_map and len(opt_map) >= 3:
                                    for p_mc, a_mc, m_mc in format_dual_mc_pair(q, opt_map, ans_key):
                                        yield {"user": p_mc, "assistant": a_mc, "mode": m_mc}
                                        if ds_name != "tau/commonsense_qa":
                                            yield {"user": p_mc, "assistant": a_mc, "mode": m_mc}
                                            if m_mc == "mmlu_raw":
                                                mmlu_cnt += 2
                                            else:
                                                arc_cnt += 2
                                        else:
                                            if m_mc == "mmlu_raw":
                                                mmlu_cnt += 1
                                            else:
                                                arc_cnt += 1
                                    c_sub += 1
                        print(f"  [Pillar 1C] Loaded {c_sub:,} `{ds_name}` ({cfg_name}) train questions.")
                    except Exception as e:
                        print(f"  [Warning] Could not stream `{ds_name}` ({cfg_name}): {e}")

                # 1D. AllenAI Winogrande (winogrande_xl train split — 40,398 coreference cloze completions!)
                try:
                    wino_ds = load_dataset("allenai/winogrande", "winogrande_xl", split="train", streaming=True)
                    c_wino = 0
                    for item in wino_ds:
                        sent = item.get("sentence", "").strip()
                        opt1 = item.get("option1", "").strip()
                        opt2 = item.get("option2", "").strip()
                        ans = str(item.get("answer", "")).strip()
                        if "_" in sent and opt1 and opt2 and ans in ("1", "2"):
                            chosen_opt = opt1 if ans == "1" else opt2
                            chosen_let = "A" if ans == "1" else "B"
                            prefix, suffix = sent.split("_", 1)
                            prefix_clean = prefix.rstrip()
                            target_continuation = f" {chosen_opt}{suffix}"
                            if len(prefix_clean) >= 6:
                                # Exact lm_eval Winogrande cloze continuation format:
                                # lm_eval scores log P(chosen_opt + suffix | prefix)
                                yield {"user": prefix_clean, "assistant": target_continuation, "mode": "raw_completion"}
                                comp_cnt += 1
                                # Also emit A/B multiple-choice format on 50% of examples
                                if c_wino % 2 == 0:
                                    mc_p = f"Fill in the blank:\n{sent}\nA. {opt1}\nB. {opt2}\nAnswer:"
                                    yield {"user": mc_p, "assistant": f" {chosen_let}", "mode": "mmlu_raw"}
                                    mmlu_cnt += 1
                                c_wino += 1
                    print(f"  [Pillar 1D] Loaded {c_wino:,} `allenai/winogrande` (winogrande_xl) train cloze completions.")
                except Exception as e:
                    print(f"  [Warning] Could not stream `allenai/winogrande`: {e}")

                # 1E. PIQA Physical Commonsense train split (16,113 physical goal-solution pairs)
                try:
                    try:
                        piqa_ds = load_dataset("ybisk/piqa", split="train", streaming=True, trust_remote_code=True)
                    except Exception:
                        piqa_ds = load_dataset("piqa", split="train", streaming=True)
                    c_piqa = 0
                    for item in piqa_ds:
                        goal = item.get("goal", "").strip()
                        sol1 = item.get("sol1", "").strip()
                        sol2 = item.get("sol2", "").strip()
                        lbl = item.get("label", -1)
                        if goal and sol1 and sol2 and int(lbl) in (0, 1):
                            chosen_sol = sol1 if int(lbl) == 0 else sol2
                            chosen_let = "A" if int(lbl) == 0 else "B"
                            # 1. Exact lm_eval PIQA format (`Goal` -> ` Solution`)
                            yield {"user": f"Question: {goal}\nAnswer:", "assistant": f" {chosen_sol}", "mode": "arc_raw"}
                            arc_cnt += 1
                            # 2. A/B MC format
                            mc_p = f"{goal}\nA. {sol1}\nB. {sol2}\nAnswer:"
                            yield {"user": mc_p, "assistant": f" {chosen_let}", "mode": "mmlu_raw"}
                            mmlu_cnt += 1
                            c_piqa += 1
                    print(f"  [Pillar 1E] Loaded {c_piqa:,} `piqa` physical commonsense train questions.")
                except Exception as e:
                    print(f"  [Warning] Could not stream `piqa`: {e}")

                # -------------------------------------------------------------------------
                # PILLARS 2, 3, 4: Broad SFT + Reasoning (SmolTalk, OpenHermes-2.5, SlimOrca)
                # -------------------------------------------------------------------------
                # 2. HuggingFaceTB/smoltalk
                smol = load_dataset("HuggingFaceTB/smoltalk", "all", split="train", streaming=True)
                c_smol = 0
                for item in smol:
                    if c_smol >= n_smol:
                        break
                    msgs = item.get("messages", [])
                    if len(msgs) >= 2 and msgs[0].get("role") == "user" and msgs[1].get("role") == "assistant":
                        u, a = msgs[0].get("content", ""), msgs[1].get("content", "")
                        if u and a:
                            yield {"user": u, "assistant": a, "mode": "auto"}
                            c_smol += 1
                            for p_mc, a_mc, m_mc in extract_mc_mmlu_and_arc_pairs(u, a):
                                yield {"user": p_mc, "assistant": a_mc, "mode": m_mc}
                                if m_mc == "mmlu_raw":
                                    mmlu_cnt += 1
                                else:
                                    arc_cnt += 1

                # 3. teknium/OpenHermes-2.5
                hermes = load_dataset("teknium/OpenHermes-2.5", split="train", streaming=True)
                c_hermes = 0
                for item in hermes:
                    if c_hermes >= n_hermes:
                        break
                    convs = item.get("conversations", [])
                    u_msg, a_msg = None, None
                    for turn in convs:
                        if turn.get("from") == "human" and u_msg is None:
                            u_msg = turn.get("value", "")
                        elif turn.get("from") == "gpt" and u_msg is not None and a_msg is None:
                            a_msg = turn.get("value", "")
                            break
                    if u_msg and a_msg:
                        yield {"user": u_msg, "assistant": a_msg, "mode": "auto"}
                        c_hermes += 1
                        for p_mc, a_mc, m_mc in extract_mc_mmlu_and_arc_pairs(u_msg, a_msg):
                            yield {"user": p_mc, "assistant": a_mc, "mode": m_mc}
                            yield {"user": p_mc, "assistant": a_mc, "mode": m_mc}
                            if m_mc == "mmlu_raw":
                                mmlu_cnt += 2
                            else:
                                arc_cnt += 2

                # 4. Open-Orca/SlimOrca
                orca = load_dataset("Open-Orca/SlimOrca", split="train", streaming=True)
                c_orca = 0
                for item in orca:
                    if c_orca >= n_orca:
                        break
                    convs = item.get("conversations", [])
                    u_msg, a_msg = None, None
                    for turn in convs:
                        if turn.get("from") == "human" and u_msg is None:
                            u_msg = turn.get("value", "")
                        elif turn.get("from") == "gpt" and u_msg is not None and a_msg is None:
                            a_msg = turn.get("value", "")
                            break
                    if u_msg and a_msg:
                        yield {"user": u_msg, "assistant": a_msg, "mode": "auto"}
                        c_orca += 1
                        for p_mc, a_mc, m_mc in extract_mc_mmlu_and_arc_pairs(u_msg, a_msg):
                            yield {"user": p_mc, "assistant": a_mc, "mode": m_mc}
                            yield {"user": p_mc, "assistant": a_mc, "mode": m_mc}
                            if m_mc == "mmlu_raw":
                                mmlu_cnt += 2
                            else:
                                arc_cnt += 2

                print(
                    f"  --> Total Extracted/Built: {mmlu_cnt:,} `mmlu_raw` (A/B/C/D) pairs, "
                    f"{arc_cnt:,} `arc_raw` (Direct QA) pairs & {comp_cnt:,} `raw_completion` (Cloze) pairs!"
                )

            raw_ds = Dataset.from_generator(
                raw_pairs_generator,
                cache_dir=args.cache_dir,
                writer_batch_size=2000,
            )
            raw_ds = raw_ds.shuffle(seed=args.seed)
            gc.collect()

            seq_len = args.seq_len
            lrad_c = args.lrad_chunk_size
            eos_id = tokenizer.eos_token_id
            has_chat_tmpl = hasattr(tokenizer, "apply_chat_template")
            mc_w = float(args.mc_Answer_weight)
            short_w = float(args.short_answer_weight)
            cc_w = float(args.cross_chunk_weight)

            def parallel_tokenize_and_binpack(examples):
                users = examples["user"]
                assistants = examples["assistant"]
                modes = examples["mode"]

                packed_input_ids = []
                packed_labels = []
                packed_valid_mask = []

                cur_ids = []
                cur_lbls = []
                cur_mask = []

                def flush_bin(c_ids, c_lbls, c_mask):
                    if len(c_ids) == 0:
                        return
                    if not any(x != -100 for x in c_lbls[1:]):
                        return
                    rem = seq_len - len(c_ids)
                    if rem > 0:
                        c_ids = c_ids + [eos_id] * rem
                        c_lbls = c_lbls + [-100] * rem
                        c_mask = c_mask + [0.0] * rem
                    packed_input_ids.append(c_ids)
                    packed_labels.append(c_lbls)
                    packed_valid_mask.append(c_mask)

                for idx, (u_txt, a_txt, mode) in enumerate(zip(users, assistants, modes)):
                    if mode in ("mmlu_raw", "arc_raw", "raw_completion"):
                        # Exact zero-shot MMLU, ARC, or Winogrande Cloze completion format
                        p_ids = tokenizer(u_txt, add_special_tokens=False)["input_ids"]
                        f_ids = tokenizer(u_txt + a_txt, add_special_tokens=False)["input_ids"] + [eos_id]
                    elif (idx % 7 == 0) and has_chat_tmpl:
                        # 15% Official Chat Template format (retains interactive chat ability)
                        p_ids = tokenizer.apply_chat_template(
                            [{"role": "user", "content": u_txt}], tokenize=True, add_generation_prompt=True
                        )
                        f_ids = tokenizer.apply_chat_template(
                            [{"role": "user", "content": u_txt}, {"role": "assistant", "content": a_txt}],
                            tokenize=True,
                            add_generation_prompt=False,
                        )
                    else:
                        # 85% Clean Zero-Shot Q&A format (matches `lm_eval` zero-shot formatting!)
                        p_str = f"Question: {u_txt.strip()}\nAnswer:"
                        f_str = f"Question: {u_txt.strip()}\nAnswer: {a_txt.strip()}"
                        p_ids = tokenizer(p_str, add_special_tokens=False)["input_ids"]
                        f_ids = tokenizer(f_str, add_special_tokens=False)["input_ids"] + [eos_id]

                    # Never cut an observation in half: skip single conversations exceeding seq_len
                    if len(f_ids) > seq_len or len(f_ids) < 4:
                        continue

                    p_len = min(len(p_ids), len(f_ids))
                    resp_len = len(f_ids) - p_len
                    if resp_len <= 0:
                        continue

                    lbls = [-100] * p_len + f_ids[p_len:]

                    # Base Focal Answer-Token Importance Weight
                    if mode == "mmlu_raw":
                        ans_w = mc_w
                    elif mode in ("arc_raw", "raw_completion") or resp_len <= 32:
                        ans_w = short_w
                    else:
                        ans_w = 1.0

                    # Determine bin start offset so we know which 64-token CS-LRAD chunks this example spans
                    if len(cur_ids) + len(f_ids) > seq_len:
                        flush_bin(cur_ids, cur_lbls, cur_mask)
                        cur_ids, cur_lbls, cur_mask = [], [], []

                    bin_start_pos = len(cur_ids)
                    prompt_start_chunk = bin_start_pos // lrad_c

                    # Apply Cross-Chunk Boundary Multiplier (`cc_w`) whenever a token sits in a later
                    # 64-token chunk than the start of its question (forcing retrieval via S_historical!)
                    token_weights = []
                    for local_i in range(len(f_ids)):
                        abs_pos = bin_start_pos + local_i
                        base_w = 1.0 if local_i < p_len else ans_w
                        if (abs_pos // lrad_c) > prompt_start_chunk:
                            base_w = base_w * cc_w
                        token_weights.append(base_w)

                    cur_ids.extend(f_ids)
                    cur_lbls.extend(lbls)
                    cur_mask.extend(token_weights)

                flush_bin(cur_ids, cur_lbls, cur_mask)
                return {
                    "input_ids": packed_input_ids,
                    "labels": packed_labels,
                    "valid_mask": packed_valid_mask,
                }

            print(f"  - Running parallel whole-conversation bin-packing (seq_len={seq_len}, num_proc={args.num_proc})...")
            packed_ds = raw_ds.map(
                parallel_tokenize_and_binpack,
                batched=True,
                batch_size=1000,
                writer_batch_size=1000,
                num_proc=args.num_proc,
                remove_columns=raw_ds.column_names,
                keep_in_memory=False,
                desc=f"Bin-Packing Expanded 8K 5-Pillar SFT Conversations ({seq_len})",
            )
            gc.collect()

            # Pillar 5: Add ~22% Pretraining Replay from FineWeb-Edu with Cross-Chunk Boundary Up-Weighting!
            if os.path.exists(args.pretrain_data_dir):
                fw_train = load_from_disk(args.pretrain_data_dir)["train"]
                num_replay = min(len(fw_train), max(500, int(len(packed_ds) * args.replay_ratio)))
                pct_of_total = 100.0 * num_replay / max(1, len(packed_ds) + num_replay)
                print(
                    f"  - Mixing {num_replay:,} FineWeb-Edu replay blocks (~{pct_of_total:.1f}% of final pool) "
                    f"with 2.0x Cross-Chunk Boundary weights..."
                )

                # Precompute the 2048-token mask where the first 16 tokens after every 64-token boundary get 2.0x weight
                fw_boundary_mask = [
                    2.0 if (pos >= lrad_c and (pos % lrad_c) < 16) else 1.0
                    for pos in range(seq_len)
                ]

                def convert_fw_to_sft_schema(examples):
                    ids_list = examples["input_ids"]
                    return {
                        "input_ids": ids_list,
                        "labels": ids_list,
                        "valid_mask": [fw_boundary_mask[:len(x)] for x in ids_list],
                    }

                replay_ds = fw_train.select(range(num_replay)).map(
                    convert_fw_to_sft_schema,
                    batched=True,
                    batch_size=1000,
                    num_proc=min(8, args.num_proc),
                    keep_in_memory=False,
                )
                packed_ds = concatenate_datasets([packed_ds, replay_ds])

            packed_ds = packed_ds.shuffle(seed=args.seed)
            total_blocks = len(packed_ds)
            val_blocks = min(args.val_blocks, max(50, total_blocks // 40))
            train_blocks = total_blocks - val_blocks

            ds_dict = DatasetDict({
                "train": packed_ds.select(range(val_blocks, val_blocks + train_blocks)),
                "validation": packed_ds.select(range(val_blocks)),
            })

            os.makedirs(args.sft_data_dir, exist_ok=True)
            ds_dict.save_to_disk(args.sft_data_dir, num_proc=min(4, args.num_proc))
            print(
                f"[Dataset Builder] Saved {train_blocks:,} train blocks ({train_blocks // blocks_per_step:,} unique steps) "
                f"& {val_blocks:,} val blocks to `{args.sft_data_dir}`!"
            )
    else:
        if accelerator.is_main_process:
            print(f"[Dataset Loader] Found existing Expanded 5-Pillar Focal SFT dataset at `{args.sft_data_dir}`. Loading from disk!")

    accelerator.wait_for_everyone()
    return load_from_disk(args.sft_data_dir)


def collate_sft(batch):
    return {
        "input_ids": torch.tensor([x["input_ids"] for x in batch], dtype=torch.long),
        "labels": torch.tensor([x["labels"] for x in batch], dtype=torch.long),
        "valid_mask": torch.tensor([x["valid_mask"] for x in batch], dtype=torch.float32),
    }


def chunked_sft_kl_and_masked_ce_loss(
    h_student, h_teacher, labels, valid_mask, student_lm_head, teacher_lm_head,
    temperature=1.0, kl_weight=0.5, ce_weight=0.5, chunk_size=128
):
    """
    Focal Answer-Weighted + Cross-Chunk Logit Distillation:
      - `valid_mask` is 0.0 on right-padding, 1.0 on intra-chunk prompts/essays, and >1.0 on concise MC/QA
        and cross-chunk tokens.
      - Both KL and CE training losses weight tokens by `valid_mask`.
      - Telemetry (`ce_s_raw`, `ce_t_raw`) reports standard unweighted token-mean CE for direct comparison.
    """
    s_dtype = student_lm_head.weight.dtype
    t_dtype = teacher_lm_head.weight.dtype

    h_s_shift = h_student[:, :-1, :].to(s_dtype)
    h_t_shift = h_teacher[:, :-1, :].detach().to(t_dtype)
    shift_labels = labels[:, 1:]
    shift_valid_mask = valid_mask[:, 1:]

    B, L_minus_1, _ = h_s_shift.shape

    def _chunk_loss_fn(h_s_c, h_t_c, labels_c, vmask_c):
        logits_s = student_lm_head(h_s_c.to(s_dtype)).float()
        with torch.no_grad():
            logits_t = teacher_lm_head(h_t_c.to(t_dtype)).float()
            v_min = min(logits_s.size(-1), logits_t.size(-1))
            logits_t_v = logits_t[..., :v_min]

            probs_t = F.softmax(logits_t_v / temperature, dim=-1)
            if (labels_c != -100).any():
                ce_t_raw = F.cross_entropy(logits_t.reshape(-1, logits_t.size(-1)), labels_c.reshape(-1), ignore_index=-100)
            else:
                ce_t_raw = torch.tensor(0.0, device=logits_s.device)

        log_probs_s = F.log_softmax(logits_s[..., :v_min] / temperature, dim=-1)
        token_kl = F.kl_div(log_probs_s, probs_t, reduction="none").sum(dim=-1) * (temperature ** 2)
        valid_kl_denom = vmask_c.sum().clamp(min=1.0)
        kl = (token_kl * vmask_c).sum() / valid_kl_denom

        if (labels_c != -100).any():
            # Unweighted CE for telemetry logging
            with torch.no_grad():
                ce_s_raw = F.cross_entropy(logits_s.reshape(-1, logits_s.size(-1)), labels_c.reshape(-1), ignore_index=-100)
            # Focal-weighted CE for gradient backpropagation
            per_tok_ce = F.cross_entropy(
                logits_s.reshape(-1, logits_s.size(-1)), labels_c.reshape(-1), ignore_index=-100, reduction="none"
            ).view_as(labels_c)
            ce_weights = vmask_c * (labels_c != -100).float()
            ce_s_weighted = (per_tok_ce * ce_weights).sum() / ce_weights.sum().clamp(min=1.0)
        else:
            ce_s_raw = torch.tensor(0.0, device=logits_s.device)
            ce_s_weighted = logits_s.sum() * 0.0

        return kl, ce_s_weighted, ce_s_raw, ce_t_raw

    total_kl, total_ce_weighted = 0.0, 0.0
    total_ce_s_raw, total_ce_t_raw = 0.0, 0.0
    valid_kl_weight_sum = 0.0
    valid_ce_weight_sum = 0.0
    raw_ce_token_sum = 0.0

    for start in range(0, L_minus_1, chunk_size):
        end = min(start + chunk_size, L_minus_1)
        h_s_c = h_s_shift[:, start:end, :].contiguous()
        h_t_c = h_t_shift[:, start:end, :].contiguous()
        labels_c = shift_labels[:, start:end].contiguous()
        vmask_c = shift_valid_mask[:, start:end].contiguous()

        kl_w_in_chunk = vmask_c.sum().item()
        ce_mask_c = (labels_c != -100).float()
        ce_w_in_chunk = (vmask_c * ce_mask_c).sum().item()
        raw_ce_tokens = ce_mask_c.sum().item()

        if h_student.requires_grad:
            kl_c, ce_w_c, ce_s_r, ce_t_r = checkpoint.checkpoint(
                _chunk_loss_fn, h_s_c, h_t_c, labels_c, vmask_c, use_reentrant=False
            )
        else:
            kl_c, ce_w_c, ce_s_r, ce_t_r = _chunk_loss_fn(h_s_c, h_t_c, labels_c, vmask_c)

        if kl_w_in_chunk > 0:
            total_kl = total_kl + kl_c * kl_w_in_chunk
            valid_kl_weight_sum += kl_w_in_chunk

        if ce_w_in_chunk > 0:
            total_ce_weighted = total_ce_weighted + ce_w_c * ce_w_in_chunk
            valid_ce_weight_sum += ce_w_in_chunk

        if raw_ce_tokens > 0:
            total_ce_s_raw = total_ce_s_raw + ce_s_r * raw_ce_tokens
            total_ce_t_raw = total_ce_t_raw + ce_t_r * raw_ce_tokens
            raw_ce_token_sum += raw_ce_tokens

    if valid_kl_weight_sum > 0:
        total_kl = total_kl / valid_kl_weight_sum
    if valid_ce_weight_sum > 0:
        total_ce_weighted = total_ce_weighted / valid_ce_weight_sum
    if raw_ce_token_sum > 0:
        total_ce_s_raw = total_ce_s_raw / raw_ce_token_sum
        total_ce_t_raw = total_ce_t_raw / raw_ce_token_sum

    loss = kl_weight * total_kl + ce_weight * total_ce_weighted
    return loss, total_kl.detach(), total_ce_s_raw.detach(), total_ce_t_raw.detach()


class Stage4Wrapper(nn.Module):
    def __init__(self, student_model):
        super().__init__()
        self.student = student_model

    def forward(self, input_ids):
        out = self.student.model(
            input_ids=input_ids,
            attention_mask=None,
            output_hidden_states=False,
            use_cache=False,
        )
        return out.last_hidden_state


@torch.no_grad()
def evaluate_stage4a(accelerator, wrapper, teacher, t_backbone, val_loader, args):
    wrapper.eval()
    unwrapped_student = accelerator.unwrap_model(wrapper).student

    local_sums = torch.zeros(4, dtype=torch.float64, device=accelerator.device)
    local_count = torch.zeros(1, dtype=torch.float64, device=accelerator.device)

    for batch_idx, batch in enumerate(val_loader):
        if args.eval_batches > 0 and batch_idx >= args.eval_batches:
            break
        input_ids = batch["input_ids"].to(accelerator.device, non_blocking=True)
        labels = batch["labels"].to(accelerator.device, non_blocking=True)
        valid_mask = batch["valid_mask"].to(accelerator.device, non_blocking=True)

        t_out = t_backbone(input_ids=input_ids, attention_mask=None, output_hidden_states=False, use_cache=False)
        s_last_h = wrapper(input_ids)

        loss, kl_val, ce_s_val, ce_t_val = chunked_sft_kl_and_masked_ce_loss(
            s_last_h, t_out.last_hidden_state, labels, valid_mask,
            unwrapped_student.lm_head, teacher.lm_head,
            temperature=args.temperature, kl_weight=args.kl_weight, ce_weight=args.ce_weight, chunk_size=args.logit_chunk_size
        )

        local_sums[0] += loss.detach().to(torch.float64)
        local_sums[1] += kl_val.to(torch.float64)
        local_sums[2] += ce_s_val.to(torch.float64)
        local_sums[3] += ce_t_val.to(torch.float64)
        local_count[0] += 1.0

    global_sums = accelerator.reduce(local_sums, reduction="sum")
    global_count = accelerator.reduce(local_count, reduction="sum").clamp(min=1.0)
    means = (global_sums / global_count).tolist()

    wrapper.train()
    return means[0], means[1], means[2], means[3]


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    use_compile = not args.no_compile
    use_gc = not args.no_grad_checkpointing
    accelerator = Accelerator(mixed_precision="bf16", gradient_accumulation_steps=args.grad_accum_steps)

    ckpt_to_load = args.stage3_ckpt_dir
    if not os.path.exists(ckpt_to_load) and os.path.exists(os.path.dirname(ckpt_to_load)):
        ckpt_to_load = os.path.dirname(ckpt_to_load)

    tokenizer = AutoTokenizer.from_pretrained(ckpt_to_load, trust_remote_code=True)

    # 1. Check disk for Expanded 8K 5-Pillar Focal-Weighted SFT dataset; build once in parallel if missing/small
    ds_dict = prepare_or_load_sft_dataset(args, tokenizer, accelerator)
    train_ds = ds_dict["train"]
    val_ds = ds_dict["validation"]
    seq_len = len(train_ds[0]["input_ids"])

    # 2. Reconstruct 0.67B Student from 0.8B structure, then load ANY Qwen3.5 Teacher (0.8B, 2B, or 4B!)
    _, base_08b_backbone, _ = load_qwen3_5_text_teacher(args.student_base_id, dtype=torch.bfloat16, device="cpu")
    student = load_hybrid_student_checkpoint(
        ckpt_to_load, teacher_backbone=base_08b_backbone, teacher_id=args.student_base_id, dtype=torch.bfloat16
    )
    del base_08b_backbone
    gc.collect()

    if getattr(student.config, "tie_word_embeddings", True):
        student.lm_head.weight = student.model.token_embedding.weight

    if use_gc:
        student.gradient_checkpointing_enable()

    if accelerator.is_main_process:
        print(f"Loading Up-Distillation Teacher: {args.teacher_id} ...")
    teacher, t_backbone, _ = load_qwen3_5_text_teacher(args.teacher_id, dtype=torch.bfloat16, device="cpu")
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    if use_compile:
        for layer in student.model.layers:
            if isinstance(layer, LRADDecoderBlock):
                layer.ln1.forward = torch.compile(layer.ln1.forward, mode=args.compile_mode, dynamic=False)
                layer.attn.chunk_forward = torch.compile(layer.attn.chunk_forward, mode=args.compile_mode, dynamic=False)
                layer.ln2.forward = torch.compile(layer.ln2.forward, mode=args.compile_mode, dynamic=False)

    t_backbone.to(accelerator.device)
    teacher.lm_head.to(accelerator.device)
    student.to(accelerator.device)

    lrad_ids = set()
    lrad_params, backbone_params = [], []
    for layer in student.model.layers:
        if isinstance(layer, LRADDecoderBlock):
            for p in list(layer.ln1.parameters()) + list(layer.attn.parameters()):
                p.requires_grad = True
                lrad_ids.add(id(p))
                lrad_params.append(p)
    seen = set(lrad_ids)
    for _, p in student.named_parameters():
        p.requires_grad = True
        if id(p) not in seen:
            seen.add(id(p))
            backbone_params.append(p)

    optimizer = torch.optim.AdamW(
        [
            {"params": lrad_params, "lr": args.lrad_lr, "weight_decay": args.weight_decay},
            {"params": backbone_params, "lr": args.backbone_lr, "weight_decay": args.weight_decay},
        ],
        betas=(0.9, 0.95),
        fused=True,
    )
    all_params = lrad_params + backbone_params

    train_loader = DataLoader(
        train_ds, batch_size=args.micro_batch_size, shuffle=True,
        collate_fn=collate_sft, num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.micro_batch_size, shuffle=False,
        collate_fn=collate_sft, num_workers=4, pin_memory=True, drop_last=True
    )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=int(args.warmup_ratio * args.max_steps), num_training_steps=args.max_steps
    )

    wrapper = Stage4Wrapper(student)
    wrapper, optimizer, train_loader, val_loader = accelerator.prepare(wrapper, optimizer, train_loader, val_loader)
    wrapper.train()

    blocks_per_step = args.micro_batch_size * accelerator.num_processes * args.grad_accum_steps
    tokens_per_step = blocks_per_step * seq_len
    epochs_covered = (args.max_steps * blocks_per_step) / max(len(train_ds), 1)

    if accelerator.is_main_process:
        val_scope = "ENTIRE validation set" if args.eval_batches <= 0 else f"{args.eval_batches * accelerator.num_processes} batches"
        print("=" * 85)
        print("STAGE 4A: EXPANDED 8K 5-PILLAR + COMMONSENSE FOCAL UP-DISTILLATION")
        print(f"  - Up-Distill Teacher   : {args.teacher_id} (Pure Logit KL, Zero Dimension Lock-in!)")
        print(f"  - Loaded Stage 3 Ckpt  : {ckpt_to_load}")
        print(f"  - Disk Dataset Path    : {args.sft_data_dir} (seq_len={seq_len})")
        print(f"  - Focal Answer Weights : MMLU={args.mc_Answer_weight}x | Concise QA/Cloze={args.short_answer_weight}x | Cross-Chunk={args.cross_chunk_weight}x")
        print(f"  - Train / Val Blocks   : {len(train_ds):,} train ({len(train_ds)*seq_len/1e6:.1f}M tok) | {len(val_ds):,} val")
        print(f"  - Epochs Covered       : {epochs_covered:.2f} epochs ({args.max_steps:,} steps / {len(train_ds)//blocks_per_step:,} max unique steps)")
        print(f"  - Gradient Checkpoint  : {'ENABLED' if use_gc else 'DISABLED'}")
        print(f"  - torch.compile CS-LRAD: {'ENABLED (' + args.compile_mode + ')' if use_compile else 'DISABLED'}")
        print(f"  - Loss Formulation     : {args.kl_weight}*Focal_KL(T={args.temperature}) + {args.ce_weight}*Focal_Resp_CE")
        print(f"  - Global Batch Size    : {tokens_per_step:,} tokens/step | Total Token Budget: {tokens_per_step * args.max_steps / 1e6:.1f}M")
        print(f"  - Train Log Averaging  : Global mean over {args.log_steps} steps ({args.log_steps * tokens_per_step:,} tokens)")
        print(f"  - Validation Scope     : {val_scope} ({len(val_ds) * seq_len / 1e6:.2f}M total val tokens)")
        print("=" * 85)

    step = 0
    t0 = time.time()
    best_val_loss = float("inf")
    running_sums = torch.zeros(4, dtype=torch.float64, device=accelerator.device)
    running_count = torch.zeros(1, dtype=torch.float64, device=accelerator.device)

    while step < args.max_steps:
        for batch in train_loader:
            input_ids = batch["input_ids"].to(accelerator.device, non_blocking=True)
            labels = batch["labels"].to(accelerator.device, non_blocking=True)
            valid_mask = batch["valid_mask"].to(accelerator.device, non_blocking=True)

            with accelerator.accumulate(wrapper):
                with torch.no_grad():
                    t_out = t_backbone(
                        input_ids=input_ids, attention_mask=None, output_hidden_states=False, use_cache=False
                    )

                s_last_h = wrapper(input_ids)
                unwrapped_student = accelerator.unwrap_model(wrapper).student

                loss, kl_val, ce_s_val, ce_t_val = chunked_sft_kl_and_masked_ce_loss(
                    s_last_h, t_out.last_hidden_state, labels, valid_mask,
                    unwrapped_student.lm_head, teacher.lm_head,
                    temperature=args.temperature, kl_weight=args.kl_weight, ce_weight=args.ce_weight, chunk_size=args.logit_chunk_size
                )

                running_sums[0] += loss.detach().to(torch.float64)
                running_sums[1] += kl_val.to(torch.float64)
                running_sums[2] += ce_s_val.to(torch.float64)
                running_sums[3] += ce_t_val.to(torch.float64)
                running_count[0] += 1.0

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(all_params, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                scheduler.step()
                step += 1
                if step == 1:
                    t0 = time.time()

                # 1. Logging
                if step == 1 or step % args.log_steps == 0:
                    g_sums = accelerator.reduce(running_sums, reduction="sum")
                    g_cnt = accelerator.reduce(running_count, reduction="sum").clamp(min=1.0)
                    avg_l, avg_kl, avg_ces, avg_cet = (g_sums / g_cnt).tolist()
                    running_sums.zero_()
                    running_count.zero_()

                    if accelerator.is_main_process:
                        elapsed = time.time() - t0
                        toks_sec = ((args.log_steps if step > 1 else 1) * tokens_per_step) / max(elapsed, 1e-5)
                        t0 = time.time()
                        lrs = scheduler.get_last_lr()
                        s_ppl = min(torch.tensor(avg_ces).exp().item(), 9999.0)
                        t_ppl = min(torch.tensor(avg_cet).exp().item(), 9999.0)
                        tok_str = "Compiling DONE" if step == 1 else f"{toks_sec:,.0f} tok/s"
                        print(
                            f"[Stage 4A] Step {step:04d}/{args.max_steps} | Total: {avg_l:.4f} | Focal_KL: {avg_kl:.4f} | "
                            f"S_RespCE: {avg_ces:.4f} vs T_RespCE: {avg_cet:.4f} (PPL: {s_ppl:.2f} vs T_PPL: {t_ppl:.2f}) | "
                            f"LR: {lrs[0]:.2e}/{lrs[1]:.2e} | {tok_str}"
                        )

                # 2. Validation Evaluation
                if step % args.eval_steps == 0 or step >= args.max_steps:
                    accelerator.wait_for_everyone()
                    val_loss, val_kl, val_ces, val_cet = evaluate_stage4a(
                        accelerator, wrapper, teacher, t_backbone, val_loader, args
                    )
                    if accelerator.is_main_process:
                        val_s_ppl = min(torch.tensor(val_ces).exp().item(), 9999.0)
                        val_t_ppl = min(torch.tensor(val_cet).exp().item(), 9999.0)
                        print(
                            f"  >>> [VAL @ Step {step:04d}] Total: {val_loss:.4f} | Focal_KL: {val_kl:.4f} | "
                            f"Student Resp_CE: {val_ces:.4f} (Val PPL: {val_s_ppl:.2f}) | "
                            f"Teacher ({args.teacher_id}) Resp_CE: {val_cet:.4f} (T_PPL: {val_t_ppl:.2f})"
                        )
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            best_dir = os.path.join(args.output_dir, "best")
                            os.makedirs(best_dir, exist_ok=True)
                            unwrapped = accelerator.unwrap_model(wrapper).student
                            unwrapped.config.save_pretrained(best_dir)
                            tokenizer.save_pretrained(best_dir)
                            torch.save(unwrapped.state_dict(), os.path.join(best_dir, "pytorch_model.bin"))
                            print(f"  >>> New Best Stage 4A Val Loss ({best_val_loss:.4f})! Saved to {best_dir}")
                    accelerator.wait_for_everyone()
                    t0 = time.time()

                # 3. Periodic Checkpointing
                if step % args.save_steps == 0 or step >= args.max_steps:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        os.makedirs(args.output_dir, exist_ok=True)
                        unwrapped = accelerator.unwrap_model(wrapper).student
                        unwrapped.config.save_pretrained(args.output_dir)
                        tokenizer.save_pretrained(args.output_dir)
                        torch.save(unwrapped.state_dict(), os.path.join(args.output_dir, "pytorch_model.bin"))
                        print(f"  --> Saved Latest Stage 4A checkpoint at step {step} to {args.output_dir}")
                    accelerator.wait_for_everyone()
                    t0 = time.time()

                if step >= args.max_steps:
                    break

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()