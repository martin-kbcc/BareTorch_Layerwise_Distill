# /home/martinkb/Desktop/BareTorch_Layerwise_Distill/train_stage4b_dpo_align_06.py
import os
import copy
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
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict

from baretorch.cs_lrad import LRADDecoderBlock
from init_student_00 import load_qwen3_5_text_teacher, load_hybrid_student_checkpoint

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 4B: 6-Pillar Contrastive DPO + DistiLLM-2 Top-K Skew-KD with Larger Teacher (2B/4B)"
    )
    # Models & Paths
    parser.add_argument("--teacher_id", type=str, default="Qwen/Qwen3.5-2B", help="Teacher HF ID for DPO & Top-K Skew-KD (Qwen/Qwen3.5-0.8B, Qwen/Qwen3.5-2B, or Qwen/Qwen3.5-4B)")
    parser.add_argument("--student_base_id", type=str, default="Qwen/Qwen3.5-0.8B", help="Base 0.8B model ID used to initialize Student structure")
    parser.add_argument("--stage4a_ckpt_dir", type=str, default="./checkpoints/stage4a_5pillar_8k_0.8B/best", help="Path to Stage 4A best student checkpoint")
    parser.add_argument("--dpo_data_dir", type=str, default="./data/stage4b_6pillar_dpo_1024", help="Disk path for pre-tokenized 6-Pillar DPO dataset")
    parser.add_argument("--cache_dir", type=str, default="./data/hf_cache_stage4b_6pillar_dpo", help="Temporary Arrow disk cache directory")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/stage4b_dpo_2B", help="Directory to save Stage 4B DPO checkpoints")

    # 6-Pillar Dataset Construction Parameters
    parser.add_argument("--mmlu_aux_dpo_samples", type=int, default=35000, help="Hard-negative MC distractor pairs from cais/mmlu auxiliary_train")
    parser.add_argument("--hellaswag_dpo_samples", type=int, default=25000, help="Narrative completion contrastive pairs from Rowan/hellaswag train")
    parser.add_argument("--winogrande_dpo_samples", type=int, default=20000, help="Cloze coreference contrastive pairs from allenai/winogrande train")
    parser.add_argument("--piqa_dpo_samples", type=int, default=16113, help="Physical commonsense contrastive pairs from ybisk/piqa Parquet train")
    parser.add_argument("--ultrafeedback_samples", type=int, default=25000, help="General reasoning preference pairs from UltraFeedback Binarized")
    parser.add_argument("--mc_answer_weight", type=float, default=6.0, help="Focal weight on single-token MMLU (A/B/C/D) chosen tokens")
    parser.add_argument("--short_answer_weight", type=float, default=3.0, help="Focal weight on concise ARC/QA/Cloze/HellaSwag chosen tokens (<= 48 tokens)")
    parser.add_argument("--cross_chunk_weight", type=float, default=1.5, help="Extra multiplier when tokens cross a 64-token CS-LRAD chunk boundary")
    parser.add_argument("--lrad_chunk_size", type=int, default=64, help="CS-LRAD chunk boundary interval (64)")
    parser.add_argument("--max_len", type=int, default=1024, help="Max sequence length per DPO pair (multiple of 64; pairs exceeding this are skipped)")
    parser.add_argument("--val_samples", type=int, default=400, help="Number of validation preference pairs reserved")
    parser.add_argument("--num_proc", type=int, default=min(16, max(1, os.cpu_count() // 2)), help="Parallel CPU workers for dataset building")

    # Training Hyperparameters (Decoupled LRs matching Stage 4A)
    parser.add_argument("--lrad_lr", type=float, default=2.0e-5, help="Peak DPO LR for CS-LRAD recurrent layers")
    parser.add_argument("--backbone_lr", type=float, default=5.0e-6, help="Peak DPO LR for Backbone MLPs & Full-Attn")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--micro_batch_size", type=int, default=2, help="Preference pairs per GPU per micro-step")
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=1500, help="1500 contrastive DPO + Top-K Skew-KD steps (<1.0 epoch)")
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # DistiLLM-2 Top-K Skew-KD + Teacher-Guided DPO + Focal NLL Weights
    parser.add_argument("--beta", type=float, default=0.1, help="DPO inverse temperature beta")
    parser.add_argument("--teacher_ref_alpha", type=float, default=0.35, help="Blend ratio of Teacher log-probs into DPO reference margin (0.35 = 65% Frozen_Stage4A + 35% Teacher)")
    parser.add_argument("--topk_logits", type=int, default=64, help="Top-K Teacher logits for Response-Only Skew-KL & Skew-RKL (eliminates 248K tail flattening)")
    parser.add_argument("--skew_alpha", type=float, default=0.1, help="DistiLLM-2 skew interpolation coefficient alpha")
    parser.add_argument("--dpo_weight", type=float, default=0.45, help="Weight on Teacher-Guided Contrastive DPO loss")
    parser.add_argument("--skew_kd_weight", type=float, default=0.30, help="Weight on Response-Only Top-K Skew-KL (Chosen) + Skew-Reverse-KL (Chosen & Rejected)")
    parser.add_argument("--nll_weight", type=float, default=0.25, help="Weight on Focal Chosen NLL anchor (RPO stabilizer)")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for Top-K Teacher Skew-KD")
    parser.add_argument("--logit_chunk_size", type=int, default=128, help="Chunk size along sequence length to avoid materializing full 248K vocab logits")

    # Logging, Evaluation & Saving
    parser.add_argument("--log_steps", type=int, default=25)
    parser.add_argument("--eval_steps", type=int, default=250)
    parser.add_argument("--eval_batches", type=int, default=-1, help="-1 = evaluate entire validation split")
    parser.add_argument("--save_steps", type=int, default=250)

    # Hardware & Compilation
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--compile_mode", type=str, default="default", choices=["default", "reduce-overhead", "max-autotune"])
    parser.add_argument("--no_compile", action="store_true", help="Disable torch.compile on CS-LRAD layers")
    parser.add_argument("--no_grad_checkpointing", action="store_true", help="Disable gradient checkpointing on Student")
    parser.add_argument("--seed", type=int, default=202)
    return parser.parse_args()


def build_contrastive_mc_pairs(stem: str, opt_map: dict, chosen_letter: str, rng: random.Random):
    """
    Given a multiple-choice question `stem`, `opt_map` ({'A': ..., 'B': ...}), and `chosen_letter`,
    picks a distractor letter (`rejected_letter != chosen_letter`) and returns contrastive pairs:
      1. `mmlu_raw`: Prompt = `Question\nA. ...\nB. ...\nAnswer:`, Chosen = ` B`, Rejected = ` D`
      2. `arc_raw` : Prompt = `Question: ...\nAnswer:`, Chosen = ` <Correct Text>`, Rejected = ` <Distractor Text>`
    """
    stem = stem.strip()
    if len(stem) < 6 or chosen_letter not in opt_map:
        return []

    distractor_letters = [k for k in ("A", "B", "C", "D") if k in opt_map and k != chosen_letter]
    if not distractor_letters:
        return []

    rej_letter = rng.choice(distractor_letters)
    results = []

    if chosen_letter in ("A", "B", "C", "D"):
        opts_formatted = "\n".join([f"{k}. {opt_map[k]}" for k in ("A", "B", "C", "D") if k in opt_map])
        mmlu_prompt = f"{stem}\n{opts_formatted}\nAnswer:"
        results.append((mmlu_prompt, f" {chosen_letter}", f" {rej_letter}", "mmlu_raw"))

    chosen_text = opt_map[chosen_letter].strip()
    rej_text = opt_map[rej_letter].strip()
    if (
        len(chosen_text) >= 2
        and len(rej_text) >= 2
        and chosen_text.lower() != rej_text.lower()
        and not chosen_text.lower().startswith(("all of the above", "none of the above", "both a and b"))
        and not rej_text.lower().startswith(("all of the above", "none of the above", "both a and b"))
    ):
        arc_prompt = f"Question: {stem}\nAnswer:"
        results.append((arc_prompt, f" {chosen_text}", f" {rej_text}", "arc_raw"))

    return results


def prepare_or_load_dpo_dataset(args, tokenizer, accelerator):
    """
    Checks if `args.dpo_data_dir` exists on disk AND contains enough preference pairs for `< 1.0` epoch.
    If not, Rank 0 builds the 6-Pillar Contrastive DPO dataset in parallel on CPU:
      - Pillar 1 (Academic & STEM MC): MMLU (`auxiliary_train`), SciQ (`train`), ARC (`train`), OBQA (`train`), CSQA (`train`)
      - Pillar 2 (Narrative Activity Completion): Rowan/hellaswag (`train`) exact `acc_norm` continuation pairs!
      - Pillar 3 (Physical Commonsense): ybisk/piqa (`refs/convert/parquet` `train`) goal -> solution pairs!
      - Pillar 4 (Coreference Cloze): allenai/winogrande (`winogrande_xl` `train`) cloze continuation pairs!
      - Pillar 5 (General Reasoning & Chat): HuggingFaceH4/ultrafeedback_binarized (`train_prefs`)
    """
    pairs_per_step = args.micro_batch_size * accelerator.num_processes * args.grad_accum_steps
    min_required_train_pairs = args.max_steps * pairs_per_step

    need_build = not os.path.exists(args.dpo_data_dir)
    if not need_build:
        try:
            existing_ds = load_from_disk(args.dpo_data_dir)
            existing_train_len = len(existing_ds["train"])
            if existing_train_len < min_required_train_pairs:
                if accelerator.is_main_process:
                    print(
                        f"[DPO Dataset Check] Existing dataset at `{args.dpo_data_dir}` has {existing_train_len:,} pairs "
                        f"(< {min_required_train_pairs:,} needed for {args.max_steps} unique steps). Rebuilding!"
                    )
                need_build = True
        except Exception:
            need_build = True

    if need_build:
        if accelerator.is_main_process:
            if os.path.exists(args.dpo_data_dir):
                shutil.rmtree(args.dpo_data_dir)
            os.makedirs(args.cache_dir, exist_ok=True)

            print("=" * 90)
            print(f"[DPO Dataset Builder] Building 6-Pillar Contrastive DPO Pool at `{args.dpo_data_dir}` ({args.num_proc} CPU workers)...")
            print(f"  - Pillar 1 (MC/QA Distractor Rejection): MMLU aux ({args.mmlu_aux_dpo_samples:,}) + SciQ + ARC + OBQA + CSQA")
            print(f"  - Pillar 2 (Narrative Completion DPO)  : Rowan/hellaswag train ({args.hellaswag_dpo_samples:,})")
            print(f"  - Pillar 3 (Physical Commonsense DPO)  : ybisk/piqa Parquet train ({args.piqa_dpo_samples:,})")
            print(f"  - Pillar 4 (Cloze Coreference DPO)     : Winogrande XL ({args.winogrande_dpo_samples:,})")
            print(f"  - Pillar 5 (General Reasoning DPO)     : UltraFeedback Binarized ({args.ultrafeedback_samples:,})")
            print(f"  - Focal Chosen Weights                 : MMLU={args.mc_answer_weight}x | Concise/Cloze={args.short_answer_weight}x | Cross-Chunk={args.cross_chunk_weight}x")
            print("=" * 90)

            def raw_dpo_generator():
                rng = random.Random(args.seed)
                letters = ["A", "B", "C", "D"]
                mmlu_cnt, arc_cnt, cloze_cnt, uf_cnt = 0, 0, 0, 0

                # 1A. Official MMLU auxiliary_train contrastive pairs
                try:
                    mmlu_aux = load_dataset("cais/mmlu", "all", split="auxiliary_train", streaming=True)
                    c_aux = 0
                    for item in mmlu_aux:
                        if c_aux >= args.mmlu_aux_dpo_samples:
                            break
                        row = item.get("train", item) if isinstance(item, dict) else item
                        q = row.get("question", "")
                        choices = row.get("choices", [])
                        ans_idx = row.get("answer", -1)
                        if q and len(choices) == 4 and 0 <= int(ans_idx) < 4:
                            opt_map = {letters[i]: str(choices[i]).strip() for i in range(4)}
                            chosen = letters[int(ans_idx)]
                            for p_txt, c_txt, r_txt, mode in build_contrastive_mc_pairs(q, opt_map, chosen, rng):
                                yield {"prompt": p_txt, "chosen": c_txt, "rejected": r_txt, "mode": mode}
                                if mode == "mmlu_raw":
                                    mmlu_cnt += 1
                                else:
                                    arc_cnt += 1
                            c_aux += 1
                    print(f"  [DPO Pillar 1A] Built contrastive pairs from {c_aux:,} `cais/mmlu` auxiliary_train questions.")
                except Exception as e:
                    print(f"  [Warning] Could not stream `cais/mmlu` auxiliary_train: {e}")

                # 1B. AllenAI SciQ STEM contrastive distractor pairs
                try:
                    sciq_tr = load_dataset("allenai/sciq", split="train", streaming=True)
                    c_sciq = 0
                    for item in sciq_tr:
                        q = item.get("question", "").strip()
                        corr = item.get("correct_answer", "").strip()
                        d1 = item.get("distractor1", "").strip()
                        d2 = item.get("distractor2", "").strip()
                        d3 = item.get("distractor3", "").strip()
                        if q and corr and d1 and d2 and d3:
                            opts = [corr, d1, d2, d3]
                            rng.shuffle(opts)
                            chosen = letters[opts.index(corr)]
                            opt_map = {letters[i]: opts[i] for i in range(4)}
                            for p_txt, c_txt, r_txt, mode in build_contrastive_mc_pairs(q, opt_map, chosen, rng):
                                yield {"prompt": p_txt, "chosen": c_txt, "rejected": r_txt, "mode": mode}
                                if mode == "mmlu_raw":
                                    mmlu_cnt += 1
                                else:
                                    arc_cnt += 1
                            c_sciq += 1
                    print(f"  [DPO Pillar 1B] Built contrastive pairs from {c_sciq:,} `allenai/sciq` STEM train questions.")
                except Exception as e:
                    print(f"  [Warning] Could not stream `allenai/sciq`: {e}")

                # 1C. ARC-Challenge, ARC-Easy, OpenBookQA, CommonsenseQA train splits
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
                                    for p_txt, c_txt, r_txt, mode in build_contrastive_mc_pairs(q, opt_map, ans_key, rng):
                                        yield {"prompt": p_txt, "chosen": c_txt, "rejected": r_txt, "mode": mode}
                                        if mode == "mmlu_raw":
                                            mmlu_cnt += 1
                                        else:
                                            arc_cnt += 1
                                    c_sub += 1
                        print(f"  [DPO Pillar 1C] Built contrastive pairs from {c_sub:,} `{ds_name}` ({cfg_name}) train questions.")
                    except Exception as e:
                        print(f"  [Warning] Could not stream `{ds_name}` ({cfg_name}): {e}")

                # 2. Rowan/hellaswag train split (Exact lm_eval raw_completion format!)
                try:
                    hswag_ds = load_dataset("Rowan/hellaswag", split="train", streaming=True)
                    c_hswag = 0
                    for item in hswag_ds:
                        if c_hswag >= args.hellaswag_dpo_samples:
                            break
                        act_label = item.get("activity_label", "").strip()
                        ctx_a = item.get("ctx_a", "").strip()
                        ctx_b = item.get("ctx_b", "").strip()
                        endings = item.get("endings", [])
                        lbl = str(item.get("label", "")).strip()
                        if ctx_a and len(endings) == 4 and lbl in ("0", "1", "2", "3"):
                            ans_idx = int(lbl)
                            ctx_b_cap = (ctx_b[0].upper() + ctx_b[1:]) if ctx_b else ""
                            # Exact lm_eval HellaSwag doc_to_text formatting:
                            full_ctx = f"{act_label}: {ctx_a} {ctx_b_cap}".strip() if act_label else f"{ctx_a} {ctx_b_cap}".strip()
                            chosen_end = str(endings[ans_idx]).strip()
                            dist_indices = [i for i in range(4) if i != ans_idx and str(endings[i]).strip()]
                            if len(full_ctx) >= 10 and chosen_end and dist_indices:
                                rej_idx = rng.choice(dist_indices)
                                rej_end = str(endings[rej_idx]).strip()
                                yield {
                                    "prompt": full_ctx,
                                    "chosen": f" {chosen_end}",
                                    "rejected": f" {rej_end}",
                                    "mode": "raw_completion",
                                }
                                cloze_cnt += 1
                                c_hswag += 1
                    print(f"  [DPO Pillar 2] Built {c_hswag:,} `Rowan/hellaswag` narrative completion contrastive pairs.")
                except Exception as e:
                    print(f"  [Warning] Could not stream `Rowan/hellaswag`: {e}")

                # 3. PIQA Physical Commonsense via Hugging Face auto-converted Parquet branch (`refs/convert/parquet`)
                try:
                    piqa_ds = load_dataset(
                        "parquet",
                        data_files="hf://datasets/ybisk/piqa@refs/convert/parquet/plain_text/train/*.parquet",
                        split="train",
                        streaming=True,
                    )
                    c_piqa = 0
                    for item in piqa_ds:
                        if c_piqa >= args.piqa_dpo_samples:
                            break
                        goal = item.get("goal", "").strip()
                        sol1 = item.get("sol1", "").strip()
                        sol2 = item.get("sol2", "").strip()
                        lbl = item.get("label", -1)
                        if goal and sol1 and sol2 and int(lbl) in (0, 1):
                            chosen_sol = sol1 if int(lbl) == 0 else sol2
                            rej_sol = sol2 if int(lbl) == 0 else sol1
                            # Exact lm_eval PIQA format: `Question: <goal>\nAnswer: <sol>`
                            yield {
                                "prompt": f"Question: {goal}\nAnswer:",
                                "chosen": f" {chosen_sol}",
                                "rejected": f" {rej_sol}",
                                "mode": "arc_raw",
                            }
                            arc_cnt += 1
                            c_piqa += 1
                    print(f"  [DPO Pillar 3] Built {c_piqa:,} `ybisk/piqa` (Parquet) physical commonsense contrastive pairs.")
                except Exception as e:
                    print(f"  [Warning] Could not stream `ybisk/piqa` Parquet: {e}")

                # 4. Winogrande XL Cloze Coreference Preference Pairs
                try:
                    wino_ds = load_dataset("allenai/winogrande", "winogrande_xl", split="train", streaming=True)
                    c_wino = 0
                    for item in wino_ds:
                        if c_wino >= args.winogrande_dpo_samples:
                            break
                        sent = item.get("sentence", "").strip()
                        opt1 = item.get("option1", "").strip()
                        opt2 = item.get("option2", "").strip()
                        ans = str(item.get("answer", "")).strip()
                        if "_" in sent and opt1 and opt2 and ans in ("1", "2"):
                            chosen_opt = opt1 if ans == "1" else opt2
                            rej_opt = opt2 if ans == "1" else opt1
                            prefix, suffix = sent.split("_", 1)
                            prefix_clean = prefix.rstrip()
                            if len(prefix_clean) >= 6:
                                yield {
                                    "prompt": prefix_clean,
                                    "chosen": f" {chosen_opt}{suffix}",
                                    "rejected": f" {rej_opt}{suffix}",
                                    "mode": "raw_completion",
                                }
                                cloze_cnt += 1
                                c_wino += 1
                    print(f"  [DPO Pillar 4] Built {c_wino:,} `allenai/winogrande` cloze preference pairs.")
                except Exception as e:
                    print(f"  [Warning] Could not stream `allenai/winogrande`: {e}")

                # 5. UltraFeedback Binarized General Instruction / Reasoning Pairs
                try:
                    uf_ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs", streaming=True)
                    c_uf = 0
                    for item in uf_ds:
                        if c_uf >= args.ultrafeedback_samples:
                            break
                        prompt = item.get("prompt", "").strip()
                        c_msgs = item.get("chosen", [])
                        r_msgs = item.get("rejected", [])
                        if prompt and len(c_msgs) >= 2 and len(r_msgs) >= 2:
                            c_txt = c_msgs[-1].get("content", "").strip()
                            r_txt = r_msgs[-1].get("content", "").strip()
                            if c_txt and r_txt and c_txt != r_txt:
                                yield {"prompt": prompt, "chosen": c_txt, "rejected": r_txt, "mode": "auto"}
                                uf_cnt += 1
                                c_uf += 1
                    print(f"  [DPO Pillar 5] Loaded {c_uf:,} `HuggingFaceH4/ultrafeedback_binarized` preference pairs.")
                except Exception as e:
                    print(f"  [Warning] Could not stream `ultrafeedback_binarized`: {e}")

                print(
                    f"  --> Total 6-Pillar Contrastive Pairs: {mmlu_cnt:,} `mmlu_raw`, "
                    f"{arc_cnt:,} `arc_raw` (ARC/SciQ/OBQA/CSQA/PIQA), "
                    f"{cloze_cnt:,} `raw_completion` (HellaSwag/Winogrande), {uf_cnt:,} `ultrafeedback`!"
                )

            raw_ds = Dataset.from_generator(
                raw_dpo_generator,
                cache_dir=args.cache_dir,
                writer_batch_size=2000,
            )
            raw_ds = raw_ds.shuffle(seed=args.seed)
            gc.collect()

            max_len = args.max_len
            lrad_c = args.lrad_chunk_size
            eos_id = tokenizer.eos_token_id
            has_chat_tmpl = hasattr(tokenizer, "apply_chat_template")
            mc_w = float(args.mc_answer_weight)
            short_w = float(args.short_answer_weight)
            cc_w = float(args.cross_chunk_weight)

            def parallel_tokenize_dpo_batch(examples):
                prompts = examples["prompt"]
                chosens = examples["chosen"]
                rejecteds = examples["rejected"]
                modes = examples["mode"]

                out_c_ids, out_c_labels, out_c_vmask = [], [], []
                out_r_ids, out_r_labels = [], []

                for idx, (u_txt, c_txt, r_txt, mode) in enumerate(zip(prompts, chosens, rejecteds, modes)):
                    if mode in ("mmlu_raw", "arc_raw", "raw_completion"):
                        p_ids = tokenizer(u_txt, add_special_tokens=False)["input_ids"]
                        c_full = tokenizer(u_txt + c_txt, add_special_tokens=False)["input_ids"] + [eos_id]
                        r_full = tokenizer(u_txt + r_txt, add_special_tokens=False)["input_ids"] + [eos_id]
                    elif (idx % 7 == 0) and has_chat_tmpl:
                        p_ids = tokenizer.apply_chat_template(
                            [{"role": "user", "content": u_txt}], tokenize=True, add_generation_prompt=True
                        )
                        c_full = tokenizer.apply_chat_template(
                            [{"role": "user", "content": u_txt}, {"role": "assistant", "content": c_txt}],
                            tokenize=True, add_generation_prompt=False,
                        )
                        r_full = tokenizer.apply_chat_template(
                            [{"role": "user", "content": u_txt}, {"role": "assistant", "content": r_txt}],
                            tokenize=True, add_generation_prompt=False,
                        )
                    else:
                        p_str = f"Question: {u_txt.strip()}\nAnswer:"
                        c_str = f"Question: {u_txt.strip()}\nAnswer: {c_txt.strip()}"
                        r_str = f"Question: {u_txt.strip()}\nAnswer: {r_txt.strip()}"
                        p_ids = tokenizer(p_str, add_special_tokens=False)["input_ids"]
                        c_full = tokenizer(c_str, add_special_tokens=False)["input_ids"] + [eos_id]
                        r_full = tokenizer(r_str, add_special_tokens=False)["input_ids"] + [eos_id]

                    # Never cut a chosen or rejected sequence in half
                    if len(c_full) > max_len or len(r_full) > max_len:
                        continue
                    p_len = len(p_ids)
                    c_resp_len = len(c_full) - p_len
                    r_resp_len = len(r_full) - p_len
                    if c_resp_len <= 0 or r_resp_len <= 0:
                        continue

                    if mode == "mmlu_raw":
                        ans_w = mc_w
                    elif mode in ("arc_raw", "raw_completion") or c_resp_len <= 48:
                        ans_w = short_w
                    else:
                        ans_w = 1.0

                    # Build padded chosen sequence + labels + focal/cross-chunk valid_mask
                    c_rem = max_len - len(c_full)
                    c_pad = c_full + [eos_id] * c_rem
                    c_lbl = [-100] * p_len + c_full[p_len:] + [-100] * c_rem

                    c_vmask = []
                    for pos in range(len(c_full)):
                        w = 1.0 if pos < p_len else ans_w
                        if (pos // lrad_c) > 0:
                            w = w * cc_w
                        c_vmask.append(w)
                    c_vmask.extend([0.0] * c_rem)

                    # Build padded rejected sequence + labels
                    r_rem = max_len - len(r_full)
                    r_pad = r_full + [eos_id] * r_rem
                    r_lbl = [-100] * p_len + r_full[p_len:] + [-100] * r_rem

                    out_c_ids.append(c_pad)
                    out_c_labels.append(c_lbl)
                    out_c_vmask.append(c_vmask)
                    out_r_ids.append(r_pad)
                    out_r_labels.append(r_lbl)

                return {
                    "chosen_ids": out_c_ids,
                    "chosen_labels": out_c_labels,
                    "chosen_valid_mask": out_c_vmask,
                    "rejected_ids": out_r_ids,
                    "rejected_labels": out_r_labels,
                }

            print(f"  - Tokenizing 6-Pillar DPO pairs (max_len={max_len}, num_proc={args.num_proc})...")
            processed_ds = raw_ds.map(
                parallel_tokenize_dpo_batch,
                batched=True,
                batch_size=1000,
                writer_batch_size=1000,
                num_proc=args.num_proc,
                remove_columns=raw_ds.column_names,
                keep_in_memory=False,
                desc=f"Tokenizing 6-Pillar DPO Pairs ({max_len})",
            )
            gc.collect()

            processed_ds = processed_ds.shuffle(seed=args.seed)
            total_pairs = len(processed_ds)
            val_cnt = min(args.val_samples, max(50, total_pairs // 40))
            train_cnt = total_pairs - val_cnt

            ds_dict = DatasetDict({
                "train": processed_ds.select(range(val_cnt, val_cnt + train_cnt)),
                "validation": processed_ds.select(range(val_cnt)),
            })

            os.makedirs(args.dpo_data_dir, exist_ok=True)
            ds_dict.save_to_disk(args.dpo_data_dir, num_proc=min(4, args.num_proc))
            print(
                f"[DPO Dataset Builder] Saved {train_cnt:,} train pairs ({train_cnt // pairs_per_step:,} unique steps) "
                f"& {val_cnt:,} val pairs to `{args.dpo_data_dir}`!"
            )
    else:
        if accelerator.is_main_process:
            print(f"[DPO Dataset Loader] Found existing 6-Pillar DPO dataset at `{args.dpo_data_dir}`. Loading from disk!")

    accelerator.wait_for_everyone()
    return load_from_disk(args.dpo_data_dir)


def collate_dpo(batch):
    return {
        "chosen_ids": torch.tensor([x["chosen_ids"] for x in batch], dtype=torch.long),
        "chosen_labels": torch.tensor([x["chosen_labels"] for x in batch], dtype=torch.long),
        "chosen_valid_mask": torch.tensor([x["chosen_valid_mask"] for x in batch], dtype=torch.float32),
        "rejected_ids": torch.tensor([x["rejected_ids"] for x in batch], dtype=torch.long),
        "rejected_labels": torch.tensor([x["rejected_labels"] for x in batch], dtype=torch.long),
    }


def chunked_distillm2_and_teacher_dpo(
    h_pol,
    h_ref,
    h_teach,
    labels,
    valid_mask,
    pol_lm_head,
    ref_lm_head,
    teach_lm_head,
    topk=64,
    skew_alpha=0.1,
    temperature=1.0,
    chunk_size=128,
    is_chosen=True,
):
    """
    Memory-safe chunked DistiLLM-2 Top-K Skew-KD + Teacher-Guided DPO over the 248,320 vocabulary:
      1. Computes per-example length-normalized sequence log-probabilities for Policy (`pol_seq_logp`),
         Frozen Stage 4A Reference Student (`ref_seq_logp`), and Larger Teacher (`teach_seq_logp`).
      2. Strictly on response tokens (`labels != -100`), selects the Top-K (`K=64`) Teacher logits,
         eliminating the 248,256 noisy tail tokens so a 2B/4B Teacher never flattens Student confidence.
      3. On Chosen sequences (`is_chosen=True`):
         - Computes 50% Skew Forward KL (`D_SKL`) + 50% Skew Reverse KL (`D_SRKL`) weighted by Focal `valid_mask`.
         - Computes Focal Chosen NLL (`focal_nll`) to prevent DPO likelihood collapse.
      4. On Rejected sequences (`is_chosen=False`):
         - Computes Mode-Seeking Skew Reverse KL (`D_SRKL`) on response tokens to suppress distractor modes.
    """
    p_dtype = pol_lm_head.weight.dtype
    r_dtype = ref_lm_head.weight.dtype
    t_dtype = teach_lm_head.weight.dtype

    h_p_shift = h_pol[:, :-1, :].to(p_dtype)
    h_r_shift = h_ref[:, :-1, :].detach().to(r_dtype)
    h_t_shift = h_teach[:, :-1, :].detach().to(t_dtype)

    shift_labels = labels[:, 1:]
    resp_mask = (shift_labels != -100).float()
    if valid_mask is not None:
        focal_resp_mask = valid_mask[:, 1:] * resp_mask
    else:
        focal_resp_mask = resp_mask

    B, L_minus_1, _ = h_p_shift.shape

    def _chunk_fn(hp_c, hr_c, ht_c, lbl_c, f_mask_c, r_mask_c):
        logits_p = pol_lm_head(hp_c.to(p_dtype)).float()
        v_s = logits_p.size(-1)

        with torch.no_grad():
            logits_r = ref_lm_head(hr_c.to(r_dtype)).float()
            logits_t = teach_lm_head(ht_c.to(t_dtype)).float()
            v_min = min(v_s, logits_t.size(-1))

        safe_lbl = lbl_c.clamp(min=0)

        logp_p_full = F.log_softmax(logits_p[..., :v_min], dim=-1)
        tok_logp_p = logp_p_full.gather(-1, safe_lbl.unsqueeze(-1)).squeeze(-1) * r_mask_c
        seq_logp_p_chunk = tok_logp_p.sum(dim=-1)

        with torch.no_grad():
            logp_r_full = F.log_softmax(logits_r[..., :v_min], dim=-1)
            tok_logp_r = logp_r_full.gather(-1, safe_lbl.unsqueeze(-1)).squeeze(-1) * r_mask_c
            seq_logp_r_chunk = tok_logp_r.sum(dim=-1)

            logp_t_full = F.log_softmax(logits_t[..., :v_min], dim=-1)
            tok_logp_t = logp_t_full.gather(-1, safe_lbl.unsqueeze(-1)).squeeze(-1) * r_mask_c
            seq_logp_t_chunk = tok_logp_t.sum(dim=-1)

            # Top-K Teacher Logit Selection
            k_eff = min(topk, v_min)
            top_vals_t, top_idx_t = torch.topk(logits_t[..., :v_min] / temperature, k=k_eff, dim=-1)
            probs_t_k = F.softmax(top_vals_t, dim=-1).clamp(min=1e-8)

        logits_p_k = (logits_p[..., :v_min] / temperature).gather(-1, top_idx_t)
        probs_p_k = F.softmax(logits_p_k, dim=-1).clamp(min=1e-8)

        # DistiLLM-2 Skew Forward KL: D_KL(P || alpha*P + (1-alpha)*Q)
        mixed_q = (skew_alpha * probs_t_k + (1.0 - skew_alpha) * probs_p_k).clamp(min=1e-8)
        skl_tok = (probs_t_k * (probs_t_k.log() - mixed_q.log())).sum(dim=-1) * (temperature ** 2)

        # DistiLLM-2 Skew Reverse KL (Mode-Seeking): D_KL(Q || (1-alpha)*P + alpha*Q)
        mixed_p = ((1.0 - skew_alpha) * probs_t_k + skew_alpha * probs_p_k).clamp(min=1e-8)
        srkl_tok = (probs_p_k * (probs_p_k.log() - mixed_p.log())).sum(dim=-1) * (temperature ** 2)

        if is_chosen:
            kd_tok = 0.5 * skl_tok + 0.5 * srkl_tok
            kd_chunk_sum = (kd_tok * f_mask_c).sum()
            nll_chunk_sum = (-tok_logp_p * f_mask_c).sum()
            raw_ce_chunk_sum = (-tok_logp_p.detach()).sum()
        else:
            kd_chunk_sum = (srkl_tok * f_mask_c).sum()
            nll_chunk_sum = logits_p.sum() * 0.0
            raw_ce_chunk_sum = torch.tensor(0.0, device=logits_p.device)

        return seq_logp_p_chunk, seq_logp_r_chunk, seq_logp_t_chunk, kd_chunk_sum, nll_chunk_sum, raw_ce_chunk_sum

    pol_logp_sum = torch.zeros(B, dtype=torch.float32, device=h_pol.device)
    ref_logp_sum = torch.zeros(B, dtype=torch.float32, device=h_pol.device)
    teach_logp_sum = torch.zeros(B, dtype=torch.float32, device=h_pol.device)
    total_kd_sum = 0.0
    total_nll_sum = 0.0
    total_raw_ce_sum = 0.0

    for start in range(0, L_minus_1, chunk_size):
        end = min(start + chunk_size, L_minus_1)
        hp_c = h_p_shift[:, start:end, :].contiguous()
        hr_c = h_r_shift[:, start:end, :].contiguous()
        ht_c = h_t_shift[:, start:end, :].contiguous()
        lbl_c = shift_labels[:, start:end].contiguous()
        f_mask_c = focal_resp_mask[:, start:end].contiguous()
        r_mask_c = resp_mask[:, start:end].contiguous()

        if h_pol.requires_grad:
            lp_p, lp_r, lp_t, kd_s, nll_s, raw_s = checkpoint.checkpoint(
                _chunk_fn, hp_c, hr_c, ht_c, lbl_c, f_mask_c, r_mask_c, use_reentrant=False
            )
        else:
            lp_p, lp_r, lp_t, kd_s, nll_s, raw_s = _chunk_fn(hp_c, hr_c, ht_c, lbl_c, f_mask_c, r_mask_c)

        pol_logp_sum = pol_logp_sum + lp_p
        ref_logp_sum = ref_logp_sum + lp_r
        teach_logp_sum = teach_logp_sum + lp_t
        total_kd_sum = total_kd_sum + kd_s
        if is_chosen:
            total_nll_sum = total_nll_sum + nll_s
            total_raw_ce_sum = total_raw_ce_sum + raw_s

    resp_tok_counts = resp_mask.sum(dim=-1).clamp(min=1.0)
    pol_seq_logp = pol_logp_sum / resp_tok_counts
    ref_seq_logp = ref_logp_sum / resp_tok_counts
    teach_seq_logp = teach_logp_sum / resp_tok_counts

    f_denom = focal_resp_mask.sum().clamp(min=1.0)
    mean_skew_kd = total_kd_sum / f_denom

    if is_chosen:
        mean_focal_nll = total_nll_sum / f_denom
        raw_ce_s = total_raw_ce_sum / resp_mask.sum().clamp(min=1.0)
        return pol_seq_logp, ref_seq_logp, teach_seq_logp, mean_skew_kd, mean_focal_nll, raw_ce_s

    return pol_seq_logp, ref_seq_logp, teach_seq_logp, mean_skew_kd


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
def evaluate_stage4b(accelerator, wrapper, ref_student, teacher, t_backbone, val_loader, args):
    wrapper.eval()
    unwrapped_student = accelerator.unwrap_model(wrapper).student

    local_sums = torch.zeros(6, dtype=torch.float64, device=accelerator.device)
    local_cnt = torch.zeros(1, dtype=torch.float64, device=accelerator.device)

    for b_idx, batch in enumerate(val_loader):
        if args.eval_batches > 0 and b_idx >= args.eval_batches:
            break
        c_ids = batch["chosen_ids"].to(accelerator.device, non_blocking=True)
        c_lbl = batch["chosen_labels"].to(accelerator.device, non_blocking=True)
        c_vm = batch["chosen_valid_mask"].to(accelerator.device, non_blocking=True)
        r_ids = batch["rejected_ids"].to(accelerator.device, non_blocking=True)
        r_lbl = batch["rejected_labels"].to(accelerator.device, non_blocking=True)

        ref_c_h = ref_student.model(input_ids=c_ids, attention_mask=None, output_hidden_states=False, use_cache=False).last_hidden_state
        ref_r_h = ref_student.model(input_ids=r_ids, attention_mask=None, output_hidden_states=False, use_cache=False).last_hidden_state
        t_c_h = t_backbone(input_ids=c_ids, attention_mask=None, output_hidden_states=False, use_cache=False).last_hidden_state
        t_r_h = t_backbone(input_ids=r_ids, attention_mask=None, output_hidden_states=False, use_cache=False).last_hidden_state

        pol_c_h = wrapper(c_ids)
        pol_r_h = wrapper(r_ids)

        pol_c_lp, ref_c_lp, teach_c_lp, kd_c, focal_nll, raw_ce_s = chunked_distillm2_and_teacher_dpo(
            pol_c_h, ref_c_h, t_c_h, c_lbl, c_vm,
            unwrapped_student.lm_head, ref_student.lm_head, teacher.lm_head,
            topk=args.topk_logits, skew_alpha=args.skew_alpha, temperature=args.temperature,
            chunk_size=args.logit_chunk_size, is_chosen=True,
        )
        pol_r_lp, ref_r_lp, teach_r_lp, kd_r = chunked_distillm2_and_teacher_dpo(
            pol_r_h, ref_r_h, t_r_h, r_lbl, None,
            unwrapped_student.lm_head, ref_student.lm_head, teacher.lm_head,
            topk=args.topk_logits, skew_alpha=args.skew_alpha, temperature=args.temperature,
            chunk_size=args.logit_chunk_size, is_chosen=False,
        )

        alpha_t = args.teacher_ref_alpha
        anchor_c_lp = (1.0 - alpha_t) * ref_c_lp + alpha_t * teach_c_lp
        anchor_r_lp = (1.0 - alpha_t) * ref_r_lp + alpha_t * teach_r_lp

        logits_diff = (pol_c_lp - anchor_c_lp) - (pol_r_lp - anchor_r_lp)
        dpo_loss = -F.logsigmoid(args.beta * logits_diff).mean()

        self_margin = (pol_c_lp - ref_c_lp) - (pol_r_lp - ref_r_lp)
        win_rate = (self_margin > 0).float().mean()
        pol_pref_acc = (pol_c_lp > pol_r_lp).float().mean()

        skew_kd_loss = 0.65 * kd_c + 0.35 * kd_r
        loss = args.dpo_weight * dpo_loss + args.skew_kd_weight * skew_kd_loss + args.nll_weight * focal_nll

        local_sums[0] += loss.to(torch.float64)
        local_sums[1] += dpo_loss.to(torch.float64)
        local_sums[2] += skew_kd_loss.to(torch.float64)
        local_sums[3] += raw_ce_s.to(torch.float64)
        local_sums[4] += win_rate.to(torch.float64)
        local_sums[5] += pol_pref_acc.to(torch.float64)
        local_cnt[0] += 1.0

    g_sums = accelerator.reduce(local_sums, reduction="sum")
    g_cnt = accelerator.reduce(local_cnt, reduction="sum").clamp(min=1.0)
    means = (g_sums / g_cnt).tolist()
    wrapper.train()
    return means[0], means[1], means[2], means[3], means[4], means[5]


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    use_compile = not args.no_compile
    use_gc = not args.no_grad_checkpointing
    accelerator = Accelerator(mixed_precision="bf16", gradient_accumulation_steps=args.grad_accum_steps)

    ckpt_to_load = args.stage4a_ckpt_dir
    if not os.path.exists(ckpt_to_load) and os.path.exists(os.path.dirname(ckpt_to_load)):
        ckpt_to_load = os.path.dirname(ckpt_to_load)

    tokenizer = AutoTokenizer.from_pretrained(ckpt_to_load, trust_remote_code=True)

    # 1. Prepare or load 6-Pillar Contrastive DPO dataset
    ds_dict = prepare_or_load_dpo_dataset(args, tokenizer, accelerator)
    train_ds = ds_dict["train"]
    val_ds = ds_dict["validation"]
    seq_len = len(train_ds[0]["chosen_ids"])

    # 2. Reconstruct 0.67B Policy Student AND Frozen Reference Student from Stage 4A Best Checkpoint
    _, base_08b_backbone, _ = load_qwen3_5_text_teacher(args.student_base_id, dtype=torch.bfloat16, device="cpu")
    student = load_hybrid_student_checkpoint(
        ckpt_to_load, teacher_backbone=base_08b_backbone, teacher_id=args.student_base_id, dtype=torch.bfloat16
    )
    del base_08b_backbone
    gc.collect()

    if getattr(student.config, "tie_word_embeddings", True):
        student.lm_head.weight = student.model.token_embedding.weight

    ref_student = copy.deepcopy(student)
    ref_student.eval()
    for p in ref_student.parameters():
        p.requires_grad = False

    if use_gc:
        student.gradient_checkpointing_enable()

    if accelerator.is_main_process:
        print(f"Loading Larger DPO + Top-{args.topk_logits} Skew-KD Teacher: {args.teacher_id} ...")
    teacher, t_backbone, _ = load_qwen3_5_text_teacher(args.teacher_id, dtype=torch.bfloat16, device="cpu")
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    if use_compile:
        for m in (student, ref_student):
            for layer in m.model.layers:
                if isinstance(layer, LRADDecoderBlock):
                    layer.ln1.forward = torch.compile(layer.ln1.forward, mode=args.compile_mode, dynamic=False)
                    layer.attn.chunk_forward = torch.compile(layer.attn.chunk_forward, mode=args.compile_mode, dynamic=False)
                    layer.ln2.forward = torch.compile(layer.ln2.forward, mode=args.compile_mode, dynamic=False)

    t_backbone.to(accelerator.device)
    teacher.lm_head.to(accelerator.device)
    ref_student.to(accelerator.device)
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
        collate_fn=collate_dpo, num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.micro_batch_size, shuffle=False,
        collate_fn=collate_dpo, num_workers=4, pin_memory=True, drop_last=True
    )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=int(args.warmup_ratio * args.max_steps), num_training_steps=args.max_steps
    )

    wrapper = Stage4Wrapper(student)
    wrapper, optimizer, train_loader, val_loader = accelerator.prepare(wrapper, optimizer, train_loader, val_loader)
    wrapper.train()

    pairs_per_step = args.micro_batch_size * accelerator.num_processes * args.grad_accum_steps
    tokens_per_step = pairs_per_step * seq_len * 2
    epochs_covered = (args.max_steps * pairs_per_step) / max(len(train_ds), 1)

    if accelerator.is_main_process:
        val_scope = "ENTIRE validation set" if args.eval_batches <= 0 else f"{args.eval_batches * accelerator.num_processes} batches"
        print("=" * 90)
        print("STAGE 4B: 6-PILLAR CONTRASTIVE DPO + DISTILLM-2 TOP-K SKEW-KD (LARGER TEACHER)")
        print(f"  - Reference Policy pi_0: Frozen Stage 4A Student ({ckpt_to_load})")
        print(f"  - Larger Teacher       : {args.teacher_id} (Top-{args.topk_logits} Skew-KL/SRKL + {int(args.teacher_ref_alpha*100)}% DPO Margin Blend)")
        print(f"  - Disk Dataset Path    : {args.dpo_data_dir} (max_len={seq_len})")
        print(f"  - Train / Val Pairs    : {len(train_ds):,} train | {len(val_ds):,} val")
        print(f"  - Epochs Covered       : {epochs_covered:.2f} epochs ({args.max_steps:,} steps / {len(train_ds)//pairs_per_step:,} max unique steps)")
        print(f"  - Decoupled LRs        : LRAD={args.lrad_lr:.2e} | Backbone={args.backbone_lr:.2e}")
        print(f"  - Loss Formulation     : {args.dpo_weight}*Teacher_DPO(beta={args.beta}) + {args.skew_kd_weight}*Top{args.topk_logits}_SkewKD + {args.nll_weight}*Focal_NLL")
        print(f"  - Global Batch Size    : {pairs_per_step} pairs/step ({tokens_per_step:,} tok/step)")
        print(f"  - Validation Scope     : {val_scope}")
        print("=" * 90)

    step = 0
    t0 = time.time()
    best_val_loss = float("inf")
    running_sums = torch.zeros(6, dtype=torch.float64, device=accelerator.device)
    running_cnt = torch.zeros(1, dtype=torch.float64, device=accelerator.device)

    while step < args.max_steps:
        for batch in train_loader:
            c_ids = batch["chosen_ids"].to(accelerator.device, non_blocking=True)
            c_lbl = batch["chosen_labels"].to(accelerator.device, non_blocking=True)
            c_vm = batch["chosen_valid_mask"].to(accelerator.device, non_blocking=True)
            r_ids = batch["rejected_ids"].to(accelerator.device, non_blocking=True)
            r_lbl = batch["rejected_labels"].to(accelerator.device, non_blocking=True)

            with accelerator.accumulate(wrapper):
                with torch.no_grad():
                    ref_c_h = ref_student.model(
                        input_ids=c_ids, attention_mask=None, output_hidden_states=False, use_cache=False
                    ).last_hidden_state
                    ref_r_h = ref_student.model(
                        input_ids=r_ids, attention_mask=None, output_hidden_states=False, use_cache=False
                    ).last_hidden_state
                    t_c_h = t_backbone(
                        input_ids=c_ids, attention_mask=None, output_hidden_states=False, use_cache=False
                    ).last_hidden_state
                    t_r_h = t_backbone(
                        input_ids=r_ids, attention_mask=None, output_hidden_states=False, use_cache=False
                    ).last_hidden_state

                pol_c_h = wrapper(c_ids)
                pol_r_h = wrapper(r_ids)
                unwrapped_student = accelerator.unwrap_model(wrapper).student

                pol_c_lp, ref_c_lp, teach_c_lp, kd_c, focal_nll, raw_ce_s = chunked_distillm2_and_teacher_dpo(
                    pol_c_h, ref_c_h, t_c_h, c_lbl, c_vm,
                    unwrapped_student.lm_head, ref_student.lm_head, teacher.lm_head,
                    topk=args.topk_logits, skew_alpha=args.skew_alpha, temperature=args.temperature,
                    chunk_size=args.logit_chunk_size, is_chosen=True,
                )
                pol_r_lp, ref_r_lp, teach_r_lp, kd_r = chunked_distillm2_and_teacher_dpo(
                    pol_r_h, ref_r_h, t_r_h, r_lbl, None,
                    unwrapped_student.lm_head, ref_student.lm_head, teacher.lm_head,
                    topk=args.topk_logits, skew_alpha=args.skew_alpha, temperature=args.temperature,
                    chunk_size=args.logit_chunk_size, is_chosen=False,
                )

                alpha_t = args.teacher_ref_alpha
                anchor_c_lp = (1.0 - alpha_t) * ref_c_lp + alpha_t * teach_c_lp
                anchor_r_lp = (1.0 - alpha_t) * ref_r_lp + alpha_t * teach_r_lp

                logits_diff = (pol_c_lp - anchor_c_lp) - (pol_r_lp - anchor_r_lp)
                dpo_loss = -F.logsigmoid(args.beta * logits_diff).mean()

                self_margin = (pol_c_lp - ref_c_lp) - (pol_r_lp - ref_r_lp)
                win_rate = (self_margin > 0).float().mean()
                pol_pref_acc = (pol_c_lp > pol_r_lp).float().mean()

                skew_kd_loss = 0.65 * kd_c + 0.35 * kd_r
                loss = args.dpo_weight * dpo_loss + args.skew_kd_weight * skew_kd_loss + args.nll_weight * focal_nll

                running_sums[0] += loss.detach().to(torch.float64)
                running_sums[1] += dpo_loss.detach().to(torch.float64)
                running_sums[2] += skew_kd_loss.detach().to(torch.float64)
                running_sums[3] += raw_ce_s.detach().to(torch.float64)
                running_sums[4] += win_rate.detach().to(torch.float64)
                running_sums[5] += pol_pref_acc.detach().to(torch.float64)
                running_cnt[0] += 1.0

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
                    g_cnt = accelerator.reduce(running_cnt, reduction="sum").clamp(min=1.0)
                    avg_l, avg_dpo, avg_kd, avg_ce, avg_wr, avg_pref = (g_sums / g_cnt).tolist()
                    running_sums.zero_()
                    running_cnt.zero_()

                    if accelerator.is_main_process:
                        elapsed = time.time() - t0
                        toks_sec = ((args.log_steps if step > 1 else 1) * tokens_per_step) / max(elapsed, 1e-5)
                        t0 = time.time()
                        lrs = scheduler.get_last_lr()
                        tok_str = "Compiling DONE" if step == 1 else f"{toks_sec:,.0f} tok/s"
                        print(
                            f"[Stage 4B] Step {step:04d}/{args.max_steps} | Total: {avg_l:.4f} | T_DPO: {avg_dpo:.4f} | "
                            f"Top{args.topk_logits}_SkewKD: {avg_kd:.4f} | Chosen_CE: {avg_ce:.4f} | "
                            f"Win-Rate: {avg_wr*100:.1f}% (Acc: {avg_pref*100:.1f}%) | LR: {lrs[0]:.2e}/{lrs[1]:.2e} | {tok_str}"
                        )

                # 2. Validation Evaluation
                if step % args.eval_steps == 0 or step >= args.max_steps:
                    accelerator.wait_for_everyone()
                    val_loss, val_dpo, val_kd, val_ce, val_wr, val_pref = evaluate_stage4b(
                        accelerator, wrapper, ref_student, teacher, t_backbone, val_loader, args
                    )
                    if accelerator.is_main_process:
                        print(
                            f"  >>> [VAL @ Step {step:04d}] Total: {val_loss:.4f} | Val T_DPO: {val_dpo:.4f} | "
                            f"Val SkewKD: {val_kd:.4f} | Val Chosen_CE: {val_ce:.4f} | "
                            f"Val Win-Rate: {val_wr*100:.1f}% (Acc: {val_pref*100:.1f}%)"
                        )
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            best_dir = os.path.join(args.output_dir, "best")
                            os.makedirs(best_dir, exist_ok=True)
                            unwrapped = accelerator.unwrap_model(wrapper).student
                            unwrapped.config.save_pretrained(best_dir)
                            tokenizer.save_pretrained(best_dir)
                            torch.save(unwrapped.state_dict(), os.path.join(best_dir, "pytorch_model.bin"))
                            print(f"  >>> New Best Stage 4B Val Loss ({best_val_loss:.4f})! Saved to {best_dir}")
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
                        print(f"  --> Saved Latest Stage 4B checkpoint at step {step} to {args.output_dir}")
                    accelerator.wait_for_everyone()
                    t0 = time.time()

                if step >= args.max_steps:
                    break

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()