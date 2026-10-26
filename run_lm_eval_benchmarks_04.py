# /home/martinkb/Desktop/BareTorch_Layerwise_Distill/run_lm_eval_benchmarks_04.py
import os
import json
import argparse
import torch
from transformers import AutoTokenizer

import lm_eval
from lm_eval.models.huggingface import HFLM

from init_student_00 import load_qwen3_5_text_teacher, load_hybrid_student_checkpoint

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")


def parse_args():
    parser = argparse.ArgumentParser(description="Run EleutherAI lm-evaluation-harness on Teacher vs Student and compute Deltas")
    parser.add_argument("--teacher_id", type=str, default="Qwen/Qwen3.5-0.8B", help="Teacher HF model ID")
    parser.add_argument("--student_ckpt", type=str, default="./checkpoints/stage3_final_distilled/best", help="Path to Student checkpoint")
    parser.add_argument(
        "--tasks",
        type=str,
        default="arc_easy,arc_challenge,piqa,winogrande,hellaswag,boolq,openbookqa,sciq,lambada_openai,mmlu",
        help="Comma-separated list of lm_eval tasks (includes MMLU by default!)",
    )
    parser.add_argument("--num_fewshot", type=int, default=0, help="Number of few-shot examples (0 for zero-shot, 5 for 5-shot MMLU)")
    parser.add_argument("--batch_size", type=int, default=16, help="Evaluation batch size for lm_eval")
    parser.add_argument("--limit", type=float, default=None, help="Optional limit per task/subtask (e.g., 100 for fast preview; None = full)")
    parser.add_argument("--device", type=str, default="cuda:0", help="GPU device to run evaluation on")
    parser.add_argument("--output_json", type=str, default="./lm_eval_comparison_results.json", help="Path to save full JSON comparison")
    return parser.parse_args()


def extract_primary_metric(task_dict: dict):
    """
    Extracts the primary accuracy or perplexity metric from an lm_eval task result dict.
    Prefers `acc_norm,none` -> `acc,none` -> `exact_match,strict-match` -> `word_perplexity,none`.
    """
    preferred_keys = [
        ("acc_norm,none", "acc_norm", True),
        ("acc,none", "acc", True),
        ("exact_match,strict-match", "exact_match", True),
        ("exact_match,flexible-extract", "exact_match", True),
        ("word_perplexity,none", "word_ppl", False),
        ("perplexity,none", "ppl", False),
        ("byte_perplexity,none", "byte_ppl", False),
    ]
    for key, short_name, is_percent in preferred_keys:
        if key in task_dict and isinstance(task_dict[key], (int, float)):
            val = float(task_dict[key])
            if is_percent and val <= 1.0:
                val *= 100.0
            return short_name, val, is_percent

    for k, v in task_dict.items():
        if isinstance(v, (int, float)) and "stderr" not in k:
            return k, float(v), False
    return "N/A", 0.0, False


def should_display_in_main_table(task_name: str, all_keys: set) -> bool:
    """
    When `mmlu` is evaluated, lm_eval populates 57 individual `mmlu_<subject>` keys
    plus the top-level `mmlu` aggregate (and 4 category keys).
    To keep the main table and overall macro-average clean (1 row for MMLU instead of 57),
    we display top-level tasks and hide the 57 individual MMLU subject sub-keys if `mmlu` is present.
    """
    if "mmlu" in all_keys and task_name.startswith("mmlu_") and task_name not in {
        "mmlu_stem", "mmlu_humanities", "mmlu_social_sciences", "mmlu_other"
    }:
        return False
    return True


def main():
    args = parse_args()
    task_list = [t.strip() for t in args.tasks.split(",") if t.strip()]

    ckpt_to_load = args.student_ckpt
    if not os.path.exists(ckpt_to_load) and os.path.exists(os.path.dirname(ckpt_to_load)):
        ckpt_to_load = os.path.dirname(ckpt_to_load)

    print("=" * 95)
    print("ELEUTHERAI LM-EVALUATION-HARNESS: TEACHER vs BARETORCH STUDENT BENCHMARK SUITE")
    print(f"  - Teacher Model : {args.teacher_id}")
    print(f"  - Student Model : {ckpt_to_load}")
    print(f"  - Tasks         : {', '.join(task_list)}")
    print(f"  - Few-Shot      : {args.num_fewshot}-shot | Batch Size: {args.batch_size} | Limit: {args.limit or 'FULL'}")
    print("=" * 95)

    tokenizer = AutoTokenizer.from_pretrained(ckpt_to_load, trust_remote_code=True)

    # -------------------------------------------------------------------------
    # 1. Load Teacher & Student, then evaluate Student first
    # -------------------------------------------------------------------------
    print("\n[1/2] Loading Teacher & Reconstructing Distilled BareTorch Student...")
    teacher, t_backbone, _ = load_qwen3_5_text_teacher(args.teacher_id, dtype=torch.bfloat16, device="cpu")
    teacher.eval()

    student = load_hybrid_student_checkpoint(
        ckpt_to_load, teacher_backbone=t_backbone, teacher_id=args.teacher_id, dtype=torch.bfloat16
    )
    student.to(args.device).eval()

    print(f"--> Running `lm_eval` on BareTorch Student ({ckpt_to_load})...")
    student_hflm = HFLM(
        pretrained=student,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_batch_size=args.batch_size,
    )
    student_eval_out = lm_eval.simple_evaluate(
        model=student_hflm,
        tasks=task_list,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
        log_samples=False,
    )
    student_results = student_eval_out["results"]

    # Free Student from GPU before moving Teacher to GPU
    del student_hflm
    student.to("cpu")
    torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # 2. Evaluate Teacher on the exact same tasks
    # -------------------------------------------------------------------------
    print(f"\n[2/2] Running `lm_eval` on Teacher ({args.teacher_id})...")
    teacher.to(args.device).eval()
    teacher_hflm = HFLM(
        pretrained=teacher,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_batch_size=args.batch_size,
    )
    teacher_eval_out = lm_eval.simple_evaluate(
        model=teacher_hflm,
        tasks=task_list,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
        log_samples=False,
    )
    teacher_results = teacher_eval_out["results"]

    # -------------------------------------------------------------------------
    # 3. Compute Deltas & Print Summary Table
    # -------------------------------------------------------------------------
    all_keys = set(teacher_results.keys()) & set(student_results.keys())
    print("\n" + "=" * 95)
    print(f"{'Task':<22} | {'Metric':<10} | {'Teacher':>12} | {'Student':>12} | {'Delta (S - T)':>15} | {'Retention':>11}")
    print("-" * 95)

    summary_rows = {}
    acc_teacher_vals = []
    acc_student_vals = []

    for task_name in sorted(all_keys):
        m_name, t_val, is_pct = extract_primary_metric(teacher_results[task_name])
        _, s_val, _ = extract_primary_metric(student_results[task_name])

        delta = s_val - t_val
        if is_pct:
            retention = (s_val / max(t_val, 1e-5)) * 100.0
            t_str = f"{t_val:.2f}%"
            s_str = f"{s_val:.2f}%"
            d_str = f"{delta:+.2f}%"
            r_str = f"{retention:.1f}%"
        else:
            retention = (t_val / max(s_val, 1e-5)) * 100.0
            t_str = f"{t_val:.2f}"
            s_str = f"{s_val:.2f}"
            d_str = f"{delta:+.2f}"
            r_str = f"{retention:.1f}%"

        summary_rows[task_name] = {
            "metric": m_name,
            "teacher": t_val,
            "student": s_val,
            "delta": delta,
            "retention_pct": retention,
        }

        if should_display_in_main_table(task_name, all_keys):
            print(f"{task_name:<22} | {m_name:<10} | {t_str:>12} | {s_str:>12} | {d_str:>15} | {r_str:>11}")
            # Exclude MMLU sub-categories from top-level macro average if top-level `mmlu` is already included
            if is_pct and not task_name.startswith("mmlu_"):
                acc_teacher_vals.append(t_val)
                acc_student_vals.append(s_val)

    if len(acc_teacher_vals) > 0:
        avg_t = sum(acc_teacher_vals) / len(acc_teacher_vals)
        avg_s = sum(acc_student_vals) / len(acc_student_vals)
        avg_d = avg_s - avg_t
        avg_ret = (avg_s / max(avg_t, 1e-5)) * 100.0
        print("-" * 95)
        print(
            f"{'MACRO AVERAGE (Acc)':<22} | {'mean_acc':<10} | "
            f"{avg_t:>11.2f}% | {avg_s:>11.2f}% | {avg_d:>+14.2f}% | {avg_ret:>10.1f}%"
        )
    print("=" * 95)

    with open(args.output_json, "w") as f:
        json.dump(summary_rows, f, indent=2)
    print(f"Saved full comparison table (including all 57 MMLU subjects) to: {args.output_json}")


if __name__ == "__main__":
    main()