# /home/martinkb/Desktop/BareTorch_Layerwise_Distill/run_generative_benchmarks_08.py
import os
import re
import ast
import json
import math
import time
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor
import torch
from accelerate import Accelerator
from transformers import AutoTokenizer
from datasets import load_dataset

from init_student_00 import load_qwen3_5_text_teacher, load_hybrid_student_checkpoint

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

NUM_REGEX = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-GPU Generative Benchmark Suite: GSM8K (CoT Math) & MBPP (Python Code Pass@1)"
    )
    parser.add_argument("--teacher_id", type=str, default="Qwen/Qwen3.5-0.8B", help="Teacher HF ID")
    parser.add_argument("--student_base_id", type=str, default="Qwen/Qwen3.5-0.8B", help="Base 0.8B architecture ID")
    parser.add_argument(
        "--student_ckpt",
        type=str,
        default="./checkpoints/stage4b_dpo_2B_4k/best",
        help="Path to Student checkpoint directory",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batched generation size per GPU")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Max generated tokens per problem")
    parser.add_argument(
        "--max_samples",
        type=int,
        default=250,
        help="Number of test problems per benchmark (-1 = full 1319 GSM8K & 500 MBPP; 250 = fast 2-min benchmark)",
    )
    parser.add_argument("--skip_teacher", action="store_true", help="Skip Teacher evaluation if cached in JSON")
    parser.add_argument(
        "--output_json",
        type=str,
        default="./generative_eval_stage4b_baseline.json",
        help="Output JSON file for results",
    )
    return parser.parse_args()


# ==============================================================================
# 1. Standard GSM8K & MBPP Verifiers
# ==============================================================================
def extract_gsm8k_answer(text: str):
    # Stop at next-question delimiters if the model continues generating
    for stop_seq in ("\nQuestion:", "\nQ:", "\n\nQuestion"):
        if stop_seq in text:
            text = text.split(stop_seq, 1)[0]

    if "####" in text:
        after = text.split("####")[-1].strip()
        m = NUM_REGEX.findall(after)
        if m:
            return m[0].replace(",", "")

    boxed = re.findall(r"\\boxed\{([^}]+)\}", text)
    if boxed:
        m = NUM_REGEX.findall(boxed[-1])
        if m:
            return m[0].replace(",", "")

    ans_phrase = re.findall(r"(?:answer is|equals|=)\s*\$?\s*([-+]?\d[\d,]*(?:\.\d+)?)", text.lower())
    if ans_phrase:
        return ans_phrase[-1].replace(",", "")

    all_nums = NUM_REGEX.findall(text)
    if all_nums:
        return all_nums[-1].replace(",", "")
    return None


def check_gsm8k_correct(gen_text: str, gold_target: str) -> bool:
    pred = extract_gsm8k_answer(gen_text)
    if pred is None:
        return False
    try:
        return math.isclose(float(pred), float(gold_target.replace(",", "")), rel_tol=1e-4, abs_tol=1e-4)
    except Exception:
        return pred.strip() == gold_target.strip()


def extract_python_code(text: str, func_sig_hint: str = "") -> str:
    if "```python" in text:
        code = text.split("```python", 1)[1].split("```")[0].strip()
    elif "```" in text:
        code = text.split("```", 1)[1].split("```")[0].strip()
    else:
        for stop_tok in ("\n# Test", "\nassert ", "\nif __name__", "\nYou are"):
            if stop_tok in text:
                text = text.split(stop_tok, 1)[0]
        code = text.strip()

    if code and not code.lstrip().startswith(("def ", "import ", "from ", "class ")) and func_sig_hint:
        indented = "\n".join("    " + line if line.strip() else line for line in code.splitlines())
        code = f"{func_sig_hint}\n{indented}"
    return code


def check_mbpp_one(args_tuple) -> tuple[bool, bool]:
    """Returns (passed_all_unit_tests: bool, valid_python_syntax: bool)."""
    gen_text, test_list, func_sig = args_tuple
    code = extract_python_code(gen_text, func_sig)
    if not code:
        return False, False
    try:
        ast.parse(code)
    except SyntaxError:
        return False, False

    test_script = code + "\n\n" + "\n".join(test_list) + "\nprint('ALL_MBPP_TESTS_PASSED')\n"
    try:
        res = subprocess.run(
            ["python3", "-I", "-c", test_script],
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        return ("ALL_MBPP_TESTS_PASSED" in res.stdout), True
    except Exception:
        return False, True


# ==============================================================================
# 2. Load Official GSM8K (`test`) & MBPP (`test`) Benchmark Prompts
# ==============================================================================
def load_gsm8k_test(max_samples=-1):
    # Standard 4-shot CoT prefix so both Base/Teacher and Student know the `#### <number>` convention
    fewshot_prefix = (
        "Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. "
        "How many clips did Natalia sell altogether in April and May?\n"
        "Answer: Let's think step by step.\n"
        "Natalia sold 48/2 = 24 clips in May.\n"
        "Natalia sold 48+24 = 72 clips altogether in April and May.\n"
        "#### 72\n\n"
        "Question: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. "
        "How much did she earn?\n"
        "Answer: Let's think step by step.\n"
        "Weng earns 12/60 = $0.2 per minute.\n"
        "Working 50 minutes, she earned 0.2 x 50 = $10.\n"
        "#### 10\n\n"
    )
    ds = load_dataset("openai/gsm8k", "main", split="test")
    items = []
    for idx, row in enumerate(ds):
        if 0 < max_samples <= idx:
            break
        q = row["question"].strip()
        ans = row["answer"].strip()
        gold_num = ans.split("####")[-1].strip().replace(",", "")
        prompt = f"{fewshot_prefix}Question: {q}\nAnswer: Let's think step by step.\n"
        items.append({"prompt": prompt, "gold_target": gold_num})
    return items


def load_mbpp_test(max_samples=-1):
    ds = load_dataset("google-research-datasets/mbpp", "full", split="test")
    items = []
    for idx, row in enumerate(ds):
        if 0 < max_samples <= idx:
            break
        desc = (row.get("text", "") or row.get("prompt", "")).strip()
        code = row.get("code", "").strip().replace("\r\n", "\n")
        tests = list(row.get("test_list", []))
        sig_match = re.search(r"def\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\([^)]*\)\s*:", code)
        sig_line = sig_match.group(0) if sig_match else ""
        first_test = tests[0] if tests else ""
        prompt = (
            "You are an expert Python programmer, and here is your task: "
            "Write a function to find the shared elements from the given two lists.\n"
            "Your code should pass these tests:\n\n"
            "assert set(similar_elements((3, 4, 5, 6),(5, 7, 4, 10))) == set((4, 5))\n\n"
            "```python\n"
            "def similar_elements(test_tup1, test_tup2):\n"
            "    return tuple(set(test_tup1) & set(test_tup2))\n"
            "```\n\n"
            f"You are an expert Python programmer, and here is your task: {desc}\n"
            f"Your code should pass these tests:\n\n{first_test}\n\n"
            f"```python\n"
        )
        items.append({"prompt": prompt, "test_list": tests, "func_sig": sig_line})
    return items


# ==============================================================================
# 3. Batched Multi-GPU Generation Runner
# ==============================================================================
@torch.no_grad()
def evaluate_model_on_generative_suite(model, tokenizer, gsm_items, mbpp_items, args, accelerator, model_label="Model"):
    model.eval()
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id

    # 1. Partition GSM8K across GPUs
    my_gsm = gsm_items[accelerator.process_index :: accelerator.num_processes]
    gsm_correct = 0
    t0 = time.time()

    for start in range(0, len(my_gsm), args.batch_size):
        batch = my_gsm[start : start + args.batch_size]
        # Left-pad batch for parallel generation
        tok_lists = [tokenizer(x["prompt"], add_special_tokens=False)["input_ids"] for x in batch]
        max_p = max(len(t) for t in tok_lists)
        padded_ids = [[pad_id] * (max_p - len(t)) + t for t in tok_lists]
        attn_mask = [[0] * (max_p - len(t)) + [1] * len(t) for t in tok_lists]

        inp_t = torch.tensor(padded_ids, dtype=torch.long, device=accelerator.device)
        mask_t = torch.tensor(attn_mask, dtype=torch.long, device=accelerator.device)

        gen_out = model.generate(
            input_ids=inp_t,
            attention_mask=mask_t,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=pad_id,
            eos_token_id=eos_id,
        )
        for i, item in enumerate(batch):
            gen_toks = gen_out[i, max_p:].tolist()
            gen_txt = tokenizer.decode(gen_toks, skip_special_tokens=True)
            if check_gsm8k_correct(gen_txt, item["gold_target"]):
                gsm_correct += 1

        if accelerator.is_main_process and ((start // args.batch_size) % 2 == 0):
            done = min(len(my_gsm), start + args.batch_size) * accelerator.num_processes
            print(f"  [{model_label} | GSM8K] Evaluated {min(done, len(gsm_items))}/{len(gsm_items)} problems ({time.time()-t0:.1f}s)...")

    # 2. Partition MBPP across GPUs
    my_mbpp = mbpp_items[accelerator.process_index :: accelerator.num_processes]
    mbpp_eval_tuples = []
    t1 = time.time()

    for start in range(0, len(my_mbpp), args.batch_size):
        batch = my_mbpp[start : start + args.batch_size]
        tok_lists = [tokenizer(x["prompt"], add_special_tokens=False)["input_ids"] for x in batch]
        max_p = max(len(t) for t in tok_lists)
        padded_ids = [[pad_id] * (max_p - len(t)) + t for t in tok_lists]
        attn_mask = [[0] * (max_p - len(t)) + [1] * len(t) for t in tok_lists]

        inp_t = torch.tensor(padded_ids, dtype=torch.long, device=accelerator.device)
        mask_t = torch.tensor(attn_mask, dtype=torch.long, device=accelerator.device)

        gen_out = model.generate(
            input_ids=inp_t,
            attention_mask=mask_t,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=pad_id,
            eos_token_id=eos_id,
        )
        for i, item in enumerate(batch):
            gen_toks = gen_out[i, max_p:].tolist()
            gen_txt = tokenizer.decode(gen_toks, skip_special_tokens=True)
            mbpp_eval_tuples.append((gen_txt, item["test_list"], item["func_sig"]))

        if accelerator.is_main_process and ((start // args.batch_size) % 2 == 0):
            done = min(len(my_mbpp), start + args.batch_size) * accelerator.num_processes
            print(f"  [{model_label} | MBPP]  Generated {min(done, len(mbpp_items))}/{len(mbpp_items)} solutions ({time.time()-t1:.1f}s)...")

    # Execute MBPP unit tests in parallel across CPU threads
    with ThreadPoolExecutor(max_workers=8) as ex:
        mbpp_results = list(ex.map(check_mbpp_one, mbpp_eval_tuples))

    mbpp_passed = sum(1 for p, _ in mbpp_results if p)
    mbpp_syntax = sum(1 for _, s in mbpp_results if s)

    local_t = torch.tensor(
        [gsm_correct, len(my_gsm), mbpp_passed, mbpp_syntax, len(my_mbpp)],
        dtype=torch.float64,
        device=accelerator.device,
    )
    global_t = accelerator.reduce(local_t, reduction="sum").tolist()

    gsm_acc = (global_t[0] / max(global_t[1], 1.0)) * 100.0
    mbpp_pass1 = (global_t[2] / max(global_t[4], 1.0)) * 100.0
    mbpp_syn_rate = (global_t[3] / max(global_t[4], 1.0)) * 100.0

    return {
        "gsm8k_cot": gsm_acc,
        "mbpp_pass@1": mbpp_pass1,
        "mbpp_syntax_valid": mbpp_syn_rate,
        "generative_macro_avg": 0.5 * (gsm_acc + mbpp_pass1),
    }


def main():
    args = parse_args()
    accelerator = Accelerator(mixed_precision="bf16")

    ckpt_to_load = args.student_ckpt
    if not os.path.exists(ckpt_to_load) and os.path.exists(os.path.dirname(ckpt_to_load)):
        ckpt_to_load = os.path.dirname(ckpt_to_load)

    tokenizer = AutoTokenizer.from_pretrained(ckpt_to_load, trust_remote_code=True)
    gsm_items = load_gsm8k_test(args.max_samples)
    mbpp_items = load_mbpp_test(args.max_samples)

    if accelerator.is_main_process:
        print("=" * 95)
        print("GENERATIVE REASONING & CODING BENCHMARK SUITE (GSM8K CoT + MBPP Python Unit-Tests)")
        print(f"  - Teacher Model : {args.teacher_id}")
        print(f"  - Student Model : {ckpt_to_load}")
        print(f"  - Test Problems : {len(gsm_items)} GSM8K Math CoT | {len(mbpp_items)} MBPP Python Unit-Test Tasks")
        print(f"  - Parallel GPUs : {accelerator.num_processes} GPUs (batch_size={args.batch_size}/GPU, max_new_tokens={args.max_new_tokens})")
        print("=" * 95)

    # 1. Evaluate Qwen3.5-0.8B Teacher
    teacher_scores = None
    if not args.skip_teacher:
        if accelerator.is_main_process:
            print(f"\n[1/2] Evaluating Teacher ({args.teacher_id}) on GSM8K & MBPP...")
        teacher, _, _ = load_qwen3_5_text_teacher(args.teacher_id, dtype=torch.bfloat16, device="cpu")
        teacher.to(accelerator.device)
        teacher.eval()
        teacher_scores = evaluate_model_on_generative_suite(
            teacher, tokenizer, gsm_items, mbpp_items, args, accelerator, model_label="Teacher"
        )
        del teacher
        torch.cuda.empty_cache()

    # 2. Evaluate BareTorch Hybrid Student
    if accelerator.is_main_process:
        print(f"\n[2/2] Evaluating BareTorch Student ({ckpt_to_load}) on GSM8K & MBPP...")
    _, base_backbone, _ = load_qwen3_5_text_teacher(args.student_base_id, dtype=torch.bfloat16, device="cpu")
    student = load_hybrid_student_checkpoint(
        ckpt_to_load, teacher_backbone=base_backbone, teacher_id=args.student_base_id, dtype=torch.bfloat16
    )
    del base_backbone
    student.to(accelerator.device)
    student.eval()

    student_scores = evaluate_model_on_generative_suite(
        student, tokenizer, gsm_items, mbpp_items, args, accelerator, model_label="Student"
    )

    if accelerator.is_main_process:
        print("\n" + "=" * 95)
        print(f"{'Generative Task':<22} | {'Metric':<12} | {'Teacher':>10} | {'Student':>10} | {'Delta (S - T)':>14} | {'Retention':>10}")
        print("-" * 95)
        rows = [
            ("gsm8k_cot (Math)", "exact_match", "gsm8k_cot"),
            ("mbpp (Python Code)", "pass@1", "mbpp_pass@1"),
            ("mbpp_syntax", "ast_valid", "mbpp_syntax_valid"),
        ]
        for label, metric, key in rows:
            s_val = student_scores[key]
            if teacher_scores is not None:
                t_val = teacher_scores[key]
                delta = s_val - t_val
                ret = (s_val / max(t_val, 1e-5)) * 100.0
                print(f"{label:<22} | {metric:<12} | {t_val:>9.2f}% | {s_val:>9.2f}% | {delta:>+13.2f}% | {ret:>9.1f}%")
            else:
                print(f"{label:<22} | {metric:<12} | {'N/A':>10} | {s_val:>9.2f}% | {'N/A':>14} | {'N/A':>10}")
        print("-" * 95)
        s_avg = student_scores["generative_macro_avg"]
        if teacher_scores is not None:
            t_avg = teacher_scores["generative_macro_avg"]
            d_avg = s_avg - t_avg
            r_avg = (s_avg / max(t_avg, 1e-5)) * 100.0
            print(f"{'GENERATIVE MACRO AVG':<22} | {'mean_acc':<12} | {t_avg:>9.2f}% | {s_avg:>9.2f}% | {d_avg:>+13.2f}% | {r_avg:>9.1f}%")
        print("=" * 95)

        with open(args.output_json, "w") as f:
            json.dump({"teacher": teacher_scores, "student": student_scores, "student_ckpt": ckpt_to_load}, f, indent=2)
        print(f"Saved generative benchmark report to: {args.output_json}")

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()