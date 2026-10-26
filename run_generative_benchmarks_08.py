# /home/martinkb/Desktop/BareTorch_Layerwise_Distill/run_generative_benchmarks_08.py
import os
import re
import ast
import json
import math
import time
import random
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
        description="5-Task Multi-GPU Generative Benchmark Suite: GSM8K, MATH-500, GPQA-Diamond, MBPP & HumanEval"
    )
    parser.add_argument("--teacher_id", type=str, default="Qwen/Qwen3.5-0.8B", help="Teacher HF ID")
    parser.add_argument("--student_base_id", type=str, default="Qwen/Qwen3.5-0.8B", help="Base student architecture ID")
    parser.add_argument(
        "--student_ckpt",
        type=str,
        default="./checkpoints/stage4b_dpo_2B_4k/best",
        help="Path to Student checkpoint directory",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="gsm8k,math500,gpqa,mbpp,humaneval",
        help="Comma-separated list of generative benchmarks to evaluate",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batched generation size per GPU")
    parser.add_argument("--max_new_tokens", type=int, default=384, help="Max generated tokens for GSM8K/MBPP/HumanEval")
    parser.add_argument(
        "--max_new_tokens_reasoning",
        type=int,
        default=512,
        help="Max generated tokens for deeper <think> derivations in MATH-500 & GPQA-Diamond",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=250,
        help="Max problems per benchmark (-1 = full test splits; 250 = fast benchmark)",
    )
    parser.add_argument("--skip_teacher", action="store_true", help="Skip Teacher evaluation if already scored")
    parser.add_argument(
        "--teacher_baseline_json",
        type=str,
        default="",
        help="Optional path to previous baseline JSON to auto-load Teacher scores when --skip_teacher is set",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="./generative_eval_stage4b_baseline.json",
        help="Output JSON file for results",
    )
    return parser.parse_args()


# ==============================================================================
# 1. Verifiers: GSM8K, MATH-500 (LaTeX/Symbolic), GPQA-Diamond, MBPP & HumanEval
# ==============================================================================
def strip_qwen_special_tokens(text: str) -> str:
    """Removes `<|...|>` control tokens while preserving `<think>` and `</think>`."""
    return re.sub(r"<\|[^|>]+?\|>", "", text)


def truncate_stop_sequences(text: str) -> str:
    text = strip_qwen_special_tokens(text)
    for stop_seq in ("\nQuestion:", "\nProblem:", "\nQ:", "\n\nQuestion", "\n\nProblem"):
        if stop_seq in text:
            text = text.split(stop_seq, 1)[0]
    return text


def extract_last_boxed(text: str):
    """Extracts content inside the last \\boxed{...} supporting nested braces."""
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    i = idx + len("\\boxed{")
    depth = 1
    out = []
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(text[i])
        i += 1
    return "".join(out).strip() if depth == 0 else None


def extract_gsm8k_answer(text: str):
    text = truncate_stop_sequences(text)
    if "####" in text:
        after = text.split("####")[-1].strip()
        m = NUM_REGEX.findall(after)
        if m:
            return m[0].replace(",", "")
    if "</think>" in text:
        after_t = text.split("</think>")[-1].strip()
        m = NUM_REGEX.findall(after_t)
        if m:
            return m[-1].replace(",", "")
    boxed = extract_last_boxed(text)
    if boxed:
        m = NUM_REGEX.findall(boxed)
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


def normalize_math_str(s: str) -> str:
    if not s:
        return ""
    s = strip_qwen_special_tokens(s).strip()
    for cmd in ("\\left", "\\right", "\\$", "$", "\\%", "%", "\\!", "\\,", "\\;", "\\:", " "):
        s = s.replace(cmd, "")
    s = re.sub(r"\\(?:text|mathrm|textbf|mathbf)\{([^}]*)\}", r"\1", s)
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = re.sub(r"\\frac(\d)(\d)", r"\\frac{\1}{\2}", s)
    s = re.sub(r"\\sqrt(\d)", r"\\sqrt{\1}", s)
    s = s.rstrip(".")
    return s.lower()


def parse_math_numeric_value(s: str):
    """Safely converts integers, decimals, LaTeX fractions, \\sqrt{a}, and \\pi into float values."""
    s_clean = normalize_math_str(s).replace(",", "")
    if not s_clean:
        return None
    try:
        return float(s_clean)
    except Exception:
        pass

    expr = s_clean
    expr = re.sub(r"\\sqrt\{([-+]?\d+(?:\.\d+)?)\}", lambda m: str(math.sqrt(max(0.0, float(m.group(1))))), expr)
    expr = expr.replace("\\pi", str(math.pi))

    m_frac = re.fullmatch(r"(-?)\\frac\{([-+]?\d+(?:\.\d+)?)\}\{([-+]?\d+(?:\.\d+)?)\}", expr)
    if m_frac:
        sign = -1.0 if m_frac.group(1) == "-" else 1.0
        num, den = float(m_frac.group(2)), float(m_frac.group(3))
        if abs(den) > 1e-12:
            return sign * (num / den)

    m_slash = re.fullmatch(r"([-+]?\d+(?:\.\d+)?)/([-+]?\d+(?:\.\d+)?)", expr)
    if m_slash:
        num, den = float(m_slash.group(1)), float(m_slash.group(2))
        if abs(den) > 1e-12:
            return num / den
    return None


def extract_math500_answer(text: str):
    text = truncate_stop_sequences(text)
    boxed = extract_last_boxed(text)
    if boxed:
        return boxed
    if "####" in text:
        after = text.split("####")[-1].strip().splitlines()
        if after and after[0].strip():
            return after[0].strip()
    if "</think>" in text:
        after_t = text.split("</think>")[-1].strip()
        lines = [ln.strip() for ln in after_t.splitlines() if ln.strip()]
        if lines:
            return lines[-1]
    ans_m = re.findall(r"(?:answer is|equals)\s*:?\s*([^\n\.]+)", text, flags=re.IGNORECASE)
    if ans_m:
        return ans_m[-1].strip()
    return None


def check_math500_correct(gen_text: str, gold_target: str) -> bool:
    pred = extract_math500_answer(gen_text)
    if pred is None:
        return False
    if normalize_math_str(pred) == normalize_math_str(gold_target):
        return True
    v_pred = parse_math_numeric_value(pred)
    v_gold = parse_math_numeric_value(gold_target)
    if v_pred is not None and v_gold is not None:
        return math.isclose(v_pred, v_gold, rel_tol=1e-4, abs_tol=1e-4)
    return False


def extract_gpqa_choice(text: str):
    text = truncate_stop_sequences(text)
    if "####" in text:
        after = text.split("####")[-1].strip()
        m = re.search(r"\b([A-D])\b", after.upper())
        if m:
            return m.group(1)
    boxed = extract_last_boxed(text)
    if boxed:
        m = re.search(r"\b([A-D])\b", boxed.upper())
        if m:
            return m.group(1)
    if "</think>" in text:
        after_t = text.split("</think>")[-1].strip()
        m = re.search(r"\b([A-D])\b", after_t.upper())
        if m:
            return m.group(1)
    m_ans = re.findall(r"(?:answer is|choice is|option is|answer:)\s*[\(\[]?([A-D])[\)\]]?", text, flags=re.IGNORECASE)
    if m_ans:
        return m_ans[-1].upper()
    m_any = re.findall(r"\b([A-D])\b", text[-40:])
    if m_any:
        return m_any[-1].upper()
    return None


def check_gpqa_correct(gen_text: str, gold_letter: str) -> bool:
    pred = extract_gpqa_choice(gen_text)
    return (pred == gold_letter.strip().upper()) if pred else False


def extract_python_code(text: str, func_sig_hint: str = "") -> str:
    text = strip_qwen_special_tokens(text)
    if "</think>" in text:
        text = text.split("</think>", 1)[-1]
    if "```python" in text:
        code = text.split("```python", 1)[1].split("```")[0].strip()
    elif "```" in text:
        # CRITICAL FIX: Since prompt already ends with ```python\n, the first ``` in generated text is the CLOSING fence!
        code = text.split("```", 1)[0].strip()
    else:
        for stop_tok in ("\n# Test", "\nassert ", "\nif __name__", "\nYou are", "\nQuestion:"):
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


def check_humaneval_one(args_tuple) -> tuple[bool, bool]:
    """Returns (passed_humaneval_check: bool, valid_python_syntax: bool)."""
    gen_text, prompt_code, test_code, entry_point = args_tuple
    gen_text = strip_qwen_special_tokens(gen_text)
    if "</think>" in gen_text:
        gen_text = gen_text.split("</think>", 1)[-1]

    if "```python" in gen_text:
        extracted = gen_text.split("```python", 1)[1].split("```")[0]
    elif "```" in gen_text:
        # CRITICAL FIX: Since prompt already ends with ```python\n, the first ``` in generated text is the CLOSING fence!
        extracted = gen_text.split("```", 1)[0]
    else:
        extracted = gen_text
        for stop_tok in ("\ndef check(", "\nif __name__", "\nassert ", "\n# Test", "\n```"):
            if stop_tok in extracted:
                extracted = extracted.split(stop_tok, 1)[0]

    if f"def {entry_point}" in extracted:
        full_code = prompt_code.split(f"def {entry_point}")[0] + extracted.strip()
    else:
        lines = [ln for ln in extracted.splitlines() if ln.strip() != ""]
        if lines and not lines[0].startswith((" ", "\t")):
            extracted = "\n".join("    " + ln if ln.strip() else ln for ln in extracted.splitlines())
        full_code = prompt_code.rstrip() + "\n" + extracted.rstrip() + "\n"

    try:
        ast.parse(full_code)
    except SyntaxError:
        return False, False

    test_script = (
        "from typing import *\nimport math\nimport collections\nimport itertools\nimport heapq\nimport functools\n\n"
        f"{full_code}\n\n{test_code}\ncheck({entry_point})\nprint('ALL_HUMANEVAL_TESTS_PASSED')\n"
    )
    try:
        res = subprocess.run(
            ["python3", "-I", "-c", test_script],
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        return ("ALL_HUMANEVAL_TESTS_PASSED" in res.stdout), True
    except Exception:
        return False, True


# ==============================================================================
# 2. Load Official Test Splits for All 5 Generative Benchmarks
# ==============================================================================
GSM8K_THINK_2SHOT = (
    "Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. "
    "How many clips did Natalia sell altogether in April and May?\n"
    "Answer: <think>\n"
    "Natalia sold 48/2 = 24 clips in May.\n"
    "Natalia sold 48+24 = 72 clips altogether in April and May.\n"
    "</think>\n"
    "#### 72\n\n"
    "Question: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. "
    "How much did she earn?\n"
    "Answer: <think>\n"
    "Weng earns 12/60 = $0.2 per minute.\n"
    "Working 50 minutes, she earned 0.2 x 50 = $10.\n"
    "</think>\n"
    "#### 10\n\n"
)

MATH500_THINK_2SHOT = (
    "Problem: What is the positive difference between $120\\%$ of 30 and $130\\%$ of 20?\n"
    "Solution: <think>\n"
    "120% of 30 is 1.2 * 30 = 36.\n"
    "130% of 20 is 1.3 * 20 = 26.\n"
    "The positive difference is 36 - 26 = 10.\n"
    "</think>\n"
    "#### 10\n\n"
    "Problem: Solve for $x$: $3x - 5 = 16$.\n"
    "Solution: <think>\n"
    "Add 5 to both sides: 3x = 21.\n"
    "Divide by 3: x = 7.\n"
    "</think>\n"
    "#### 7\n\n"
)

GPQA_THINK_2SHOT = (
    "Question: Two stars A and B have the same luminosity, but star A has twice the surface temperature of star B. "
    "What is the ratio of the radius of star A to the radius of star B?\n"
    "A. 1/4\nB. 1/2\nC. 2\nD. 4\n"
    "Answer: <think>\n"
    "By the Stefan-Boltzmann law, Luminosity L = 4 * pi * R^2 * sigma * T^4.\n"
    "Since L_A = L_B and T_A = 2 * T_B, we have R_A^2 * (2 * T_B)^4 = R_B^2 * T_B^4.\n"
    "Thus 16 * R_A^2 = R_B^2, which gives R_A / R_B = 1/4 (Option A).\n"
    "</think>\n"
    "#### A\n\n"
)


def load_gsm8k_test(max_samples=-1):
    ds = load_dataset("openai/gsm8k", "main", split="test")
    items = []
    for idx, row in enumerate(ds):
        if 0 < max_samples <= idx:
            break
        q = row["question"].strip()
        ans = row["answer"].strip()
        gold_num = ans.split("####")[-1].strip().replace(",", "")
        prompt = f"{GSM8K_THINK_2SHOT}Question: {q}\nAnswer: <think>\n"
        items.append({"prompt": prompt, "gold_target": gold_num})
    return items


def load_math500_test(max_samples=-1):
    try:
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    except Exception:
        ds = load_dataset("lighteval/MATH", "all", split="test")
    items = []
    for idx, row in enumerate(ds):
        if 0 < max_samples <= idx:
            break
        prob = (row.get("problem", "") or row.get("question", "")).strip()
        gold = row.get("answer", "")
        if not gold and "solution" in row:
            gold = extract_last_boxed(row["solution"]) or ""
        if prob and gold:
            prompt = f"{MATH500_THINK_2SHOT}Problem: {prob}\nSolution: <think>\n"
            items.append({"prompt": prompt, "gold_target": str(gold).strip()})
    return items


def load_gpqa_test(max_samples=-1):
    items = []
    rng = random.Random(42)
    letters = ["A", "B", "C", "D"]

    loaded_rows = None
    for repo, subset, sp in [
        ("Idavidrein/gpqa", "gpqa_diamond", "train"),
        ("ankner/gpqa", None, "train"),
        ("Wanfq/gpqa", "gpqa_diamond", "train"),
    ]:
        try:
            loaded_rows = load_dataset(repo, subset, split=sp) if subset else load_dataset(repo, split=sp)
            break
        except Exception:
            continue

    if loaded_rows is not None:
        for idx, row in enumerate(loaded_rows):
            if 0 < max_samples <= idx:
                break
            q = (row.get("Question", "") or row.get("question", "")).strip()
            corr = (row.get("Correct Answer", "") or row.get("correct_answer", "")).strip()
            inc1 = (row.get("Incorrect Answer 1", "") or row.get("incorrect_answer_1", "")).strip()
            inc2 = (row.get("Incorrect Answer 2", "") or row.get("incorrect_answer_2", "")).strip()
            inc3 = (row.get("Incorrect Answer 3", "") or row.get("incorrect_answer_3", "")).strip()
            if q and corr and inc1 and inc2 and inc3:
                opts = [corr, inc1, inc2, inc3]
                rng.shuffle(opts)
                gold_let = letters[opts.index(corr)]
                opts_str = "\n".join(f"{letters[i]}. {opts[i]}" for i in range(4))
                prompt = f"{GPQA_THINK_2SHOT}Question: {q}\n{opts_str}\nAnswer: <think>\n"
                items.append({"prompt": prompt, "gold_target": gold_let})
    else:
        mmlu_pro = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
        for row in mmlu_pro:
            if 0 < max_samples <= len(items):
                break
            cat = str(row.get("category", "")).lower()
            if cat in ("physics", "chemistry", "biology", "math", "engineering", "computer science"):
                q = row.get("question", "").strip()
                choices = row.get("options", [])
                ans_let = str(row.get("answer", "")).strip().upper()
                if q and len(choices) >= 4 and ans_let in letters:
                    opts_str = "\n".join(f"{letters[i]}. {choices[i]}" for i in range(4))
                    prompt = f"{GPQA_THINK_2SHOT}Question: {q}\n{opts_str}\nAnswer: <think>\n"
                    items.append({"prompt": prompt, "gold_target": ans_let})
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


def load_humaneval_test(max_samples=-1):
    ds = load_dataset("openai/openai_humaneval", split="test")
    items = []
    for idx, row in enumerate(ds):
        if 0 < max_samples <= idx:
            break
        prompt_code = row["prompt"]
        test_code = row["test"]
        entry_point = row["entry_point"]
        prompt = (
            "Complete the following Python function implementation. Output only valid Python code:\n\n"
            f"```python\n{prompt_code}"
        )
        items.append({
            "prompt": prompt,
            "prompt_code": prompt_code,
            "test_code": test_code,
            "entry_point": entry_point,
        })
    return items


# ==============================================================================
# 3. Batched Multi-GPU Generation Runner Across All Selected Tasks
# ==============================================================================
@torch.no_grad()
def run_batched_generation(model, tokenizer, task_items, max_new_toks, batch_size, accelerator, label_str):
    my_items = task_items[accelerator.process_index :: accelerator.num_processes]
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id
    outputs = []
    t0 = time.time()

    for start in range(0, len(my_items), batch_size):
        batch = my_items[start : start + batch_size]
        tok_lists = [tokenizer(x["prompt"], add_special_tokens=False)["input_ids"] for x in batch]
        max_p = max(len(t) for t in tok_lists)
        padded_ids = [[pad_id] * (max_p - len(t)) + t for t in tok_lists]
        attn_mask = [[0] * (max_p - len(t)) + [1] * len(t) for t in tok_lists]

        inp_t = torch.tensor(padded_ids, dtype=torch.long, device=accelerator.device)
        mask_t = torch.tensor(attn_mask, dtype=torch.long, device=accelerator.device)

        gen_out = model.generate(
            input_ids=inp_t,
            attention_mask=mask_t,
            max_new_tokens=max_new_toks,
            do_sample=False,
            use_cache=True,
            pad_token_id=pad_id,
            eos_token_id=eos_id,
        )
        for i, item in enumerate(batch):
            gen_toks = gen_out[i, max_p:].tolist()
            if eos_id in gen_toks:
                gen_toks = gen_toks[: gen_toks.index(eos_id)]
            gen_txt = tokenizer.decode(gen_toks, skip_special_tokens=False)
            outputs.append((item, gen_txt))

        if accelerator.is_main_process and ((start // batch_size) % 2 == 0):
            done = min(len(my_items), start + batch_size) * accelerator.num_processes
            print(f"  [{label_str}] Evaluated {min(done, len(task_items))}/{len(task_items)} problems ({time.time()-t0:.1f}s)...")

    return outputs


@torch.no_grad()
def evaluate_model_on_generative_suite(model, tokenizer, suites, args, accelerator, model_label="Model"):
    model.eval()
    scores = {}
    macro_components = []

    # 1. GSM8K
    if "gsm8k" in suites and suites["gsm8k"]:
        outs = run_batched_generation(
            model, tokenizer, suites["gsm8k"], args.max_new_tokens, args.batch_size, accelerator, f"{model_label} | GSM8K"
        )
        corr = sum(1 for item, txt in outs if check_gsm8k_correct(txt, item["gold_target"]))
        t_stat = accelerator.reduce(torch.tensor([corr, len(outs)], dtype=torch.float64, device=accelerator.device), reduction="sum").tolist()
        acc = (t_stat[0] / max(t_stat[1], 1.0)) * 100.0
        scores["gsm8k_cot"] = acc
        macro_components.append(acc)

    # 2. MATH-500
    if "math500" in suites and suites["math500"]:
        outs = run_batched_generation(
            model, tokenizer, suites["math500"], args.max_new_tokens_reasoning, args.batch_size, accelerator, f"{model_label} | MATH-500"
        )
        corr = sum(1 for item, txt in outs if check_math500_correct(txt, item["gold_target"]))
        t_stat = accelerator.reduce(torch.tensor([corr, len(outs)], dtype=torch.float64, device=accelerator.device), reduction="sum").tolist()
        acc = (t_stat[0] / max(t_stat[1], 1.0)) * 100.0
        scores["math500_cot"] = acc
        macro_components.append(acc)

    # 3. GPQA-Diamond
    if "gpqa" in suites and suites["gpqa"]:
        outs = run_batched_generation(
            model, tokenizer, suites["gpqa"], args.max_new_tokens_reasoning, args.batch_size, accelerator, f"{model_label} | GPQA-Diamond"
        )
        corr = sum(1 for item, txt in outs if check_gpqa_correct(txt, item["gold_target"]))
        t_stat = accelerator.reduce(torch.tensor([corr, len(outs)], dtype=torch.float64, device=accelerator.device), reduction="sum").tolist()
        acc = (t_stat[0] / max(t_stat[1], 1.0)) * 100.0
        scores["gpqa_diamond"] = acc
        macro_components.append(acc)

    # 4. MBPP
    if "mbpp" in suites and suites["mbpp"]:
        outs = run_batched_generation(
            model, tokenizer, suites["mbpp"], args.max_new_tokens, args.batch_size, accelerator, f"{model_label} | MBPP"
        )
        eval_tuples = [(txt, item["test_list"], item["func_sig"]) for item, txt in outs]
        with ThreadPoolExecutor(max_workers=8) as ex:
            res_list = list(ex.map(check_mbpp_one, eval_tuples))
        passed = sum(1 for p, _ in res_list if p)
        syntax = sum(1 for _, s in res_list if s)
        t_stat = accelerator.reduce(torch.tensor([passed, syntax, len(outs)], dtype=torch.float64, device=accelerator.device), reduction="sum").tolist()
        pass1 = (t_stat[0] / max(t_stat[2], 1.0)) * 100.0
        syn_rate = (t_stat[1] / max(t_stat[2], 1.0)) * 100.0
        scores["mbpp_pass@1"] = pass1
        scores["mbpp_syntax_valid"] = syn_rate
        macro_components.append(pass1)

    # 5. HumanEval
    if "humaneval" in suites and suites["humaneval"]:
        outs = run_batched_generation(
            model, tokenizer, suites["humaneval"], args.max_new_tokens, args.batch_size, accelerator, f"{model_label} | HumanEval"
        )
        eval_tuples = [(txt, item["prompt_code"], item["test_code"], item["entry_point"]) for item, txt in outs]
        with ThreadPoolExecutor(max_workers=8) as ex:
            res_list = list(ex.map(check_humaneval_one, eval_tuples))
        passed = sum(1 for p, _ in res_list if p)
        syntax = sum(1 for _, s in res_list if s)
        t_stat = accelerator.reduce(torch.tensor([passed, syntax, len(outs)], dtype=torch.float64, device=accelerator.device), reduction="sum").tolist()
        pass1 = (t_stat[0] / max(t_stat[2], 1.0)) * 100.0
        syn_rate = (t_stat[1] / max(t_stat[2], 1.0)) * 100.0
        scores["humaneval_pass@1"] = pass1
        scores["humaneval_syntax_valid"] = syn_rate
        macro_components.append(pass1)

    scores["generative_macro_avg"] = sum(macro_components) / max(len(macro_components), 1)
    return scores


def main():
    args = parse_args()
    accelerator = Accelerator(mixed_precision="bf16")

    ckpt_to_load = args.student_ckpt
    if not os.path.exists(ckpt_to_load) and os.path.exists(os.path.dirname(ckpt_to_load)):
        ckpt_to_load = os.path.dirname(ckpt_to_load)

    tokenizer = AutoTokenizer.from_pretrained(ckpt_to_load, trust_remote_code=True)
    task_list = [t.strip().lower() for t in args.tasks.split(",") if t.strip()]

    suites = {}
    if "gsm8k" in task_list:
        suites["gsm8k"] = load_gsm8k_test(args.max_samples)
    if "math500" in task_list:
        suites["math500"] = load_math500_test(args.max_samples)
    if "gpqa" in task_list:
        suites["gpqa"] = load_gpqa_test(args.max_samples)
    if "mbpp" in task_list:
        suites["mbpp"] = load_mbpp_test(args.max_samples)
    if "humaneval" in task_list:
        suites["humaneval"] = load_humaneval_test(args.max_samples)

    if accelerator.is_main_process:
        counts_str = " | ".join(f"{k.upper()}: {len(v)}" for k, v in suites.items())
        print("=" * 98)
        print("5-TASK GENERATIVE REASONING & CODING SUITE (GSM8K, MATH-500, GPQA-Diamond, MBPP, HumanEval)")
        print(f"  - Teacher Model : {args.teacher_id}")
        print(f"  - Student Model : {ckpt_to_load}")
        print(f"  - Test Problems : {counts_str}")
        print(f"  - Parallel GPUs : {accelerator.num_processes} GPUs (batch_size={args.batch_size}/GPU, max_tokens={args.max_new_tokens}/{args.max_new_tokens_reasoning})")
        print("=" * 98)

    # 1. Evaluate or Load Cached Teacher Scores
    teacher_scores = None
    if not args.skip_teacher:
        if accelerator.is_main_process:
            print(f"\n[1/2] Evaluating Teacher ({args.teacher_id}) across {len(suites)} generative benchmarks...")
        teacher, _, _ = load_qwen3_5_text_teacher(args.teacher_id, dtype=torch.bfloat16, device="cpu")
        teacher.to(accelerator.device)
        teacher.eval()
        teacher_scores = evaluate_model_on_generative_suite(
            teacher, tokenizer, suites, args, accelerator, model_label="Teacher"
        )
        del teacher
        torch.cuda.empty_cache()
    elif args.teacher_baseline_json and os.path.exists(args.teacher_baseline_json):
        try:
            with open(args.teacher_baseline_json, "r") as f:
                teacher_scores = json.load(f).get("teacher", None)
        except Exception:
            teacher_scores = None

    # 2. Evaluate BareTorch Hybrid Student
    if accelerator.is_main_process:
        print(f"\n[2/2] Evaluating BareTorch Student ({ckpt_to_load}) across {len(suites)} generative benchmarks...")
    _, base_backbone, _ = load_qwen3_5_text_teacher(args.student_base_id, dtype=torch.bfloat16, device="cpu")
    student = load_hybrid_student_checkpoint(
        ckpt_to_load, teacher_backbone=base_backbone, teacher_id=args.student_base_id, dtype=torch.bfloat16
    )
    del base_backbone
    student.to(accelerator.device)
    student.eval()

    student_scores = evaluate_model_on_generative_suite(
        student, tokenizer, suites, args, accelerator, model_label="Student"
    )

    if accelerator.is_main_process:
        print("\n" + "=" * 98)
        print(f"{'Generative Benchmark':<24} | {'Metric':<12} | {'Teacher':>10} | {'Student':>10} | {'Delta (S - T)':>14} | {'Retention':>10}")
        print("-" * 98)
        row_defs = [
            ("gsm8k_cot (Math)", "exact_match", "gsm8k_cot"),
            ("math500 (Comp Math)", "boxed_match", "math500_cot"),
            ("gpqa_diamond (PhD STEM)", "mc_think", "gpqa_diamond"),
            ("mbpp (Python Code)", "pass@1", "mbpp_pass@1"),
            ("humaneval (Python Code)", "pass@1", "humaneval_pass@1"),
            ("mbpp_syntax", "ast_valid", "mbpp_syntax_valid"),
            ("humaneval_syntax", "ast_valid", "humaneval_syntax_valid"),
        ]
        for label, metric, key in row_defs:
            if key not in student_scores:
                continue
            s_val = student_scores[key]
            if teacher_scores is not None and key in teacher_scores:
                t_val = teacher_scores[key]
                delta = s_val - t_val
                ret = (s_val / max(t_val, 1e-5)) * 100.0
                print(f"{label:<24} | {metric:<12} | {t_val:>9.2f}% | {s_val:>9.2f}% | {delta:>+13.2f}% | {ret:>9.1f}%")
            else:
                print(f"{label:<24} | {metric:<12} | {'N/A':>10} | {s_val:>9.2f}% | {'N/A':>14} | {'N/A':>10}")
        print("-" * 98)
        s_avg = student_scores["generative_macro_avg"]
        if teacher_scores is not None and "generative_macro_avg" in teacher_scores:
            t_avg = teacher_scores["generative_macro_avg"]
            d_avg = s_avg - t_avg
            r_avg = (s_avg / max(t_avg, 1e-5)) * 100.0
            print(f"{'5-TASK GENERATIVE AVG':<24} | {'macro_avg':<12} | {t_avg:>9.2f}% | {s_avg:>9.2f}% | {d_avg:>+13.2f}% | {r_avg:>9.1f}%")
        else:
            print(f"{'5-TASK GENERATIVE AVG':<24} | {'macro_avg':<12} | {'N/A':>10} | {s_avg:>9.2f}% | {'N/A':>14} | {'N/A':>10}")
        print("=" * 98)

        with open(args.output_json, "w") as f:
            json.dump({"teacher": teacher_scores, "student": student_scores, "student_ckpt": ckpt_to_load}, f, indent=2)
        print(f"Saved 5-task generative benchmark report to: {args.output_json}")

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()