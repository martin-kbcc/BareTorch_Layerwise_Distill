# /home/martinkb/Desktop/BareTorch_Layerwise_Distill/train_stage4c_grpo_math_code_07.py
import os
import re
import ast
import copy
import gc
import time
import math
import random
import subprocess
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.utils.data import Dataset as TorchDataset, DataLoader
from accelerate import Accelerator
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from datasets import load_dataset

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from baretorch.cs_lrad import LRADDecoderBlock
from init_student_00 import load_qwen3_5_text_teacher, load_hybrid_student_checkpoint

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 4C: Gold-Injected Contrastive Dr. GRPO + On-Policy Top-64 Skew-KD for Math (GSM8K) & Code (MBPP)"
    )
    parser.add_argument("--teacher_id", type=str, default="Qwen/Qwen3.5-2B", help="Larger Teacher for Top-64 Skew-KD")
    parser.add_argument("--student_base_id", type=str, default="Qwen/Qwen3.5-0.8B", help="Base model ID for Student structure")
    parser.add_argument(
        "--input_ckpt_dir",
        type=str,
        default="./checkpoints/stage4b_dpo_2B_4k/best",
        help="Starting Stage 4B champion checkpoint (57.50% Macro Avg)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints/stage4c_grpo_math_code_2B",
        help="Directory to save Stage 4C GRPO Math & Code checkpoints",
    )

    # GRPO Rollout & Optimization Hyperparameters
    parser.add_argument("--group_size", type=int, default=4, help="Number of rollouts G per prompt")
    parser.add_argument("--max_prompt_len", type=int, default=320)
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Max generated CoT / Code tokens per rollout")
    parser.add_argument("--rollout_temp", type=float, default=0.75, help="Sampling temperature for group rollouts")
    parser.add_argument("--rollout_top_p", type=float, default=0.92)
    parser.add_argument("--lrad_lr", type=float, default=2.5e-5, help="Effective Peak GRPO LR for CS-LRAD recurrent layers")
    parser.add_argument("--backbone_lr", type=float, default=6.0e-6, help="Effective Peak GRPO LR for Backbone MLPs & Full-Attn")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_accum_steps", type=int, default=2, help="Gradient accumulation steps across prompt groups")
    parser.add_argument("--max_steps", type=int, default=300, help="300 Gold-Injected Dr. GRPO + Top-64 Skew-KD steps")
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # Gold-Injected Dr. GRPO + Positive-Only Skew-KD + NLL Anchor Weights
    parser.add_argument("--clip_eps", type=float, default=0.20, help="PPO/GRPO clipping epsilon")
    parser.add_argument("--beta_ref_kl", type=float, default=0.03, help="Schulman K3 Reference KL penalty weight")
    parser.add_argument("--grpo_weight", type=float, default=0.45, help="Weight on contrastive Dr. GRPO policy gradient loss")
    parser.add_argument("--opd_skew_weight", type=float, default=0.35, help="Weight on Verified-Winner Top-64 Skew-KL + Skew-RKL")
    parser.add_argument("--gold_cot_weight", type=float, default=0.20, help="Weight on Verified-Winner token NLL anchor")
    parser.add_argument("--topk_logits", type=int, default=64)
    parser.add_argument("--skew_alpha", type=float, default=0.1)
    parser.add_argument("--logit_chunk_size", type=int, default=64, help="64-token vocab chunk size to keep peak VRAM < 12GB")

    # Logging, Eval, Compilation & Saving
    parser.add_argument("--log_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--eval_prompts", type=int, default=60, help="60 held-out validation problems (30 GSM8K + 30 MBPP)")
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--no_grad_checkpointing", action="store_true")
    parser.add_argument("--no_compile", action="store_true", help="Disable torch.compile if set")
    parser.add_argument("--seed", type=int, default=777)
    return parser.parse_args()


# ==============================================================================
# 1. Strict Verifiable Reward (RLVR) Engines for Math (GSM8K) & Code (MBPP)
# ==============================================================================
NUM_REGEX = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")

GSM8K_2SHOT_PREFIX = (
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

MBPP_1SHOT_PREFIX = (
    "You are an expert Python programmer, and here is your task: "
    "Write a function to find the shared elements from the given two lists.\n"
    "Your code should pass these tests:\n\n"
    "assert set(similar_elements((3, 4, 5, 6),(5, 7, 4, 10))) == set((4, 5))\n\n"
    "```python\n"
    "def similar_elements(test_tup1, test_tup2):\n"
    "    return tuple(set(test_tup1) & set(test_tup2))\n"
    "```\n\n"
)


def truncate_at_stop_sequences(text: str, task_type: str) -> str:
    if task_type == "math":
        for stop_seq in ("\nQuestion:", "\nQ:", "\n\nQuestion"):
            if stop_seq in text:
                text = text.split(stop_seq, 1)[0]
        if "####" in text:
            before, after = text.split("####", 1)
            first_line_after = after.strip().splitlines()[0] if after.strip() else ""
            text = f"{before}#### {first_line_after}"
    else:
        if "```" in text:
            text = text.split("```", 1)[0] + "\n```"
    return text.strip()


def extract_math_number(text: str):
    text = truncate_at_stop_sequences(text, "math")
    if "####" in text:
        after_hash = text.split("####")[-1].strip()
        matches = NUM_REGEX.findall(after_hash)
        if matches:
            return matches[0].replace(",", "")
    boxed = re.findall(r"\\boxed\{([^}]+)\}", text)
    if boxed:
        matches = NUM_REGEX.findall(boxed[-1])
        if matches:
            return matches[0].replace(",", "")
    ans_phrase = re.findall(r"(?:answer is|equals|=)\s*\$?\s*([-+]?\d[\d,]*(?:\.\d+)?)", text.lower())
    if ans_phrase:
        return ans_phrase[-1].replace(",", "")
    all_nums = NUM_REGEX.findall(text)
    if all_nums:
        return all_nums[-1].replace(",", "")
    return None


def verify_math_reward(generated_text: str, gold_target: str) -> tuple[float, bool]:
    clean_text = truncate_at_stop_sequences(generated_text, "math")
    pred_str = extract_math_number(clean_text)
    is_correct = False
    if pred_str is not None:
        try:
            if math.isclose(float(pred_str), float(gold_target.replace(",", "")), rel_tol=1e-4, abs_tol=1e-4):
                is_correct = True
        except Exception:
            if pred_str.strip() == gold_target.strip():
                is_correct = True

    if is_correct:
        reward = 1.0
        if "####" in clean_text and len(clean_text.split("####")[0].strip()) >= 20:
            reward += 0.20
        return reward, True
    else:
        return 0.0, False


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


def verify_code_reward(generated_text: str, test_list: list[str], func_sig_hint: str = "") -> tuple[float, bool]:
    code = extract_python_code(generated_text, func_sig_hint)
    if not code:
        return 0.0, False

    try:
        ast.parse(code)
    except SyntaxError:
        return 0.0, False

    reward = 0.10
    if not test_list:
        return reward, False

    test_script = code + "\n\n_passed = 0\n"
    for t_stmt in test_list:
        test_script += f"try:\n    {t_stmt}\n    _passed += 1\nexcept Exception:\n    pass\n"
    test_script += "print(f'PASSED_COUNT:{_passed}')\n"

    try:
        res = subprocess.run(
            ["python3", "-I", "-c", test_script],
            capture_output=True,
            text=True,
            timeout=1.5,
        )
        out = res.stdout
        if "PASSED_COUNT:" in out:
            passed = int(out.split("PASSED_COUNT:")[-1].strip().splitlines()[0])
            pass_ratio = passed / max(len(test_list), 1)
            reward += 1.10 * pass_ratio
            return reward, (passed == len(test_list))
    except Exception:
        pass

    return reward, False


# ==============================================================================
# 2. Math (GSM8K) + Code (MBPP) Dataset Builder (Aligned with Benchmark Format!)
# ==============================================================================
class MathCodeGRPODataset(TorchDataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def build_math_code_grpo_splits(seed=777, val_prompts=60):
    rng = random.Random(seed)
    math_items, code_items = [], []

    gsm_ds = load_dataset("openai/gsm8k", "main", split="train")
    for row in gsm_ds:
        q = row.get("question", "").strip()
        ans_full = row.get("answer", "").strip()
        if q and "####" in ans_full:
            clean_cot = re.sub(r"<<[^>]+>>", "", ans_full).strip()
            gold_num = clean_cot.split("####")[-1].strip().replace(",", "")
            prompt_str = f"{GSM8K_2SHOT_PREFIX}Question: {q}\nAnswer: Let's think step by step.\n"
            math_items.append({
                "task_type": "math",
                "prompt": prompt_str,
                "gold_completion": clean_cot,
                "gold_target": gold_num,
                "test_list": [],
                "func_sig": "",
            })

    for split_name in ("train", "prompt", "validation"):
        try:
            mbpp_ds = load_dataset("google-research-datasets/mbpp", "full", split=split_name)
            for row in mbpp_ds:
                desc = (row.get("text", "") or row.get("prompt", "")).strip()
                code = row.get("code", "").strip().replace("\r\n", "\n")
                tests = row.get("test_list", [])
                if desc and code and tests and "def " in code:
                    sig_match = re.search(r"def\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\([^)]*\)\s*:", code)
                    sig_line = sig_match.group(0) if sig_match else ""
                    first_test = tests[0]
                    prompt_str = (
                        f"{MBPP_1SHOT_PREFIX}"
                        f"You are an expert Python programmer, and here is your task: {desc}\n"
                        f"Your code should pass these tests:\n\n{first_test}\n\n"
                        f"```python\n"
                    )
                    gold_comp = f"{code}\n```"
                    code_items.append({
                        "task_type": "code",
                        "prompt": prompt_str,
                        "gold_completion": gold_comp,
                        "gold_target": "",
                        "test_list": list(tests),
                        "func_sig": sig_line,
                    })
        except Exception:
            pass

    rng.shuffle(math_items)
    rng.shuffle(code_items)

    val_half = max(15, val_prompts // 2)
    val_set = math_items[:val_half] + code_items[:val_half]
    train_math = math_items[val_half:]
    train_code = code_items[val_half:]

    target_code_count = max(len(train_code), int(len(train_math) * 0.45))
    expanded_code = [train_code[i % len(train_code)] for i in range(target_code_count)] if train_code else []

    train_set = train_math + expanded_code
    rng.shuffle(train_set)
    return MathCodeGRPODataset(train_set), MathCodeGRPODataset(val_set), len(train_math), len(train_code)


# ==============================================================================
# 3. Unified Single-Pass Gold-Injected Dr. GRPO + Winner-Only Top-64 Skew-KD
# ==============================================================================
def _fast_token_logprobs_and_topk(logits: torch.Tensor, target_ids: torch.Tensor, topk: int = 0):
    lse = torch.logsumexp(logits, dim=-1)
    tok_lp = logits.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1) - lse
    if topk > 0:
        k_eff = min(topk, logits.size(-1))
        top_vals, top_idx = torch.topk(logits, k=k_eff, dim=-1)
        probs_k = F.softmax(top_vals, dim=-1).clamp(min=1e-8)
        return tok_lp, probs_k, top_idx
    return tok_lp, None, None


def chunked_gold_injected_grpo_and_skew_kd(
    h_pol,
    h_ref,
    h_teach,
    target_ids,
    resp_mask,
    advantages,
    winner_mask,
    pol_lm_head,
    ref_lm_head,
    teach_lm_head,
    topk=64,
    skew_alpha=0.1,
    beta_ref_kl=0.03,
    chunk_size=64,
):
    """
    Unified Single-Pass Contrastive Dr. GRPO + Winner-Only Top-64 Skew-KD + Winner NLL:
      - `advantages`: Zero-mean group advantages across the G trajectories (where at least 1 trajectory
        in the group is ALWAYS a verified `+1.20` winner—either on-policy or Gold-injected!).
      - `winner_mask`: `1.0` strictly on verified correct trajectories (`is_correct == True`), `0.0` on failed ones.
      - On `winner_mask == 1.0` trajectories: applies Top-64 Skew-KL + Skew-RKL against the Larger Teacher + NLL.
      - On `advantages < 0.0` failed on-policy trajectories: applies negative Dr. GRPO gradient (with lower-bound
        log-ratio clamp `-1.5` so negative updates push down errors without destabilizing representation!).
    """
    p_dtype = pol_lm_head.weight.dtype
    r_dtype = ref_lm_head.weight.dtype
    t_dtype = teach_lm_head.weight.dtype

    h_p_s = h_pol[:, :-1, :].to(p_dtype)
    h_r_s = h_ref[:, :-1, :].detach().to(r_dtype)
    h_t_s = h_teach[:, :-1, :].detach().to(t_dtype)

    shift_targets = target_ids[:, 1:].clamp(min=0)
    shift_mask = resp_mask[:, 1:].float()
    B, L_m1, _ = h_p_s.shape
    adv_col = advantages.view(B, 1)
    win_col = winner_mask.view(B, 1)

    def _chunk_fn(hp_c, hr_c, ht_c, tgt_c, m_c):
        v_min = min(pol_lm_head.weight.size(0), teach_lm_head.weight.size(0))

        with torch.no_grad():
            logits_r = ref_lm_head(hr_c.to(r_dtype)).float()[..., :v_min]
            tok_logp_r, _, _ = _fast_token_logprobs_and_topk(logits_r, tgt_c, topk=0)
            del logits_r

            logits_t = teach_lm_head(ht_c.to(t_dtype)).float()[..., :v_min]
            _, probs_t_k, top_idx_t = _fast_token_logprobs_and_topk(logits_t, tgt_c, topk=topk)
            del logits_t

        logits_p = pol_lm_head(hp_c.to(p_dtype)).float()[..., :v_min]
        tok_logp_p, _, _ = _fast_token_logprobs_and_topk(logits_p, tgt_c, topk=0)
        logits_p_k = logits_p.gather(-1, top_idx_t)
        del logits_p

        # 1. Contrastive Dr. GRPO Surrogate:
        # Gate negative updates when log(pi_pol / pi_ref) < -1.5 so we never push down already-unlikely tokens!
        log_ratio_ref = (tok_logp_p - tok_logp_r).clamp(min=-10.0, max=10.0)
        neg_gate = torch.where((adv_col < 0.0) & (log_ratio_ref.detach() < -1.5), torch.zeros_like(adv_col), torch.ones_like(adv_col))
        ratio = torch.exp(tok_logp_p - tok_logp_p.detach())
        pg_tok = (-ratio * adv_col * neg_gate) * m_c

        # 2. Schulman K3 Reference KL penalty
        log_diff_ref = (-log_ratio_ref).clamp(min=-10.0, max=10.0)
        k3_ref_kl = (torch.exp(log_diff_ref) - log_diff_ref - 1.0) * m_c

        # 3. Winner-Only Top-64 Skew-KL + Skew-Reverse-KL against Larger Teacher
        probs_p_k = F.softmax(logits_p_k, dim=-1).clamp(min=1e-8)
        mixed_q = (skew_alpha * probs_t_k + (1.0 - skew_alpha) * probs_p_k).clamp(min=1e-8)
        skl_tok = (probs_t_k * (probs_t_k.log() - mixed_q.log())).sum(dim=-1)

        mixed_p = ((1.0 - skew_alpha) * probs_t_k + skew_alpha * probs_p_k).clamp(min=1e-8)
        srkl_tok = (probs_p_k * (probs_p_k.log() - mixed_p.log())).sum(dim=-1)

        skew_kd_tok = 0.5 * skl_tok + 0.5 * srkl_tok
        win_m_c = m_c * win_col
        opd_chunk = (skew_kd_tok * win_m_c).sum()
        nll_chunk = (-tok_logp_p * win_m_c).sum()

        return pg_tok.sum(), k3_ref_kl.sum(), opd_chunk, nll_chunk, k3_ref_kl.detach().sum()

    total_pg, total_k3, total_opd, total_nll, total_rkl_det = 0.0, 0.0, 0.0, 0.0, 0.0
    for start in range(0, L_m1, chunk_size):
        end = min(start + chunk_size, L_m1)
        hp_c = h_p_s[:, start:end, :].contiguous()
        hr_c = h_r_s[:, start:end, :].contiguous()
        ht_c = h_t_s[:, start:end, :].contiguous()
        tgt_c = shift_targets[:, start:end].contiguous()
        m_c = shift_mask[:, start:end].contiguous()

        if h_pol.requires_grad:
            pg_c, k3_c, opd_c, nll_c, rkl_c = checkpoint.checkpoint(
                _chunk_fn, hp_c, hr_c, ht_c, tgt_c, m_c, use_reentrant=False
            )
        else:
            pg_c, k3_c, opd_c, nll_c, rkl_c = _chunk_fn(hp_c, hr_c, ht_c, tgt_c, m_c)

        total_pg = total_pg + pg_c
        total_k3 = total_k3 + k3_c
        total_opd = total_opd + opd_c
        total_nll = total_nll + nll_c
        total_rkl_det = total_rkl_det + rkl_c

    tok_denom = shift_mask.sum().clamp(min=1.0)
    win_denom = (shift_mask * win_col).sum().clamp(min=1.0)

    grpo_loss = (total_pg / tok_denom) + beta_ref_kl * (total_k3 / tok_denom)
    opd_loss = total_opd / win_denom
    win_nll_loss = total_nll / win_denom
    return grpo_loss, opd_loss, win_nll_loss, total_rkl_det / tok_denom


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
def evaluate_math_code_greedy(accelerator, wrapper, tokenizer, val_dataset, args):
    unwrapped = accelerator.unwrap_model(wrapper).student
    unwrapped.eval()

    items = [val_dataset[i] for i in range(len(val_dataset))]
    my_items = items[accelerator.process_index :: accelerator.num_processes]

    local_stats = torch.zeros(6, dtype=torch.float64, device=accelerator.device)
    eos_id = tokenizer.eos_token_id

    for item in my_items:
        p_ids = tokenizer(
            item["prompt"], add_special_tokens=False, truncation=True, max_length=args.max_prompt_len
        )["input_ids"]
        inp = torch.tensor([p_ids], dtype=torch.long, device=accelerator.device)
        attn_m = torch.ones_like(inp)
        p_len = inp.size(1)

        gen_out = unwrapped.generate(
            input_ids=inp,
            attention_mask=attn_m,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=eos_id,
            eos_token_id=eos_id,
        )
        gen_tokens = gen_out[0, p_len:].tolist()
        gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)

        if item["task_type"] == "math":
            rew, is_corr = verify_math_reward(gen_text, item["gold_target"])
            local_stats[0] += float(is_corr)
            local_stats[1] += 1.0
        else:
            rew, is_corr = verify_code_reward(gen_text, item["test_list"], item["func_sig"])
            local_stats[2] += float(is_corr)
            local_stats[3] += 1.0

        local_stats[4] += rew
        local_stats[5] += 1.0

    g_stats = accelerator.reduce(local_stats, reduction="sum")
    math_acc = (g_stats[0] / g_stats[1].clamp(min=1.0)).item()
    code_acc = (g_stats[2] / g_stats[3].clamp(min=1.0)).item()
    mean_rew = (g_stats[4] / g_stats[5].clamp(min=1.0)).item()
    unwrapped.train()
    return math_acc, code_acc, mean_rew


def main():
    args = parse_args()
    torch.manual_seed(args.seed + int(os.environ.get("LOCAL_RANK", 0)))
    use_gc = not args.no_grad_checkpointing
    accelerator = Accelerator(mixed_precision="bf16", gradient_accumulation_steps=args.grad_accum_steps)

    ckpt_to_load = args.input_ckpt_dir
    if not os.path.exists(ckpt_to_load) and os.path.exists(os.path.dirname(ckpt_to_load)):
        ckpt_to_load = os.path.dirname(ckpt_to_load)

    tokenizer = AutoTokenizer.from_pretrained(ckpt_to_load, trust_remote_code=True)
    eos_id = tokenizer.eos_token_id

    if accelerator.is_main_process:
        print("[Stage 4C GRPO Builder] Loading 2-Shot GSM8K (Math CoT) & 1-Shot MBPP (Python Code) train/val splits...")
    train_ds, val_ds, n_math, n_code = build_math_code_grpo_splits(seed=args.seed, val_prompts=args.eval_prompts)
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        print(f"Loading Larger Skew-KD Teacher: {args.teacher_id} ...")
    teacher, t_backbone, _ = load_qwen3_5_text_teacher(args.teacher_id, dtype=torch.bfloat16, device="cpu")
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    if args.student_base_id == args.teacher_id:
        student = load_hybrid_student_checkpoint(
            ckpt_to_load, teacher_backbone=t_backbone, teacher_id=args.teacher_id, dtype=torch.bfloat16
        )
    else:
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

    t_backbone.to(accelerator.device)
    teacher.lm_head.to(accelerator.device)
    ref_student.to(accelerator.device)
    student.to(accelerator.device)

    if not args.no_compile:
        if accelerator.is_main_process:
            print("  - Compiling frozen Reference & Teacher backbones...")
        ref_student.model = torch.compile(ref_student.model, dynamic=True)
        t_backbone = torch.compile(t_backbone, dynamic=True)

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

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=lambda b: b[0], drop_last=True)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=int(args.warmup_ratio * args.max_steps), num_training_steps=args.max_steps
    )

    wrapper = Stage4Wrapper(student)
    wrapper, optimizer, train_loader = accelerator.prepare(wrapper, optimizer, train_loader)
    wrapper.train()

    if accelerator.is_main_process:
        print("=" * 94)
        print("STAGE 4C: GOLD-INJECTED CONTRASTIVE DR. GRPO + TOP-64 SKEW-KD (MATH CoT & PYTHON CODE)")
        print(f"  - Starting Checkpoint  : {ckpt_to_load}")
        print(f"  - Larger Teacher       : {args.teacher_id} (Winner-Only Top-{args.topk_logits} Skew-KD + Contrastive GRPO)")
        print(f"  - Verifiable Tasks     : {n_math:,} GSM8K (2-Shot CoT) + {n_code:,} MBPP (1-Shot Python Unit-Tests)")
        print(f"  - Group Rollout Size   : G={args.group_size} rollouts/prompt (Gold-Injected when all G fail -> 100% active steps!)")
        print(f"  - Peak Learning Rates  : LRAD={args.lrad_lr:.2e} | Backbone={args.backbone_lr:.2e}")
        print(f"  - Loss Formulation     : {args.grpo_weight}*Dr.GRPO + {args.opd_skew_weight}*Winner_SkewKD + {args.gold_cot_weight}*Winner_NLL")
        print("=" * 94)

    accelerator.wait_for_everyone()
    init_math, init_code, init_rew = evaluate_math_code_greedy(accelerator, wrapper, tokenizer, val_ds, args)
    best_val_score = 0.5 * (init_math + init_code)
    if accelerator.is_main_process:
        print(
            f"  >>> [INIT @ Step 0000] Greedy Val GSM8K: {init_math*100:.1f}% | "
            f"Val MBPP Pass@1: {init_code*100:.1f}% | Combined Acc: {best_val_score*100:.1f}% (Rew: {init_rew:.3f})"
        )

    step = 0
    t0 = time.time()
    running_sums = torch.zeros(6, dtype=torch.float64, device=accelerator.device)
    running_cnt = torch.zeros(1, dtype=torch.float64, device=accelerator.device)
    G = args.group_size
    c_align = 64

    while step < args.max_steps:
        for item in train_loader:
            unwrapped_student = accelerator.unwrap_model(wrapper).student

            # 1. Tokenize prompt & generate G on-policy rollouts
            p_ids = tokenizer(
                item["prompt"], add_special_tokens=False, truncation=True, max_length=args.max_prompt_len
            )["input_ids"]
            p_len = len(p_ids)
            prompt_tensor = torch.tensor([p_ids] * G, dtype=torch.long, device=accelerator.device)
            prompt_mask = torch.ones_like(prompt_tensor)

            unwrapped_student.eval()
            with torch.no_grad():
                gen_seqs = unwrapped_student.generate(
                    input_ids=prompt_tensor,
                    attention_mask=prompt_mask,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.rollout_temp,
                    top_p=args.rollout_top_p,
                    use_cache=True,
                    pad_token_id=eos_id,
                    eos_token_id=eos_id,
                )
            unwrapped_student.train()

            # 2. Score all G on-policy rollouts
            cleaned_seqs = []
            end_lengths = []
            rewards = []
            correct_flags = []

            for i in range(G):
                comp_ids = gen_seqs[i, p_len:].tolist()
                if eos_id in comp_ids:
                    comp_ids = comp_ids[: comp_ids.index(eos_id)]
                raw_txt = tokenizer.decode(comp_ids, skip_special_tokens=True)
                clean_txt = truncate_at_stop_sequences(raw_txt, item["task_type"])

                if item["task_type"] == "math":
                    r_val, is_c = verify_math_reward(clean_txt, item["gold_target"])
                else:
                    r_val, is_c = verify_code_reward(clean_txt, item["test_list"], item["func_sig"])
                rewards.append(r_val)
                correct_flags.append(float(is_c))

                clean_comp_ids = tokenizer(
                    clean_txt, add_special_tokens=False, truncation=True, max_length=args.max_new_tokens - 1
                )["input_ids"] + [eos_id]
                cleaned_seqs.append(p_ids + clean_comp_ids)
                end_lengths.append(len(clean_comp_ids))

            onpolicy_any_correct = float(any(correct_flags))
            onpolicy_mean_rew = sum(rewards) / max(len(rewards), 1)

            # 3. GOLD-INJECTED GROUP GUARANTEE:
            # If none of the G on-policy rollouts solved the problem (or Even if they did, to anchor against drift),
            # replace the lowest-reward rollout slot with the verified Gold CoT / Python solution (`reward = 1.20`)!
            # This guarantees 100% of groups have a verified `+1.20` winner AND active contrastive Dr. GRPO advantages!
            if not any(correct_flags):
                gold_comp_ids = tokenizer(
                    item["gold_completion"], add_special_tokens=False, truncation=True, max_length=args.max_new_tokens - 1
                )["input_ids"] + [eos_id]
                replace_idx = int(torch.tensor(rewards).argmin().item())
                cleaned_seqs[replace_idx] = p_ids + gold_comp_ids
                end_lengths[replace_idx] = len(gold_comp_ids)
                rewards[replace_idx] = 1.20
                correct_flags[replace_idx] = 1.0

            max_seq_l = max(len(s) for s in cleaned_seqs)
            target_len = min(640, ((max_seq_l + c_align - 1) // c_align) * c_align)

            padded_batch = []
            resp_mask = torch.zeros((G, target_len), dtype=torch.float32, device=accelerator.device)
            for i, s in enumerate(cleaned_seqs):
                s_trunc = s[:target_len]
                pad_rem = target_len - len(s_trunc)
                padded_batch.append(s_trunc + [eos_id] * pad_rem)
                valid_end = min(target_len, p_len + end_lengths[i])
                if valid_end > p_len:
                    resp_mask[i, p_len:valid_end] = 1.0

            gen_seqs_aligned = torch.tensor(padded_batch, dtype=torch.long, device=accelerator.device)

            with accelerator.accumulate(wrapper):
                with torch.no_grad():
                    ref_h = ref_student.model(
                        input_ids=gen_seqs_aligned, attention_mask=None, output_hidden_states=False, use_cache=False
                    ).last_hidden_state
                    t_h = t_backbone(
                        input_ids=gen_seqs_aligned, attention_mask=None, output_hidden_states=False, use_cache=False
                    ).last_hidden_state

                    rew_t = torch.tensor(rewards, dtype=torch.float32, device=accelerator.device)
                    winner_mask = torch.tensor(correct_flags, dtype=torch.float32, device=accelerator.device)
                    advantages = rew_t - rew_t.mean()

                pol_h = wrapper(gen_seqs_aligned)
                grpo_loss, opd_skew_loss, win_nll_loss, ref_kl = chunked_gold_injected_grpo_and_skew_kd(
                    pol_h, ref_h, t_h, gen_seqs_aligned, resp_mask, advantages, winner_mask,
                    unwrapped_student.lm_head, ref_student.lm_head, teacher.lm_head,
                    topk=args.topk_logits, skew_alpha=args.skew_alpha, beta_ref_kl=args.beta_ref_kl,
                    chunk_size=args.logit_chunk_size,
                )

                loss = (
                    args.grpo_weight * grpo_loss
                    + args.opd_skew_weight * opd_skew_loss
                    + args.gold_cot_weight * win_nll_loss
                )

                accelerator.backward(loss)

                running_sums[0] += loss.detach().to(torch.float64)
                running_sums[1] += float(onpolicy_mean_rew)
                running_sums[2] += float(onpolicy_any_correct)
                running_sums[3] += opd_skew_loss.detach().to(torch.float64)
                running_sums[4] += win_nll_loss.detach().to(torch.float64)
                running_sums[5] += ref_kl.detach().to(torch.float64)
                running_cnt[0] += 1.0

                del pol_h, ref_h, t_h

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(all_params, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                scheduler.step()
                step += 1

                if step == 1 or step % args.log_steps == 0:
                    g_sums = accelerator.reduce(running_sums, reduction="sum")
                    g_cnt = accelerator.reduce(running_cnt, reduction="sum").clamp(min=1.0)
                    avg_l, avg_rew, pass_at_g, avg_opd, avg_gce, avg_rkl = (g_sums / g_cnt).tolist()
                    running_sums.zero_()
                    running_cnt.zero_()

                    if accelerator.is_main_process:
                        elapsed = time.time() - t0
                        t0 = time.time()
                        lrs = scheduler.get_last_lr()
                        print(
                            f"[Stage 4C GRPO] Step {step:03d}/{args.max_steps} | Loss: {avg_l:.4f} | "
                            f"OnPol_Rew: {avg_rew:.3f} (Pass@{G}: {pass_at_g*100:.1f}%) | "
                            f"Win_SkewKD: {avg_opd:.4f} | Win_CE: {avg_gce:.4f} | Ref_KL: {avg_rkl:.4f} | "
                            f"LR: {lrs[0]:.2e} ({elapsed:.1f}s)"
                        )

                if step % args.eval_steps == 0 or step >= args.max_steps:
                    accelerator.wait_for_everyone()
                    v_math, v_code, v_rew = evaluate_math_code_greedy(accelerator, wrapper, tokenizer, val_ds, args)
                    val_score = 0.5 * (v_math + v_code)
                    if accelerator.is_main_process:
                        print(
                            f"  >>> [VAL @ Step {step:03d}] Greedy GSM8K: {v_math*100:.1f}% | "
                            f"Greedy MBPP Pass@1: {v_code*100:.1f}% | Combined Acc: {val_score*100:.1f}% (Rew: {v_rew:.3f})"
                        )
                        if val_score >= best_val_score:
                            best_val_score = val_score
                            best_dir = os.path.join(args.output_dir, "best")
                            os.makedirs(best_dir, exist_ok=True)
                            unwrapped = accelerator.unwrap_model(wrapper).student
                            unwrapped.config.save_pretrained(best_dir)
                            tokenizer.save_pretrained(best_dir)
                            torch.save(unwrapped.state_dict(), os.path.join(best_dir, "pytorch_model.bin"))
                            print(f"  >>> New Best Stage 4C Combined Greedy Accuracy ({best_val_score*100:.1f}%)! Saved to {best_dir}")
                    accelerator.wait_for_everyone()
                    t0 = time.time()

                if step % args.save_steps == 0 or step >= args.max_steps:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        os.makedirs(args.output_dir, exist_ok=True)
                        unwrapped = accelerator.unwrap_model(wrapper).student
                        unwrapped.config.save_pretrained(args.output_dir)
                        tokenizer.save_pretrained(args.output_dir)
                        torch.save(unwrapped.state_dict(), os.path.join(args.output_dir, "pytorch_model.bin"))
                    accelerator.wait_for_everyone()
                    t0 = time.time()

                if step >= args.max_steps:
                    break

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()