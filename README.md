# BareTorch Layerwise Up-Distillation & `<think>` GRPO Engine 🐻🔥

> **Surgical Sub-Quadratic Model Conversion, Larger-Teacher Skew-KD & Verifiable Reinforcement Learning under Model Rampage**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6+-ee4c2c.svg?logo=pytorch)](https://pytorch.org/)
[![Paper PDF](https://img.shields.io/badge/Paper-PDF-red.svg)](https://github.com/martin-kbcc/baretorch-experiments/blob/main/paper.pdf)
[![Production Framework](https://img.shields.io/badge/Core_Engine-BareTorch-black?logo=github)](https://github.com/martin-kbcc/baretorch)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model%20Rampage-yellow)](https://huggingface.co/model-rampage)

Welcome to the official **BareTorch Layerwise Up-Distillation & `<think>` GRPO Engine**. This repository houses our end-to-end multi-stage conversion, distillation, and verifiable reinforcement learning pipeline designed to transform pretrained foundation models (such as `Qwen/Qwen3.5-0.8B`, `2B`, and `4B`) into hardware-portable, **3:1 Interleaved CS-LRAD (Chunked Selective Low-Rank Associative Delta)** hybrid models in hours on an 8x NVIDIA H100 cluster.

Unlike existing sub-quadratic architectures (such as Mamba-2, RWKV, and Gated DeltaNet) that require hardware-locked C++/CUDA/Triton extensions (`causal-conv1d`, `flash-linear-attention`), **BareTorch** formulates sequence mixing using **100% standard matrix multiplications (GEMMs)** and an exact **$O(1)$ low-rank recurrent associative state update** for autoregressive decoding—running out-of-the-box at peak speed across NVIDIA GPUs (`torch.compile`), Google Cloud TPUs (OpenXLA), and Apple Silicon (MLX).

---

## 🚀 Released Hugging Face Models (`model-rampage`)

All models produced by this pipeline are self-contained (`trust_remote_code=True`), require **zero external C++/CUDA packages**, and include a built-in `model.compile_cs_lrad(mode="default", dynamic=True)` helper:

| Model Repository | Hybrid Topology | Base $\rightarrow$ Up-Distill Teacher | Status & Key Highlights |
| :--- | :--- | :--- | :--- |
| **[BareTorch-Qwen3.5-4B-Hybrid](https://huggingface.co/model-rampage/BareTorch-Qwen3.5-4B-Hybrid)** | `24 CS-LRAD (r=32)` + `8 Full-Attn` | `Qwen3.5-4B` $\rightarrow$ `Qwen3.5-9B` | ✅ **Live on HF** — **100.7%** `lm_eval` retention & **103.5%** `<think>` generative retention (**73.8% GSM8K**, **36.2% MBPP**, **25.6% HumanEval**); **1.35x** faster prefill & **1.67 GB less 32K VRAM**. |
| **[BareTorch-Qwen3.5-2B-Hybrid](https://huggingface.co/model-rampage/BareTorch-Qwen3.5-2B-Hybrid)** | `18 CS-LRAD (r=32)` + `6 Full-Attn` | `Qwen3.5-2B` $\rightarrow$ `Qwen3.5-4B` | ✅ **Live on HF** — **105.0%** `lm_eval` retention & **117.7%** `<think>` generative retention vs. native `Qwen3.5-2B`; **1.63x** faster prefill & **2.3x** smaller short-context cache. |
| **[BareTorch-Qwen3.5-0.8B-Hybrid](https://huggingface.co/model-rampage/BareTorch-Qwen3.5-0.8B-Hybrid)** | `18 CS-LRAD (r=16)` + `6 Full-Attn` | `Qwen3.5-0.8B` $\rightarrow$ `Qwen3.5-4B` | 🔥 **Training on 8x H100** — Ultra-light edge hybrid; **1.63x** faster prefill (`82,767 tok/s` at 2K), **3.2x** smaller cache, **31% lower 32K VRAM** (`2.70 GB`). |

---

## 🏗️ Architectural Topology: 3:1 Interleaved CS-LRAD Hybrid

The pipeline converts pretrained `Qwen3.5` backbones into a **3:1 Interleaved CS-LRAD Hybrid Topology**:
- **75% Pure-GEMM CS-LRAD Recurrent Layers (`18/24` on 0.8B/2B, `24/32` on 4B):** Replaces `GatedDeltaNet` token mixers with block-parallel **CS-LRAD** (`chunk_size=64`, low-rank factor $r \in \{16, 32\}$) associative delta blocks. During prefill, intra-chunk and inter-chunk states resolve via standard parallel GEMMs; during decoding, each layer updates a compact recurrent state $S_t \in \mathbb{R}^{B \times H \times r \times d_h}$ in **$O(1)$ time and memory**.
- **25% Spliced Full-Attention Anchor Layers (`{3, 7, 11, ...}`):** Every 4th block retains the intact `Qwen3_5DecoderLayer` Full-Attention mechanism (with QK-Norm and RoPE) to preserve exact long-horizon token retrieval out to `32,768` tokens.

```text
[CS-LRAD] ──> [CS-LRAD] ──> [CS-LRAD] ──> [Qwen3.5 Full-Attention] ──> ... (x6 for 0.8B/2B, x8 for 4B)
  (L0)          (L1)          (L2)                  (L3)
```

---

## 🔄 The 8-Stage Surgical Up-Distillation & `<think>` GRPO Pipeline

Rather than training from scratch on hundreds of billions of tokens, our master pipeline (`run_master_pipeline_8xh100_{0.8B,2B,4B}.sh`) executes a mathematically staged progression that eliminates representation drift before up-distilling reasoning and coding capabilities from a $2\times$–$5\times$ larger teacher:

```text
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                 BARETORCH 8-STAGE SURGICAL UP-DISTILLATION & <think> GRPO ENGINE                 │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
  Stage 0 & 0-Verify (`init_student_00.py`, `verify_init.py`)
    ├── Copies Embeddings, tied LM Head, RMSNorms (+1.0 unit offset), SwiGLU MLPs & Full-Attn layers.
    ├── Warm-starts CS-LRAD projections (W_q, W_k, W_v, W_swish, W_out) from Teacher GatedDeltaNet.
    └── Runs 6-point verification: 0.000000 scaffold diff & 100.0% FP32 cross-chunk cache equivalence.

  Stage 0B (`prepare_dataset_00b.py`)
    └── Streams & packs 600M tokens of FineWeb-Edu (`sample-10BT`) into zero-padded 2,048-token blocks.

  Stage 1 (`train_stage1_mixer_align_01.py`) — Parallel Teacher-Forced Mixer Alignment
    ├── Freezes all Embeddings, MLPs, and Full-Attention layers; trains ONLY CS-LRAD mixers.
    └── Feeds clean Teacher hidden states into each Student CS-LRAD block: L = (1 - cos) + 0.1 * MSE.

  Stage 2 (`train_stage2_end2end_02.py`) — End-to-End Hidden-State & Logit Distillation
    ├── Unfreezes CS-LRAD blocks (mixers, norms, MLPs) while keeping Embeddings & Full-Attn frozen.
    └── Optimizes L = 0.5 * KL(T=1.0) + 0.3 * CE + 0.2 * Layerwise_Hidden_Cosine.

  Stage 3 (`train_stage3_full_kd_03.py`) — Full-Model Differential-LR Knowledge Distillation
    ├── Unfreezes 100% of parameters with 10x differential LRs (LR_lrad = 1.2e-4, LR_backbone = 1.2e-5).
    └── Optimizes 50% Full-Vocab (248,320) Chunked KL + 50% Next-Token CE across FineWeb-Edu.

  Stage 3B (`train_stage3b_long_context_03b.py`) — 32,768-Token (512-Chunk) Horizon Extension
    └── Trains on 32,768-token sequences (512 chunks) with 1.5x focal weight on positions [2048..32768].

  Stage 4A (`train_stage4a_updistill_sft_05.py`) — 5-Pillar Focal SFT (Native Subspace Preservation)
    ├── 373.6M-token 5-Pillar SFT mixture (Math, Code, SmolTalk, Commonsense/SciQ/ARC, MMLU + 15% Replay).
    └── Distills from the native same-scale teacher with 12x MMLU, 4x QA, and 1.5x cross-chunk focal weights.

  Stage 4B (`train_stage4b_dpo_align_06.py`) — 6-Pillar Contrastive DPO + Larger-Teacher Top-64 Skew-KD
    ├── Introduces Larger Teacher (`Qwen3.5-4B` or `9B`) across 208,086 contrastive (chosen, rejected) pairs.
    └── Optimizes L = 0.45 * Teacher_DPO(beta=0.1) + 0.30 * Top64_SkewKD(alpha=0.1) + 0.25 * Focal_NLL.

  Stage 4C (`train_stage4c_grpo_math_code_07.py`) — 4-Pillar <think> Dr. GRPO + Winner-Only Skew-KD
    ├── 1,500–2,000 steps of on-policy Dr. GRPO (G=12–16 rollouts/prompt) on GSM8K, MATH, STEM-MC & MBPP.
    ├── Live sandboxed Python unit-test & exact boxed/numeric reward verifier (+ <think> format bonus).
    └── Applies Larger-Teacher Top-64 Skew-KD exclusively on the highest-reward winning rollout.
```

---

## 📊 Comprehensive Benchmark Results (`2B` & `4B` Hybrids vs. Native `Qwen3.5`)

### 1. Zero-Shot Non-Generative Suite (`lm-evaluation-harness` 10-Task Suite)

Evaluated via `run_lm_eval_benchmarks_04.py` across 10 standard zero-shot benchmarks (including all 57 MMLU subjects). Both `BareTorch-Qwen3.5-2B-Hybrid` and `BareTorch-Qwen3.5-4B-Hybrid` **exceed 100% overall retention** against their native `Qwen3.5` teachers:

| Benchmark Task | Metric | `Qwen3.5-2B` | **BareTorch-2B** | Retention (2B) | `Qwen3.5-4B` | **BareTorch-4B** | Retention (4B) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ARC-Challenge** | `acc_norm` | 41.38% | **53.84%** | **130.1%** | 54.01% | **63.40%** | **117.4%** |
| **ARC-Easy** | `acc_norm` | 66.25% | **81.40%** | **122.9%** | 75.59% | **85.73%** | **113.4%** |
| **BoolQ** | `acc` | 71.80% | **79.88%** | **111.2%** | **86.18%** | 85.90% | 99.7% |
| **HellaSwag** | `acc_norm` | **62.05%** | 58.19% | 93.8% | **73.11%** | 69.87% | 95.6% |
| **LAMBADA (OpenAI)** | `acc` | **53.29%** | 46.81% | 87.8% | **64.70%** | 57.31% | 88.6% |
| **MMLU (57 Subjects)** | `acc` | **59.17%** | 49.50% | 83.7% | **74.37%** | 62.35% | 83.8% |
| **OpenBookQA** | `acc_norm` | 35.60% | **38.40%** | **107.9%** | 40.60% | **42.80%** | **105.4%** |
| **PIQA** | `acc_norm` | 72.85% | **73.83%** | **101.3%** | **78.40%** | 77.69% | 99.1% |
| **SciQ** | `acc_norm` | 88.50% | **97.00%** | **109.6%** | 95.50% | **97.40%** | **102.0%** |
| **WinoGrande** | `acc` | 63.06% | **65.67%** | **104.1%** | 69.46% | **74.74%** | **107.6%** |
| **10-Task Macro Average** | **`mean_acc`** | 61.40% | **64.45%** | **105.0% (+3.05%)** | 71.19% | **71.72%** | **100.7% (+0.53%)** |

---

### 2. Generative Reasoning & Python Coding Suite (5-Task `<think>` & Unit-Test Suite)

Evaluated via `run_generative_benchmarks_08.py` using multi-step autoregressive decoding (up to `512` tokens) and sandboxed Python unit testing:

| Generative Benchmark | Metric | `Qwen3.5-2B` | **BareTorch-2B** | Retention (2B) | `Qwen3.5-4B` | **BareTorch-4B** | Retention (4B) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **GSM8K** (`<think>` CoT Math) | `exact_match` | **65.60%** | 47.40% | 72.3% | **90.20%** | 73.80% | 81.8% |
| **MATH-500** (`<think>` Boxed) | `boxed_match` | **20.60%** | 15.60% | 75.7% | **48.60%** | 33.00% | 67.9% |
| **GPQA-Diamond** (PhD STEM) | `mc_think` | 4.69% | **25.67%** | **547.6%** | 2.68% | **16.52%** | **616.7%** |
| **MBPP** (Python Synthesis) | `pass@1` | 1.60% | **24.20%** | **1512.5%** | 21.00% | **36.20%** | **172.4%** |
| **HumanEval** (Python Synthesis) | `pass@1` | **15.85%** | 14.63% | 92.3% | 16.46% | **25.61%** | **155.6%** |
| *MBPP AST Syntax Validity* | `ast_valid` | 78.20% | **92.00%** | **117.6%** | 85.40% | **97.00%** | **113.6%** |
| *HumanEval AST Syntax Validity* | `ast_valid` | 78.66% | **79.27%** | **100.8%** | 73.17% | **76.22%** | **104.2%** |
| **5-Task Generative Average** | **`macro_avg`** | 21.67% | **25.50%** | **117.7% (+3.83%)** | 35.79% | **37.03%** | **103.5% (+1.24%)** |

---

## ⚡ Multi-Scale Hardware Speed & Memory Benchmarks (NVIDIA RTX 4090 — 512 to 32,768 Context)

Evaluated via `run_hardware_benchmarks_09.py` on an **NVIDIA GeForce RTX 4090 (24GB)**. Using `model.compile_cs_lrad(mode="default", dynamic=True)`, BareTorch's pure-GEMM `CS-LRAD` blocks outperform native `Qwen3.5` across **every model scale (`4B`, `2B`, `0.8B`) and every context length (`512` to `32,768`)**:

### A. `BareTorch-Qwen3.5-4B-Hybrid` (`r=32`, Compiled) vs. Native `Qwen3.5-4B`
| Context Length | Model | TTFT (ms) | Prefill (tok/s) | Decode (tok/s) | KV-Cache (MB) | Peak VRAM (GB) | Speedup / Memory Savings |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **512** | **BareTorch-Qwen3.5-4B-Hybrid (r=32, Compiled)** | **39.8 ms** | **12,877 tok/s** | **37.2 tok/s** | **25.5 MB** | **7.67 GB** | **1.35x Prefill, 1.15x Decode, 2.6x Smaller Cache** |
| 512 | Qwen3.5-4B (Native) | 53.5 ms | 9,568 tok/s | 32.3 tok/s | 67.5 MB | 7.97 GB | Baseline |
| **2,048** | **BareTorch-Qwen3.5-4B-Hybrid (r=32, Compiled)** | **132.5 ms** | **15,461 tok/s** | **36.9 tok/s** | **73.5 MB** | **7.78 GB** | **1.11x Prefill, 1.18x Decode, 1.6x Smaller Cache** |
| 2,048 | Qwen3.5-4B (Native) | 147.0 ms | 13,928 tok/s | 31.4 tok/s | 115.5 MB | 8.20 GB | Baseline |
| **8,192** | **BareTorch-Qwen3.5-4B-Hybrid (r=32, Compiled)** | **579.8 ms** | **14,128 tok/s** | **37.3 tok/s** | **265.5 MB** | **8.47 GB** | **1.13x Prefill, 1.17x Decode, 670 MB Less VRAM** |
| 8,192 | Qwen3.5-4B (Native) | 655.3 ms | 12,502 tok/s | 31.8 tok/s | 307.5 MB | 9.14 GB | Baseline |
| **16,384** | **BareTorch-Qwen3.5-4B-Hybrid (r=32, Compiled)** | **1235.8 ms** | **13,257 tok/s** | **37.0 tok/s** | **521.5 MB** | **9.38 GB** | **1.12x Prefill, 1.12x Decode, 1.01 GB Less VRAM** |
| 16,384 | Qwen3.5-4B (Native) | 1389.9 ms | 11,788 tok/s | 32.9 tok/s | 563.5 MB | 10.39 GB | Baseline |
| **32,768** | **BareTorch-Qwen3.5-4B-Hybrid (r=32, Compiled)** | **2839.9 ms** | **11,538 tok/s** | **37.0 tok/s** | **1033.5 MB** | **11.22 GB** | **1.08x Prefill, 1.12x Decode, 1.67 GB Less VRAM (-13%)** |
| 32,768 | Qwen3.5-4B (Native) | 3071.2 ms | 10,669 tok/s | 33.0 tok/s | 1075.5 MB | 12.89 GB | Baseline |

### B. `BareTorch-Qwen3.5-2B-Hybrid` (`r=32`, Compiled) vs. Native `Qwen3.5-2B`
| Context Length | Model | TTFT (ms) | Prefill (tok/s) | Decode (tok/s) | KV-Cache (MB) | Peak VRAM (GB) | Speedup / Memory Savings |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **512** | **BareTorch-Qwen3.5-2B-Hybrid (r=32, Compiled)** | **24.4 ms** | **20,967 tok/s** | **47.6 tok/s** | **11.2 MB** | **3.65 GB** | **1.63x Prefill, 1.07x Decode, 2.3x Smaller Cache** |
| 512 | Qwen3.5-2B (Native) | 39.9 ms | 12,827 tok/s | 44.5 tok/s | 25.6 MB | 3.57 GB | Baseline |
| **4,096** | **BareTorch-Qwen3.5-2B-Hybrid (r=32, Compiled)** | **119.9 ms** | **34,169 tok/s** | **47.8 tok/s** | **53.2 MB** | **3.84 GB** | **1.03x Prefill, 1.05x Decode, 1.3x Smaller Cache** |
| 4,096 | Qwen3.5-2B (Native) | 123.1 ms | 33,263 tok/s | 45.5 tok/s | 67.6 MB | 3.86 GB | Baseline |
| **16,384** | **BareTorch-Qwen3.5-2B-Hybrid (r=32, Compiled)** | **534.2 ms** | **30,672 tok/s** | **48.1 tok/s** | **197.2 MB** | **4.71 GB** | **1.08x Prefill, 1.09x Decode, 140 MB Less VRAM** |
| 16,384 | Qwen3.5-2B (Native) | 577.0 ms | 28,395 tok/s | 44.2 tok/s | 211.6 MB | 4.85 GB | Baseline |
| **32,768** | **BareTorch-Qwen3.5-2B-Hybrid (r=32, Compiled)** | **1191.9 ms** | **27,493 tok/s** | **46.8 tok/s** | **389.2 MB** | **5.87 GB** | **1.06x Prefill, 1.03x Decode, 310 MB Less VRAM** |
| 32,768 | Qwen3.5-2B (Native) | 1262.6 ms | 25,952 tok/s | 45.3 tok/s | 403.6 MB | 6.18 GB | Baseline |

### C. `BareTorch-Qwen3.5-0.8B-Hybrid` (`r=16`, Compiled) vs. Native `Qwen3.5-0.8B`
| Context Length | Model | TTFT (ms) | Prefill (tok/s) | Decode (tok/s) | KV-Cache (MB) | Peak VRAM (GB) | Speedup / Memory Savings |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **512** | **BareTorch-Qwen3.5-0.8B-Hybrid (r=16, Compiled)** | **23.8 ms** | **21,555 tok/s** | **49.6 tok/s** | **7.9 MB** | **1.34 GB** | **1.63x Prefill, 1.10x Decode, 3.2x Smaller Cache** |
| 512 | Qwen3.5-0.8B (Native) | 38.8 ms | 13,205 tok/s | 44.9 tok/s | 25.6 MB | 1.49 GB | Baseline |
| **2,048** | **BareTorch-Qwen3.5-0.8B-Hybrid (r=16, Compiled)** | **24.7 ms** | **82,767 tok/s** | **49.7 tok/s** | **25.9 MB** | **1.34 GB** | **1.58x Prefill, 1.11x Decode, 1.7x Smaller Cache** |
| 2,048 | Qwen3.5-0.8B (Native) | 39.2 ms | 52,277 tok/s | 44.8 tok/s | 43.6 MB | 1.61 GB | Baseline |
| **8,192** | **BareTorch-Qwen3.5-0.8B-Hybrid (r=16, Compiled)** | **100.4 ms** | **81,586 tok/s** | **50.0 tok/s** | **97.9 MB** | **1.61 GB** | **1.31x Prefill, 1.13x Decode, 22% Lower VRAM** |
| 8,192 | Qwen3.5-0.8B (Native) | 131.9 ms | 62,122 tok/s | 44.3 tok/s | 115.6 MB | 2.07 GB | Baseline |
| **32,768** | **BareTorch-Qwen3.5-0.8B-Hybrid (r=16, Compiled)** | **587.5 ms** | **55,775 tok/s** | **49.9 tok/s** | **385.9 MB** | **2.70 GB** | **1.25x Prefill, 1.11x Decode, 1.21 GB Less VRAM (-31%)** |
| 32,768 | Qwen3.5-0.8B (Native) | 731.9 ms | 44,770 tok/s | 45.1 tok/s | 403.6 MB | 3.91 GB | Baseline |

---

## 📁 Repository Structure

```text
BareTorch_Layerwise_Distill/
├── baretorch/                                # Core BareTorch Hybrid modeling & CS-LRAD engine
│   ├── configuration_baretorch.py            # Unified BareTorchConfig (Hugging Face AutoConfig ready)
│   ├── modeling_baretorch.py                 # BareTorchForCausalLM + compile_cs_lrad() + DynamicCache
│   ├── cs_lrad.py                            # Pure-GEMM CS-LRAD chunk_forward & O(1) step_forward
│   └── transformer.py                        # Full-Attention & KV-cache utilities
├── init_student_00.py                        # Stage 0: Surgical weight splicing & GatedDeltaNet warm-start
├── verify_init.py                            # Stage 0-Verify: 6-point scaffold & FP32 cross-chunk cache verifier
├── prepare_dataset_00b.py                    # Stage 0B: Parallel 48-worker FineWeb-Edu 600M-token packer
├── train_stage1_mixer_align_01.py            # Stage 1: Teacher-forced parallel CS-LRAD mixer alignment
├── train_stage2_end2end_02.py                # Stage 2: End-to-end hidden-state cosine + chunked full-vocab KL
├── train_stage3_full_kd_03.py                # Stage 3: Full-model joint KD with 10x differential learning rates
├── train_stage3b_long_context_03b.py         # Stage 3B: 32,768-token (512-chunk) late-horizon extension
├── run_lm_eval_benchmarks_04.py              # 10-task EleutherAI lm-evaluation-harness comparator
├── train_stage4a_updistill_sft_05.py         # Stage 4A: 5-Pillar Focal SFT with native teacher subspace KL
├── train_stage4b_dpo_align_06.py             # Stage 4B: 6-Pillar Contrastive DPO + Larger-Teacher Top-64 Skew-KD
├── train_stage4c_grpo_math_code_07.py        # Stage 4C: 4-Pillar <think> Dr. GRPO + Winner-Only Skew-KD
├── run_generative_benchmarks_08.py           # 5-task generative suite (GSM8K, MATH-500, GPQA, MBPP, HumanEval)
├── run_hardware_benchmarks_09.py             # Multi-scale TTFT, Prefill, Decode & VRAM profiler (512..32768)
├── export_to_hf_hub_2B_10.py                 # Self-contained HF Hub exporter & API verifier for 2B Hybrid
├── export_to_hf_hub_4B_10.py                 # Self-contained HF Hub exporter & API verifier for 4B Hybrid
├── run_master_pipeline_8xh100_0.8B.sh        # Master 8x H100 orchestration script for 0.8B Hybrid (r=16 -> 4B KD)
├── run_master_pipeline_8xh100_2B.sh          # Master 8x H100 orchestration script for 2B Hybrid (r=32 -> 4B KD)
└── run_master_pipeline_8xh100_4B.sh          # Master 8x H100 orchestration script for 4B Hybrid (r=32 -> 9B KD)
```

---

## 🚀 Execution Guide

### 1. Run the Complete 8-Stage Master Distillation Pipeline (`8x NVIDIA H100`)

Each master orchestration script is idempotent (automatically skipping completed stages via `.done` markers and checkpoint verification) and runs the entire pipeline from Stage 0 initialization through Stage 4C GRPO and final benchmark reports:

```bash
# Distill BareTorch-Qwen3.5-0.8B-Hybrid (Base: Qwen3.5-0.8B, Rank=16 -> Up-Distill Teacher: Qwen3.5-4B)
bash run_master_pipeline_8xh100_0.8B.sh

# Distill BareTorch-Qwen3.5-2B-Hybrid (Base: Qwen3.5-2B, Rank=32 -> Up-Distill Teacher: Qwen3.5-4B)
bash run_master_pipeline_8xh100_2B.sh

# Distill BareTorch-Qwen3.5-4B-Hybrid (Base: Qwen3.5-4B, Rank=32 -> Up-Distill Teacher: Qwen3.5-9B)
bash run_master_pipeline_8xh100_4B.sh
```

### 2. Run Multi-Scale Hardware Speed & VRAM Benchmarks (`512` to `32,768` Tokens)

```bash
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" python run_hardware_benchmarks_09.py \
  --teacher_id Qwen/Qwen3.5-4B \
  --student_ckpt ./checkpoints_4B_hybrid/stage4c_grpo_math_code_1500steps_9B/best \
  --rank 32 \
  --chunk_size 64 \
  --context_lengths 512 1024 2048 4096 8192 16384 32768 \
  --gen_tokens 64 \
  --device cuda:0
```

### 3. Export Self-Contained Model to Hugging Face Hub & Run API Sanity Check

```bash
# Export, verify single-point & batched generation, and push 2B or 4B Hybrid to Hugging Face Hub
python export_to_hf_hub_2B_10.py --push_to_hub --device cuda:0
python export_to_hf_hub_4B_10.py --push_to_hub --device cuda:0
```

---

## 📜 Citation

If you use BareTorch, CS-LRAD, or our Surgical Layerwise Up-Distillation Engine in your research, please cite:

```bibtex
@article{kovacevic2026baretorch,
  title={BareTorch: Challenging State-of-The-Art Sequence Mixing Topologies via Kernel-Free, Pure GEMM-Compliant Architectures},
  author={Kovacevic Buvinic, Martin Ignacio},
  journal={BareTorch Framework Laboratory Technical Report},
  year={2026},
  url={[https://github.com/martin-kbcc/baretorch](https://github.com/martin-kbcc/baretorch)}
}
```

---

## ⚖️ License

All source code, distillation pipelines, and exported model weights in this repository are open-source software licensed under the **Apache License 2.0**. See [LICENSE](LICENSE) for full details.