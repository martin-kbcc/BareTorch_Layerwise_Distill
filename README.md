# BareTorch Layerwise Distillation Engine

[![PyTorch](https://img.shields.io/badge/PyTorch-2.6+-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![BareTorch Ecosystem](https://img.shields.io/badge/Ecosystem-BareTorch-green.svg)](https://github.com/martin-kbcc/baretorch)

**BareTorch Layerwise Distillation Engine** is the production-grade pipeline designed to distill state-of-the-art dense Transformer models (e.g., Qwen 3.5) into sub-quadratic, **3:1 Hybrid CS-LRAD** (Chunk-Segmented Low-Rank Delta Engine) BareTorch models.

By combining layerwise hidden state feature matching with global logit alignment, this framework converts standard $O(N^2)$ Softmax attention architectures into kernel-free, pure GEMM-compliant sub-quadratic models that achieve significant inference speedups, reduced VRAM footprints, and long-context resilience without custom CUDA or Triton kernels.

---

## 📄 License

This project is licensed under the **Apache License 2.0**. See the [LICENSE](LICENSE) file for details.

---

## 🔗 Related BareTorch Ecosystem Repositories

* 📄 **Research Paper:** [BareTorch: Challenging State-of-The-Art Sequence Mixing Topologies via Kernel-Free, Pure GEMM-Compliant Architectures (PDF)](https://github.com/martin-kbcc/baretorch-experiments/blob/main/paper.pdf)
* ⚡ **Pre-training Framework:** [BareTorch Core Pre-training Repository](https://github.com/martin-kbcc/baretorch)
* 🔬 **Experimental & Benchmarking Suite:** [BareTorch Experiments & Evaluation Harness](https://github.com/martin-kbcc/baretorch-experiments/tree/main)

---

## 🏗️ Architectural Topology: 3:1 Hybrid CS-LRAD

The distilled models follow a **3:1 Interleaved CS-LRAD Hybrid Pattern**:
* **75% Linear Recurrent Layers:** 3 consecutive **CS-LRAD** (Chunk-Segmented Low-Rank Delta Engine) blocks that replace full Softmax attention with rank-$r$ subspace projections ($r=8$) and chunkwise recurrent updates.
* **25% Full Attention Layers:** Every 4th block retains the native Teacher Transformer attention mechanism to preserve global context expressivity.

    [CS-LRAD] ──> [CS-LRAD] ──> [CS-LRAD] ──> [Teacher Transformer] ──> ...
       (L0)          (L1)          (L2)              (L3)

This hybrid topology reduces active KV-cache VRAM traffic during decoding from $O(L)$ to $O(1)$ across 75% of the model layers, yielding up to **12.5x CUDA decode throughput** and eliminating Out-of-Memory (OOM) crashes on long contexts.

---

## 🔄 The 5-Stage Production Pipeline

The distillation workflow is structured into 5 end-to-end execution stages:

    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                       BareTorch Distillation Pipeline                       │
    └─────────────────────────────────────────────────────────────────────────────┘
      Stage 1: Multi-Domain Dataset Preparation & Cloud Sync
        └── Syncs pre-tokenized binary datasets (20M / 100M tokens across 7 domains).
      
      Stage 2: Layerwise Hidden State Extraction & Single-Layer Matching
        ├── Extracts Teacher hidden states using TransformerEngine FP8 acceleration.
        └── Trains individual CS-LRAD layers via MSE loss with SVD warm-start initialization.
      
      Stage 3: Teacher Logit Dataset Sync
        └── Downloads 1B / 5B token multi-domain teacher prediction binary shards.
      
      Stage 4: End-to-End Unfrozen Global Logit Distillation
        └── Assembles full model & fine-tunes with 50% Cross-Entropy + 50% Top-4 Teacher KL Loss.
      
      Stage 5: Supervised Fine-Tuning (SFT) & Instruction Alignment
        └── Multi-task ChatML fine-tuning (Chat, Code, Math, Reasoning) with completion masking.

---

## 📁 Repository Structure

    BareTorch_Layerwise_Distill/
    ├── baretorch/                         # Core BareTorch architecture library & modules
    ├── hidden_states_extraction.py        # Multi-GPU FP8 exact-quota hidden state extractor
    ├── sync_20m_dataset.py                # Cloudflare R2 dataset sync helper (20M / 100M tokens)
    ├── sync_1b_dataset.py                 # Cloudflare R2 teacher logit sync helper (1B / 5B tokens)
    ├── train_stage_1.py                   # Stage 2: Single-layer CS-LRAD feature matching trainer
    ├── train_stage_2.py                   # Stage 4: Full-model unfrozen global logit alignment
    ├── train_stage_3.py                   # Stage 5: Supervised Fine-Tuning (SFT) engine (8-Bit AdamW)
    ├── Qwen3.5-0.8B/                      # Runner directory for Qwen 3.5 0.8B (Local 2x GPU)
    │   ├── run_stage_1.sh                 # Sync 20M tokens
    │   ├── run_stage_2.sh                 # Single-layer feature extraction & distillation
    │   ├── run_stage_3.sh                 # Sync 1B teacher predictions dataset
    │   ├── run_stage_4.sh                 # Global logit alignment (1B tokens)
    │   └── run_stage_5_sft.sh             # SFT instruction fine-tuning
    └── Qwen3.5-2B/                        # Runner directory for Qwen 3.5 2B (8x H100 Cluster)
        ├── run_stage_1.sh                 # Sync 100M tokens
        ├── run_stage_2.sh                 # Single-layer feature extraction & distillation
        ├── run_stage_3.sh                 # Sync 5B teacher predictions dataset
        └── run_stage_4.sh                 # Global logit alignment (5B tokens)

---

## 🚀 Execution Guide

### Option A: Local Execution (Qwen3.5-0.8B on 2x GPUs)

    cd Qwen3.5-0.8B/

    # Stage 1: Sync 20M Dataset
    ./run_stage_1.sh

    # Stage 2: Layerwise Single-Layer Distillation
    ./run_stage_2.sh

    # Stage 3: Sync 1B Teacher Logit Dataset
    ./run_stage_3.sh

    # Stage 4: Global Logit Alignment
    ./run_stage_4.sh

    # Stage 5: Supervised Fine-Tuning
    ./run_stage_5_sft.sh

### Option B: High-Throughput Cluster Execution (Qwen3.5-2B on 8x H100 GPUs)

    cd Qwen3.5-2B/

    # Stage 1: Sync 100M Dataset
    ./run_stage_1.sh

    # Stage 2: Layerwise Distillation across 8x H100s
    ./run_stage_2.sh

    # Stage 3: Sync 5B Teacher Logit Dataset
    ./run_stage_3.sh

    # Stage 4: Global Logit Alignment (5B Tokens, Effective Batch Size 128)
    ./run_stage_4.sh

---

## 🛠️ Key Technical Features

* **Kernel-Free Pure GEMM Execution:** Designed entirely with standard PyTorch tensor ops mapping directly to native BLAS/GEMM routines, making it 100% portable across NVIDIA GPUs, Apple Silicon (MLX), and TPUs.
* **Truncated SVD Warm-Start:** Initializes CS-LRAD low-rank matrices ($W_u$, $W_r$) via singular value decomposition over teacher projection weights.
* **Zero-Copy Memory-Mapped Dataset:** Employs `np.memmap` binary readers with worker isolation to eliminate CPU RAM bottlenecks during high-throughput feature streaming.
* **Asynchronous Cloudflare R2 Sync:** Features background `rclone` callbacks to stream completed layer weights and intermediate checkpoints directly to cloud storage without stalling GPU training.
* **Mixed Precision & 8-Bit Optimization:** Built natively for `bfloat16` execution paired with `bitsandbytes` 8-Bit AdamW optimizer for minimal VRAM overhead.

---

## 📜 Citation & Reference

If you use BareTorch or this distillation framework in your research, please cite:

    @article{kovacevic2025baretorch,
      title={BareTorch: Challenging State-of-The-Art Sequence Mixing Topologies via Kernel-Free, Pure GEMM-Compliant Architectures},
      author={Kovacevic Buvinic, Martin Ignacio},
      journal={Independent Research / BareTorch Framework Laboratory},
      year={2025}
    }