import argparse
import glob
import logging
import multiprocessing as mp
import os
import random
import time
import warnings
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# Silence environment noise, warnings, and internal library thread thrashing
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")

# Force Python to spawn clean worker processes to prevent Rust/C++ memory fork panics
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

# Limit internal PyArrow C++ worker threads inside each multiprocessing process
pa.set_cpu_count(1)

import transformers

transformers.logging.set_verbosity_error()
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)


def extract_text_from_batch(batch):
    """
    Dynamically identifies and extracts text fields across web, code, 
    math, and Synthetic CoT/OpenR1 schemas into Qwen ChatML format.
    """
    raw_texts = []
    top_level_cols = batch.schema.names  # Inspect ONLY top-level schema fields

    # 1. Prioritize OpenR1 / ChatML 'messages' schema
    if "messages" in top_level_cols:
        try:
            messages_list = batch["messages"].to_pylist()
            for msg_group in messages_list:
                if not msg_group or not isinstance(msg_group, (list, tuple)):
                    continue
                
                formatted_conv = []
                for m in msg_group:
                    if isinstance(m, dict):
                        role = m.get("role", "user")
                        content = m.get("content", "")
                        if content:
                            formatted_conv.append(f"<|im_start|>{role}\n{content}<|im_end|>")
                
                if formatted_conv:
                    raw_texts.append("\n".join(formatted_conv))
            if raw_texts:
                return raw_texts
        except Exception:
            pass  # Fall through to standard column parsing if struct extraction fails

    # 2. Standard single-column text/content/solution schemas
    if "text" in top_level_cols:
        raw_texts = batch["text"].to_pylist()
    elif "content" in top_level_cols:
        raw_texts = batch["content"].to_pylist()
    elif "solution" in top_level_cols:
        problems = batch["problem"].to_pylist() if "problem" in top_level_cols else [""] * len(batch)
        solutions = batch["solution"].to_pylist()
        raw_texts = [f"{p}\n{s}".strip() for p, s in zip(problems, solutions)]
    else:
        # Fallback to first string column found
        first_col = top_level_cols[0]
        raw_texts = batch[first_col].to_pylist()

    # Filter out None/null values and non-string types safely
    return [str(t) for t in raw_texts if t is not None and len(str(t).strip()) > 0]


def process_shard(args):
    """Worker function to tokenize a Parquet shard using atomic .tmp file writes."""
    shard_path, output_dir, tokenizer_name, shard_idx, dtype_str = args
    shard_name = os.path.basename(shard_path)

    dtype = np.dtype(dtype_str)
    bytes_per_token = dtype.itemsize

    bin_filename = f"shard_{shard_idx:05d}.bin"
    out_bin_path = os.path.join(output_dir, bin_filename)
    tmp_bin_path = out_bin_path + ".tmp"

    # 1. Skip if fully completed .bin file already exists
    if os.path.exists(out_bin_path) and os.path.getsize(out_bin_path) > 0:
        total_tokens = os.path.getsize(out_bin_path) // bytes_per_token
        return shard_name, total_tokens

    # 2. Clean up leftover temporary file from any interrupted run
    if os.path.exists(tmp_bin_path):
        os.remove(tmp_bin_path)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name, 
        local_files_only=False, 
        use_fast=True,
        trust_remote_code=True
    )
    tokenizer.model_max_length = 10**9  # Suppress sequence length warnings

    eos_token_id = (
        tokenizer.eos_token_id
        if tokenizer.eos_token_id is not None
        else tokenizer.pad_token_id
    )
    total_tokens = 0

    try:
        pf = pq.ParquetFile(shard_path)

        # Write to .tmp file first
        with open(tmp_bin_path, "wb") as f:
            for batch in pf.iter_batches(batch_size=1000):
                texts = extract_text_from_batch(batch)
                if not texts:
                    continue

                tokenized_batch = tokenizer(
                    texts, add_special_tokens=False, truncation=False, padding=False
                )["input_ids"]

                chunk_tokens = []
                for doc_tokens in tokenized_batch:
                    chunk_tokens.extend(doc_tokens)
                    chunk_tokens.append(eos_token_id)

                if chunk_tokens:
                    arr = np.array(chunk_tokens, dtype=dtype)
                    f.write(arr.tobytes())
                    total_tokens += len(arr)

        # Atomic rename once completely finished
        os.replace(tmp_bin_path, out_bin_path)
        return shard_name, total_tokens

    except Exception as e:
        print(f"\n❌ Error processing shard {shard_path}: {e}")
        if os.path.exists(tmp_bin_path):
            os.remove(tmp_bin_path)
        return None, 0


def main():
    parser = argparse.ArgumentParser(
        description="BareTorch Ultra-Fast Parquet-to-Binary Tokenizer"
    )
    parser.add_argument(
        "--raw_dir",
        type=str,
        default="./raw_dclm_100bt",
        help="Directory containing raw Parquet files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./tokenized_bin/dclm_100bt",
        help="Output directory for binary files.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="Qwen/Qwen3.8-27B",
        help="Hugging Face tokenizer identifier.",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=12,
        help="Number of parallel worker processes.",
    )
    parser.add_argument(
        "--val_shards",
        type=int,
        default=2,
        help="Number of initial shards reserved for validation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling shards before split.",
    )

    args = parser.parse_args()

    # 1. Dynamically evaluate vocab size to set uint16 vs uint32 safely
    from transformers import AutoTokenizer

    print(f"Loading Tokenizer Metadata: '{args.tokenizer_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, trust_remote_code=True)
    vocab_size = getattr(tokenizer, "vocab_size", len(tokenizer))

    if vocab_size < 65536:
        dtype_str = "uint16"
    else:
        dtype_str = "uint32"

    print("=" * 70)
    print(
        f"Tokenizer: '{args.tokenizer_name}' | Vocab Size: {vocab_size:,} |"
        f" Selected Dtype: {dtype_str}"
    )
    print("=" * 70)

    os.makedirs(args.output_dir, exist_ok=True)
    val_dir = os.path.join(args.output_dir, "val")
    train_dir = os.path.join(args.output_dir, "train")
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)

    # 2. Discover recursive Parquet files
    parquet_shards = sorted(
        glob.glob(f"{args.raw_dir}/**/*.parquet", recursive=True)
    )
    if not parquet_shards:
        print(f"❌ Error: No .parquet files found in '{args.raw_dir}'.")
        return

    # 3. Shuffle shards with a fixed seed to get balanced train/val splits
    random.seed(args.seed)
    random.shuffle(parquet_shards)

    print(
        f"\n🚀 Found {len(parquet_shards)} Parquet shards in '{args.raw_dir}'."
    )
    print(
        f"Starting tokenization with {args.num_proc} processes (Val shards:"
        f" {args.val_shards})...\n"
    )

    tasks = []
    for idx, shard_path in enumerate(parquet_shards):
        target_dir = val_dir if idx < args.val_shards else train_dir
        tasks.append((shard_path, target_dir, args.tokenizer_name, idx, dtype_str))

    start_time = time.time()
    total_tokens = 0

    with mp.Pool(processes=args.num_proc) as pool:
        pbar = tqdm(
            pool.imap_unordered(process_shard, tasks),
            total=len(tasks),
            desc="Tokenizing Shards",
            unit="shard",
        )
        for result in pbar:
            if result[0] is not None:
                total_tokens += result[1]
                pbar.set_postfix({"Total Tokens": f"{total_tokens:,}"})

    bytes_per_token = 2 if dtype_str == "uint16" else 4
    elapsed_min = (time.time() - start_time) / 60
    total_gb = (total_tokens * bytes_per_token) / (1024**3)

    print("\n" + "=" * 70)
    print(
        f"🎉 SUCCESS: Tokenized {len(parquet_shards)} shards in"
        f" {elapsed_min:.2f} minutes!"
    )
    print(f"📦 Total Tokens Generated: {total_tokens:,}")
    print(f"💾 Total {dtype_str} Dataset Size: {total_gb:.2f} GB")
    print(f"📁 Binary Dataset Path: {os.path.abspath(args.output_dir)}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()