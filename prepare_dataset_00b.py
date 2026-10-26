# /home/martinkb/Desktop/BareTorch_Layerwise_Distill/prepare_dataset_00b.py
import os
import gc
import argparse
from itertools import chain
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Memory-safe parallel dataset downloader, tokenizer, and packer.")
    parser.add_argument("--dataset_name", type=str, default="HuggingFaceFW/fineweb-edu", help="HF dataset name")
    parser.add_argument("--dataset_config", type=str, default="sample-10BT", help="HF dataset config/subset")
    parser.add_argument("--tokenizer_path", type=str, default="./checkpoints/stage0_initialized", help="Tokenizer path")
    parser.add_argument("--output_dir", type=str, default="./data/fineweb_edu_packed_2048", help="Directory to save packed dataset")
    parser.add_argument("--cache_dir", type=str, default="./data/hf_cache", help="Disk cache directory for intermediate Arrow files")
    parser.add_argument("--seq_len", type=int, default=2048, help="Exact packed sequence length")
    parser.add_argument("--target_tokens_millions", type=float, default=500.0, help="Total packed train tokens (in Millions)")
    parser.add_argument("--val_tokens_millions", type=float, default=5.0, help="Validation tokens (in Millions)")
    parser.add_argument("--num_proc", type=int, default=min(16, max(1, os.cpu_count() // 2)), help="Number of CPU worker processes (default capped at 16 for RAM safety)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.cache_dir, exist_ok=True)

    print("=" * 80)
    print(f"MEMORY-SAFE PACKED DATASET BUILDER: {args.target_tokens_millions}M Train Tokens | seq_len={args.seq_len}")
    print(f"Using {args.num_proc} CPU workers | Disk Cache: {args.cache_dir}")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    eos_id = tokenizer.eos_token_id

    total_target_tokens = int((args.target_tokens_millions + args.val_tokens_millions) * 1_000_000)
    # FineWeb-Edu averages ~700-750 tokens per document; 650 is a safe conservative divisor
    estimated_docs = int(total_target_tokens / 650)

    print(f"1. Streaming ~{estimated_docs:,} raw documents directly to disk-mapped Arrow shards (Zero RAM bloat)...")

    def stream_text_only_generator():
        stream_ds = load_dataset(
            args.dataset_name,
            name=args.dataset_config,
            split="train",
            streaming=True,
        )
        for idx, item in enumerate(stream_ds):
            if idx >= estimated_docs:
                break
            yield {"text": item["text"]}

    # Dataset.from_generator streams directly to disk-backed Arrow files instead of holding a Python list in RAM!
    raw_dataset = Dataset.from_generator(
        stream_text_only_generator,
        cache_dir=args.cache_dir,
        writer_batch_size=2000,
    )
    gc.collect()
    print(f"   Streamed {len(raw_dataset):,} documents to disk.")

    # 2. Fused Tokenization + Sequence Packing in a SINGLE disk-mapped pass!
    seq_len = args.seq_len

    def tokenize_and_pack(examples):
        # Tokenize batch of texts without truncation or padding
        tokenized = tokenizer(examples["text"], add_special_tokens=False)["input_ids"]
        # Append EOS token to each document and flatten
        concatenated = []
        for doc_ids in tokenized:
            concatenated.extend(doc_ids)
            concatenated.append(eos_id)

        # Slice into exact multiples of seq_len
        total_len = (len(concatenated) // seq_len) * seq_len
        packed_blocks = [
            concatenated[i : i + seq_len] for i in range(0, total_len, seq_len)
        ]
        return {"input_ids": packed_blocks}

    print(f"2. Tokenizing & Packing into exact {seq_len}-token blocks with num_proc={args.num_proc}...")
    packed_ds = raw_dataset.map(
        tokenize_and_pack,
        batched=True,
        batch_size=500,          # Small batch size per worker keeps per-process RAM < 150MB
        writer_batch_size=500,   # Flush Arrow chunks to NVMe disk frequently
        num_proc=args.num_proc,
        remove_columns=raw_dataset.column_names,
        keep_in_memory=False,    # Force memory-mapping to disk
        desc=f"Tokenizing & Packing ({seq_len})",
    )
    gc.collect()

    total_blocks = len(packed_ds)
    total_packed_tokens = total_blocks * seq_len
    print(f"   Created {total_blocks:,} packed blocks ({total_packed_tokens / 1e6:.2f}M tokens total).")

    # 3. Split into Train and Validation sets
    val_blocks = max(64, int((args.val_tokens_millions * 1_000_000) // seq_len))
    train_blocks = min(total_blocks - val_blocks, int((args.target_tokens_millions * 1_000_000) // seq_len))

    packed_ds = packed_ds.shuffle(seed=args.seed)
    val_dataset = packed_ds.select(range(val_blocks))
    train_dataset = packed_ds.select(range(val_blocks, val_blocks + train_blocks))

    ds_dict = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
    })

    os.makedirs(args.output_dir, exist_ok=True)
    ds_dict.save_to_disk(args.output_dir, num_proc=min(4, args.num_proc))
    print(f"\n3. Saved Packed DatasetDict to: {args.output_dir}")
    print(f"   - Train blocks      : {len(train_dataset):,} ({len(train_dataset) * seq_len / 1e6:.2f}M tokens)")
    print(f"   - Validation blocks : {len(val_dataset):,} ({len(val_dataset) * seq_len / 1e6:.2f}M tokens)")
    print("=" * 80)


if __name__ == "__main__":
    main()