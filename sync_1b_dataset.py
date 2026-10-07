# /home/martinkb/Desktop/BareTorch_Layerwise_Distill/sync_1b_dataset.py
import argparse
import json
import os
import subprocess

# 1B Token Multi-Domain Proportional Ratios (7 Datasets)
RATIOS = {
    "fineweb_edu_100bt": 0.32,  # 320M tokens | High-quality educational prose
    "stack_dedup":       0.24,  # 240M tokens | Code syntax & structure
    "dclm_100bt":        0.16,  # 160M tokens | Broad web crawl diversity
    "finemath_4plus":    0.10,  # 100M tokens | Mathematical reasoning
    "cosmopedia_v2":     0.10,  # 100M tokens | Synthetic textbook prose
    "finepdfs_100bt":    0.07,  #  70M tokens | Technical & structured PDFs
    "openr1_math":       0.01,  #  10M tokens | High-signal synthetic CoT math
}

BYTES_PER_TOKEN_PACKED = 4  # uint32 = 4 bytes per token


def get_remote_shards(remote_path):
    """Query rclone for file list in JSON format."""
    cmd = ["rclone", "lsjson", remote_path, "--recursive"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        return []
    try:
        return json.loads(res.stdout)
    except Exception:
        return []


def main():
    parser = argparse.ArgumentParser(
        description="Sync 1B Token Teacher Distillation Shard Triplets across 7 Domains from Cloudflare R2"
    )
    parser.add_argument(
        "--remote",
        type=str,
        default="r2:baretorch-data/teacher_predictions",
        help="R2 remote base path for teacher predictions",
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        default="./data_1B/train",
        help="Local target directory for downloaded shard triplets",
    )
    parser.add_argument(
        "--total_tokens",
        type=float,
        default=1.0e9,
        help="Total target tokens across all domains (default: 1.0 Billion)",
    )
    args = parser.parse_args()

    os.makedirs(args.target_dir, exist_ok=True)

    print("=" * 70)
    print("🚀 Preparing 1B Token Proportional Dataset Sync from R2 (7 Domains)")
    print(f"Remote Base        : {args.remote}")
    print(f"Target Total Tokens: {args.total_tokens / 1e9:.2f} Billion")
    print(f"Target Directory   : {args.target_dir}")
    print("=" * 70)

    for ds_name, ratio in RATIOS.items():
        target_tokens = args.total_tokens * ratio
        target_packed_bytes = target_tokens * BYTES_PER_TOKEN_PACKED

        print(
            f"\n📦 Processing '{ds_name}' (Target: {target_tokens/1e6:.1f}M tokens | ~{target_packed_bytes/1e6:.1f} MB packed tokens)"
        )

        ds_remote_path = f"{args.remote}/{ds_name}"
        files = get_remote_shards(ds_remote_path)

        if not files:
            print(f"  ⚠ Warning: No files found in remote path '{ds_remote_path}'")
            continue

        # Filter binary token shards (matching prefix_packed_tokens.bin)
        packed_files = [
            f for f in files if f["Path"].endswith("_packed_tokens.bin") and not f.get("IsDir", False)
        ]

        accumulated_bytes = 0
        selected_prefixes = []

        # Select minimum number of shard triplets required to satisfy token budget
        for p_file in sorted(packed_files, key=lambda x: x["Path"]):
            prefix = p_file["Path"].replace("_packed_tokens.bin", "")
            accumulated_bytes += p_file["Size"]
            selected_prefixes.append(prefix)
            if accumulated_bytes >= target_packed_bytes:
                break

        print(
            f"  └─ Selected {len(selected_prefixes)} shard triplet(s) ({accumulated_bytes / 1e6:.1f} MB packed token data)"
        )

        local_ds_dir = os.path.join(args.target_dir, ds_name)
        os.makedirs(local_ds_dir, exist_ok=True)

        # Download selected shard triplets via rclone
        for prefix in selected_prefixes:
            for suffix in [
                "_packed_tokens.bin",
                "_teacher_indices.bin",
                "_teacher_values.bin",
            ]:
                rel_file = f"{prefix}{suffix}"
                file_remote = f"{ds_remote_path}/{rel_file}"
                file_local = os.path.join(
                    local_ds_dir, os.path.basename(rel_file)
                )

                if not os.path.exists(file_local):
                    print(f"    ├─ Downloading: {os.path.basename(rel_file)}...")
                    cmd = [
                        "rclone",
                        "copyto",
                        file_remote,
                        file_local,
                        "--transfers",
                        "8",
                        "--s3-chunk-size",
                        "64M",
                    ]
                    subprocess.run(cmd)
                else:
                    print(f"    ├─ Existing: {os.path.basename(rel_file)} (Skipping download)")

    print("\n" + "=" * 70)
    print("✅ 1B Token Teacher Predictions Dataset Sync Complete across 7 Domains!")
    print("=" * 70)


if __name__ == "__main__":
    main()