# /home/martinkb/Desktop/BareTorch_Layerwise_Distill/sync_20m_dataset.py
import argparse
import json
import os
import subprocess

# 20M Token Multi-Domain Ratios (7 Datasets)
RATIOS = {
    "fineweb_edu_100bt": 0.25,  # High-quality educational web text
    "stack_dedup":       0.20,  # Code syntax, indentation, and structure
    "dclm_100bt":        0.15,  # Broad general web crawl diversity
    "cosmopedia_v2":     0.12,  # Synthetic textbook prose & instruction styles
    "finemath_4plus":    0.12,  # Web mathematical reasoning & equations
    "finepdfs_100bt":    0.08,  # PDF layouts & structured document formats
    "openr1_math":       0.08,  # High-signal synthetic math & chain-of-thought
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
        description="Sync 20M Tokenized Binary Sub-dataset across 7 Domains from Cloudflare R2"
    )
    parser.add_argument(
        "--remote",
        type=str,
        default="r2:baretorch-data/tokenized_bin",
        help="R2 remote base path for tokenized binaries",
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        default="./data_20M/train",
        help="Local target directory for downloaded shards",
    )
    parser.add_argument(
        "--total_tokens",
        type=float,
        default=20e6,
        help="Total target tokens across all datasets (default: 20 Million)",
    )
    args = parser.parse_args()

    os.makedirs(args.target_dir, exist_ok=True)

    print("=" * 70)
    print("🚀 Preparing 20M Token Proportional Dataset Sync from R2 (7 Domains)")
    print(f"Remote Base: {args.remote}")
    print(f"Target Total Tokens: {args.total_tokens / 1e6:.2f} Million")
    print("=" * 70)

    for ds_name, ratio in RATIOS.items():
        target_tokens = args.total_tokens * ratio
        target_packed_bytes = target_tokens * BYTES_PER_TOKEN_PACKED

        print(
            f"\n📦 Processing '{ds_name}' (Target: {target_tokens / 1e6:.2f}M tokens | ~{target_packed_bytes / 1e6:.2f} MB packed tokens)"
        )

        ds_remote_path = f"{args.remote}/{ds_name}/train"
        files = get_remote_shards(ds_remote_path)

        if not files:
            print(f"  ⚠ Warning: No files found in remote path '{ds_remote_path}'")
            continue

        # Filter binary token shards
        bin_files = [
            f for f in files if f["Path"].endswith(".bin") and not f.get("IsDir", False)
        ]

        accumulated_bytes = 0
        selected_files = []

        # Select minimum number of .bin shards required to satisfy target token budget
        for p_file in sorted(bin_files, key=lambda x: x["Path"]):
            accumulated_bytes += p_file["Size"]
            selected_files.append(p_file["Path"])
            if accumulated_bytes >= target_packed_bytes:
                break

        print(
            f"  └─ Selected {len(selected_files)} shard file(s) ({accumulated_bytes / 1e6:.2f} MB token binary data)"
        )

        local_ds_dir = os.path.join(args.target_dir, ds_name)
        os.makedirs(local_ds_dir, exist_ok=True)

        # Download selected .bin shards via rclone
        for rel_file in selected_files:
            file_remote = f"{ds_remote_path}/{rel_file}"
            file_local = os.path.join(local_ds_dir, os.path.basename(rel_file))

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

    print("\n✅ 20M Tokenized Dataset Sync Complete across 7 Domains!")


if __name__ == "__main__":
    main()