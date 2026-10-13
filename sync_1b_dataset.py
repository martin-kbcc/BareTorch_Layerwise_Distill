# /home/martinkb/Desktop/BareTorch_Layerwise_Distill/sync_1b_dataset.py
import argparse
import json
import os
import subprocess
import tempfile

# Proportional Ratios (7 Domains)
RATIOS = {
    "fineweb_edu_100bt": 0.32,
    "stack_dedup":       0.24,
    "dclm_100bt":        0.16,
    "finemath_4plus":    0.10,
    "cosmopedia_v2":     0.10,
    "finepdfs_100bt":    0.07,
    "openr1_math":       0.01,
}

BYTES_PER_TOKEN_PACKED = 4  # uint32 = 4 bytes per token


def get_remote_shards(remote_path):
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
        description="Global Multi-Domain Shard Sync from Cloudflare R2"
    )
    parser.add_argument(
        "--remote",
        type=str,
        default="r2:baretorch-data/teacher_predictions",
        help="R2 remote base path containing domain folders",
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        default="./data_20B/train",
        help="Local root target directory",
    )
    parser.add_argument(
        "--total_tokens",
        type=float,
        default=2.0e10,
        help="Total target tokens across all domains (default: 20 Billion)",
    )
    parser.add_argument(
        "--transfers",
        type=int,
        default=32,
        help="Number of concurrent file transfers",
    )
    args = parser.parse_args()

    os.makedirs(args.target_dir, exist_ok=True)

    print("=" * 70)
    print("🚀 BareTorch Global Multi-Domain Dataset Sync")
    print(f"Remote Base        : {args.remote}")
    print(f"Local Target Dir   : {args.target_dir}")
    print(f"Target Total Tokens: {args.total_tokens / 1e9:.2f} Billion")
    print(f"Concurrent Streams : {args.transfers}")
    print("=" * 70)

    global_files_to_sync = []
    total_planned_tokens = 0

    # 1. Inventory and select shards across all domains
    for ds_name, ratio in RATIOS.items():
        target_tokens = args.total_tokens * ratio
        target_packed_bytes = target_tokens * BYTES_PER_TOKEN_PACKED

        print(f"\n🔍 Inventorying '{ds_name}' (Target: {target_tokens/1e6:.1f}M tokens)...")

        ds_remote_path = f"{args.remote}/{ds_name}"
        files = get_remote_shards(ds_remote_path)

        if not files:
            print(f"  ⚠️ Warning: No files found in remote path '{ds_remote_path}'")
            continue

        packed_files = [
            f for f in files
            if f["Path"].endswith("_packed_tokens.bin") and not f.get("IsDir", False)
        ]

        accumulated_bytes = 0
        selected_prefixes = []

        for p_file in sorted(packed_files, key=lambda x: x["Path"]):
            prefix = p_file["Path"].replace("_packed_tokens.bin", "")
            accumulated_bytes += p_file["Size"]
            selected_prefixes.append(prefix)
            if accumulated_bytes >= target_packed_bytes:
                break

        actual_tokens = accumulated_bytes / BYTES_PER_TOKEN_PACKED
        total_planned_tokens += actual_tokens

        if accumulated_bytes < target_packed_bytes:
            print(
                f"  ⚠️ DEFICIT ALERT: Only {actual_tokens/1e6:.1f}M tokens available "
                f"(Target: {target_tokens/1e6:.1f}M)!"
            )
        else:
            print(f"  └─ Selected {len(selected_prefixes)} unique shard triplets ({actual_tokens/1e6:.1f}M tokens)")

        local_ds_dir = os.path.join(args.target_dir, ds_name)
        os.makedirs(local_ds_dir, exist_ok=True)

        # Build relative paths including domain prefix: "domain_name/prefix_xxx.bin"
        for prefix in selected_prefixes:
            for suffix in [
                "_packed_tokens.bin",
                "_teacher_indices.bin",
                "_teacher_values.bin",
            ]:
                rel_path = f"{ds_name}/{prefix}{suffix}"
                local_path = os.path.join(args.target_dir, rel_path)
                if not os.path.exists(local_path):
                    global_files_to_sync.append(rel_path)

    # 2. Check if work is already completed
    if not global_files_to_sync:
        print("\n" + "=" * 70)
        print("✅ All requested files already exist locally. Nothing to download!")
        print("=" * 70)
        return

    print("\n" + "=" * 70)
    print(f"📦 Total files to download : {len(global_files_to_sync):,}")
    print(f"📊 Total effective tokens  : {total_planned_tokens / 1e9:.2f} Billion")
    print("=" * 70)

    # 3. Write single global manifest and execute one rclone copy
    with tempfile.NamedTemporaryFile("w", delete=False) as f:
        for item in global_files_to_sync:
            f.write(f"{item}\n")
        manifest_file = f.name

    try:
        print(f"🔥 Launching unified rclone transfer ({args.transfers} concurrent streams across all domains)...\n")
        cmd = [
            "rclone",
            "copy",
            args.remote,
            args.target_dir,
            "--files-from",
            manifest_file,
            "--transfers",
            str(args.transfers),
            "--checkers",
            str(args.transfers),
            "--s3-chunk-size",
            "64M",
            "-P",
        ]
        subprocess.run(cmd, check=True)
    finally:
        if os.path.exists(manifest_file):
            os.remove(manifest_file)

    print("\n" + "=" * 70)
    print("🎉 Global Multi-Domain Dataset Sync Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()