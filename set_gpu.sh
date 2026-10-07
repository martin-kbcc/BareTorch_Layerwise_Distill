#!/bin/bash
set -e

echo "🔧 Optimizing RTX 4090 GPU Settings..."

# Enable persistence mode
sudo nvidia-smi -pm 1

# Set 300W power cap per GPU
sudo nvidia-smi -pl 300

# Lock core clock to 2100 MHz
sudo nvidia-smi -lgc 2100