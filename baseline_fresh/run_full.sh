#!/bin/bash
set -euo pipefail

echo "=== FER2013 Baseline Fresh — Full Run ==="
echo "Seeds: 42, 123, 456 | Resolution: 224 | Epochs: 100"
echo "Device: $(python3 -c 'import torch; print("cuda" if torch.cuda.is_available() else "cpu")')"

python3 main.py \
    --config config.yaml \
    --resolution 224 \
    --epochs 100 \
    --seeds 42 123 456 \
    --batch-size 64 \
    --output-dir .

echo "=== Done. Check metrics.json and figures/ ==="
