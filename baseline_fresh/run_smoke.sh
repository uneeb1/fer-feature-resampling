#!/bin/bash
set -euo pipefail

echo "=== SMOKE TEST — 1 seed, 112x112, 2 epochs, 500 samples ==="

python3 main.py \
    --config config.yaml \
    --resolution 112 \
    --epochs 2 \
    --seeds 42 \
    --batch-size 32 \
    --subset 500 \
    --no-tta \
    --num-workers 0 \
    --output-dir .

echo "=== Smoke test done. Check figures/ and features/ ==="
