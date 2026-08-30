#!/bin/bash
set -euo pipefail

echo "=== Overfitting-Reduction Sweep (seed=42, 224x224) ==="

for config in g0 g1 g2 g3; do
    out_dir="./sweep_${config}"
    echo ""
    echo "============================================================"
    echo "Running ${config^^} (config_${config}.yaml) -> ${out_dir}"
    echo "============================================================"

    python3 main.py \
        --config "config_${config}.yaml" \
        --seeds 42 \
        --output-dir "${out_dir}"

    echo "${config^^} done. Results in ${out_dir}/metrics.json"
done

echo ""
echo "=== Sweep complete. Generating comparison table... ==="

python3 -c "
import json, os

configs = ['g0', 'g1', 'g2', 'g3']
labels = {'g0': 'G0 (baseline)', 'g1': 'G1 (freeze+discLR)', 'g2': 'G2 (G1+mixup+wd)', 'g3': 'G3 (full+stacked reg)'}
rows = []
for c in configs:
    path = f'./sweep_{c}/metrics.json'
    if not os.path.exists(path):
        continue
    with open(path) as f:
        m = json.load(f)
    s = m['per_seed']['42']
    rows.append({
        'config': labels[c],
        'train_f1': s['train_f1_at_best'],
        'val_f1': s['val_f1'],
        'test_f1': s['test_f1'],
        'gap': s['train_val_gap'],
        'best_ep': s['val_epoch'],
        'stop_ep': s['stop_epoch'],
        'disgust': s['test_per_class_f1']['disgust'],
        'fear': s['test_per_class_f1']['fear'],
        'sad': s['test_per_class_f1']['sad'],
        'verdict': s['verdict'],
    })

print()
print(f\"{'Config':<25} {'Train-F1':>8} {'Val-F1':>8} {'Test-F1':>8} {'Gap':>8} {'Best':>5} {'Stop':>5} {'Disgust':>8} {'Fear':>8} {'Sad':>8}  Verdict\")
print('-' * 130)
for r in rows:
    print(f\"{r['config']:<25} {r['train_f1']:8.4f} {r['val_f1']:8.4f} {r['test_f1']:8.4f} {r['gap']:8.4f} {r['best_ep']:5d} {r['stop_ep']:5d} {r['disgust']:8.4f} {r['fear']:8.4f} {r['sad']:8.4f}  {r['verdict']}\")
"

echo "=== Done ==="
