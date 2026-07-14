"""
Print a summary table of all completed experiments under ./experiments/
"""

import os
import re

EXPERIMENTS_DIR = "./experiments"

rows = []

for exp in sorted(os.listdir(EXPERIMENTS_DIR)):
    exp_path = os.path.join(EXPERIMENTS_DIR, exp)
    config_path  = os.path.join(exp_path, "config.txt")
    results_path = os.path.join(exp_path, "results.txt")

    if not os.path.exists(results_path):
        continue

    config = {}
    with open(config_path) as f:
        for line in f:
            if ":" in line:
                k, v = line.split(":", 1)
                config[k.strip()] = v.strip()

    with open(results_path) as f:
        content = f.read()

    best_f1   = re.search(r"Best Macro-F1\s*:\s*([\d.]+)", content)
    best_ep   = re.search(r"Best epoch\s*:\s*(\d+)", content)
    macro_avg = re.search(r"macro avg\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", content)

    best_f1   = float(best_f1.group(1))   if best_f1   else 0.0
    best_ep   = int(best_ep.group(1))     if best_ep   else 0
    macro_f1  = float(macro_avg.group(3)) if macro_avg else 0.0

    rows.append({
        "exp":        exp,
        "scheduler":  config.get("Scheduler", "?"),
        "backbone_lr": config.get("Backbone LR", "?"),
        "wd":         config.get("Weight decay", "?"),
        "dropout":    config.get("Dropout", "?"),
        "best_epoch": best_ep,
        "best_f1":    best_f1,
    })

print(f"\n{'Exp':<8} {'Backbone LR':<14} {'WD':<8} {'Dropout':<10} {'Scheduler':<35} {'BestEp':<8} {'Val Macro-F1'}")
print("-" * 100)
for r in rows:
    print(f"{r['exp']:<8} {r['backbone_lr']:<14} {r['wd']:<8} {r['dropout']:<10} {r['scheduler']:<35} {r['best_epoch']:<8} {r['best_f1']:.4f}")

print(f"\nTotal experiments: {len(rows)}")
