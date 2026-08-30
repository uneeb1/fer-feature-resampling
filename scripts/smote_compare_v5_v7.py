"""
SVMSMOTE comparison: v5 vs v7 features.
Resample all minority classes up to majority (happy=7215).
3 seeds, fresh linear heads, evaluate 4 conditions.
"""
print("=" * 60)
print("START — smote_compare_v5_v7.py")
print("=" * 60)

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, classification_report
from imblearn.over_sampling import SVMSMOTE

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = "./results/smote_compare"
os.makedirs(OUT_DIR, exist_ok=True)

CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
SEEDS = [42, 123, 456]
TARGET_COUNT = 7215
LR = 1e-4
LABEL_SMOOTHING = 0.1
MAX_EPOCHS = 50
PATIENCE = 5

# ── Load features ────────────────────────────────────────────────────────────

def load_features(prefix):
    tr = np.load(f"{prefix}_train.npz")
    va = np.load(f"{prefix}_val.npz")
    te = np.load(f"{prefix}_test.npz")
    return (tr["features"], tr["labels"],
            va["features"], va["labels"],
            te["features"], te["labels"])

v5_data = load_features("experiments/fer2013_final/results/smote/features")
v7_data = load_features("results/smote_compare/features_v7")

print(f"v5 train: {v5_data[0].shape}, val: {v5_data[2].shape}, test: {v5_data[4].shape}")
print(f"v7 train: {v7_data[0].shape}, val: {v7_data[2].shape}, test: {v7_data[4].shape}")

# ── SVMSMOTE resampling ──────────────────────────────────────────────────────

def apply_svmsmote(X, y, seed):
    counts = np.bincount(y, minlength=7)
    strategy = {}
    for i in range(7):
        if counts[i] < TARGET_COUNT:
            strategy[i] = TARGET_COUNT
    k = min(5, min(counts) - 1)
    if k < 5:
        print(f"  Warning: reduced k_neighbors to {k} (smallest class has {min(counts)} samples)")
    smote = SVMSMOTE(sampling_strategy=strategy, k_neighbors=k, random_state=seed)
    X_res, y_res = smote.fit_resample(X, y)
    print(f"  Seed {seed}: {X.shape[0]} -> {X_res.shape[0]} samples, dist: {np.bincount(y_res)}")
    return X_res, y_res

# ── Train + evaluate head ────────────────────────────────────────────────────

def train_head(X_train, y_train, X_val, y_val, X_test, y_test, dropout, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    test_ds = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)

    head = nn.Sequential(nn.Dropout(dropout), nn.Linear(512, 7)).to(DEVICE)
    optimizer = optim.Adam(head.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    best_val_f1, best_state, patience_ctr = 0.0, None, 0

    for epoch in range(MAX_EPOCHS):
        head.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(head(xb), yb)
            loss.backward()
            optimizer.step()

        head.eval()
        preds, labels = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                p = head(xb.to(DEVICE)).argmax(1).cpu().numpy()
                preds.extend(p)
                labels.extend(yb.numpy())
        val_f1 = f1_score(labels, preds, average="macro", zero_division=0)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                break

    head.load_state_dict(best_state)
    head.eval()

    results = {}
    for name, loader in [("val", val_loader), ("test", test_loader)]:
        preds, labels = [], []
        with torch.no_grad():
            for xb, yb in loader:
                p = head(xb.to(DEVICE)).argmax(1).cpu().numpy()
                preds.extend(p)
                labels.extend(yb.numpy())
        macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
        report = classification_report(labels, preds, target_names=CLASS_NAMES,
                                       zero_division=0, output_dict=True)
        per_class = {c: report[c]["f1-score"] for c in CLASS_NAMES}
        per_class_recall = {c: report[c]["recall"] for c in CLASS_NAMES}
        results[name] = {"macro_f1": macro_f1, "per_class_f1": per_class,
                         "per_class_recall": per_class_recall,
                         "accuracy": report["accuracy"]}
    return results

# ── Run all 4 conditions ─────────────────────────────────────────────────────

conditions = {}

for baseline_name, data, dropout in [("v5", v5_data, 0.3), ("v7", v7_data, 0.4)]:
    X_tr, y_tr, X_va, y_va, X_te, y_te = data

    # Baseline (no SMOTE)
    print(f"\n--- {baseline_name} baseline (no SMOTE) ---")
    seed_results = []
    for s in SEEDS:
        r = train_head(X_tr, y_tr, X_va, y_va, X_te, y_te, dropout, s)
        seed_results.append(r)
        print(f"  seed {s}: test macro-F1={r['test']['macro_f1']:.4f}")
    conditions[f"{baseline_name}_baseline"] = seed_results

    # + SVMSMOTE
    print(f"\n--- {baseline_name} + SVMSMOTE ---")
    seed_results = []
    for s in SEEDS:
        print(f"  Resampling seed {s}...")
        X_res, y_res = apply_svmsmote(X_tr, y_tr, s)
        r = train_head(X_res, y_res, X_va, y_va, X_te, y_te, dropout, s)
        seed_results.append(r)
        print(f"  seed {s}: test macro-F1={r['test']['macro_f1']:.4f}")
    conditions[f"{baseline_name}_smote"] = seed_results

# ── Aggregate results ────────────────────────────────────────────────────────

print("\n" + "=" * 80)
print("COMPARISON TABLE — Test Set (mean ± std over 3 seeds)")
print("=" * 80)

summary = {}
header = f"{'Condition':<20} {'macro-F1':>12} {'disgust F1':>12} {'fear F1':>12} {'sad F1':>12}"
print(header)
print("-" * len(header))

for cond_name, seed_results in conditions.items():
    macro_f1s = [r["test"]["macro_f1"] for r in seed_results]
    disgust_f1s = [r["test"]["per_class_f1"]["disgust"] for r in seed_results]
    fear_f1s = [r["test"]["per_class_f1"]["fear"] for r in seed_results]
    sad_f1s = [r["test"]["per_class_f1"]["sad"] for r in seed_results]

    row = {
        "macro_f1_mean": float(np.mean(macro_f1s)),
        "macro_f1_std": float(np.std(macro_f1s)),
        "per_class_mean": {c: float(np.mean([r["test"]["per_class_f1"][c] for r in seed_results]))
                          for c in CLASS_NAMES},
        "per_class_std": {c: float(np.std([r["test"]["per_class_f1"][c] for r in seed_results]))
                         for c in CLASS_NAMES},
        "per_class_recall_mean": {c: float(np.mean([r["test"]["per_class_recall"][c] for r in seed_results]))
                                  for c in CLASS_NAMES},
        "per_class_recall_std": {c: float(np.std([r["test"]["per_class_recall"][c] for r in seed_results]))
                                for c in CLASS_NAMES},
    }
    summary[cond_name] = row

    print(f"{cond_name:<20} "
          f"{row['macro_f1_mean']:.3f}±{row['macro_f1_std']:.3f}  "
          f"{row['per_class_mean']['disgust']:.3f}±{row['per_class_std']['disgust']:.3f}  "
          f"{row['per_class_mean']['fear']:.3f}±{row['per_class_std']['fear']:.3f}  "
          f"{row['per_class_mean']['sad']:.3f}±{row['per_class_std']['sad']:.3f}")

# ── Save ─────────────────────────────────────────────────────────────────────

with open(os.path.join(OUT_DIR, "comparison_results.json"), "w") as f:
    json.dump({"summary": summary, "raw": {k: [r for r in v] for k, v in conditions.items()}}, f, indent=2)

print(f"\nResults saved to {OUT_DIR}/comparison_results.json")

print("\n" + "=" * 60)
print("END — smote_compare_v5_v7.py")
print("=" * 60)
