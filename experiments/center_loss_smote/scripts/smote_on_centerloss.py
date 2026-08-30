"""
Stage 2: SVMSMOTE on Center Loss features.
Loads best_model.pth from Stage 1, extracts 512-d features, applies SVMSMOTE
(full balancing), trains a linear head, evaluates with imbalanced + balanced protocol.
"""

import os
import sys
import copy
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, models, transforms
from sklearn.metrics import f1_score, classification_report
from collections import Counter
from imblearn.over_sampling import SVMSMOTE

# ── Config ────────────────────────────────────────────────────────────────────

NUM_CLASSES = 7
CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
BATCH_SIZE  = 64
DROPOUT     = 0.3
SEED        = 42
N_DRAWS     = 100
SAMPLES_PER_CLASS = 55

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
BASE_DIR      = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
DATA_DIR      = os.path.join(BASE_DIR, "data")
STAGE1_DIR    = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "results", "stage1_centerloss_v2"))
RESULTS_DIR   = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "results", "stage2_smote"))
CHECKPOINT    = os.path.join(STAGE1_DIR, "best_model.pth")

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# ── Seed ──────────────────────────────────────────────────────────────────────

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ── Helpers ───────────────────────────────────────────────────────────────────

val_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# No augmentation for feature extraction
clean_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def build_model():
    model = models.resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Sequential(nn.Dropout(DROPOUT), nn.Linear(model.fc.in_features, NUM_CLASSES))
    return model


def extract_features(model, loader):
    hook_output = {}
    def hook_fn(module, inp, out):
        hook_output["feat"] = out
    handle = model.avgpool.register_forward_hook(hook_fn)
    model.eval()
    feats, labs = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(DEVICE)
            _ = model(inputs)
            f = hook_output["feat"].squeeze(-1).squeeze(-1)
            feats.append(f.cpu().numpy())
            labs.append(labels.numpy())
    handle.remove()
    return np.concatenate(feats), np.concatenate(labs)


def train_head(train_X, train_y, val_X, val_y, num_epochs=50):
    train_ds = TensorDataset(torch.FloatTensor(train_X), torch.LongTensor(train_y))
    val_ds   = TensorDataset(torch.FloatTensor(val_X),   torch.LongTensor(val_y))
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False)

    head = nn.Sequential(nn.Dropout(DROPOUT), nn.Linear(512, NUM_CLASSES)).to(DEVICE)
    optimizer = optim.Adam(head.parameters(), lr=1e-4, weight_decay=3e-4)
    criterion = nn.CrossEntropyLoss()

    best_f1 = 0.0
    best_state = None
    best_epoch = 0
    best_acc = 0.0

    for epoch in range(num_epochs):
        head.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(head(X_batch), y_batch)
            loss.backward()
            optimizer.step()

        head.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                out = head(X_batch.to(DEVICE))
                all_preds.extend(out.argmax(1).cpu().numpy())
                all_labels.extend(y_batch.numpy())

        val_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        val_acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_acc = val_acc
            best_epoch = epoch + 1
            best_state = copy.deepcopy(head.state_dict())

    head.load_state_dict(best_state)
    return head, best_f1, best_acc, best_epoch


def predict(head, features):
    ds = TensorDataset(torch.FloatTensor(features))
    loader = DataLoader(ds, batch_size=512, shuffle=False)
    preds = []
    with torch.no_grad():
        for (X_batch,) in loader:
            out = head(X_batch.to(DEVICE))
            preds.extend(out.argmax(1).cpu().numpy())
    return np.array(preds)


def balanced_eval(preds, labels):
    rng = np.random.RandomState(0)
    macro_scores = []
    class_scores = {c: [] for c in CLASS_NAMES}

    for _ in range(N_DRAWS):
        indices = []
        for c in range(NUM_CLASSES):
            cls_idx = np.where(labels == c)[0]
            chosen = rng.choice(cls_idx, size=min(SAMPLES_PER_CLASS, len(cls_idx)), replace=False)
            indices.extend(chosen)
        indices = np.array(indices)
        y_true = labels[indices]
        y_pred = preds[indices]

        macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
        macro_scores.append(macro)
        per_class = f1_score(y_true, y_pred, average=None, labels=list(range(NUM_CLASSES)), zero_division=0)
        for ci, cname in enumerate(CLASS_NAMES):
            class_scores[cname].append(per_class[ci])

    macro_scores = np.array(macro_scores)
    class_scores = {k: np.array(v) for k, v in class_scores.items()}
    return macro_scores, class_scores


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("START — Stage 2: SMOTE on Center Loss Features")
    print("=" * 70)
    print(f"Device     : {DEVICE}")
    print(f"Checkpoint : {CHECKPOINT}")
    print()

    if not os.path.exists(CHECKPOINT):
        print(f"ERROR: Stage 1 checkpoint not found at {CHECKPOINT}")
        print("Run Stage 1 first (train_centerloss.py)")
        sys.exit(1)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load model
    model = build_model()
    state = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model = model.to(DEVICE)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    print("Model loaded and frozen.")

    # Extract features
    train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=clean_transforms)
    val_dataset   = datasets.ImageFolder(os.path.join(DATA_DIR, "test"),  transform=val_transforms)
    train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    val_loader    = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    print("Extracting train features...")
    train_feats, train_labels = extract_features(model, train_loader)
    print(f"  Train: {train_feats.shape}, dist: {dict(Counter(train_labels.tolist()))}")

    print("Extracting val features...")
    val_feats, val_labels = extract_features(model, val_loader)
    print(f"  Val: {val_feats.shape}")

    # SVMSMOTE with full balancing
    print("\nApplying SVMSMOTE (full balancing)...")
    counts = Counter(train_labels.tolist())
    majority = max(counts.values())
    targets = {c: majority for c in counts if counts[c] < majority}
    k = min(5, min(counts.values()) - 1)

    sampler = SVMSMOTE(sampling_strategy=targets, random_state=SEED, k_neighbors=k)
    X_res, y_res = sampler.fit_resample(train_feats, train_labels)
    print(f"  Resampled: {X_res.shape[0]} samples")
    print(f"  New dist: {dict(Counter(y_res.tolist()))}")

    # Train linear head
    print("\nTraining linear head on resampled features...")
    head, best_f1, best_acc, best_epoch = train_head(X_res, y_res, val_feats, val_labels)
    print(f"  Best head epoch: {best_epoch}, F1={best_f1:.4f}, Acc={best_acc:.4f}")

    # Imbalanced eval
    val_preds = predict(head, val_feats)
    imbal_f1 = f1_score(val_labels, val_preds, average="macro", zero_division=0)
    imbal_acc = (val_preds == val_labels).mean()
    imbal_per_class = f1_score(val_labels, val_preds, average=None,
                                labels=list(range(NUM_CLASSES)), zero_division=0)

    print(f"\nImbalanced eval: F1={imbal_f1:.4f}, Acc={imbal_acc:.4f}")
    for i, c in enumerate(CLASS_NAMES):
        print(f"  {c:<10s}: {imbal_per_class[i]:.4f}")

    # Balanced eval
    print(f"\nBalanced eval ({N_DRAWS} draws, {SAMPLES_PER_CLASS}/class)...")
    bal_macros, bal_class = balanced_eval(val_preds, val_labels)
    bal_mean = bal_macros.mean()
    bal_ci = 1.96 * bal_macros.std() / np.sqrt(len(bal_macros))
    print(f"  Balanced F1: {bal_mean:.4f} +/- {bal_ci:.4f}")

    # Save results
    smote_results = {
        "method": "SVMSMOTE_full_on_centerloss",
        "seed": SEED,
        "head_best_epoch": best_epoch,
        "imbalanced_macro_f1": round(float(imbal_f1), 4),
        "imbalanced_accuracy": round(float(imbal_acc), 4),
        "imbalanced_per_class_f1": {CLASS_NAMES[i]: round(float(imbal_per_class[i]), 4) for i in range(NUM_CLASSES)},
        "balanced_macro_f1_mean": round(float(bal_mean), 4),
        "balanced_macro_f1_ci95": round(float(bal_ci), 4),
        "balanced_per_class_f1": {
            c: {
                "mean": round(float(bal_class[c].mean()), 4),
                "ci95": round(float(1.96 * bal_class[c].std() / np.sqrt(len(bal_class[c]))), 4),
            }
            for c in CLASS_NAMES
        },
        "resampled_size": int(len(y_res)),
    }

    with open(os.path.join(RESULTS_DIR, "smote_results.json"), "w") as f:
        json.dump(smote_results, f, indent=2)

    balanced_detail = {
        "n_draws": N_DRAWS,
        "samples_per_class": SAMPLES_PER_CLASS,
        "macro_f1_mean": round(float(bal_mean), 4),
        "macro_f1_ci95": round(float(bal_ci), 4),
        "per_class": {
            c: {
                "mean": round(float(bal_class[c].mean()), 4),
                "ci95": round(float(1.96 * bal_class[c].std() / np.sqrt(len(bal_class[c]))), 4),
            }
            for c in CLASS_NAMES
        },
    }
    with open(os.path.join(RESULTS_DIR, "balanced_eval.json"), "w") as f:
        json.dump(balanced_detail, f, indent=2)

    # ── Center Loss no-SMOTE control ─────────────────────────────────────────

    print("\n--- Center Loss backbone (no SMOTE) control ---")
    print("Training linear head on ORIGINAL (unbalanced) center loss features...")
    head_nosm, nosm_f1, nosm_acc, nosm_epoch = train_head(
        train_feats, train_labels, val_feats, val_labels)
    print(f"  Best head epoch: {nosm_epoch}, F1={nosm_f1:.4f}, Acc={nosm_acc:.4f}")

    nosm_preds = predict(head_nosm, val_feats)
    nosm_imbal_f1 = f1_score(val_labels, nosm_preds, average="macro", zero_division=0)
    nosm_imbal_acc = (nosm_preds == val_labels).mean()
    nosm_imbal_pc = f1_score(val_labels, nosm_preds, average=None,
                              labels=list(range(NUM_CLASSES)), zero_division=0)

    print(f"  Imbalanced eval: F1={nosm_imbal_f1:.4f}, Acc={nosm_imbal_acc:.4f}")

    nosm_bal_macros, nosm_bal_class = balanced_eval(nosm_preds, val_labels)
    nosm_bal_mean = nosm_bal_macros.mean()
    nosm_bal_ci = 1.96 * nosm_bal_macros.std() / np.sqrt(len(nosm_bal_macros))
    print(f"  Balanced F1: {nosm_bal_mean:.4f} +/- {nosm_bal_ci:.4f}")

    nosm_results = {
        "method": "centerloss_no_smote",
        "seed": SEED,
        "head_best_epoch": nosm_epoch,
        "imbalanced_macro_f1": round(float(nosm_imbal_f1), 4),
        "imbalanced_accuracy": round(float(nosm_imbal_acc), 4),
        "imbalanced_per_class_f1": {CLASS_NAMES[i]: round(float(nosm_imbal_pc[i]), 4) for i in range(NUM_CLASSES)},
        "balanced_macro_f1_mean": round(float(nosm_bal_mean), 4),
        "balanced_macro_f1_ci95": round(float(nosm_bal_ci), 4),
        "balanced_per_class_f1": {
            c: {
                "mean": round(float(nosm_bal_class[c].mean()), 4),
                "ci95": round(float(1.96 * nosm_bal_class[c].std() / np.sqrt(len(nosm_bal_class[c]))), 4),
            }
            for c in CLASS_NAMES
        },
    }
    with open(os.path.join(RESULTS_DIR, "centerloss_no_smote_results.json"), "w") as f:
        json.dump(nosm_results, f, indent=2)

    # ── Comparison table ──────────────────────────────────────────────────────

    ref = {
        "Frozen baseline": {
            "imbal": 0.6440, "bal": 0.6301,
            "angry": 0.6194, "disgust": 0.5758, "fear": 0.4887,
            "happy": 0.8820, "neutral": 0.5771, "sad": 0.5377, "surprise": 0.7305,
        },
        "SMOTE on frozen features": {
            "imbal": 0.6367, "bal": 0.6647,
            "angry": 0.6131, "disgust": 0.6337, "fear": 0.5155,
            "happy": 0.8677, "neutral": 0.5596, "sad": 0.5230, "surprise": 0.7402,
        },
    }

    print("\n" + "=" * 130)
    print("COMPARISON TABLE (imbalanced per-class F1 | balanced macro F1 with 95% CI)")
    print("=" * 130)
    header = f"{'Config':<32s} {'Imbal F1':>9s} {'Bal F1':>14s}"
    for c in CLASS_NAMES:
        header += f" {c[:7]:>8s}"
    print(header)
    print("-" * len(header))

    for name, vals in ref.items():
        row = f"{name:<32s} {vals['imbal']:>9.4f} {vals['bal']:>9.4f}       "
        for c in CLASS_NAMES:
            row += f" {vals[c]:>8.4f}"
        print(row)

    row = f"{'CenterLoss backbone (no SMOTE)':<32s} {nosm_imbal_f1:>9.4f} {nosm_bal_mean:.4f}+/-{nosm_bal_ci:.4f}"
    for c in CLASS_NAMES:
        row += f" {nosm_results['imbalanced_per_class_f1'][c]:>8.4f}"
    print(row)

    row = f"{'CenterLoss + SMOTE':<32s} {imbal_f1:>9.4f} {bal_mean:.4f}+/-{bal_ci:.4f}"
    for c in CLASS_NAMES:
        row += f" {smote_results['imbalanced_per_class_f1'][c]:>8.4f}"
    print(row)
    print("-" * len(header))

    # Per-class balanced F1 with CIs
    print(f"\nBALANCED PER-CLASS F1 (100 draws, 55/class, mean +/- 95% CI):")
    print(f"{'Class':<10s} {'CL no SMOTE':>18s} {'CL + SMOTE':>18s} {'Delta':>10s}")
    print("-" * 60)
    for c in CLASS_NAMES:
        ns_m = nosm_bal_class[c].mean()
        ns_ci = 1.96 * nosm_bal_class[c].std() / np.sqrt(len(nosm_bal_class[c]))
        sm_m = bal_class[c].mean()
        sm_ci = 1.96 * bal_class[c].std() / np.sqrt(len(bal_class[c]))
        delta = sm_m - ns_m
        sign = "+" if delta >= 0 else ""
        print(f"{c:<10s} {ns_m:.4f}+/-{ns_ci:.4f}   {sm_m:.4f}+/-{sm_ci:.4f}   {sign}{delta:.4f}")
    ns_macro_m = nosm_bal_mean
    sm_macro_m = bal_mean
    print(f"{'MACRO':<10s} {ns_macro_m:.4f}+/-{nosm_bal_ci:.4f}   {sm_macro_m:.4f}+/-{bal_ci:.4f}   {'+' if sm_macro_m >= ns_macro_m else ''}{sm_macro_m - ns_macro_m:.4f}")

    print("\n" + "=" * 70)
    print("END — Stage 2: SMOTE on Center Loss Features")
    print("=" * 70)


if __name__ == "__main__":
    main()
