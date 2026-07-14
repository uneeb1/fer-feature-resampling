"""
Systematic SMOTE experiment runner for FER2013 thesis.
Feature extraction -> SMOTE variants x strategies x seeds -> class-weighted comparison.
"""

import os
import sys
import json
import time
import copy
import traceback
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, models, transforms
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from collections import Counter

from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE, ADASYN
from imblearn.combine import SMOTETomek

# ── Config ───────────────────────────────────────────────────────────────────

DATA_DIR = "./data"
BASELINE_CKPT = "./experiments/exp_03/best_model.pth"
OUT_DIR = "./experiments"
FEATURES_DIR = os.path.join(OUT_DIR, "features")
SMOTE_DIR = os.path.join(OUT_DIR, "smote_results")
WEIGHTED_DIR = os.path.join(OUT_DIR, "weighted_loss")

NUM_CLASSES = 7
BATCH_SIZE = 64
DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
SEEDS = [42, 123, 456]

SMOTE_VARIANTS = {
    "SMOTE": SMOTE,
    "BorderlineSMOTE": BorderlineSMOTE,
    "SVMSMOTE": SVMSMOTE,
    "ADASYN": ADASYN,
    "SMOTETomek": SMOTETomek,
}

# ── Helpers ──────────────────────────────────────────────────────────────────

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_backbone():
    model = models.resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, NUM_CLASSES))
    return model


def compute_strategy_targets(labels, strategy):
    """Return sampling_strategy dict for imblearn."""
    counts = Counter(labels)
    if strategy == "full":
        majority = max(counts.values())
        return {c: majority for c in counts if counts[c] < majority}

    elif strategy == "median":
        median_count = int(np.median(list(counts.values())))
        return {c: median_count for c in counts if counts[c] < median_count}

    elif strategy == "minority_only":
        minority_classes = [CLASS_NAMES.index(n) for n in ["disgust", "fear", "surprise"]]
        non_minority = {c: v for c, v in counts.items() if c not in minority_classes}
        target = min(non_minority.values()) if non_minority else max(counts.values())
        return {c: target for c in minority_classes if counts[c] < target}

    raise ValueError(f"Unknown strategy: {strategy}")


# ── Step 1: Feature Extraction ───────────────────────────────────────────────

def extract_features():
    os.makedirs(FEATURES_DIR, exist_ok=True)
    train_f = os.path.join(FEATURES_DIR, "train_features.npy")
    train_l = os.path.join(FEATURES_DIR, "train_labels.npy")
    val_f = os.path.join(FEATURES_DIR, "val_features.npy")
    val_l = os.path.join(FEATURES_DIR, "val_labels.npy")

    if all(os.path.exists(p) for p in [train_f, train_l, val_f, val_l]):
        print("Feature files exist, loading from cache...")
        return (np.load(train_f), np.load(train_l),
                np.load(val_f), np.load(val_l))

    print("Extracting features from frozen backbone...")
    clean_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    model = build_backbone()
    state = torch.load(BASELINE_CKPT, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model = model.to(DEVICE)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    hook_output = []
    def hook_fn(module, input, output):
        hook_output.append(output.squeeze().cpu().numpy())
    handle = model.avgpool.register_forward_hook(hook_fn)

    for split, path in [("train", "train"), ("val", "test")]:
        ds = datasets.ImageFolder(os.path.join(DATA_DIR, path), transform=clean_transform)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        all_features, all_labels = [], []
        for i, (imgs, labs) in enumerate(loader):
            hook_output.clear()
            with torch.no_grad():
                model(imgs.to(DEVICE))
            all_features.append(hook_output[0] if hook_output[0].ndim == 2
                                else hook_output[0].reshape(-1, 512))
            all_labels.append(labs.numpy())
            if (i + 1) % 50 == 0:
                print(f"  {split}: {(i+1)*BATCH_SIZE}/{len(ds)} samples")

        features = np.concatenate(all_features)
        labels = np.concatenate(all_labels)
        feat_path = train_f if split == "train" else val_f
        lab_path = train_l if split == "train" else val_l
        np.save(feat_path, features)
        np.save(lab_path, labels)
        print(f"  {split}: saved {features.shape}")

    handle.remove()
    return (np.load(train_f), np.load(train_l),
            np.load(val_f), np.load(val_l))


# ── Step 2: SMOTE Head Training ──────────────────────────────────────────────

def train_head(train_X, train_y, val_X, val_y, seed, num_epochs=50):
    set_seed(seed)
    train_ds = TensorDataset(torch.FloatTensor(train_X), torch.LongTensor(train_y))
    val_ds = TensorDataset(torch.FloatTensor(val_X), torch.LongTensor(val_y))
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    head = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, NUM_CLASSES)).to(DEVICE)
    optimizer = optim.Adam(head.parameters(), lr=1e-4, weight_decay=3e-4)
    criterion = nn.CrossEntropyLoss()

    best_f1 = 0.0
    best_epoch = 0
    best_state = None
    best_report = None
    best_cm = None
    best_acc = 0.0

    for epoch in range(num_epochs):
        head.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            out = head(X_batch)
            loss = criterion(out, y_batch)
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
            best_report = classification_report(
                all_labels, all_preds, target_names=CLASS_NAMES,
                zero_division=0, output_dict=True)
            best_cm = confusion_matrix(all_labels, all_preds).tolist()

    return {
        "val_macro_f1": round(best_f1, 4),
        "val_accuracy": round(best_acc, 4),
        "best_epoch": best_epoch,
        "classification_report": best_report,
        "confusion_matrix": best_cm,
    }


def run_smote_experiments(train_features, train_labels, val_features, val_labels):
    os.makedirs(SMOTE_DIR, exist_ok=True)
    strategies = ["full", "median", "minority_only"]
    total = len(SMOTE_VARIANTS) * len(strategies) * len(SEEDS)
    done = 0

    for variant_name, variant_cls in SMOTE_VARIANTS.items():
        for strategy in strategies:
            for seed in SEEDS:
                done += 1
                fname = f"{variant_name}_{strategy}_{seed}.json"
                fpath = os.path.join(SMOTE_DIR, fname)
                if os.path.exists(fpath):
                    print(f"[{done}/{total}] SKIP {fname} (exists)")
                    continue

                print(f"[{done}/{total}] {variant_name} / {strategy} / seed={seed} ...", end=" ", flush=True)
                t0 = time.time()

                try:
                    targets = compute_strategy_targets(train_labels.tolist(), strategy)
                    if not targets:
                        print("no resampling needed, skipping")
                        continue

                    set_seed(seed)
                    if variant_name == "SMOTETomek":
                        sampler = variant_cls(
                            smote=SMOTE(sampling_strategy=targets, random_state=seed, k_neighbors=min(5, min(Counter(train_labels).values()) - 1)),
                            random_state=seed)
                    else:
                        k = min(5, min(Counter(train_labels).values()) - 1)
                        if variant_name == "ADASYN":
                            sampler = variant_cls(sampling_strategy=targets, random_state=seed, n_neighbors=k)
                        else:
                            sampler = variant_cls(sampling_strategy=targets, random_state=seed, k_neighbors=k)

                    X_res, y_res = sampler.fit_resample(train_features, train_labels)
                    result = train_head(X_res, y_res, val_features, val_labels, seed)
                    result["variant"] = variant_name
                    result["strategy"] = strategy
                    result["seed"] = seed
                    result["resampled_size"] = len(y_res)
                    result["resampled_dist"] = dict(Counter(y_res.tolist()))

                    with open(fpath, "w") as f:
                        json.dump(result, f, indent=2)

                    dt = time.time() - t0
                    print(f"F1={result['val_macro_f1']:.4f} ({dt:.1f}s)")

                except Exception as e:
                    dt = time.time() - t0
                    print(f"ERROR ({dt:.1f}s): {e}")
                    traceback.print_exc()


# ── Step 3: Class-Weighted Loss (full fine-tune) ─────────────────────────────

def run_weighted_loss():
    os.makedirs(WEIGHTED_DIR, exist_ok=True)

    train_transforms_aug = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transforms_aug)
    val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=val_transform)

    class_counts = np.array([0] * NUM_CLASSES)
    for _, label in train_dataset.samples:
        class_counts[label] += 1
    weights = 1.0 / class_counts.astype(float)
    weights = weights / weights.sum() * NUM_CLASSES
    class_weights = torch.FloatTensor(weights).to(DEVICE)
    print(f"Class weights: {dict(zip(CLASS_NAMES, [f'{w:.3f}' for w in weights]))}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    for seed in SEEDS:
        fname = f"weighted_{seed}.json"
        fpath = os.path.join(WEIGHTED_DIR, fname)
        if os.path.exists(fpath):
            print(f"SKIP {fname} (exists)")
            continue

        print(f"\nClass-weighted loss / seed={seed}")
        set_seed(seed)

        model = build_backbone()
        model = model.to(DEVICE)

        backbone_params = [p for n, p in model.named_parameters() if "fc" not in n]
        head_params = [p for n, p in model.named_parameters() if "fc" in n]
        optimizer = optim.Adam([
            {"params": backbone_params, "lr": 3e-5},
            {"params": head_params, "lr": 1e-4},
        ], weight_decay=3e-4)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

        best_f1 = 0.0
        best_epoch = 0
        best_state = None
        patience_counter = 0

        for epoch in range(30):
            model.train()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                out = model(inputs)
                loss = criterion(out, labels)
                loss.backward()
                optimizer.step()

            scheduler.step()

            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for inputs, labels in val_loader:
                    out = model(inputs.to(DEVICE))
                    all_preds.extend(out.argmax(1).cpu().numpy())
                    all_labels.extend(labels.numpy())

            val_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
            val_acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
            print(f"  Epoch {epoch+1}/30  Val F1={val_f1:.4f}  Acc={val_acc:.4f}")

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_epoch = epoch + 1
                patience_counter = 0
                best_state = copy.deepcopy(model.state_dict())
                best_report = classification_report(
                    all_labels, all_preds, target_names=CLASS_NAMES,
                    zero_division=0, output_dict=True)
                best_cm = confusion_matrix(all_labels, all_preds).tolist()
                best_acc = val_acc
            else:
                patience_counter += 1
                if patience_counter >= 5:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        result = {
            "method": "ClassWeighted",
            "seed": seed,
            "val_macro_f1": round(best_f1, 4),
            "val_accuracy": round(best_acc, 4),
            "best_epoch": best_epoch,
            "classification_report": best_report,
            "confusion_matrix": best_cm,
        }
        with open(fpath, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Best F1={best_f1:.4f} at epoch {best_epoch}")


# ── Step 4: Summary ──────────────────────────────────────────────────────────

def generate_summary():
    rows = []

    # Baseline row
    rows.append({
        "Method": "Baseline", "Strategy": "—",
        "Mean F1": 0.6440, "Std": "—",
        "Mean Disgust F1": 0.53, "Mean Fear F1": 0.49,
    })

    # SMOTE results
    for variant in SMOTE_VARIANTS:
        for strategy in ["full", "median", "minority_only"]:
            f1s, disgust_f1s, fear_f1s = [], [], []
            for seed in SEEDS:
                fpath = os.path.join(SMOTE_DIR, f"{variant}_{strategy}_{seed}.json")
                if not os.path.exists(fpath):
                    continue
                with open(fpath) as f:
                    r = json.load(f)
                f1s.append(r["val_macro_f1"])
                cr = r["classification_report"]
                disgust_f1s.append(cr["disgust"]["f1-score"])
                fear_f1s.append(cr["fear"]["f1-score"])

            if f1s:
                rows.append({
                    "Method": variant, "Strategy": strategy,
                    "Mean F1": round(np.mean(f1s), 4),
                    "Std": round(np.std(f1s), 4),
                    "Mean Disgust F1": round(np.mean(disgust_f1s), 4),
                    "Mean Fear F1": round(np.mean(fear_f1s), 4),
                })

    # Weighted loss
    f1s, disgust_f1s, fear_f1s = [], [], []
    for seed in SEEDS:
        fpath = os.path.join(WEIGHTED_DIR, f"weighted_{seed}.json")
        if not os.path.exists(fpath):
            continue
        with open(fpath) as f:
            r = json.load(f)
        f1s.append(r["val_macro_f1"])
        cr = r["classification_report"]
        disgust_f1s.append(cr["disgust"]["f1-score"])
        fear_f1s.append(cr["fear"]["f1-score"])

    if f1s:
        rows.append({
            "Method": "ClassWeighted", "Strategy": "—",
            "Mean F1": round(np.mean(f1s), 4),
            "Std": round(np.std(f1s), 4),
            "Mean Disgust F1": round(np.mean(disgust_f1s), 4),
            "Mean Fear F1": round(np.mean(fear_f1s), 4),
        })

    # Print and save
    header = f"{'Method':<18} {'Strategy':<15} {'Mean F1':>8} {'Std':>7} {'Disgust F1':>11} {'Fear F1':>9}"
    sep = "-" * len(header)

    lines = [sep, header, sep]
    for r in rows:
        std_str = f"{r['Std']:.4f}" if isinstance(r["Std"], float) else r["Std"]
        lines.append(
            f"{r['Method']:<18} {r['Strategy']:<15} {r['Mean F1']:>8.4f} {std_str:>7} "
            f"{r['Mean Disgust F1']:>11.4f} {r['Mean Fear F1']:>9.4f}"
        ) if isinstance(r["Std"], float) else lines.append(
            f"{r['Method']:<18} {r['Strategy']:<15} {r['Mean F1']:>8.4f} {r['Std']:>7} "
            f"{r['Mean Disgust F1']:>11.4f} {r['Mean Fear F1']:>9.4f}"
        )
    lines.append(sep)

    summary_txt = "\n".join(lines)
    print("\n" + summary_txt)

    with open(os.path.join(OUT_DIR, "summary.txt"), "w") as f:
        f.write(summary_txt + "\n")

    # CSV
    import csv
    with open(os.path.join(OUT_DIR, "summary.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved: {os.path.join(OUT_DIR, 'summary.txt')}")
    print(f"Saved: {os.path.join(OUT_DIR, 'summary.csv')}")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print(f"Seeds: {SEEDS}")
    print()

    # Step 1
    train_features, train_labels, val_features, val_labels = extract_features()
    print(f"Train features: {train_features.shape}, Val features: {val_features.shape}")
    print(f"Train class dist: {dict(Counter(train_labels.tolist()))}")
    print()

    # Step 2
    print("=" * 60)
    print("SMOTE EXPERIMENTS")
    print("=" * 60)
    run_smote_experiments(train_features, train_labels, val_features, val_labels)
    print()

    # Step 3
    print("=" * 60)
    print("CLASS-WEIGHTED LOSS EXPERIMENTS")
    print("=" * 60)
    run_weighted_loss()
    print()

    # Step 4
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    generate_summary()
