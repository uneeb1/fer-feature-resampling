"""Run only the class-weighted loss experiments + generate summary."""
import os, sys, json, time, copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from collections import Counter

DATA_DIR = "./data"
OUT_DIR = "./experiments"
WEIGHTED_DIR = os.path.join(OUT_DIR, "weighted_loss")
SMOTE_DIR = os.path.join(OUT_DIR, "smote_results")
NUM_CLASSES = 7
BATCH_SIZE = 64
CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
SEEDS = [42, 123, 456]
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

def log(msg):
    print(msg, flush=True)

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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
log(f"Device: {DEVICE}")
log(f"Class weights: {dict(zip(CLASS_NAMES, [f'{w:.3f}' for w in weights]))}")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

for seed in SEEDS:
    fname = f"weighted_{seed}.json"
    fpath = os.path.join(WEIGHTED_DIR, fname)
    if os.path.exists(fpath):
        log(f"SKIP {fname}")
        continue

    log(f"\n{'='*50}")
    log(f"Class-weighted loss / seed={seed}")
    log(f"{'='*50}")
    set_seed(seed)

    model = models.resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, NUM_CLASSES))
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
    best_acc = 0.0
    best_report = None
    best_cm = None
    patience_counter = 0

    for epoch in range(30):
        t0 = time.time()
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
        dt = time.time() - t0
        log(f"  Epoch {epoch+1}/30  Val F1={val_f1:.4f}  Acc={val_acc:.4f}  ({dt:.1f}s)")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch + 1
            patience_counter = 0
            best_acc = val_acc
            best_report = classification_report(
                all_labels, all_preds, target_names=CLASS_NAMES,
                zero_division=0, output_dict=True)
            best_cm = confusion_matrix(all_labels, all_preds).tolist()
        else:
            patience_counter += 1
            if patience_counter >= 5:
                log(f"  Early stopping at epoch {epoch+1}")
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
    log(f"  SAVED: {fname}  Best F1={best_f1:.4f} at epoch {best_epoch}")

# Generate summary from all results
log("\n" + "=" * 60)
log("GENERATING SUMMARY")
log("=" * 60)

SMOTE_VARIANTS = ["SMOTE", "BorderlineSMOTE", "SVMSMOTE", "ADASYN", "SMOTETomek"]
rows = [{"Method": "Baseline", "Strategy": "—", "Mean F1": 0.6440, "Std": "—", "Mean Disgust F1": 0.53, "Mean Fear F1": 0.49}]

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
                "Mean F1": round(np.mean(f1s), 4), "Std": round(np.std(f1s), 4),
                "Mean Disgust F1": round(np.mean(disgust_f1s), 4),
                "Mean Fear F1": round(np.mean(fear_f1s), 4),
            })

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
        "Mean F1": round(np.mean(f1s), 4), "Std": round(np.std(f1s), 4),
        "Mean Disgust F1": round(np.mean(disgust_f1s), 4),
        "Mean Fear F1": round(np.mean(fear_f1s), 4),
    })

header = f"{'Method':<18} {'Strategy':<15} {'Mean F1':>8} {'Std':>7} {'Disgust F1':>11} {'Fear F1':>9}"
sep = "-" * len(header)
lines = [sep, header, sep]
for r in rows:
    std_str = f"{r['Std']:.4f}" if isinstance(r["Std"], float) else r["Std"]
    lines.append(f"{r['Method']:<18} {r['Strategy']:<15} {r['Mean F1']:>8.4f} {std_str:>7} {r['Mean Disgust F1']:>11.4f} {r['Mean Fear F1']:>9.4f}")
lines.append(sep)
summary_txt = "\n".join(lines)
log("\n" + summary_txt)

with open(os.path.join(OUT_DIR, "summary.txt"), "w") as f:
    f.write(summary_txt + "\n")
import csv
with open(os.path.join(OUT_DIR, "summary.csv"), "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

log(f"\nSaved: {os.path.join(OUT_DIR, 'summary.txt')}")
log(f"Saved: {os.path.join(OUT_DIR, 'summary.csv')}")
log("DONE")
