"""
Stage 1: Full Center Loss Fine-Tuning on ResNet-18/ImageNet.
Loss = CE + λ*CenterLoss (λ=0.003), learnable class centers.
Hyperparameters match Manifold Mixup Config 1 runs exactly.
"""

import os
import sys
import time
import copy
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.metrics import f1_score, classification_report
from sklearn.metrics import silhouette_score, silhouette_samples

# ── Config (matches Manifold Mixup Config 1) ─────────────────────────────────

NUM_CLASSES     = 7
CLASS_NAMES     = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
BATCH_SIZE      = 64
NUM_EPOCHS      = 30
BACKBONE_LR     = 3e-5
HEAD_LR         = 1e-4
WEIGHT_DECAY    = 3e-4
DROPOUT         = 0.3
PATIENCE        = 7
NUM_WORKERS     = 4
SEED            = 42

CENTER_LOSS_LAMBDA = 0.003
CENTER_LR          = 0.5

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
BASE_DIR    = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
DATA_DIR    = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "results", "stage1_centerloss"))

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

# ── Center Loss ───────────────────────────────────────────────────────────────

class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, features, labels):
        batch_centers = self.centers[labels]
        return ((features - batch_centers) ** 2).sum(dim=1).mean()

# ── Transforms (identical to baseline/mixup) ──────────────────────────────────

train_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ── Feature extraction via hook ───────────────────────────────────────────────

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


def per_class_silhouette(features, labels):
    overall = silhouette_score(features, labels)
    sample_scores = silhouette_samples(features, labels)
    per_class = {}
    for c in range(NUM_CLASSES):
        mask = labels == c
        if mask.sum() > 0:
            per_class[CLASS_NAMES[c]] = float(np.mean(sample_scores[mask]))
    return overall, per_class

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("START — Stage 1: Center Loss Fine-Tuning")
    print("=" * 70)
    print(f"Device       : {DEVICE}")
    print(f"Seed         : {SEED}")
    print(f"Center λ     : {CENTER_LOSS_LAMBDA}")
    print(f"Center LR    : {CENTER_LR}")
    print(f"Epochs       : {NUM_EPOCHS}")
    print(f"Patience     : {PATIENCE}")
    print(f"Data dir     : {DATA_DIR}")
    print(f"Results dir  : {RESULTS_DIR}")
    print()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Data
    train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transforms)
    val_dataset   = datasets.ImageFolder(os.path.join(DATA_DIR, "test"),  transform=val_transforms)
    train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
    val_loader    = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples  : {len(val_dataset)}")

    # Model — fresh ImageNet-pretrained ResNet-18
    model = models.resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Sequential(
        nn.Dropout(DROPOUT),
        nn.Linear(model.fc.in_features, NUM_CLASSES)
    )
    model = model.to(DEVICE)

    # Center loss module
    center_loss_fn = CenterLoss(NUM_CLASSES, 512).to(DEVICE)

    # Optimizers
    backbone_params = [p for n, p in model.named_parameters() if "fc" not in n]
    head_params     = [p for n, p in model.named_parameters() if "fc" in n]
    optimizer = optim.Adam([
        {"params": backbone_params, "lr": BACKBONE_LR},
        {"params": head_params,     "lr": HEAD_LR},
    ], weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    center_optimizer = optim.SGD(center_loss_fn.parameters(), lr=CENTER_LR)

    criterion = nn.CrossEntropyLoss()

    # Hook for extracting features during training
    hook_output = {}
    def hook_fn(module, inp, out):
        hook_output["feat"] = out

    # Training
    best_f1 = 0.0
    best_wts = None
    best_epoch = 0
    patience_ctr = 0
    training_log = []

    for epoch in range(NUM_EPOCHS):
        t0 = time.time()
        model.train()
        running_ce = 0.0
        running_center = 0.0
        n_samples = 0
        all_preds, all_labels = [], []

        handle = model.avgpool.register_forward_hook(hook_fn)

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            center_optimizer.zero_grad()

            outputs = model(inputs)
            ce_loss = criterion(outputs, labels)

            feats = hook_output["feat"].squeeze(-1).squeeze(-1)
            c_loss = center_loss_fn(feats, labels)
            loss = ce_loss + CENTER_LOSS_LAMBDA * c_loss

            loss.backward()
            optimizer.step()
            center_optimizer.step()

            bs = inputs.size(0)
            running_ce += ce_loss.item() * bs
            running_center += c_loss.item() * bs
            n_samples += bs
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        handle.remove()

        train_ce = running_ce / n_samples
        train_center = running_center / n_samples
        train_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

        # Validation
        model.eval()
        val_preds, val_labels_list = [], []
        val_loss_sum = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                val_loss_sum += criterion(outputs, labels).item() * inputs.size(0)
                val_preds.extend(outputs.argmax(1).cpu().numpy())
                val_labels_list.extend(labels.cpu().numpy())

        val_preds = np.array(val_preds)
        val_labels_arr = np.array(val_labels_list)
        val_f1 = f1_score(val_labels_arr, val_preds, average="macro", zero_division=0)
        val_acc = (val_preds == val_labels_arr).mean()
        val_per_class_f1 = f1_score(val_labels_arr, val_preds, average=None,
                                     labels=list(range(NUM_CLASSES)), zero_division=0)

        # Per-class silhouette on val features
        val_feats, val_labs = extract_features(model, val_loader)
        overall_sil, cls_sil = per_class_silhouette(val_feats, val_labs)

        scheduler.step()
        elapsed = time.time() - t0

        epoch_log = {
            "epoch": epoch + 1,
            "train_ce_loss": round(train_ce, 4),
            "train_center_loss": round(train_center, 4),
            "train_total_loss": round(train_ce + CENTER_LOSS_LAMBDA * train_center, 4),
            "train_f1": round(float(train_f1), 4),
            "val_macro_f1": round(float(val_f1), 4),
            "val_accuracy": round(float(val_acc), 4),
            "val_per_class_f1": {CLASS_NAMES[i]: round(float(val_per_class_f1[i]), 4) for i in range(NUM_CLASSES)},
            "overall_silhouette": round(overall_sil, 4),
            "per_class_silhouette": cls_sil,
            "time_s": round(elapsed, 1),
        }
        training_log.append(epoch_log)

        # Print
        print(f"Epoch {epoch+1:2d}/{NUM_EPOCHS} | "
              f"CE={train_ce:.4f} Center={train_center:.4f} | "
              f"val_F1={val_f1:.4f} acc={val_acc:.4f} sil={overall_sil:.4f} | "
              f"{elapsed:.0f}s", end="")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch + 1
            patience_ctr = 0
            best_wts = copy.deepcopy(model.state_dict())
            torch.save(best_wts, os.path.join(RESULTS_DIR, "best_model.pth"))
            print(f" * BEST")
        else:
            patience_ctr += 1
            print(f"  (pat {patience_ctr}/{PATIENCE})")
            if patience_ctr >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    # Save training log
    with open(os.path.join(RESULTS_DIR, "training_log.json"), "w") as f:
        json.dump(training_log, f, indent=2)

    # ── Final evaluation at best epoch ────────────────────────────────────────

    model.load_state_dict(best_wts)
    model.eval()

    # Get best epoch's log entry
    best_log = training_log[best_epoch - 1]

    # Recompute silhouette on best model's features
    val_feats, val_labs = extract_features(model, val_loader)
    overall_sil, cls_sil = per_class_silhouette(val_feats, val_labs)

    final_sil = {
        "overall": round(overall_sil, 4),
        "per_class": {k: round(v, 4) for k, v in cls_sil.items()},
    }
    with open(os.path.join(RESULTS_DIR, "final_silhouette.json"), "w") as f:
        json.dump(final_sil, f, indent=2)

    # ── Summary table ─────────────────────────────────────────────────────────

    frozen_sil = {
        "overall": 0.0380,
        "angry": -0.0369, "disgust": 0.0644, "fear": -0.0658,
        "happy": 0.1261, "neutral": 0.0548, "sad": -0.0100, "surprise": 0.1068,
    }

    print("\n" + "=" * 70)
    print("STAGE 1 SUMMARY — Center Loss Fine-Tuning")
    print("=" * 70)
    print(f"Best epoch     : {best_epoch}")
    print(f"Best val F1    : {best_log['val_macro_f1']:.4f}")
    print(f"Val accuracy   : {best_log['val_accuracy']:.4f}")

    print(f"\nPer-class F1 at best epoch:")
    for c in CLASS_NAMES:
        print(f"  {c:<10s}: {best_log['val_per_class_f1'][c]:.4f}")

    print(f"\nPer-class silhouette (best model vs frozen baseline):")
    header = f"  {'Class':<10s} {'Frozen':>8s} {'CenterLoss':>12s} {'Delta':>8s}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for c in CLASS_NAMES:
        frozen_val = frozen_sil[c]
        cl_val = cls_sil.get(c, 0.0)
        delta = cl_val - frozen_val
        marker = ""
        if c == "fear":
            marker = "  <-- WATCH: fear was -0.066"
        print(f"  {c:<10s} {frozen_val:>8.4f} {cl_val:>12.4f} {delta:>+8.4f}{marker}")

    frozen_ov = frozen_sil["overall"]
    print(f"  {'OVERALL':<10s} {frozen_ov:>8.4f} {overall_sil:>12.4f} {overall_sil - frozen_ov:>+8.4f}")

    fear_sil = cls_sil.get("fear", 0.0)
    print()
    if fear_sil < 0:
        print(f"NOTE: Fear silhouette is STILL negative ({fear_sil:.4f}) after full training.")
        print("This is a research finding — center loss alone cannot separate fear from its neighbors.")
    else:
        print(f"Fear silhouette improved to {fear_sil:.4f} (was -0.066). Center loss helped separation.")

    print("\n" + "=" * 70)
    print("END — Stage 1: Center Loss Fine-Tuning")
    print("=" * 70)


if __name__ == "__main__":
    main()
