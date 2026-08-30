"""
Stage 1 v2: Full Center Loss Fine-Tuning on ResNet-18/ImageNet.
Loss = CE + λ*CenterLoss (λ=0.003), learnable class centers.
LR schedule: CosineAnnealingLR (T_max=20, eta_min=1e-6).
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

# ── Config ────────────────────────────────────────────────────────────────────

NUM_CLASSES     = 7
CLASS_NAMES     = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
BATCH_SIZE      = 64
NUM_EPOCHS      = 20
MAX_EPOCHS      = 30
BACKBONE_LR     = 3e-5
HEAD_LR         = 1e-4
WEIGHT_DECAY    = 3e-4
DROPOUT         = 0.3
PATIENCE        = 7
NUM_WORKERS     = 4
SEED            = 42

CENTER_LOSS_LAMBDA = 0.003
CENTER_LR          = 0.5
COSINE_ETA_MIN     = 1e-6

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
BASE_DIR    = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
DATA_DIR    = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "results", "stage1_centerloss_v2"))

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
    print("START — Stage 1 v2: Center Loss Fine-Tuning (CosineAnnealingLR)")
    print("=" * 70)
    print(f"Device       : {DEVICE}")
    print(f"Seed         : {SEED}")
    print(f"Center λ     : {CENTER_LOSS_LAMBDA}")
    print(f"Center LR    : {CENTER_LR}")
    print(f"Initial epochs: {NUM_EPOCHS} (extend to {MAX_EPOCHS} if F1 < 0.60)")
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

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=COSINE_ETA_MIN)

    center_optimizer = optim.SGD(center_loss_fn.parameters(), lr=CENTER_LR)

    criterion = nn.CrossEntropyLoss()

    # ── Verification ──────────────────────────────────────────────────────────
    print(f"Scheduler    : {type(scheduler).__name__}")
    print(f"  T_max      : {NUM_EPOCHS}")
    print(f"  eta_min    : {COSINE_ETA_MIN}")
    print(f"Initial LRs  : backbone={optimizer.param_groups[0]['lr']}, head={optimizer.param_groups[1]['lr']}")
    print(f"Center optim : SGD, lr={CENTER_LR} (NOT cosine-annealed)")
    assert isinstance(scheduler, optim.lr_scheduler.CosineAnnealingLR), "Scheduler is NOT CosineAnnealingLR!"
    print("VERIFIED: Scheduler is CosineAnnealingLR. Starting training.\n")

    # Header
    print(f"{'Epoch':<7} {'LR_bb':>10} {'LR_head':>10} {'CE_loss':>9} {'Ctr_loss':>10} {'Val_F1':>8} {'Val_Acc':>8} {'Sil':>7} {'Note':>10}")
    print("-" * 90)

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
    total_epochs = NUM_EPOCHS

    for epoch in range(MAX_EPOCHS):
        if epoch >= total_epochs:
            break

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
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
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

        current_lr_bb = optimizer.param_groups[0]['lr']
        current_lr_head = optimizer.param_groups[1]['lr']
        scheduler.step()
        elapsed = time.time() - t0

        epoch_log = {
            "epoch": epoch + 1,
            "lr_backbone": round(current_lr_bb, 8),
            "lr_head": round(current_lr_head, 8),
            "train_ce_loss": round(train_ce, 4),
            "train_center_loss": round(train_center, 4),
            "train_total_loss": round(train_ce + CENTER_LOSS_LAMBDA * train_center, 4),
            "train_f1": round(float(train_f1), 4),
            "val_macro_f1": round(float(val_f1), 4),
            "val_accuracy": round(float(val_acc), 4),
            "val_per_class_f1": {CLASS_NAMES[i]: round(float(val_per_class_f1[i]), 4) for i in range(NUM_CLASSES)},
            "overall_silhouette": round(overall_sil, 4),
            "per_class_silhouette": {k: round(v, 4) for k, v in cls_sil.items()},
            "time_s": round(elapsed, 1),
        }
        training_log.append(epoch_log)

        note = ""
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch + 1
            patience_ctr = 0
            best_wts = copy.deepcopy(model.state_dict())
            torch.save(best_wts, os.path.join(RESULTS_DIR, "best_model.pth"))
            note = "* BEST"
        else:
            patience_ctr += 1
            note = f"pat {patience_ctr}/{PATIENCE}"

        print(f"{epoch+1:<7d} {current_lr_bb:>10.2e} {current_lr_head:>10.2e} "
              f"{train_ce:>9.4f} {train_center:>10.2f} "
              f"{val_f1:>8.4f} {val_acc:>8.4f} {overall_sil:>7.4f} {note:>10}")

        # Save final epoch checkpoint regardless
        final_wts = copy.deepcopy(model.state_dict())
        torch.save(final_wts, os.path.join(RESULTS_DIR, "final_epoch_model.pth"))

        if patience_ctr >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

        # At epoch 20, check if we need to extend
        if epoch + 1 == NUM_EPOCHS and best_f1 < 0.60:
            print(f"\n*** Val F1 = {best_f1:.4f} < 0.60 at epoch {NUM_EPOCHS}. Extending to {MAX_EPOCHS} epochs. ***")
            total_epochs = MAX_EPOCHS
            patience_ctr = 0

    final_epoch_num = epoch + 1

    # Save training log
    with open(os.path.join(RESULTS_DIR, "training_log.json"), "w") as f:
        json.dump(training_log, f, indent=2)

    # ── Helper for printing a checkpoint summary ──────────────────────────────

    frozen_sil = {
        "overall": 0.0380,
        "angry": -0.0369, "disgust": 0.0644, "fear": -0.0658,
        "happy": 0.1261, "neutral": 0.0548, "sad": -0.0100, "surprise": 0.1068,
    }

    def print_checkpoint_summary(label, ckpt_wts, ckpt_epoch):
        model.load_state_dict(ckpt_wts)
        model.eval()

        vp, vl = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                vp.extend(outputs.argmax(1).cpu().numpy())
                vl.extend(labels.cpu().numpy())
        vp, vl = np.array(vp), np.array(vl)

        macro_f1 = f1_score(vl, vp, average="macro", zero_division=0)
        acc = (vp == vl).mean()
        pc_f1 = f1_score(vl, vp, average=None, labels=list(range(NUM_CLASSES)), zero_division=0)

        feats, labs = extract_features(model, val_loader)
        ov_sil, cs_sil = per_class_silhouette(feats, labs)

        print(f"\n{'='*70}")
        print(f"{label} (epoch {ckpt_epoch})")
        print(f"{'='*70}")
        print(f"Val macro-F1 : {macro_f1:.4f}")
        print(f"Val accuracy : {acc:.4f}")

        print(f"\nPer-class F1:")
        for i, c in enumerate(CLASS_NAMES):
            flag = "  *** COLLAPSED ***" if pc_f1[i] == 0.0 else ""
            print(f"  {c:<10s}: {pc_f1[i]:.4f}{flag}")

        print(f"\nPer-class silhouette (vs frozen baseline):")
        hdr = f"  {'Class':<10s} {'Frozen':>8s} {'CenterLoss':>12s} {'Delta':>8s}"
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for c in CLASS_NAMES:
            fv = frozen_sil[c]
            cv = cs_sil.get(c, 0.0)
            marker = "  <-- WATCH" if c == "fear" else ""
            print(f"  {c:<10s} {fv:>8.4f} {cv:>12.4f} {cv - fv:>+8.4f}{marker}")
        fov = frozen_sil["overall"]
        print(f"  {'OVERALL':<10s} {fov:>8.4f} {ov_sil:>12.4f} {ov_sil - fov:>+8.4f}")

        return ov_sil, cs_sil

    # ── Print both checkpoints ────────────────────────────────────────────────

    best_sil_ov, best_sil_cls = print_checkpoint_summary(
        "BEST CHECKPOINT (by val macro-F1)", best_wts, best_epoch)

    final_sil_ov, final_sil_cls = print_checkpoint_summary(
        "FINAL EPOCH CHECKPOINT", final_wts, final_epoch_num)

    # Save silhouette files for both
    for name, sil_ov, sil_cls in [("final_silhouette_best.json", best_sil_ov, best_sil_cls),
                                    ("final_silhouette_epoch20.json", final_sil_ov, final_sil_cls)]:
        data = {"overall": round(sil_ov, 4), "per_class": {k: round(v, 4) for k, v in sil_cls.items()}}
        with open(os.path.join(RESULTS_DIR, name), "w") as f:
            json.dump(data, f, indent=2)

    # Fear finding
    fear_best = best_sil_cls.get("fear", 0.0)
    fear_final = final_sil_cls.get("fear", 0.0)
    print()
    if fear_best < 0 and fear_final < 0:
        print(f"FINDING: Fear silhouette negative at BOTH checkpoints (best={fear_best:.4f}, final={fear_final:.4f}).")
        print("Center loss alone cannot separate fear from its neighbors.")
    elif fear_final >= 0:
        print(f"Fear silhouette recovered by final epoch ({fear_final:.4f}).")

    print("\n" + "=" * 70)
    print("END — Stage 1 v2: Center Loss Fine-Tuning")
    print("=" * 70)


if __name__ == "__main__":
    main()
