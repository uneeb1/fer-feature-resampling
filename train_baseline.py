"""
FER2013 Baseline — ResNet-18 fine-tuned on ImageNet weights
Backbone LR: 3e-5 | Head LR: 1e-4 | WD: 3e-4 | Dropout: 0.3
Scheduler: ReduceLROnPlateau | Early stopping patience: 5
"""

import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR        = "./data"
EXPERIMENTS_DIR = "./experiments"
NUM_CLASSES     = 7
BATCH_SIZE      = 64
NUM_EPOCHS      = 30
BACKBONE_LR     = 3e-5
HEAD_LR         = 1e-4
WEIGHT_DECAY    = 3e-4
DROPOUT         = 0.3
PATIENCE        = 5
NUM_WORKERS     = 0
DEVICE          = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# ── Auto experiment folder ────────────────────────────────────────────────────

os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
existing = [d for d in os.listdir(EXPERIMENTS_DIR) if d.startswith("exp_")]
next_num = len(existing) + 1
EXP_DIR  = os.path.join(EXPERIMENTS_DIR, f"exp_{next_num:02d}")
os.makedirs(EXP_DIR, exist_ok=True)

print(f"Using device : {DEVICE}")
print(f"Experiment   : {EXP_DIR}")
print(f"Backbone LR  : {BACKBONE_LR}")
print(f"Head LR      : {HEAD_LR}")
print(f"Weight decay : {WEIGHT_DECAY}")
print(f"Dropout      : {DROPOUT}")
print(f"Patience     : {PATIENCE}")
print(f"Scheduler    : StepLR (step=5, gamma=0.1)")

with open(os.path.join(EXP_DIR, "config.txt"), "w") as f:
    f.write(f"Experiment   : {EXP_DIR}\n")
    f.write(f"Device       : {DEVICE}\n")
    f.write(f"Backbone LR  : {BACKBONE_LR}\n")
    f.write(f"Head LR      : {HEAD_LR}\n")
    f.write(f"Weight decay : {WEIGHT_DECAY}\n")
    f.write(f"Dropout      : {DROPOUT}\n")
    f.write(f"Batch size   : {BATCH_SIZE}\n")
    f.write(f"Max epochs   : {NUM_EPOCHS}\n")
    f.write(f"Patience     : {PATIENCE}\n")
    f.write(f"Scheduler    : StepLR step=5 gamma=0.1\n")

# ── Transforms ────────────────────────────────────────────────────────────────

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

# ── Dataset & DataLoader ──────────────────────────────────────────────────────

train_dataset = datasets.ImageFolder(
    root=os.path.join(DATA_DIR, "train"), transform=train_transforms)
val_dataset   = datasets.ImageFolder(
    root=os.path.join(DATA_DIR, "test"),  transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          shuffle=True,  num_workers=NUM_WORKERS)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=NUM_WORKERS)

print(f"Train samples : {len(train_dataset)}")
print(f"Val   samples : {len(val_dataset)}")
print(f"Classes       : {train_dataset.classes}")

# ── Model ─────────────────────────────────────────────────────────────────────

model    = models.resnet18(weights="IMAGENET1K_V1")
model.fc = nn.Sequential(
    nn.Dropout(DROPOUT),
    nn.Linear(model.fc.in_features, NUM_CLASSES)
)
model = model.to(DEVICE)

# ── Differential LR optimizer ─────────────────────────────────────────────────

backbone_params = [p for n, p in model.named_parameters() if "fc" not in n]
head_params     = [p for n, p in model.named_parameters() if "fc"     in n]

optimizer = optim.Adam([
    {"params": backbone_params, "lr": BACKBONE_LR},
    {"params": head_params,     "lr": HEAD_LR},
], weight_decay=WEIGHT_DECAY)

criterion = nn.CrossEntropyLoss()

# ReduceLROnPlateau — drops LR only when val Macro-F1 stops improving
# factor=0.5: halves LR (gentle), patience=3: waits 3 bad epochs first
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# ── History & state ───────────────────────────────────────────────────────────

history = {
    "train_loss": [], "val_loss": [],
    "train_acc":  [], "val_acc":  [],
    "train_f1":   [], "val_f1":   [],
}

best_model_wts   = copy.deepcopy(model.state_dict())
best_macro_f1    = 0.0
patience_counter = 0
best_epoch       = 0

# ── Training Loop ─────────────────────────────────────────────────────────────

for epoch in range(NUM_EPOCHS):
    epoch_start = time.time()
    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
    print("-" * 40)

    # Train phase
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    train_loss = running_loss / len(train_dataset)
    train_acc  = correct / total
    train_f1   = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    # Val phase
    model.eval()
    val_loss_sum, val_correct, val_total = 0.0, 0, 0
    val_preds, val_labels = [], []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss    = criterion(outputs, labels)
            val_loss_sum += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            val_correct += (preds == labels).sum().item()
            val_total   += labels.size(0)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    val_loss = val_loss_sum / len(val_dataset)
    val_acc  = val_correct / val_total
    val_f1   = f1_score(val_labels, val_preds, average="macro", zero_division=0)

    scheduler.step()

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)
    history["train_f1"].append(train_f1)
    history["val_f1"].append(val_f1)

    epoch_time = time.time() - epoch_start
    print(f"  Train — Loss: {train_loss:.4f}  Acc: {train_acc:.4f}  Macro-F1: {train_f1:.4f}")
    print(f"  Val   — Loss: {val_loss:.4f}  Acc: {val_acc:.4f}  Macro-F1: {val_f1:.4f}")
    print(f"  Time  : {epoch_time:.1f}s")

    if val_f1 > best_macro_f1:
        best_macro_f1    = val_f1
        best_epoch       = epoch + 1
        patience_counter = 0
        best_model_wts   = copy.deepcopy(model.state_dict())
        torch.save(best_model_wts, os.path.join(EXP_DIR, "best_model.pth"))
        print(f"  ✓ Best model saved (Macro-F1: {best_macro_f1:.4f})")
    else:
        patience_counter += 1
        print(f"  No improvement. Patience: {patience_counter}/{PATIENCE}")
        if patience_counter >= PATIENCE:
            print(f"\n⚡ Early stopping at epoch {epoch+1}")
            print(f"   Best epoch: {best_epoch}  Best val Macro-F1: {best_macro_f1:.4f}")
            break

# ── Final Evaluation ──────────────────────────────────────────────────────────

print("\n" + "=" * 50)
print("FINAL EVALUATION ON BEST MODEL")
print("=" * 50)
print(f"Best epoch    : {best_epoch}")
print(f"Best Macro-F1 : {best_macro_f1:.4f}")

model.load_state_dict(best_model_wts)
model.eval()

final_preds, final_labels = [], []
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        outputs = model(inputs)
        preds   = outputs.argmax(dim=1)
        final_preds.extend(preds.cpu().numpy())
        final_labels.extend(labels.cpu().numpy())

report = classification_report(final_labels, final_preds,
                                target_names=CLASS_NAMES, zero_division=0)
print(report)

with open(os.path.join(EXP_DIR, "results.txt"), "w") as f:
    f.write(f"Best epoch    : {best_epoch}\n")
    f.write(f"Best Macro-F1 : {best_macro_f1:.4f}\n\n")
    f.write(report)

# ── Plots ─────────────────────────────────────────────────────────────────────

epochs_ran = range(1, len(history["train_loss"]) + 1)

def save_plot(train_vals, val_vals, ylabel, title, filename):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs_ran, train_vals, "b-o", markersize=4, label=f"Train {ylabel}")
    ax.plot(epochs_ran, val_vals,   "r-o", markersize=4, label=f"Val {ylabel}")
    if best_epoch <= len(list(epochs_ran)):
        ax.axvline(x=best_epoch, color="green", linestyle="--",
                   label=f"Best epoch ({best_epoch})")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(EXP_DIR, filename), dpi=150)
    plt.close()
    print(f"✓ Saved: {filename}")

save_plot(history["train_loss"], history["val_loss"],
          "Loss", "Training vs Validation Loss", "loss_curve.png")

save_plot([a * 100 for a in history["train_acc"]],
          [a * 100 for a in history["val_acc"]],
          "Accuracy (%)", "Training vs Validation Accuracy", "accuracy_curve.png")

save_plot(history["train_f1"], history["val_f1"],
          "Macro-F1", "Training vs Validation Macro-F1", "f1_curve.png")

# Confusion matrix
cm      = confusion_matrix(final_labels, final_preds)
cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            linewidths=0.5, ax=ax)
ax.set_title("Confusion Matrix (Normalized) — Best Model", fontsize=14, fontweight="bold")
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(EXP_DIR, "confusion_matrix.png"), dpi=150)
plt.close()
print("✓ Saved: confusion_matrix.png")

print(f"\nAll results saved to: {EXP_DIR}")
