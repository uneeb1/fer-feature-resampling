"""Baseline v3: ResNet-18 with class-agnostic regularization only (NO augmentation)."""
import os
import json
import random
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import f1_score, confusion_matrix

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data_lossless")
RESULTS = os.path.join(BASE, "results", "baseline_v3")
LOGS = os.path.join(BASE, "logs")
CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
SEED = 42

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def build_model():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # Freeze first two stages: conv1, bn1, layer1, layer2
    frozen = ["conv1", "bn1", "layer1", "layer2"]
    for name, param in model.named_parameters():
        if any(name.startswith(f) for f in frozen):
            param.requires_grad = False

    model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(512, 7))
    return model

def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0; correct = 0; total = 0
    all_preds = []; all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    avg_loss = total_loss / total
    accuracy = correct / total
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    per_class_f1 = f1_score(all_labels, all_preds, average=None, labels=list(range(7)))
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(7)))
    return avg_loss, accuracy, macro_f1, per_class_f1, cm, all_preds, all_labels

def main():
    print("=" * 60)
    print("BASELINE v3: ResNet-18 — NO augmentation, class-agnostic regularization")
    print("  Dropout 0.5, weight decay 1e-3, label smoothing 0.1")
    print("  Frozen: conv1, bn1, layer1, layer2")
    print("=" * 60)

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # NO augmentation — grayscale + resize + normalize only
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(os.path.join(DATA, "train"), transform=transform)
    val_ds = datasets.ImageFolder(os.path.join(DATA, "validation"), transform=transform)
    test_ds = datasets.ImageFolder(os.path.join(DATA, "test"), transform=transform)

    print(f"Class to idx: {train_ds.class_to_idx}")
    assert list(train_ds.class_to_idx.keys()) == CLASSES, "Class order mismatch!"
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    # Print class counts
    from collections import Counter
    train_counts = Counter(train_ds.targets)
    for i, cls in enumerate(CLASSES):
        print(f"  {cls:>10}: {train_counts[i]:,}")

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    model = build_model().to(device)

    # Label smoothing 0.1, NO class weights
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Only optimize unfrozen parameters
    backbone_params = [p for n, p in model.named_parameters()
                       if not n.startswith("fc.") and p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if n.startswith("fc.")]

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total_params:,} ({100*trainable/total_params:.1f}%)")

    optimizer = torch.optim.Adam([
        {"params": backbone_params, "lr": 3e-5, "weight_decay": 1e-3},
        {"params": head_params, "lr": 1e-4, "weight_decay": 1e-3},
    ])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    history = []
    best_val_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    max_epochs = 30
    patience = 5

    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(LOGS, exist_ok=True)

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0; train_correct = 0; train_total = 0
        t0 = time.time()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * labels.size(0)
            _, preds = outputs.max(1)
            train_correct += preds.eq(labels).sum().item()
            train_total += labels.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        val_loss, val_acc, val_f1, val_per_f1, _, _, _ = evaluate(model, val_loader, device, criterion)
        scheduler.step(val_f1)

        elapsed = time.time() - t0
        print(f"Epoch {epoch:2d}/{max_epochs} ({elapsed:.0f}s) | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f}")

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_accuracy": round(train_acc, 6),
            "val_loss": round(val_loss, 6),
            "val_accuracy": round(val_acc, 6),
            "val_macro_f1": round(val_f1, 6),
        })

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(RESULTS, f"best_model_s{SEED}.pth"))
            print(f"  -> New best! Val F1={val_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch} (best epoch: {best_epoch})")
                break

    with open(os.path.join(LOGS, f"baseline_v3_history_s{SEED}.json"), "w") as f:
        json.dump(history, f, indent=2)

    model.load_state_dict(torch.load(os.path.join(RESULTS, f"best_model_s{SEED}.pth"), weights_only=True))

    print("\n" + "=" * 60)
    print("EVALUATION — VALIDATION SET")
    print("=" * 60)
    val_loss, val_acc, val_f1, val_per_f1, val_cm, _, _ = evaluate(model, val_loader, device, criterion)
    print(f"Accuracy: {val_acc:.4f}")
    print(f"Macro-F1: {val_f1:.4f}")
    for i, cls in enumerate(CLASSES):
        print(f"  {cls:>10}: F1={val_per_f1[i]:.4f}")

    print("\n" + "=" * 60)
    print("EVALUATION — TEST SET")
    print("=" * 60)
    test_loss, test_acc, test_f1, test_per_f1, test_cm, test_preds, test_labels = evaluate(model, test_loader, device, criterion)
    print(f"Accuracy: {test_acc:.4f}")
    print(f"Macro-F1: {test_f1:.4f}")
    for i, cls in enumerate(CLASSES):
        print(f"  {cls:>10}: F1={test_per_f1[i]:.4f}")

    results = {
        "seed": SEED,
        "best_epoch": best_epoch,
        "config": {
            "augmentation": "NONE",
            "dropout": 0.5,
            "weight_decay": 1e-3,
            "label_smoothing": 0.1,
            "frozen_stages": ["conv1", "bn1", "layer1", "layer2"],
            "backbone_lr": 3e-5,
            "head_lr": 1e-4,
            "batch_size": 64,
            "max_epochs": 30,
        },
        "validation": {
            "accuracy": round(val_acc, 6),
            "macro_f1": round(val_f1, 6),
            "per_class_f1": {cls: round(float(val_per_f1[i]), 6) for i, cls in enumerate(CLASSES)},
            "confusion_matrix": val_cm.tolist(),
        },
        "test": {
            "accuracy": round(test_acc, 6),
            "macro_f1": round(test_f1, 6),
            "per_class_f1": {cls: round(float(test_per_f1[i]), 6) for i, cls in enumerate(CLASSES)},
            "confusion_matrix": test_cm.tolist(),
        },
    }
    with open(os.path.join(RESULTS, f"baseline_v3_s{SEED}.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {os.path.join(RESULTS, f'baseline_v3_s{SEED}.json')}")

    np.savez(os.path.join(RESULTS, f"test_predictions_s{SEED}.npz"),
             preds=np.array(test_preds), labels=np.array(test_labels))

    # Report overfitting gap
    best_h = history[best_epoch - 1]
    gap = best_h["train_accuracy"] - best_h["val_accuracy"]
    print(f"\n{'=' * 60}")
    print(f"BASELINE v3 COMPLETE")
    print(f"  Best epoch: {best_epoch}")
    print(f"  Train acc @ best: {best_h['train_accuracy']:.4f}")
    print(f"  Val acc @ best:   {best_h['val_accuracy']:.4f}")
    print(f"  Overfitting gap:  {gap:.4f} ({100*gap:.1f}%)")
    print(f"  Val macro-F1:     {best_val_f1:.4f}")
    print(f"  Test macro-F1:    {test_f1:.4f}")
    print(f"  Test accuracy:    {test_acc:.4f}")
    print(f"{'=' * 60}")

if __name__ == "__main__":
    main()
