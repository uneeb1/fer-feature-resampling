import math
import time
import torch
import torch.nn as nn
from sklearn.metrics import f1_score


def get_lr(epoch, base_lr, warmup_epochs, total_epochs):
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)
        all_preds.extend(logits.argmax(1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    n = len(all_labels)
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / n
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    per_class_f1 = f1_score(all_labels, all_preds, average=None)
    return total_loss / n, acc, macro_f1, per_class_f1, all_preds, all_labels


@torch.no_grad()
def evaluate_tta(model, dataset, device, resolution=224, batch_size=32):
    from .transforms import get_tta_transform, IMAGENET_MEAN, IMAGENET_STD
    from torchvision import transforms
    from torch.utils.data import DataLoader
    from .dataset import FER2013Dataset

    tta_tf = get_tta_transform(resolution)
    flip_tta_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.Resize((int(resolution * 1.14), int(resolution * 1.14))),
        transforms.TenCrop(resolution),
        transforms.Lambda(lambda crops: [transforms.ToTensor()(c) for c in crops]),
        transforms.Lambda(lambda tensors: [transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)(t) for t in tensors]),
    ])

    model.eval()
    all_preds, all_labels = [], []

    for pixels, label in dataset.data:
        from PIL import Image
        img = Image.fromarray(pixels, mode="L").convert("RGB")
        crops = tta_tf(img)
        flip_crops = flip_tta_tf(img)
        all_crops = torch.stack(crops + flip_crops)  # 20 crops
        all_crops = all_crops.to(device)

        logits = model(all_crops)
        avg_logits = logits.mean(0)
        all_preds.append(avg_logits.argmax().item())
        all_labels.append(label)

    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    per_class_f1 = f1_score(all_labels, all_preds, average=None)
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    return acc, macro_f1, per_class_f1, all_preds, all_labels


def train_model(model, train_loader, val_loader, cfg, device, save_path, log_fn=print):
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["training"]["label_smoothing"])
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg["training"]["lr"],
        momentum=cfg["training"]["momentum"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    epochs = cfg["training"]["epochs"]
    warmup = cfg["training"]["warmup_epochs"]
    base_lr = cfg["training"]["lr"]
    best_f1, best_epoch = 0.0, 0
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": [], "lr": []}

    for epoch in range(epochs):
        lr = get_lr(epoch, base_lr, warmup, epochs)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        history["lr"].append(lr)

        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1, val_pcf1, _, _ = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        log_fn(f"Epoch {epoch+1:3d}/{epochs} | LR {lr:.6f} | "
               f"Train Loss {train_loss:.4f} Acc {train_acc:.4f} | "
               f"Val Loss {val_loss:.4f} Acc {val_acc:.4f} F1 {val_f1:.4f} | "
               f"{elapsed:.1f}s")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch + 1
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "val_f1": float(val_f1),
                "val_acc": float(val_acc),
            }, save_path)
            log_fn(f"  -> New best val F1: {val_f1:.4f} (saved)")

    log_fn(f"Best val F1: {best_f1:.4f} at epoch {best_epoch}")
    return history, best_epoch, best_f1
