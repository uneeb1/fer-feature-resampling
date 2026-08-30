"""
Config 1: Manifold Mixup at ResNet-18 layer4.
Interpolates features AND labels (soft labels).
Tests alpha in {0.2, 1.0, 2.0}, seeds {42, 123, 456}.
"""

import os
import sys
import time
import copy
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.metrics import classification_report, f1_score
from scipy import stats

NUM_CLASSES = 7
CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
BATCH_SIZE = 64
NUM_EPOCHS = 30
BACKBONE_LR = 3e-5
HEAD_LR = 1e-4
WEIGHT_DECAY = 3e-4
DROPOUT = 0.3
PATIENCE = 7
NUM_WORKERS = 4
DATA_DIR = "../../data"
DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


class ManifoldMixupResNet(nn.Module):
    def __init__(self, num_classes=7, dropout=0.3):
        super().__init__()
        base = models.resnet18(weights="IMAGENET1K_V1")
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward_head(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward(self, x):
        return self.forward_head(self.forward_features(x))


def soft_cross_entropy(logits, targets):
    log_probs = torch.log_softmax(logits, dim=1)
    return -(targets * log_probs).sum(dim=1).mean()


def manifold_mixup_batch(features, labels_onehot, alpha):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = features.size(0)
    index = torch.randperm(batch_size, device=features.device)
    mixed_features = lam * features + (1 - lam) * features[index]
    mixed_labels = lam * labels_onehot + (1 - lam) * labels_onehot[index]
    return mixed_features, mixed_labels


def to_onehot(labels, num_classes):
    return torch.zeros(labels.size(0), num_classes, device=labels.device).scatter_(1, labels.unsqueeze(1), 1.0)


def get_transforms():
    train_t = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_t = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_t, val_t


def balanced_eval(model, val_features, val_labels, n_draws=100, n_per_class=55):
    """Balanced evaluation: subsample 55/class, 100 draws."""
    rng = np.random.RandomState(0)
    macro_scores = []
    class_scores = {c: [] for c in CLASS_NAMES}

    for _ in range(n_draws):
        indices = []
        for c in range(NUM_CLASSES):
            cls_idx = np.where(val_labels == c)[0]
            chosen = rng.choice(cls_idx, size=min(n_per_class, len(cls_idx)), replace=False)
            indices.extend(chosen)
        indices = np.array(indices)
        y_true = val_labels[indices]
        y_pred = val_features[indices]
        macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
        macro_scores.append(macro)
        per_class = f1_score(y_true, y_pred, average=None, labels=list(range(NUM_CLASSES)), zero_division=0)
        for ci, cname in enumerate(CLASS_NAMES):
            class_scores[cname].append(per_class[ci])

    return np.array(macro_scores), {k: np.array(v) for k, v in class_scores.items()}


def train_one_config(alpha, seed, out_dir):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    os.makedirs(out_dir, exist_ok=True)

    train_t, val_t = get_transforms()
    train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_t)
    val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=val_t)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    model = ManifoldMixupResNet(NUM_CLASSES, DROPOUT).to(DEVICE)

    backbone_params = [p for n, p in model.named_parameters() if "fc" not in n]
    head_params = [p for n, p in model.named_parameters() if "fc" in n]
    optimizer = optim.Adam([
        {"params": backbone_params, "lr": BACKBONE_LR},
        {"params": head_params, "lr": HEAD_LR},
    ], weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    best_f1 = 0.0
    best_wts = None
    patience_ctr = 0
    best_epoch = 0

    print(f"\n{'='*60}")
    print(f"Manifold Mixup | alpha={alpha} | seed={seed}")
    print(f"{'='*60}")

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        n_samples = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            labels_oh = to_onehot(labels, NUM_CLASSES)

            features = model.forward_features(inputs)
            mixed_features, mixed_labels = manifold_mixup_batch(features, labels_oh, alpha)
            logits = model.forward_head(mixed_features)
            loss = soft_cross_entropy(logits, mixed_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            n_samples += inputs.size(0)

        scheduler.step()
        train_loss = running_loss / n_samples

        # Validation (no mixup)
        model.eval()
        val_preds, val_labels_all = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                logits = model(inputs)
                val_preds.extend(logits.argmax(1).cpu().numpy())
                val_labels_all.extend(labels.cpu().numpy())

        val_preds = np.array(val_preds)
        val_labels_all = np.array(val_labels_all)
        val_f1 = f1_score(val_labels_all, val_preds, average="macro", zero_division=0)

        print(f"  Epoch {epoch+1:2d} | train_loss={train_loss:.4f} | val_macro_f1={val_f1:.4f}", end="")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch + 1
            patience_ctr = 0
            best_wts = copy.deepcopy(model.state_dict())
            print(f" * best")
        else:
            patience_ctr += 1
            print(f"  (pat {patience_ctr}/{PATIENCE})")
            if patience_ctr >= PATIENCE:
                print(f"  Early stop at epoch {epoch+1}")
                break

    # Load best and do full eval
    model.load_state_dict(best_wts)
    model.eval()
    val_preds, val_labels_all = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            logits = model(inputs)
            val_preds.extend(logits.argmax(1).cpu().numpy())
            val_labels_all.extend(labels.cpu().numpy())

    val_preds = np.array(val_preds)
    val_labels_all = np.array(val_labels_all)

    # Imbalanced eval
    imbal_f1 = f1_score(val_labels_all, val_preds, average="macro", zero_division=0)
    per_class_f1 = f1_score(val_labels_all, val_preds, average=None, labels=list(range(NUM_CLASSES)), zero_division=0)
    report = classification_report(val_labels_all, val_preds, target_names=CLASS_NAMES, zero_division=0, output_dict=True)

    # Balanced eval
    bal_macros, bal_class = balanced_eval(model, val_preds, val_labels_all)
    bal_mean = bal_macros.mean()
    bal_ci = 1.96 * bal_macros.std() / np.sqrt(len(bal_macros))

    results = {
        "config": "manifold_mixup",
        "alpha": alpha,
        "seed": seed,
        "best_epoch": best_epoch,
        "imbalanced_macro_f1": round(float(imbal_f1), 4),
        "imbalanced_per_class_f1": {CLASS_NAMES[i]: round(float(per_class_f1[i]), 4) for i in range(NUM_CLASSES)},
        "balanced_macro_f1_mean": round(float(bal_mean), 4),
        "balanced_macro_f1_ci95": round(float(bal_ci), 4),
        "balanced_per_class_f1": {
            cname: {"mean": round(float(bal_class[cname].mean()), 4),
                    "ci95": round(float(1.96 * bal_class[cname].std() / np.sqrt(len(bal_class[cname]))), 4)}
            for cname in CLASS_NAMES
        },
    }

    # Save
    result_path = os.path.join(out_dir, f"mmixup_a{alpha}_s{seed}.json")
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)

    model_path = os.path.join(out_dir, f"mmixup_a{alpha}_s{seed}.pth")
    torch.save(best_wts, model_path)

    print(f"\n  Results: imbal_F1={imbal_f1:.4f} | bal_F1={bal_mean:.4f}±{bal_ci:.4f}")
    print(f"  Per-class (imbal): {' | '.join(f'{c}={per_class_f1[i]:.3f}' for i, c in enumerate(CLASS_NAMES))}")
    print(f"  Saved: {result_path}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, nargs="+", default=[0.2, 1.0, 2.0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--out_dir", type=str, default="./results")
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    all_results = []
    for alpha in args.alpha:
        for seed in args.seeds:
            r = train_one_config(alpha, seed, args.out_dir)
            all_results.append(r)

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY — Manifold Mixup Config 1")
    print(f"{'='*80}")
    print(f"{'Alpha':<8}{'Seed':<8}{'Imbal F1':<12}{'Bal F1':<16}{'Fear':<8}{'Sad':<8}{'Neutral':<8}")
    print("-" * 70)
    for r in all_results:
        pc = r["imbalanced_per_class_f1"]
        print(f"{r['alpha']:<8}{r['seed']:<8}{r['imbalanced_macro_f1']:<12.4f}"
              f"{r['balanced_macro_f1_mean']:.4f}±{r['balanced_macro_f1_ci95']:.4f}  "
              f"{pc['fear']:<8.4f}{pc['sad']:<8.4f}{pc['neutral']:<8.4f}")

    # Save combined
    with open(os.path.join(args.out_dir, "config1_all_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    # Best alpha by mean imbal F1 across seeds
    alpha_means = {}
    for alpha in args.alpha:
        runs = [r for r in all_results if r["alpha"] == alpha]
        alpha_means[alpha] = np.mean([r["imbalanced_macro_f1"] for r in runs])
    best_alpha = max(alpha_means, key=alpha_means.get)
    print(f"\nBest alpha: {best_alpha} (mean imbal F1 = {alpha_means[best_alpha]:.4f})")
    print(f"Alpha means: {alpha_means}")


if __name__ == "__main__":
    main()
