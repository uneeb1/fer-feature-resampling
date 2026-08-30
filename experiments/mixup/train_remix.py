"""
Config 2: Remix — Manifold Mixup with disentangled label mixing (Chou et al., 2020).
Feature mixing identical to Config 1. Labels tilt toward minority class.
"""

import os
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
from sklearn.metrics import classification_report, f1_score, accuracy_score
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

# FER2013 training set class counts (from ImageFolder alphabetical order)
# angry=0, disgust=1, fear=2, happy=3, neutral=4, sad=5, surprise=6
CLASS_COUNTS = {0: 3995, 1: 547, 2: 4097, 3: 8989, 4: 4965, 5: 4830, 6: 4002}


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


def to_onehot(labels, num_classes):
    return torch.zeros(labels.size(0), num_classes, device=labels.device).scatter_(1, labels.unsqueeze(1), 1.0)


def remix_batch(features, labels_onehot, labels_int, alpha, tau, class_counts):
    """
    Remix: features mix with λ, labels mix with λ̃ (tilted toward minority).

    For each pair (i, j):
      - If n_i < n_j (i is minority): λ̃ = max(λ, τ)
      - If n_i > n_j (j is minority): λ̃ = min(λ, 1-τ)
      - If n_i == n_j: λ̃ = λ
    """
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = features.size(0)
    index = torch.randperm(batch_size, device=features.device)

    # Feature mixing — identical to Manifold Mixup
    mixed_features = lam * features + (1 - lam) * features[index]

    # Compute per-sample λ̃ for label mixing
    labels_i = labels_int.cpu().numpy()
    labels_j = labels_int[index].cpu().numpy()
    counts_i = np.array([class_counts[int(c)] for c in labels_i])
    counts_j = np.array([class_counts[int(c)] for c in labels_j])

    lam_tilde = np.full(batch_size, lam)
    minority_i = counts_i < counts_j  # i is minority
    minority_j = counts_i > counts_j  # j is minority
    lam_tilde[minority_i] = max(lam, tau)
    lam_tilde[minority_j] = min(lam, 1 - tau)
    # equal counts: lam_tilde stays at lam

    lam_tilde_t = torch.FloatTensor(lam_tilde).to(features.device).unsqueeze(1)
    mixed_labels = lam_tilde_t * labels_onehot + (1 - lam_tilde_t) * labels_onehot[index]

    # Stats for logging
    n_tilted = int(minority_i.sum() + minority_j.sum())
    mean_lam_tilde = float(lam_tilde.mean())

    return mixed_features, mixed_labels, lam, mean_lam_tilde, n_tilted, batch_size


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


def balanced_eval(val_preds, val_labels, n_draws=100, n_per_class=55):
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
        y_pred = val_preds[indices]
        macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
        macro_scores.append(macro)
        per_class = f1_score(y_true, y_pred, average=None, labels=list(range(NUM_CLASSES)), zero_division=0)
        for ci, cname in enumerate(CLASS_NAMES):
            class_scores[cname].append(per_class[ci])

    return np.array(macro_scores), {k: np.array(v) for k, v in class_scores.items()}


def verify_remix_logic(tau):
    """Print verification examples before training starts."""
    print(f"\n{'='*70}")
    print(f"REMIX LABEL MIXING VERIFICATION (τ={tau})")
    print(f"{'='*70}")
    print(f"Class counts: { {CLASS_NAMES[k]: v for k, v in CLASS_COUNTS.items()} }")
    print()

    test_cases = [
        (1, 3, 0.3, "disgust(547) vs happy(8989): i is minority"),
        (3, 1, 0.3, "happy(8989) vs disgust(547): j is minority"),
        (0, 2, 0.7, "angry(3995) vs fear(4097): i is minority"),
        (2, 0, 0.7, "fear(4097) vs angry(3995): j is minority"),
        (0, 6, 0.5, "angry(3995) vs surprise(4002): i is minority"),
        (4, 4, 0.6, "neutral vs neutral: equal counts"),
    ]

    print(f"{'Pair':<45} {'n_i':>6} {'n_j':>6} {'λ':>6} {'λ̃':>6} {'Tilt?':<12}")
    print("-" * 85)
    for ci, cj, lam, desc in test_cases:
        ni, nj = CLASS_COUNTS[ci], CLASS_COUNTS[cj]
        if ni < nj:
            lam_tilde = max(lam, tau)
            tilt = "→ i (min)"
        elif ni > nj:
            lam_tilde = min(lam, 1 - tau)
            tilt = "→ j (min)"
        else:
            lam_tilde = lam
            tilt = "none"
        print(f"{desc:<45} {ni:>6} {nj:>6} {lam:>6.3f} {lam_tilde:>6.3f} {tilt:<12}")

    print()
    print("Interpretation: λ̃ > λ means label tilts toward sample i;")
    print("                λ̃ < λ means label tilts toward sample j.")
    print("                Tilt always favors the minority class in the pair.")
    print(f"{'='*70}\n")


def train_remix(alpha, tau, seed, out_dir):
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
    print(f"Remix | alpha={alpha} | tau={tau} | seed={seed}")
    print(f"{'='*60}")

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        n_samples = 0
        epoch_lam_sum = 0.0
        epoch_lam_tilde_sum = 0.0
        epoch_tilted = 0
        epoch_total_pairs = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            labels_oh = to_onehot(labels, NUM_CLASSES)

            features = model.forward_features(inputs)
            mixed_features, mixed_labels, lam, mean_lt, n_tilted, bs = remix_batch(
                features, labels_oh, labels, alpha, tau, CLASS_COUNTS
            )
            logits = model.forward_head(mixed_features)
            loss = soft_cross_entropy(logits, mixed_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            n_samples += inputs.size(0)
            epoch_lam_sum += lam * bs
            epoch_lam_tilde_sum += mean_lt * bs
            epoch_tilted += n_tilted
            epoch_total_pairs += bs

        scheduler.step()
        train_loss = running_loss / n_samples
        avg_lam = epoch_lam_sum / epoch_total_pairs
        avg_lam_tilde = epoch_lam_tilde_sum / epoch_total_pairs
        tilt_frac = epoch_tilted / epoch_total_pairs

        # Validation
        model.eval()
        val_preds, val_labels_all = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                logits = model(inputs)
                val_preds.extend(logits.argmax(1).cpu().numpy())
                val_labels_all.extend(labels.cpu().numpy())

        val_preds_arr = np.array(val_preds)
        val_labels_arr = np.array(val_labels_all)
        val_f1 = f1_score(val_labels_arr, val_preds_arr, average="macro", zero_division=0)

        print(f"  Epoch {epoch+1:2d} | loss={train_loss:.4f} | val_f1={val_f1:.4f} | "
              f"λ={avg_lam:.3f} λ̃={avg_lam_tilde:.3f} tilt={tilt_frac:.1%}", end="")

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

    # Final eval on best model
    model.load_state_dict(best_wts)
    model.eval()
    val_preds, val_labels_all = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            logits = model(inputs)
            val_preds.extend(logits.argmax(1).cpu().numpy())
            val_labels_all.extend(labels.cpu().numpy())

    val_preds_arr = np.array(val_preds)
    val_labels_arr = np.array(val_labels_all)

    # Imbalanced eval
    imbal_f1 = f1_score(val_labels_arr, val_preds_arr, average="macro", zero_division=0)
    imbal_acc = accuracy_score(val_labels_arr, val_preds_arr)
    per_class_f1 = f1_score(val_labels_arr, val_preds_arr, average=None, labels=list(range(NUM_CLASSES)), zero_division=0)
    report = classification_report(val_labels_arr, val_preds_arr, target_names=CLASS_NAMES, zero_division=0, output_dict=True)

    # Balanced eval
    bal_macros, bal_class = balanced_eval(val_preds_arr, val_labels_arr)
    bal_mean = bal_macros.mean()
    bal_ci = 1.96 * bal_macros.std() / np.sqrt(len(bal_macros))

    results = {
        "config": "remix",
        "alpha": alpha,
        "tau": tau,
        "seed": seed,
        "best_epoch": best_epoch,
        "imbalanced_macro_f1": round(float(imbal_f1), 4),
        "imbalanced_accuracy": round(float(imbal_acc), 4),
        "imbalanced_per_class_f1": {CLASS_NAMES[i]: round(float(per_class_f1[i]), 4) for i in range(NUM_CLASSES)},
        "imbalanced_per_class_precision": {CLASS_NAMES[i]: round(float(report[CLASS_NAMES[i]]["precision"]), 4) for i in range(NUM_CLASSES)},
        "imbalanced_per_class_recall": {CLASS_NAMES[i]: round(float(report[CLASS_NAMES[i]]["recall"]), 4) for i in range(NUM_CLASSES)},
        "balanced_macro_f1_mean": round(float(bal_mean), 4),
        "balanced_macro_f1_ci95": round(float(bal_ci), 4),
        "balanced_per_class_f1": {
            cname: {"mean": round(float(bal_class[cname].mean()), 4),
                    "ci95": round(float(1.96 * bal_class[cname].std() / np.sqrt(len(bal_class[cname]))), 4)}
            for cname in CLASS_NAMES
        },
    }

    result_path = os.path.join(out_dir, f"remix_a{alpha}_t{tau}_s{seed}.json")
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)

    model_path = os.path.join(out_dir, f"remix_a{alpha}_t{tau}_s{seed}.pth")
    torch.save(best_wts, model_path)

    print(f"\n  === FINAL RESULTS ===")
    print(f"  Imbal macro-F1: {imbal_f1:.4f} | Accuracy: {imbal_acc:.4f}")
    print(f"  Bal macro-F1:   {bal_mean:.4f} ± {bal_ci:.4f}")
    print(f"  Per-class F1 (imbal): {' | '.join(f'{c}={per_class_f1[i]:.3f}' for i, c in enumerate(CLASS_NAMES))}")
    print(f"  Per-class F1 (bal mean): {' | '.join(f'{c}={bal_class[c].mean():.3f}' for c in CLASS_NAMES)}")
    print(f"  Saved: {result_path}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--out_dir", type=str, default="./results")
    parser.add_argument("--verify_only", action="store_true")
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Always verify first
    verify_remix_logic(args.tau)

    if args.verify_only:
        return

    all_results = []
    for seed in args.seeds:
        r = train_remix(args.alpha, args.tau, seed, args.out_dir)
        all_results.append(r)

    if len(all_results) > 1:
        print(f"\n{'='*80}")
        print("SUMMARY — Remix Config 2")
        print(f"{'='*80}")
        for r in all_results:
            pc = r["imbalanced_per_class_f1"]
            print(f"  seed={r['seed']} | imbal_F1={r['imbalanced_macro_f1']:.4f} | "
                  f"bal_F1={r['balanced_macro_f1_mean']:.4f}±{r['balanced_macro_f1_ci95']:.4f} | "
                  f"fear={pc['fear']:.4f} sad={pc['sad']:.4f} neutral={pc['neutral']:.4f}")


if __name__ == "__main__":
    main()
