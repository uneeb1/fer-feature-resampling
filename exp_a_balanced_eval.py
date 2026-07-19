"""
Experiment A: Balanced vs imbalanced val evaluation.
Retrains a small slice of heads (baseline, SVMSMOTE/full, ADASYN/minority_only),
then evaluates on both the real imbalanced val set and repeated balanced subsamples
(55/class, 100 draws) to test whether resampling wins under balanced eval.
"""

import os
import copy
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score
from collections import Counter
from imblearn.over_sampling import SMOTE, SVMSMOTE, ADASYN

FEATURES_DIR = "./experiments/features"
OUT_DIR = "./experiments/exp_a_balanced_eval"
NUM_CLASSES = 7
CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

N_DRAWS = 100
SAMPLES_PER_CLASS = 55


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    best_state = None

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
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = copy.deepcopy(head.state_dict())

    head.load_state_dict(best_state)
    return head, best_f1


def evaluate_head(head, features, labels):
    head.eval()
    ds = TensorDataset(torch.FloatTensor(features), torch.LongTensor(labels))
    loader = DataLoader(ds, batch_size=256, shuffle=False)
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            out = head(X_batch.to(DEVICE))
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_labels.extend(y_batch.numpy())
    return f1_score(all_labels, all_preds, average="macro", zero_division=0)


def balanced_subsample(features, labels, n_per_class, rng):
    indices = []
    for c in range(NUM_CLASSES):
        cls_idx = np.where(labels == c)[0]
        chosen = rng.choice(cls_idx, size=min(n_per_class, len(cls_idx)), replace=False)
        indices.extend(chosen)
    rng.shuffle(indices)
    return features[indices], labels[indices]


def balanced_eval(head, val_features, val_labels, n_draws, n_per_class, base_seed=0):
    rng = np.random.RandomState(base_seed)
    scores = []
    for _ in range(n_draws):
        bf, bl = balanced_subsample(val_features, val_labels, n_per_class, rng)
        scores.append(evaluate_head(head, bf, bl))
    scores = np.array(scores)
    mean = scores.mean()
    ci = 1.96 * scores.std() / np.sqrt(len(scores))
    return mean, ci, scores


def compute_strategy_targets(labels, strategy):
    counts = Counter(labels)
    if strategy == "full":
        majority = max(counts.values())
        return {c: majority for c in counts if counts[c] < majority}
    elif strategy == "minority_only":
        minority_classes = [CLASS_NAMES.index(n) for n in ["disgust", "fear", "surprise"]]
        non_minority = {c: v for c, v in counts.items() if c not in minority_classes}
        target = min(non_minority.values()) if non_minority else max(counts.values())
        return {c: target for c in minority_classes if counts[c] < target}
    raise ValueError(f"Unknown strategy: {strategy}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Device: {DEVICE}")
    print("Loading features...")
    train_features = np.load(os.path.join(FEATURES_DIR, "train_features.npy"))
    train_labels = np.load(os.path.join(FEATURES_DIR, "train_labels.npy"))
    val_features = np.load(os.path.join(FEATURES_DIR, "val_features.npy"))
    val_labels = np.load(os.path.join(FEATURES_DIR, "val_labels.npy"))

    print(f"Train: {train_features.shape}, Val: {val_features.shape}")
    print(f"Val class dist: {dict(Counter(val_labels.tolist()))}")
    min_val = min(Counter(val_labels.tolist()).values())
    print(f"Min val class size: {min_val} -> using {SAMPLES_PER_CLASS}/class")

    configs = [
        ("baseline", None, None, 42),
        ("SVMSMOTE_full", SVMSMOTE, "full", 456),
        ("ADASYN_minority_only", ADASYN, "minority_only", 123),
    ]

    results = {}

    for name, sampler_cls, strategy, seed in configs:
        print(f"\n{'='*60}")
        print(f"Config: {name} (seed={seed})")
        print(f"{'='*60}")

        if sampler_cls is not None:
            targets = compute_strategy_targets(train_labels.tolist(), strategy)
            set_seed(seed)
            k = min(5, min(Counter(train_labels).values()) - 1)
            if sampler_cls == ADASYN:
                sampler = sampler_cls(sampling_strategy=targets, random_state=seed, n_neighbors=k)
            else:
                sampler = sampler_cls(sampling_strategy=targets, random_state=seed, k_neighbors=k)
            X_train, y_train = sampler.fit_resample(train_features, train_labels)
            print(f"  Resampled: {len(y_train)} samples")
        else:
            X_train, y_train = train_features, train_labels

        print("  Training head...")
        head, imbalanced_f1 = train_head(X_train, y_train, val_features, val_labels, seed)

        head_path = os.path.join(OUT_DIR, f"{name}_seed{seed}.pth")
        torch.save(head.state_dict(), head_path)
        print(f"  Saved head: {head_path}")

        print(f"  Imbalanced val macro-F1: {imbalanced_f1:.4f}")

        print(f"  Running {N_DRAWS} balanced draws ({SAMPLES_PER_CLASS}/class)...")
        bal_mean, bal_ci, bal_scores = balanced_eval(
            head, val_features, val_labels, N_DRAWS, SAMPLES_PER_CLASS
        )
        print(f"  Balanced val macro-F1: {bal_mean:.4f} ± {bal_ci:.4f} (95% CI)")

        results[name] = {
            "seed": seed,
            "imbalanced_val_f1": round(float(imbalanced_f1), 4),
            "balanced_val_f1_mean": round(float(bal_mean), 4),
            "balanced_val_f1_ci95": round(float(bal_ci), 4),
            "balanced_val_f1_std": round(float(bal_scores.std()), 4),
            "balanced_val_f1_min": round(float(bal_scores.min()), 4),
            "balanced_val_f1_max": round(float(bal_scores.max()), 4),
            "n_draws": N_DRAWS,
            "samples_per_class": SAMPLES_PER_CLASS,
        }

    print(f"\n{'='*60}")
    print("COMPARISON TABLE")
    print(f"{'='*60}")
    print(f"{'Config':<25} {'Imbal F1':>10} {'Bal F1 (mean)':>14} {'95% CI':>10}")
    print("-" * 62)
    for name, r in results.items():
        print(f"{name:<25} {r['imbalanced_val_f1']:>10.4f} {r['balanced_val_f1_mean']:>14.4f} {'±'+str(r['balanced_val_f1_ci95']):>10}")

    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUT_DIR}/results.json")


if __name__ == "__main__":
    main()
