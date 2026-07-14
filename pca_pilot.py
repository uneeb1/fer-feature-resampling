"""
PCA + SMOTE pilot test. Compares PCA 50/100/200/512 with plain SMOTE full strategy, seed 42.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, f1_score
from collections import Counter
from imblearn.over_sampling import SMOTE
import copy

FEATURES_DIR = "./experiments/features"
NUM_CLASSES = 7
CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
SEED = 42
DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

BASELINE_F1 = 0.6440
BEST_NONPCA_SMOTE_F1 = 0.6367


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_head(train_X, train_y, val_X, val_y, input_dim, seed):
    set_seed(seed)
    train_ds = TensorDataset(torch.FloatTensor(train_X), torch.LongTensor(train_y))
    val_ds = TensorDataset(torch.FloatTensor(val_X), torch.LongTensor(val_y))
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    head = nn.Sequential(nn.Dropout(0.3), nn.Linear(input_dim, NUM_CLASSES)).to(DEVICE)
    optimizer = optim.Adam(head.parameters(), lr=1e-4, weight_decay=3e-4)
    criterion = nn.CrossEntropyLoss()

    best_f1 = 0.0
    best_report = None

    for epoch in range(50):
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
            best_epoch = epoch + 1
            best_report = classification_report(
                all_labels, all_preds, target_names=CLASS_NAMES, zero_division=0)

    return best_f1, best_epoch, best_report


if __name__ == "__main__":
    print(f"Device: {DEVICE}")

    train_features = np.load(f"{FEATURES_DIR}/train_features.npy")
    train_labels = np.load(f"{FEATURES_DIR}/train_labels.npy")
    val_features = np.load(f"{FEATURES_DIR}/val_features.npy")
    val_labels = np.load(f"{FEATURES_DIR}/val_labels.npy")
    print(f"Loaded: train {train_features.shape}, val {val_features.shape}")

    majority_count = max(Counter(train_labels).values())
    counts = Counter(train_labels)
    smote_targets = {c: majority_count for c in counts if counts[c] < majority_count}
    k = min(5, min(counts.values()) - 1)

    pca_dims = [50, 100, 200, 512]
    results = []

    for dim in pca_dims:
        print(f"\n{'='*60}")
        print(f"PCA {dim} + SMOTE full / seed {SEED}")
        print(f"{'='*60}")

        if dim < 512:
            pca = PCA(n_components=dim, random_state=42)
            t_feat = pca.fit_transform(train_features)
            v_feat = pca.transform(val_features)
            var_explained = pca.explained_variance_ratio_.sum()
            print(f"PCA variance retained: {var_explained:.1%}")
        else:
            t_feat = train_features
            v_feat = val_features

        set_seed(SEED)
        sampler = SMOTE(sampling_strategy=smote_targets, random_state=SEED, k_neighbors=k)
        X_res, y_res = sampler.fit_resample(t_feat, train_labels)
        print(f"Resampled: {len(train_labels)} -> {len(y_res)}")

        f1, best_ep, report = train_head(X_res, y_res, v_feat, val_labels, dim, SEED)

        print(f"\nVal Macro-F1: {f1:.4f}  (best epoch {best_ep})")
        print(report)

        results.append({"dim": dim, "f1": f1, "epoch": best_ep})

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY — PCA + SMOTE PILOT (seed 42, full strategy)")
    print("=" * 70)
    header = f"{'PCA Dims':>10} {'SMOTE Val F1':>13} {'vs Baseline':>12} {'vs Best Non-PCA':>16}"
    print(header)
    print("-" * len(header))
    for r in results:
        dim_label = f"{r['dim']}" if r["dim"] < 512 else "512 (ref)"
        delta_base = r["f1"] - BASELINE_F1
        delta_smote = r["f1"] - BEST_NONPCA_SMOTE_F1
        sign_b = "+" if delta_base >= 0 else ""
        sign_s = "+" if delta_smote >= 0 else ""
        print(f"{dim_label:>10} {r['f1']:>13.4f} {sign_b}{delta_base:>11.4f} {sign_s}{delta_smote:>15.4f}")
