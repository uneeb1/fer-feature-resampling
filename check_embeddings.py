"""
Visualize and analyze the 512-dim embeddings from the baseline model.
- t-SNE 2D plot (real samples, colored by class)
- t-SNE with SMOTE synthetic samples overlaid
- Per-class cluster stats (inter/intra-class distances)
- Class separability analysis
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
from imblearn.over_sampling import SMOTE
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR      = "./data"
BASELINE_CKPT = "./experiments/exp_03/best_model.pth"
OUT_DIR       = "./outputs/embedding_analysis"
NUM_CLASSES   = 7
BATCH_SIZE    = 64
NUM_WORKERS   = 0
DEVICE        = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
CLASS_COLORS = ["#e74c3c", "#8e44ad", "#2c3e50", "#f39c12", "#1abc9c", "#3498db", "#e67e22"]

os.makedirs(OUT_DIR, exist_ok=True)

# ── Transforms ───────────────────────────────────────────────────────────────

base_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ── Dataset ──────────────────────────────────────────────────────────────────

train_dataset = datasets.ImageFolder(
    root=os.path.join(DATA_DIR, "train"), transform=base_transforms)
val_dataset = datasets.ImageFolder(
    root=os.path.join(DATA_DIR, "test"), transform=base_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=NUM_WORKERS)

# ── Load model ───────────────────────────────────────────────────────────────

model = models.resnet18(weights="IMAGENET1K_V1")
model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.fc.in_features, NUM_CLASSES)
)
model.load_state_dict(torch.load(BASELINE_CKPT, map_location=DEVICE, weights_only=True))
model = model.to(DEVICE)
model.eval()

# ── Extract embeddings ───────────────────────────────────────────────────────

hook_output = {}

def hook_fn(module, input, output):
    hook_output["feat"] = output

def extract_embeddings(loader):
    emb_list, lab_list = [], []
    handle = model.avgpool.register_forward_hook(hook_fn)
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(DEVICE)
            _ = model(inputs)
            feats = hook_output["feat"].squeeze(-1).squeeze(-1)
            emb_list.append(feats.cpu().numpy())
            lab_list.append(labels.numpy())
    handle.remove()
    return np.concatenate(emb_list), np.concatenate(lab_list)

print("Extracting train embeddings...")
train_emb, train_lab = extract_embeddings(train_loader)
print(f"  Shape: {train_emb.shape}")

print("Extracting val embeddings...")
val_emb, val_lab = extract_embeddings(val_loader)
print(f"  Shape: {val_emb.shape}")

# ── 1. Per-class statistics ──────────────────────────────────────────────────

print("\n" + "=" * 60)
print("PER-CLASS EMBEDDING STATISTICS")
print("=" * 60)

class_centroids = {}
class_intra_dist = {}

for c in range(NUM_CLASSES):
    mask = train_lab == c
    class_emb = train_emb[mask]
    centroid = class_emb.mean(axis=0)
    class_centroids[c] = centroid

    dists = np.linalg.norm(class_emb - centroid, axis=1)
    class_intra_dist[c] = dists.mean()

print(f"\n{'Class':>10s}  {'N':>5s}  {'Intra-dist (mean)':>18s}  {'Std of norms':>14s}")
print("-" * 55)
for c in range(NUM_CLASSES):
    mask = train_lab == c
    class_emb = train_emb[mask]
    norms = np.linalg.norm(class_emb, axis=1)
    print(f"{CLASS_NAMES[c]:>10s}  {mask.sum():>5d}  {class_intra_dist[c]:>18.4f}  {norms.std():>14.4f}")

# ── 2. Inter-class distances ─────────────────────────────────────────────────

print("\n" + "=" * 60)
print("INTER-CLASS CENTROID DISTANCES")
print("=" * 60)

centroid_matrix = np.stack([class_centroids[c] for c in range(NUM_CLASSES)])
inter_dists = cdist(centroid_matrix, centroid_matrix, metric="euclidean")

print(f"\n{'':>10s}", end="")
for c in range(NUM_CLASSES):
    print(f"  {CLASS_NAMES[c][:5]:>6s}", end="")
print()

for i in range(NUM_CLASSES):
    print(f"{CLASS_NAMES[i]:>10s}", end="")
    for j in range(NUM_CLASSES):
        print(f"  {inter_dists[i, j]:>6.2f}", end="")
    print()

# Find closest class pairs
print("\nClosest class pairs (hardest to separate):")
pairs = []
for i in range(NUM_CLASSES):
    for j in range(i + 1, NUM_CLASSES):
        pairs.append((inter_dists[i, j], CLASS_NAMES[i], CLASS_NAMES[j]))
pairs.sort()
for dist, c1, c2 in pairs[:5]:
    print(f"  {c1:>10s} <-> {c2:<10s}  distance: {dist:.4f}")

# ── 3. Separability ratio (inter / intra) ───────────────────────────────────

print("\n" + "=" * 60)
print("CLASS SEPARABILITY (inter-class dist / intra-class spread)")
print("=" * 60)

print(f"\n{'Class':>10s}  {'Nearest neighbor':>16s}  {'Inter dist':>10s}  {'Intra dist':>10s}  {'Ratio':>6s}")
print("-" * 60)
for c in range(NUM_CLASSES):
    nearest_dist = float("inf")
    nearest_cls = -1
    for j in range(NUM_CLASSES):
        if j != c and inter_dists[c, j] < nearest_dist:
            nearest_dist = inter_dists[c, j]
            nearest_cls = j
    ratio = nearest_dist / class_intra_dist[c] if class_intra_dist[c] > 0 else 0
    print(f"{CLASS_NAMES[c]:>10s}  {CLASS_NAMES[nearest_cls]:>16s}  {nearest_dist:>10.4f}  {class_intra_dist[c]:>10.4f}  {ratio:>6.2f}")

# ── 4. Silhouette score ─────────────────────────────────────────────────────

print("\nComputing silhouette score (may take a minute)...")
n_subsample = min(5000, len(train_emb))
idx = np.random.RandomState(42).choice(len(train_emb), n_subsample, replace=False)
sil = silhouette_score(train_emb[idx], train_lab[idx])
print(f"Silhouette score (train, n={n_subsample}): {sil:.4f}")
print("  (>0.5 = good separation, 0.25-0.5 = overlapping, <0.25 = poor)")

sil_val = silhouette_score(val_emb, val_lab)
print(f"Silhouette score (val, n={len(val_emb)}): {sil_val:.4f}")

# ── 5. t-SNE visualization — real embeddings ────────────────────────────────

print("\nRunning t-SNE on train embeddings (subsampled)...")
n_tsne = min(5000, len(train_emb))
idx_tsne = np.random.RandomState(42).choice(len(train_emb), n_tsne, replace=False)
tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
emb_2d = tsne.fit_transform(train_emb[idx_tsne])
lab_2d = train_lab[idx_tsne]

fig, ax = plt.subplots(figsize=(10, 8))
for c in range(NUM_CLASSES):
    mask = lab_2d == c
    ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1],
               c=CLASS_COLORS[c], label=CLASS_NAMES[c],
               alpha=0.5, s=10, edgecolors="none")
ax.set_title("t-SNE of Train Embeddings (Baseline exp_03)", fontsize=14, fontweight="bold")
ax.legend(markerscale=3, fontsize=10)
ax.set_xticks([])
ax.set_yticks([])
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "tsne_train_real.png"), dpi=150)
plt.close()
print(f"Saved: {OUT_DIR}/tsne_train_real.png")

# ── 6. t-SNE — real vs SMOTE synthetic ──────────────────────────────────────

print("\nApplying SMOTE and visualizing synthetic samples...")
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(train_emb, train_lab)

n_real = len(train_emb)
synthetic_emb = X_smote[n_real:]
synthetic_lab = y_smote[n_real:]
print(f"  Synthetic samples generated: {len(synthetic_emb)}")

n_syn_sample = min(3000, len(synthetic_emb))
idx_syn = np.random.RandomState(42).choice(len(synthetic_emb), n_syn_sample, replace=False)

combined = np.vstack([train_emb[idx_tsne], synthetic_emb[idx_syn]])
combined_lab = np.concatenate([lab_2d, synthetic_lab[idx_syn]])
is_synthetic = np.array([False] * n_tsne + [True] * n_syn_sample)

print("Running t-SNE on combined (real + synthetic)...")
tsne2 = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
combined_2d = tsne2.fit_transform(combined)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Left: real only
ax = axes[0]
real_mask = ~is_synthetic
for c in range(NUM_CLASSES):
    mask = real_mask & (combined_lab == c)
    ax.scatter(combined_2d[mask, 0], combined_2d[mask, 1],
               c=CLASS_COLORS[c], label=CLASS_NAMES[c],
               alpha=0.5, s=10, edgecolors="none")
ax.set_title("Real Embeddings", fontsize=13, fontweight="bold")
ax.legend(markerscale=3, fontsize=9)
ax.set_xticks([])
ax.set_yticks([])

# Right: synthetic overlaid
ax = axes[1]
for c in range(NUM_CLASSES):
    mask_real = real_mask & (combined_lab == c)
    ax.scatter(combined_2d[mask_real, 0], combined_2d[mask_real, 1],
               c=CLASS_COLORS[c], alpha=0.2, s=8, edgecolors="none")

for c in range(NUM_CLASSES):
    mask_syn = is_synthetic & (combined_lab == c)
    if mask_syn.sum() > 0:
        ax.scatter(combined_2d[mask_syn, 0], combined_2d[mask_syn, 1],
                   c=CLASS_COLORS[c], label=f"{CLASS_NAMES[c]} (syn)",
                   alpha=0.8, s=20, edgecolors="black", linewidths=0.3, marker="*")
ax.set_title("SMOTE Synthetic Overlaid on Real", fontsize=13, fontweight="bold")
ax.legend(markerscale=2, fontsize=8)
ax.set_xticks([])
ax.set_yticks([])

plt.suptitle("Feature-Space SMOTE: Where Do Synthetic Samples Land?",
             fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "tsne_real_vs_smote.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {OUT_DIR}/tsne_real_vs_smote.png")

# ── 7. Val embeddings t-SNE ─────────────────────────────────────────────────

print("\nRunning t-SNE on val embeddings...")
tsne3 = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
val_2d = tsne3.fit_transform(val_emb)

fig, ax = plt.subplots(figsize=(10, 8))
for c in range(NUM_CLASSES):
    mask = val_lab == c
    ax.scatter(val_2d[mask, 0], val_2d[mask, 1],
               c=CLASS_COLORS[c], label=CLASS_NAMES[c],
               alpha=0.5, s=10, edgecolors="none")
ax.set_title("t-SNE of Val Embeddings (Baseline exp_03)", fontsize=14, fontweight="bold")
ax.legend(markerscale=3, fontsize=10)
ax.set_xticks([])
ax.set_yticks([])
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "tsne_val.png"), dpi=150)
plt.close()
print(f"Saved: {OUT_DIR}/tsne_val.png")

# ── Summary ──────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Train silhouette : {sil:.4f}")
print(f"Val silhouette   : {sil_val:.4f}")
print(f"Synthetic samples: {len(synthetic_emb)}")
print(f"\nPlots saved to: {OUT_DIR}/")
print("  - tsne_train_real.png       (train embeddings by class)")
print("  - tsne_real_vs_smote.png    (real vs SMOTE synthetic)")
print("  - tsne_val.png              (val embeddings by class)")
