"""
Layer-wise feature analysis for ResNet-18 backbone.
Extracts features from layer1-4 + avgpool, computes silhouette scores,
inter-class distances, per-class separation, t-SNE plots, and PCA analysis.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import OrderedDict

# ── Config ───────────────────────────────────────────────────────────────────

DATA_DIR = "./data"
BASELINE_CKPT = "./experiments/exp_03/best_model.pth"
OUT_DIR = "./experiments/layer_analysis"
NUM_CLASSES = 7
BATCH_SIZE = 64
SILHOUETTE_SAMPLE = 5000
DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

LAYER_NAMES = ["layer1", "layer2", "layer3", "layer4", "avgpool"]
LAYER_DIMS = {"layer1": 64, "layer2": 128, "layer3": 256, "layer4": 512, "avgpool": 512}

# ── Model ────────────────────────────────────────────────────────────────────

def load_model():
    model = models.resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, NUM_CLASSES))
    state = torch.load(BASELINE_CKPT, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model = model.to(DEVICE)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def get_transform():
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

# ── Feature Extraction ──────────────────────────────────────────────────────

def extract_all_layers(model, data_path):
    ds = datasets.ImageFolder(data_path, transform=get_transform())
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    hook_outputs = {name: [] for name in LAYER_NAMES}
    handles = []

    def make_hook(name):
        def hook_fn(module, inp, out):
            if name == "avgpool":
                hook_outputs[name].append(out.squeeze().cpu().numpy())
            else:
                pooled = nn.functional.adaptive_avg_pool2d(out, 1).squeeze(-1).squeeze(-1)
                hook_outputs[name].append(pooled.cpu().numpy())
        return hook_fn

    for name in LAYER_NAMES:
        module = getattr(model, name)
        handles.append(module.register_forward_hook(make_hook(name)))

    all_labels = []
    for i, (imgs, labs) in enumerate(loader):
        with torch.no_grad():
            model(imgs.to(DEVICE))
        all_labels.append(labs.numpy())
        if (i + 1) % 100 == 0:
            print(f"  {(i+1)*BATCH_SIZE}/{len(ds)} samples")

    for h in handles:
        h.remove()

    labels = np.concatenate(all_labels)
    features = {}
    for name in LAYER_NAMES:
        feat = np.concatenate(hook_outputs[name])
        if feat.ndim == 1:
            feat = feat.reshape(-1, LAYER_DIMS[name])
        features[name] = feat

    return features, labels

# ── Analysis ─────────────────────────────────────────────────────────────────

def compute_silhouette(features, labels, n_sample=SILHOUETTE_SAMPLE):
    n = len(labels)
    if n > n_sample:
        rng = np.random.RandomState(42)
        idx = rng.choice(n, n_sample, replace=False)
        features = features[idx]
        labels = labels[idx]

    overall = silhouette_score(features, labels)
    sample_scores = silhouette_samples(features, labels)

    per_class = {}
    for c in range(NUM_CLASSES):
        mask = labels == c
        if mask.sum() > 0:
            per_class[CLASS_NAMES[c]] = float(np.mean(sample_scores[mask]))

    return overall, per_class


def compute_centroid_distances(features, labels):
    centroids = []
    for c in range(NUM_CLASSES):
        centroids.append(features[labels == c].mean(axis=0))
    centroids = np.array(centroids)

    from scipy.spatial.distance import cdist
    dist_matrix = cdist(centroids, centroids, metric="euclidean")
    return dist_matrix


def run_tsne(features, labels, name, split, n_sample=5000):
    n = len(labels)
    if n > n_sample:
        rng = np.random.RandomState(42)
        idx = rng.choice(n, n_sample, replace=False)
        features = features[idx]
        labels = labels[idx]

    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    emb = tsne.fit_transform(features)

    plt.figure(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, NUM_CLASSES))
    for c in range(NUM_CLASSES):
        mask = labels == c
        plt.scatter(emb[mask, 0], emb[mask, 1], c=[colors[c]], label=CLASS_NAMES[c],
                    alpha=0.5, s=8)
    plt.legend(markerscale=3)
    plt.title(f"t-SNE: {name} ({split})")
    plt.tight_layout()
    fname = f"{name}_tsne_{split}.png"
    plt.savefig(os.path.join(OUT_DIR, fname), dpi=150)
    plt.close()
    print(f"  Saved {fname}")

# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Device: {DEVICE}")

    model = load_model()

    results_lines = []
    def log(msg=""):
        print(msg)
        results_lines.append(msg)

    log("=" * 80)
    log("LAYER-WISE FEATURE ANALYSIS — ResNet-18 (exp_03)")
    log("=" * 80)

    # Extract features
    layer_data = {}
    for split, path in [("train", "train"), ("val", "test")]:
        print(f"\nExtracting {split} features from all layers...")
        features, labels = extract_all_layers(model, os.path.join(DATA_DIR, path))
        layer_data[split] = {"features": features, "labels": labels}
        for name in LAYER_NAMES:
            print(f"  {name}: {features[name].shape}")

    # Layer comparison table
    log("\n" + "=" * 80)
    log("LAYER COMPARISON — SILHOUETTE SCORES")
    log("=" * 80)

    header = f"{'Layer':<10} {'Dims':>5} {'Train Sil':>11} {'Val Sil':>9} {'Best Class':>18} {'Worst Class':>18}"
    log(header)
    log("-" * len(header))

    layer_results = {}
    for name in LAYER_NAMES:
        train_sil, train_per_class = compute_silhouette(
            layer_data["train"]["features"][name], layer_data["train"]["labels"])
        val_sil, val_per_class = compute_silhouette(
            layer_data["val"]["features"][name], layer_data["val"]["labels"])

        best_cls = max(train_per_class, key=train_per_class.get)
        worst_cls = min(train_per_class, key=train_per_class.get)

        dim = LAYER_DIMS[name]
        log(f"{name:<10} {dim:>5} {train_sil:>11.4f} {val_sil:>9.4f} "
            f"{best_cls + ' (' + f'{train_per_class[best_cls]:.3f}' + ')':>18} "
            f"{worst_cls + ' (' + f'{train_per_class[worst_cls]:.3f}' + ')':>18}")

        layer_results[name] = {
            "train_sil": train_sil, "val_sil": val_sil,
            "train_per_class": train_per_class, "val_per_class": val_per_class,
        }

    # Per-class silhouette detail
    log("\n" + "=" * 80)
    log("PER-CLASS SILHOUETTE SCORES (TRAIN)")
    log("=" * 80)

    header2 = f"{'Class':<10}" + "".join(f"{name:>12}" for name in LAYER_NAMES)
    log(header2)
    log("-" * len(header2))
    for cls_name in CLASS_NAMES:
        vals = [layer_results[name]["train_per_class"].get(cls_name, 0) for name in LAYER_NAMES]
        row = f"{cls_name:<10}" + "".join(f"{v:>12.4f}" for v in vals)
        log(row)

    log("\nPER-CLASS SILHOUETTE SCORES (VAL)")
    log("-" * len(header2))
    for cls_name in CLASS_NAMES:
        vals = [layer_results[name]["val_per_class"].get(cls_name, 0) for name in LAYER_NAMES]
        row = f"{cls_name:<10}" + "".join(f"{v:>12.4f}" for v in vals)
        log(row)

    # Inter-class distance matrices
    log("\n" + "=" * 80)
    log("INTER-CLASS CENTROID DISTANCES (TRAIN, AVGPOOL)")
    log("=" * 80)

    dist_matrix = compute_centroid_distances(
        layer_data["train"]["features"]["avgpool"], layer_data["train"]["labels"])

    header3 = f"{'':>10}" + "".join(f"{c[:7]:>10}" for c in CLASS_NAMES)
    log(header3)
    for i, cls_name in enumerate(CLASS_NAMES):
        row = f"{cls_name[:7]:>10}" + "".join(f"{dist_matrix[i, j]:>10.3f}" for j in range(NUM_CLASSES))
        log(row)

    # Closest and farthest pairs
    log("\nClosest class pairs (hardest to separate):")
    pairs = []
    for i in range(NUM_CLASSES):
        for j in range(i + 1, NUM_CLASSES):
            pairs.append((dist_matrix[i, j], CLASS_NAMES[i], CLASS_NAMES[j]))
    pairs.sort()
    for dist, c1, c2 in pairs[:5]:
        log(f"  {c1:>10} <-> {c2:<10}  dist={dist:.3f}")

    # t-SNE plots
    log("\n" + "=" * 80)
    log("GENERATING t-SNE PLOTS")
    log("=" * 80)

    for name in LAYER_NAMES:
        for split in ["train", "val"]:
            run_tsne(layer_data[split]["features"][name],
                     layer_data[split]["labels"], name, split)

    # PCA analysis
    log("\n" + "=" * 80)
    log("PCA DIMENSIONALITY REDUCTION — AVGPOOL FEATURES")
    log("=" * 80)

    pca_dims = [512, 200, 100, 50]
    pca_header = f"{'PCA Dims':>10} {'Train Sil':>11} {'Val Sil':>9}"
    log(pca_header)
    log("-" * len(pca_header))

    train_feat = layer_data["train"]["features"]["avgpool"]
    val_feat = layer_data["val"]["features"]["avgpool"]
    train_labels = layer_data["train"]["labels"]
    val_labels = layer_data["val"]["labels"]

    for dim in pca_dims:
        if dim >= train_feat.shape[1]:
            t_sil, _ = compute_silhouette(train_feat, train_labels)
            v_sil, _ = compute_silhouette(val_feat, val_labels)
            label = f"{dim} (raw)"
        else:
            pca = PCA(n_components=dim, random_state=42)
            t_reduced = pca.fit_transform(train_feat)
            v_reduced = pca.transform(val_feat)
            t_sil, _ = compute_silhouette(t_reduced, train_labels)
            v_sil, _ = compute_silhouette(v_reduced, val_labels)
            var = pca.explained_variance_ratio_.sum()
            label = f"{dim} ({var:.1%})"

        log(f"{label:>10} {t_sil:>11.4f} {v_sil:>9.4f}")

    # Save results
    results_path = os.path.join(OUT_DIR, "results.txt")
    with open(results_path, "w") as f:
        f.write("\n".join(results_lines) + "\n")
    print(f"\nAll results saved to {results_path}")
    print(f"Plots saved to {OUT_DIR}/")
