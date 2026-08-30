"""
FERPlus baseline (frozen ImageNet ResNet-18) + silhouette analysis.
Drop contempt, remap class names, 7-class setup matching FER2013.
"""

import os, sys, copy, json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, models, transforms
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.metrics import silhouette_score, silhouette_samples
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Config ────────────────────────────────────────────────────────────────────

NUM_CLASSES = 7
CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
FERPLUS_FOLDER_MAP = {
    "anger": 0, "disgust": 1, "fear": 2, "happiness": 3,
    "neutral": 4, "sadness": 5, "surprise": 6,
}
BATCH_SIZE = 64
DROPOUT = 0.3
SEED = 42
HEAD_EPOCHS = 50

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
EXP_DIR     = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
BASE_DIR    = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
FERPLUS_DIR = os.path.join(BASE_DIR, "Fer_plus", "fer2013plus", "fer2013")
RESULTS_DIR = os.path.join(EXP_DIR, "results")
FIGURES_DIR = os.path.join(EXP_DIR, "figures")

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

img_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ── Helpers ───────────────────────────────────────────────────────────────────

def build_model():
    model = models.resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Sequential(nn.Dropout(DROPOUT), nn.Linear(model.fc.in_features, NUM_CLASSES))
    return model


def load_ferplus_split(split_dir):
    """Load FERPlus images, drop contempt, remap labels."""
    all_images = []
    all_labels = []
    for folder_name, class_idx in sorted(FERPLUS_FOLDER_MAP.items(), key=lambda x: x[1]):
        folder_path = os.path.join(split_dir, folder_name)
        if not os.path.isdir(folder_path):
            print(f"  WARNING: folder not found: {folder_path}")
            continue
        files = sorted([f for f in os.listdir(folder_path) if f.endswith('.png')])
        for fname in files:
            all_images.append(os.path.join(folder_path, fname))
            all_labels.append(class_idx)
    return all_images, all_labels


class FERPlusDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        from PIL import Image
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


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
            labs.append(labels.numpy() if isinstance(labels, torch.Tensor) else np.array(labels))
    handle.remove()
    return np.concatenate(feats), np.concatenate(labs)


def train_head(train_X, train_y, val_X, val_y, num_epochs=HEAD_EPOCHS):
    train_ds = TensorDataset(torch.FloatTensor(train_X), torch.LongTensor(train_y))
    val_ds   = TensorDataset(torch.FloatTensor(val_X),   torch.LongTensor(val_y))
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False)

    head = nn.Sequential(nn.Dropout(DROPOUT), nn.Linear(512, NUM_CLASSES)).to(DEVICE)
    optimizer = optim.Adam(head.parameters(), lr=1e-4, weight_decay=3e-4)
    criterion = nn.CrossEntropyLoss()

    best_f1, best_state, best_epoch, best_acc = 0.0, None, 0, 0.0

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
        val_acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_acc = val_acc
            best_epoch = epoch + 1
            best_state = copy.deepcopy(head.state_dict())

    head.load_state_dict(best_state)
    return head, best_f1, best_acc, best_epoch


def predict(head, features):
    ds = TensorDataset(torch.FloatTensor(features))
    loader = DataLoader(ds, batch_size=512, shuffle=False)
    preds = []
    with torch.no_grad():
        for (X_batch,) in loader:
            out = head(X_batch.to(DEVICE))
            preds.extend(out.argmax(1).cpu().numpy())
    return np.array(preds)


# ── Figures ───────────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, title, save_path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    im0 = axes[0].imshow(cm, interpolation='nearest', cmap='Blues')
    axes[0].set_title(f"{title} — Counts", fontsize=12)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            axes[0].text(j, i, str(cm[i, j]), ha='center', va='center',
                         color='white' if cm[i, j] > cm.max()/2 else 'black', fontsize=8)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
    axes[1].set_title(f"{title} — Row-Normalized", fontsize=12)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            axes[1].text(j, i, f"{cm_norm[i, j]:.2f}", ha='center', va='center',
                         color='white' if cm_norm[i, j] > 0.5 else 'black', fontsize=8)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.set_xticks(range(NUM_CLASSES))
        ax.set_yticks(range(NUM_CLASSES))
        ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(CLASS_NAMES, fontsize=9)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_class_counts_comparison(ferplus_train, ferplus_test, fer2013_train, fer2013_test, save_path):
    x = np.arange(NUM_CLASSES)
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Train
    ax = axes[0]
    fp_vals = [ferplus_train.get(c, 0) for c in CLASS_NAMES]
    f13_vals = [fer2013_train.get(c, 0) for c in CLASS_NAMES]
    ax.bar(x - width/2, f13_vals, width, label="FER2013", color="#4e79a7")
    ax.bar(x + width/2, fp_vals, width, label="FERPlus", color="#f28e2b")
    ax.set_title("Train Split — Class Counts", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    for i, (v1, v2) in enumerate(zip(f13_vals, fp_vals)):
        ax.text(i - width/2, v1 + 100, str(v1), ha='center', va='bottom', fontsize=7)
        ax.text(i + width/2, v2 + 100, str(v2), ha='center', va='bottom', fontsize=7)

    # Test
    ax = axes[1]
    fp_vals = [ferplus_test.get(c, 0) for c in CLASS_NAMES]
    f13_vals = [fer2013_test.get(c, 0) for c in CLASS_NAMES]
    ax.bar(x - width/2, f13_vals, width, label="FER2013", color="#4e79a7")
    ax.bar(x + width/2, fp_vals, width, label="FERPlus", color="#f28e2b")
    ax.set_title("Test Split — Class Counts", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    for i, (v1, v2) in enumerate(zip(f13_vals, fp_vals)):
        ax.text(i - width/2, v1 + 20, str(v1), ha='center', va='bottom', fontsize=7)
        ax.text(i + width/2, v2 + 20, str(v2), ha='center', va='bottom', fontsize=7)

    plt.suptitle("FER2013 vs FERPlus — Per-Class Sample Counts (7 classes, contempt dropped)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_silhouette_comparison(fer2013_sil, ferplus_sil, save_path):
    classes = CLASS_NAMES + ["overall"]
    x = np.arange(len(classes))
    width = 0.35

    f13_vals = [fer2013_sil[c] for c in classes]
    fp_vals = [ferplus_sil[c] for c in classes]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, f13_vals, width, label="FER2013", color="#4e79a7")
    bars2 = ax.bar(x + width/2, fp_vals, width, label="FERPlus", color="#f28e2b")

    ax.set_title("Per-Class Silhouette Score — FER2013 vs FERPlus (Frozen ImageNet Features)", fontsize=13)
    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Silhouette Score", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=10)
    ax.legend(fontsize=11)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(axis='y', alpha=0.3)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            va = 'bottom' if h >= 0 else 'top'
            offset = 0.003 if h >= 0 else -0.003
            ax.annotate(f'{h:.3f}', xy=(bar.get_x() + bar.get_width()/2, h + offset),
                        ha='center', va=va, fontsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("START — FERPlus Baseline + Silhouette Analysis")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"FERPlus data: {FERPLUS_DIR}")
    print()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # ── Load FERPlus data (drop contempt) ────────────────────────────────────

    print("Loading FERPlus data (7 classes, contempt dropped)...")
    train_imgs, train_labels = load_ferplus_split(os.path.join(FERPLUS_DIR, "train"))
    test_imgs, test_labels = load_ferplus_split(os.path.join(FERPLUS_DIR, "test"))
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    ferplus_train_counts = {CLASS_NAMES[c]: int((train_labels == c).sum()) for c in range(NUM_CLASSES)}
    ferplus_test_counts = {CLASS_NAMES[c]: int((test_labels == c).sum()) for c in range(NUM_CLASSES)}

    print(f"  Train: {len(train_labels)} images")
    for c in CLASS_NAMES:
        print(f"    {c:<10s}: {ferplus_train_counts[c]}")
    print(f"  Test:  {len(test_labels)} images")
    for c in CLASS_NAMES:
        print(f"    {c:<10s}: {ferplus_test_counts[c]}")

    train_dataset = FERPlusDataset(train_imgs, train_labels, transform=img_transforms)
    test_dataset  = FERPlusDataset(test_imgs, test_labels, transform=img_transforms)
    train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader   = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # ── Class counts comparison figure ───────────────────────────────────────

    fer2013_train = {"angry": 3995, "disgust": 436, "fear": 4097, "happy": 7215,
                     "neutral": 4965, "sad": 4830, "surprise": 3171}
    fer2013_test  = {"angry": 958, "disgust": 111, "fear": 1024, "happy": 1774,
                     "neutral": 1233, "sad": 1247, "surprise": 831}

    plot_class_counts_comparison(ferplus_train_counts, ferplus_test_counts,
                                  fer2013_train, fer2013_test,
                                  os.path.join(FIGURES_DIR, "class_counts_comparison.png"))
    print("\n  Saved: class_counts_comparison.png")

    # ── Build frozen model + extract features ────────────────────────────────

    print("\nBuilding frozen ImageNet ResNet-18...")
    model = build_model()
    model = model.to(DEVICE)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    print("Extracting train features...")
    train_feats, train_labs = extract_features(model, train_loader)
    print(f"  Shape: {train_feats.shape}")

    print("Extracting test features...")
    test_feats, test_labs = extract_features(model, test_loader)
    print(f"  Shape: {test_feats.shape}")

    # ── Step 1: Baseline ─────────────────────────────────────────────────────

    print("\n" + "=" * 70)
    print("STEP 1: FERPlus Baseline (frozen ImageNet, linear head)")
    print("=" * 70)

    head, best_f1, best_acc, best_epoch = train_head(
        train_feats, train_labs, test_feats, test_labs)
    print(f"  Best epoch: {best_epoch}, F1={best_f1:.4f}, Acc={best_acc:.4f}")

    preds = predict(head, test_feats)
    imbal_f1 = f1_score(test_labs, preds, average="macro", zero_division=0)
    imbal_acc = float((preds == test_labs).mean())
    imbal_pc = f1_score(test_labs, preds, average=None,
                         labels=list(range(NUM_CLASSES)), zero_division=0)

    print(f"\n  Imbalanced val results:")
    print(f"    Macro-F1:  {imbal_f1:.4f}")
    print(f"    Accuracy:  {imbal_acc:.4f}")
    for i, c in enumerate(CLASS_NAMES):
        print(f"    {c:<10s}: {imbal_pc[i]:.4f}")

    baseline_results = {
        "method": "FERPlus_frozen_ImageNet_baseline",
        "seed": SEED,
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "head_best_epoch": best_epoch,
        "imbalanced_macro_f1": round(float(imbal_f1), 4),
        "imbalanced_accuracy": round(float(imbal_acc), 4),
        "imbalanced_per_class_f1": {CLASS_NAMES[i]: round(float(imbal_pc[i]), 4) for i in range(NUM_CLASSES)},
        "train_class_counts": ferplus_train_counts,
        "test_class_counts": ferplus_test_counts,
        "note": "FERPlus class distribution differs from FER2013: neutral is majority (10308 train), fear collapsed to 652, disgust to 191. Contempt dropped.",
    }
    with open(os.path.join(RESULTS_DIR, "baseline_ferplus.json"), "w") as f:
        json.dump(baseline_results, f, indent=2)
    print("  Saved: baseline_ferplus.json")

    # Confusion matrix
    plot_confusion_matrix(test_labs, preds, "FERPlus Baseline (Frozen ImageNet)",
                          os.path.join(FIGURES_DIR, "confusion_matrix_ferplus_baseline.png"))
    print("  Saved: confusion_matrix_ferplus_baseline.png")

    # ── Step 2: Silhouette analysis ──────────────────────────────────────────

    print("\n" + "=" * 70)
    print("STEP 2: Silhouette Analysis (FERPlus val features)")
    print("=" * 70)

    print("Computing silhouette scores on test/val features...")
    print(f"  Feature shape: {test_feats.shape}, {len(test_labs)} labels")

    sample_sil = silhouette_samples(test_feats, test_labs, metric='euclidean')
    overall_sil = float(silhouette_score(test_feats, test_labs, metric='euclidean'))

    ferplus_sil = {"overall": round(overall_sil, 4)}
    print(f"\n  Overall silhouette: {overall_sil:.4f}")
    print(f"\n  Per-class silhouette:")
    for c in range(NUM_CLASSES):
        mask = test_labs == c
        class_sil = float(sample_sil[mask].mean())
        ferplus_sil[CLASS_NAMES[c]] = round(class_sil, 4)
        print(f"    {CLASS_NAMES[c]:<10s}: {class_sil:.4f}")

    # FER2013 reference
    fer2013_sil = {
        "overall": 0.038, "angry": -0.037, "disgust": 0.064, "fear": -0.066,
        "happy": 0.126, "neutral": 0.055, "sad": -0.010, "surprise": 0.107,
    }

    # Comparison table
    print(f"\n  {'Class':<10s} {'FER2013':>10s} {'FERPlus':>10s} {'Delta':>10s}")
    print("  " + "-" * 42)
    for c in CLASS_NAMES + ["overall"]:
        f13 = fer2013_sil[c]
        fp = ferplus_sil[c]
        delta = fp - f13
        sign = "+" if delta >= 0 else ""
        print(f"  {c:<10s} {f13:>10.4f} {fp:>10.4f} {sign}{delta:>9.4f}")

    sil_results = {
        "method": "silhouette_ferplus_val_features",
        "feature_source": "frozen_ImageNet_ResNet18_avgpool",
        "metric": "euclidean",
        "ferplus_silhouette": ferplus_sil,
        "fer2013_silhouette_reference": fer2013_sil,
        "delta": {c: round(ferplus_sil[c] - fer2013_sil[c], 4) for c in CLASS_NAMES + ["overall"]},
    }
    with open(os.path.join(RESULTS_DIR, "silhouette_ferplus.json"), "w") as f:
        json.dump(sil_results, f, indent=2)
    print("\n  Saved: silhouette_ferplus.json")

    # Silhouette comparison figure
    plot_silhouette_comparison(fer2013_sil, ferplus_sil,
                                os.path.join(FIGURES_DIR, "silhouette_comparison.png"))
    print("  Saved: silhouette_comparison.png")

    # ── Final summary ────────────────────────────────────────────────────────

    print("\n--- Saved files ---")
    for d, label in [(RESULTS_DIR, "results"), (FIGURES_DIR, "figures")]:
        for f in sorted(os.listdir(d)):
            fpath = os.path.join(d, f)
            if os.path.isfile(fpath):
                print(f"  {label}/{f}")

    print("\n" + "=" * 70)
    print("END — FERPlus Baseline + Silhouette Analysis")
    print("=" * 70)


if __name__ == "__main__":
    main()
