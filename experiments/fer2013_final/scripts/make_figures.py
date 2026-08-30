"""Step 2: Generate all thesis figures (300 dpi PNG)."""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(BASE, "results", "baseline")
LOGS = os.path.join(BASE, "logs")
FIGURES = os.path.join(BASE, "figures", "baseline")
DATA = os.path.join(BASE, "data_lossless")

CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
CLASS_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
SEED = 42

# Validated categorical palette (7 classes = 7 slots)
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "axes.grid": False,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
})

def save(fig, name):
    path = os.path.join(FIGURES, name)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path

def fig1_class_distribution():
    """Bar chart of per-class TRAIN counts."""
    print("\nFigure 1: Class Distribution")
    counts = []
    for cls in CLASSES:
        counts.append(len(os.listdir(os.path.join(DATA, "train", cls))))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(CLASS_LABELS, counts, color=COLORS, width=0.65, edgecolor="white", linewidth=0.5)
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 80,
                f"{c:,}", ha="center", va="bottom", fontsize=9, color="#52514e")
    ax.set_ylabel("Number of Samples")
    ax.set_title("FER2013 Training Set — Class Distribution")
    ax.set_ylim(0, max(counts) * 1.12)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    return save(fig, "class_distribution.png")

def fig2_training_curves():
    """Train vs val accuracy and loss per epoch."""
    print("\nFigure 2: Training Curves")
    with open(os.path.join(LOGS, f"baseline_history_s{SEED}.json")) as f:
        history = json.load(f)

    epochs = [h["epoch"] for h in history]
    train_acc = [h["train_accuracy"] for h in history]
    val_acc = [h["val_accuracy"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]

    with open(os.path.join(RESULTS, f"baseline_s{SEED}.json")) as f:
        results = json.load(f)
    best_epoch = results["best_epoch"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(epochs, train_acc, color="#2a78d6", linewidth=2, label="Train")
    ax1.plot(epochs, val_acc, color="#eb6834", linewidth=2, label="Validation")
    ax1.axvline(best_epoch, color="#52514e", linestyle="--", linewidth=1, alpha=0.7, label=f"Best epoch ({best_epoch})")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("(a) Accuracy")
    ax1.legend(frameon=False, fontsize=9)

    ax2.plot(epochs, train_loss, color="#2a78d6", linewidth=2, label="Train")
    ax2.plot(epochs, val_loss, color="#eb6834", linewidth=2, label="Validation")
    ax2.axvline(best_epoch, color="#52514e", linestyle="--", linewidth=1, alpha=0.7, label=f"Best epoch ({best_epoch})")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.set_title("(b) Loss")
    ax2.legend(frameon=False, fontsize=9)

    fig.suptitle("Baseline Training Curves", fontsize=13, y=1.02)
    fig.tight_layout()
    return save(fig, "training_curves.png")

def fig3_confusion_matrix():
    """Confusion matrix on TEST set — raw counts and row-normalized."""
    print("\nFigure 3: Confusion Matrix")
    with open(os.path.join(RESULTS, f"baseline_s{SEED}.json")) as f:
        results = json.load(f)
    cm = np.array(results["test"]["confusion_matrix"])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax, data, fmt, title in [
        (ax1, cm, "d", "(a) Raw Counts"),
        (ax2, cm_norm, ".2f", "(b) Row-Normalized"),
    ]:
        im = ax.imshow(data, cmap="Blues", aspect="auto")
        ax.set_xticks(range(7))
        ax.set_yticks(range(7))
        ax.set_xticklabels(CLASS_LABELS, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(CLASS_LABELS, fontsize=9)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title)
        thresh = data.max() / 2
        for i in range(7):
            for j in range(7):
                val = format(data[i, j], fmt)
                color = "white" if data[i, j] > thresh else "black"
                ax.text(j, i, val, ha="center", va="center", fontsize=8, color=color)

    fig.suptitle("Baseline Confusion Matrix — Test Set", fontsize=13, y=1.02)
    fig.tight_layout()
    return save(fig, "confusion_matrix_baseline.png")

def fig4_per_class_f1():
    """Bar chart of per-class F1 on test with macro-F1 reference line."""
    print("\nFigure 4: Per-Class F1")
    with open(os.path.join(RESULTS, f"baseline_s{SEED}.json")) as f:
        results = json.load(f)
    f1s = [results["test"]["per_class_f1"][cls] for cls in CLASSES]
    macro = results["test"]["macro_f1"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(CLASS_LABELS, f1s, color=COLORS, width=0.65, edgecolor="white", linewidth=0.5)
    ax.axhline(macro, color="#52514e", linestyle="--", linewidth=1.5, label=f"Macro-F1 = {macro:.3f}")
    for bar, f1 in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{f1:.3f}", ha="center", va="bottom", fontsize=9, color="#52514e")
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-Class F1 Score — Baseline on Test Set")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=10)
    return save(fig, "per_class_f1_baseline.png")

def fig5_tsne():
    """t-SNE of 512-d test features from fine-tuned baseline backbone."""
    print("\nFigure 5: t-SNE")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, 7))
    model.load_state_dict(torch.load(os.path.join(RESULTS, f"best_model_s{SEED}.pth"),
                                     map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    # Feature extractor: everything before fc
    feature_model = nn.Sequential(*list(model.children())[:-1])  # ends at avgpool

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_ds = datasets.ImageFolder(os.path.join(DATA, "test"), transform=transform)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4)

    features = []
    labels = []
    with torch.no_grad():
        for images, lbls in test_loader:
            images = images.to(device)
            feat = feature_model(images).squeeze(-1).squeeze(-1)
            features.append(feat.cpu().numpy())
            labels.append(lbls.numpy())
    features = np.concatenate(features)
    labels = np.concatenate(labels)
    print(f"  Extracted features: {features.shape}")

    # Save features for Stage 2
    np.savez(os.path.join(RESULTS, f"test_features_s{SEED}.npz"),
             features=features, labels=labels)

    tsne = TSNE(n_components=2, perplexity=30, random_state=SEED, max_iter=1000)
    coords = tsne.fit_transform(features)

    fig, ax = plt.subplots(figsize=(8, 7))
    for i, (cls, color) in enumerate(zip(CLASS_LABELS, COLORS)):
        mask = labels == i
        ax.scatter(coords[mask, 0], coords[mask, 1], c=color, s=8, alpha=0.6, label=cls, rasterized=True)
    ax.set_title("t-SNE of Baseline Test Features (512-d)")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(frameon=False, fontsize=9, markerscale=3, loc="best")
    ax.set_xticks([])
    ax.set_yticks([])
    return save(fig, "tsne_baseline.png")

def main():
    os.makedirs(FIGURES, exist_ok=True)
    print("=" * 60)
    print("STEP 2: Generating thesis figures")
    print("=" * 60)

    paths = []
    paths.append(fig1_class_distribution())
    paths.append(fig2_training_curves())
    paths.append(fig3_confusion_matrix())
    paths.append(fig4_per_class_f1())
    paths.append(fig5_tsne())

    print("\n" + "=" * 60)
    print(f"ALL {len(paths)} FIGURES SAVED:")
    for p in paths:
        print(f"  {p}")
    print("=" * 60)

if __name__ == "__main__":
    main()
