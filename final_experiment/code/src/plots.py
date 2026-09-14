import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, silhouette_samples

CLASSES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
COLORS = ["#e74c3c", "#8e44ad", "#2980b9", "#f39c12", "#1abc9c", "#e67e22", "#7f8c8d"]


def plot_class_distribution(splits_counts, save_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(7)
    w = 0.25
    for i, (split, counts) in enumerate(splits_counts.items()):
        vals = [counts.get(c, 0) for c in range(7)]
        bars = ax.bar(x + i * w, vals, w, label=split)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                    str(v), ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x + w)
    ax.set_xticklabels(CLASSES, rotation=30)
    ax.set_ylabel("Count")
    ax.set_title("FER2013 Class Distribution by Split")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_training_curves(history, save_path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(epochs, history["train_loss"], label="Train Loss")
    ax1.plot(epochs, history["val_loss"], label="Val Loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss"); ax1.legend()
    ax2.plot(epochs, history["val_f1"], label="Val Macro-F1", color="green")
    ax2.plot(epochs, history["val_acc"], label="Val Accuracy", color="blue")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Score")
    ax2.set_title("Validation Macro-F1 & Accuracy"); ax2.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_lr_schedule(history, save_path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(history["lr"]) + 1), history["lr"])
    ax.set_xlabel("Epoch"); ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule (Warmup + Cosine)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_confusion_matrix(labels, preds, save_path, normalize=False):
    cm = confusion_matrix(labels, preds)
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(7)); ax.set_yticks(range(7))
    ax.set_xticklabels(CLASSES, rotation=45, ha="right")
    ax.set_yticklabels(CLASSES)
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(7):
        for j in range(7):
            ax.text(j, i, format(cm[i, j], fmt), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=9)
    title = "Confusion Matrix (Normalized)" if normalize else "Confusion Matrix (Counts)"
    ax.set_title(title); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_per_class_f1(per_class_f1, save_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(CLASSES, per_class_f1, color=COLORS)
    for bar, v in zip(bars, per_class_f1):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("F1 Score"); ax.set_title("Per-Class F1 (Test)")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_tsne(features, labels, save_path, title="t-SNE", max_samples=5000):
    if len(features) > max_samples:
        idx = np.random.RandomState(42).choice(len(features), max_samples, replace=False)
        features, labels = features[idx], labels[idx]
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    emb = tsne.fit_transform(features)
    fig, ax = plt.subplots(figsize=(10, 8))
    for c in range(7):
        mask = labels == c
        ax.scatter(emb[mask, 0], emb[mask, 1], s=5, alpha=0.5, label=CLASSES[c], color=COLORS[c])
    ax.legend(markerscale=4); ax.set_title(title)
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_silhouette(features, labels, save_path, max_samples=5000):
    if len(features) > max_samples:
        idx = np.random.RandomState(42).choice(len(features), max_samples, replace=False)
        features, labels = features[idx], labels[idx]
    sil_vals = silhouette_samples(features, labels)
    per_class = []
    for c in range(7):
        mask = labels == c
        per_class.append(sil_vals[mask].mean())
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(CLASSES, per_class, color=COLORS)
    for bar, v in zip(bars, per_class):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=10)
    ax.axhline(y=0, color="black", linestyle="--", linewidth=0.8)
    ax.set_ylabel("Mean Silhouette Score"); ax.set_title("Per-Class Silhouette (Test Features)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    return per_class
