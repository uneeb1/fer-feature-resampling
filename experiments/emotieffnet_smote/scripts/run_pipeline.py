"""
EmotiEffNet-B0 (AffectNet-pretrained) frozen feature extractor on FER2013.
Exploratory experiment: architecture + pretraining + dimension all change vs ResNet-18 baseline.
"""

import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from sklearn.metrics import (
    f1_score, accuracy_score, classification_report, confusion_matrix,
    silhouette_samples,
)
from sklearn.linear_model import LogisticRegression
from collections import Counter
from imblearn.over_sampling import SVMSMOTE
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ── Config ────────────────────────────────────────────────────────────────────

SEED = 42
NUM_CLASSES = 7
CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
N_DRAWS = 100
SAMPLES_PER_CLASS = 55

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
EXP_DIR     = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
BASE_DIR    = os.path.abspath(os.path.join(EXP_DIR, "..", ".."))
DATA_DIR    = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(EXP_DIR, "results")
FIGURES_DIR = os.path.join(EXP_DIR, "figures")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Reference: ImageNet ResNet-18 fine-tuned silhouette scores
RESNET_SIL = {
    "overall": 0.038, "angry": -0.037, "disgust": 0.064, "fear": -0.066,
    "happy": 0.126, "neutral": 0.055, "sad": -0.010, "surprise": 0.107,
}

# Reference: ImageNet ResNet-18 baseline + SMOTE results
RESNET_BASELINE = {"imbal_macro_f1": 0.6440, "bal_macro_f1": 0.6301}
RESNET_SMOTE    = {"imbal_macro_f1": 0.6367, "bal_macro_f1": 0.6647}

print("=" * 70)
print("=== START: EmotiEffNet-B0 Pipeline ===")
print("=" * 70)

# ── Step 1: Load backbone ────────────────────────────────────────────────────

print("\n--- Step 1: Loading EmotiEffNet-B0 backbone ---")
from hsemotion.facial_emotions import HSEmotionRecognizer

MODEL_NAME = "enet_b0_8_best_afew"
rec = HSEmotionRecognizer(model_name=MODEL_NAME, device=DEVICE)
print(f"  Model: {MODEL_NAME}")
print(f"  Architecture: EfficientNet-B0")
print(f"  Pretraining: VGGFace2 → AffectNet (8-class expression)")
print(f"  CONTAMINATION CHECK: Model was trained on AffectNet, NOT FER2013. No train/test contamination.")
print(f"  Usage: Frozen feature extractor (classifier replaced with Identity)")
print(f"  Normalization: ImageNet stats (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]) — as specified by library")
print(f"  Device: {DEVICE}")

# ── Step 2: Feature extraction ───────────────────────────────────────────────

print("\n--- Step 2 & 3: Feature extraction ---")
print("  CAVEAT: FER2013 is 48x48 grayscale → upsampled to 224x224 RGB (bicubic via PIL + bilinear via torchvision)")
print("  Grayscale → RGB: 1-channel replicated to 3 channels")

cached_train = os.path.join(RESULTS_DIR, "train_features.npy")
cached_val   = os.path.join(RESULTS_DIR, "val_features.npy")

if os.path.exists(cached_train) and os.path.exists(cached_val):
    print("  Loading cached features...")
    train_features = np.load(cached_train)
    train_labels   = np.load(os.path.join(RESULTS_DIR, "train_labels.npy"))
    val_features   = np.load(cached_val)
    val_labels     = np.load(os.path.join(RESULTS_DIR, "val_labels.npy"))
    print(f"  Train: {train_features.shape} (cached)")
    print(f"  Val:   {val_features.shape} (cached)")
else:
    def extract_features_from_folder(data_path, rec):
        dataset = datasets.ImageFolder(data_path)
        features_list = []
        labels_list = []
        for i, (img_path, label) in enumerate(dataset.samples):
            img = Image.open(img_path).convert("RGB")
            img_np = np.array(img.resize((224, 224), Image.BICUBIC))
            feat = rec.extract_features(img_np)
            features_list.append(feat.flatten())
            labels_list.append(label)
            if (i + 1) % 5000 == 0:
                print(f"    Processed {i+1}/{len(dataset.samples)}")
        return np.array(features_list), np.array(labels_list)

    t0 = time.time()
    train_features, train_labels = extract_features_from_folder(os.path.join(DATA_DIR, "train"), rec)
    print(f"  Train: {train_features.shape} extracted in {time.time()-t0:.1f}s")

    t0 = time.time()
    val_features, val_labels = extract_features_from_folder(os.path.join(DATA_DIR, "test"), rec)
    print(f"  Val:   {val_features.shape} extracted in {time.time()-t0:.1f}s")

    np.save(os.path.join(RESULTS_DIR, "train_features.npy"), train_features)
    np.save(os.path.join(RESULTS_DIR, "train_labels.npy"), train_labels)
    np.save(os.path.join(RESULTS_DIR, "val_features.npy"), val_features)
    np.save(os.path.join(RESULTS_DIR, "val_labels.npy"), val_labels)

print(f"  Embedding dimension: {train_features.shape[1]} (vs 512 for ResNet-18)")
print(f"  NOTE: This run changes architecture, pretraining, AND dimension simultaneously — exploratory, not controlled.")

# ── Step 4: Silhouette analysis ──────────────────────────────────────────────

print("\n--- Step 4: Silhouette analysis (val set) ---")
sil_vals = silhouette_samples(val_features, val_labels, metric="euclidean")
sil_results = {"overall": float(np.mean(sil_vals))}
print(f"  Overall silhouette: {sil_results['overall']:.4f} (ResNet-18: {RESNET_SIL['overall']:.4f})")

for c in range(NUM_CLASSES):
    mask = val_labels == c
    score = float(np.mean(sil_vals[mask]))
    sil_results[CLASS_NAMES[c]] = score
    ref = RESNET_SIL[CLASS_NAMES[c]]
    delta = score - ref
    flag = " <<<" if CLASS_NAMES[c] == "fear" else ""
    print(f"  {CLASS_NAMES[c]:>10s}: {score:+.4f} (ResNet-18: {ref:+.4f}, Δ={delta:+.4f}){flag}")

print(f"\n  FEAR highlight: {sil_results['fear']:+.4f} vs ResNet-18 {RESNET_SIL['fear']:+.4f} (Δ={sil_results['fear']-RESNET_SIL['fear']:+.4f})")

with open(os.path.join(RESULTS_DIR, "silhouette_emotieffnet.json"), "w") as f:
    json.dump(sil_results, f, indent=2)
print(f"  Saved: {os.path.join(RESULTS_DIR, 'silhouette_emotieffnet.json')}")

# ── Step 5a: Baseline linear head ────────────────────────────────────────────

print("\n--- Step 5a: Baseline (linear head on frozen embeddings) ---")
clf_base = LogisticRegression(max_iter=2000, random_state=SEED, solver="lbfgs")
clf_base.fit(train_features, train_labels)

val_pred_base = clf_base.predict(val_features)
base_acc = accuracy_score(val_labels, val_pred_base)
base_macro_f1 = f1_score(val_labels, val_pred_base, average="macro")
base_per_class_f1 = f1_score(val_labels, val_pred_base, average=None)

print(f"  Accuracy:  {base_acc:.4f}")
print(f"  Macro F1:  {base_macro_f1:.4f} (ResNet-18: {RESNET_BASELINE['imbal_macro_f1']:.4f})")
for i, name in enumerate(CLASS_NAMES):
    print(f"    {name:>10s}: {base_per_class_f1[i]:.4f}")

base_cm = confusion_matrix(val_labels, val_pred_base)
baseline_results = {
    "accuracy": float(base_acc),
    "macro_f1": float(base_macro_f1),
    "per_class_f1": {CLASS_NAMES[i]: float(base_per_class_f1[i]) for i in range(NUM_CLASSES)},
    "confusion_matrix": base_cm.tolist(),
}
with open(os.path.join(RESULTS_DIR, "emotieffnet_baseline.json"), "w") as f:
    json.dump(baseline_results, f, indent=2)
print(f"  Saved: {os.path.join(RESULTS_DIR, 'emotieffnet_baseline.json')}")

# ── Step 5b: SVMSMOTE ───────────────────────────────────────────────────────

print("\n--- Step 5b: SVMSMOTE (full balancing) ---")
print(f"  Train distribution before: {dict(sorted(Counter(train_labels).items()))}")

smote = SVMSMOTE(random_state=SEED, sampling_strategy="not majority")
X_resampled, y_resampled = smote.fit_resample(train_features, train_labels)
print(f"  Train distribution after:  {dict(sorted(Counter(y_resampled).items()))}")
print(f"  Resampled: {X_resampled.shape[0]} samples (was {train_features.shape[0]})")

clf_smote = LogisticRegression(max_iter=2000, random_state=SEED, solver="lbfgs")
clf_smote.fit(X_resampled, y_resampled)

# Imbalanced eval
val_pred_smote = clf_smote.predict(val_features)
smote_acc = accuracy_score(val_labels, val_pred_smote)
smote_macro_f1 = f1_score(val_labels, val_pred_smote, average="macro")
smote_per_class_f1 = f1_score(val_labels, val_pred_smote, average=None)

print(f"\n  Imbalanced eval:")
print(f"    Accuracy:  {smote_acc:.4f}")
print(f"    Macro F1:  {smote_macro_f1:.4f} (ResNet-18+SMOTE: {RESNET_SMOTE['imbal_macro_f1']:.4f})")
for i, name in enumerate(CLASS_NAMES):
    print(f"      {name:>10s}: {smote_per_class_f1[i]:.4f}")

smote_cm = confusion_matrix(val_labels, val_pred_smote)

smote_results = {
    "accuracy": float(smote_acc),
    "imbal_macro_f1": float(smote_macro_f1),
    "per_class_f1_imbal": {CLASS_NAMES[i]: float(smote_per_class_f1[i]) for i in range(NUM_CLASSES)},
    "confusion_matrix": smote_cm.tolist(),
}

# Balanced eval (55/class, 100 draws)
print(f"\n  Balanced eval ({SAMPLES_PER_CLASS}/class, {N_DRAWS} draws):")
bal_f1_draws = []
bal_acc_draws = []
bal_per_class_f1_draws = []

for draw in range(N_DRAWS):
    rng = np.random.RandomState(draw)
    indices = []
    for c in range(NUM_CLASSES):
        c_idx = np.where(val_labels == c)[0]
        chosen = rng.choice(c_idx, size=SAMPLES_PER_CLASS, replace=False)
        indices.extend(chosen)
    indices = np.array(indices)
    y_bal = val_labels[indices]
    pred_bal = clf_smote.predict(val_features[indices])
    bal_f1_draws.append(f1_score(y_bal, pred_bal, average="macro"))
    bal_acc_draws.append(accuracy_score(y_bal, pred_bal))
    bal_per_class_f1_draws.append(f1_score(y_bal, pred_bal, average=None))

bal_f1_mean = float(np.mean(bal_f1_draws))
bal_f1_ci = (float(np.percentile(bal_f1_draws, 2.5)), float(np.percentile(bal_f1_draws, 97.5)))
bal_acc_mean = float(np.mean(bal_acc_draws))
bal_per_class_mean = np.mean(bal_per_class_f1_draws, axis=0)
bal_per_class_ci_lo = np.percentile(bal_per_class_f1_draws, 2.5, axis=0)
bal_per_class_ci_hi = np.percentile(bal_per_class_f1_draws, 97.5, axis=0)

print(f"    Balanced Macro F1: {bal_f1_mean:.4f} [{bal_f1_ci[0]:.4f}, {bal_f1_ci[1]:.4f}]")
print(f"    Balanced Accuracy: {bal_acc_mean:.4f}")
print(f"    (ResNet-18+SMOTE balanced F1: {RESNET_SMOTE['bal_macro_f1']:.4f})")
for i, name in enumerate(CLASS_NAMES):
    print(f"      {name:>10s}: {bal_per_class_mean[i]:.4f} [{bal_per_class_ci_lo[i]:.4f}, {bal_per_class_ci_hi[i]:.4f}]")

smote_results["bal_macro_f1"] = bal_f1_mean
smote_results["bal_macro_f1_95ci"] = list(bal_f1_ci)
smote_results["bal_accuracy"] = bal_acc_mean
smote_results["per_class_f1_bal"] = {CLASS_NAMES[i]: float(bal_per_class_mean[i]) for i in range(NUM_CLASSES)}
smote_results["per_class_f1_bal_95ci"] = {
    CLASS_NAMES[i]: [float(bal_per_class_ci_lo[i]), float(bal_per_class_ci_hi[i])] for i in range(NUM_CLASSES)
}

with open(os.path.join(RESULTS_DIR, "emotieffnet_smote.json"), "w") as f:
    json.dump(smote_results, f, indent=2)
print(f"  Saved: {os.path.join(RESULTS_DIR, 'emotieffnet_smote.json')}")

balanced_eval = {
    "protocol": f"{SAMPLES_PER_CLASS}/class, {N_DRAWS} draws, 95% CI",
    "macro_f1_mean": bal_f1_mean,
    "macro_f1_95ci": list(bal_f1_ci),
    "accuracy_mean": bal_acc_mean,
    "per_class_f1_mean": {CLASS_NAMES[i]: float(bal_per_class_mean[i]) for i in range(NUM_CLASSES)},
    "per_class_f1_95ci": {
        CLASS_NAMES[i]: [float(bal_per_class_ci_lo[i]), float(bal_per_class_ci_hi[i])] for i in range(NUM_CLASSES)
    },
}
with open(os.path.join(RESULTS_DIR, "balanced_eval.json"), "w") as f:
    json.dump(balanced_eval, f, indent=2)
print(f"  Saved: {os.path.join(RESULTS_DIR, 'balanced_eval.json')}")

# ── Figures ──────────────────────────────────────────────────────────────────

print("\n--- Figures ---")

# 1. Silhouette comparison
fig, ax = plt.subplots(figsize=(10, 5))
x_pos = np.arange(len(CLASS_NAMES) + 1)
labels = CLASS_NAMES + ["overall"]
resnet_vals = [RESNET_SIL[c] for c in CLASS_NAMES] + [RESNET_SIL["overall"]]
emoti_vals = [sil_results[c] for c in CLASS_NAMES] + [sil_results["overall"]]
w = 0.35
ax.bar(x_pos - w/2, resnet_vals, w, label="ImageNet ResNet-18", color="#4878CF")
ax.bar(x_pos + w/2, emoti_vals, w, label="EmotiEffNet-B0", color="#D65F5F")
ax.set_xticks(x_pos)
ax.set_xticklabels(labels, rotation=30, ha="right")
ax.set_ylabel("Silhouette Score")
ax.set_title("Per-Class Silhouette: ImageNet ResNet-18 vs EmotiEffNet-B0")
ax.legend()
ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
plt.tight_layout()
path = os.path.join(FIGURES_DIR, "silhouette_comparison.png")
fig.savefig(path, dpi=300)
plt.close(fig)
print(f"  Saved: {path}")

# 2. Confusion matrices
for name, cm, title_suffix in [
    ("confusion_matrix_emotieffnet_baseline.png", base_cm, "Baseline"),
    ("confusion_matrix_emotieffnet_smote.png", smote_cm, "SVMSMOTE"),
]:
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"EmotiEffNet-B0 {title_suffix}")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}")

# 3. Per-class F1 comparison (4 configs)
# Load ResNet per-class from existing results if available, else use placeholders
resnet_base_pcf1 = None
resnet_smote_pcf1 = None
for rpath in [
    os.path.join(BASE_DIR, "experiments", "smote_results", "SVMSMOTE_full_42.json"),
]:
    if os.path.exists(rpath):
        with open(rpath) as f:
            d = json.load(f)
            if "per_class_f1" in d:
                resnet_smote_pcf1 = d["per_class_f1"]

for rpath in [
    os.path.join(BASE_DIR, "experiments", "features", "val_features.npy"),
]:
    pass  # baseline per-class from existing results

fig, ax = plt.subplots(figsize=(10, 5))
x_pos = np.arange(len(CLASS_NAMES))
w = 0.2
ax.bar(x_pos - 1.5*w, base_per_class_f1, w, label="EmotiEffNet Baseline", color="#4878CF")
ax.bar(x_pos - 0.5*w, smote_per_class_f1, w, label="EmotiEffNet SMOTE", color="#D65F5F")
if resnet_smote_pcf1:
    rsm_vals = [resnet_smote_pcf1.get(c, 0) for c in CLASS_NAMES]
    ax.bar(x_pos + 0.5*w, rsm_vals, w, label="ResNet-18 SMOTE", color="#6ACC65", alpha=0.8)
ax.set_xticks(x_pos)
ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right")
ax.set_ylabel("F1 Score")
ax.set_title("Per-Class F1 Comparison")
ax.legend()
plt.tight_layout()
path = os.path.join(FIGURES_DIR, "per_class_f1_comparison.png")
fig.savefig(path, dpi=300)
plt.close(fig)
print(f"  Saved: {path}")

# ── Final comparison table ───────────────────────────────────────────────────

print("\n" + "=" * 110)
print("COMPARISON TABLE")
print("=" * 110)
header = f"{'Config':<30s} {'Imbal F1':>9s} {'Bal F1':>9s}"
for c in CLASS_NAMES:
    header += f" {c:>8s}"
print(header)
print("-" * 110)

# ResNet baseline
row = f"{'ImageNet ResNet-18 base':<30s} {0.6440:>9.4f} {0.6301:>9.4f}"
for c in CLASS_NAMES:
    row += f" {'---':>8s}"
print(row)

# ResNet SMOTE
row = f"{'ImageNet + SVMSMOTE':<30s} {0.6367:>9.4f} {0.6647:>9.4f}"
if resnet_smote_pcf1:
    for c in CLASS_NAMES:
        row += f" {resnet_smote_pcf1.get(c, 0):>8.4f}"
else:
    for c in CLASS_NAMES:
        row += f" {'---':>8s}"
print(row)

# EmotiEffNet baseline
row = f"{'EmotiEffNet baseline':<30s} {base_macro_f1:>9.4f} {'n/a':>9s}"
for i in range(NUM_CLASSES):
    row += f" {base_per_class_f1[i]:>8.4f}"
print(row)

# EmotiEffNet SMOTE
row = f"{'EmotiEffNet + SVMSMOTE':<30s} {smote_macro_f1:>9.4f} {bal_f1_mean:>9.4f}"
for i in range(NUM_CLASSES):
    row += f" {smote_per_class_f1[i]:>8.4f}"
print(row)

print("=" * 110)

# ── Summary ──────────────────────────────────────────────────────────────────

print("\n--- All output files ---")
for f in sorted(os.listdir(RESULTS_DIR)):
    print(f"  results/{f}")
for f in sorted(os.listdir(FIGURES_DIR)):
    print(f"  figures/{f}")

print("\n" + "=" * 70)
print("=== END: EmotiEffNet-B0 Pipeline ===")
print("=" * 70)
