"""
Matched silhouette comparison: FER2013 vs FERPlus on the SAME fresh
frozen ImageNet ResNet-18 backbone. Label quality is the only variable.
"""

import os, sys, copy, json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, models, transforms
from sklearn.metrics import f1_score, silhouette_score, silhouette_samples
from collections import Counter
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NUM_CLASSES = 7
CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
FERPLUS_FOLDER_MAP = {
    "anger": 0, "disgust": 1, "fear": 2, "happiness": 3,
    "neutral": 4, "sadness": 5, "surprise": 6,
}
SEED = 42
DROPOUT = 0.3
BATCH_SIZE = 64
HEAD_EPOCHS = 50

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
EXP_DIR     = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
BASE_DIR    = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
FER2013_DIR = os.path.join(BASE_DIR, "data")
FERPLUS_DIR = os.path.join(BASE_DIR, "Fer_plus", "fer2013plus", "fer2013")
RESULTS_DIR = os.path.join(EXP_DIR, "results")
FIGURES_DIR = os.path.join(EXP_DIR, "figures")

DEVICE = "cpu"
torch.manual_seed(SEED)
np.random.seed(SEED)

img_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class FERPlusDataset(torch.utils.data.Dataset):
    def __init__(self, split_dir, transform=None):
        self.image_paths, self.labels = [], []
        self.transform = transform
        for folder_name, class_idx in sorted(FERPLUS_FOLDER_MAP.items(), key=lambda x: x[1]):
            folder_path = os.path.join(split_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue
            for fname in sorted(os.listdir(folder_path)):
                if fname.endswith('.png'):
                    self.image_paths.append(os.path.join(folder_path, fname))
                    self.labels.append(class_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
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


def train_head(train_X, train_y, val_X, val_y):
    train_ds = TensorDataset(torch.FloatTensor(train_X), torch.LongTensor(train_y))
    val_ds   = TensorDataset(torch.FloatTensor(val_X),   torch.LongTensor(val_y))
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False)

    head = nn.Sequential(nn.Dropout(DROPOUT), nn.Linear(512, NUM_CLASSES)).to(DEVICE)
    optimizer = optim.Adam(head.parameters(), lr=1e-4, weight_decay=3e-4)
    criterion = nn.CrossEntropyLoss()

    best_f1, best_state, best_epoch, best_acc = 0.0, None, 0, 0.0
    for epoch in range(HEAD_EPOCHS):
        head.train()
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            criterion(head(X_b), y_b).backward()
            optimizer.step()

        head.eval()
        preds, labels = [], []
        with torch.no_grad():
            for X_b, y_b in val_loader:
                preds.extend(head(X_b.to(DEVICE)).argmax(1).cpu().numpy())
                labels.extend(y_b.numpy())
        f1 = f1_score(labels, preds, average="macro", zero_division=0)
        acc = sum(p == l for p, l in zip(preds, labels)) / len(labels)
        if f1 > best_f1:
            best_f1, best_acc, best_epoch = f1, acc, epoch + 1
            best_state = copy.deepcopy(head.state_dict())

    head.load_state_dict(best_state)
    return head, best_f1, best_acc, best_epoch


def predict(head, features):
    ds = TensorDataset(torch.FloatTensor(features))
    loader = DataLoader(ds, batch_size=512, shuffle=False)
    preds = []
    with torch.no_grad():
        for (X_b,) in loader:
            preds.extend(head(X_b.to(DEVICE)).argmax(1).cpu().numpy())
    return np.array(preds)


def compute_silhouette(feats, labels):
    sample_sil = silhouette_samples(feats, labels, metric='euclidean')
    overall = float(silhouette_score(feats, labels, metric='euclidean'))
    result = {"overall": round(overall, 4)}
    for c in range(NUM_CLASSES):
        mask = labels == c
        result[CLASS_NAMES[c]] = round(float(sample_sil[mask].mean()), 4)
    return result


def plot_silhouette_comparison(fer2013_sil, ferplus_sil, save_path):
    classes = CLASS_NAMES + ["overall"]
    x = np.arange(len(classes))
    width = 0.35

    f13 = [fer2013_sil[c] for c in classes]
    fp = [ferplus_sil[c] for c in classes]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, f13, width, label="FER2013", color="#4e79a7")
    bars2 = ax.bar(x + width/2, fp, width, label="FERPlus", color="#f28e2b")

    ax.set_title("Per-Class Silhouette — FER2013 vs FERPlus\n(Same Frozen ImageNet Backbone, Label Quality Only)", fontsize=13)
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


def main():
    print("=" * 70)
    print("START — Matched Silhouette: FER2013 vs FERPlus (Fresh ImageNet)")
    print("=" * 70)
    print(f"Device: {DEVICE}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Build ONE fresh frozen ImageNet backbone
    print("\nBuilding fresh frozen ImageNet ResNet-18...")
    model = models.resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Sequential(nn.Dropout(DROPOUT), nn.Linear(model.fc.in_features, NUM_CLASSES))
    model.to(DEVICE)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # ── Extract features from both datasets ──────────────────────────────────

    # FER2013
    print("\nExtracting FER2013 features...")
    fer_train = datasets.ImageFolder(os.path.join(FER2013_DIR, "train"), transform=img_transforms)
    fer_test  = datasets.ImageFolder(os.path.join(FER2013_DIR, "test"),  transform=img_transforms)
    fer_train_loader = DataLoader(fer_train, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    fer_test_loader  = DataLoader(fer_test,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    fer_train_feats, fer_train_labs = extract_features(model, fer_train_loader)
    print(f"  Train: {fer_train_feats.shape}, dist: {dict(Counter(fer_train_labs.tolist()))}")
    fer_test_feats, fer_test_labs = extract_features(model, fer_test_loader)
    print(f"  Test:  {fer_test_feats.shape}")

    # FERPlus
    print("\nExtracting FERPlus features...")
    fp_train = FERPlusDataset(os.path.join(FERPLUS_DIR, "train"), transform=img_transforms)
    fp_test  = FERPlusDataset(os.path.join(FERPLUS_DIR, "test"),  transform=img_transforms)
    fp_train_loader = DataLoader(fp_train, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    fp_test_loader  = DataLoader(fp_test,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    fp_train_feats, fp_train_labs = extract_features(model, fp_train_loader)
    print(f"  Train: {fp_train_feats.shape}, dist: {dict(Counter(fp_train_labs.tolist()))}")
    fp_test_feats, fp_test_labs = extract_features(model, fp_test_loader)
    print(f"  Test:  {fp_test_feats.shape}")

    # ── Step 1: FER2013 silhouette ───────────────────────────────────────────

    print("\n" + "=" * 70)
    print("STEP 1: FER2013 Silhouette (fresh ImageNet features)")
    print("=" * 70)

    fer_sil = compute_silhouette(fer_test_feats, fer_test_labs)
    print(f"\n  Overall: {fer_sil['overall']}")
    for c in CLASS_NAMES:
        print(f"  {c:<10s}: {fer_sil[c]}")

    with open(os.path.join(RESULTS_DIR, "silhouette_fer2013_freshimagenet.json"), "w") as f:
        json.dump({"feature_source": "fresh_ImageNet_ResNet18_avgpool", "silhouette": fer_sil}, f, indent=2)
    print("  Saved: silhouette_fer2013_freshimagenet.json")

    # ── Step 2: FERPlus silhouette ───────────────────────────────────────────

    print("\n" + "=" * 70)
    print("STEP 2: FERPlus Silhouette (fresh ImageNet features)")
    print("=" * 70)

    fp_sil = compute_silhouette(fp_test_feats, fp_test_labs)
    print(f"\n  Overall: {fp_sil['overall']}")
    for c in CLASS_NAMES:
        print(f"  {c:<10s}: {fp_sil[c]}")

    with open(os.path.join(RESULTS_DIR, "silhouette_ferplus_freshimagenet.json"), "w") as f:
        json.dump({"feature_source": "fresh_ImageNet_ResNet18_avgpool", "silhouette": fp_sil}, f, indent=2)
    print("  Saved: silhouette_ferplus_freshimagenet.json")

    # ── Step 3: Side-by-side ─────────────────────────────────────────────────

    print("\n" + "=" * 70)
    print("STEP 3: Side-by-Side Comparison (both on fresh ImageNet)")
    print("=" * 70)

    print(f"\n  {'Class':<10s} {'FER2013':>10s} {'FERPlus':>10s} {'Delta':>10s}")
    print("  " + "-" * 42)
    for c in CLASS_NAMES + ["overall"]:
        f13 = fer_sil[c]
        fp = fp_sil[c]
        delta = fp - f13
        sign = "+" if delta >= 0 else ""
        print(f"  {c:<10s} {f13:>10.4f} {fp:>10.4f} {sign}{delta:>9.4f}")

    # ── Baselines: retrained heads on matched backbone ───────────────────────

    print("\n" + "=" * 70)
    print("BASELINES: Retrained heads on fresh ImageNet backbone")
    print("=" * 70)

    # FER2013 baseline
    print("\nFER2013 baseline (fresh ImageNet + linear head)...")
    head_fer, fer_f1, fer_acc, fer_ep = train_head(
        fer_train_feats, fer_train_labs, fer_test_feats, fer_test_labs)
    print(f"  Best epoch: {fer_ep}, F1={fer_f1:.4f}, Acc={fer_acc:.4f}")

    fer_preds = predict(head_fer, fer_test_feats)
    fer_imbal_f1 = f1_score(fer_test_labs, fer_preds, average="macro", zero_division=0)
    fer_imbal_acc = float((fer_preds == fer_test_labs).mean())
    fer_pc = f1_score(fer_test_labs, fer_preds, average=None, labels=list(range(NUM_CLASSES)), zero_division=0)

    print(f"  Macro-F1: {fer_imbal_f1:.4f}, Acc: {fer_imbal_acc:.4f}")
    for i, c in enumerate(CLASS_NAMES):
        print(f"    {c:<10s}: {fer_pc[i]:.4f}")

    # FERPlus baseline
    print("\nFERPlus baseline (fresh ImageNet + linear head)...")
    head_fp, fp_f1, fp_acc, fp_ep = train_head(
        fp_train_feats, fp_train_labs, fp_test_feats, fp_test_labs)
    print(f"  Best epoch: {fp_ep}, F1={fp_f1:.4f}, Acc={fp_acc:.4f}")

    fp_preds = predict(head_fp, fp_test_feats)
    fp_imbal_f1 = f1_score(fp_test_labs, fp_preds, average="macro", zero_division=0)
    fp_imbal_acc = float((fp_preds == fp_test_labs).mean())
    fp_pc = f1_score(fp_test_labs, fp_preds, average=None, labels=list(range(NUM_CLASSES)), zero_division=0)

    print(f"  Macro-F1: {fp_imbal_f1:.4f}, Acc: {fp_imbal_acc:.4f}")
    for i, c in enumerate(CLASS_NAMES):
        print(f"    {c:<10s}: {fp_pc[i]:.4f}")

    # Comparison table
    print(f"\n  {'Config':<30s} {'Macro-F1':>10s} {'Acc':>10s}", end="")
    for c in CLASS_NAMES:
        print(f" {c[:7]:>8s}", end="")
    print()
    print("  " + "-" * 100)
    row = f"  {'FER2013 (fresh IN)':<30s} {fer_imbal_f1:>10.4f} {fer_imbal_acc:>10.4f}"
    for v in fer_pc:
        row += f" {v:>8.4f}"
    print(row)
    row = f"  {'FERPlus (fresh IN)':<30s} {fp_imbal_f1:>10.4f} {fp_imbal_acc:>10.4f}"
    for v in fp_pc:
        row += f" {v:>8.4f}"
    print(row)

    # Save baseline results
    baselines = {
        "feature_source": "fresh_ImageNet_ResNet18_avgpool",
        "FER2013": {
            "macro_f1": round(float(fer_imbal_f1), 4),
            "accuracy": round(float(fer_imbal_acc), 4),
            "per_class_f1": {CLASS_NAMES[i]: round(float(fer_pc[i]), 4) for i in range(NUM_CLASSES)},
            "best_epoch": fer_ep,
        },
        "FERPlus": {
            "macro_f1": round(float(fp_imbal_f1), 4),
            "accuracy": round(float(fp_imbal_acc), 4),
            "per_class_f1": {CLASS_NAMES[i]: round(float(fp_pc[i]), 4) for i in range(NUM_CLASSES)},
            "best_epoch": fp_ep,
        },
    }
    with open(os.path.join(RESULTS_DIR, "baselines_freshimagenet.json"), "w") as f:
        json.dump(baselines, f, indent=2)
    print("\n  Saved: baselines_freshimagenet.json")

    # ── Figure ───────────────────────────────────────────────────────────────

    plot_silhouette_comparison(fer_sil, fp_sil,
                                os.path.join(FIGURES_DIR, "silhouette_comparison_matched.png"))
    print("  Saved: silhouette_comparison_matched.png")

    # ── Saved files ──────────────────────────────────────────────────────────

    print("\n--- Saved files ---")
    for d, label in [(RESULTS_DIR, "results"), (FIGURES_DIR, "figures")]:
        for fn in sorted(os.listdir(d)):
            if os.path.isfile(os.path.join(d, fn)):
                print(f"  {label}/{fn}")

    print("\n" + "=" * 70)
    print("END — Matched Silhouette Comparison")
    print("=" * 70)


if __name__ == "__main__":
    main()
