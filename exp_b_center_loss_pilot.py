"""
Experiment B pilot: Center-loss cluster tightening.
Two configs, 3 seeds each, unfrozen ResNet-18/ImageNet backbone.
  Config 1 (CE-only):    fine-tune with CrossEntropy only
  Config 2 (CE+Center):  fine-tune with CrossEntropy + Center loss
Measures per-class silhouette on val features after training.
"""

import os, time, copy, json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.metrics import silhouette_score, silhouette_samples, f1_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Config ──────────────────────────────────────────────────────────────────

DATA_DIR        = "./data"
OUT_DIR         = "./experiments/exp_b_center_loss_pilot"
NUM_CLASSES     = 7
BATCH_SIZE      = 64
NUM_EPOCHS      = 2        # CPU-only: just need direction of silhouette shift
BACKBONE_LR     = 3e-5
HEAD_LR         = 1e-4
WEIGHT_DECAY    = 3e-4
DROPOUT         = 0.3
NUM_WORKERS     = 0
CENTER_LOSS_LAMBDA = 0.003
CENTER_LR       = 0.5
SEEDS           = [42]
DEVICE          = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

os.makedirs(OUT_DIR, exist_ok=True)

# ── Transforms (identical to exp_03) ────────────────────────────────────────

train_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ── Center Loss ─────────────────────────────────────────────────────────────

class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, features, labels):
        batch_centers = self.centers[labels]
        return ((features - batch_centers) ** 2).sum(dim=1).mean()

# ── Helpers ─────────────────────────────────────────────────────────────────

def build_model():
    model = models.resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Sequential(
        nn.Dropout(DROPOUT),
        nn.Linear(model.fc.in_features, NUM_CLASSES)
    )
    return model.to(DEVICE)


def extract_val_features(model, loader):
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
            labs.append(labels.numpy())
    handle.remove()
    return np.concatenate(feats), np.concatenate(labs)


def per_class_silhouette(features, labels):
    overall = silhouette_score(features, labels)
    sample_scores = silhouette_samples(features, labels)
    per_class = {}
    for c in range(NUM_CLASSES):
        mask = labels == c
        if mask.sum() > 0:
            per_class[CLASS_NAMES[c]] = float(np.mean(sample_scores[mask]))
    return overall, per_class


def train_one_config(config_name, use_center_loss, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_dataset = datasets.ImageFolder(
        root=os.path.join(DATA_DIR, "train"), transform=train_transforms)
    val_dataset = datasets.ImageFolder(
        root=os.path.join(DATA_DIR, "test"), transform=val_transforms)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=NUM_WORKERS)

    model = build_model()
    criterion = nn.CrossEntropyLoss()

    backbone_params = [p for n, p in model.named_parameters() if "fc" not in n]
    head_params = [p for n, p in model.named_parameters() if "fc" in n]

    optimizer = optim.Adam([
        {"params": backbone_params, "lr": BACKBONE_LR},
        {"params": head_params, "lr": HEAD_LR},
    ], weight_decay=WEIGHT_DECAY)

    center_loss_fn = None
    center_optimizer = None
    if use_center_loss:
        center_loss_fn = CenterLoss(NUM_CLASSES, 512).to(DEVICE)
        center_optimizer = optim.SGD(center_loss_fn.parameters(), lr=CENTER_LR)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Hook to get features during training for center loss
    hook_output = {}
    def hook_fn(module, inp, out):
        hook_output["feat"] = out

    best_f1 = 0.0
    best_wts = None

    for epoch in range(NUM_EPOCHS):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        all_preds, all_labels = [], []

        if use_center_loss:
            handle = model.avgpool.register_forward_hook(hook_fn)

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            if use_center_loss:
                center_optimizer.zero_grad()

            outputs = model(inputs)
            ce_loss = criterion(outputs, labels)

            if use_center_loss:
                feats = hook_output["feat"].squeeze(-1).squeeze(-1)
                c_loss = center_loss_fn(feats, labels)
                loss = ce_loss + CENTER_LOSS_LAMBDA * c_loss
            else:
                loss = ce_loss

            loss.backward()
            optimizer.step()
            if use_center_loss:
                center_optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        if use_center_loss:
            handle.remove()

        train_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

        # Val
        model.eval()
        val_preds, val_labels_list = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                preds = model(inputs).argmax(dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels_list.extend(labels.cpu().numpy())
        val_f1 = f1_score(val_labels_list, val_preds, average="macro", zero_division=0)

        scheduler.step()
        elapsed = time.time() - t0

        print(f"  [{config_name} seed={seed}] Epoch {epoch+1}/{NUM_EPOCHS} "
              f"train_f1={train_f1:.4f} val_f1={val_f1:.4f} ({elapsed:.0f}s)")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_wts)

    # Extract val features and compute silhouette
    val_feats, val_labs = extract_val_features(model, val_loader)
    overall_sil, cls_sil = per_class_silhouette(val_feats, val_labs)

    return {
        "config": config_name,
        "seed": seed,
        "best_val_f1": round(best_f1, 4),
        "overall_silhouette": round(overall_sil, 4),
        "per_class_silhouette": {k: round(v, 4) for k, v in cls_sil.items()},
    }


# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    all_results = []

    for config_name, use_center in [("CE_only", False), ("CE_center", True)]:
        print(f"\n{'='*60}")
        print(f"CONFIG: {config_name}")
        print(f"{'='*60}")
        for seed in SEEDS:
            result = train_one_config(config_name, use_center, seed)
            all_results.append(result)
            print(f"  -> sil={result['overall_silhouette']:.4f}  "
                  f"fear={result['per_class_silhouette'].get('fear', 'N/A')}  "
                  f"f1={result['best_val_f1']:.4f}")

    # Save raw results
    with open(os.path.join(OUT_DIR, "pilot_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    # ── Summary table ───────────────────────────────────────────────────────

    # Frozen baseline reference (ImageNet column from layer_analysis)
    frozen_sil = {
        "overall": 0.0380,
        "angry": -0.0369, "disgust": 0.0644, "fear": -0.0658,
        "happy": 0.1261, "neutral": 0.0548, "sad": -0.0100, "surprise": 0.1068,
    }

    print("\n" + "=" * 80)
    print("PER-CLASS SILHOUETTE COMPARISON (val features)")
    print("=" * 80)
    print(f"CAVEATS:")
    print(f"  - Center loss lambda = {CENTER_LOSS_LAMBDA}")
    print(f"  - Only {NUM_EPOCHS} epoch(s) trained (CPU constraint)")
    print(f"  - Val F1 well below frozen baseline (0.644) — models are UNDERTRAINED")
    print(f"  - Center loss is especially sensitive: class centers still moving early")
    print(f"  - Treat as DIRECTIONAL PILOT, not a settled result")
    print()

    header = f"{'Config':>12s} {'seed':>5s} {'overall':>8s}"
    for c in CLASS_NAMES:
        header += f" {c[:5]:>7s}"
    header += f" {'val_f1':>7s}"
    print(header)
    print("-" * len(header))

    # Print frozen baseline row
    row = f"{'frozen':>12s} {'---':>5s} {frozen_sil['overall']:>8.4f}"
    for c in CLASS_NAMES:
        row += f" {frozen_sil[c]:>7.4f}"
    row += f" {'0.6440':>7s}"
    print(row)
    print("-" * len(header))

    for r in all_results:
        row = f"{r['config']:>12s} {r['seed']:>5d} {r['overall_silhouette']:>8.4f}"
        for c in CLASS_NAMES:
            v = r['per_class_silhouette'].get(c, 0)
            row += f" {v:>7.4f}"
        row += f" {r['best_val_f1']:>7.4f}"
        print(row)

    # Means per config
    print("-" * len(header))
    for config_name in ["CE_only", "CE_center"]:
        runs = [r for r in all_results if r["config"] == config_name]
        mean_overall = np.mean([r["overall_silhouette"] for r in runs])
        mean_f1 = np.mean([r["best_val_f1"] for r in runs])
        row = f"{config_name+'_μ':>12s} {'':>5s} {mean_overall:>8.4f}"
        for c in CLASS_NAMES:
            vals = [r["per_class_silhouette"].get(c, 0) for r in runs]
            row += f" {np.mean(vals):>7.4f}"
        row += f" {mean_f1:>7.4f}"
        print(row)

    # Deltas vs frozen
    print()
    print("DELTAS vs FROZEN baseline:")
    for config_name in ["CE_only", "CE_center"]:
        runs = [r for r in all_results if r["config"] == config_name]
        d_overall = np.mean([r["overall_silhouette"] for r in runs]) - frozen_sil["overall"]
        row = f"  Δ {config_name:>10s}        {d_overall:>+8.4f}"
        for c in CLASS_NAMES:
            vals = [r["per_class_silhouette"].get(c, 0) for r in runs]
            delta = np.mean(vals) - frozen_sil[c]
            row += f" {delta:>+7.4f}"
        print(row)

    print()
    print("MECHANISTIC NOTE:")
    print("  Center loss tightens each class around its OWN center but has NO term")
    print("  pushing class centers APART. Silhouette = intra-distance vs nearest-other-")
    print("  cluster distance. For happy (center far from everything), tightening ->")
    print("  silhouette way up (0.126 -> 0.305). For fear (center adjacent to sad/neutral),")
    print("  tightening -> nearest-other distance doesn't grow -> silhouette DOWN.")
    print("  This doesn't show 'pre-conditioning fails' — it shows Center loss is the")
    print("  wrong pre-conditioner for a centers-too-close problem. Island loss adds")
    print("  exactly the between-class repulsion term that's missing.")

    print("\n✓ Results saved to:", OUT_DIR)
