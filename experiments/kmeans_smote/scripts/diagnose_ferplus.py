"""
Diagnostic checks for FERPlus pipeline vs FER2013.
"""

import os, sys
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from collections import Counter
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
FER2013_DIR = os.path.join(BASE_DIR, "data")
FERPLUS_DIR = os.path.join(BASE_DIR, "Fer_plus", "fer2013plus", "fer2013")
FER2013_CKPT = os.path.join(BASE_DIR, "experiments", "exp_03", "best_model.pth")

NUM_CLASSES = 7
CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
FERPLUS_FOLDER_MAP = {
    "anger": 0, "disgust": 1, "fear": 2, "happiness": 3,
    "neutral": 4, "sadness": 5, "surprise": 6,
}

DEVICE = "cpu"

img_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class FERPlusDataset(torch.utils.data.Dataset):
    def __init__(self, split_dir, transform=None):
        self.image_paths = []
        self.labels = []
        self.folder_names = []
        self.transform = transform
        for folder_name, class_idx in sorted(FERPLUS_FOLDER_MAP.items(), key=lambda x: x[1]):
            folder_path = os.path.join(split_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue
            files = sorted([f for f in os.listdir(folder_path) if f.endswith('.png')])
            for fname in files:
                self.image_paths.append(os.path.join(folder_path, fname))
                self.labels.append(class_idx)
                self.folder_names.append(folder_name)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def build_model():
    model = models.resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.fc.in_features, NUM_CLASSES))
    return model


def main():
    print("=" * 70)
    print("DIAGNOSTIC: FERPlus Pipeline Check")
    print("=" * 70)

    # ── CHECK 1: PREPROCESSING ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("CHECK 1: PREPROCESSING")
    print("=" * 70)

    print("\nTransform pipeline (identical for both):")
    print(f"  {img_transforms}")

    print("\nNormalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]")

    # Sample FER2013 images
    print("\n--- FER2013 tensor stats (3 samples) ---")
    fer_dataset = datasets.ImageFolder(os.path.join(FER2013_DIR, "test"), transform=img_transforms)
    for i in [0, 100, 500]:
        tensor, label = fer_dataset[i]
        path = fer_dataset.samples[i][0]
        fname = os.path.basename(path)
        raw = Image.open(path)
        print(f"  {fname}: raw_mode={raw.mode}, raw_size={raw.size}")
        print(f"    tensor: shape={tensor.shape}, dtype={tensor.dtype}, "
              f"min={tensor.min():.4f}, max={tensor.max():.4f}, mean={tensor.mean():.4f}")

    # Sample FERPlus images
    print("\n--- FERPlus tensor stats (3 samples) ---")
    fp_dataset = FERPlusDataset(os.path.join(FERPLUS_DIR, "test"), transform=img_transforms)
    for i in [0, 100, 500]:
        tensor, label = fp_dataset[i]
        path = fp_dataset.image_paths[i]
        fname = os.path.basename(path)
        raw = Image.open(path)
        print(f"  {fname}: raw_mode={raw.mode}, raw_size={raw.size}")
        print(f"    tensor: shape={tensor.shape}, dtype={tensor.dtype}, "
              f"min={tensor.min():.4f}, max={tensor.max():.4f}, mean={tensor.mean():.4f}")

    # Check if Grayscale(3) on RGB-stored-grayscale gives same result
    print("\n--- RGB-as-grayscale handling check ---")
    fp_path = fp_dataset.image_paths[0]
    raw_rgb = Image.open(fp_path)
    raw_l = raw_rgb.convert("L")
    # Apply Grayscale(3) to RGB
    gs3 = transforms.Grayscale(num_output_channels=3)
    from_rgb = np.array(gs3(raw_rgb))
    from_l = np.array(gs3(raw_l))
    print(f"  Grayscale(3) on RGB vs on L: identical={np.array_equal(from_rgb, from_l)}")
    print(f"  RGB channels before Grayscale: R==G==B={np.array_equal(np.array(raw_rgb)[:,:,0], np.array(raw_rgb)[:,:,1])}")
    arr_rgb = np.array(from_rgb)
    arr_l = np.array(from_l)
    print(f"  After Grayscale(3) from RGB: shape={arr_rgb.shape}, range=[{arr_rgb.min()}, {arr_rgb.max()}]")
    print(f"  After Grayscale(3) from L:   shape={arr_l.shape}, range=[{arr_l.min()}, {arr_l.max()}]")
    if not np.array_equal(from_rgb, from_l):
        print(f"  MAX DIFF: {np.abs(arr_rgb.astype(int) - arr_l.astype(int)).max()}")

    # ── CHECK 2: BACKBONE ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("CHECK 2: BACKBONE WEIGHTS")
    print("=" * 70)

    # Fresh ImageNet model (what ferplus script used)
    model_fresh = build_model()
    print("\nFresh ImageNet ResNet-18 weight norms:")
    for name in ["conv1.weight", "layer1.0.conv1.weight", "layer4.1.conv2.weight"]:
        w = dict(model_fresh.named_parameters())[name]
        print(f"  {name}: norm={w.data.norm():.4f}, shape={w.shape}")

    # FER2013 checkpoint
    model_ckpt = build_model()
    state = torch.load(FER2013_CKPT, map_location="cpu", weights_only=True)
    model_ckpt.load_state_dict(state)
    print(f"\nFER2013 checkpoint ({FER2013_CKPT}) weight norms:")
    for name in ["conv1.weight", "layer1.0.conv1.weight", "layer4.1.conv2.weight"]:
        w = dict(model_ckpt.named_parameters())[name]
        print(f"  {name}: norm={w.data.norm():.4f}, shape={w.shape}")

    # Compare: are backbone weights identical?
    print("\nBackbone weights comparison (fresh ImageNet vs FER2013 checkpoint):")
    all_same = True
    for name, p_fresh in model_fresh.named_parameters():
        if name.startswith("fc."):
            continue
        p_ckpt = dict(model_ckpt.named_parameters())[name]
        if not torch.equal(p_fresh.data, p_ckpt.data):
            diff = (p_fresh.data - p_ckpt.data).abs().max().item()
            print(f"  DIFFERS: {name}, max_diff={diff:.6f}")
            all_same = False
    if all_same:
        print("  All backbone weights IDENTICAL (ImageNet pretrained, not fine-tuned)")
    else:
        print("  WARNING: FER2013 checkpoint has DIFFERENT backbone weights (was fine-tuned)")

    # ── CHECK 3: LABEL MAPPING ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("CHECK 3: LABEL MAPPING")
    print("=" * 70)

    # ImageFolder mapping for FER2013
    fer_if = datasets.ImageFolder(os.path.join(FER2013_DIR, "test"))
    print(f"\nFER2013 ImageFolder class_to_idx: {fer_if.class_to_idx}")

    # FERPlus custom dataset mapping
    print(f"FERPlus FERPLUS_FOLDER_MAP: {FERPLUS_FOLDER_MAP}")

    # Spot check: 3 images per class
    print("\nFERPlus spot check (3 images per class):")
    class_samples = {c: [] for c in range(NUM_CLASSES)}
    for i in range(len(fp_dataset)):
        c = fp_dataset.labels[i]
        if len(class_samples[c]) < 3:
            class_samples[c].append(i)
        if all(len(v) >= 3 for v in class_samples.values()):
            break

    for c in range(NUM_CLASSES):
        print(f"\n  Class {c} ({CLASS_NAMES[c]}):")
        for idx in class_samples[c]:
            fname = os.path.basename(fp_dataset.image_paths[idx])
            folder = fp_dataset.folder_names[idx]
            label = fp_dataset.labels[idx]
            print(f"    {fname} | folder={folder} | label_idx={label} | label_name={CLASS_NAMES[label]}")

    # ── CHECK 4: SANITY BASELINE ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("CHECK 4: FER2013-TRAINED MODEL ON FERPLUS (no retraining)")
    print("=" * 70)

    # Load the exact FER2013 checkpoint (backbone + head)
    model_ckpt.eval()
    model_ckpt.to(DEVICE)

    # First verify it on FER2013 test
    print("\nVerifying FER2013 checkpoint on FER2013 test...")
    fer_test = datasets.ImageFolder(os.path.join(FER2013_DIR, "test"), transform=img_transforms)
    fer_loader = DataLoader(fer_test, batch_size=64, shuffle=False, num_workers=4)
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in fer_loader:
            out = model_ckpt(inputs)
            all_preds.extend(out.argmax(1).numpy())
            all_labels.extend(labels.numpy())
    fer_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    fer_acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    print(f"  FER2013 test: macro-F1={fer_f1:.4f}, acc={fer_acc:.4f}")
    fer_pc = f1_score(all_labels, all_preds, average=None, labels=list(range(NUM_CLASSES)), zero_division=0)
    for i, c in enumerate(CLASS_NAMES):
        print(f"    {c:<10s}: {fer_pc[i]:.4f}")

    # Check FER2013 ImageFolder class ordering matches ours
    print(f"\n  FER2013 class_to_idx: {fer_test.class_to_idx}")
    print(f"  Our CLASS_NAMES:      {CLASS_NAMES}")

    # Now run same model on FERPlus test
    print("\nRunning FER2013 checkpoint on FERPlus test (no retraining)...")
    # Need to check label alignment: FER2013 ImageFolder has its own class_to_idx
    # FERPlus uses our FERPLUS_FOLDER_MAP
    # Are they aligned?
    fer_c2i = fer_test.class_to_idx
    print(f"\n  FER2013 class_to_idx: {fer_c2i}")
    print(f"  FERPlus folder map:   {FERPLUS_FOLDER_MAP}")
    print(f"  Alignment check:")
    fer_idx_to_class = {v: k for k, v in fer_c2i.items()}
    for fer_class, fer_idx in sorted(fer_c2i.items(), key=lambda x: x[1]):
        fp_equiv = None
        for fp_folder, fp_idx in FERPLUS_FOLDER_MAP.items():
            if fp_idx == fer_idx:
                fp_equiv = fp_folder
        print(f"    idx={fer_idx}: FER2013='{fer_class}' -> FERPlus='{fp_equiv}'")

    fp_test = FERPlusDataset(os.path.join(FERPLUS_DIR, "test"), transform=img_transforms)
    fp_loader = DataLoader(fp_test, batch_size=64, shuffle=False, num_workers=4)
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in fp_loader:
            out = model_ckpt(inputs)
            all_preds.extend(out.argmax(1).numpy())
            all_labels.extend(labels if isinstance(labels, list) else labels.numpy())
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    fp_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    fp_acc = (all_preds == all_labels).mean()
    print(f"\n  FERPlus test (FER2013 model, no retrain): macro-F1={fp_f1:.4f}, acc={fp_acc:.4f}")
    fp_pc = f1_score(all_labels, all_preds, average=None, labels=list(range(NUM_CLASSES)), zero_division=0)
    for i, c in enumerate(CLASS_NAMES):
        print(f"    {c:<10s}: {fp_pc[i]:.4f}")

    # Prediction distribution
    print(f"\n  Prediction distribution on FERPlus:")
    pred_counts = Counter(all_preds.tolist())
    for c in range(NUM_CLASSES):
        print(f"    {CLASS_NAMES[c]:<10s}: {pred_counts.get(c, 0)} predictions")
    print(f"  True label distribution on FERPlus:")
    true_counts = Counter(all_labels.tolist())
    for c in range(NUM_CLASSES):
        print(f"    {CLASS_NAMES[c]:<10s}: {true_counts.get(c, 0)} true")

    print("\n" + "=" * 70)
    print("END — Diagnostics")
    print("=" * 70)


if __name__ == "__main__":
    main()
