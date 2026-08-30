import csv
import hashlib
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2


CLASSES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
USAGE_MAP = {"Training": "train", "PublicTest": "val", "PrivateTest": "test"}


def _pixel_hash(pixels_str):
    return hashlib.md5(pixels_str.strip().encode()).hexdigest()


def load_fer2013_csv(csv_path, leakage_filter=True):
    raw = {"train": [], "val": [], "test": []}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = int(row["emotion"])
            pixels_str = row["pixels"]
            h = _pixel_hash(pixels_str)
            pixels = np.array(pixels_str.split(), dtype=np.uint8).reshape(48, 48)
            split = USAGE_MAP[row["Usage"]]
            raw[split].append((pixels, label, h))

    if not leakage_filter:
        return {k: [(p, l) for p, l, _ in v] for k, v in raw.items()}

    # Val and test are NEVER modified (official splits)
    val_hashes = set(h for _, _, h in raw["val"])
    test_hashes = set(h for _, _, h in raw["test"])
    train_orig = len(raw["train"])

    # Remove train rows whose hash appears in val or test
    raw["train"] = [r for r in raw["train"] if r[2] not in val_hashes and r[2] not in test_hashes]

    removed = train_orig - len(raw["train"])
    print(f"  Leakage filter: removed {removed} train rows (hash found in val/test)")
    print(f"  Train: {train_orig} -> {len(raw['train'])}")
    print(f"  Val: {len(raw['val'])} (untouched)")
    print(f"  Test: {len(raw['test'])} (untouched)")

    # Verify
    train_hashes = set(h for _, _, h in raw["train"])
    tv = len(train_hashes & val_hashes)
    tt = len(train_hashes & test_hashes)
    vt = len(val_hashes & test_hashes)
    assert tv == 0, f"train∩val = {tv}"
    assert tt == 0, f"train∩test = {tt}"
    print(f"  train∩val: {tv} (OK)")
    print(f"  train∩test: {tt} (OK)")
    print(f"  val∩test: {vt} (official splits, not altered)")

    return {k: [(p, l) for p, l, _ in v] for k, v in raw.items()}


def verify_splits(splits):
    counts = {}
    for split_name, data in splits.items():
        counts[split_name] = {}
        for _, label in data:
            counts[split_name][label] = counts[split_name].get(label, 0) + 1
    return counts


class FER2013Dataset(Dataset):
    def __init__(self, data, transform=None, clahe=False):
        self.data = data
        self.transform = transform
        self.clahe = clahe
        if clahe:
            self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pixels, label = self.data[idx]
        if self.clahe:
            pixels = self._clahe.apply(pixels)
        img = Image.fromarray(pixels, mode="L").convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label
