import csv
import hashlib
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2


CLASSES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
USAGE_MAP = {"Training": "train", "PublicTest": "val", "PrivateTest": "test"}
SPLIT_PRIORITY = {"test": 0, "val": 1, "train": 2}


def _pixel_hash(pixels_str):
    return hashlib.md5(pixels_str.strip().encode()).hexdigest()


def load_fer2013_csv(csv_path, dedupe=True):
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

    if not dedupe:
        return {k: [(p, l) for p, l, _ in v] for k, v in raw.items()}

    original = {k: len(v) for k, v in raw.items()}

    # Collect all hashes by split
    hash_sets = {k: set(h for _, _, h in v) for k, v in raw.items()}

    # Priority: test > val > train
    # 1. Drop train rows whose hash appears in val or test
    train_keep_hashes = hash_sets["train"] - hash_sets["val"] - hash_sets["test"]
    raw["train"] = [r for r in raw["train"] if r[2] in train_keep_hashes]

    # 2. Drop val rows whose hash appears in test
    val_keep_hashes = hash_sets["val"] - hash_sets["test"]
    raw["val"] = [r for r in raw["val"] if r[2] in val_keep_hashes]

    # 3. Collapse within-split duplicates (keep first occurrence)
    for split in raw:
        seen = set()
        deduped = []
        for p, l, h in raw[split]:
            if h not in seen:
                seen.add(h)
                deduped.append((p, l, h))
        raw[split] = deduped

    # Report
    for split in ["train", "val", "test"]:
        removed = original[split] - len(raw[split])
        print(f"  Dedup {split}: {original[split]} -> {len(raw[split])} (removed {removed})")

    # Build final hash sets and assert zero cross-split overlap
    final_hashes = {k: set(h for _, _, h in v) for k, v in raw.items()}
    for a in final_hashes:
        for b in final_hashes:
            if a < b:
                overlap = len(final_hashes[a] & final_hashes[b])
                assert overlap == 0, f"Leakage remains: {a}∩{b} = {overlap}"
                print(f"  {a} ∩ {b}: {overlap} (OK)")

    splits = {k: [(p, l) for p, l, _ in v] for k, v in raw.items()}

    # Sanity check: disgust train count
    disgust_train = sum(1 for _, l in splits["train"] if l == 1)
    print(f"  Disgust train count after dedup: {disgust_train} (expect ~300-440; originals minus duplicates)")

    return splits


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
