import csv
import hashlib
import sys
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2


CLASSES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
USAGE_MAP = {"Training": "train", "PublicTest": "val", "PrivateTest": "test"}

# FERPlus vote columns (file order) -> FER7 integer label
# neutral=0,happiness=1,surprise=2,sadness=3,anger=4,disgust=5,fear=6,contempt=7,unknown=8,NF=9
COL_TO_FER7 = {0: 6, 1: 3, 2: 5, 3: 4, 4: 0, 5: 1, 6: 2}


def _pixel_hash(pixels_str):
    return hashlib.md5(pixels_str.strip().encode()).hexdigest()


def _apply_leakage_filter(raw):
    """Remove train rows whose pixel hash appears in val or test. Shared by both loaders."""
    val_hashes = set(h for _, _, h in raw["val"])
    test_hashes = set(h for _, _, h in raw["test"])
    train_orig = len(raw["train"])

    raw["train"] = [r for r in raw["train"] if r[2] not in val_hashes and r[2] not in test_hashes]

    removed = train_orig - len(raw["train"])
    print(f"  Leakage filter: removed {removed} train rows (hash found in val/test)")
    print(f"  Train: {train_orig} -> {len(raw['train'])}")
    print(f"  Val: {len(raw['val'])} (untouched)")
    print(f"  Test: {len(raw['test'])} (untouched)")

    train_hashes = set(h for _, _, h in raw["train"])
    tv = len(train_hashes & val_hashes)
    tt = len(train_hashes & test_hashes)
    vt = len(val_hashes & test_hashes)
    assert tv == 0, f"train∩val = {tv}"
    assert tt == 0, f"train∩test = {tt}"
    print(f"  train∩val: {tv} (OK)")
    print(f"  train∩test: {tt} (OK)")
    print(f"  val∩test: {vt} (official splits, not altered)")


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

    _apply_leakage_filter(raw)

    return {k: [(p, l) for p, l, _ in v] for k, v in raw.items()}


def ferplus_majority_label(votes):
    """Microsoft FERPlus majority-vote scheme (single-vote outlier removal)."""
    votes = list(votes)
    threshold = 1.0 + sys.float_info.epsilon
    votes = [v if v >= threshold else 0.0 for v in votes]
    total = sum(votes)
    if total <= 0:
        return None
    maxval = max(votes)
    if maxval <= 0.5 * total:
        return None
    idx = votes.index(maxval)
    return COL_TO_FER7.get(idx)


def load_ferplus_csv(fer_csv, ferplus_csv, leakage_filter=True):
    """Load FERPlus-relabeled data. Pixels from fer2013.csv, labels from fer2013new.csv."""
    fer_rows = []
    with open(fer_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fer_rows.append(row)

    fp_rows = []
    with open(ferplus_csv, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            fp_rows.append(row)

    assert len(fer_rows) == len(fp_rows), \
        f"Row count mismatch: fer2013={len(fer_rows)} ferplus={len(fp_rows)}"

    raw = {"train": [], "val": [], "test": []}
    drop_reasons = {"no_majority": 0, "contempt": 0, "unknown": 0, "NF": 0}
    usage_mismatch = 0

    for i, (fr, fpr) in enumerate(zip(fer_rows, fp_rows)):
        pixels_str = fr["pixels"]
        fer_usage = fr["Usage"]
        fp_usage = fpr[0] if len(fpr) > 0 else ""
        if fer_usage != fp_usage:
            usage_mismatch += 1

        vote_cols = [float(fpr[j]) for j in range(2, 12)]
        label = ferplus_majority_label(vote_cols)

        if label is None:
            filtered_votes = [v if v >= 1.0 + sys.float_info.epsilon else 0.0 for v in vote_cols]
            total = sum(filtered_votes)
            if total <= 0:
                maxidx = vote_cols.index(max(vote_cols))
            else:
                maxval = max(filtered_votes)
                if maxval <= 0.5 * total:
                    drop_reasons["no_majority"] += 1
                    continue
                maxidx = filtered_votes.index(maxval)
            if maxidx == 9:
                drop_reasons["NF"] += 1
            elif maxidx == 8:
                drop_reasons["unknown"] += 1
            elif maxidx == 7:
                drop_reasons["contempt"] += 1
            else:
                drop_reasons["no_majority"] += 1
            continue

        h = _pixel_hash(pixels_str)
        pixels = np.array(pixels_str.split(), dtype=np.uint8).reshape(48, 48)
        split = USAGE_MAP[fer_usage]
        raw[split].append((pixels, label, h))

    total_kept = sum(len(v) for v in raw.values())
    total_dropped = len(fer_rows) - total_kept
    print(f"  FERPlus: kept {total_kept}, dropped {total_dropped} of {len(fer_rows)}")
    print(f"    Drop reasons: {drop_reasons}")
    if usage_mismatch > 0:
        print(f"  WARNING: {usage_mismatch} rows have mismatched Usage columns")

    if not leakage_filter:
        return {k: [(p, l) for p, l, _ in v] for k, v in raw.items()}

    _apply_leakage_filter(raw)

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
