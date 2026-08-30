import csv
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2


CLASSES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
USAGE_MAP = {"Training": "train", "PublicTest": "val", "PrivateTest": "test"}


def load_fer2013_csv(csv_path):
    splits = {"train": [], "val": [], "test": []}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = int(row["emotion"])
            pixels = np.array(row["pixels"].split(), dtype=np.uint8).reshape(48, 48)
            split = USAGE_MAP[row["Usage"]]
            splits[split].append((pixels, label))
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
