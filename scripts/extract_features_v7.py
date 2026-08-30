"""
Extract 512-d features from frozen v7 backbone for SMOTE comparison.
Uses fer2013_final data splits (28709 train / 3589 val / 3589 test) to match v5.
NO augmentation at extraction time.
"""
print("=" * 60)
print("START — extract_features_v7.py")
print("=" * 60)

import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "./experiments/fer2013_final/data"
MODEL_PATH = "./results/baseline_v7/best_model.pth"
OUT_DIR = "./results/smote_compare"
os.makedirs(OUT_DIR, exist_ok=True)

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

model = models.resnet18(weights=None)
model.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(512, 7))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.eval()

feature_extractor = nn.Sequential(*list(model.children())[:-1])
feature_extractor.eval()
feature_extractor.to(DEVICE)

CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

splits = {
    "train": os.path.join(DATA_DIR, "train"),
    "val": os.path.join(DATA_DIR, "validation"),
    "test": os.path.join(DATA_DIR, "test"),
}

for split_name, split_path in splits.items():
    ds = datasets.ImageFolder(split_path, transform=transform)
    assert list(ds.class_to_idx.keys()) == CLASSES
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

    all_feats, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            feats = feature_extractor(inputs.to(DEVICE)).squeeze(-1).squeeze(-1)
            all_feats.append(feats.cpu().numpy())
            all_labels.append(labels.numpy())

    features = np.concatenate(all_feats)
    labels = np.concatenate(all_labels)
    out_path = os.path.join(OUT_DIR, f"features_v7_{split_name}.npz")
    np.savez(out_path, features=features, labels=labels)
    print(f"  {split_name}: {features.shape} -> {out_path}")

print("\n" + "=" * 60)
print("END — extract_features_v7.py")
print("=" * 60)
