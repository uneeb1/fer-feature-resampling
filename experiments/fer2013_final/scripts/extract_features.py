"""Extract 512-d penultimate features from frozen v5 baseline backbone."""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data_lossless")
RESULTS_SMOTE = os.path.join(BASE, "results", "smote")
CHECKPOINT = os.path.join(BASE, "results", "baseline_v5", "best_model_s42.pth")
CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


def build_model():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(512, 7))
    return model


def extract(model, loader, device):
    features_list, labels_list = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            feat = model(images)
            features_list.append(feat.cpu().numpy())
            labels_list.append(labels.numpy())
    return np.concatenate(features_list), np.concatenate(labels_list)


def main():
    print("=" * 60)
    print("STAGE 2 — Step 1: Feature Extraction from v5 Backbone")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model, strip the classifier head to get 512-d features
    full_model = build_model()
    full_model.load_state_dict(torch.load(CHECKPOINT, weights_only=True, map_location=device))
    full_model.eval()

    # Feature extractor: everything before the fc head
    # ResNet18 avgpool outputs (B, 512, 1, 1) -> flatten -> 512
    feature_extractor = nn.Sequential(
        full_model.conv1, full_model.bn1, full_model.relu, full_model.maxpool,
        full_model.layer1, full_model.layer2, full_model.layer3, full_model.layer4,
        full_model.avgpool, nn.Flatten(),
    )
    feature_extractor.to(device)
    feature_extractor.eval()

    # No augmentation — deterministic features
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    splits = {
        "train": os.path.join(DATA, "train"),
        "val": os.path.join(DATA, "validation"),
        "test": os.path.join(DATA, "test"),
    }

    os.makedirs(RESULTS_SMOTE, exist_ok=True)

    for split_name, split_path in splits.items():
        ds = datasets.ImageFolder(split_path, transform=transform)
        assert list(ds.class_to_idx.keys()) == CLASSES, f"Class order mismatch in {split_name}!"
        loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

        print(f"\nExtracting {split_name} ({len(ds)} images)...")
        feats, labels = extract(feature_extractor, loader, device)
        print(f"  Features shape: {feats.shape}, Labels shape: {labels.shape}")

        out_path = os.path.join(RESULTS_SMOTE, f"features_{split_name}.npz")
        np.savez(out_path, features=feats, labels=labels)
        print(f"  Saved: {out_path}")

        # Per-class counts
        for i, cls in enumerate(CLASSES):
            count = int((labels == i).sum())
            print(f"    {cls:>10}: {count}")

    print("\n" + "=" * 60)
    print("Feature extraction COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
