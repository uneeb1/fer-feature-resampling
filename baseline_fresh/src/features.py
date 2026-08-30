import numpy as np
import torch
from torch.utils.data import DataLoader


@torch.no_grad()
def extract_features(model, dataset, device, batch_size=64):
    from .transforms import get_val_transform
    from .dataset import FER2013Dataset

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    all_feats, all_labels = [], []
    for images, labels in loader:
        images = images.to(device)
        _, feats = model(images, return_features=True)
        all_feats.append(feats.cpu().numpy())
        all_labels.append(labels.numpy() if isinstance(labels, torch.Tensor) else np.array(labels))
    features = np.concatenate(all_feats, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    assert (features >= 0).all(), "Features must be non-negative (post-ReLU)"
    return features, labels


def save_features(features, labels, path_prefix):
    np.save(f"{path_prefix}_features.npy", features)
    np.save(f"{path_prefix}_labels.npy", labels)
    print(f"Saved features {features.shape} and labels {labels.shape} to {path_prefix}_*.npy")
