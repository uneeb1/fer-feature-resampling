#!/usr/bin/env python3
"""FER2013 Baseline — Fresh, imbalance-agnostic training pipeline."""

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.dataset import load_fer2013_csv, verify_splits, FER2013Dataset, CLASSES
from src.model import FERResNet18
from src.transforms import get_train_transform, get_val_transform
from src.train import train_model, evaluate, evaluate_tta
from src.features import extract_features, save_features
from src.plots import (
    plot_class_distribution, plot_training_curves, plot_lr_schedule,
    plot_confusion_matrix, plot_per_class_f1, plot_tsne, plot_silhouette,
)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    p = argparse.ArgumentParser(description="FER2013 Baseline Training")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--csv", default=None, help="Override CSV path")
    p.add_argument("--resolution", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--subset", type=int, default=None, help="Use N samples per split for smoke test")
    p.add_argument("--clahe", action="store_true", default=False)
    p.add_argument("--no-tta", action="store_true", default=False)
    p.add_argument("--output-dir", default=".")
    p.add_argument("--num-workers", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    base_dir = args.output_dir
    os.makedirs(f"{base_dir}/checkpoints", exist_ok=True)
    os.makedirs(f"{base_dir}/features", exist_ok=True)
    os.makedirs(f"{base_dir}/figures", exist_ok=True)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.csv:
        cfg["data"]["csv_path"] = args.csv
    if args.resolution:
        cfg["data"]["resolution"] = args.resolution
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    if args.seeds:
        cfg["seeds"] = args.seeds
    if args.batch_size:
        cfg["training"]["batch_size"] = args.batch_size
    if args.clahe:
        cfg["data"]["clahe"] = True
    if args.no_tta:
        cfg["evaluation"]["tta"] = False
    if args.num_workers is not None:
        cfg["data"]["num_workers"] = args.num_workers

    res = cfg["data"]["resolution"]
    seeds = cfg["seeds"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Resolution: {res}, Epochs: {cfg['training']['epochs']}, Seeds: {seeds}")

    # Load data
    csv_path = cfg["data"]["csv_path"]
    if not os.path.isabs(csv_path):
        csv_path = os.path.join(os.path.dirname(args.config), csv_path)
    print(f"Loading data from {csv_path} ...")
    splits = load_fer2013_csv(csv_path)

    if args.subset:
        for k in splits:
            splits[k] = splits[k][:args.subset]
        print(f"SMOKE TEST: using {args.subset} samples per split")

    # Verify and print counts
    counts = verify_splits(splits)
    print("\n=== Split Counts ===")
    header = f"{'Class':<10}" + "".join(f"{s:<10}" for s in counts.keys())
    print(header)
    for c in range(7):
        row = f"{CLASSES[c]:<10}" + "".join(f"{counts[s].get(c, 0):<10}" for s in counts.keys())
        print(row)
    totals = f"{'TOTAL':<10}" + "".join(f"{sum(counts[s].values()):<10}" for s in counts.keys())
    print(totals)

    # Leakage check (pixel-hash based)
    print("\nVerifying zero sample leakage across splits...")
    split_hashes = {}
    for split_name, data in splits.items():
        hashes = set()
        for pixels, _ in data:
            hashes.add(pixels.tobytes())
        split_hashes[split_name] = hashes
    for a in split_hashes:
        for b in split_hashes:
            if a < b:
                overlap = len(split_hashes[a] & split_hashes[b])
                status = "OK" if overlap == 0 else f"WARNING: {overlap} overlaps!"
                print(f"  {a} ∩ {b}: {overlap} ({status})")

    # Plot class distribution
    plot_class_distribution(counts, f"{base_dir}/figures/class_distribution.png")
    print("Saved class_distribution.png")

    # Transforms
    train_tf = get_train_transform(res, cfg["augmentation"]["rotation_degrees"],
                                    cfg["augmentation"]["random_crop_pad"])
    val_tf = get_val_transform(res)
    clahe = cfg["data"]["clahe"]
    print(f"\nTrain transforms: {train_tf}")
    print(f"Val transforms: {val_tf}")
    print(f"CLAHE: {clahe}")

    # Train per seed
    all_results = {}
    best_overall_f1, best_seed = 0.0, None

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"SEED {seed}")
        print(f"{'='*60}")
        seed_everything(seed)

        train_ds = FER2013Dataset(splits["train"], transform=train_tf, clahe=clahe)
        val_ds = FER2013Dataset(splits["val"], transform=val_tf, clahe=clahe)

        nw = cfg["data"]["num_workers"]
        train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"],
                                  shuffle=True, num_workers=nw, pin_memory=(device.type == "cuda"))
        val_loader = DataLoader(val_ds, batch_size=cfg["training"]["batch_size"],
                                shuffle=False, num_workers=nw, pin_memory=(device.type == "cuda"))

        model = FERResNet18(num_classes=7).to(device)
        ckpt_path = f"{base_dir}/checkpoints/best_seed{seed}.pt"

        history, best_epoch, best_f1 = train_model(
            model, train_loader, val_loader, cfg, device, ckpt_path
        )

        # Plot curves for this seed
        plot_training_curves(history, f"{base_dir}/figures/training_curves_seed{seed}.png")
        plot_lr_schedule(history, f"{base_dir}/figures/lr_schedule.png")

        # Load best checkpoint for eval
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])

        # Test evaluation
        test_ds_eval = FER2013Dataset(splits["test"], transform=val_tf, clahe=clahe)
        test_loader = DataLoader(test_ds_eval, batch_size=cfg["training"]["batch_size"],
                                 shuffle=False, num_workers=nw)

        if cfg["evaluation"]["tta"] and not args.no_tta:
            print("Running TTA evaluation on test set...")
            test_ds_raw = FER2013Dataset(splits["test"], transform=None, clahe=clahe)
            test_acc, test_f1, test_pcf1, test_preds, test_labels = evaluate_tta(
                model, test_ds_raw, device, resolution=res
            )
        else:
            from torch.nn import CrossEntropyLoss
            criterion = CrossEntropyLoss(label_smoothing=cfg["training"]["label_smoothing"])
            _, test_acc, test_f1, test_pcf1, test_preds, test_labels = evaluate(
                model, test_loader, criterion, device
            )

        print(f"\nSeed {seed} TEST: Acc={test_acc:.4f} Macro-F1={test_f1:.4f}")
        for i, name in enumerate(CLASSES):
            print(f"  {name:<10}: F1={test_pcf1[i]:.4f}")

        all_results[seed] = {
            "val_f1": best_f1,
            "val_epoch": best_epoch,
            "test_acc": test_acc,
            "test_f1": test_f1,
            "test_per_class_f1": {CLASSES[i]: float(test_pcf1[i]) for i in range(7)},
            "history": {k: [float(v) for v in vals] for k, vals in history.items()},
        }

        if best_f1 > best_overall_f1:
            best_overall_f1 = best_f1
            best_seed = seed

    # Use best seed for figures and features
    print(f"\n{'='*60}")
    print(f"Best seed: {best_seed} (val F1={best_overall_f1:.4f})")
    print(f"{'='*60}")

    ckpt = torch.load(f"{base_dir}/checkpoints/best_seed{best_seed}.pt",
                       map_location=device, weights_only=True)
    model = FERResNet18(num_classes=7).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    # Best-seed training curves as main figure
    plot_training_curves(all_results[best_seed]["history"],
                         f"{base_dir}/figures/training_curves.png")

    # Confusion matrices (test, best seed)
    test_ds_eval = FER2013Dataset(splits["test"], transform=val_tf, clahe=clahe)
    test_loader = DataLoader(test_ds_eval, batch_size=cfg["training"]["batch_size"],
                             shuffle=False, num_workers=0)

    if cfg["evaluation"]["tta"] and not args.no_tta:
        test_ds_raw = FER2013Dataset(splits["test"], transform=None, clahe=clahe)
        _, _, _, test_preds, test_labels = evaluate_tta(
            model, test_ds_raw, device, resolution=res
        )
    else:
        from torch.nn import CrossEntropyLoss
        criterion = CrossEntropyLoss(label_smoothing=cfg["training"]["label_smoothing"])
        _, _, _, _, test_preds, test_labels = evaluate(
            model, test_loader, criterion, device
        )

    plot_confusion_matrix(test_labels, test_preds,
                          f"{base_dir}/figures/confusion_matrix_raw.png", normalize=False)
    plot_confusion_matrix(test_labels, test_preds,
                          f"{base_dir}/figures/confusion_matrix_norm.png", normalize=True)
    plot_per_class_f1(all_results[best_seed]["test_per_class_f1"].values(),
                      f"{base_dir}/figures/per_class_f1.png")
    print("Saved confusion matrices and per_class_f1.png")

    # Feature extraction (best seed model)
    print("\nExtracting 512-d features (best seed model)...")
    for split_name in ["train", "val", "test"]:
        ds = FER2013Dataset(splits[split_name], transform=val_tf, clahe=clahe)
        feats, lbls = extract_features(model, ds, device, batch_size=cfg["training"]["batch_size"])
        save_features(feats, lbls, f"{base_dir}/features/{split_name}")

    # t-SNE
    print("Computing t-SNE (this may take a while)...")
    for split_name in ["train", "test"]:
        feats = np.load(f"{base_dir}/features/{split_name}_features.npy")
        lbls = np.load(f"{base_dir}/features/{split_name}_labels.npy")
        plot_tsne(feats, lbls, f"{base_dir}/figures/tsne_{split_name}_features.png",
                  title=f"t-SNE of {split_name.capitalize()} Features (512-d)")

    # Silhouette
    print("Computing silhouette scores...")
    test_feats = np.load(f"{base_dir}/features/test_features.npy")
    test_lbls = np.load(f"{base_dir}/features/test_labels.npy")
    sil_scores = plot_silhouette(test_feats, test_lbls,
                                  f"{base_dir}/figures/per_class_silhouette.png")

    # Aggregate metrics
    test_f1s = [all_results[s]["test_f1"] for s in seeds]
    test_accs = [all_results[s]["test_acc"] for s in seeds]
    metrics = {
        "seeds": seeds,
        "per_seed": {str(s): all_results[s] for s in seeds},
        "aggregate": {
            "test_macro_f1_mean": float(np.mean(test_f1s)),
            "test_macro_f1_std": float(np.std(test_f1s)),
            "test_acc_mean": float(np.mean(test_accs)),
            "test_acc_std": float(np.std(test_accs)),
        },
        "best_seed": best_seed,
        "silhouette_per_class": {CLASSES[i]: float(sil_scores[i]) for i in range(7)},
    }

    # Remove history from metrics.json (too large)
    for s in seeds:
        if str(s) in metrics["per_seed"]:
            metrics["per_seed"][str(s)].pop("history", None)

    with open(f"{base_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics.json")

    # Summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Test Macro-F1: {np.mean(test_f1s):.4f} ± {np.std(test_f1s):.4f}")
    print(f"Test Accuracy: {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}")
    print(f"\nPer-class F1 (best seed {best_seed}):")
    for name, f1 in all_results[best_seed]["test_per_class_f1"].items():
        print(f"  {name:<10}: {f1:.4f}")
    print(f"\nSilhouette per class:")
    for i, name in enumerate(CLASSES):
        print(f"  {name:<10}: {sil_scores[i]:.4f}")


if __name__ == "__main__":
    main()
