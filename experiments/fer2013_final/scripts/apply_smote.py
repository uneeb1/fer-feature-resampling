"""SVMSMOTE feature-space resampling + classifier head training."""
import os
import json
import random
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, confusion_matrix
from imblearn.over_sampling import SVMSMOTE

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_SMOTE = os.path.join(BASE, "results", "smote")
LOGS_SMOTE = os.path.join(BASE, "logs", "smote")
CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
SEEDS = [42, 123, 456]

BASELINE_TEST_F1 = {
    "angry": 0.598, "disgust": 0.693, "fear": 0.541,
    "happy": 0.880, "neutral": 0.679, "sad": 0.551, "surprise": 0.819,
}
BASELINE_MACRO_F1 = 0.680


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0
    total = 0
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            loss = criterion(out, y)
            total_loss += loss.item() * y.size(0)
            total += y.size(0)
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    per_class = f1_score(all_labels, all_preds, average=None, labels=list(range(7)))
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(7)))
    return acc, macro_f1, per_class, cm, total_loss / total


def main():
    print("=" * 60)
    print("STAGE 2 — Step 2: SVMSMOTE + Classifier Head")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load features
    train_data = np.load(os.path.join(RESULTS_SMOTE, "features_train.npz"))
    val_data = np.load(os.path.join(RESULTS_SMOTE, "features_val.npz"))
    test_data = np.load(os.path.join(RESULTS_SMOTE, "features_test.npz"))

    X_train, y_train = train_data["features"], train_data["labels"]
    X_val, y_val = val_data["features"], val_data["labels"]
    X_test, y_test = test_data["features"], test_data["labels"]

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # Original class counts
    print("\nOriginal train class counts:")
    for i, cls in enumerate(CLASSES):
        print(f"  {cls:>10}: {int((y_train == i).sum())}")

    # Val/test loaders (never resampled)
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val)),
        batch_size=256, shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test)),
        batch_size=256, shuffle=False,
    )

    os.makedirs(RESULTS_SMOTE, exist_ok=True)
    os.makedirs(LOGS_SMOTE, exist_ok=True)

    all_seed_results = []
    smote_info = {}

    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f"SEED {seed}")
        print(f"{'='*60}")
        set_seed(seed)

        # SVMSMOTE with fallback k_neighbors
        X_resampled, y_resampled = None, None
        k_used = None
        for k in [5, 4, 3]:
            try:
                smote = SVMSMOTE(k_neighbors=k, random_state=seed)
                X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
                k_used = k
                break
            except Exception as e:
                print(f"  SVMSMOTE k_neighbors={k} failed: {e}")
                if k == 3:
                    raise RuntimeError("SVMSMOTE failed with k_neighbors=3") from e

        print(f"  SVMSMOTE fit with k_neighbors={k_used}")
        print(f"\n  Resampled train class counts:")
        for i, cls in enumerate(CLASSES):
            orig = int((y_train == i).sum())
            resampled = int((y_resampled == i).sum())
            print(f"    {cls:>10}: {orig:>5} → {resampled:>5} (+{resampled - orig})")

        if seed == SEEDS[0]:
            smote_info["k_neighbors_used"] = k_used
            smote_info["original_counts"] = {cls: int((y_train == i).sum()) for i, cls in enumerate(CLASSES)}
            smote_info["resampled_counts"] = {cls: int((y_resampled == i).sum()) for i, cls in enumerate(CLASSES)}
            # Save resampled features for t-SNE
            np.savez(
                os.path.join(RESULTS_SMOTE, "features_train_resampled.npz"),
                features=X_resampled, labels=y_resampled,
                n_original=len(X_train),
            )

        # Train classifier head
        train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_resampled), torch.LongTensor(y_resampled)),
            batch_size=256, shuffle=True,
        )

        model = nn.Sequential(nn.Dropout(0.4), nn.Linear(512, 7)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        best_val_f1 = 0.0
        best_state = None
        patience_counter = 0
        max_epochs = 200
        patience = 15
        history = []

        for epoch in range(1, max_epochs + 1):
            model.train()
            epoch_loss = 0; n = 0
            for X_b, y_b in train_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                optimizer.zero_grad()
                out = model(X_b)
                loss = criterion(out, y_b)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * y_b.size(0)
                n += y_b.size(0)

            val_acc, val_f1, val_per_f1, _, val_loss = evaluate(model, val_loader, device)

            if epoch % 20 == 0 or epoch == 1:
                print(f"  Epoch {epoch:3d} | Train Loss: {epoch_loss/n:.4f} | Val F1: {val_f1:.4f} Acc: {val_acc:.4f}")

            history.append({"epoch": epoch, "train_loss": round(epoch_loss/n, 6),
                           "val_f1": round(val_f1, 6), "val_acc": round(val_acc, 6)})

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch}")
                    break

        model.load_state_dict(best_state)

        # Evaluate
        val_acc, val_f1, val_per_f1, _, _ = evaluate(model, val_loader, device)
        test_acc, test_f1, test_per_f1, test_cm, _ = evaluate(model, test_loader, device)

        print(f"\n  Val  — Macro-F1: {val_f1:.4f}, Acc: {val_acc:.4f}")
        print(f"  Test — Macro-F1: {test_f1:.4f}, Acc: {test_acc:.4f}")
        print(f"  Per-class test F1:")
        for i, cls in enumerate(CLASSES):
            delta = test_per_f1[i] - BASELINE_TEST_F1[cls]
            print(f"    {cls:>10}: {test_per_f1[i]:.4f} (baseline: {BASELINE_TEST_F1[cls]:.3f}, Δ={delta:+.3f})")

        seed_result = {
            "seed": seed,
            "k_neighbors_used": k_used,
            "validation": {
                "accuracy": round(float(val_acc), 6),
                "macro_f1": round(float(val_f1), 6),
                "per_class_f1": {cls: round(float(val_per_f1[i]), 6) for i, cls in enumerate(CLASSES)},
            },
            "test": {
                "accuracy": round(float(test_acc), 6),
                "macro_f1": round(float(test_f1), 6),
                "per_class_f1": {cls: round(float(test_per_f1[i]), 6) for i, cls in enumerate(CLASSES)},
                "confusion_matrix": test_cm.tolist(),
            },
        }
        all_seed_results.append(seed_result)

        with open(os.path.join(LOGS_SMOTE, f"smote_history_s{seed}.json"), "w") as f:
            json.dump(history, f, indent=2)

        # Save best model
        torch.save(best_state, os.path.join(RESULTS_SMOTE, f"smote_head_s{seed}.pth"))

    # Aggregate
    test_f1s = [r["test"]["macro_f1"] for r in all_seed_results]
    test_accs = [r["test"]["accuracy"] for r in all_seed_results]
    val_f1s = [r["validation"]["macro_f1"] for r in all_seed_results]
    val_accs = [r["validation"]["accuracy"] for r in all_seed_results]

    per_class_test = {}
    for cls in CLASSES:
        vals = [r["test"]["per_class_f1"][cls] for r in all_seed_results]
        per_class_test[cls] = {"mean": round(float(np.mean(vals)), 4), "std": round(float(np.std(vals)), 4)}

    summary = {
        "method": "SVMSMOTE_feature_space",
        "smote_info": smote_info,
        "seeds": SEEDS,
        "per_seed": all_seed_results,
        "mean_std": {
            "validation": {
                "macro_f1_mean": round(float(np.mean(val_f1s)), 4),
                "macro_f1_std": round(float(np.std(val_f1s)), 4),
                "accuracy_mean": round(float(np.mean(val_accs)), 4),
                "accuracy_std": round(float(np.std(val_accs)), 4),
            },
            "test": {
                "macro_f1_mean": round(float(np.mean(test_f1s)), 4),
                "macro_f1_std": round(float(np.std(test_f1s)), 4),
                "accuracy_mean": round(float(np.mean(test_accs)), 4),
                "accuracy_std": round(float(np.std(test_accs)), 4),
                "per_class_f1": per_class_test,
            },
        },
    }

    with open(os.path.join(RESULTS_SMOTE, "smote_results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved: {os.path.join(RESULTS_SMOTE, 'smote_results.json')}")

    # Final comparison table
    print(f"\n{'='*70}")
    print("COMPARISON: Baseline v5 vs SVMSMOTE (Test)")
    print(f"{'='*70}")
    print(f"{'Class':>10} | {'Baseline':>10} | {'SMOTE mean':>10} | {'SMOTE std':>10} | {'Delta':>8}")
    print("-" * 60)
    for cls in CLASSES:
        bl = BASELINE_TEST_F1[cls]
        sm = per_class_test[cls]["mean"]
        sd = per_class_test[cls]["std"]
        print(f"{cls:>10} | {bl:>10.3f} | {sm:>10.4f} | {sd:>10.4f} | {sm - bl:>+8.3f}")
    print("-" * 60)
    sm_macro = summary["mean_std"]["test"]["macro_f1_mean"]
    sm_std = summary["mean_std"]["test"]["macro_f1_std"]
    print(f"{'Macro-F1':>10} | {BASELINE_MACRO_F1:>10.3f} | {sm_macro:>10.4f} | {sm_std:>10.4f} | {sm_macro - BASELINE_MACRO_F1:>+8.3f}")

    print(f"\nVal  Macro-F1: {np.mean(val_f1s):.4f} ± {np.std(val_f1s):.4f}")
    print(f"Test Macro-F1: {np.mean(test_f1s):.4f} ± {np.std(test_f1s):.4f}")
    print(f"Test Accuracy: {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}")

    # Flag findings
    print(f"\n{'='*60}")
    print("KEY FINDINGS")
    print(f"{'='*60}")
    dg = per_class_test["disgust"]["mean"]
    print(f"Disgust: {dg:.4f} vs baseline 0.693 → {'IMPROVED' if dg > 0.693 else 'NOT improved'}")
    for cls in ["fear", "sad"]:
        v = per_class_test[cls]["mean"]
        bl = BASELINE_TEST_F1[cls]
        d = v - bl
        print(f"{cls.capitalize()}: {v:.4f} vs baseline {bl:.3f} (Δ={d:+.3f}) → {'flat/entangled' if abs(d) < 0.02 else 'moved'}")

    print(f"\n{'='*60}")
    print("SVMSMOTE STAGE 2 COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
