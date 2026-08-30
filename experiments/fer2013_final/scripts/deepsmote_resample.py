"""Stage 4: DeepSMOTE — learn latent space, SMOTE there, decode back, train head."""
import os
import sys
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deepsmote_model import FeatureAutoencoder

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEAT_DIR = os.path.join(BASE, "results", "smote")
RESULTS = os.path.join(BASE, "results", "deepsmote")
LOGS = os.path.join(BASE, "logs", "deepsmote")
CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
SEEDS = [42, 123, 456]

BASELINE_TEST_F1 = {
    "angry": 0.598, "disgust": 0.693, "fear": 0.541,
    "happy": 0.880, "neutral": 0.679, "sad": 0.551, "surprise": 0.819,
}
BASELINE_MACRO_F1 = 0.680
SVMSMOTE_FULL = {"macro_f1": 0.674, "disgust": 0.626, "fear": 0.542, "sad": 0.569}
SVMSMOTE_2000 = {"macro_f1": 0.678, "disgust": 0.648, "fear": 0.547, "sad": 0.566}


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
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            all_preds.extend(model(X).argmax(1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    acc = float(np.mean(np.array(all_preds) == np.array(all_labels)))
    macro_f1 = float(f1_score(all_labels, all_preds, average="macro"))
    per_class = f1_score(all_labels, all_preds, average=None, labels=list(range(7)))
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(7)))
    return acc, macro_f1, per_class, cm


def build_sampling_strategy(y, target):
    counts = Counter(y)
    strategy = {}
    for cls_idx in range(7):
        if counts[cls_idx] < target:
            strategy[cls_idx] = target
    return strategy if strategy else "auto"


def train_autoencoder(X_train, device, seed, ae_epochs=200, lr=1e-3, batch_size=256):
    set_seed(seed)
    ae = FeatureAutoencoder(input_dim=512, hidden_dim=256, latent_dim=128).to(device)
    optimizer = torch.optim.Adam(ae.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
    criterion = nn.MSELoss()

    loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train)),
        batch_size=batch_size, shuffle=True,
    )

    history = []
    for epoch in range(1, ae_epochs + 1):
        ae.train()
        epoch_loss = 0; n = 0
        for (X_b,) in loader:
            X_b = X_b.to(device)
            recon, _ = ae(X_b)
            loss = criterion(recon, X_b)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * X_b.size(0)
            n += X_b.size(0)

        avg_loss = epoch_loss / n
        scheduler.step(avg_loss)
        history.append({"epoch": epoch, "recon_loss": round(avg_loss, 8)})

        if epoch % 50 == 0 or epoch == 1:
            print(f"    AE Epoch {epoch:3d} | Recon Loss: {avg_loss:.6f}")

    return ae, history


def main():
    print("=" * 70)
    print("STAGE 4 — DeepSMOTE: Learned Latent Space Interpolation")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_data = np.load(os.path.join(FEAT_DIR, "features_train.npz"))
    val_data = np.load(os.path.join(FEAT_DIR, "features_val.npz"))
    test_data = np.load(os.path.join(FEAT_DIR, "features_test.npz"))

    X_train, y_train = train_data["features"], train_data["labels"]
    X_val, y_val = val_data["features"], val_data["labels"]
    X_test, y_test = test_data["features"], test_data["labels"]

    orig_counts = Counter(y_train)
    print("Original train counts:")
    for i, cls in enumerate(CLASSES):
        print(f"  {cls:>10}: {orig_counts[i]}")

    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val)),
        batch_size=256, shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test)),
        batch_size=256, shuffle=False,
    )

    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(LOGS, exist_ok=True)

    targets_to_test = [("full", 7215), ("target_2000", 2000)]
    all_results = {}

    for target_name, target_count in targets_to_test:
        print(f"\n{'='*70}")
        print(f"DeepSMOTE — {target_name} (target={target_count})")
        print(f"{'='*70}")

        seed_results = []

        for seed in SEEDS:
            print(f"\n  --- Seed {seed} ---")

            # Step 1: Train autoencoder
            print(f"  Training autoencoder...")
            ae, ae_history = train_autoencoder(X_train, device, seed)

            final_loss = ae_history[-1]["recon_loss"]
            print(f"  Final recon loss: {final_loss:.6f}")

            if seed == SEEDS[0] and target_name == "full":
                with open(os.path.join(LOGS, "ae_history.json"), "w") as f:
                    json.dump(ae_history, f, indent=2)
                torch.save(ae.state_dict(), os.path.join(RESULTS, "autoencoder.pth"))

            # Step 2: Encode train features to latent space
            ae.eval()
            with torch.no_grad():
                Z_train = ae.encode(torch.FloatTensor(X_train).to(device)).cpu().numpy()
            print(f"  Latent shape: {Z_train.shape}")

            # Step 3: SMOTE in latent space
            strategy = build_sampling_strategy(y_train, target_count)
            set_seed(seed)

            k_used = None
            for k in [5, 4, 3]:
                try:
                    smote = SMOTE(sampling_strategy=strategy, k_neighbors=k, random_state=seed)
                    Z_resampled, y_resampled = smote.fit_resample(Z_train, y_train)
                    k_used = k
                    break
                except Exception as e:
                    print(f"    SMOTE k={k} failed: {e}")
                    if k == 3:
                        raise

            n_synthetic = len(Z_resampled) - len(Z_train)
            print(f"  SMOTE in latent space: {len(Z_train)} → {len(Z_resampled)} (+{n_synthetic} synthetic)")

            if seed == SEEDS[0]:
                res_counts = Counter(y_resampled)
                for i, cls in enumerate(CLASSES):
                    print(f"    {cls:>10}: {orig_counts[i]:>5} → {res_counts[i]:>5}")

            # Step 4: Decode synthetic latent points back to 512-d
            # Only decode the synthetic points; keep real features as-is
            Z_synthetic = Z_resampled[len(Z_train):]
            y_synthetic = y_resampled[len(Z_train):]

            with torch.no_grad():
                X_synthetic_decoded = ae.decode(
                    torch.FloatTensor(Z_synthetic).to(device)
                ).cpu().numpy()

            X_combined = np.concatenate([X_train, X_synthetic_decoded], axis=0)
            y_combined = np.concatenate([y_train, y_synthetic], axis=0)
            print(f"  Combined features: {X_combined.shape}")

            # Save for t-SNE (first seed, full target only)
            if seed == SEEDS[0] and target_name == "full":
                np.savez(
                    os.path.join(RESULTS, "features_train_deepsmote.npz"),
                    features=X_combined, labels=y_combined,
                    n_original=len(X_train),
                )

            # Step 5: Train classifier head
            train_loader = DataLoader(
                TensorDataset(torch.FloatTensor(X_combined), torch.LongTensor(y_combined)),
                batch_size=256, shuffle=True,
            )

            set_seed(seed)
            model = nn.Sequential(nn.Dropout(0.4), nn.Linear(512, 7)).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

            best_val_f1 = 0.0
            best_state = None
            patience_counter = 0

            for epoch in range(1, 201):
                model.train()
                for X_b, y_b in train_loader:
                    X_b, y_b = X_b.to(device), y_b.to(device)
                    optimizer.zero_grad()
                    loss = criterion(model(X_b), y_b)
                    loss.backward()
                    optimizer.step()

                val_acc, val_f1, _, _ = evaluate(model, val_loader, device)
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= 15:
                        break

            model.load_state_dict(best_state)
            val_acc, val_f1, val_per, _ = evaluate(model, val_loader, device)
            test_acc, test_f1, test_per, test_cm = evaluate(model, test_loader, device)

            print(f"  Test macro-F1: {test_f1:.4f}, disgust: {test_per[1]:.4f}")

            seed_results.append({
                "seed": seed,
                "k_neighbors_used": k_used,
                "ae_final_recon_loss": final_loss,
                "val_macro_f1": round(val_f1, 6),
                "val_accuracy": round(val_acc, 6),
                "test_macro_f1": round(test_f1, 6),
                "test_accuracy": round(test_acc, 6),
                "test_per_class_f1": {cls: round(float(test_per[i]), 6) for i, cls in enumerate(CLASSES)},
                "test_confusion_matrix": test_cm.tolist(),
            })

        # Aggregate
        test_f1s = [r["test_macro_f1"] for r in seed_results]
        test_accs = [r["test_accuracy"] for r in seed_results]
        per_class = {}
        for cls in CLASSES:
            vals = [r["test_per_class_f1"][cls] for r in seed_results]
            per_class[cls] = {"mean": round(float(np.mean(vals)), 4), "std": round(float(np.std(vals)), 4)}

        summary = {
            "target": target_count,
            "target_name": target_name,
            "per_seed": seed_results,
            "test_macro_f1_mean": round(float(np.mean(test_f1s)), 4),
            "test_macro_f1_std": round(float(np.std(test_f1s)), 4),
            "test_accuracy_mean": round(float(np.mean(test_accs)), 4),
            "test_accuracy_std": round(float(np.std(test_accs)), 4),
            "test_per_class_f1": per_class,
        }
        all_results[target_name] = summary

        print(f"\n  Mean test macro-F1: {np.mean(test_f1s):.4f} ± {np.std(test_f1s):.4f}")
        print(f"  Mean disgust F1: {per_class['disgust']['mean']:.4f} ± {per_class['disgust']['std']:.4f}")

    # Save
    with open(os.path.join(RESULTS, "deepsmote_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved: {os.path.join(RESULTS, 'deepsmote_results.json')}")

    # Comparison table
    print(f"\n{'='*90}")
    print("COMPARISON TABLE: Baseline vs SVMSMOTE vs DeepSMOTE (Test)")
    print(f"{'='*90}")
    header = f"{'Method':<22} | {'Test macro-F1':>16} | {'disgust F1':>12} | {'fear F1':>10} | {'sad F1':>10}"
    print(header)
    print("-" * len(header))
    print(f"{'Baseline v5':<22} | {BASELINE_MACRO_F1:>16.3f} | {BASELINE_TEST_F1['disgust']:>12.3f} | {BASELINE_TEST_F1['fear']:>10.3f} | {BASELINE_TEST_F1['sad']:>10.3f}")
    print(f"{'SVMSMOTE (full)':<22} | {SVMSMOTE_FULL['macro_f1']:>16.3f} | {SVMSMOTE_FULL['disgust']:>12.3f} | {SVMSMOTE_FULL['fear']:>10.3f} | {SVMSMOTE_FULL['sad']:>10.3f}")
    print(f"{'SVMSMOTE (2000)':<22} | {SVMSMOTE_2000['macro_f1']:>16.3f} | {SVMSMOTE_2000['disgust']:>12.3f} | {SVMSMOTE_2000['fear']:>10.3f} | {SVMSMOTE_2000['sad']:>10.3f}")

    for name, label in [("full", "DeepSMOTE (full)"), ("target_2000", "DeepSMOTE (2000)")]:
        s = all_results[name]
        mf = s["test_macro_f1_mean"]
        ms = s["test_macro_f1_std"]
        df = s["test_per_class_f1"]["disgust"]["mean"]
        ff = s["test_per_class_f1"]["fear"]["mean"]
        sf = s["test_per_class_f1"]["sad"]["mean"]
        print(f"{label:<22} | {mf:.4f} ± {ms:.4f} | {df:>12.4f} | {ff:>10.4f} | {sf:>10.4f}")

    # Key findings
    print(f"\n{'='*70}")
    print("KEY FINDINGS")
    print(f"{'='*70}")

    for name, label in [("full", "DeepSMOTE (full)"), ("target_2000", "DeepSMOTE (2000)")]:
        s = all_results[name]
        mf = s["test_macro_f1_mean"]
        df = s["test_per_class_f1"]["disgust"]["mean"]
        ff = s["test_per_class_f1"]["fear"]["mean"]
        sf = s["test_per_class_f1"]["sad"]["mean"]
        print(f"\n{label}:")
        print(f"  Macro-F1: {mf:.4f} vs baseline 0.680 → {'BEAT' if mf > 0.680 else 'DID NOT BEAT'}")
        print(f"  Disgust:  {df:.4f} vs baseline 0.693 → {'IMPROVED' if df > 0.693 else 'REGRESSED'} (Δ={df-0.693:+.4f})")
        print(f"  Fear:     {ff:.4f} vs baseline 0.541 (Δ={ff-0.541:+.4f}) → {'moved' if abs(ff-0.541) >= 0.02 else 'flat'}")
        print(f"  Sad:      {sf:.4f} vs baseline 0.551 (Δ={sf-0.551:+.4f}) → {'moved' if abs(sf-0.551) >= 0.02 else 'flat'}")

    print(f"\nAutoencoder final recon loss: {all_results['full']['per_seed'][0]['ae_final_recon_loss']:.6f}")

    print(f"\n{'='*70}")
    print("STAGE 4 DeepSMOTE COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
