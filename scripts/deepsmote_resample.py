"""
Stage 4: DeepSMOTE on BOTH v5 and v7 baselines.
Train AE per baseline, SMOTE in 128-d latent, decode, train head, evaluate.
"""
print("=" * 70)
print("START — DeepSMOTE dual-baseline comparison")
print("=" * 70)

import os
import sys
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, classification_report
from sklearn.manifold import TSNE
from imblearn.over_sampling import SMOTE
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deepsmote_model import FeatureAutoencoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = "./results/deepsmote"
os.makedirs(RESULTS_DIR, exist_ok=True)

CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
SEEDS = [42, 123, 456]
TARGET_COUNT = 7215
AE_EPOCHS = 200
AE_LR = 1e-3
HEAD_LR = 1e-4
HEAD_PATIENCE = 15
HEAD_MAX_EPOCHS = 200
LABEL_SMOOTHING = 0.1


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_features(prefix):
    tr = np.load(f"{prefix}_train.npz")
    va = np.load(f"{prefix}_val.npz")
    te = np.load(f"{prefix}_test.npz")
    return (tr["features"], tr["labels"],
            va["features"], va["labels"],
            te["features"], te["labels"])


def train_autoencoder(X_train, device, seed):
    set_seed(seed)
    ae = FeatureAutoencoder().to(device)
    optimizer = torch.optim.Adam(ae.parameters(), lr=AE_LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10)
    criterion = nn.MSELoss()
    loader = DataLoader(TensorDataset(torch.FloatTensor(X_train)),
                        batch_size=256, shuffle=True)

    history = []
    for epoch in range(1, AE_EPOCHS + 1):
        ae.train()
        epoch_loss, n = 0, 0
        for (xb,) in loader:
            xb = xb.to(device)
            recon, _ = ae(xb)
            loss = criterion(recon, xb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
            n += xb.size(0)
        avg = epoch_loss / n
        scheduler.step(avg)
        history.append({"epoch": epoch, "recon_loss": round(avg, 8)})
        if epoch % 50 == 0 or epoch == 1:
            print(f"      AE epoch {epoch:3d} | loss: {avg:.6f}")
    return ae, history


def evaluate_head(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            p = model(xb.to(device)).argmax(1).cpu().numpy()
            preds.extend(p)
            labels.extend(yb.numpy())
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    per_class = f1_score(labels, preds, average=None, labels=list(range(7)), zero_division=0)
    report = classification_report(labels, preds, target_names=CLASSES,
                                   zero_division=0, output_dict=True)
    per_class_recall = {c: report[c]["recall"] for c in CLASSES}
    return (macro_f1,
            {c: float(per_class[i]) for i, c in enumerate(CLASSES)},
            per_class_recall,
            report["accuracy"])


def run_deepsmote_for_baseline(name, feat_prefix, dropout, device):
    print(f"\n{'='*70}")
    print(f"DeepSMOTE — {name} (dropout={dropout})")
    print(f"{'='*70}")

    X_tr, y_tr, X_va, y_va, X_te, y_te = load_features(feat_prefix)
    print(f"  Train: {X_tr.shape}, Val: {X_va.shape}, Test: {X_te.shape}")
    print(f"  Train dist: {np.bincount(y_tr)}")

    val_loader = DataLoader(TensorDataset(torch.FloatTensor(X_va), torch.LongTensor(y_va)),
                            batch_size=512, shuffle=False)
    test_loader = DataLoader(TensorDataset(torch.FloatTensor(X_te), torch.LongTensor(y_te)),
                             batch_size=512, shuffle=False)

    all_ae_histories = []
    seed_results = []
    tsne_data = None

    for seed in SEEDS:
        print(f"\n  --- Seed {seed} ---")

        # 1. Train AE
        print(f"    Training autoencoder...")
        ae, ae_hist = train_autoencoder(X_tr, device, seed)
        all_ae_histories.append(ae_hist)

        # 2. Encode
        ae.eval()
        with torch.no_grad():
            Z_train = ae.encode(torch.FloatTensor(X_tr).to(device)).cpu().numpy()
        print(f"    Latent: {Z_train.shape}")

        # 3. SMOTE in latent
        counts = Counter(y_tr)
        strategy = {i: TARGET_COUNT for i in range(7) if counts[i] < TARGET_COUNT}
        k = min(5, min(counts.values()) - 1)
        if k < 5:
            print(f"    Warning: k_neighbors={k}")

        set_seed(seed)
        smote = SMOTE(sampling_strategy=strategy, k_neighbors=k, random_state=seed)
        Z_res, y_res = smote.fit_resample(Z_train, y_tr)
        n_synth = len(Z_res) - len(Z_train)
        print(f"    SMOTE: {len(Z_train)} -> {len(Z_res)} (+{n_synth})")

        # 4. Decode synthetic only, keep originals
        Z_synth = Z_res[len(Z_train):]
        y_synth = y_res[len(Z_train):]
        with torch.no_grad():
            X_synth_decoded = ae.decode(torch.FloatTensor(Z_synth).to(device)).cpu().numpy()
        X_combined = np.concatenate([X_tr, X_synth_decoded])
        y_combined = np.concatenate([y_tr, y_synth])

        # Save t-SNE data for first seed
        if seed == SEEDS[0]:
            tsne_data = {
                "X_original": X_tr, "y_original": y_tr,
                "X_synthetic": X_synth_decoded, "y_synthetic": y_synth,
            }

        # 5. Train head
        train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_combined), torch.LongTensor(y_combined)),
            batch_size=256, shuffle=True)

        set_seed(seed)
        head = nn.Sequential(nn.Dropout(dropout), nn.Linear(512, 7)).to(device)
        optimizer = torch.optim.Adam(head.parameters(), lr=HEAD_LR)
        criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

        best_f1, best_state, pat = 0.0, None, 0
        for epoch in range(1, HEAD_MAX_EPOCHS + 1):
            head.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                criterion(head(xb), yb).backward()
                optimizer.step()
            vf1, _, _, _ = evaluate_head(head, val_loader, device)
            if vf1 > best_f1:
                best_f1 = vf1
                best_state = {k: v.clone() for k, v in head.state_dict().items()}
                pat = 0
            else:
                pat += 1
                if pat >= HEAD_PATIENCE:
                    break

        head.load_state_dict(best_state)
        val_f1, val_pc, val_recall, val_acc = evaluate_head(head, val_loader, device)
        test_f1, test_pc, test_recall, test_acc = evaluate_head(head, test_loader, device)
        print(f"    Test macro-F1: {test_f1:.4f}, disgust: {test_pc['disgust']:.4f}, disgust recall: {test_recall['disgust']:.4f}")

        seed_results.append({
            "seed": seed,
            "ae_final_loss": ae_hist[-1]["recon_loss"],
            "val_macro_f1": val_f1, "val_accuracy": val_acc,
            "test_macro_f1": test_f1, "test_accuracy": test_acc,
            "test_per_class_f1": test_pc,
            "test_per_class_recall": test_recall,
        })

    # Aggregate
    macro_f1s = [r["test_macro_f1"] for r in seed_results]
    summary = {
        "macro_f1_mean": float(np.mean(macro_f1s)),
        "macro_f1_std": float(np.std(macro_f1s)),
        "per_class_mean": {c: float(np.mean([r["test_per_class_f1"][c] for r in seed_results]))
                          for c in CLASSES},
        "per_class_std": {c: float(np.std([r["test_per_class_f1"][c] for r in seed_results]))
                         for c in CLASSES},
        "per_class_recall_mean": {c: float(np.mean([r["test_per_class_recall"][c] for r in seed_results]))
                                  for c in CLASSES},
        "per_class_recall_std": {c: float(np.std([r["test_per_class_recall"][c] for r in seed_results]))
                                for c in CLASSES},
        "per_seed": seed_results,
        "ae_histories": all_ae_histories,
    }

    print(f"\n  Mean test macro-F1: {summary['macro_f1_mean']:.4f} ± {summary['macro_f1_std']:.4f}")
    print(f"  Mean disgust F1:   {summary['per_class_mean']['disgust']:.4f}")

    return summary, tsne_data


# ── Run both baselines ───────────────────────────────────────────────────────

device = torch.device(DEVICE)
print(f"Device: {device}")

v5_summary, v5_tsne = run_deepsmote_for_baseline(
    "v5", "experiments/fer2013_final/results/smote/features", dropout=0.3, device=device)

v7_summary, v7_tsne = run_deepsmote_for_baseline(
    "v7", "results/smote_compare/features_v7", dropout=0.4, device=device)

# ── Save everything ──────────────────────────────────────────────────────────

results = {"v5_deepsmote": v5_summary, "v7_deepsmote": v7_summary}
with open(os.path.join(RESULTS_DIR, "deepsmote_dual_results.json"), "w") as f:
    json.dump(results, f, indent=2, default=str)

# Save t-SNE data
for name, td in [("v5", v5_tsne), ("v7", v7_tsne)]:
    np.savez(os.path.join(RESULTS_DIR, f"tsne_data_{name}.npz"),
             X_original=td["X_original"], y_original=td["y_original"],
             X_synthetic=td["X_synthetic"], y_synthetic=td["y_synthetic"])

# ── Load SVMSMOTE results for comparison ─────────────────────────────────────

svmsmote_path = "results/smote_compare/comparison_results.json"
if os.path.exists(svmsmote_path):
    with open(svmsmote_path) as f:
        svm_data = json.load(f)["summary"]
else:
    svm_data = None

# ── Print full 6-way comparison ─────────────────────────────────────────────

print(f"\n{'='*90}")
print("6-WAY COMPARISON TABLE — Test Set (mean ± std)")
print(f"{'='*90}")

rows = [
    ("v5 baseline",     0.678, 0.666, 0.544, 0.563),
    ("v5 + SVMSMOTE",
     svm_data["v5_smote"]["macro_f1_mean"] if svm_data else 0,
     svm_data["v5_smote"]["per_class_mean"]["disgust"] if svm_data else 0,
     svm_data["v5_smote"]["per_class_mean"]["fear"] if svm_data else 0,
     svm_data["v5_smote"]["per_class_mean"]["sad"] if svm_data else 0),
    ("v5 + DeepSMOTE",
     v5_summary["macro_f1_mean"],
     v5_summary["per_class_mean"]["disgust"],
     v5_summary["per_class_mean"]["fear"],
     v5_summary["per_class_mean"]["sad"]),
    ("v7 baseline",     0.648, 0.582, 0.514, 0.532),
    ("v7 + SVMSMOTE",
     svm_data["v7_smote"]["macro_f1_mean"] if svm_data else 0,
     svm_data["v7_smote"]["per_class_mean"]["disgust"] if svm_data else 0,
     svm_data["v7_smote"]["per_class_mean"]["fear"] if svm_data else 0,
     svm_data["v7_smote"]["per_class_mean"]["sad"] if svm_data else 0),
    ("v7 + DeepSMOTE",
     v7_summary["macro_f1_mean"],
     v7_summary["per_class_mean"]["disgust"],
     v7_summary["per_class_mean"]["fear"],
     v7_summary["per_class_mean"]["sad"]),
]

header = f"{'Method':<22} | {'macro-F1':>10} | {'disgust':>10} | {'fear':>10} | {'sad':>10}"
print(header)
print("-" * len(header))
for name, mf, df, ff, sf in rows:
    print(f"{name:<22} | {mf:>10.4f} | {df:>10.4f} | {ff:>10.4f} | {sf:>10.4f}")

print(f"\n{'='*70}")
print("END — DeepSMOTE dual-baseline comparison")
print(f"{'='*70}")
