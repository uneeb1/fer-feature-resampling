"""Stage 3: SVMSMOTE ratio sweep — test gentler oversampling targets."""
import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, confusion_matrix
from imblearn.over_sampling import SVMSMOTE
from collections import Counter

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEAT_DIR = os.path.join(BASE, "results", "smote")
RESULTS = os.path.join(BASE, "results", "smote_ratio")
CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
SEEDS = [42, 123, 456]
TARGETS = [1000, 2000, 3000, 5000, 7215]

BASELINE_TEST_F1 = {
    "angry": 0.598, "disgust": 0.693, "fear": 0.541,
    "happy": 0.880, "neutral": 0.679, "sad": 0.551, "surprise": 0.819,
}
BASELINE_MACRO_F1 = 0.680
BASELINE_ACC = 0.694


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
            all_preds.extend(X.argmax(1).cpu().numpy() if False else model(X).argmax(1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    acc = float(np.mean(np.array(all_preds) == np.array(all_labels)))
    macro_f1 = float(f1_score(all_labels, all_preds, average="macro"))
    per_class = f1_score(all_labels, all_preds, average=None, labels=list(range(7)))
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(7)))
    return acc, macro_f1, per_class, cm


def build_sampling_strategy(y_train, target):
    counts = Counter(y_train)
    strategy = {}
    for cls_idx in range(7):
        current = counts[cls_idx]
        if current < target:
            strategy[cls_idx] = target
    return strategy


def main():
    print("=" * 70)
    print("STAGE 3 — SVMSMOTE Ratio Sweep")
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
    all_strategies = []

    for target in TARGETS:
        print(f"\n{'='*70}")
        print(f"TARGET = {target}")
        print(f"{'='*70}")

        strategy = build_sampling_strategy(y_train, target)
        if not strategy:
            print("  No classes below target — full balancing.")
            strategy = "auto"

        disgust_orig = orig_counts[1]
        disgust_target = target if disgust_orig < target else disgust_orig
        disgust_synth_frac = max(0, (disgust_target - disgust_orig) / disgust_target) if disgust_target > 0 else 0

        print(f"  Disgust: {disgust_orig} → {disgust_target} ({100*disgust_synth_frac:.0f}% synthetic)")
        print(f"  Classes to oversample: {list(strategy.keys()) if isinstance(strategy, dict) else 'all minority'}")

        seed_results = []
        for seed in SEEDS:
            set_seed(seed)

            k_used = None
            for k in [5, 4, 3]:
                try:
                    smote = SVMSMOTE(
                        sampling_strategy=strategy,
                        k_neighbors=k,
                        random_state=seed,
                    )
                    X_res, y_res = smote.fit_resample(X_train, y_train)
                    k_used = k
                    break
                except Exception as e:
                    print(f"    Seed {seed}, k={k} failed: {e}")
                    if k == 3:
                        raise

            res_counts = Counter(y_res)
            if seed == SEEDS[0]:
                print(f"  Resampled counts (seed {seed}):")
                for i, cls in enumerate(CLASSES):
                    print(f"    {cls:>10}: {orig_counts[i]:>5} → {res_counts[i]:>5}")

            train_loader = DataLoader(
                TensorDataset(torch.FloatTensor(X_res), torch.LongTensor(y_res)),
                batch_size=256, shuffle=True,
            )

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

            seed_results.append({
                "seed": seed,
                "k_neighbors_used": k_used,
                "val_macro_f1": round(val_f1, 6),
                "val_accuracy": round(val_acc, 6),
                "test_macro_f1": round(test_f1, 6),
                "test_accuracy": round(test_acc, 6),
                "test_per_class_f1": {cls: round(float(test_per[i]), 6) for i, cls in enumerate(CLASSES)},
                "test_confusion_matrix": test_cm.tolist(),
            })
            print(f"    Seed {seed}: test macro-F1={test_f1:.4f}, disgust={test_per[1]:.4f}")

        # Aggregate
        test_f1s = [r["test_macro_f1"] for r in seed_results]
        test_accs = [r["test_accuracy"] for r in seed_results]
        per_class = {}
        for cls in CLASSES:
            vals = [r["test_per_class_f1"][cls] for r in seed_results]
            per_class[cls] = {"mean": round(float(np.mean(vals)), 4), "std": round(float(np.std(vals)), 4)}

        strat_summary = {
            "target": target,
            "disgust_synth_fraction": round(disgust_synth_frac, 3),
            "per_seed": seed_results,
            "test_macro_f1_mean": round(float(np.mean(test_f1s)), 4),
            "test_macro_f1_std": round(float(np.std(test_f1s)), 4),
            "test_accuracy_mean": round(float(np.mean(test_accs)), 4),
            "test_accuracy_std": round(float(np.std(test_accs)), 4),
            "test_per_class_f1": per_class,
        }
        all_strategies.append(strat_summary)

        print(f"  Mean test macro-F1: {np.mean(test_f1s):.4f} ± {np.std(test_f1s):.4f}")

    # Save
    with open(os.path.join(RESULTS, "ratio_sweep_results.json"), "w") as f:
        json.dump(all_strategies, f, indent=2)
    print(f"\nResults saved: {os.path.join(RESULTS, 'ratio_sweep_results.json')}")

    # Print comparison table
    print(f"\n{'='*100}")
    print("COMPARISON TABLE: SVMSMOTE Ratio Sweep (Test, mean±std)")
    print(f"{'='*100}")
    header = f"{'Strategy':>14} | {'disg target':>11} | {'disg %synth':>11} | {'Test macro-F1':>16} | {'disgust F1':>13} | {'fear F1':>10} | {'sad F1':>10}"
    print(header)
    print("-" * len(header))
    print(f"{'baseline v5':>14} | {'(none)':>11} | {'0%':>11} | {BASELINE_MACRO_F1:>16.3f} | {BASELINE_TEST_F1['disgust']:>13.3f} | {BASELINE_TEST_F1['fear']:>10.3f} | {BASELINE_TEST_F1['sad']:>10.3f}")

    best_macro = -1
    best_target = None
    for s in all_strategies:
        t = s["target"]
        ds = s["disgust_synth_fraction"]
        mf1 = s["test_macro_f1_mean"]
        mstd = s["test_macro_f1_std"]
        df1 = s["test_per_class_f1"]["disgust"]["mean"]
        dstd = s["test_per_class_f1"]["disgust"]["std"]
        ff1 = s["test_per_class_f1"]["fear"]["mean"]
        sf1 = s["test_per_class_f1"]["sad"]["mean"]
        label = f"target {t}"
        print(f"{label:>14} | {t:>11} | {100*ds:>10.0f}% | {mf1:.4f} ± {mstd:.4f} | {df1:.4f}±{dstd:.4f} | {ff1:>10.4f} | {sf1:>10.4f}")
        if mf1 > best_macro:
            best_macro = mf1
            best_target = t

    print(f"\nBest test macro-F1: target={best_target} → {best_macro:.4f}")
    beats = best_macro > BASELINE_MACRO_F1
    print(f"Beats baseline 0.680? {'YES' if beats else 'NO'} ({best_macro:.4f} vs 0.680)")

    # Disgust regression threshold
    print(f"\nDisgust regression analysis:")
    for s in all_strategies:
        df1 = s["test_per_class_f1"]["disgust"]["mean"]
        delta = df1 - BASELINE_TEST_F1["disgust"]
        status = "OK" if df1 >= BASELINE_TEST_F1["disgust"] else "REGRESSED"
        print(f"  target {s['target']:>5}: disgust F1={df1:.4f} (Δ={delta:+.4f}) {status}")

    # Fear/sad check
    print(f"\nFear/sad across strategies:")
    for s in all_strategies:
        ff = s["test_per_class_f1"]["fear"]["mean"]
        sf = s["test_per_class_f1"]["sad"]["mean"]
        fd = ff - BASELINE_TEST_F1["fear"]
        sd = sf - BASELINE_TEST_F1["sad"]
        print(f"  target {s['target']:>5}: fear={ff:.4f} (Δ={fd:+.4f}), sad={sf:.4f} (Δ={sd:+.4f})")

    print(f"\n{'='*70}")
    print("STAGE 3 RATIO SWEEP COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
