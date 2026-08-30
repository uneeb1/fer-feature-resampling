"""Step 0: Build official three-way split for FER2013 final experiment."""
import os
import shutil
from collections import defaultdict

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJ = os.path.dirname(os.path.dirname(BASE))
SRC_TRAIN = os.path.join(PROJ, "data", "train")
SRC_TEST = os.path.join(PROJ, "data", "test")
DST = os.path.join(BASE, "data")

CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
EXPECTED_TRAIN = {"angry": 3995, "disgust": 436, "fear": 4097, "happy": 7215,
                  "sad": 4830, "surprise": 3171, "neutral": 4965}

def main():
    print("=" * 60)
    print("STEP 0: Building official three-way split")
    print("=" * 60)

    counts = {"train": defaultdict(int), "validation": defaultdict(int), "test": defaultdict(int)}
    all_files = {"train": set(), "validation": set(), "test": set()}

    # Copy training data (symlink for speed)
    for cls in CLASSES:
        src = os.path.join(SRC_TRAIN, cls)
        dst = os.path.join(DST, "train", cls)
        os.makedirs(dst, exist_ok=True)
        for f in os.listdir(src):
            src_path = os.path.join(src, f)
            dst_path = os.path.join(dst, f)
            if not os.path.exists(dst_path):
                shutil.copy2(src_path, dst_path)
            counts["train"][cls] += 1
            all_files["train"].add(f)

    # Split test into validation (PublicTest) and test (PrivateTest)
    for cls in CLASSES:
        src = os.path.join(SRC_TEST, cls)
        for f in os.listdir(src):
            src_path = os.path.join(src, f)
            if f.startswith("PublicTest_"):
                split = "validation"
            elif f.startswith("PrivateTest_"):
                split = "test"
            else:
                raise ValueError(f"Unknown prefix: {f}")
            dst_path = os.path.join(DST, split, cls, f)
            if not os.path.exists(dst_path):
                shutil.copy2(src_path, dst_path)
            counts[split][cls] += 1
            all_files[split].add(f)

    # Report
    for split in ["train", "validation", "test"]:
        total = sum(counts[split].values())
        print(f"\n{split.upper()} ({total} images):")
        for cls in CLASSES:
            marker = ""
            if split == "train" and counts[split][cls] != EXPECTED_TRAIN.get(cls, -1):
                marker = " *** MISMATCH ***"
            print(f"  {cls:>10}: {counts[split][cls]}{marker}")

    # Verify train counts
    for cls, expected in EXPECTED_TRAIN.items():
        actual = counts["train"][cls]
        assert actual == expected, f"Train {cls}: expected {expected}, got {actual}"
    print("\n[OK] Train counts verified.")

    # Verify zero overlap
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        overlap = all_files[a] & all_files[b]
        assert len(overlap) == 0, f"Overlap between {a} and {b}: {len(overlap)} files"
    print("[OK] Zero filename overlap between all splits.")

    print(f"\nTotal: train={sum(counts['train'].values())}, "
          f"validation={sum(counts['validation'].values())}, "
          f"test={sum(counts['test'].values())}")
    print("=" * 60)
    print("STEP 0 COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
