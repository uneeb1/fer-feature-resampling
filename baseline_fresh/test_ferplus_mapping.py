#!/usr/bin/env python3
"""Unit test: verify FERPlus majority-vote loader against ground-truth counts.

Self-contained — does not import src.dataset (avoids numpy dependency at test time).
Replicates the majority-vote logic inline to validate the mapping.
"""

import csv
import sys
import os
import urllib.request

CLASSES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
USAGE_MAP = {"Training": "train", "PublicTest": "val", "PrivateTest": "test"}
COL_TO_FER7 = {0: 6, 1: 3, 2: 5, 3: 4, 4: 0, 5: 1, 6: 2}

FERPLUS_URL = "https://raw.githubusercontent.com/microsoft/FERPlus/master/fer2013new.csv"
FERPLUS_LOCAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fer2013new_test.csv")

EXPECTED = {
    "train": {"angry": 2100, "disgust": 119, "fear": 532, "happy": 7287,
              "sad": 3014, "surprise": 3149, "neutral": 8740},
    "val":   {"angry": 287, "disgust": 24, "fear": 62, "happy": 865,
              "sad": 351, "surprise": 415, "neutral": 1182},
    "test":  {"angry": 273, "disgust": 18, "fear": 83, "happy": 893,
              "sad": 384, "surprise": 396, "neutral": 1090},
}
EXPECTED_TOTAL_KEPT = 31264
EXPECTED_TOTAL_DROPPED = 4623


def ferplus_majority_label(votes):
    votes = list(votes)
    threshold = 1.0 + sys.float_info.epsilon
    votes = [v if v >= threshold else 0.0 for v in votes]
    total = sum(votes)
    if total <= 0:
        return None
    maxval = max(votes)
    if maxval <= 0.5 * total:
        return None
    idx = votes.index(maxval)
    return COL_TO_FER7.get(idx)


def fetch_ferplus_csv():
    if os.path.exists(FERPLUS_LOCAL):
        return FERPLUS_LOCAL
    print(f"Downloading fer2013new.csv from {FERPLUS_URL}...")
    urllib.request.urlretrieve(FERPLUS_URL, FERPLUS_LOCAL)
    return FERPLUS_LOCAL


def test_mapping():
    fp_path = fetch_ferplus_csv()

    counts = {"train": {}, "val": {}, "test": {}}
    total_rows = 0
    dropped = 0

    with open(fp_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            total_rows += 1
            usage_str = row[0]
            split = USAGE_MAP.get(usage_str)
            if split is None:
                dropped += 1
                continue

            votes = [float(row[j]) for j in range(2, 12)]
            label = ferplus_majority_label(votes)
            if label is None:
                dropped += 1
                continue

            cname = CLASSES[label]
            counts[split][cname] = counts[split].get(cname, 0) + 1

    total_kept = sum(sum(v.values()) for v in counts.values())

    print(f"Total rows: {total_rows}")
    print(f"Kept: {total_kept}, Dropped: {dropped}")
    print()

    for split in ["train", "val", "test"]:
        print(f"{split}: {counts[split]}")

    assert total_kept == EXPECTED_TOTAL_KEPT, \
        f"Total kept {total_kept} != {EXPECTED_TOTAL_KEPT}"
    assert dropped == EXPECTED_TOTAL_DROPPED, \
        f"Dropped {dropped} != {EXPECTED_TOTAL_DROPPED}"

    for split in ["train", "val", "test"]:
        for cls in CLASSES:
            actual = counts[split].get(cls, 0)
            expected = EXPECTED[split].get(cls, 0)
            assert actual == expected, \
                f"{split}/{cls}: {actual} != {expected}"

    assert counts["train"]["angry"] == 2100
    assert counts["train"]["disgust"] == 119
    assert counts["train"]["neutral"] == 8740

    print("\nAll assertions passed!")


if __name__ == "__main__":
    test_mapping()
