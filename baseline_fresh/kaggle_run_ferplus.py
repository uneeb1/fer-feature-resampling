#!/usr/bin/env python3
"""
Kaggle orchestrator for FERPlus replication of the geometry-loss experiment.

Paste this entire file into a single Kaggle notebook cell, or run as a script.
Prerequisites:
  - FER2013 dataset added (fer2013.csv somewhere under /kaggle/input)
  - FERPlus dataset added (fer2013new.csv somewhere under /kaggle/input)
  - GPU T4 enabled
"""

import os
import subprocess
import sys
import glob
import json
import shutil
import zipfile

# ── 0. Find both CSVs ─────────────────────────────────────────────────────
def find_file(name):
    for root, _, files in os.walk("/kaggle/input"):
        for f in files:
            if f == name:
                return os.path.join(root, f)
    raise FileNotFoundError(f"{name} not found under /kaggle/input")

CSV_PATH = find_file("fer2013.csv")
FERPLUS_CSV_PATH = find_file("fer2013new.csv")
print(f"Found fer2013.csv: {CSV_PATH}")
print(f"Found fer2013new.csv: {FERPLUS_CSV_PATH}")

# ── 1. Clone repo (ferplus branch) ────────────────────────────────────────
REPO_URL = "https://github.com/uneeb1/fer-feature-resampling.git"
BRANCH = "ferplus"
WORK = "/kaggle/working"
REPO_DIR = os.path.join(WORK, "repo")

if os.path.exists(REPO_DIR):
    shutil.rmtree(REPO_DIR)

token = None
try:
    from kaggle_secrets import UserSecretsClient
    token = UserSecretsClient().get_secret("GITHUB_TOKEN")
except Exception:
    pass

clone_url = REPO_URL
if token:
    clone_url = REPO_URL.replace("https://", f"https://{token}@")

subprocess.run(["git", "clone", "-b", BRANCH, "--single-branch", clone_url, REPO_DIR],
               check=True)
os.chdir(os.path.join(REPO_DIR, "baseline_fresh"))
print(f"Working dir: {os.getcwd()}")

# ── 2. Install deps ──────────────────────────────────────────────────────
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "PyYAML", "scikit-learn", "matplotlib", "Pillow",
                "opencv-python-headless", "seaborn", "imbalanced-learn"], check=True)

# ── 3. Training runs ─────────────────────────────────────────────────────
RUNS = [
    {"lam_center": "0",     "lam_sep": "0",    "tag": "baseline"},
    {"lam_center": "0.001", "lam_sep": "0.01", "tag": "geometry"},
]

for run in RUNS:
    lc, ls = run["lam_center"], run["lam_sep"]
    print(f"\n{'='*60}")
    print(f"RUN ({run['tag']}): lam_center={lc}  lam_sep={ls}")
    print(f"{'='*60}")
    cmd = [
        sys.executable, "train_geometry_loss.py",
        "--config", "config_m2.yaml",
        "--csv", CSV_PATH,
        "--ferplus-csv", FERPLUS_CSV_PATH,
        "--dataset", "ferplus",
        "--lam-center", lc,
        "--lam-sep", ls,
        "--seeds", "42", "123", "456",
        "--extract-features",
        "--num-workers", "2",
    ]
    subprocess.run(cmd, check=True)

# ── 4. Gaussian-only resampling on each feature dir ──────────────────────
RESAMPLING_RUNS = [
    {
        "feat_dir": "ferplus_M2geo_lamc0_lams0/features",
        "out": "results_resampling_ferplus_baseline.csv",
    },
    {
        "feat_dir": "ferplus_M2geo_lamc0.001_lams0.01/features",
        "out": "results_resampling_ferplus_geometry.csv",
    },
]

for rr in RESAMPLING_RUNS:
    print(f"\n{'='*60}")
    print(f"RESAMPLING: {rr['feat_dir']}")
    print(f"{'='*60}")
    cmd = [
        sys.executable, "resampling_experiment.py",
        "--feat-dir", rr["feat_dir"],
        "--methods", "gaussian",
        "--out", rr["out"],
    ]
    subprocess.run(cmd, check=True)

# ── 5. Bundle each ferplus_M2geo_* dir to a zip ─────────────────────────
BUNDLE_DIRS = ["metrics.json", "history.json", "config.json",
               "geometry", "predictions", "graphs", "features", "checkpoints"]

run_dirs = sorted(glob.glob("ferplus_M2geo_*"))
print(f"\nFound {len(run_dirs)} output dirs: {run_dirs}")

for d in run_dirs:
    zip_name = f"bundle_{d}.zip"
    zip_path = os.path.join(WORK, zip_name)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for item in BUNDLE_DIRS:
            full = os.path.join(d, item)
            if os.path.isfile(full):
                zf.write(full)
            elif os.path.isdir(full):
                for root, _, files in os.walk(full):
                    for f in files:
                        fp = os.path.join(root, f)
                        zf.write(fp)
    print(f"Created {zip_path} ({os.path.getsize(zip_path)/1e6:.1f} MB)")

# Copy resampling CSVs to /kaggle/working
for rr in RESAMPLING_RUNS:
    src = rr["out"]
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(WORK, os.path.basename(src)))

# Display download links in Kaggle
try:
    from IPython.display import FileLink, display
    for d in run_dirs:
        zip_name = f"bundle_{d}.zip"
        display(FileLink(os.path.join(WORK, zip_name)))
    for rr in RESAMPLING_RUNS:
        csv_dest = os.path.join(WORK, os.path.basename(rr["out"]))
        if os.path.exists(csv_dest):
            display(FileLink(csv_dest))
except ImportError:
    print("(Not in notebook — download zips from /kaggle/working/)")

# ── 6. Summary ───────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("FERPLUS EXPERIMENT SUMMARY")
print(f"{'='*60}")
for d in run_dirs:
    mf = os.path.join(d, "metrics.json")
    if os.path.exists(mf):
        with open(mf) as f:
            m = json.load(f)
        agg = m.get("aggregate", {})
        print(f"{d}:")
        print(f"  Test Macro-F1: {agg.get('test_macro_f1_mean', 0):.4f} ± {agg.get('test_macro_f1_std', 0):.4f}")
        print(f"  Test Accuracy: {agg.get('test_acc_mean', 0):.4f} ± {agg.get('test_acc_std', 0):.4f}")
        for seed_key, sd in m.get("per_seed", {}).items():
            pcf1 = sd.get("test_per_class_f1", {})
            print(f"  Seed {seed_key}: test_f1={sd.get('test_f1', 0):.4f}  "
                  + "  ".join(f"{c}={pcf1.get(c, 0):.3f}" for c in ["fear", "angry", "sad", "neutral", "disgust"]))

for rr in RESAMPLING_RUNS:
    if os.path.exists(rr["out"]):
        print(f"\nResampling results ({rr['out']}):")
        with open(rr["out"]) as f:
            print(f.read())
