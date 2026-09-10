#!/usr/bin/env python3
"""
Kaggle orchestrator for Phase 5 geometry-loss experiment (Stage A, seed 42).

Paste this entire file into a single Kaggle notebook cell, or run as a script.
Prerequisites:
  - FER2013 dataset added (fer2013.csv somewhere under /kaggle/input)
  - GPU T4 enabled
  - (Optional) GITHUB_TOKEN in Kaggle Secrets for auto-push
"""

import os
import subprocess
import sys
import glob
import json
import shutil
import zipfile

# ── 0. Find fer2013.csv ─────────────────────────────────────────────────────
def find_csv():
    for root, _, files in os.walk("/kaggle/input"):
        for f in files:
            if f == "fer2013.csv":
                return os.path.join(root, f)
    raise FileNotFoundError("fer2013.csv not found under /kaggle/input")

CSV_PATH = find_csv()
print(f"Found CSV: {CSV_PATH}")

# ── 1. Clone repo (geometry-loss branch) ────────────────────────────────────
REPO_URL = "https://github.com/uneeb1/fer-feature-resampling.git"
BRANCH = "geometry-loss"
WORK = "/kaggle/working"
REPO_DIR = os.path.join(WORK, "repo")

if os.path.exists(REPO_DIR):
    shutil.rmtree(REPO_DIR)

# Try authenticated clone if token available
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

# ── 2. Install deps ────────────────────────────────────────────────────────
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "PyYAML", "scikit-learn", "matplotlib", "Pillow",
                "opencv-python-headless", "seaborn"], check=True)

# ── 3. Stage A runs ────────────────────────────────────────────────────────
RUNS = [
    {"lam_center": "0",     "lam_sep": "0"},
    {"lam_center": "0.001", "lam_sep": "0"},
    {"lam_center": "0.001", "lam_sep": "0.001"},
    {"lam_center": "0.001", "lam_sep": "0.01"},
    {"lam_center": "0.001", "lam_sep": "0.1"},
]

for run in RUNS:
    lc, ls = run["lam_center"], run["lam_sep"]
    print(f"\n{'='*60}")
    print(f"RUN: lam_center={lc}  lam_sep={ls}")
    print(f"{'='*60}")
    cmd = [
        sys.executable, "train_geometry_loss.py",
        "--config", "config_m2.yaml",
        "--csv", CSV_PATH,
        "--lam-center", lc,
        "--lam-sep", ls,
        "--extract-features",
        "--num-workers", "2",
    ]
    subprocess.run(cmd, check=True)

# ── 4. Bundle each run into a zip ──────────────────────────────────────────
BUNDLE_DIRS = ["metrics.json", "history.json", "config.json",
               "geometry", "predictions", "graphs", "features", "checkpoints"]

run_dirs = sorted(glob.glob("M2geo_*"))
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

# Display download links in Kaggle
try:
    from IPython.display import FileLink, display
    for d in run_dirs:
        zip_name = f"bundle_{d}.zip"
        display(FileLink(os.path.join(WORK, zip_name)))
except ImportError:
    print("(Not in notebook — download zips from /kaggle/working/)")

# ── 5. Push light artifacts to git ─────────────────────────────────────────
LIGHT_PATTERNS = ["metrics.json", "history.json", "config.json",
                  "geometry/geometry.json", "graphs/*.png"]
LOCKED = {"M2_seed42_backup (1)", "results_resampling_v2_symmetric.csv"}

if token:
    subprocess.run(["git", "config", "user.email", "sheikhuneeb90@gmail.com"], check=True)
    subprocess.run(["git", "config", "user.name", "uneeb1"], check=True)

    for d in run_dirs:
        for pat in LIGHT_PATTERNS:
            for fp in glob.glob(os.path.join(d, pat)):
                if not any(locked in fp for locked in LOCKED):
                    subprocess.run(["git", "add", fp])

    result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    if result.stdout.strip():
        subprocess.run(["git", "commit", "-m",
                         "Add Stage A geometry-loss results (metrics, history, geometry, graphs)\n\n"
                         "Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n"
                         "Claude-Session: https://claude.ai/code/session_01FgXfJG3Pmzs4PHCrfrV4P5"],
                        check=True)
        subprocess.run(["git", "push"], check=True)
        print("Pushed light artifacts to geometry-loss branch.")
    else:
        print("Nothing to push.")
else:
    print("No GITHUB_TOKEN — skipping git push. Results saved locally in zips.")

# ── 6. Summary ──────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("STAGE A SUMMARY")
print(f"{'='*60}")
for d in run_dirs:
    mf = os.path.join(d, "metrics.json")
    if os.path.exists(mf):
        with open(mf) as f:
            m = json.load(f)
        seed_data = list(m["per_seed"].values())[0]
        print(f"{d}: val_f1={seed_data['val_f1']:.4f}  test_f1={seed_data['test_f1']:.4f}")
