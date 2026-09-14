#!/bin/bash
# Step 2: run the existing resampling pipeline on GEOMETRY-loss features.
#
# Usage (from baseline_fresh/):
#   bash run_resampling_on_geometry.sh                                  # default geometry dir
#   bash run_resampling_on_geometry.sh M2geo_lamc0.001_lams0.01/features  # explicit dir
#
# Compare substrates by running on each:
#   (a) M2 features:        bash run_resampling_on_center.sh "M2_seed42_backup (1)/features"
#   (b) Revised CE baseline: bash run_resampling_on_geometry.sh M2geo_lamc0_lams0/features
#   (c) Geometry features:   bash run_resampling_on_geometry.sh M2geo_lamc0.001_lams0.01/features
#
# Output: results_resampling_geometry.csv (NEVER overwrites results_resampling_v2_symmetric.csv)

set -euo pipefail

FEAT_DIR="${1:-M2geo_lamc0.001_lams0.01/features}"
ORIG_SCRIPT="M2_seed42_backup (1)/resampling_experiment_v2_symmetric.py"
OUT_CSV="results_resampling_geometry.csv"

echo "============================================================"
echo "Resampling on geometry features: $FEAT_DIR"
echo "Output: $OUT_CSV"
echo "============================================================"

if [ ! -f "$FEAT_DIR/train_features.npy" ]; then
    echo "ERROR: $FEAT_DIR/train_features.npy not found."
    echo "Run train_geometry_loss.py --extract-features first."
    exit 1
fi

# Patch the resampling script (read-only copy): FEAT_DIR + output path
TMPSCRIPT=$(mktemp /tmp/resample_geometry_XXXXXX.py)
sed -e "s|^FEAT_DIR = .*|FEAT_DIR = Path(\"$FEAT_DIR\")|" \
    -e "s|results_resampling_v2_symmetric.csv|$OUT_CSV|" \
    "$ORIG_SCRIPT" > "$TMPSCRIPT"

echo "Running patched resampling script..."
python "$TMPSCRIPT"
rm -f "$TMPSCRIPT"

echo "Done. Results in: $OUT_CSV"
