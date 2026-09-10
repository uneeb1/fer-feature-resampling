#!/bin/bash
# Step 2 stub: run the existing resampling pipeline on center-loss features.
# Point FEAT_DIR at the center-loss feature directory.
#
# Usage (from baseline_fresh/):
#   FEAT_DIR=M2center_seed42/features python "M2_seed42_backup (1)/resampling_experiment_v2_symmetric.py"
#
# Or edit the FEAT_DIR line at the top of the resampling script to point to
# M2center_seed42/features/ instead of M2_seed42_backup (1)/features/.
#
# The resampling script reads {train,val,test}_{features,labels}.npy from FEAT_DIR.
# No code changes needed beyond the path.

set -euo pipefail

FEAT_DIR="${1:-M2center_lam0.01/features}"

echo "Running resampling pipeline on features from: $FEAT_DIR"
echo "Make sure features have been extracted first (train_center_loss.py --extract-features)"

# The resampling script has FEAT_DIR hardcoded — create a patched copy
TMPSCRIPT=$(mktemp /tmp/resample_center_XXXXXX.py)
sed "s|^FEAT_DIR = .*|FEAT_DIR = Path(\"$FEAT_DIR\")|" \
    "M2_seed42_backup (1)/resampling_experiment_v2_symmetric.py" > "$TMPSCRIPT"

echo "Running patched resampling script..."
python "$TMPSCRIPT"
rm -f "$TMPSCRIPT"
