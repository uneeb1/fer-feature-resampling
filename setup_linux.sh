#!/usr/bin/env bash
set -euo pipefail

python3 -m pip install --upgrade pip
pip install -r requirements.txt

echo "Linux setup complete."
echo "Activate the environment with: source venv/bin/activate"
