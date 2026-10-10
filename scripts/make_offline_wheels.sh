#!/usr/bin/env bash
# Run this on a machine WITH internet. It downloads every Python package the offline
# Snowflake run needs into ./offline_wheels, then you upload that folder to a Snowflake stage.
set -euo pipefail
OUT=${1:-offline_wheels}
mkdir -p "$OUT"
# CPU wheels are small; for GPU builds use: --index-url https://download.pytorch.org/whl/cu124
python -m pip download -d "$OUT" -r requirements-offline.txt
echo "Wheels in $OUT. Install offline on the target with:"
echo "  pip install --no-index --find-links $OUT -r requirements-offline.txt"
