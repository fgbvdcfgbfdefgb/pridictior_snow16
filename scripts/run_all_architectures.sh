#!/usr/bin/env bash
# Trains the three Price Predictor architectures (multiple instances), then evaluates and selects.
# The Market Analyser is trained once and shared by all instances.
# Usage: scripts/run_all_architectures.sh [SIZE=auto] [NPROC=auto]
set -euo pipefail
cd "$(dirname "$0")/.."
SIZE=${1:-auto}
NPROC=${2:-$(python -c "import torch;print(max(1,torch.cuda.device_count()))" 2>/dev/null || echo 1)}

python scripts/prepare_dataset.py
python scripts/train_analyser.py --size "$SIZE" --device "$(python -c "import torch;print('cuda' if torch.cuda.is_available() else 'cpu')")"
for ARCH in transformer tcn gru; do
  if [ "$NPROC" -gt 1 ]; then
    python -m torch.distributed.run --standalone --nproc_per_node="$NPROC" scripts/train_predictor.py --arch "$ARCH" --size "$SIZE"
  else
    python scripts/train_predictor.py --arch "$ARCH" --size "$SIZE"
  fi
done
for CK in models/*/*/final.pt; do python scripts/evaluate.py --ckpt "$CK"; done
python scripts/select_best.py
