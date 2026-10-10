#!/usr/bin/env bash
# End-to-end smoke test on one month (January 2020) with tiny models. Runs on CPU in a few minutes.
# It checks that every stage runs and that the live engine reproduces the offline pipeline.
set -euo pipefail
cd "$(dirname "$0")/.."
T=$(mktemp -d)
echo "work dir: $T"
python scripts/prepare_dataset.py --start 2020-01 --end 2020-01 --out "$T/proc" \
  --train-end 2020-01-20T00:00:00 --test-start 2020-01-25T00:00:00
python scripts/train_analyser.py --data-dir "$T/proc" --out "$T/models/analyser" --size small --steps 30 --window 128 --batch 16
for ARCH in transformer tcn gru; do
  python scripts/train_predictor.py --arch "$ARCH" --data-dir "$T/proc" --out "$T/models" --size small \
    --batch 8 --stride 600 --max-anchors 96 --log-every 4 --ckpt-every 8 --run-name smoke
  python scripts/evaluate.py --ckpt "$T/models/$ARCH/smoke/final.pt" --data-dir "$T/proc" --stride 1800 --stab-seconds 300
done
python scripts/select_best.py --models-dir "$T/models" --data-dir "$T/proc" --out "$T/models/best"
python tests/test_engine_consistency.py --bundle "$T/models/best" --data-dir "$T/proc" \
  --parquet data/parquet/BTCUSDT_1s_2020-01.parquet
echo "SMOKE TEST PASSED ($T)"
