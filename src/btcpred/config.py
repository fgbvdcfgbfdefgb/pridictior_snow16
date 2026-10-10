"""Shared constants, size presets and time splits for the BTC price predictor."""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Market / horizon constants (1 row = 1 second)
SYMBOL = "BTCUSDT"
LOOKBACK = 12 * 3600          # 12 h of second-by-second history fed to the predictor
HORIZON = 25 * 60             # predict 25 minutes ahead (1500 s)
KNOT_EVERY = 10               # predictor outputs one value every 10 s, linearly interpolated to 1 s
N_KNOTS = HORIZON // KNOT_EVERY  # 150
PATCH = 60                    # predictor groups 60 seconds into one token (720 tokens for 12 h)
CONTEXT_FOR_FEATURES = 3600   # rolling features (ret_3600 / vol_3600) need 1 h of history

# Time splits (UTC). Train on everything before TRAIN_END, validate on H1-2026, test from TEST_START.
TRAIN_END = "2026-01-01T00:00:00"
TEST_START = "2026-07-01T00:00:00"

# Default locations (override with env BTCPRED_DATA / BTCPRED_MODELS)
DATA_DIR = Path(os.environ.get("BTCPRED_DATA", REPO_ROOT / "data"))
PARQUET_DIR = DATA_DIR / "parquet"
PROCESSED_DIR = Path(os.environ.get("BTCPRED_PROCESSED", DATA_DIR / "processed"))
MODELS_DIR = Path(os.environ.get("BTCPRED_MODELS", REPO_ROOT / "models"))

# Model size presets. "auto" picks one from detected hardware (see hardware.py).
PRESETS = {
    "small":  dict(an_width=64,  an_layers=8,  emb=16, pred_d=128, pred_layers=4,  pred_heads=4),
    "medium": dict(an_width=128, an_layers=10, emb=32, pred_d=256, pred_layers=6,  pred_heads=8),
    "large":  dict(an_width=256, an_layers=12, emb=48, pred_d=512, pred_layers=8,  pred_heads=8),
    "xlarge": dict(an_width=384, an_layers=13, emb=64, pred_d=768, pred_layers=12, pred_heads=12),
}

ARCHITECTURES = ("transformer", "tcn", "gru")
N_SIGNALS = 3  # analyser signals: fwd ret 60 s, fwd ret 300 s, fwd log realised-vol 300 s
