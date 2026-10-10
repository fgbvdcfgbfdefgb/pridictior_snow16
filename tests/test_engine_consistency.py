"""The live engine (bar-by-bar, as in production) must reproduce the offline pipeline exactly.

Run:  python tests/test_engine_consistency.py --bundle models/best --data-dir data/processed --parquet data/parquet/BTCUSDT_1s_2020-01.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from btcpred.config import CONTEXT_FOR_FEATURES, LOOKBACK  # noqa: E402
from btcpred.dataset import ProcessedSeries  # noqa: E402
from btcpred.engine import LiveEngine  # noqa: E402
from btcpred.models import PricePredictor  # noqa: E402


def offline_knots(bundle: Path, ds: ProcessedSeries, anchor: int) -> np.ndarray:
    ck = torch.load(bundle / "predictor.pt", map_location="cpu", weights_only=False)
    m = PricePredictor(ck["arch"], n_in=ck["n_in"], d_model=ck["d_model"], n_layers=ck["n_layers"],
                       n_heads=ck["n_heads"], max_tokens=ck["lookback"] // ck["patch"] + 8)
    m.load_state_dict(ck["state_dict"]); m.eval()
    x = torch.from_numpy(ds.windows(np.array([anchor]), LOOKBACK))
    with torch.no_grad():
        return m(x)[0].numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, default=ROOT / "models" / "best")
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data" / "processed")
    ap.add_argument("--parquet", type=Path, required=True, help="raw 1 s parquet, contiguous, same month as data-dir")
    args = ap.parse_args()

    ds = ProcessedSeries(args.data_dir)
    raw = pd.read_parquet(args.parquet)[["ts", "open", "high", "low", "close", "volume",
                                         "taker_buy_volume", "trades"]].copy()
    raw["missing"] = 0.0
    raw = raw.reset_index(drop=True)
    # processed series starts at the first ts of the first file; align raw to it
    off = int(raw["ts"].iloc[0]) - int(ds.ts[0])
    assert off == 0, f"raw/processed offset {off}"

    anchor = int(ds.meta["test_start_idx"]) + 5000
    cap = LOOKBACK + CONTEXT_FOR_FEATURES + 10
    hist = raw.iloc[anchor - cap + 1: anchor].reset_index(drop=True)   # history strictly before anchor
    eng = LiveEngine(args.bundle, device="cpu")
    eng.bootstrap(hist)
    r = raw.iloc[anchor]
    eng.push_bar(ts=int(r["ts"]), open_=float(r["open"]), high=float(r["high"]), low=float(r["low"]),
                 close=float(r["close"]), volume=float(r["volume"]), taker_buy=float(r["taker_buy_volume"]),
                 trades=float(r["trades"]), missing=0)
    live = eng.forecast()["knots_pct"]
    ref = offline_knots(args.bundle, ds, anchor)
    diff = float(np.max(np.abs(live - ref)))
    print(f"max |live - offline| over 150 knots = {diff:.6f} (% log-return units)")
    assert diff < 1e-2, "engine does not match the offline pipeline"

    # a second, later anchor after several bars pushed one by one
    base = anchor + 1000
    eng2 = LiveEngine(args.bundle, device="cpu")
    eng2.bootstrap(raw.iloc[base - cap: base].reset_index(drop=True))
    for i in range(base, base + 30):
        r = raw.iloc[i]
        eng2.push_bar(ts=int(r["ts"]), open_=float(r["open"]), high=float(r["high"]), low=float(r["low"]),
                      close=float(r["close"]), volume=float(r["volume"]),
                      taker_buy=float(r["taker_buy_volume"]), trades=float(r["trades"]))
    live2 = eng2.forecast()["knots_pct"]
    ref2 = offline_knots(args.bundle, ds, base + 29)
    diff2 = float(np.max(np.abs(live2 - ref2)))
    print(f"after 30 pushed bars: max |live - offline| = {diff2:.6f}")
    assert diff2 < 1e-2
    print("ENGINE CONSISTENCY OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
