"""Step 4: score one checkpoint on the held-out test period against the STORED (real) prices.

For every evaluated second t it compares the 25-minute forecast with the price actually
recorded 25 minutes later. It also measures stability: how much consecutive forecasts
disagree about the same future second (raw, and after the EMA used in live trading).

Usage:
    python scripts/evaluate.py --ckpt models/gru/<run>/final.pt
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from btcpred.config import HORIZON, KNOT_EVERY, PATCH  # noqa: E402
from btcpred.dataset import ProcessedSeries  # noqa: E402
from btcpred.models import PricePredictor, ema_stabilise, knots_to_path  # noqa: E402


def load_model(ckpt_path: Path, device: str):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    m = PricePredictor(ck["arch"], n_in=ck["n_in"], d_model=ck["d_model"], n_layers=ck["n_layers"],
                       n_heads=ck["n_heads"], max_tokens=ck["lookback"] // ck["patch"] + 8)
    m.load_state_dict(ck["state_dict"])
    return m.eval().to(device), ck


@torch.no_grad()
def predict(model, ds: ProcessedSeries, anchors: np.ndarray, lookback: int, device: str,
            batch: int = 64) -> np.ndarray:
    out = []
    for i in range(0, len(anchors), batch):
        x = torch.from_numpy(ds.windows(anchors[i:i + batch], lookback)).to(device)
        out.append(model(x).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, HORIZON // KNOT_EVERY))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data" / "processed")
    ap.add_argument("--start-idx", type=int, default=None, help="default: test_start_idx from meta")
    ap.add_argument("--end-idx", type=int, default=None)
    ap.add_argument("--stride", type=int, default=60, help="seconds between scored anchors")
    ap.add_argument("--stab-seconds", type=int, default=1800, help="consecutive seconds for stability test")
    ap.add_argument("--ema-alpha", type=float, default=0.3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    ds = ProcessedSeries(args.data_dir)
    model, ck = load_model(args.ckpt, args.device)
    lookback = ck["lookback"]
    start = args.start_idx if args.start_idx is not None else ds.meta["test_start_idx"]
    end = args.end_idx if args.end_idx is not None else ds.n
    lo, hi = ds.anchor_range(start, end, lookback)
    t0 = time.time()

    # ---- accuracy at the 25-min horizon, every `stride` seconds ----
    anchors = np.arange(lo, hi, args.stride, dtype=np.int64)
    knots = predict(model, ds, anchors, lookback, args.device)
    actual_knots = ds.targets(anchors)                     # % log-return realised at each knot
    c0 = ds.close[anchors]
    pred_end = c0 * np.exp(knots[:, -1] / 100)
    act_end = ds.close[anchors + HORIZON]
    naive = c0                                              # "price stays flat" baseline
    mape_end = float(np.mean(np.abs(pred_end - act_end) / act_end) * 100)
    naive_mape = float(np.mean(np.abs(naive - act_end) / act_end) * 100)
    knot_idx = anchors[:, None] + KNOT_EVERY * np.arange(1, knots.shape[1] + 1)[None, :]
    pred_knot_px = c0[:, None] * np.exp(knots / 100)
    act_knot_px = ds.close[knot_idx]
    knot_mape = float(np.mean(np.abs(pred_knot_px - act_knot_px) / act_knot_px) * 100)
    dir_acc = float(np.mean(np.sign(knots[:, -1]) == np.sign(actual_knots[:, -1])))

    # ---- stability: consecutive seconds forecast the same future seconds ----
    s0 = lo + (hi - lo) // 3
    block = np.arange(s0, min(hi, s0 + args.stab_seconds), dtype=np.int64)
    bk = predict(model, ds, block, lookback, args.device)
    paths = knots_to_path(bk)                                                  # (n, HORIZON) % log-ret
    P = ds.close[block][:, None] * np.exp(paths / 100)                         # absolute price paths
    raw_jit = np.abs(P[1:, 1:] - P[:-1, :-1]) / P[:-1, :-1]                    # same absolute second
    ema = np.empty_like(P)
    ema[0] = P[0]
    for i in range(1, len(P)):
        ema[i] = ema_stabilise(ema[i - 1], P[i], args.ema_alpha)
    ema_jit = np.abs(ema[1:, 1:] - ema[:-1, :-1]) / ema[:-1, :-1]

    res = {
        "ckpt": str(args.ckpt),
        "arch": ck["arch"],
        "preset": ck["preset"],
        "train_step": ck["step"],
        "scored_anchors": int(len(anchors)),
        "test_range_unix": [int(ds.ts[start]), int(ds.ts[min(end, ds.n) - 1])],
        "mape_25min_pct": round(mape_end, 4),
        "naive_flat_mape_25min_pct": round(naive_mape, 4),
        "skill_vs_flat": round(1 - mape_end / naive_mape, 4),
        "mae_25min_usd": round(float(np.mean(np.abs(pred_end - act_end))), 2),
        "knot_mape_pct": round(knot_mape, 4),
        "direction_accuracy_25min": round(dir_acc, 4),
        "stability_raw_pct_per_s": round(float(raw_jit.mean() * 100), 6),
        "stability_ema_pct_per_s": round(float(ema_jit.mean() * 100), 6),
        "stability_ema_max_jump_pct": round(float(ema_jit.max() * 100), 4),
        "eval_seconds": round(time.time() - t0, 1),
    }
    out = args.out or args.ckpt.with_suffix(".eval.json")
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
