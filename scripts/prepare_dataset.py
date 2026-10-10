"""Step 1: turn the monthly Parquet files into one continuous, causal, per-second feature series.

Input : data/parquet/BTCUSDT_1s_*.parquet   (from scripts/download_binance_1s.py)
Output: data/processed/{features.f32, close.f64, ts.i64, meta.json, stats.json}

- Missing seconds (no trades in the archive) are inserted and forward-filled with
  flag missing=1, so the series is strictly one row per second.
- Features are normalised with mean/std of the TRAINING range only (no test leakage).

Usage:
    python scripts/prepare_dataset.py
    python scripts/prepare_dataset.py --parquet-dir data/parquet --out data/processed \
        --train-end 2026-01-01T00:00:00 --test-start 2026-07-01T00:00:00
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from btcpred.config import CONTEXT_FOR_FEATURES, PARQUET_DIR, PROCESSED_DIR, TEST_START, TRAIN_END  # noqa: E402
from btcpred.features import FEATURE_COLS, N_FEATURES, RAW_COLS, build_features, normalize  # noqa: E402


def parquet_files(folder: Path, start: str | None, end: str | None) -> list[Path]:
    files = sorted(folder.glob("BTCUSDT_1s_*.parquet"))
    out = []
    for f in files:
        stamp = f.stem.replace("BTCUSDT_1s_", "")[:10]  # YYYY-MM or YYYY-MM-DD
        if start and stamp < start[:len(stamp)]:
            continue
        if end and stamp > end[:len(stamp)]:
            continue
        out.append(f)
    return out


def regularise(df: pd.DataFrame, last_ts: int | None, last_close: float | None) -> pd.DataFrame:
    """Put one file on a continuous 1-second grid, continuing from the previous file."""
    df = df.drop_duplicates("ts").sort_values("ts")
    first = int(df["ts"].iloc[0]) if last_ts is None else last_ts + 1
    grid = np.arange(first, int(df["ts"].iloc[-1]) + 1, dtype=np.int64)
    out = df.set_index("ts").reindex(grid)
    missing = out["close"].isna().to_numpy()
    out["missing"] = missing.astype(np.float32)
    if last_close is not None and missing[0]:
        out.iloc[0, out.columns.get_loc("close")] = last_close
    out["close"] = out["close"].ffill()
    for col in ["open", "high", "low"]:
        out[col] = out[col].where(~missing, out["close"])
    for col in ["volume", "taker_buy_volume", "trades"]:
        out[col] = out[col].fillna(0)
    out = out.reset_index().rename(columns={"index": "ts"})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet-dir", type=Path, default=PARQUET_DIR)
    ap.add_argument("--out", type=Path, default=PROCESSED_DIR)
    ap.add_argument("--start", default=None, help="YYYY or YYYY-MM to include from")
    ap.add_argument("--end", default=None, help="YYYY or YYYY-MM to include until")
    ap.add_argument("--train-end", default=TRAIN_END)
    ap.add_argument("--test-start", default=TEST_START)
    args = ap.parse_args()

    files = parquet_files(args.parquet_dir, args.start, args.end)
    if not files:
        print(f"no parquet files in {args.parquet_dir}", file=sys.stderr)
        return 1
    args.out.mkdir(parents=True, exist_ok=True)
    paths = {k: args.out / v for k, v in {"f": "features.f32", "c": "close.f64", "t": "ts.i64"}.items()}
    fh = {k: open(p, "wb") for k, p in paths.items()}

    n = 0
    last_ts = last_close = None
    tail = None  # last CONTEXT_FOR_FEATURES raw rows, to keep rolling features continuous
    n_missing = 0
    for i, f in enumerate(files, 1):
        raw = pd.read_parquet(f)
        grid_df = regularise(raw, last_ts, last_close)
        if grid_df.empty:
            continue
        n_missing += int(grid_df["missing"].sum())
        joined = grid_df if tail is None else pd.concat([tail, grid_df], ignore_index=True)
        feats = build_features(joined)
        keep = len(grid_df)
        feats = feats[-keep:]
        fh["f"].write(feats.astype(np.float32).tobytes())
        fh["c"].write(grid_df["close"].to_numpy(np.float64).tobytes())
        fh["t"].write(grid_df["ts"].to_numpy(np.int64).tobytes())
        n += keep
        tail = joined.iloc[-CONTEXT_FOR_FEATURES:][["ts", *RAW_COLS]].reset_index(drop=True)
        last_ts = int(grid_df["ts"].iloc[-1])
        last_close = float(grid_df["close"].iloc[-1])
        print(f"[{i}/{len(files)}] {f.name}: {keep:,} s  (total {n:,})", flush=True)
    for h in fh.values():
        h.close()

    # ---- normalisation stats from the TRAINING range only ----
    feats = np.memmap(paths["f"], dtype=np.float32, mode="r+", shape=(n, N_FEATURES))
    ts = np.memmap(paths["t"], dtype=np.int64, mode="r", shape=(n,))
    train_end_idx = int(np.searchsorted(ts, int(np.datetime64(args.train_end, "s").astype("int64"))))
    test_start_idx = int(np.searchsorted(ts, int(np.datetime64(args.test_start, "s").astype("int64"))))
    sample = np.asarray(feats[0:max(train_end_idx, 1):50], dtype=np.float64)
    mean = sample.mean(axis=0)
    std = np.maximum(sample.std(axis=0), 1e-6)
    stats = {"feature_cols": FEATURE_COLS, "mean": mean.tolist(), "std": std.tolist()}
    (args.out / "stats.json").write_text(json.dumps(stats, indent=2))

    chunk = 5_000_000
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        feats[s:e] = normalize(np.asarray(feats[s:e]), mean, std)
    feats.flush()
    del feats, ts

    t0 = int(np.fromfile(paths["t"], dtype=np.int64, count=1)[0])
    meta = {
        "n": n,
        "n_feat": N_FEATURES,
        "feature_cols": FEATURE_COLS,
        "n_extra": 0,
        "ts_first": dt.datetime.fromtimestamp(t0, dt.timezone.utc).isoformat(),
        "ts_last": dt.datetime.fromtimestamp(int(last_ts), dt.timezone.utc).isoformat(),
        "train_end": args.train_end,
        "test_start": args.test_start,
        "train_end_idx": train_end_idx,
        "test_start_idx": test_start_idx,
        "missing_seconds": n_missing,
        "source": "Binance spot BTCUSDT 1s klines (data.binance.vision)",
        "files": [f.name for f in files],
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: meta[k] for k in ["n", "ts_first", "ts_last", "train_end_idx",
                                           "test_start_idx", "missing_seconds"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
