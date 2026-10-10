"""
Download Binance SPOT BTCUSDT 1-second klines (2020-01 -> today) and convert
them to compressed Parquet, one file per month (daily files for the current month).

Source: https://data.binance.vision (public archive, no API key needed)

Output: data/parquet/BTCUSDT_1s_YYYY-MM.parquet
Columns (lean schema, ~40 MB per month with zstd):
    ts                 int64    open time, UTC unix seconds
    open, high, low, close  float32
    volume             float32  base-asset volume (BTC)
    taker_buy_volume   float32  base volume bought by takers (buy pressure)
    trades             int32    number of trades in the second

Usage:
    python scripts/download_binance_1s.py                 # 2020-01 .. today
    python scripts/download_binance_1s.py --start 2024-01 # custom start
    python scripts/download_binance_1s.py --keep-zip      # keep raw zips
"""
from __future__ import annotations

import argparse
import datetime as dt
import io
import os
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

BASE = "https://data.binance.vision/data/spot"
SYMBOL = "BTCUSDT"
INTERVAL = "1s"
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "parquet"
RAW_DIR = ROOT / "data" / "raw"

COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "trades", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]


def month_url(ym: str) -> str:
    return f"{BASE}/monthly/klines/{SYMBOL}/{INTERVAL}/{SYMBOL}-{INTERVAL}-{ym}.zip"


def day_url(d: dt.date) -> str:
    return f"{BASE}/daily/klines/{SYMBOL}/{INTERVAL}/{SYMBOL}-{INTERVAL}-{d.isoformat()}.zip"


def fetch(url: str, retries: int = 5) -> bytes | None:
    """Return zip bytes, or None if the file does not exist (404)."""
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            err = e
        except Exception as e:  # network hiccup
            err = e
        wait = 2 ** attempt
        print(f"  retry {attempt}/{retries} for {url.rsplit('/', 1)[-1]} ({err}); sleeping {wait}s", flush=True)
        time.sleep(wait)
    raise RuntimeError(f"failed to download {url}")


def normalise_time(s: pd.Series) -> pd.Series:
    """Binance switched some archives to microseconds in 2025; normalise to seconds."""
    s = s.astype("int64")
    return np.where(s > 1e14, s // 1_000_000, s // 1_000)  # us -> s, ms -> s


def zip_to_frame(payload: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        name = z.namelist()[0]
        with z.open(name) as f:
            df = pd.read_csv(f, header=None, names=COLS)
    # Some files carry a header row; drop it if present.
    if not str(df["open_time"].iloc[0]).lstrip("-").isdigit():
        df = df.iloc[1:].reset_index(drop=True)
    out = pd.DataFrame({
        "ts": normalise_time(df["open_time"]),
        "open": df["open"].astype("float32"),
        "high": df["high"].astype("float32"),
        "low": df["low"].astype("float32"),
        "close": df["close"].astype("float32"),
        "volume": df["volume"].astype("float32"),
        "taker_buy_volume": df["taker_buy_volume"].astype("float32"),
        "trades": df["trades"].astype("int32"),
    })
    out = out.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    return out


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False, compression="zstd", compression_level=9)
    os.replace(tmp, path)


def months_between(start: str, today: dt.date):
    y, m = map(int, start.split("-"))
    cur_y, cur_m = today.year, today.month
    while (y, m) < (cur_y, cur_m):
        yield f"{y:04d}-{m:02d}"
        m += 1
        if m == 13:
            y, m = y + 1, 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2020-01", help="first month YYYY-MM")
    ap.add_argument("--keep-zip", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    today = dt.datetime.now(dt.timezone.utc).date()

    # 1) completed months
    for ym in months_between(args.start, today):
        out = OUT_DIR / f"{SYMBOL}_{INTERVAL}_{ym}.parquet"
        if out.exists():
            print(f"skip {out.name} (exists)", flush=True)
            continue
        print(f"month {ym} ...", flush=True)
        payload = fetch(month_url(ym))
        if payload is None:
            print(f"  no monthly file for {ym}", flush=True)
            continue
        if args.keep_zip:
            (RAW_DIR / f"{SYMBOL}-{INTERVAL}-{ym}.zip").write_bytes(payload)
        df = zip_to_frame(payload)
        write_parquet(df, out)
        print(f"  wrote {out.name}: {len(df):,} rows", flush=True)

    # 2) current (incomplete) month -> daily files, one parquet per day
    cur_ym = f"{today.year:04d}-{today.month:02d}"
    if args.start <= cur_ym:
        d = dt.date(today.year, today.month, 1)
        while d < today:  # up to yesterday (today is still open)
            out = OUT_DIR / f"{SYMBOL}_{INTERVAL}_{d.isoformat()}.parquet"
            if not out.exists():
                payload = fetch(day_url(d))
                if payload is not None:
                    write_parquet(zip_to_frame(payload), out)
                    print(f"  wrote {out.name}", flush=True)
            d += dt.timedelta(days=1)

    print("done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
