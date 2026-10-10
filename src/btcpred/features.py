"""Causal per-second features. Used identically offline (prepare_dataset) and live (engine).

Every feature at second t only uses data at or before t, so the live simulator and the
offline replay produce the same numbers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLS = [
    "r1",          # 1-second log return (%)
    "hl",          # log(high/low) (%)
    "co",          # log(close/open) (%)
    "lvol",        # log1p(volume)
    "taker",       # taker-buy share of volume, centred on 0
    "ltrades",     # log1p(number of trades)
    "missing",     # 1 if this second had no trades in the archive (forward-filled)
    "ret60",       # 60 s log return (%)
    "ret300",      # 5 min log return (%)
    "ret3600",     # 1 h log return (%)
    "vol300",      # 5 min std of r1 (%)
    "vol3600",     # 1 h std of r1 (%)
]
N_FEATURES = len(FEATURE_COLS)
RAW_COLS = ["open", "high", "low", "close", "volume", "taker_buy_volume", "trades", "missing"]


def build_features(df: pd.DataFrame) -> np.ndarray:
    """df: contiguous 1-second rows with RAW_COLS (+ optional ts). Returns (n, N_FEATURES) float32.

    The first rows lack history for long rolling windows; callers should drop the first
    CONTEXT_FOR_FEATURES rows of any chunk that is not the very beginning of the series.
    """
    c = df["close"].to_numpy(np.float64)
    o = df["open"].to_numpy(np.float64)
    h = df["high"].to_numpy(np.float64)
    lo = df["low"].to_numpy(np.float64)
    vol = df["volume"].to_numpy(np.float64)
    taker = df["taker_buy_volume"].to_numpy(np.float64)
    trades = df["trades"].to_numpy(np.float64)
    missing = df["missing"].to_numpy(np.float64)

    lc = np.log(c)
    r1 = np.concatenate([[0.0], np.diff(lc)]) * 100
    s = pd.Series(r1)
    with np.errstate(divide="ignore", invalid="ignore"):
        taker_share = np.where(vol > 0, taker / np.where(vol > 0, vol, 1.0), 0.5) - 0.5

    def lag_ret(n: int) -> np.ndarray:
        out = np.zeros_like(lc)
        out[n:] = (lc[n:] - lc[:-n]) * 100
        return out

    feats = np.column_stack([
        r1,
        np.log(h / lo) * 100,
        np.log(c / o) * 100,
        np.log1p(vol),
        taker_share,
        np.log1p(trades),
        missing,
        lag_ret(60),
        lag_ret(300),
        lag_ret(3600),
        s.rolling(300, min_periods=2).std().fillna(0).to_numpy(),
        s.rolling(3600, min_periods=2).std().fillna(0).to_numpy(),
    ])
    assert feats.shape[1] == N_FEATURES
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def normalize(feats: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return np.clip((feats - mean) / std, -10, 10).astype(np.float32)
