"""Memory-mapped access to the processed per-second series and training/eval batches."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .config import HORIZON, KNOT_EVERY, LOOKBACK, N_KNOTS


class ProcessedSeries:
    """Reads the arrays written by scripts/prepare_dataset.py (+ analyser.f16 if present)."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.meta = json.loads((self.root / "meta.json").read_text())
        n, f = self.meta["n"], self.meta["n_feat"]
        self.n, self.n_feat = n, f
        self.features = np.memmap(self.root / "features.f32", dtype=np.float32, mode="r", shape=(n, f))
        self.close = np.memmap(self.root / "close.f64", dtype=np.float64, mode="r", shape=(n,))
        self.ts = np.memmap(self.root / "ts.i64", dtype=np.int64, mode="r", shape=(n,))
        self.extra = None
        self.n_extra = 0
        if (self.root / "analyser.f16").exists():
            self.n_extra = self.meta["n_extra"]
            self.extra = np.memmap(self.root / "analyser.f16", dtype=np.float16, mode="r",
                                   shape=(n, self.n_extra))

    @property
    def n_in(self) -> int:
        return self.n_feat + self.n_extra

    def idx_of(self, iso_ts: str) -> int:
        """First row index with timestamp >= iso_ts (UTC)."""
        t = int(np.datetime64(iso_ts, "s").astype("int64"))
        return int(np.searchsorted(self.ts, t, side="left"))

    def windows(self, anchors: np.ndarray, lookback: int = LOOKBACK):
        """Inputs for the predictor at each anchor (second index a): rows a-L+1 .. a."""
        anchors = np.asarray(anchors, dtype=np.int64)
        idx = anchors[:, None] + np.arange(-lookback + 1, 1)[None, :]
        x = self.features[idx]
        if self.extra is not None:
            x = np.concatenate([x, self.extra[idx].astype(np.float32)], axis=-1)
        return x

    def targets(self, anchors: np.ndarray, horizon: int = HORIZON, knot_every: int = KNOT_EVERY):
        """Realised log-return (%) at each knot after each anchor, relative to the anchor's close."""
        anchors = np.asarray(anchors, dtype=np.int64)
        kidx = anchors[:, None] + knot_every * np.arange(1, horizon // knot_every + 1)[None, :]
        c0 = np.log(self.close[anchors])
        ck = np.log(self.close[kidx])
        return ((ck - c0[:, None]) * 100).astype(np.float32)

    def batch(self, anchors: np.ndarray, lookback: int = LOOKBACK):
        x = torch.from_numpy(self.windows(anchors, lookback))
        y = torch.from_numpy(self.targets(anchors))
        return x, y

    def anchor_range(self, start_idx: int, end_idx: int, lookback: int = LOOKBACK) -> tuple[int, int]:
        """Valid anchors [lo, hi): need a full lookback window and a full horizon of targets."""
        lo = max(start_idx, lookback - 1)
        hi = min(end_idx, self.n - HORIZON - 1)
        return lo, hi


def load_stats(root: str | Path) -> dict:
    return json.loads((Path(root) / "stats.json").read_text())
