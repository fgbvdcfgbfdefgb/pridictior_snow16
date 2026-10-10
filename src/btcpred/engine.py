"""Live inference engine: raw 1-second bars -> features -> Market Analyser (CPU) -> Price Predictor.

The same code path is used by the molab notebook (live biquote feed) and by the replay tests.
Every second, push_bar() appends one bar; forecast() returns the 25-minute path.

A "bundle" directory holds everything needed to run the model:
    analyser.pt, predictor.pt, stats.json (feature mean/std), analyser_stats.json, bundle.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import CONTEXT_FOR_FEATURES, HORIZON, LOOKBACK, N_SIGNALS
from .features import FEATURE_COLS, build_features, normalize
from .models import MarketAnalyser, PricePredictor, embed_range, knots_to_path

# features that need real traded volume; a quote-only feed (e.g. biquote) sends 0 here, so we
# feed the training-average value (0 after normalisation) instead of a fake "no volume" signal.
VOLUME_FEATURES = ("lvol", "taker", "ltrades")


def save_bundle(out_dir: Path, analyser: MarketAnalyser, analyser_meta: dict,
                predictor: PricePredictor, predictor_meta: dict, stats_src: Path,
                analyser_stats_src: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": analyser.state_dict(), "n_feat": analyser.inp.in_channels,
                "width": analyser.inp.out_channels, "layers": len(analyser.blocks),
                "emb": analyser.emb_dim, **analyser_meta}, out_dir / "analyser.pt")
    torch.save({"state_dict": predictor.state_dict(), **predictor_meta}, out_dir / "predictor.pt")
    (out_dir / "stats.json").write_text(stats_src.read_text())
    (out_dir / "analyser_stats.json").write_text(analyser_stats_src.read_text())


class LiveEngine:
    def __init__(self, bundle_dir: str | Path, device: str = "cpu", predictor_device: str | None = None,
                 lookback: int = LOOKBACK, quote_only: bool = False):
        """quote_only=True: the live feed has no traded volume (biquote). Volume features are then
        held at the training average both in the bootstrap history and in live bars (consistent)."""
        b = Path(bundle_dir)
        self.lookback = lookback
        self.quote_only = quote_only
        self.cap = lookback + CONTEXT_FOR_FEATURES + 10  # rows kept in the rolling buffer
        self.device_an = torch.device(device)
        self.device_pr = torch.device(predictor_device or device)

        st = json.loads((b / "stats.json").read_text())
        self.f_mean = np.asarray(st["mean"], np.float32)
        self.f_std = np.asarray(st["std"], np.float32)
        ast = json.loads((b / "analyser_stats.json").read_text())
        self.e_mean = np.asarray(ast["mean"], np.float32)
        self.e_std = np.asarray(ast["std"], np.float32)

        ack = torch.load(b / "analyser.pt", map_location="cpu", weights_only=False)
        self.analyser = MarketAnalyser(ack["n_feat"], ack["width"], ack["layers"], ack["emb"])
        self.analyser.load_state_dict(ack["state_dict"])
        self.analyser.eval().to(self.device_an)
        self.rf = self.analyser.receptive_field
        self.y_mean = np.asarray(ack["y_mean"])  # kept for reference / debugging

        pck = torch.load(b / "predictor.pt", map_location="cpu", weights_only=False)
        self.predictor = PricePredictor(pck["arch"], n_in=pck["n_in"], d_model=pck["d_model"],
                                        n_layers=pck["n_layers"], n_heads=pck["n_heads"],
                                        max_tokens=pck["lookback"] // pck["patch"] + 8)
        self.predictor.load_state_dict(pck["state_dict"])
        self.predictor.eval().to(self.device_pr)
        self.arch = pck["arch"]
        self.meta = {"arch": pck["arch"], "preset": pck.get("preset"), "step": pck.get("step")}

        self.raw = pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume",
                                         "taker_buy_volume", "trades", "missing"], dtype=float)
        self.feat = np.zeros((0, len(FEATURE_COLS)), np.float32)  # normalised features
        self.emb = np.zeros((0, self.analyser.emb_dim + N_SIGNALS), np.float32)  # normalised analyser out

    # ------------------------------------------------------------------ data in
    def _normalise(self, raw: pd.DataFrame) -> np.ndarray:
        feats = normalize(build_features(raw), self.f_mean, self.f_std)
        if self.quote_only:
            feats[:, [FEATURE_COLS.index(c) for c in VOLUME_FEATURES]] = 0.0
        return feats

    def _set_buffers(self, raw: pd.DataFrame) -> None:
        raw = raw.iloc[-self.cap:].reset_index(drop=True)
        feats = self._normalise(raw)
        emb = embed_range(self.analyser, feats, 0, len(feats), device=str(self.device_an))
        self.raw = raw
        self.feat = feats
        self.emb = ((emb.astype(np.float32) - self.e_mean) / self.e_std).astype(np.float32)

    def bootstrap(self, history: pd.DataFrame) -> None:
        """Load history (>= lookback + 1 h of 1-second bars). Uses the full history for warm-up."""
        need = self.lookback + CONTEXT_FOR_FEATURES
        if len(history) < need:
            raise ValueError(f"need at least {need} seconds of history, got {len(history)}")
        self._set_buffers(history)

    def push_bar(self, ts: int, open_: float, high: float, low: float, close: float,
                 volume: float = 0.0, taker_buy: float = 0.0, trades: float = 0.0, missing: int = 0) -> None:
        """Append one finished 1-second bar and update features + analyser output (CPU, O(RF))."""
        row = pd.DataFrame([{"ts": ts, "open": open_, "high": high, "low": low, "close": close,
                             "volume": volume, "taker_buy_volume": taker_buy, "trades": trades,
                             "missing": missing}], dtype=float)
        raw = pd.concat([self.raw, row], ignore_index=True).iloc[-self.cap:].reset_index(drop=True)
        self.raw = raw
        self.feat = self._normalise(raw)
        # analyser: only the newest second needs computing, with RF rows of warm-up
        w = self.feat[-(self.rf + 1):]
        e = embed_range(self.analyser, w, len(w) - 1, len(w), device=str(self.device_an))
        e = (e.astype(np.float32) - self.e_mean) / self.e_std
        self.emb = np.concatenate([self.emb[-(self.cap - 1):], e], axis=0)

    # ------------------------------------------------------------------ predict
    @torch.no_grad()
    def forecast(self) -> dict:
        n = min(len(self.feat), len(self.emb))
        if n < self.lookback:
            raise RuntimeError(f"not enough history yet: {n}/{self.lookback} seconds")
        x = np.concatenate([self.feat[-self.lookback:], self.emb[-self.lookback:]], axis=1)
        xt = torch.from_numpy(x).unsqueeze(0).to(self.device_pr)
        knots = self.predictor(xt)[0].float().cpu().numpy()          # % log-return at 10 s knots
        path_pct = knots_to_path(knots)                                # (HORIZON,) per second
        last_close = float(self.raw["close"].iloc[-1])
        last_ts = int(self.raw["ts"].iloc[-1])
        prices = last_close * np.exp(path_pct / 100.0)
        return {"ts": last_ts, "last_price": last_close, "knots_pct": knots, "path_pct": path_pct,
                "prices": prices, "times": last_ts + np.arange(1, HORIZON + 1)}
