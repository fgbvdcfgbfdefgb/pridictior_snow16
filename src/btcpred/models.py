"""Market Analyser (runs on CPU) and Price Predictor (runs on GPU) models."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import HORIZON, KNOT_EVERY, N_KNOTS, N_SIGNALS, PATCH


# ----------------------------------------------------------------------------
# building blocks
# ----------------------------------------------------------------------------
class ChanLayerNorm(nn.Module):
    """LayerNorm over channels only (never over time, so it stays causal)."""

    def __init__(self, ch: int):
        super().__init__()
        self.ln = nn.LayerNorm(ch)

    def forward(self, x):  # (B, C, T)
        return self.ln(x.transpose(1, 2)).transpose(1, 2)


class CausalResBlock(nn.Module):
    def __init__(self, ch: int, dilation: int, k: int = 3):
        super().__init__()
        self.pad = (k - 1) * dilation
        self.norm = ChanLayerNorm(ch)
        self.conv = nn.Conv1d(ch, ch, k, dilation=dilation)
        self.out = nn.Conv1d(ch, ch, 1)

    def forward(self, x):  # (B, C, T)
        h = self.norm(x)
        h = self.conv(F.pad(h, (self.pad, 0)))  # left-pad only -> causal
        return x + self.out(F.gelu(h))


def receptive_field(n_layers: int) -> int:
    return 1 + 2 * sum(2 ** (i % 9) for i in range(n_layers))


@torch.no_grad()
def embed_range(model: "MarketAnalyser", feats: np.ndarray, start: int, end: int,
                device: str = "cpu", chunk: int = 200_000) -> np.ndarray:
    """Analyser outputs for rows [start, end) of a normalised feature array.

    Uses receptive-field warm-up rows before `start`, so results are identical to running
    the analyser over the whole history (causal, no look-ahead).
    """
    rf = model.receptive_field
    out = np.empty((end - start, model.emb_dim + model.n_signals), dtype=np.float16)
    model.eval().to(device)
    for s in range(start, end, chunk):
        e = min(end, s + chunk)
        w0 = max(0, s - rf)
        x = torch.from_numpy(np.array(feats[w0:e], dtype=np.float32)).unsqueeze(0).to(device)
        y = model(x)[0, s - w0:].cpu().numpy()
        out[s - start:e - start] = y.astype(np.float16)
    return out


# ----------------------------------------------------------------------------
# Market Analyser: causal dilated conv over per-second features.
# Output per second = [embedding (E) | signals (S)]
#   signals: fwd return 60 s, fwd return 300 s, fwd log realised-vol 300 s
# ----------------------------------------------------------------------------
class MarketAnalyser(nn.Module):
    def __init__(self, n_feat: int, width: int = 64, n_layers: int = 8, emb_dim: int = 16):
        super().__init__()
        self.emb_dim = emb_dim
        self.n_signals = N_SIGNALS
        self.inp = nn.Conv1d(n_feat, width, 1)
        self.blocks = nn.ModuleList([CausalResBlock(width, 2 ** (i % 9)) for i in range(n_layers)])
        self.norm = ChanLayerNorm(width)
        self.emb = nn.Conv1d(width, emb_dim, 1)
        self.sig = nn.Conv1d(width, N_SIGNALS, 1)
        self.receptive_field = receptive_field(n_layers)

    def forward(self, x):  # x: (B, T, F) -> (B, T, E+S)
        h = self.inp(x.transpose(1, 2))
        for b in self.blocks:
            h = b(h)
        h = self.norm(h)
        return torch.cat([self.emb(h), self.sig(h)], dim=1).transpose(1, 2)


# ----------------------------------------------------------------------------
# Price Predictor: 12 h of per-second input -> 150 knots (every 10 s) of the
# 25-minute log-return path (in percent). Three interchangeable architectures.
# ----------------------------------------------------------------------------
class PricePredictor(nn.Module):
    def __init__(self, arch: str, n_in: int, d_model: int = 128, n_layers: int = 4,
                 n_heads: int = 4, max_tokens: int = 1024, patch: int = PATCH,
                 n_knots: int = N_KNOTS):
        super().__init__()
        if arch not in ("transformer", "tcn", "gru"):
            raise ValueError(arch)
        self.arch, self.patch, self.n_knots = arch, patch, n_knots
        self.proj = nn.Sequential(nn.Linear(patch * n_in, d_model), nn.LayerNorm(d_model))
        if arch == "transformer":
            self.max_tokens = max_tokens
            self.pos = nn.Parameter(torch.randn(1, max_tokens, d_model) * 0.02)
            layer = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout=0.0,
                                               activation="gelu", batch_first=True, norm_first=True)
            self.body = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        elif arch == "tcn":
            self.body = nn.Sequential(*[CausalResBlock(d_model, 2 ** (i % 6)) for i in range(n_layers)])
        else:  # gru
            self.body = nn.GRU(d_model, d_model, num_layers=n_layers, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 2 * d_model),
                                  nn.GELU(), nn.Linear(2 * d_model, n_knots))

    def forward(self, x):  # x: (B, L, n_in) with L % patch == 0 -> (B, n_knots)
        B, L, C = x.shape
        T = L // self.patch
        t = self.proj(x.reshape(B, T, self.patch * C))
        if self.arch == "transformer":
            if T > self.max_tokens:
                raise ValueError("increase max_tokens")
            h = self.body(t + self.pos[:, :T])[:, -1]
        elif self.arch == "tcn":
            h = self.body(t.transpose(1, 2))[:, :, -1]
        else:
            out, _ = self.body(t)
            h = out[:, -1]
        return self.head(h)


# ----------------------------------------------------------------------------
# loss, interpolation and stabilisation helpers
# ----------------------------------------------------------------------------
def predictor_loss(pred: torch.Tensor, target: torch.Tensor, smooth_w: float = 0.1) -> torch.Tensor:
    """Huber on knots + penalty on 2nd differences so 25-min paths stay smooth."""
    base = F.smooth_l1_loss(pred, target)
    curvature = pred[:, 2:] - 2 * pred[:, 1:-1] + pred[:, :-2]
    return base + smooth_w * curvature.pow(2).mean()


def knots_to_path(knots_pct: np.ndarray) -> np.ndarray:
    """(…, N_KNOTS) knot log-returns (%) -> (…, HORIZON) per-second path, starting at 0 at t=0."""
    knots_pct = np.asarray(knots_pct, dtype=np.float64)
    tk = np.arange(0, HORIZON + 1, KNOT_EVERY)
    tau = np.arange(1, HORIZON + 1)
    full = np.concatenate([np.zeros(knots_pct.shape[:-1] + (1,)), knots_pct], axis=-1)
    flat = full.reshape(-1, full.shape[-1])
    out = np.stack([np.interp(tau, tk, row) for row in flat])
    return out.reshape(knots_pct.shape[:-1] + (HORIZON,))


def ema_stabilise(prev_path: np.ndarray | None, new_path: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """Blend the new 25-min forecast with the previous one (shifted by 1 s) to stop jitter.

    Pass both paths in ABSOLUTE units (e.g. log price) so they are comparable.
    prev_path[tau] was forecast for absolute time t0+tau; one second later that same absolute
    time is index tau-1 relative to the new anchor, so the previous path is shifted by one.
    """
    if prev_path is None:
        return new_path
    shifted = np.concatenate([prev_path[1:], prev_path[-1:]])
    return alpha * new_path + (1 - alpha) * shifted
