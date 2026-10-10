"""Step 2: train the Market Analyser (one model, CPU by default) and cache its per-second output.

Self-supervised targets (computed from the realised future, never from the input):
    fwd return 60 s, fwd return 300 s, fwd log realised volatility over the next 300 s.
Its per-second output [embedding | signals] is saved to data/processed/analyser.f16 so the
Price Predictor can read it as extra input (the predictor never runs the analyser itself).

Usage:
    python scripts/train_analyser.py --size auto --steps 3000
    python scripts/train_analyser.py --size small --steps 50 --window 128   # quick test
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from btcpred.config import MODELS_DIR, PRESETS, N_SIGNALS  # noqa: E402
from btcpred.dataset import ProcessedSeries  # noqa: E402
from btcpred.hardware import choose_preset, detect  # noqa: E402
from btcpred.models import MarketAnalyser, embed_range, receptive_field  # noqa: E402

H1, H2, VOL_WIN = 60, 300, 300


def make_batch(ds: ProcessedSeries, starts: np.ndarray, rf: int, window: int):
    """Windows of length rf + window + H2. Loss only on positions [rf, rf+window)."""
    T = rf + window + H2
    idx = starts[:, None] + np.arange(T)[None, :]
    x = torch.from_numpy(np.asarray(ds.features[idx], dtype=np.float32))            # (B, T, F)
    lc = torch.log(torch.from_numpy(np.asarray(ds.close[idx], dtype=np.float64)).float())  # (B, T)
    p = torch.arange(rf, rf + window)
    ret60 = (lc[:, p + H1] - lc[:, p]) * 100
    ret300 = (lc[:, p + H2] - lc[:, p]) * 100
    r1 = (lc[:, 1:] - lc[:, :-1]) * 100                                              # (B, T-1)
    fut = r1.unfold(1, VOL_WIN, 1)                                                   # (B, T-1-299, 300)
    # future window for position p is r1[p .. p+299]  (r1[k] = lc[k+1]-lc[k])
    vol300 = torch.log(fut[:, p].std(dim=-1) + 1e-6)
    y = torch.stack([ret60, ret300, vol300], dim=-1)                                 # (B, W, 3)
    return x, y, rf


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data" / "processed")
    ap.add_argument("--out", type=Path, default=MODELS_DIR / "analyser")
    ap.add_argument("--size", default="auto", choices=["auto", *PRESETS])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--window", type=int, default=512, help="loss positions per sequence")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--device", default="cpu", help="cpu (runtime placement) or cuda for faster training")
    ap.add_argument("--no-precompute", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_num_threads(max(1, (torch.get_num_threads())))
    hw = detect()
    size = choose_preset(hw, args.size)
    cfg = PRESETS[size]
    print(f"hardware={hw}\nanalyser preset={size} {cfg}", flush=True)

    ds = ProcessedSeries(args.data_dir)
    model = MarketAnalyser(ds.n_feat, cfg["an_width"], cfg["an_layers"], cfg["emb"])
    rf = model.receptive_field
    print(f"receptive field = {rf} s, params = {sum(p.numel() for p in model.parameters()):,}", flush=True)
    dev = torch.device(args.device)
    model.to(dev)

    # target scaling from a random sample (fixed afterwards)
    lo, hi = 0, ds.meta["train_end_idx"] - (rf + args.window + H2) - 1
    sample_starts = np.random.randint(lo, hi, size=256)
    _, ys, _ = make_batch(ds, sample_starts, rf, args.window)
    y_mean = ys.reshape(-1, N_SIGNALS).mean(0)
    y_std = ys.reshape(-1, N_SIGNALS).std(0).clamp_min(1e-3)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    log = []
    t0 = time.time()
    for step in range(1, args.steps + 1):
        starts = np.random.randint(lo, hi, size=args.batch)
        x, y, _ = make_batch(ds, starts, rf, args.window)
        x, y = x.to(dev), ((y - y_mean) / y_std).to(dev)
        pred = model(x)[:, rf:rf + args.window, -N_SIGNALS:]  # signals are the last channels
        loss = F.smooth_l1_loss(pred, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 50 == 0 or step == 1:
            log.append({"step": step, "loss": float(loss.detach())})
            print(f"step {step:6d}  loss {float(loss.detach()):.4f}  ({time.time() - t0:.0f}s)", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "n_feat": ds.n_feat, "width": cfg["an_width"],
                "layers": cfg["an_layers"], "emb": cfg["emb"], "preset": size,
                "y_mean": y_mean.tolist(), "y_std": y_std.tolist()}, args.out / "analyser.pt")
    (args.out / "train_log.json").write_text(json.dumps(log, indent=2))
    print(f"saved {args.out / 'analyser.pt'}", flush=True)

    if args.no_precompute:
        return 0

    # ---- precompute the analyser output for the whole series (causal, once) ----
    # Normalisation stats come from the first 2M seconds (2020), streamed in blocks: no full copy in RAM.
    n_sample = min(ds.n, 2_000_000)
    raw_sample = embed_range(model, ds.features, 0, n_sample, device=args.device).astype(np.float32)
    mu = raw_sample[::50].mean(0)
    sd = np.maximum(raw_sample[::50].std(0), 1e-6)
    n_out = raw_sample.shape[1]
    out = np.memmap(ds.root / "analyser.f16", dtype=np.float16, mode="w+", shape=(ds.n, n_out))
    block = 5_000_000
    for s0 in range(0, ds.n, block):
        e0 = min(ds.n, s0 + block)
        blk = embed_range(model, ds.features, s0, e0, device=args.device).astype(np.float32)
        out[s0:e0] = ((blk - mu) / sd).astype(np.float16)
        print(f"  analyser rows {e0:,}/{ds.n:,}", flush=True)
    out.flush()
    (ds.root / "analyser_stats.json").write_text(json.dumps({"mean": mu.tolist(), "std": sd.tolist()}))
    meta = json.loads((ds.root / "meta.json").read_text())
    meta["n_extra"] = int(n_out)
    meta["analyser_preset"] = size
    (ds.root / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote analyser.f16 with {n_out} extra columns for {ds.n:,} seconds", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
