"""Step 3: train one Price Predictor instance (transformer | tcn | gru) on all visible GPUs.

Training protocol ("real-time reward", no epochs):
  * The training range is consumed ONCE, in time order (a single pass of the stream).
  * For each batch of anchor seconds t the model first forecasts the next 25 minutes
    (prediction made only from data <= t), the forecast is scored against the REALISED
    prices (the stored data 25 minutes later), and only then the weights are updated.
    The logged error is therefore always out-of-sample: a prequential / online reward.
  * Every checkpoint is kept, so each training run can be evaluated and compared later.

Distributed: launched with torchrun; each GPU takes every WORLD_SIZE-th anchor and gradients
are averaged with DistributedDataParallel. With no GPU it falls back to a single CPU process.

Usage:
    torchrun --standalone --nproc_per_node=4 scripts/train_predictor.py --arch transformer
    python scripts/train_predictor.py --arch gru --size small --max-anchors 20000   # quick test
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from btcpred.config import ARCHITECTURES, HORIZON, LOOKBACK, MODELS_DIR, PATCH, PRESETS  # noqa: E402
from btcpred.dataset import ProcessedSeries  # noqa: E402
from btcpred.hardware import choose_preset, detect  # noqa: E402
from btcpred.models import PricePredictor, predictor_loss  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=ARCHITECTURES)
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data" / "processed")
    ap.add_argument("--out", type=Path, default=MODELS_DIR)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--size", default="auto", choices=["auto", *PRESETS])
    ap.add_argument("--batch", type=int, default=32, help="anchors per GPU per step")
    ap.add_argument("--stride", type=int, default=10, help="seconds between training anchors")
    ap.add_argument("--max-anchors", type=int, default=0, help="cap on anchors (0 = full pass)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--lookback", type=int, default=LOOKBACK)
    ap.add_argument("--ckpt-every", type=int, default=2000)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    world = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    use_cuda = torch.cuda.is_available()
    if world > 1:
        dist.init_process_group("nccl" if use_cuda else "gloo")
    device = torch.device(f"cuda:{local_rank}" if use_cuda else "cpu")
    if use_cuda:
        torch.cuda.set_device(local_rank)
    torch.manual_seed(args.seed + rank)
    is_main = rank == 0

    hw = detect()
    size = choose_preset(hw, args.size)
    cfg = PRESETS[size]
    ds = ProcessedSeries(args.data_dir)
    if ds.n_extra == 0:
        raise SystemExit("analyser.f16 missing: run scripts/train_analyser.py first")
    run = args.run_name or f"{args.arch}_{size}_{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = args.out / args.arch / run
    if is_main:
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"hardware: {json.dumps(hw)}\npreset={size} {cfg}\nworld={world} device={device}\nrun={run_dir}",
              flush=True)

    model = PricePredictor(args.arch, n_in=ds.n_in, d_model=cfg["pred_d"], n_layers=cfg["pred_layers"],
                           n_heads=cfg["pred_heads"], max_tokens=args.lookback // PATCH + 8).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    net = DDP(model, device_ids=[local_rank] if use_cuda else None) if world > 1 else model

    # anchors: one pass over the training range, in time order, split across ranks
    rng = np.random.default_rng(args.seed)
    # targets look 25 min ahead, so the last training anchor must be >= HORIZON s before train_end
    # (otherwise test-period prices would leak into training targets)
    lo, hi = ds.anchor_range(0, ds.meta["train_end_idx"] - HORIZON - 1, args.lookback)
    anchors = np.arange(lo + int(rng.integers(0, args.stride)), hi, args.stride, dtype=np.int64)
    if args.max_anchors:
        anchors = anchors[: args.max_anchors]
    anchors = anchors[: (len(anchors) // world) * world]  # equal length per rank -> no DDP hang
    anchors = anchors[rank::world]
    global_bs = args.batch * world
    total_steps = max(1, len(anchors) // args.batch)
    warmup = min(500, max(1, total_steps // 10))

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def lr_at(step: int) -> float:
        if step < warmup:
            return args.lr * (step + 1) / warmup
        p = (step - warmup) / max(1, total_steps - warmup)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, p))))

    if is_main:
        info = {"arch": args.arch, "preset": size, "params": n_params, "world": world,
                "lookback": args.lookback, "patch": PATCH, "n_in": ds.n_in, "stride": args.stride,
                "batch_per_gpu": args.batch, "global_batch": global_bs, "total_steps": total_steps,
                "anchors_total": int(len(anchors) * world), "train_end": ds.meta["train_end"],
                "hardware": hw, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
        (run_dir / "run_info.json").write_text(json.dumps(info, indent=2))
        print(json.dumps(info, indent=2), flush=True)
        metrics = open(run_dir / "metrics.jsonl", "w")

    t0 = time.time()
    running = []
    net.train()
    for step in range(total_steps):
        a = anchors[step * args.batch:(step + 1) * args.batch]
        x, y = ds.batch(a, args.lookback)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        for g in opt.param_groups:
            g["lr"] = lr_at(step)

        pred = net(x)
        # prequential reward: error of the forecast made BEFORE learning from this batch
        online_mae = (pred.detach() - y).abs().mean()
        loss = predictor_loss(pred, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        running.append(torch.stack([loss.detach(), online_mae]))

        if (step + 1) % args.log_every == 0 or step == 0:
            r = torch.stack(running).mean(0)
            if world > 1:
                dist.all_reduce(r, op=dist.ReduceOp.AVG)
            running = []
            if is_main:
                rec = {"step": step + 1, "loss": float(r[0]), "online_mae_pct": float(r[1]),
                       "lr": lr_at(step), "elapsed_s": round(time.time() - t0, 1)}
                metrics.write(json.dumps(rec) + "\n")
                metrics.flush()
                print(f"step {step + 1}/{total_steps} loss {rec['loss']:.4f} "
                      f"online_mae {rec['online_mae_pct']:.4f}% ({rec['elapsed_s']}s)", flush=True)

        if is_main and ((step + 1) % args.ckpt_every == 0):
            save(model, run_dir / f"step_{step + 1:07d}.pt", args, size, ds, step + 1)

    if is_main:
        save(model, run_dir / "final.pt", args, size, ds, total_steps)
        metrics.close()
        info = json.loads((run_dir / "run_info.json").read_text())
        info.update({"finished": time.strftime("%Y-%m-%d %H:%M:%S"), "wall_time_s": round(time.time() - t0, 1)})
        (run_dir / "run_info.json").write_text(json.dumps(info, indent=2))
        print(f"saved final model -> {run_dir / 'final.pt'}", flush=True)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
    return 0


def save(model, path: Path, args, size: str, ds: ProcessedSeries, step: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "arch": args.arch,
        "preset": size,
        "n_in": ds.n_in,
        "d_model": PRESETS[size]["pred_d"],
        "n_layers": PRESETS[size]["pred_layers"],
        "n_heads": PRESETS[size]["pred_heads"],
        "lookback": args.lookback,
        "patch": PATCH,
        "step": step,
    }, path)


if __name__ == "__main__":
    sys.exit(main())
