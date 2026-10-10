"""OFFLINE training driver for Snowflake Notebooks / Snowpark container GPU pools.

Snowflake has no internet access, so this script:
  * never downloads anything (no data fetch, no hub/wandb calls, no pip),
  * only uses Python packages that are already installed or were uploaded to a stage
    (see README "Snowflake offline setup"),
  * runs the same four stages as the local pipeline, using the same code in src/.

Expected layout inside the Snowflake stage / notebook working dir:
    <REPO>/src, <REPO>/scripts, <REPO>/snowflake
    <REPO>/data/parquet/BTCUSDT_1s_*.parquet      (copied from the GitHub repo, offline)

Typical use (inside a Snowflake notebook cell or a container job):
    python snowflake/train_snowflake.py --repo /path/to/pridictior_snow16 --stages all --arch transformer
    python snowflake/train_snowflake.py --repo ... --stages predictor --arch tcn --nproc 4
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# ---- hard-disable every network-using library before anything imports them ----
for k, v in {
    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "WANDB_MODE": "disabled",
    "WANDB_DISABLED": "true", "NO_ALBUMENTATIONS_UPDATE": "1", "PIP_NO_INDEX": "1",
    "MPLBACKEND": "Agg", "TOKENIZERS_PARALLELISM": "false",
}.items():
    os.environ.setdefault(k, v)


def run(cmd: list[str], cwd: Path) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--stages", default="all",
                    help="comma list of: prepare,analyser,predictor,evaluate,select  (or 'all')")
    ap.add_argument("--arch", default="transformer", choices=["transformer", "tcn", "gru", "all"])
    ap.add_argument("--size", default="auto")
    ap.add_argument("--nproc", type=int, default=0, help="GPUs for DDP (0 = auto-detect)")
    ap.add_argument("--max-anchors", type=int, default=0)
    args = ap.parse_args()

    repo = args.repo.resolve()
    sys.path.insert(0, str(repo / "src"))
    from btcpred.hardware import choose_preset, detect  # noqa: E402

    hw = detect()
    print("Detected hardware:", hw, flush=True)
    if hw["n_gpus"] == 0:
        print("WARNING: no GPU visible - predictor training will be very slow on CPU.", flush=True)
    nproc = args.nproc or max(1, hw["n_gpus"])
    size = choose_preset(hw, args.size)
    print(f"Using preset: {size}, GPUs for DDP: {nproc}", flush=True)

    stages = ["prepare", "analyser", "predictor", "evaluate", "select"] if args.stages == "all" \
        else [s.strip() for s in args.stages.split(",")]
    archs = ["transformer", "tcn", "gru"] if args.arch == "all" else [args.arch]
    py = sys.executable

    for st in stages:
        if st == "prepare":
            run([py, "scripts/prepare_dataset.py"], repo)
        elif st == "analyser":
            run([py, "scripts/train_analyser.py", "--size", size], repo)
        elif st == "predictor":
            for a in archs:
                # torch.distributed.run works even when the `torchrun` console script is not on PATH
                cmd = ([py] if nproc == 1 else
                       [py, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={nproc}"])
                cmd += ["scripts/train_predictor.py", "--arch", a, "--size", size]
                if args.max_anchors:
                    cmd += ["--max-anchors", str(args.max_anchors)]
                run(cmd, repo)
        elif st == "evaluate":
            for ck in sorted((repo / "models").glob("*/*/final.pt")):
                run([py, "scripts/evaluate.py", "--ckpt", str(ck)], repo)
        elif st == "select":
            run([py, "scripts/select_best.py"], repo)
        else:
            raise SystemExit(f"unknown stage {st}")
    print("Offline run finished.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
