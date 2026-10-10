"""Step 5: rank all evaluated checkpoints, pick the best, and export a self-contained bundle.

score = mape_25min_pct + stab_weight * stability_ema_pct_per_s
  (lower is better: accurate AND stable forecasts; stability matters because the
   forecasts drive trading decisions)

Outputs:
  models/leaderboard.csv, models/leaderboard.md
  models/best/  -> analyser.pt, predictor.pt, stats.json, analyser_stats.json, bundle.json
                   (the folder the molab notebook and LiveEngine load)

Usage:
    python scripts/select_best.py
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from btcpred.config import MODELS_DIR, PROCESSED_DIR  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    ap.add_argument("--data-dir", type=Path, default=PROCESSED_DIR)
    ap.add_argument("--stab-weight", type=float, default=1.0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    evals = sorted(args.models_dir.glob("*/*/*.eval.json"))
    if not evals:
        print("no *.eval.json found; run scripts/evaluate.py first", file=sys.stderr)
        return 1
    rows = []
    for p in evals:
        r = json.loads(p.read_text())
        r["file"] = str(p.relative_to(args.models_dir))
        r["score"] = r["mape_25min_pct"] + args.stab_weight * r["stability_ema_pct_per_s"]
        rows.append(r)
    rows.sort(key=lambda r: r["score"])

    cols = ["score", "arch", "preset", "train_step", "mape_25min_pct", "naive_flat_mape_25min_pct",
            "skill_vs_flat", "mae_25min_usd", "knot_mape_pct", "direction_accuracy_25min",
            "stability_raw_pct_per_s", "stability_ema_pct_per_s", "file"]
    with open(args.models_dir / "leaderboard.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    md = ["| rank | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for i, r in enumerate(rows, 1):
        md.append(f"| {i} | " + " | ".join(str(round(r[c], 4)) if isinstance(r[c], float) else str(r[c])
                                          for c in cols) + " |")
    (args.models_dir / "leaderboard.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))

    best = rows[0]
    ckpt = (args.models_dir / best["file"]).with_suffix("").with_suffix(".pt")  # X.eval.json -> X.pt
    out = args.out or (args.models_dir / "best")
    out.mkdir(parents=True, exist_ok=True)
    analyser_dir = args.models_dir / "analyser"
    shutil.copy(analyser_dir / "analyser.pt", out / "analyser.pt")
    shutil.copy(ckpt, out / "predictor.pt")
    shutil.copy(args.data_dir / "stats.json", out / "stats.json")
    shutil.copy(args.data_dir / "analyser_stats.json", out / "analyser_stats.json")
    (out / "bundle.json").write_text(json.dumps({
        "selected_from": best["file"], "score": best["score"], "arch": best["arch"],
        "preset": best["preset"], "metrics": {k: best[k] for k in cols if k in best and k != "file"},
    }, indent=2))
    print(f"\nBest: {best['arch']} ({best['file']}) score={best['score']:.4f} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
