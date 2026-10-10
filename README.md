# pridictior_snow16 — Bitcoin 25-minute price predictor

Predicts the BTCUSDT price **25 minutes ahead**, updated **every second**, from:

* a **Market Analyser** (one model, runs on **CPU**): causal conv net over per-second features,
  outputs an embedding + signals (fwd return 60 s / 300 s, fwd realised volatility);
* a **Price Predictor** (runs on **GPU**, several instances on any number of GPUs): reads the
  **last 12 h of second-by-second data** plus the Analyser output and forecasts the next
  1500 seconds. Three interchangeable architectures: `transformer`, `tcn`, `gru`.

Data: Binance spot **BTCUSDT 1-second klines, 2020-01-01 → today** (public archive,
`data.binance.vision`), stored as monthly Parquet files in `data/parquet/`.

---

## 1. Repository layout

```
data/parquet/                  1-second BTCUSDT data (committed in chunks, one file per month)
scripts/download_binance_1s.py refresh / extend the dataset from data.binance.vision
scripts/prepare_dataset.py     Parquet -> continuous causal feature series (+ normalisation stats)
scripts/train_analyser.py      Market Analyser (CPU) + precomputed per-second output
scripts/train_predictor.py     Price Predictor, DDP over all GPUs, one pass, prequential reward
scripts/evaluate.py            scores every checkpoint vs the STORED (real) prices + stability
scripts/select_best.py         leaderboard + exports the best model bundle to models/best/
scripts/run_all_architectures.sh   the whole pipeline in one command
scripts/make_offline_wheels.sh     builds offline wheels for Snowflake (run with internet)
scripts/push_data_chunks.sh        pushes the dataset to GitHub in ~6-month batches
src/btcpred/                   shared code: config, hardware detection, features, models,
                               dataset access, live inference engine
snowflake/train_snowflake.py   OFFLINE driver for Snowflake (no network calls at all)
notebooks/molab_live_predictor.py  marimo notebook: live biquote chart + live predictions (molab)
tests/test_engine_consistency.py   proves the live engine == offline pipeline bit-for-bit (±float16)
requirements.txt / requirements-offline.txt
```

## 2. Setup (online machine: laptop, cloud GPU box, molab)

```bash
git clone https://github.com/fgbvdcfgbfdefgb/pridictior_snow16.git
cd pridictior_snow16
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # for GPUs install the torch build matching your CUDA (pytorch.org)
```

Check what compute is visible (the trainer does this automatically):

```bash
PYTHONPATH=src python -m btcpred.hardware    # CPUs, RAM, GPUs and the size preset it would choose
```

Size presets (`small | medium | large | xlarge`) are picked from the hardware (`--size auto`) and
scale **both** the Market Analyser (width/depth) and the Price Predictor (d_model/layers/heads).

## 3. Run the whole pipeline

```bash
# 0) (optional) refresh the dataset up to today
python scripts/download_binance_1s.py                 # 2020-01 .. today, resumable

# 1) features (CPU, roughly 10-15 min for 2020-2026)
python scripts/prepare_dataset.py

# 2) Market Analyser: one model, trained on CPU; caching its output for all ~190M seconds
#    takes roughly an hour or more on CPU, so add --device cuda when a GPU is available
python scripts/train_analyser.py --size auto --device cpu          # use --device cuda if you have a GPU

# 3) Price Predictor: multiple instances, DDP over every GPU
torchrun --standalone --nproc_per_node=$(nvidia-smi -L | wc -l) scripts/train_predictor.py --arch transformer
torchrun --standalone --nproc_per_node=$(nvidia-smi -L | wc -l) scripts/train_predictor.py --arch tcn
torchrun --standalone --nproc_per_node=$(nvidia-smi -L | wc -l) scripts/train_predictor.py --arch gru
#   (or simply: scripts/run_all_architectures.sh auto)
#   every checkpoint is kept in models/<arch>/<run>/step_XXXXXXX.pt and final.pt
#   --stride (default 10 s) sets how many anchor seconds are used; a full pass at stride 10
#   is ~19M anchors, so use --stride 60 or --max-anchors to shorten a first run

# 4) evaluate every final model against the stored prices, then pick the best
for CK in models/*/*/final.pt; do python scripts/evaluate.py --ckpt "$CK"; done
python scripts/select_best.py          # -> models/leaderboard.md and models/best/ (bundle)
```

### How training works (what "real-time reward" means here)

* **No epochs.** The training range (2020-01 → 2025-12) is consumed **once, in time order**.
* For each anchor second *t* the predictor first forecasts +25 min using only data ≤ *t*,
  the forecast is scored against the **stored prices** 25 minutes later, and only then are the
  weights updated. The logged `online_mae_pct` is therefore always out-of-sample.
* Forecasts are 150 knots (every 10 s), linearly interpolated to 1 s. A curvature penalty keeps
  the 25-minute path smooth.

### Stability (why the predictions don't jump)

* Training: curvature penalty on each forecast path.
* Inference: `ema_stabilise()` blends each new forecast with the previous one (α = 0.3) in
  absolute price units, the same rule used in the notebook.
* Evaluation: `stability_raw_pct_per_s` and `stability_ema_pct_per_s` measure how much
  consecutive forecasts disagree about the same future second.
* Model selection score: `mape_25min_pct + stability_ema_pct_per_s`.

### Evaluation (against real data)

`scripts/evaluate.py` compares, on the test period 2026-07-01 → now:

| metric | meaning |
|---|---|
| `mape_25min_pct` | error of the 25-min forecast vs the price actually stored 25 min later |
| `naive_flat_mape_25min_pct` | baseline "price stays where it is" |
| `skill_vs_flat` | 1 − model/baseline; **must be > 0 to be worth anything** |
| `direction_accuracy_25min` | how often the sign of the 25-min move was right |
| `stability_*` | forecast jitter (see above) |

> Crypto returns at a 25-minute horizon are very hard to predict. If `skill_vs_flat` is not
> clearly positive out-of-sample, do not trade on the model.

## 4. Live on molab (GPU) with marimo

1. Produce a bundle with `scripts/select_best.py` (folder `models/best/`). If the bundle is
   larger than GitHub's 100 MB file limit (e.g. `xlarge`), upload it as a GitHub Release asset
   and download it into molab.
2. Open `notebooks/molab_live_predictor.py` in molab (or `marimo edit notebooks/molab_live_predictor.py`).
3. Set **Model bundle folder** to that folder, then switch **Run live feed** on.

What happens:

* **Warm-up:** the last 13 h of real 1-second BTCUSDT bars are loaded from the Binance public
  REST API (`data-api.binance.vision`, no key).
* **Live:** biquote (`https://biquote.io/api/BTCUSD`, no key) is polled once per second. Each
  second becomes one bar; the Market Analyser and the Price Predictor run; the forecast is drawn.
* **One chart:** a single widget is created once and redrawn in place every second, in a
  CoinMarketCap-style layout (price header, range tabs 5M / 1H / 12H / ALL, blue area line,
  dashed orange 25-minute forecast, price bubble, crosshair).

Notes:

* biquote quotes come from an MT5 broker and have no traded volume. The notebook runs the engine
  with `quote_only=True`, which holds the volume features at their training average, both in the
  warm-up and live. This is consistent, but it means volume-based signals are not used live.
* Binance and biquote differ by a small offset. The notebook measures it once at start-up and
  aligns biquote to the Binance level (`basis` in the status line). It is not re-estimated.
* The 12 h of context is built from Binance, so the first forecast is ready immediately.

## 5. Snowflake (no internet): offline setup

Snowflake Notebooks / container jobs cannot reach the internet, so **nothing is downloaded at run time**.

1. On a machine **with** internet:
   ```bash
   bash scripts/make_offline_wheels.sh offline_wheels     # packages only
   ```
   Upload the `offline_wheels/` folder and the whole repo (including `data/parquet/`) to a Snowflake
   stage, e.g. `@BTC_STAGE/pridictior_snow16/`.
2. Inside the Snowflake notebook / container:
   ```bash
   pip install --no-index --find-links /path/to/offline_wheels -r requirements-offline.txt
   python snowflake/train_snowflake.py --repo /path/to/pridictior_snow16 --stages all --arch all
   # or stage by stage: --stages prepare,analyser ; --stages predictor --arch transformer
   ```
3. The driver detects the GPUs (`--nproc` defaults to the count it finds) and uses `torch.distributed.run`
   for multi-GPU training. The same code, offline, produces the same `models/` layout.

The driver sets `HF_HUB_OFFLINE`, `WANDB_MODE=disabled` and similar switches, and never calls
`pip`, `curl` or any URL.

## 6. Dataset details and size

| item | value |
|---|---|
| source | Binance spot `BTCUSDT`, `1s` klines (`data.binance.vision`) |
| rows | 1 per second, contiguous; seconds without trades are inserted with `missing=1` (forward-filled) |
| raw Parquet | ≈ 40 MB per month (`zstd`), ≈ 3.2 GB for 2020-01 → 2026-10 |
| columns | `ts` (unix s), `open`, `high`, `low`, `close`, `volume`, `taker_buy_volume`, `trades` |
| processed | `data/processed/` (not committed): 12 causal features, 19 analyser columns, per-second |

GitHub limits: a hard 100 MB cap per file, a warning above 50 MB, and a recommended repo size under 1 GB.
Monthly files are 40-63 MB, so they pass the hard cap but some trigger the 50 MB warning (harmless).
The full dataset is about 3.3 GB, so the push is split into batches (`scripts/push_data_chunks.sh`).

## 7. Tests

```bash
bash tests/smoke_test.sh                       # full chain on a 1-month slice, tiny models (CPU, ~5 min)
python tests/test_engine_consistency.py \
  --bundle models/best --data-dir data/processed --parquet data/parquet/BTCUSDT_1s_2020-01.parquet
```

## 8. Known limits

* Training was not run on a GPU in this build environment (no GPU available here); the code paths
  for DDP were exercised on CPU with the `gloo` backend. Run the real training on your GPUs.
* The 12 h history per prediction makes each forecast heavier than a plain tick model; the
  predictor sees the whole 12 h as 60-second tokens (720 tokens), so it stays fast.
* This is research code, not financial advice. Check `skill_vs_flat` before using any output.
