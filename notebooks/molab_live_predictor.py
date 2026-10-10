# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.14.0",
#     "anywidget>=0.9.0",
#     "traitlets>=5.9",
#     "numpy>=1.26",
#     "pandas>=2.1",
#     "pyarrow>=15.0",
#     "torch>=2.3",
# ]
# ///
"""BTC 25-minute price predictor - LIVE on molab (marimo).

* Live ticks: biquote REST (https://biquote.io/api/BTCUSD), no key needed.
* 12-hour warm-up history: Binance public 1-second klines (data-api.binance.vision).
* Market Analyser runs on CPU, Price Predictor runs on the GPU if molab provides one.
* ONE chart widget is created once and updated in place every second (no new graphs).

Run on molab:  open this file in molab (or `marimo edit notebooks/molab_live_predictor.py`).
Set "Model bundle folder" to a folder produced by scripts/select_best.py (models/best).
"""

import marimo

__generated_with = "0.14.0"
app = marimo.App(width="full", app_title="BTC 25-min predictor (live)")


@app.cell
def _():
    import json
    import os
    import subprocess
    import sys
    import threading
    import time
    import urllib.request
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import pandas as pd

    return Path, json, mo, np, os, pd, subprocess, sys, threading, time, urllib


@app.cell
def _(Path, mo, os, subprocess, sys):
    # --- get the project code (src/ + models/) ---------------------------------------
    REPO_URL = "https://github.com/fgbvdcfgbfdefgb/pridictior_snow16.git"
    _local = Path("src/btcpred")
    if _local.exists():
        ROOT = Path(".")
    else:
        ROOT = Path("pridictior_snow16")
        if not (ROOT / "src" / "btcpred").exists():
            subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(ROOT)], check=True)
    sys.path.insert(0, str(ROOT / "src"))

    bundle_path = mo.ui.text(
        value=os.environ.get("BTC_BUNDLE", str(ROOT / "models" / "best")),
        label="Model bundle folder (analyser.pt, predictor.pt, stats.json, analyser_stats.json)",
    )
    start_switch = mo.ui.switch(label="Run live feed", value=False)
    mo.vstack([bundle_path, start_switch])
    return ROOT, bundle_path, start_switch


@app.cell
def _(mo):
    # ---- the single chart widget (created once; updated in place) -----------------
    import anywidget
    import traitlets

    CHART_ESM = r"""
const TF = [
  { k: "5M", s: 300 }, { k: "1H", s: 3600 }, { k: "12H", s: 43200 }, { k: "ALL", s: Infinity },
];
const BLUE = "#3861fb", ORANGE = "#ea8c00", GREEN = "#16c784", RED = "#ea3943", GREY = "#808a9d";

function fmtUsd(p) {
  return "$" + Number(p).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}
function fmtTime(t, span) {
  const d = new Date(t * 1000), pad = (n) => String(n).padStart(2, "0");
  if (span > 86400 * 2) return `${d.getMonth() + 1}/${d.getDate()}`;
  if (span > 3600 * 6) return `${pad(d.getHours())}:${pad(d.getMinutes())}`;
  return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
}

export function render({ model, el }) {
  el.innerHTML = "";
  const wrap = document.createElement("div");
  wrap.className = "cmc";
  wrap.innerHTML = `
    <div class="cmc-head">
      <div class="coin">
        <span class="logo">&#8383;</span>
        <div>
          <div class="name">Bitcoin <span class="sym">BTC</span></div>
          <div class="src">BTCUSD &middot; live ticks from biquote &middot; 25-min model forecast</div>
        </div>
      </div>
      <div class="price"><span class="p" data-price>&mdash;</span><span class="chg" data-chg></span></div>
    </div>
    <div class="cmc-tabs"></div>
    <div class="cmc-stage"><canvas></canvas><div class="tip" style="display:none"></div></div>
    <div class="cmc-foot"><span data-fc>25-min forecast: waiting for data</span><span data-status>starting&hellip;</span></div>`;
  el.appendChild(wrap);

  const canvas = wrap.querySelector("canvas");
  const ctx = canvas.getContext("2d");
  const tip = wrap.querySelector(".tip");
  const tabs = wrap.querySelector(".cmc-tabs");
  let range = "1H";
  let hist = (model.get("history") || []).slice();
  let fc = [];
  let stats = {};
  let hover = null;

  TF.forEach((r) => {
    const b = document.createElement("button");
    b.textContent = r.k;
    b.className = r.k === range ? "on" : "";
    b.onclick = () => {
      range = r.k;
      [...tabs.children].forEach((x) => (x.className = x.textContent === range ? "on" : ""));
      draw();
    };
    tabs.appendChild(b);
  });

  function header() {
    const last = hist.length ? hist[hist.length - 1][1] : null;
    const tf = TF.find((x) => x.k === range);
    const data = tf.s === Infinity ? hist : hist.slice(Math.max(0, hist.length - tf.s));
    const first = data.length ? data[0][1] : null;
    const pEl = wrap.querySelector("[data-price]");
    const cEl = wrap.querySelector("[data-chg]");
    if (last !== null) pEl.textContent = fmtUsd(last);
    if (last !== null && first !== null && first !== 0) {
      const pct = ((last - first) / first) * 100;
      cEl.textContent = `${pct >= 0 ? "\u25B2" : "\u25BC"} ${Math.abs(pct).toFixed(3)}% (${range})`;
      cEl.style.color = pct >= 0 ? GREEN : RED;
    }
    if (stats.fc_end !== undefined) {
      const d = stats.fc_change_pct;
      wrap.querySelector("[data-fc]").innerHTML =
        `25-min forecast: <b>${fmtUsd(stats.fc_end)}</b> ` +
        `<span style="color:${d >= 0 ? GREEN : RED}">${d >= 0 ? "+" : ""}${d.toFixed(3)}%</span>` +
        ` &middot; stabilised (EMA)`;
    }
    if (stats.status) wrap.querySelector("[data-status]").textContent = stats.status;
  }

  function draw() {
    const dpr = window.devicePixelRatio || 1;
    const W = canvas.clientWidth, H = canvas.clientHeight;
    if (!W || !H) return;
    if (canvas.width !== Math.round(W * dpr) || canvas.height !== Math.round(H * dpr)) {
      canvas.width = Math.round(W * dpr);
      canvas.height = Math.round(H * dpr);
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, W, H);

    const tf = TF.find((x) => x.k === range);
    const data = tf.s === Infinity ? hist : hist.slice(Math.max(0, hist.length - tf.s));
    if (!data.length) {
      ctx.fillStyle = GREY;
      ctx.font = "14px system-ui, sans-serif";
      ctx.fillText("Waiting for live data\u2026", 16, 28);
      return;
    }
    const padL = 10, padR = 84, padT = 14, padB = 26;
    const t0 = data[0][0], tLast = data[data.length - 1][0];
    const tFc = fc.length ? fc[fc.length - 1][0] : tLast;
    const span = tFc - t0;
    const X = (t) => padL + ((t - t0) / (span || 1)) * (W - padL - padR);

    let yMin = Infinity, yMax = -Infinity;
    for (const p of data) { if (p[1] < yMin) yMin = p[1]; if (p[1] > yMax) yMax = p[1]; }
    for (const p of fc) { if (p[1] < yMin) yMin = p[1]; if (p[1] > yMax) yMax = p[1]; }
    const pad = (yMax - yMin) * 0.12 || yMax * 0.0005;
    yMin -= pad; yMax += pad;
    const Y = (p) => padT + (1 - (p - yMin) / (yMax - yMin || 1)) * (H - padT - padB);

    // grid + right-hand price scale (CoinMarketCap style)
    ctx.strokeStyle = "#eef0f3";
    ctx.lineWidth = 1;
    ctx.fillStyle = GREY;
    ctx.font = "11px system-ui, sans-serif";
    ctx.textAlign = "left";
    for (let i = 0; i <= 5; i++) {
      const v = yMin + ((yMax - yMin) * i) / 5, y = Y(v);
      ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(W - padR, y); ctx.stroke();
      ctx.fillText(Number(v).toLocaleString(undefined, { maximumFractionDigits: 0 }), W - padR + 8, y + 4);
    }
    ctx.textAlign = "center";
    for (let i = 0; i <= 4; i++) {
      const t = t0 + (span * i) / 4;
      ctx.fillText(fmtTime(t, span), X(t), H - 8);
    }

    // area + line for the observed price
    const pts = data.map((p) => [X(p[0]), Y(p[1])]);
    const grad = ctx.createLinearGradient(0, padT, 0, H - padB);
    grad.addColorStop(0, "rgba(56,97,251,0.28)");
    grad.addColorStop(1, "rgba(56,97,251,0.00)");
    ctx.beginPath();
    pts.forEach((q, i) => (i ? ctx.lineTo(q[0], q[1]) : ctx.moveTo(q[0], q[1])));
    ctx.lineTo(pts[pts.length - 1][0], H - padB);
    ctx.lineTo(pts[0][0], H - padB);
    ctx.closePath();
    ctx.fillStyle = grad;
    ctx.fill();

    ctx.beginPath();
    pts.forEach((q, i) => (i ? ctx.lineTo(q[0], q[1]) : ctx.moveTo(q[0], q[1])));
    ctx.strokeStyle = BLUE;
    ctx.lineWidth = 2;
    ctx.lineJoin = "round";
    ctx.stroke();

    // 25-minute forecast (dashed), starts at the last observed price
    if (fc.length) {
      const last = data[data.length - 1];
      ctx.beginPath();
      ctx.moveTo(X(last[0]), Y(last[1]));
      fc.forEach((p) => ctx.lineTo(X(p[0]), Y(p[1])));
      ctx.setLineDash([6, 4]);
      ctx.strokeStyle = ORANGE;
      ctx.lineWidth = 2;
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // current price bubble
    const lastP = data[data.length - 1][1];
    const yl = Y(lastP);
    ctx.fillStyle = BLUE;
    ctx.fillRect(W - padR + 2, yl - 10, padR - 6, 20);
    ctx.fillStyle = "#fff";
    ctx.textAlign = "left";
    ctx.font = "600 11px system-ui, sans-serif";
    ctx.fillText(Number(lastP).toLocaleString(undefined, { maximumFractionDigits: 2 }), W - padR + 7, yl + 4);

    // crosshair
    if (hover !== null) {
      let best = 0, bd = Infinity;
      pts.forEach((q, i) => { const d = Math.abs(q[0] - hover); if (d < bd) { bd = d; best = i; } });
      const q = pts[best];
      ctx.strokeStyle = "rgba(23,23,23,0.25)";
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(q[0], padT); ctx.lineTo(q[0], H - padB); ctx.stroke();
      ctx.fillStyle = BLUE;
      ctx.beginPath(); ctx.arc(q[0], q[1], 4, 0, Math.PI * 2); ctx.fill();
      tip.style.display = "block";
      tip.style.left = Math.min(W - 170, q[0] + 10) + "px";
      tip.style.top = Math.max(0, q[1] - 40) + "px";
      tip.innerHTML = `${fmtUsd(data[best][1])}<br>${new Date(data[best][0] * 1000).toISOString().slice(0, 19).replace("T", " ")} UTC`;
    }
  }

  canvas.addEventListener("mousemove", (e) => {
    const r = canvas.getBoundingClientRect();
    hover = e.clientX - r.left;
    draw();
  });
  canvas.addEventListener("mouseleave", () => { hover = null; tip.style.display = "none"; draw(); });

  model.on("change:history", () => {
    hist = (model.get("history") || []).slice();
    header(); draw();
  });
  model.on("change:tick", () => {
    const t = model.get("tick");
    if (!t || t.t === undefined) return;
    if (!hist.length || hist[hist.length - 1][0] !== t.t) hist.push([t.t, t.price]);
    if (hist.length > 120000) hist.splice(0, hist.length - 120000);
    fc = t.forecast || [];
    stats = t.stats || {};
    header(); draw();
  });

  const ro = new ResizeObserver(() => draw());
  ro.observe(wrap);
  header();
  draw();
  return () => ro.disconnect();
}
"""

    CHART_CSS = r"""
.cmc { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  color: #171717; background: #fff; border: 1px solid #e0e3eb; border-radius: 12px;
  padding: 16px 18px; box-sizing: border-box; max-width: 1100px; }
.cmc-head { display: flex; justify-content: space-between; align-items: flex-end; flex-wrap: wrap; gap: 8px; }
.coin { display: flex; gap: 10px; align-items: center; }
.logo { width: 36px; height: 36px; border-radius: 50%; background: #f7931a; color: #fff; display: inline-flex;
  align-items: center; justify-content: center; font-weight: 700; font-size: 20px; }
.name { font-weight: 600; font-size: 16px; }
.sym { color: #808a9d; font-weight: 500; margin-left: 6px; }
.src { color: #808a9d; font-size: 12px; }
.price { display: flex; align-items: baseline; gap: 10px; }
.p { font-size: 28px; font-weight: 700; }
.chg { font-size: 14px; font-weight: 600; }
.cmc-tabs { display: flex; gap: 6px; margin: 12px 0 6px; }
.cmc-tabs button { border: 0; background: #f1f4f9; color: #55607a; border-radius: 6px; padding: 4px 10px;
  font-size: 12px; cursor: pointer; }
.cmc-tabs button.on { background: #3861fb; color: #fff; }
.cmc-stage { position: relative; width: 100%; }
.cmc-stage canvas { display: block; width: 100%; height: 340px; }
.cmc-foot { display: flex; justify-content: space-between; color: #808a9d; font-size: 12px; margin-top: 8px;
  flex-wrap: wrap; gap: 6px; }
.tip { position: absolute; pointer-events: none; background: #171717; color: #fff; padding: 4px 8px;
  border-radius: 6px; font-size: 12px; line-height: 1.4; }
"""

    class LiveChart(anywidget.AnyWidget):
        _esm = CHART_ESM
        _css = CHART_CSS
        history = traitlets.List([]).tag(sync=True)   # [[unix_s, price], ...] (warm-up + live)
        tick = traitlets.Dict({}).tag(sync=True)      # latest second: {t, price, forecast, stats}

    chart = LiveChart()
    mo.md("### Live chart (one widget, updated in place)")
    return LiveChart, chart


@app.cell
def _(ROOT, bundle_path, mo):
    # ---- load the trained model bundle -------------------------------------------------
    import torch

    from btcpred.engine import LiveEngine

    _dev_pred = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        engine = LiveEngine(bundle_path.value, device="cpu", predictor_device=_dev_pred, quote_only=True)
        _info = mo.md(
            f"Loaded **{engine.arch}** predictor (preset `{engine.meta['preset']}`, step {engine.meta['step']}) "
            f"| Market Analyser on **CPU** | Predictor on **{_dev_pred.upper()}**"
            + (f" ({torch.cuda.get_device_name(0)})" if _dev_pred == "cuda" else "")
        )
    except Exception as e:  # bundle missing -> explain instead of crashing
        engine = None
        _info = mo.md(
            "**No model bundle found.** Put the output of `scripts/select_best.py` (folder `models/best`) "
            f"in the bundle path above.\n\n`{type(e).__name__}: {e}`"
        )
    _info
    return (engine,)


@app.cell
def _(json, np, pd, time, urllib):
    # ---- data sources ------------------------------------------------------------------
    BIQUOTE_URL = "https://biquote.io/api/BTCUSD"
    BINANCE_KLINES = "https://data-api.binance.vision/api/v3/klines"

    def fetch_biquote(timeout=3.0):
        with urllib.request.urlopen(BIQUOTE_URL, timeout=timeout) as r:
            d = json.loads(r.read().decode())
        if d.get("stale") or not d.get("mid"):
            return None
        return float(d["mid"])

    def binance_1s(seconds: int):
        """Last `seconds` of BTCUSDT 1-second klines (REST, 1000 per call), regularised to 1 row/s."""
        end_ms = (int(time.time()) - 1) * 1000  # only completed seconds
        cur = end_ms - seconds * 1000
        rows = []
        while cur < end_ms:
            url = f"{BINANCE_KLINES}?symbol=BTCUSDT&interval=1s&startTime={cur}&endTime={end_ms}&limit=1000"
            with urllib.request.urlopen(url, timeout=20) as r:
                chunk = json.loads(r.read().decode())
            if not chunk:
                break
            rows.extend(chunk)
            cur = chunk[-1][0] + 1000
        df = pd.DataFrame(rows).iloc[:, :10]
        df.columns = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume",
                      "trades", "taker_buy_volume"]
        out = pd.DataFrame({
            "ts": (df["open_time"].astype("int64") // 1000).to_numpy(),
            "open": df["open"].astype(float).to_numpy(), "high": df["high"].astype(float).to_numpy(),
            "low": df["low"].astype(float).to_numpy(), "close": df["close"].astype(float).to_numpy(),
            "volume": df["volume"].astype(float).to_numpy(),
            "taker_buy_volume": df["taker_buy_volume"].astype(float).to_numpy(),
            "trades": df["trades"].astype(float).to_numpy(),
        }).drop_duplicates("ts").sort_values("ts")
        grid = np.arange(out["ts"].iloc[0], out["ts"].iloc[-1] + 1)
        out = out.set_index("ts").reindex(grid)
        missing = out["close"].isna().to_numpy()
        out["close"] = out["close"].ffill()
        for c in ["open", "high", "low"]:
            out[c] = out[c].where(~missing, out["close"])
        for c in ["volume", "taker_buy_volume", "trades"]:
            out[c] = out[c].fillna(0)
        out["missing"] = missing.astype(float)
        return out.reset_index().rename(columns={"index": "ts"})

    return binance_1s, fetch_biquote


@app.cell
def _(binance_1s, chart, engine, fetch_biquote, np, threading, time):
    # ---- the live runner: one bar per second, one forecast per second, one chart update ----
    # Created once. Starting/stopping only flips a flag, so no second thread is ever spawned.
    from btcpred.config import CONTEXT_FOR_FEATURES, HORIZON, LOOKBACK
    from btcpred.models import ema_stabilise

    class LiveRunner:
        def __init__(self):
            self.thread = None
            self.stop_evt = threading.Event()
            self.basis = 0.0

        def alive(self):
            return self.thread is not None and self.thread.is_alive()

        def start(self):
            if engine is None or self.alive():
                self.stop_evt.clear()
                return
            self.stop_evt.clear()
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()

        def stop(self):
            self.stop_evt.set()

        def _publish(self, s, price, stab, status):
            times = s + np.arange(1, HORIZON + 1)
            fc_pts = [[int(t), float(p)] for t, p in zip(times[::5], stab[::5])]
            end_p = float(stab[-1])
            chart.tick = {
                "t": int(s), "price": float(price), "forecast": fc_pts,
                "stats": {"fc_end": end_p, "fc_change_pct": (end_p / price - 1) * 100,
                          "status": status, "basis_usd": round(self.basis, 2)},
            }

        def _run(self):
            try:
                chart.tick = {}
                bars = binance_1s(LOOKBACK + CONTEXT_FOR_FEATURES + 120)   # Binance 1 s warm-up
                engine.bootstrap(bars)
                chart.history = [[int(t), float(c)] for t, c in zip(bars["ts"], bars["close"])]
                px = fetch_biquote()
                # biquote (MT5 quotes) and Binance differ by a small basis: align to Binance level
                self.basis = (float(bars["close"].iloc[-1]) - px) if px is not None else 0.0

                prev = None
                cur_s = int(bars["ts"].iloc[-1]) + 1      # continue right after the warm-up
                bucket, last_px = [], float(bars["close"].iloc[-1])
                while not self.stop_evt.is_set():
                    now_s = int(time.time())
                    while cur_s < now_s and not self.stop_evt.is_set():   # close finished seconds
                        if bucket:
                            o, h, lo, c, miss = bucket[0], max(bucket), min(bucket), bucket[-1], 0
                        else:
                            o = h = lo = c = last_px
                            miss = 1                                      # no tick that second
                        engine.push_bar(ts=cur_s, open_=o, high=h, low=lo, close=c, missing=miss)
                        fc = engine.forecast()
                        stab = ema_stabilise(prev, fc["prices"], alpha=0.3)
                        prev = stab
                        status = (f"live | bar {'ok' if miss == 0 else 'held (no tick)'} | "
                                  f"basis {self.basis:+.2f} USD | {engine.arch}")
                        self._publish(cur_s, c, stab, status)
                        last_px = c
                        bucket = []
                        cur_s += 1
                    try:
                        px = fetch_biquote(timeout=1.5)
                        if px is not None:
                            bucket.append(px + self.basis)
                    except Exception:
                        pass
                    time.sleep(max(0.05, (cur_s + 1) - time.time()))
            except Exception as exc:  # report errors on the chart's status line
                chart.tick = {"t": int(time.time()), "price": 0.0, "forecast": [],
                              "stats": {"status": f"error: {type(exc).__name__}: {exc}"}}

    runner = LiveRunner()
    return (runner,)


@app.cell
def _(runner, start_switch):
    # ---- switch on/off (this cell re-runs only when the switch changes) ----
    if start_switch.value:
        runner.start()
    else:
        runner.stop()
    return


if __name__ == "__main__":
    app.run()
