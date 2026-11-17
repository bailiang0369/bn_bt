#!/usr/bin/env python3
"""Chronos-Bolt 零样本探针（Kaggle/魔搭 GPU）：通用概率基础模型能否押对短 h 方向。

与 TimesFM/Kronos 探针完全同口径：futures(um) log(close)，cal 段定阈值、test 评估，
二元目标 close[t+h] > close[t]，赔率 0.8（BE 55.56%）。

用 amazon/chronos-bolt-mini（非自回归，比原 Chronos-T5 快 250×）；输出分位数，
取 0.5 分位（中位数）作为方向判断：pred = median_forecast[:, h-1] > last_log_close。

env
---
  BAR=1|5  N_ANCHORS=3000  CONTEXT=512  HORIZONS=3,10  CAL_FRAC=0.6  TOP_FRAC=0.1
  CKPT=amazon/chronos-bolt-mini  DEVICE=cuda|cpu
  OUT=/kaggle/working/chronos_probe_b1.parquet
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import requests

T0 = time.time()
BAR = int(os.environ.get("BAR", "1"))
N_ANCHORS = int(os.environ.get("N_ANCHORS", "3000"))
CONTEXT = int(os.environ.get("CONTEXT", "512"))
HORIZONS = [int(x) for x in os.environ.get("HORIZONS", "3,10").split(",")]
CAL_FRAC = float(os.environ.get("CAL_FRAC", "0.6"))
TOP_FRAC = float(os.environ.get("TOP_FRAC", "0.1"))
ODDS = float(os.environ.get("ODDS", "0.8"))
CKPT = os.environ.get("CKPT", "amazon/chronos-bolt-mini")
DEVICE = os.environ.get("DEVICE", "cuda")
YEARS = os.environ.get("YEARS", "2020,2021,2022,2023,2024,2025,2026").split(",")
OUT = os.environ.get("OUT", f"/kaggle/working/chronos_probe_b{BAR}.parquet")
REPO, TAG = "bailiang0369/bn_hisdata", "vision-base"
BE = 1.0 / (1.0 + ODDS) * 100


def log(msg: str) -> None:
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def setup():
    import importlib.util
    if importlib.util.find_spec("chronos") is None or importlib.util.find_spec("torch") is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                        "chronos-forecasting", "torch"], check=True)
    from chronos import BaseChronosPipeline
    return BaseChronosPipeline


def fetch_year(year: str) -> pl.DataFrame:
    name = f"BTCUSDT_1m_klines_{year}.parquet"
    cache = Path(os.environ.get("DATA_DIR", "/tmp"))
    cache.mkdir(parents=True, exist_ok=True)
    p = cache / name
    bases = (f"https://github.com/{REPO}/releases/download/{TAG}",
             f"https://ghfast.top/https://github.com/{REPO}/releases/download/{TAG}")
    if not p.exists() or p.stat().st_size < 1e6:
        for base in bases:
            try:
                with requests.get(f"{base}/{name}", stream=True, timeout=120) as r:
                    r.raise_for_status()
                    with open(p.with_suffix(".part"), "wb") as f:
                        for chunk in r.iter_content(1 << 20):
                            if chunk:
                                f.write(chunk)
                p.with_suffix(".part").replace(p)
                log(f"[data] {name} {p.stat().st_size / 1e6:.1f}MB")
                break
            except Exception as e:  # noqa: BLE001
                log(f"[data] {year} via {base[:38]}: {type(e).__name__}")
        else:
            raise SystemExit(f"数据下载失败: {name}")
    return pl.read_parquet(p, columns=["close_time", "close"])


def load_data() -> pd.DataFrame:
    frames = [fetch_year(y) for y in YEARS]
    df = pl.concat(frames).sort("close_time").unique(subset=["close_time"], keep="last")
    if BAR > 1:
        ms = BAR * 60_000
        df = (df.with_columns((pl.col("close_time") // ms).alias("b"))
                .group_by("b").agg(pl.col("close_time").last(), pl.col("close").last())
                .sort("close_time"))
    out = df.to_pandas()
    out["time"] = pd.to_datetime(out["close_time"], unit="ms", utc=True)
    log(f"data BAR={BAR}m rows={len(out):,} {out['time'].iloc[0]} ~ {out['time'].iloc[-1]}")
    return out


def auc(y, s):
    y = np.asarray(y).astype(int)
    s = np.asarray(s)
    order = np.argsort(s)
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    n1 = int(y.sum())
    n0 = len(y) - n1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def main() -> None:
    import torch
    Cls = setup()
    log(f"loading {CKPT} on {DEVICE} ...")
    dtype = torch.bfloat16 if DEVICE == "cuda" else torch.float32
    pipeline = Cls.from_pretrained(CKPT, device_map=DEVICE, torch_dtype=dtype)
    log(f"pipeline loaded: {CKPT} dtype={dtype}")

    df = load_data()
    c = df["close"].to_numpy(dtype=float)
    lc = np.log(c)
    n = len(c)
    lo, hi = CONTEXT + 1, n - max(HORIZONS) - 2
    anchors = np.linspace(lo, hi, N_ANCHORS).astype(int)
    cal_cut = int(len(anchors) * CAL_FRAC)
    log(f"anchors={len(anchors)} cal={cal_cut} test={len(anchors)-cal_cut}")

    # Chronos-Bolt: list of 1d context -> forecast [num_series, num_quantiles, pred_len]
    contexts = [lc[t - CONTEXT:t] for t in anchors]
    log(f"forecast batch n={len(contexts)} ctx={CONTEXT} ...")
    fc = pipeline.predict(context=torch.tensor(np.asarray(contexts, dtype=np.float32)),
                          prediction_length=max(HORIZONS))
    fc = fc.detach().cpu().numpy() if hasattr(fc, "detach") else np.asarray(fc)
    log(f"forecast raw shape={fc.shape}")
    if fc.ndim == 3:
        q = fc.shape[1]
        med_idx = q // 2                      # 分位数近似等距，0.5 在中间
        fc = fc[:, med_idx, :]                # [n, pred_len] 中位数路径
    log(f"median forecast shape={fc.shape}")

    rows = []
    for i, t in enumerate(anchors):
        last = c[t - 1]
        rec = {"t": int(t), "sec": str(df["time"].iloc[t]),
               "is_test": int(i >= cal_cut), "ret_prev": float(last / c[t - 2] - 1)}
        for h in HORIZONS:
            if h - 1 < fc.shape[1]:
                rec[f"pred_ret_{h}"] = float(np.exp(fc[i, h - 1] - np.log(last)) - 1)
                rec[f"true_up_{h}"] = int(c[t + h] > c[t])
        rows.append(rec)
    res = pd.DataFrame(rows)
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    res.to_parquet(OUT, index=False)
    log(f"saved {OUT}: {len(res)} rows")

    cal, test = res[res.is_test == 0], res[res.is_test == 1]
    log(f"\n=== Chronos-Bolt BAR={BAR}m ODDS={ODDS} BE={BE:.2f}% ===")
    for h in HORIZONS:
        s, y = f"pred_ret_{h}", f"true_up_{h}"
        cv, tv = cal[[s, y]].dropna(), test[[s, y]].dropna()
        if len(cv) < 50 or len(tv) < 50:
            continue
        thr = float(np.quantile(np.abs(cv[s].to_numpy()), 1 - TOP_FRAC))
        sel = tv[np.abs(tv[s]) >= thr]
        a = auc(tv[y], tv[s])
        win_all = float(((tv[s] > 0) == (tv[y] == 1)).mean()) * 100
        if len(sel):
            w = float(((sel[s] > 0) == (sel[y] == 1)).mean())
            se = np.sqrt(w * (1 - w) / len(sel))
            ci = f"[{max(0,w-1.96*se)*100:.1f},{min(1,w+1.96*se)*100:.1f}]"
            ev = ODDS * w - (1 - w)
        else:
            w, ci, ev = float("nan"), "n/a", float("nan")
        log(f"h={h:2d} thr={thr:.6f} cal_n={len(cv)} test_n={len(tv)} "
            f"AUC={a:.4f} win@0={win_all:.2f}%")
        log(f"     test top{int(TOP_FRAC*100)}% n={len(sel)} win={w*100:.2f}% "
            f"CI95={ci} EV={ev:+.4f} {'PASS' if w*100>BE else 'FAIL'} vs BE")


if __name__ == "__main__":
    main()
