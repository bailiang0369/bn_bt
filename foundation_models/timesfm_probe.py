#!/usr/bin/env python3
"""TimesFM 2.5 零样本探针（Kaggle，GPU）：通用时序基础模型能否押对 BTCUSDT 短 h 方向？

与 Kronos 探针可比：同样 futures(um) close、同样二元目标、同样 cal 阈值/test 评估。
TimesFM 是单变量模型，喂 log(close) 序列（不吃 OHLCV / volume），一次前向批量出全部 horizon，
因此速度比自回归的 Kronos 快几十倍。

env
---
  BAR=1|5  N_ANCHORS=3000  CONTEXT=512  HORIZONS=3,10  CAL_FRAC=0.6  TOP_FRAC=0.1
  YEARS=2020,...,2026  ODDS=0.8  OUT=/kaggle/working/timesfm_probe_b{BAR}.parquet
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
YEARS = os.environ.get("YEARS", "2020,2021,2022,2023,2024,2025,2026").split(",")
OUT = os.environ.get("OUT", f"/kaggle/working/timesfm_probe_b{BAR}.parquet")
REPO, TAG = "bailiang0369/bn_hisdata", "vision-base"
BE = 1.0 / (1.0 + ODDS) * 100


def log(msg: str) -> None:
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def setup():
    # 2.5 checkpoint 对应 timesfm==2.0.2，API = TimesFM_2p5_200M_torch（不是 TimesFm）。
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--force-reinstall",
                    "timesfm[torch]==2.0.2"], check=True)
    from timesfm import TimesFM_2p5_200M_torch, ForecastConfig
    log("timesfm[torch]==2.0.2 (TimesFM_2p5_200M_torch) ready")
    return TimesFM_2p5_200M_torch, ForecastConfig


def fetch_year(year: str) -> pl.DataFrame:
    url = f"https://github.com/{REPO}/releases/download/{TAG}/BTCUSDT_1m_klines_{year}.parquet"
    p = Path(f"/tmp/BTCUSDT_1m_klines_{year}.parquet")
    if not p.exists() or p.stat().st_size < 1e6:
        for attempt in range(5):
            try:
                with requests.get(url, stream=True, timeout=120) as r:
                    r.raise_for_status()
                    with open(p.with_suffix(".part"), "wb") as f:
                        for chunk in r.iter_content(1 << 20):
                            if chunk:
                                f.write(chunk)
                p.with_suffix(".part").replace(p)
                break
            except Exception as e:  # noqa: BLE001
                log(f"[dl] {year} attempt {attempt}: {type(e).__name__}")
                time.sleep(2 ** attempt)
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


def main() -> None:
    Cls, ForecastConfig = setup()
    import torch
    torch.set_float32_matmul_precision("high")
    tfm = Cls.from_pretrained("google/timesfm-2.5-200m-pytorch",
                              torch_compile=False)
    cfg = ForecastConfig(max_context=CONTEXT, max_horizon=max(HORIZONS),
                         per_core_batch_size=64, infer_is_positive=False)
    tfm.compile(cfg)
    log(f"TimesFM 2.5 loaded, ctx={CONTEXT} horizon={max(HORIZONS)}")

    df = load_data()
    c = df["close"].to_numpy(dtype=float)
    n = len(c)
    lc = np.log(c)
    lo, hi = CONTEXT + 1, n - max(HORIZONS) - 2
    anchors = np.linspace(lo, hi, N_ANCHORS).astype(int)
    cal_cut = int(len(anchors) * CAL_FRAC)
    log(f"anchors={len(anchors)} cal={cal_cut} test={len(anchors)-cal_cut}")

    # 一次批量前向：list of 1d log(close) 序列，返回 (points, quantiles)
    inputs = [lc[t - CONTEXT:t] for t in anchors]
    log(f"forecast batch n={len(inputs)} ctx={CONTEXT} ...")
    fc, _q = tfm.forecast(max(HORIZONS), inputs)
    fc = np.asarray(fc)
    log(f"forecast done shape={fc.shape}")

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

    def auc(y, s):  # 手动 rank AUC，避免 numpy 降级后 sklearn 导入失败
        y = np.asarray(y).astype(int)
        s = np.asarray(s)
        order = np.argsort(s)
        ranks = np.empty(len(s), dtype=float)
        ranks[order] = np.arange(1, len(s) + 1)
        n1 = int(y.sum())
        n0 = len(y) - n1
        return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")

    log(f"\n=== TimesFM 2.5 BAR={BAR}m ODDS={ODDS} BE={BE:.2f}% ===")
    for h in HORIZONS:
        s, y = f"pred_ret_{h}", f"true_up_{h}"
        cv, tv = cal[[s, y]].dropna(), test[[s, y]].dropna()
        if len(cv) < 50 or len(tv) < 50:
            continue
        thr = float(np.quantile(np.abs(cv[s].to_numpy()), 1 - TOP_FRAC))
        sel = tv[np.abs(tv[s]) >= thr]
        auc_val = auc(tv[y], tv[s])
        win_all = float(((tv[s] > 0) == (tv[y] == 1)).mean()) * 100
        if len(sel):
            w = float(((sel[s] > 0) == (sel[y] == 1)).mean())
            se = np.sqrt(w * (1 - w) / len(sel))
            ci = f"[{max(0,w-1.96*se)*100:.1f},{min(1,w+1.96*se)*100:.1f}]"
            ev = ODDS * w - (1 - w)
        else:
            w, ci, ev = float("nan"), "n/a", float("nan")
        log(f"h={h:2d} thr={thr:.6f} cal_n={len(cv)} test_n={len(tv)} "
            f"AUC={auc_val:.4f} win@0={win_all:.2f}%")
        log(f"     test top{int(TOP_FRAC*100)}% n={len(sel)} win={w*100:.2f}% "
            f"CI95={ci} EV={ev:+.4f} {'PASS' if w*100>BE else 'FAIL'} vs BE")


if __name__ == "__main__":
    main()
