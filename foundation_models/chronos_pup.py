#!/usr/bin/env python3
"""Chronos-Bolt 分位数方向概率 P(up) 校准分析。

核心想法：模型输出 9 个分位数 [0.1..0.9]。在 h 根处，数有多少分位数 > 当前 close，
得到 P(up) ∈ {0, 1/9, .., 1}。检验：
  1. P(up) 作为方向 score 的 AUC（对比只用中位数的旧结果）
  2. 校准曲线：P(up) 分档的实际上涨频率 vs 名义概率
  3. 阈值下单：P(up) >= p_thr 的实际胜率，看能否过 55.56% BE
  所有阈值/分档在 cal 段定，test 只评估（防止事后挑）。

env: BAR=1 N_ANCHORS=10000 HORIZONS=3,10,60 CAL_FRAC=0.6 DEVICE=cpu
     OUT=/kaggle/working/chronos_pup.parquet
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import requests

T0 = time.time()
BAR = int(os.environ.get("BAR", "1"))
N_ANCHORS = int(os.environ.get("N_ANCHORS", "10000"))
CONTEXT = int(os.environ.get("CONTEXT", "512"))
HORIZONS = [int(x) for x in os.environ.get("HORIZONS", "3,10,60").split(",")]
CAL_FRAC = float(os.environ.get("CAL_FRAC", "0.6"))
ODDS = float(os.environ.get("ODDS", "0.8"))
CKPT = os.environ.get("CKPT", "amazon/chronos-bolt-mini")
DEVICE = os.environ.get("DEVICE", "cpu")
YEARS = os.environ.get("YEARS", "2020,2021,2022,2023,2024,2025,2026").split(",")
OUT = os.environ.get("OUT", "/kaggle/working/chronos_pup.parquet")
REPO, TAG = "bailiang0369/bn_hisdata", "vision-base"
BE = 1.0 / (1.0 + ODDS) * 100


def log(m):
    print(f"[{time.time()-T0:6.1f}s] {m}", flush=True)


def auc(y, s):
    y = np.asarray(y).astype(int); s = np.asarray(s)
    order = np.argsort(s); ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1)
    n1 = int(y.sum()); n0 = len(y) - n1
    return float((ranks[y == 1].sum() - n1*(n1+1)/2)/(n1*n0)) if n1 and n0 else float("nan")


def fetch_year(year):
    name = f"BTCUSDT_1m_klines_{year}.parquet"
    cache = Path(os.environ.get("DATA_DIR", "/tmp")); cache.mkdir(parents=True, exist_ok=True)
    p = cache / name
    bases = (f"https://github.com/{REPO}/releases/download/{TAG}",
             f"https://ghfast.top/https://github.com/{REPO}/releases/download/{TAG}")
    if not p.exists() or p.stat().st_size < 1e6:
        for b in bases:
            try:
                with requests.get(f"{b}/{name}", stream=True, timeout=120) as r:
                    r.raise_for_status()
                    with open(p.with_suffix(".part"), "wb") as f:
                        for ch in r.iter_content(1 << 20):
                            if ch: f.write(ch)
                p.with_suffix(".part").replace(p); break
            except Exception as e:  # noqa: BLE001
                log(f"[data] {year}: {type(e).__name__}")
        else:
            raise SystemExit(f"下载失败 {name}")
    return pl.read_parquet(p, columns=["close_time", "close"])


def main():
    import subprocess
    import sys
    import importlib.util
    if importlib.util.find_spec("chronos") is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "chronos-forecasting"],
                       check=True)
        log("pip installed chronos-forecasting")
    import torch
    from chronos import BaseChronosPipeline
    log(f"loading {CKPT} on {DEVICE} ...")
    dtype = torch.bfloat16 if DEVICE == "cuda" else torch.float32
    pipe = BaseChronosPipeline.from_pretrained(CKPT, device_map=DEVICE, torch_dtype=dtype)

    frames = [fetch_year(y) for y in YEARS]
    df = pl.concat(frames).sort("close_time").unique(subset=["close_time"], keep="last")
    if BAR > 1:
        ms = BAR * 60_000
        df = (df.with_columns((pl.col("close_time")//ms).alias("b"))
                .group_by("b").agg(pl.col("close_time").last(), pl.col("close").last())
                .sort("close_time"))
    d = df.to_pandas()
    c = d["close"].to_numpy(float); lc = np.log(c); n = len(c)
    lo, hi = CONTEXT + 1, n - max(HORIZONS) - 2
    anchors = np.linspace(lo, hi, N_ANCHORS).astype(int)
    cal_cut = int(len(anchors) * CAL_FRAC)
    log(f"anchors={len(anchors)} cal={cal_cut} test={len(anchors)-cal_cut} horizons={HORIZONS}")

    contexts = [lc[t - CONTEXT:t] for t in anchors]
    fc = pipe.predict(torch.tensor(np.asarray(contexts, np.float32)),
                      prediction_length=max(HORIZONS))
    fc = fc.detach().cpu().numpy() if hasattr(fc, "detach") else np.asarray(fc)
    log(f"forecast shape={fc.shape}")   # [n, n_quantiles, pred_len]

    rows = []
    for i, t in enumerate(anchors):
        last_log = np.log(c[t - 1])
        rec = {"t": int(t), "is_test": int(i >= cal_cut)}
        for h in HORIZONS:
            if h - 1 >= fc.shape[2]:
                continue
            qs = fc[i, :, h - 1]                       # 9 个分位数（log 价）
            rec[f"pup_{h}"] = float((qs > last_log).mean())   # P(up)
            rec[f"pup_med_{h}"] = float(qs[len(qs)//2] > last_log)
            rec[f"true_up_{h}"] = int(c[t + h] > c[t])
        rows.append(rec)
    res = pd.DataFrame(rows)
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    res.to_parquet(OUT, index=False)
    log(f"saved {OUT}: {len(res)} rows")

    cal, test = res[res.is_test == 0], res[res.is_test == 1]
    log(f"\n=== Chronos P(up) BAR={BAR}m BE={BE:.2f}% ===")
    for h in HORIZONS:
        pu, y = f"pup_{h}", f"true_up_{h}"
        cv, tv = cal[[pu, y]].dropna(), test[[pu, y]].dropna()
        if len(tv) < 100:
            continue
        a = auc(tv[y], tv[pu])
        win_med = float(((tv[f"pup_med_{h}"] == 1) == (tv[y] == 1)).mean()) * 100
        log(f"\nh={h}  n_test={len(tv)}  AUC(Pup)={a:.4f}  win@median={win_med:.2f}%")
        # 校准曲线：P(up) 名义分档 vs 实际上涨频率
        log("  校准  P(up)档 : n / 实际上涨率")
        for lo_p, hi_p in ((0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)):
            m = (tv[pu] >= lo_p) & (tv[pu] < hi_p)
            if m.sum() >= 20:
                log(f"    [{lo_p:.1f},{min(hi_p,1):.1f}]  n={int(m.sum()):5d}  "
                    f"实际上涨={tv[y][m].mean()*100:5.2f}%  名义中心≈{(lo_p+min(hi_p,1))/2*100:.0f}%")
        # 阈值下单：阈值在 cal 段按分位定，test 评估
        log("  阈值下单 (cal 定分位 -> test 胜率)")
        for frac in (0.25, 0.10, 0.05):
            thr = float(np.quantile(cv[pu], 1 - frac))
            sel = tv[tv[pu] >= thr]
            if len(sel) < 20:
                continue
            # 高 P(up) 档：预测涨，看实际涨的比例；低 P(up) 档同理
            hi_sel = sel[sel[pu] >= 0.5]
            if len(hi_sel) < 20:
                continue
            w = float(hi_sel[y].mean())  # 预测涨的胜率
            se = np.sqrt(w*(1-w)/len(hi_sel))
            log(f"    top{int(frac*100):2d}% thr={thr:.3f} n_pred_up={len(hi_sel):4d} "
                f"胜率={w*100:.2f}% CI95=[{max(0,w-1.96*se)*100:.1f},{min(1,w+1.96*se)*100:.1f}] "
                f"{'PASS' if w*100>BE else 'FAIL'}")


if __name__ == "__main__":
    main()
