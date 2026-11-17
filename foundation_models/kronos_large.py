#!/usr/bin/env python3
"""Kronos 大样本验证：校准段定阈值 + test 评估；支持 1m / 5m 基础 bar。

与 kronos_probe.py 的区别（修正上一次的两个方法论问题）：
  1. N_ANCHORS 大幅提升（默认 3000），top10% 不再只有 30 样本；
  2. **阈值在 cal 段定，test 段只评估**——杜绝"test 上事后切 topK"的前视；
  3. BAR 参数：1m 或 5m 真 K 线重采样，验证"1m 随机扰动多、5m 更干净"的假设。

口径：
  - 数据：vision-base BTCUSDT futures(um)，OHLCV；5m 时 OHLCV 按桶真聚合。
  - 目标：close[t+h] > close[t]，h 以 bar 为单位；赔率 ODDS=0.8 → BE 55.56%。
  - 无前视：输入 t 及之前 LOOKBACK 根；cal/test 严格按时间先后切分。
  - 选信号：|pred_ret| >= 阈值（阈值 = cal 段 |pred_ret| 的 (1-TOP_FRAC) 分位），
    方向由 sign(pred_ret) 决定；test 用 cal 阈值固定不变。
  - 判据：test 胜率 vs BE、test topK 胜率 vs always-up、AUC、按天 tpd。

env
---
  BAR=1|5  N_ANCHORS=3000  PRED_LEN=3  HORIZONS=3  LOOKBACK=512
  CAL_FRAC=0.6  TOP_FRAC=0.1  ODDS=0.8  YEARS=2020,...,2026
  MODEL=NeoQuasar/Kronos-small  OUT=/kaggle/working/kronos_large_b{BAR}.parquet
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
PRED_LEN = int(os.environ.get("PRED_LEN", "3"))
HORIZONS = [int(x) for x in os.environ.get("HORIZONS", "3").split(",")]
LOOKBACK = int(os.environ.get("LOOKBACK", "512"))
CAL_FRAC = float(os.environ.get("CAL_FRAC", "0.6"))
TOP_FRAC = float(os.environ.get("TOP_FRAC", "0.1"))
ODDS = float(os.environ.get("ODDS", "0.8"))
YEARS = os.environ.get("YEARS", "2020,2021,2022,2023,2024,2025,2026").split(",")
MODEL = os.environ.get("MODEL", "NeoQuasar/Kronos-small")
TOKENIZER = os.environ.get("TOKENIZER", "NeoQuasar/Kronos-Tokenizer-base")
OUT = os.environ.get("OUT", f"/kaggle/working/kronos_large_b{BAR}.parquet")
REPO, TAG = "bailiang0369/bn_hisdata", "vision-base"
BE = 1.0 / (1.0 + ODDS) * 100


def log(msg: str) -> None:
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def setup() -> Path:
    dst = Path("/tmp/Kronos")
    if not (dst / "model.py").exists():
        subprocess.run(["git", "clone", "--depth=1", "https://github.com/shiyu-coder/Kronos.git",
                        str(dst)], check=True)
    import importlib.util
    for mod in ("einops", "safetensors", "huggingface_hub", "tqdm"):
        if importlib.util.find_spec(mod) is None:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", mod], check=True)
    sys.path.insert(0, str(dst))
    return dst


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
    return pl.read_parquet(p)


def load_data() -> pd.DataFrame:
    frames = [fetch_year(y).select("close_time", "open", "high", "low", "close",
                                   "um_buy_vol", "um_sell_vol") for y in YEARS]
    df = pl.concat(frames).sort("close_time").unique(subset=["close_time"], keep="last")
    df = df.with_columns((pl.col("um_buy_vol") + pl.col("um_sell_vol")).alias("volume"))
    if BAR > 1:
        ms = BAR * 60_000
        df = (df.with_columns((pl.col("close_time") // ms).alias("b"))
                .group_by("b").agg(
                    pl.col("close_time").last(),
                    pl.col("open").first(), pl.col("high").max(),
                    pl.col("low").min(), pl.col("close").last(),
                    pl.col("volume").sum())
                .sort("close_time"))
    out = df.select(["open", "high", "low", "close", "volume", "close_time"]).to_pandas()
    out["time"] = pd.to_datetime(out["close_time"], unit="ms", utc=True)
    log(f"data BAR={BAR}m rows={len(out):,} {out['time'].iloc[0]} ~ {out['time'].iloc[-1]}")
    return out


def main() -> None:
    setup()
    import torch
    from model import Kronos, KronosTokenizer, KronosPredictor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"device={device}")
    tokenizer = KronosTokenizer.from_pretrained(TOKENIZER)
    model = Kronos.from_pretrained(MODEL).to(device)
    try:
        predictor = KronosPredictor(model, tokenizer, max_context=LOOKBACK, device=device)
    except TypeError:
        predictor = KronosPredictor(model, tokenizer, max_context=LOOKBACK)
    log(f"model loaded {sum(p.numel() for p in model.parameters()):,} params")

    df = load_data()
    c = df["close"].to_numpy(dtype=float)
    n = len(c)
    lo, hi = LOOKBACK + 1, n - PRED_LEN - 2
    anchors = np.linspace(lo, hi, N_ANCHORS).astype(int)
    cal_cut = int(len(anchors) * CAL_FRAC)
    log(f"anchors={len(anchors)} cal={cal_cut} test={len(anchors)-cal_cut} "
        f"horizons={HORIZONS} top_frac={TOP_FRAC}")

    rows = []
    for i, t in enumerate(anchors):
        x_df = df.iloc[t - LOOKBACK:t][["open", "high", "low", "close", "volume"]].reset_index(drop=True)
        x_ts = pd.Series(df.iloc[t - LOOKBACK:t]["time"].to_numpy())
        y_ts = pd.Series(pd.to_datetime(
            df["close_time"].iloc[t] + np.arange(1, PRED_LEN + 1) * BAR * 60_000,
            unit="ms", utc=True))
        try:
            pred = predictor.predict(df=x_df, x_timestamp=x_ts, y_timestamp=y_ts,
                                     pred_len=PRED_LEN, T=1.0, top_p=0.9, sample_count=1)
        except Exception as e:  # noqa: BLE001
            if i < 3:
                log(f"[pred] anchor {t}: {type(e).__name__} {str(e)[:100]}")
            continue
        pc = pred["close"].to_numpy(dtype=float).reshape(-1)
        last = float(x_df["close"].iloc[-1])
        rec = {"t": int(t), "sec": str(df["time"].iloc[t]),
               "is_test": int(i >= cal_cut), "ret_prev": float(last / c[t - 1] - 1)}
        for h in HORIZONS:
            if h - 1 < len(pc):
                rec[f"pred_ret_{h}"] = float(pc[h - 1] / last - 1)
                rec[f"true_up_{h}"] = int(c[t + h] > c[t])
        rows.append(rec)
        if (i + 1) % 250 == 0:
            log(f"pred {i+1}/{len(anchors)}")

    res = pd.DataFrame(rows)
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    res.to_parquet(OUT, index=False)
    log(f"saved {OUT}: {len(res)} rows")

    cal = res[res["is_test"] == 0]
    test = res[res["is_test"] == 1]
    log(f"\n=== Kronos large BAR={BAR}m ODDS={ODDS} BE={BE:.2f}% ===")
    for h in HORIZONS:
        s, y = f"pred_ret_{h}", f"true_up_{h}"
        cv, tv = cal[[s, y]].dropna(), test[[s, y]].dropna()
        if len(cv) < 50 or len(tv) < 50:
            log(f"h={h} 样本不足 cal={len(cv)} test={len(tv)}")
            continue
        # 阈值只在 cal 定
        thr = float(np.quantile(np.abs(cv[s].to_numpy()), 1 - TOP_FRAC))
        sel = tv[np.abs(tv[s]) >= thr].copy()
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(tv[y], tv[s])
        win_all = float(((tv[s] > 0) == (tv[y] == 1)).mean()) * 100
        if len(sel):
            win_top = float(((sel[s] > 0) == (sel[y] == 1)).mean()) * 100
            up_top = float(sel[y].mean()) * 100
            ev = ODDS * win_top / 100 - (1 - win_top / 100)
            # 正态近似 95% CI
            se = np.sqrt(win_top / 100 * (1 - win_top / 100) / len(sel))
            ci = f"[{max(0,(win_top-1.96*se*100)):.1f},{min(100,(win_top+1.96*se*100)):.1f}]"
        else:
            win_top = up_top = ev = float("nan")
            ci = "n/a"
        log(f"h={h:2d} thr={thr:.6f} cal_n={len(cv)} test_n={len(tv)} "
            f"test_AUC={auc:.4f} win@0={win_all:.2f}%")
        log(f"     test top{int(TOP_FRAC*100)}% n={len(sel)} win={win_top:.2f}% "
            f"CI95={ci} always_up={up_top:.2f}% EV={ev:+.4f} "
            f"{'PASS' if win_top > BE else 'FAIL'} vs BE")


if __name__ == "__main__":
    main()
