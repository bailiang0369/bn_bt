#!/usr/bin/env python3
"""Kronos 零样本探针（Kaggle）：金融 K 线基础模型能否押对 BTCUSDT 1m 的短 h 方向？

口径（严格对齐项目）：
  - 数据：vision-base BTCUSDT 1m，**合约（um）** OHLC + volume = um_buy_vol+um_sell_vol。
    理由：我们的 target = close[t+h] > close[t] 用的是 um close（two_legs.add_target），
    现货是另一个标的；喂现货去预测合约方向是错配。
  - 目标：与 two_legs.add_target 完全相同；h∈{3,10}，无手续费、无止损止盈。
  - 无前视：模型只看 t 及之前的 512 根，预测 t+1..t+H；标签用 t+h 的 close。
  - 判据：test 胜率 vs 55.56%（赔率 0.8 平衡线）+ random-walk 基线 + 动量基线 + AUC。
  - 不做参数搜索：MODEL / LOOKBACK / T / top_p 固定；只报结果。

env
---
  YEARS=2024,2025,2026  N_ANCHORS=300  HORIZONS=3,10  PRED_LEN=10
  LOOKBACK=512  MODEL=NeoQuasar/Kronos-small  TOKENIZER=NeoQuasar/Kronos-Tokenizer-base
  SAMPLE_COUNT=1  T=1.0  TOP_P=0.9  ODDS=0.8  SEED=42
  OUT=/kaggle/working/kronos_probe.parquet
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
YEARS = os.environ.get("YEARS", "2024,2025,2026").split(",")
N_ANCHORS = int(os.environ.get("N_ANCHORS", "300"))
HORIZONS = [int(x) for x in os.environ.get("HORIZONS", "3,10").split(",")]
PRED_LEN = int(os.environ.get("PRED_LEN", "10"))
LOOKBACK = int(os.environ.get("LOOKBACK", "512"))
MODEL = os.environ.get("MODEL", "NeoQuasar/Kronos-small")
TOKENIZER = os.environ.get("TOKENIZER", "NeoQuasar/Kronos-Tokenizer-base")
SAMPLE_COUNT = int(os.environ.get("SAMPLE_COUNT", "1"))
TEMP = float(os.environ.get("T", "1.0"))
TOP_P = float(os.environ.get("TOP_P", "0.9"))
ODDS = float(os.environ.get("ODDS", "0.8"))
SEED = int(os.environ.get("SEED", "42"))
OUT = os.environ.get("OUT", "/kaggle/working/kronos_probe.parquet")
REPO = "bailiang0369/bn_hisdata"
TAG = "vision-base"


def log(msg: str) -> None:
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def setup() -> Path:
    """clone Kronos 到 /tmp；只补缺失依赖（Kaggle 预装 pandas/tqdm，不动版本避免 pyproject 构建失败）。"""
    dst = Path("/tmp/Kronos")
    if not (dst / "model.py").exists():
        subprocess.run(["git", "clone", "--depth=1", "https://github.com/shiyu-coder/Kronos.git",
                        str(dst)], check=True)
    import importlib.util
    for mod, pkg in (("einops", "einops"), ("safetensors", "safetensors"),
                     ("huggingface_hub", "huggingface_hub"), ("tqdm", "tqdm")):
        if importlib.util.find_spec(mod) is None:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)
            log(f"pip installed {pkg}")
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
    log(f"data {year}: {p.stat().st_size / 1e6:.1f}MB")
    return pl.read_parquet(p)


def load_data() -> pd.DataFrame:
    frames = [fetch_year(y).select(
        "close_time", "open", "high", "low", "close", "um_buy_vol", "um_sell_vol")
        for y in YEARS]
    df = pl.concat(frames).sort("close_time").unique(subset=["close_time"], keep="last")
    df = df.with_columns((pl.col("um_buy_vol") + pl.col("um_sell_vol")).alias("volume"))
    out = df.select(["open", "high", "low", "close", "volume", "close_time"]).to_pandas()
    out["time"] = pd.to_datetime(out["close_time"], unit="ms", utc=True)
    log(f"futures data {len(out):,} rows {out['time'].iloc[0]} ~ {out['time'].iloc[-1]}")
    return out


def main() -> None:
    setup()
    from model import Kronos, KronosTokenizer, KronosPredictor  # noqa: E402

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"device={device} (cuda_available={torch.cuda.is_available()})")
    log(f"loading {MODEL} + {TOKENIZER} ...")
    tokenizer = KronosTokenizer.from_pretrained(TOKENIZER)
    model = Kronos.from_pretrained(MODEL).to(device)
    try:
        predictor = KronosPredictor(model, tokenizer, max_context=LOOKBACK, device=device)
    except TypeError:
        predictor = KronosPredictor(model, tokenizer, max_context=LOOKBACK)
    log(f"model loaded: {sum(p.numel() for p in model.parameters()):,} params on {device}")

    df = load_data()
    c = df["close"].to_numpy(dtype=float)
    n = len(c)
    rng = np.random.default_rng(SEED)
    # 固定锚点：等间距取 N_ANCHORS 个，留足 lookback 与 PRED_LEN；不做任何筛选/挑优
    lo, hi = LOOKBACK + 1, n - PRED_LEN - 2
    anchors = np.linspace(lo, hi, N_ANCHORS).astype(int)
    log(f"anchors {len(anchors)} from {lo} to {hi}; horizons {HORIZONS}")

    rows = []
    for i, t in enumerate(anchors):
        x_df = df.iloc[t - LOOKBACK:t][["open", "high", "low", "close", "volume"]].reset_index(drop=True)
        x_ts = pd.Series(df.iloc[t - LOOKBACK:t]["time"].to_numpy())
        y_ts = pd.Series(pd.to_datetime(
            df["close_time"].iloc[t] + np.arange(1, PRED_LEN + 1) * 60_000,
            unit="ms", utc=True))
        try:
            pred = predictor.predict(df=x_df, x_timestamp=x_ts, y_timestamp=y_ts,
                                     pred_len=PRED_LEN, T=TEMP, top_p=TOP_P,
                                     sample_count=SAMPLE_COUNT)
        except Exception as e:  # noqa: BLE001
            log(f"[pred] anchor {t}: {type(e).__name__} {str(e)[:80]}")
            continue
        pc = pred["close"].to_numpy(dtype=float).reshape(-1)
        last = float(x_df["close"].iloc[-1])
        rec = {"t": int(t), "close_t": last,
               "ret_prev": float(last / c[t - 1] - 1)}  # 动量基线：上一根收益
        for h in HORIZONS:
            if h > PRED_LEN or h - 1 >= len(pc):
                continue
            rec[f"pred_ret_{h}"] = float(pc[h - 1] / last - 1)
            rec[f"true_up_{h}"] = bool(c[t + h] > c[t])
        rows.append(rec)
        if (i + 1) % 25 == 0:
            log(f"pred {i + 1}/{len(anchors)}")

    res = pd.DataFrame(rows)
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    res.to_parquet(OUT, index=False)
    log(f"saved {OUT}: {len(res)} rows")

    be = 1.0 / (1.0 + ODDS) * 100
    log(f"\n=== Kronos zero-shot ({MODEL}, futures um, ODDS={ODDS}, BE={be:.2f}%) ===")
    for h in HORIZONS:
        key, truth = f"pred_ret_{h}", f"true_up_{h}"
        if key not in res or truth not in res:
            continue
        d = res[[key, truth]].dropna()
        if len(d) < 30:
            log(f"h={h}: 样本不足 {len(d)}")
            continue
        s = d[key].to_numpy()
        y = d[truth].to_numpy().astype(int)
        pred = (s > 0).astype(int)
        win = float((pred == y).mean()) * 100
        yw = float(y.mean()) * 100                       # 基线：永远押涨
        mw = float(((d["ret_prev"].to_numpy() > 0).astype(int) == y).mean()) * 100
        # AUC（用预测收益做分数）
        order = np.argsort(s)
        ranks = np.empty(len(s), dtype=float)
        ranks[order] = np.arange(1, len(s) + 1)
        n1, n0 = y.sum(), len(y) - y.sum()
        auc = float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")
        ev = ODDS * win / 100 - (1 - win / 100)
        log(f"h={h:2d}  n={len(d):4d}  Kronos胜率={win:5.2f}%  always-up={yw:5.2f}%  "
            f"momentum={mw:5.2f}%  AUC={auc:.4f}  EV/bet={ev:+.4f}  "
            f"{'PASS' if win > be else 'FAIL'} vs BE")


if __name__ == "__main__":
    main()
