#!/usr/bin/env python3
"""方向 #3：资金费率特征生产（Kaggle 侧运行，本地不跑）。

数据源：Binance Vision S3（REST API fapi.binance.com 对 Kaggle IP 返回 451）。
路径：data/futures/um/monthly/fundingRate/{SYMBOL}/{SYMBOL}-fundingRate-YYYY-MM.zip
每月一个 zip，内含 CSV（列：calc_time, funding_time, funding_rate）。

对齐：1m 网格 asof 前向填充，close_time=t 只用 funding_time<=t 的已公布费率（因果）。
输出：/kaggle/working/funding_{SYMBOL}.parquet
"""
from __future__ import annotations

import io
import os
import re
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import polars as pl
import requests

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

SYMBOL = os.environ.get("SYMBOL", "BTCUSDT")
S3 = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision/"
PREFIX = f"data/futures/um/monthly/fundingRate/{SYMBOL}/"
OUT = os.environ.get("OUT", f"/kaggle/working/funding_{SYMBOL}.parquet")
YEARS = range(2020, 2027)


def log(msg: str) -> None:
    print(f"[{time.time():.0f}] {msg}", flush=True)


def list_keys() -> list[str]:
    keys, token = [], None
    while True:
        params = {"list-type": "2", "prefix": PREFIX}
        if token:
            params["continuation-token"] = token
        r = requests.get(S3, params=params, timeout=60)
        r.raise_for_status()
        keys += re.findall(r"<Key>([^<]+)</Key>", r.text)
        m = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", r.text)
        if not m:
            break
        token = m.group(1)
    return [k for k in keys if k.endswith(".zip")]


def fetch_one(key: str) -> pl.DataFrame | None:
    url = S3 + key
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        name = [n for n in z.namelist() if n.endswith(".csv")][0]
        raw = z.read(name)
    # Vision fundingRate CSV: calc_time, funding_time, funding_rate（有 header）
    try:
        df = pl.read_csv(raw, try_parse_dates=False)
    except Exception:
        df = pl.read_csv(raw, has_header=False, new_columns=["calc_time", "funding_time", "funding_rate"])
    cols = {c.lower(): c for c in df.columns}
    ft = cols.get("funding_time", cols.get("fundingtime"))
    fr = cols.get("funding_rate", cols.get("fundingrate"))
    if ft is None or fr is None:
        log(f"  跳过 {key}：列 {df.columns}")
        return None
    return df.select(pl.col(ft).cast(pl.Int64).alias("funding_time"),
                     pl.col(fr).cast(pl.Float64).alias("fr"))


def main() -> None:
    keys = [k for k in list_keys() if any(f"-{y}-" in k for y in YEARS)]
    log(f"找到 {len(keys)} 个月度文件")
    parts = []
    for i, k in enumerate(keys):
        d = fetch_one(k)
        if d is not None and d.height:
            parts.append(d)
        if (i + 1) % 12 == 0:
            log(f"  已处理 {i+1}/{len(keys)}")
    if not parts:
        raise SystemExit("无数据")
    fr = pl.concat(parts).unique(subset="funding_time").sort("funding_time")
    log(f"费率记录 {fr.height:,} 条 | fr 范围 [{fr['fr'].min():.6f}, {fr['fr'].max():.6f}]")

    # 在费率序列上算统计量（8h 频率），再回填到 1m 网格
    fr2 = fr.with_columns([
        pl.col("fr").rolling_mean(21, min_samples=3).alias("fr_ma21"),
        pl.col("fr").rolling_std(21, min_samples=3).alias("fr_std21"),
        ((pl.col("fr") - pl.col("fr").rolling_mean(21, min_samples=3))
         / (pl.col("fr").rolling_std(21, min_samples=3) + 1e-12)).alias("fr_z21"),
        (pl.col("fr") - pl.col("fr").shift(1)).alias("fr_change"),
    ])
    sign = fr2["fr"].sign().to_numpy()
    streak = np.zeros(len(sign), dtype=np.float32)
    cur = 0
    for i in range(1, len(sign)):
        cur = cur + 1 if sign[i] == sign[i - 1] and sign[i] != 0 else 0
        streak[i] = cur
    fr2 = fr2.with_columns(pl.Series("fr_sign_streak", streak))

    lo = (fr["funding_time"].min() // 60_000) * 60_000
    hi = (fr["funding_time"].max() // 60_000) * 60_000
    grid = pl.DataFrame({"close_time": np.arange(lo, hi + 1, 60_000, dtype=np.int64)})
    cols = ["funding_time", "fr", "fr_ma21", "fr_std21", "fr_z21", "fr_change", "fr_sign_streak"]
    out = grid.join_asof(fr2.select(cols), left_on="close_time",
                         right_on="funding_time", strategy="backward")
    out.write_parquet(OUT)
    log(f"写出 {OUT} {out.height:,} 行 | 列={out.columns}")


if __name__ == "__main__":
    main()
