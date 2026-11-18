#!/usr/bin/env python3
"""方向 #3：资金费率特征生产（Kaggle 侧取数，本地不跑）。

从币安 USD-M futures REST API 取历史 fundingRate（8h 一次，每次 ≤1000 条，
分页回溯到 2020），对齐到 1m 时间轴（close_time 语义：该分钟结束时已知的最新费率），
生成因果特征并发布为 parquet。

输出：/kaggle/working/funding_{SYMBOL}.parquet
列：close_time, fr              （该时刻最近一次已公布费率；未公布为 null）
     fr_z60/fr_z240             （费率在滚动窗口内的 z-score）
     fr_change                  （相对上一次费率的变化）
     fr_sign_streak             （连续同号次数）
     fr_next_in                 （距下次结算的分钟数，因果）

注意：fundingRate 的 fundingTime 是结算时刻；t 时刻只能用 fundingTime <= t 的费率。
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import requests

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

SYMBOL = os.environ.get("SYMBOL", "BTCUSDT")
API = "https://fapi.binance.com/fapi/v1/fundingRate"
START = int(os.environ.get("START_MS", "1577836800000"))  # 2020-01-01 UTC
END = int(os.environ.get("END_MS", str(int(time.time() * 1000))))
OUT = os.environ.get("OUT", f"/kaggle/working/funding_{SYMBOL}.parquet")


def log(msg: str) -> None:
    print(f"[{time.time():.0f}] {msg}", flush=True)


def fetch() -> pl.DataFrame:
    rows = []
    end = END
    while True:
        r = requests.get(API, params={"symbol": SYMBOL, "startTime": START,
                                      "endTime": end, "limit": 1000}, timeout=60)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        rows.extend((int(x["fundingTime"]), float(x["fundingRate"])) for x in batch)
        log(f"  batch {len(batch)} 条，最新 {batch[-1]['fundingTime']}")
        if len(batch) < 1000:
            break
        end = int(batch[0]["fundingTime"]) - 1  # 向前翻页
        time.sleep(0.3)
    if not rows:
        raise SystemExit("无 fundingRate 数据")
    df = (pl.DataFrame(rows, schema=["funding_time", "fr"], orient="row")
          .unique(subset="funding_time").sort("funding_time"))
    log(f"共 {df.height:,} 条费率记录 {df['funding_time'].min()}~{df['funding_time'].max()}")
    return df


def build_features(fr: pl.DataFrame) -> pl.DataFrame:
    # 以 1m 网格前向填充：close_time=t 用 fundingTime<=t 的最近费率（因果）
    lo = (fr["funding_time"].min() // 60_000) * 60_000
    hi = (END // 60_000) * 60_000
    grid = pl.DataFrame({"close_time": np.arange(lo, hi + 1, 60_000, dtype=np.int64)})
    out = grid.join_asof(fr, left_on="close_time", right_on="funding_time", strategy="backward")
    # 费率 8h 更新一次，1m 网格上用 asof 前向填充；所有特征只用当前及过去。
    # 在费率序列（而非 1m 网格）上算统计量，再回填，避免 1m 滚动把同值重复 480 次。
    fr2 = fr.with_columns([
        pl.col("fr").rolling_mean(21, min_samples=3).alias("fr_ma21"),   # ~7 天
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
    cols = ["funding_time", "fr", "fr_ma21", "fr_std21", "fr_z21", "fr_change", "fr_sign_streak"]
    out = grid.join_asof(fr2.select(cols), left_on="close_time",
                         right_on="funding_time", strategy="backward")
    return out


def main() -> None:
    fr = fetch()
    out = build_features(fr)
    out.write_parquet(OUT)
    log(f"写出 {OUT} {out.height:,} 行 | 列={out.columns}")


if __name__ == "__main__":
    main()
