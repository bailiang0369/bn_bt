#!/usr/bin/env python3
"""量化「OKX 400 档覆盖范围 vs 币安 2000 档」的不对称程度。

对同一天、同一分钟，分别算：
  1. OKX / 币安各自档位相对 mid 的最大 bps 距离（覆盖范围）
  2. OKX 覆盖范围内，币安有多少档 / 多少量（被增强的部分）
  3. OKX 覆盖范围外，币安有多少档 / 多少量（失真的部分）

用途：决定按价位合并方案是否可行，以及是否需要裁剪到共同范围。
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from morph import ob_kline as ok

DAY0 = int(dt.datetime(2025, 10, 1, tzinfo=dt.timezone.utc).timestamp() * 1000)
DAY1 = DAY0 + 86_400_000
log = lambda m: print(m, flush=True)


def main() -> None:
    bn = pl.read_parquet(ok.fetch("data-merged", "binance_futures_minute_band_2025-10.parquet"))
    bn = bn.filter((pl.col("bucket") >= DAY0) & (pl.col("bucket") < DAY1))
    okx = pl.read_parquet(sorted(Path("/kaggle/input").rglob("*minute_band_2025-10.parquet"))[0]
                          ).filter((pl.col("bucket") >= DAY0) & (pl.col("bucket") < DAY1))
    log(f"binance={bn.height:,} rows, OKX={okx.height:,} rows")

    # 相对各自 mid 的 bps 距离
    bn = bn.with_columns(((pl.col("px") - pl.col("mid")).abs() / pl.col("mid") * 1e4).alias("d_bps"))
    okx = okx.with_columns(((pl.col("px") - pl.col("mid")).abs() / pl.col("mid") * 1e4).alias("d_bps"))

    # 每分钟的最大覆盖 bps
    bn_max = bn.group_by("bucket").agg(pl.col("d_bps").max().alias("bn_max_bps"),
                                       pl.col("q").sum().alias("bn_q"),
                                       pl.col("d_bps").len().alias("bn_levels"))
    okx_max = okx.group_by("bucket").agg(pl.col("d_bps").max().alias("okx_max_bps"))
    j = bn_max.join(okx_max, on="bucket", how="inner")
    log("\n=== 每分钟最大覆盖 bps（相对各自 mid）===")
    for c in ("bn_max_bps", "okx_max_bps"):
        s = j[c]
        log(f"  {c}: median={s.median():.2f} p95={s.quantile(.95):.2f} "
            f"max={s.max():.2f} min={s.min():.2f}")
    log(f"  OKX覆盖/币安覆盖 = {(j['okx_max_bps']/j['bn_max_bps']).median():.3f} (median)")

    # 关键：OKX 覆盖范围内（用 OKX 每分钟 max_bps 做界），币安有多少档/多少量落在外面
    cov = okx_max.select(["bucket", "okx_max_bps"])
    b2 = bn.join(cov, on="bucket", how="inner")
    inside = b2.filter(pl.col("d_bps") <= pl.col("okx_max_bps"))
    outside = b2.filter(pl.col("d_bps") > pl.col("okx_max_bps"))
    in_q, out_q = inside["q"].sum(), outside["q"].sum()
    in_n, out_n = inside.height, outside.height
    log("\n=== 币安档位在 OKX 覆盖范围 内/外 的分布 ===")
    log(f"  档数: 内={in_n:,} ({in_n/(in_n+out_n)*100:.1f}%) "
        f"外={out_n:,} ({out_n/(in_n+out_n)*100:.1f}%)")
    log(f"  挂单量: 内={in_q:,.0f} ({in_q/(in_q+out_q)*100:.1f}%) "
        f"外={out_q:,.0f} ({out_q/(in_q+out_q)*100:.1f}%)")
    log(f"  → 合并只增强 {in_q/(in_q+out_q)*100:.1f}% 的币安深度，"
        f"剩余 {out_q/(in_q+out_q)*100:.1f}% 无 OKX 对应量，相对特征会失真")
    log("DONE")


if __name__ == "__main__":
    main()
