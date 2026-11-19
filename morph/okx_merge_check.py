#!/usr/bin/env python3
"""验证「按价格档位合并币安+OKX 挂单量」是否可行。

检查三件事：
1. 价格刻度：两边 px 的最小间隔是否一致（tick 对齐才能按 px 直接相加）
2. 档位重叠：同一分钟、同一侧，两边 px 的重叠率
3. mid 差异：同一分钟两边 mid 的差（bps）

用法（Kaggle，挂载 okx features dataset + 从 Release 拉币安 band）：
  python morph/okx_merge_check.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from morph import ob_kline as ok

DATE = "2025-10-01"
log = lambda m: print(m, flush=True)


def main() -> None:
    # 币安 futures band（从 Release 拉整月会很大，这里取 2025-10 全月但只分析一天）
    a = ok._assets("data-merged")
    name = "binance_futures_minute_band_2025-10.parquet"
    bn_path = ok.fetch("data-merged", name)
    bn = pl.read_parquet(bn_path).filter(pl.col("bucket") // 86_400_000 == int(
        __import__("datetime").datetime(2025, 10, 1, tzinfo=__import__("datetime").timezone.utc
                                        ).timestamp()))
    log(f"binance band day1: {bn.height:,} rows, px range "
        f"{bn['px'].min():.2f}~{bn['px'].max():.2f}")

    # OKX band（Kaggle dataset 里只有 features，没有 band；需从 okx-month kernel 产物拉）
    # 这里先看 dataset 里有没有 band 文件
    okx_files = sorted(Path("/kaggle/input").rglob("*minute_band*.parquet"))
    if not okx_files:
        log("no OKX band file in /kaggle/input; skip OKX side")
        # 只分析币安侧 tick
        for side in ("bid", "ask"):
            s = bn.filter(pl.col("side") == side)["px"].sort()
            d = s.diff().drop_nulls()
            log(f"  binance {side}: median tick={d.median():.4f}, "
                f"p10={d.quantile(.1):.4f}, p90={d.quantile(.9):.4f}")
        return

    okx = pl.read_parquet(okx_files[0]).filter(pl.col("bucket") // 86_400_000 == int(
        __import__("datetime").datetime(2025, 10, 1, tzinfo=__import__("datetime").timezone.utc
                                        ).timestamp()))
    log(f"OKX band day1: {okx.height:,} rows")

    for side in ("bid", "ask"):
        bs = bn.filter(pl.col("side") == side)["px"].sort()
        os_ = okx.filter(pl.col("side") == side)["px"].sort()
        bd = bs.diff().drop_nulls()
        od = os_.diff().drop_nulls()
        log(f"[{side}] tick: binance med={bd.median():.4f} | OKX med={od.median():.4f}")
        log(f"[{side}] levels/bucket: binance={bn.filter(pl.col('side')==side).height/1440:.0f} "
            f"| OKX={okx.filter(pl.col('side')==side).height/1440:.0f}")

    # mid 差异（用同一分钟）
    bmid = pl.read_parquet(ok.fetch("data-merged",
                                    "binance_futures_minute_band_mid_2025-10.parquet"))
    omid_files = [f for f in Path("/kaggle/input").rglob("*band_mid*.parquet")]
    if omid_files:
        omid = pl.read_parquet(omid_files[0])
        j = bmid.join(omid, on="bucket", suffix="_okx").select(
            "bucket", "mid", "mid_okx")
        j = j.with_columns(((pl.col("mid_okx") - pl.col("mid")) / pl.col("mid") * 1e4
                           ).alias("mid_diff_bps"))
        log(f"mid diff (OKX-Binance): mean={j['mid_diff_bps'].mean():.3f}bps "
            f"std={j['mid_diff_bps'].std():.3f}bps "
            f"p95={j['mid_diff_bps'].quantile(.95):.3f}bps")


if __name__ == "__main__":
    main()
