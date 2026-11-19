#!/usr/bin/env python3
"""最小可行性验证：把 OKX 挂单按价位合并进币安 band，看特征变化幅度。

判据（预登记）：核心特征（ob_obi_total / ob_q_bid / ob_q_ask / ob_vwap）的
平均相对变化 > 5% 才值得继续；< 1% 则"合并等于没加"，方向关闭。

合并规则（用户定义）：
  同一 (bucket, side, px) 的 q = 币安 q + OKX q
  OKX 只覆盖币安 2000 档中的 400 档，其余价位 q 不变
  两边 mid 差异 ~0.28bps，直接按绝对价格 px 对齐（tick 都是 0.1）
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from morph import ob_kline as ok
from morph.ob_features_okx import build_l1

DAY0 = int(dt.datetime(2025, 10, 1, tzinfo=dt.timezone.utc).timestamp() * 1000)
DAY1 = DAY0 + 86_400_000
log = lambda m: print(m, flush=True)


def main() -> None:
    # 币安 band（单日）
    a = ok._assets("data-merged")
    bn = pl.read_parquet(ok.fetch("data-merged", "binance_futures_minute_band_2025-10.parquet"))
    bn = bn.filter((pl.col("bucket") >= DAY0) & (pl.col("bucket") < DAY1))
    log(f"binance band: {bn.height:,} rows")

    # OKX band（dataset）
    okx_files = sorted(Path("/kaggle/input").rglob("*minute_band_2025-10.parquet"))
    okx = pl.read_parquet(okx_files[0]).filter(
        (pl.col("bucket") >= DAY0) & (pl.col("bucket") < DAY1))
    log(f"OKX band: {okx.height:,} rows")

    # 对齐统计：OKX 档位中有多少能在币安找到同 (bucket,side,px)
    j = okx.select(["bucket", "side", "px", "q"]).join(
        bn.select(["bucket", "side", "px", "q"]).rename({"q": "q_bn"}),
        on=["bucket", "side", "px"], how="left")
    matched = j.filter(pl.col("q_bn").is_not_null())
    log(f"OKX levels matched in binance: {matched.height:,}/{okx.height:,} "
        f"({matched.height/okx.height*100:.1f}%)")

    # 量级对比：匹配档位上两边 q 的中位数比
    both = matched.with_columns((pl.col("q") / pl.col("q_bn")).alias("ratio"))
    log(f"q_okx / q_bn at matched levels: median={both['ratio'].median():.3f}, "
        f"mean={both['ratio'].mean():.3f}, total OKX={matched['q'].sum():.0f} "
        f"vs binance={matched['q_bn'].sum():.0f}")

    # 合并：币安 band + OKX q（按同 bucket/side/px 加，OKX 独有价位忽略，因为量极小）
    merged = bn.join(
        okx.select(["bucket", "side", "px", "q"]).rename({"q": "q_okx"}),
        on=["bucket", "side", "px"], how="left").with_columns(
        (pl.col("q") + pl.col("q_okx").fill_null(0)).alias("q"))
    merged = merged.select(["bucket", "side", "px", "q", "mid"])
    log(f"merged band: {merged.height:,} rows")

    # 两边都算特征（bps 分箱，同口径）
    b_mid = pl.read_parquet(ok.fetch("data-merged",
                                     "binance_futures_minute_band_mid_2025-10.parquet"))
    b_mid = b_mid.filter((pl.col("bucket") >= DAY0) & (pl.col("bucket") < DAY1))
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p1, p2 = Path(td)/"b.parquet", Path(td)/"m.parquet"
        bn.write_parquet(p1); b_mid.write_parquet(p2)
        fb = build_l1(str(p1), str(p2), bin_mode="bps")
        merged.write_parquet(p1)
        fm = build_l1(str(p1), str(p2), bin_mode="bps")

    # 关键特征变化
    keys = ["ob_obi_total", "ob_q_bid", "ob_q_ask", "ob_spread_bps", "ob_obi_near"]
    log("\nfeature change (merged vs binance-only):")
    for c in keys:
        if c not in fb.columns:
            continue
        a_, b_ = fb[c].to_numpy(), fm[c].to_numpy()
        valid = np.isfinite(a_) & np.isfinite(b_) & (np.abs(a_) > 1e-9)
        rel = np.abs(b_[valid] - a_[valid]) / np.abs(a_[valid])
        log(f"  {c}: mean|Δ|/|x|={rel.mean()*100:.2f}% p95={np.percentile(rel,95)*100:.2f}% "
            f"max={rel.max()*100:.1f}%")
    log("DONE")


if __name__ == "__main__":
    import numpy as np
    main()
