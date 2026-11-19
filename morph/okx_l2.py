#!/usr/bin/env python3
"""对 ob_features 输出做 reindex + 时间窗滚动（L2），gap-safe。

输入：ob_features.py 的产物（主键 close_time，含 ob_mid 等 L1 截面特征）。
输出：同一批行 + 滚动/差分特征。缺分钟（源端断流/单边有量）被 reindex 成
      `ob_mid=null` 的行并打 `ob_missing=1`，滚动窗口按**日历时间**滑过缺口，
      而不是按行偏移压缩缺口。

关键点
------
1. 先 reindex 到连续分钟网格（close_time 从 min 到 max，步长 60000），
   之后 `shift(n)` 恰好等于 n 分钟前、`rolling(period=...)` 按时间窗滑动。
2. 滚动窗口用 `rolling(index_column=ts, period=...)`：窗口 `[t-w, t]`，null 行
   被聚合算子跳过（例如 5m 窗口里缺了 1 分钟，std 就在剩下 4 分钟上算）。
3. 时序特征本身只用「t 及之前」的信息，无前视：ob bucket=t 是第 t 根 bar
   收盘时的盘口，用它预测 t+1 及以后才因果。别拿本表特征解释同一根 bar 的
   收益率——那是同一时刻，有内生性。

用法
----
    python features/ob_rolling.py --in ob_features_binance_futures_2026-09.parquet \
        --out ob_rolling_binance_futures_2026-09.parquet
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import polars as pl

T = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T:7.1f}s] {msg}", flush=True)


def _reindex(df: pl.DataFrame) -> pl.DataFrame:
    """补齐到连续分钟网格，缺分钟行 ob_* 全 null 并标记。"""
    lo, hi = int(df["close_time"].min()), int(df["close_time"].max())
    grid = pl.DataFrame({
        "close_time": pl.int_range(lo, hi + 1, 60_000, dtype=pl.Int64, eager=True)
    })
    out = grid.join(df.sort("close_time"), on="close_time", how="left")
    return out.with_columns(
        pl.col("ob_mid").is_null().cast(pl.Int8).alias("ob_missing")
    )


def _add_rolling(df: pl.DataFrame, label: str, period: str) -> pl.DataFrame:
    """加一套 (mean / std) 滚动特征，窗口按日历时间。

    `rolling(index_column=..., period=...)` 的输出与输入同长度、同行序，
    所以直接按列横向拼回，不 join。
    """
    agg = df.with_columns(
        pl.from_epoch(pl.col("close_time"), time_unit="ms").alias("_ts")
    ).rolling(index_column="_ts", period=period).agg(
        pl.col("ob_ret_1m_bps").std(ddof=1).alias(f"ob_rv_{label}_bps"),
        pl.col("ob_obi_total").mean().alias(f"ob_obi_mean_{label}"),
        pl.col("ob_obi_total").std(ddof=1).alias(f"ob_obi_std_{label}"),
        pl.col("ob_spread_bps").mean().alias(f"ob_spread_mean_{label}"),
        pl.col("ob_spread_bps").std(ddof=1).alias(f"ob_spread_std_{label}"),
        pl.col("_impact").mean().alias(f"ob_impact_mean_{label}"),
    ).drop("_ts")
    return pl.concat([df, agg], how="horizontal")


# 波动率分桶边界：桶边界 = 最近 7 天 |1m 收益| 中位数 × 这些倍率。
# 6 档（0..5）：<0.8 / 0.8~1.0 / 1.0~1.25 / 1.25~1.6 / 1.6~2.0 / >=2.0 倍。
VOL_MULTS = (0.8, 1.0, 1.25, 1.6, 2.0)
VOL_BASE_WIN = "7d"    # 基准尺度窗口
VOL_CUR_WIN = "60m"    # 当前尺度窗口


def _add_vol_bucket(df: pl.DataFrame) -> pl.DataFrame:
    """按「最近波动率中位数 × 倍率」切波动档，桶边界随波动水平整体平移。

    为什么不用固定 bps 阈值：周末低波动时 1~2bp 已算"大波动"，消息面/美股开盘
    前后 20bp 也可能只是常态。用相对尺度（当前 / 最近 7 天中位数）分档，档位在
    不同时段之间可比。

    尺度统一取 1 分钟收益绝对值的中位数（bps）：
      ob_vol_med_7d_bps   过去 7 天中位数，shift(1) 后才是严格因果的基准
      ob_vol_med_60m_bps  最近 60 分钟中位数（当前尺度）
      ob_vol_ratio        当前 / 基准
      ob_vol_bucket       0..5，ratio 落在 VOL_MULTS 的哪一段
      ob_ret_vol_norm     1m 收益 / 基准中位数，跨波动时段可比的收益幅度
    """
    src = df.with_columns(
        pl.col("ob_ret_1m_bps").abs().alias("_absret"),
        pl.from_epoch(pl.col("close_time"), time_unit="ms").alias("_ts"),
    ).select(["_ts", "_absret"])
    m7 = src.rolling(index_column="_ts", period=VOL_BASE_WIN).agg(
        pl.col("_absret").median().alias("_m7")).drop("_ts")
    m60 = src.rolling(index_column="_ts", period=VOL_CUR_WIN).agg(
        pl.col("_absret").median().alias("_m60")).drop("_ts")
    out = pl.concat([df, m7, m60], how="horizontal").with_columns(
        pl.col("_m7").shift(1).alias("ob_vol_med_7d_bps"),
        pl.col("_m60").alias("ob_vol_med_60m_bps"),
    )
    out = out.with_columns(
        (pl.col("ob_vol_med_60m_bps") / pl.col("ob_vol_med_7d_bps")).alias("ob_vol_ratio")
    )
    ratio = pl.col("ob_vol_ratio")
    bucket = pl.lit(len(VOL_MULTS), dtype=pl.Int8)
    # 从大到小套 when：最外层是最低阈值，判断顺序自然变成 0 -> n
    for i in range(len(VOL_MULTS) - 1, -1, -1):
        bucket = pl.when(ratio < VOL_MULTS[i]).then(pl.lit(i, dtype=pl.Int8)).otherwise(bucket)
    out = out.with_columns(
        bucket.alias("ob_vol_bucket"),
        (pl.col("ob_ret_1m_bps") / pl.col("ob_vol_med_7d_bps")).alias("ob_ret_vol_norm"),
    )
    return out.drop(["_m7", "_m60"])


def _sanitize(df: pl.DataFrame) -> pl.DataFrame:
    """把 inf / -inf / NaN 统一置 null，避免除零产生的非常数静默污染下游模型。"""
    cols = [c for c in df.columns if df[c].dtype in (pl.Float32, pl.Float64)]
    return df.with_columns([
        pl.when(pl.col(c).is_finite()).then(pl.col(c)).otherwise(None).alias(c)
        for c in cols
    ])


def build_l2(df: pl.DataFrame) -> pl.DataFrame:
    """在 reindex 后的 df 上算差分 + 滚动特征。"""
    df = df.with_columns(
        (pl.col("ob_mid").pct_change() * 1e4).alias("ob_ret_1m_bps"),
        ((pl.col("ob_impact_bid_bps_per_btc") + pl.col("ob_impact_ask_bps_per_btc"))
         / 2).alias("_impact"),
    )
    # 差分（reindex 后 shift(n) == n 分钟前）
    df = df.with_columns(
        (pl.col("ob_obi_total") - pl.col("ob_obi_total").shift(1)).alias("ob_dobi_1m"),
        (pl.col("ob_obi_total") - pl.col("ob_obi_total").shift(5)).alias("ob_dobi_5m"),
        (pl.col("ob_q_bid") - pl.col("ob_q_bid").shift(1)).alias("ob_ddepth_bid_1m"),
        (pl.col("ob_q_ask") - pl.col("ob_q_ask").shift(1)).alias("ob_ddepth_ask_1m"),
        (pl.col("ob_q_bid") - pl.col("ob_q_bid").shift(5)).alias("ob_ddepth_bid_5m"),
        (pl.col("ob_wmid_bps")
         - pl.col("ob_wmid_bps").shift(1)).alias("ob_dwmid_1m"),
        (pl.col("ob_spread_bps") - pl.col("ob_spread_bps").shift(1)).alias("ob_dspread_1m"),
    )
    # 时间窗只取短周期 {5m, 15m}：订单簿挂单只对短期有效（挂单随时可撤/新增），
    # 对齐预测周期 3/5/10/15 分钟，不用 60m 这类长窗。
    for label, period in (("5m", "5m"), ("15m", "15m")):
        df = _add_rolling(df, label, period)
    # z-score
    df = df.with_columns(
        ((pl.col("ob_obi_total") - pl.col("ob_obi_mean_5m"))
         / pl.col("ob_obi_std_5m")).alias("ob_obi_z_5m"),
        ((pl.col("ob_obi_total") - pl.col("ob_obi_mean_15m"))
         / pl.col("ob_obi_std_15m")).alias("ob_obi_z_15m"),
        ((pl.col("ob_spread_bps") - pl.col("ob_spread_mean_15m"))
         / pl.col("ob_spread_std_15m")).alias("ob_spread_z_15m"),
    )
    df = _add_vol_bucket(df)
    return _sanitize(df.drop(["_impact"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    log("reading L1 ...")
    df = pl.read_parquet(args.inp)
    log(f"L1 rows: {df.height:,}")
    df = _reindex(df)
    log(f"after reindex: {df.height:,} rows, missing={df['ob_missing'].sum():,}")

    df = build_l2(df)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(args.out)
    log(f"wrote {args.out}: {df.height:,} rows x {df.width} cols")

    with pl.Config(tbl_cols=10, tbl_rows=6):
        print(df.select(["close_time", "ob_missing", "ob_ret_1m_bps", "ob_rv_5m_bps",
                         "ob_obi_z_5m", "ob_dobi_1m", "ob_spread_z_15m"]).head(6))


if __name__ == "__main__":
    main()
