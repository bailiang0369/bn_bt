#!/usr/bin/env python3
"""从分钟级 band(+mid) 快照计算 L2 订单簿截面特征（L1）。

输入（`data-YYYY-MM` release 的日包，或 `data-merged` 的月度文件）
------------------------------------------------------------------
band : bucket(Int64) side(str) px(Float64) q(Float32) mid(Float64)
    每分钟 × 每档一行，仅保留 mid ±0.3% 内的档位。
mid  : bucket(Int64) best_bid best_ask mid spread（全部 Float64）

`--band` / `--mid` 支持通配，可直接读一个月的日包（`data-YYYY-MM/{date}.zip`
解压后的 30 个日 parquet）：日文件之间在边界重叠一分钟，`unique(bucket, side, px)`
会去掉重复；`rolling_std` 跨文件续算，等价于月度文件。

输出
----
主键 `close_time = bucket + 60_000`（Int64 毫秒），与 `vision-base` 的
`close_time` 同键同语义：

    bucket = t 的内容是「第 t 分钟内最后一条消息生效后的盘口」，而第 t 分钟内
    最后一条消息之后直到 t+60_000 不再有消息，所以该状态一直持续到分钟边界，
    即 bucket=t 就是 K 线 bar [t, t+60000) **收盘时刻**的盘口。因此
    close_time = bucket + 60000 是同一瞬间的右闭标签。

消费端须知（生产端未修的两处 quirk，必须在这里兜掉）
-------------------------------------------------
1. 重复分钟：相邻小时文件在每小时的第 59 分钟重叠，同一分钟被写两遍
   （实测 2026-09 futures：43891 行 -> 43172 个 bucket）。重复行内容基本相同
   （单日仅 ~1.7% 的档位 q 有 <1% 差异），故按 (bucket, side, px) 去重即可。
2. 缺分钟：源端断流 / 该分钟只有单边有量，月度文件里整分钟消失
   （2026-09 futures 缺 29 分钟、spot 缺 58 分钟）。本模块**不补洞**：
   输出就是实际存在的分钟，reindex 与滚动窗口交给 `ob_rolling.py`。

特征口径
--------
所有距离统一为**相对 mid 的 bps**（`d = |px-mid|/mid*1e4`），所有量都是 BTC 枚数。
这样跨交易所（futures tick 0.1 / spot tick 0.01，档位数差一个量级）和跨价位都可比。
深度默认按 **vol8** 分箱：距离 / 当期 σ_t，边界 1 / 2 / 4 / 8 σ（v0..v4）。
σ 中位数约 3.8bps、band 到 ±30bps（≈8σ），所以 1/2/4/8 正好铺满整个 band；
单月实测桶占比最均衡（futures max 桶 .336 / 熵 .939，bps 口径为 .374 / .848）。
固定 bps 口径（b0..b4）保留，用 `--bin-mode bps` 可复现历史结果。

计算结构（避免把原始档位表展开成宽表）
--------------------------------------
150M 行 --group_by(bucket, side, bin)--> ~430K 行 --pivot--> 43K 行宽表。
唯一的重活是那一次 group_by，其余都在小表上做。

用法
----
    python features/ob_features.py \
        --band data/ob/binance_futures_minute_band_2026-09.parquet \
        --mid  data/ob/binance_futures_minute_band_mid_2026-09.parquet \
        --out  data/ob/ob_features_binance_futures_2026-09.parquet
"""
from __future__ import annotations

import argparse
import glob
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

# 两种分箱口径。都切 5 个桶，列数一致，便于同维度 A/B。
#   bps：固定距离 [0,2) [2,5) [5,10) [10,20) [20,30] bps（历史口径）
#   vol：距离 / 当期波动尺度 sigma（无量纲），阈值 [0,0.5) [0.5,1) [1,2) [2,4) [4,inf)
#        —— 周末低波动时 2bps 就算"远"，美股开盘前后 20bps 可能只是"近"，固定 bps 表达不了。
BIN_MODES = {
    "bps": ((2.0, 5.0, 10.0, 20.0), ("b0", "b1", "b2", "b3", "b4")),
    # vol：0.5/1/2/4 倍 sigma；vol8：1/2/4/8 倍 —— sigma 中位数约 3.8bps、band 到 30bps
    # （≈8σ），用 4σ 封顶会把近一半挂单挤进最后一桶，8σ 才覆盖满 band。
    "vol": ((0.5, 1.0, 2.0, 4.0), ("v0", "v1", "v2", "v3", "v4")),
    "vol8": ((1.0, 2.0, 4.0, 8.0), ("v0", "v1", "v2", "v3", "v4")),
    # vol02：0.2/0.5/1/2 倍 sigma，贴价侧重（σ≈3.8bps 时 2σ≈7.6bps）
    "vol02": ((0.2, 0.5, 1.0, 2.0), ("v0", "v1", "v2", "v3", "v4")),
}
DEFAULT_BIN_MODE = "vol8"                    # 生产口径：波动率倍数（见模块 docstring）
BIN_EDGES, BIN_NAMES = BIN_MODES[DEFAULT_BIN_MODE]
SIGMA_WINDOW = 30        # 波动尺度：过去 N 个 1 分钟对数收益的标准差
SIGMA_FLOOR = 0.5        # sigma 下限（bps），防除零与极端放大
DATE_DIR = re.compile(r"^\d{4}-\d{2}-\d{2}$")   # 日包解压目录名
T = time.time()


def _groups(mode: str) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """按口径返回 (近层, 至x, 远层) 的桶名组合。"""
    names = BIN_MODES[mode][1]
    return (names[:2], names[:3], names[3:])


def log(msg: str) -> None:
    print(f"[{time.time() - T:7.1f}s] {msg}", flush=True)


def _sanitize(df: pl.DataFrame) -> pl.DataFrame:
    """把 inf / -inf / NaN 统一置 null，避免除零产生的非常数静默污染下游模型。"""
    cols = [c for c in df.columns if df[c].dtype in (pl.Float32, pl.Float64)]
    return df.with_columns([
        pl.when(pl.col(c).is_finite()).then(pl.col(c)).otherwise(None).alias(c)
        for c in cols
    ])


def _bin_expr(d: pl.Expr, edges: tuple[float, ...]) -> pl.Expr:
    """距离 -> 分箱索引 0..len(edges)；超出最后一个上界归到最后一箱。"""
    e = pl.lit(len(edges), dtype=pl.Int8)
    for i in reversed(range(len(edges))):
        e = pl.when(d < edges[i]).then(pl.lit(i, dtype=pl.Int8)).otherwise(e)
    return e


def _sigma_from_mid(mid: pl.DataFrame, window: int = SIGMA_WINDOW) -> pl.DataFrame:
    """每分钟的波动尺度 sigma（bps）= 过去 window 个 1 分钟对数收益的标准差。

    shift(1) 是防前视的硬前提：t 时刻的分箱边界只能用 t-1 及之前实现的收益。
    开头 window 分钟不足、以及 mid 断档处的 null，用当月中位数填（不用未来值）。
    """
    s = (
        mid.sort("bucket")
        .select("bucket", "mid")
        .with_columns((pl.col("mid") / pl.col("mid").shift(1)).log().mul(1e4).alias("r"))
        .with_columns(pl.col("r").rolling_std(window).shift(1).alias("sigma"))
        .select("bucket", "sigma")
    )
    med = s["sigma"].median()
    if med is None or not med > 0:
        raise SystemExit("sigma 全为空，无法用波动率分箱（mid 序列可能断档过多）")
    log(f"sigma: median={med:.2f}bps, null={s['sigma'].null_count():,}")
    return s.with_columns(pl.col("sigma").fill_null(med).clip(lower_bound=SIGMA_FLOOR))


def _prepare(band: pl.LazyFrame, edges: tuple[float, ...],
             sigma: pl.DataFrame | None) -> pl.LazyFrame:
    """逐行派生 sc / d(bps) / 分箱尺度 / bin / key。

    去重不在这里做：调用方按文件先 `unique` 再喂进来（见 `_scan_band`），
    否则整月 1.5 亿行一次性去重会把内核内存打爆（Kaggle 上 SIGKILL 过）。

    sigma 为 None 时按固定 bps 分箱；否则先用 `d/σ` 归一化再按倍数切桶。
    """
    lf = band.with_columns(
        pl.when(pl.col("side") == "ask").then(1).otherwise(0)
        .cast(pl.Int8).alias("sc"),
        (((pl.col("px") - pl.col("mid")) / pl.col("mid")) * 1e4).abs().alias("d_bps"),
    )
    if sigma is not None:
        lf = lf.join(sigma.lazy(), on="bucket", how="left").with_columns(
            (pl.col("d_bps") / pl.col("sigma")).alias("scal")
        )
    else:
        lf = lf.with_columns(pl.col("d_bps").alias("scal"))

    return (
        lf.with_columns(_bin_expr(pl.col("scal"), edges).alias("b"))
        .with_columns((pl.col("sc") * 8 + pl.col("b")).cast(pl.Int16).alias("key"))
    )


_AGG = [
    pl.col("q").sum().alias("q_sum"),
    pl.len().cast(pl.Int32).alias("n"),
    (pl.col("q") * pl.col("px")).sum().alias("qx"),
    pl.col("q").max().alias("q_max"),
    (pl.col("q") ** 2).sum().alias("q2"),
    # 距离用 bps（人看得懂），不用归一化后的无量纲值
    pl.col("d_bps").sort_by(pl.col("q"), descending=True).first().alias("d_at_maxq"),
    pl.col("mid").first().alias("mid"),
]


def _collect(agg: pl.LazyFrame) -> pl.DataFrame:
    """流式引擎优先，不支持时退回内存引擎。"""
    last = None
    for engine in ("streaming", "in-memory"):
        try:
            return agg.collect(engine=engine)
        except Exception as exc:  # noqa: BLE001 - 流式算子缺失时退回内存引擎
            last = exc
            log(f"collect(engine={engine}) failed: {type(exc).__name__}")
    raise SystemExit(f"cannot collect bin aggregation: {last}")


def _scan_band(files: list[str]) -> list[pl.LazyFrame]:
    """每个日 parquet 一条 lazy 管线：切出「自己那一天」并去掉重复档位。

    两件事都在**文件级**做，因为整月一次性去重（1.5 亿行 hash set）会把内核
    内存打爆——Kaggle 上跑 2025-08 时被 SIGKILL 过。

    1. 日包开头带前一天最后一分钟（warmup），相邻两个文件各存一份：
       目录名是日期时按 [D 00:00, D+1 00:00) 切，彻底消除跨文件重复。
    2. 文件内部也有重复（源端每小时文件在 :59 重叠），按 (bucket, side, px) 去重。
    """
    out = []
    for f in files:
        lf = pl.scan_parquet(f)
        day = Path(f).parent.name
        if DATE_DIR.match(day):
            t0 = int(datetime.fromisoformat(day).replace(
                tzinfo=timezone.utc).timestamp()) * 1000
            lf = lf.filter((pl.col("bucket") >= t0) & (pl.col("bucket") < t0 + 86_400_000))
        out.append(lf.unique(subset=["bucket", "side", "px"], keep="any"))
    return out


# 跨日合并：加和量、取最大单档，并用「q_max 最大的那一份」决定 d_at_maxq
_MERGE = [
    pl.col("q_sum").sum(),
    pl.col("n").sum(),
    pl.col("qx").sum(),
    pl.col("q_max").max(),
    pl.col("q2").sum(),
    pl.col("d_at_maxq").sort_by(pl.col("q_max"), descending=True).first(),
    pl.col("mid").first(),
]


def _bin_agg(files: list[str], edges: tuple[float, ...],
             sigma: pl.DataFrame | None,
             limit_rows: int | None = None) -> pl.DataFrame:
    """重活：每日 4~5M 行 -> (bucket, key) 聚合，再跨日合并成整月。

    逐日聚合是刻意的：一次性对整月 1.5 亿行去重/分组需要十几 GB，Kaggle 上
    会被 OOM killer 直接 SIGKILL。按文件切分后峰值只跟单日有关。
    """
    parts = []
    for i, lf in enumerate(_scan_band(files)):
        if limit_rows is not None:
            if i > 0:
                break
            lf = lf.head(limit_rows)
        parts.append(_collect(
            _prepare(lf, edges, sigma).group_by(["bucket", "key"]).agg(_AGG)))
    if not parts:
        raise SystemExit(f"band 文件全部为空: {files[:1]}")
    merged = _collect(
        pl.concat(parts).lazy().group_by(["bucket", "key"]).agg(_MERGE))
    return merged.with_columns(pl.col("n").cast(pl.Int32))


def _side_totals(bin_agg: pl.DataFrame) -> pl.DataFrame:
    """分箱聚合 -> (bucket, side) 总量。"""
    return (
        bin_agg.with_columns((pl.col("key") // 8).cast(pl.Int8).alias("sc"))
        .group_by(["bucket", "sc"])
        .agg(
            pl.col("q_sum").sum().alias("q"),
            pl.col("n").sum().alias("n"),
            pl.col("qx").sum().alias("qx"),
            pl.col("q2").sum().alias("q2"),
            pl.col("q_max").max().alias("q_max"),
            pl.col("d_at_maxq").sort_by(pl.col("q_max"), descending=True)
            .first().alias("q_max_dist_bps"),
            pl.col("mid").first().alias("mid"),
        )
    )


def _pivot_bins(bin_agg: pl.DataFrame, bin_names: tuple[str, ...]) -> pl.DataFrame:
    """分箱量 / 档数 / 分箱 VWAP(相对 mid 的 bps) 转宽表。

    pivot 输出列名是 `{value}_{key}`，这里显式改名成 `ob_q_bid_b0` 这类，
    不依赖隐含规则。某分钟某箱为空 -> 量与档数补 0，VWAP 保持 null（真没深度）。
    """
    idx = {str(i): n for i, n in enumerate(bin_names)}
    wide = (
        bin_agg.with_columns(
            pl.when(pl.col("key") // 8 == 0).then(pl.lit("bid"))
            .otherwise(pl.lit("ask")).alias("side"),
            ((pl.col("qx") / pl.col("q_sum") - pl.col("mid")) / pl.col("mid") * 1e4)
            .alias("vwap_bps"),
        )
        .with_columns(
            (pl.col("side") + "_" + (pl.col("key") % 8).cast(pl.Utf8)
             .replace_strict(idx)).alias("k")
        )
        .pivot(on="k", index="bucket", values=["q_sum", "n", "vwap_bps"])
    )
    ren = {}
    for s in ("bid", "ask"):
        for bn in bin_names:
            ren[f"q_sum_{s}_{bn}"] = f"ob_q_{s}_{bn}"
            ren[f"n_{s}_{bn}"] = f"ob_n_{s}_{bn}"
            ren[f"vwap_bps_{s}_{bn}"] = f"ob_vwap_{s}_{bn}_bps"
    wide = wide.rename({k: v for k, v in ren.items() if k in wide.columns})
    missing = [v for v in ren.values() if v not in wide.columns]
    if missing:  # 整月都没有某个箱（极端行情）
        wide = wide.with_columns([pl.lit(None).alias(c) for c in missing])
    wide = wide.with_columns([
        pl.col(f"ob_q_{s}_{bn}").fill_null(0.0)
        for s in ("bid", "ask") for bn in bin_names
    ] + [
        pl.col(f"ob_n_{s}_{bn}").fill_null(0)
        for s in ("bid", "ask") for bn in bin_names
    ])
    # 显式定列序：pivot 的列序取决于分组出现的先后（逐日聚合 vs 单文件聚合不同），
    # 不定序会让同一份数据在不同运行里列序漂移，下游按位置取列就静默错位。
    order = ["bucket"]
    for metric in ("q", "n", "vwap"):
        for s in ("bid", "ask"):
            for bn in bin_names:
                order.append({"q": f"ob_q_{s}_{bn}",
                              "n": f"ob_n_{s}_{bn}",
                              "vwap": f"ob_vwap_{s}_{bn}_bps"}[metric])
    return wide.select(order)


def build_l1(band_path: str | Path, mid_path: str | Path,
             limit_rows: int | None = None,
             bin_mode: str = DEFAULT_BIN_MODE) -> pl.DataFrame:
    """band(+mid) -> 分钟截面特征表（主键 close_time）。

    `band_path` / `mid_path` 可以是单文件、通配（一个月的日包）或 glob 列表。
    多文件时逐日聚合再合并，日界重复的那一分钟按目录名切掉，不会重复计入。

    bin_mode="vol8"（默认）：按 `距离 / σ_t` 的波动率倍数分箱（v0..v4，1/2/4/8σ），
    并多输出 `ob_sigma_bps`（当期 1 分钟波动尺度）。
    bin_mode="vol"/"vol02"：同上但边界不同（0.5/1/2/4σ、0.2/0.5/1/2σ），仅作对照。
    bin_mode="bps"：历史口径，按固定 bps 距离分箱（b0..b4）。
    所有口径都是 5 个桶、列数一致。

    `limit_rows` 只用于本地冒烟测试：截取文件最前面的 N 行跑通逻辑，避免在
    本地对 1.5 亿行做聚合（那是 Kaggle 的活）。全量计算必须走流式引擎。
    """
    if bin_mode not in BIN_MODES:
        raise SystemExit(f"unknown bin_mode: {bin_mode}")
    edges, names = BIN_MODES[bin_mode]
    near, upto10, far = _groups(bin_mode)

    mid = (
        pl.scan_parquet(str(mid_path))
        .unique(subset=["bucket"], keep="any")
        .select("bucket", "mid", "spread")
        .collect()
        .sort("bucket")
    )
    # 除了 "bps" 以外都是波动率口径（vol / vol8 ...），都要先算 sigma
    sigma = _sigma_from_mid(mid) if bin_mode != "bps" else None

    files = sorted(glob.glob(str(band_path)))
    if not files:
        raise SystemExit(f"没有 band 文件匹配 {band_path}")
    log(f"band files: {len(files)}")

    log(f"aggregating bins (mode={bin_mode}) ...")
    bin_agg = _bin_agg(files, edges, sigma, limit_rows=limit_rows)
    log(f"bin rows: {bin_agg.height:,}")

    df = _pivot_bins(bin_agg, names).join(mid, on="bucket", how="left")
    if sigma is not None:
        df = df.join(sigma.rename({"sigma": "ob_sigma_bps"}), on="bucket", how="left")
    for sc, s in ((0, "bid"), (1, "ask")):
        df = df.join(
            _side_totals(bin_agg).filter(pl.col("sc") == sc).select(
                "bucket",
                pl.col("q").alias(f"ob_q_{s}"),
                pl.col("n").alias(f"ob_n_{s}"),
                pl.col("qx").alias(f"_qx_{s}"),
                pl.col("q2").alias(f"_q2_{s}"),
                pl.col("q_max").alias(f"ob_qmax_{s}"),
                pl.col("q_max_dist_bps").alias(f"ob_qmax_dist_bps_{s}"),
            ),
            on="bucket", how="left",
        )

    qb, qa = pl.col("ob_q_bid"), pl.col("ob_q_ask")

    def q(side: str, bins: tuple[str, ...]) -> pl.Expr:
        return pl.sum_horizontal([pl.col(f"ob_q_{side}_{b}") for b in bins])

    df = df.with_columns(
        (pl.col("spread") / pl.col("mid") * 1e4).alias("ob_spread_bps"),
        # ---- 不平衡 ----
        ((qb - qa) / (qb + qa)).alias("ob_obi_total"),
        ((pl.col(f"ob_q_bid_{names[0]}") - pl.col(f"ob_q_ask_{names[0]}"))
         / (pl.col(f"ob_q_bid_{names[0]}") + pl.col(f"ob_q_ask_{names[0]}")
            )).alias(f"ob_obi_{names[0]}"),
        ((q("bid", near) - q("ask", near))
         / (q("bid", near) + q("ask", near))).alias("ob_obi_near"),
        ((q("bid", far) - q("ask", far))
         / (q("bid", far) + q("ask", far))).alias("ob_obi_far"),
        # ---- 价格（用 mid 锚定的 band VWAP，不用单档） ----
        (((pl.col("_qx_ask") / qa) * qb + (pl.col("_qx_bid") / qb) * qa) / (qb + qa)
         - pl.col("mid")).truediv(pl.col("mid")).mul(1e4).alias("ob_wmid_bps"),
        # ---- 冲击 / 形态 ----
        (10.0 / q("bid", upto10)).alias("ob_impact_bid_bps_per_btc"),
        (10.0 / q("ask", upto10)).alias("ob_impact_ask_bps_per_btc"),
        (q("bid", near) / qb).alias("ob_depth_conc_bid"),
        (q("ask", near) / qa).alias("ob_depth_conc_ask"),
        (pl.col("ob_n_bid") / pl.col("ob_n_ask")).alias("ob_levels_ratio"),
        (qb / pl.col("ob_n_bid")).alias("ob_qmean_bid"),
        (qa / pl.col("ob_n_ask")).alias("ob_qmean_ask"),
        (pl.col("_q2_bid") / qb**2).alias("ob_hhi_bid"),
        (pl.col("_q2_ask") / qa**2).alias("ob_hhi_ask"),
        (pl.col("ob_qmax_bid") * pl.col("ob_n_bid") / qb).alias("ob_qmax_ratio_bid"),
        (pl.col("ob_qmax_ask") * pl.col("ob_n_ask") / qa).alias("ob_qmax_ratio_ask"),
        ((pl.col("_q2_bid") / pl.col("ob_n_bid")
          - (qb / pl.col("ob_n_bid")) ** 2).sqrt()
         / (qb / pl.col("ob_n_bid"))).alias("ob_qcv_bid"),
        ((pl.col("_q2_ask") / pl.col("ob_n_ask")
          - (qa / pl.col("ob_n_ask")) ** 2).sqrt()
         / (qa / pl.col("ob_n_ask"))).alias("ob_qcv_ask"),
    ).with_columns(
        (pl.col("ob_obi_far") - pl.col("ob_obi_near")).alias("ob_obi_tilt"),
        ((pl.col("ob_impact_bid_bps_per_btc") - pl.col("ob_impact_ask_bps_per_btc"))
         / (pl.col("ob_impact_bid_bps_per_btc")
            + pl.col("ob_impact_ask_bps_per_btc"))).alias("ob_impact_imb"),
    )

    df = df.with_columns(
        (pl.col("bucket") + 60_000).alias("close_time"),
        pl.col("mid").alias("ob_mid"),
    ).drop(["bucket", "mid", "_qx_bid", "_qx_ask", "_q2_bid", "_q2_ask"])
    cols = ["close_time", "ob_mid"]
    cols += [c for c in df.columns if c.startswith("ob_") and c != "ob_mid"]
    df = df.select(cols).sort("close_time")
    df = df.with_columns([
        pl.col(c).cast(pl.Float32) for c in df.columns
        if c not in ("close_time", "ob_mid")
    ])
    return _sanitize(df)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--band", required=True,
                    help="*_minute_band.parquet；支持通配，如 'ob/*/binance_futures_minute_band.parquet'")
    ap.add_argument("--mid", required=True,
                    help="*_minute_band_mid.parquet；同样支持通配（日包解压后的多个文件）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit-rows", type=int, default=None,
                    help="只读前 N 行做本地冒烟测试；不传则全量")
    ap.add_argument("--bin-mode", choices=tuple(BIN_MODES), default=DEFAULT_BIN_MODE,
                    help="vol8=1/2/4/8σ(默认/生产口径)；bps=固定 bps(历史)；"
                         "vol=0.5/1/2/4σ；vol02=0.2/0.5/1/2σ")
    ap.add_argument("--force", action="store_true",
                    help="确认在本地跑全量（>256MB 的 band）。默认拒绝，避免占满本机")
    args = ap.parse_args()

    # band 可能是通配（一个月的日包），体积按所有匹配文件求和
    band_mb = sum(p.stat().st_size for p in map(Path, glob.glob(args.band))) / 1e6
    if args.limit_rows is None and band_mb > 256 and not args.force:
        raise SystemExit(
            f"band 文件 {band_mb:.0f} MB，全量聚合约 1.5 亿行，会把本机 CPU/内存打满。\n"
            "本地只做冒烟：加 --limit-rows 5000000 跑前几百万行；\n"
            "全量请放到 Kaggle 内核上跑（确认过再 --force）。"
        )

    df = build_l1(args.band, args.mid, limit_rows=args.limit_rows,
                  bin_mode=args.bin_mode)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(args.out)
    log(f"wrote {args.out}: {df.height:,} rows x {df.width} cols")

    with pl.Config(tbl_cols=10, tbl_rows=6):
        print(df.select(["close_time", "ob_spread_bps", "ob_obi_total", "ob_obi_near",
                         "ob_wmid_bps", "ob_impact_bid_bps_per_btc"]).head(6))
    nulls = {c: int(v) for c, v in df.null_count().to_dicts()[0].items() if v}
    print("null cols:", nulls)


if __name__ == "__main__":
    main()
