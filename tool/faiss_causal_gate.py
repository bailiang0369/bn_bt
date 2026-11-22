"""faiss ace 因果版每日配额：阈值只用历史段定，test 逐 bar 触发、每日最多 N 单。

上界诊断显示日内 top-N 可达 62~66%，但那是前视；这里改成可执行口径：
- cal: 2025 全年，按 1-N/1440 分位定 ace 绝对阈值；
- test: 2026-01~，逐 bar 判断 |ace|>=thr，方向 sign(ace)，当天取前 N 单（先到先得）。
不重训任何模型，分数来自 morph/morph.py（cluster/edge 只用 2025 前数据）。
"""
from pathlib import Path

import numpy as np
import polars as pl

OUT = Path(r"e:/codebuddy_wrokspace/bn_bt/.codebuddy/artifacts")
N = 7
PAYOUT = 0.8


def load(sym):
    d = pl.read_parquet(OUT / f"faiss_scores_{sym}.parquet")
    return d.with_columns(pl.from_epoch(pl.col("close_time"), time_unit="ms")
                          .dt.strftime("%Y-%m-%d").alias("day"))


def report(tag, sub):
    if sub.height == 0:
        print(tag, "n=0"); return
    y = sub["target"].to_numpy()
    ace = sub["ace"].to_numpy()
    w = (np.where(ace > 0, 1, 0) == y)
    nd = len(np.unique(sub["day"].to_numpy()))
    se = np.sqrt(w.mean() * (1 - w.mean()) / max(len(w) / 15, 1))
    print(f"{tag}: n={len(w)} per_day={len(w)/nd:.1f} win={w.mean()*100:.2f}% "
          f"SE={se*100:.2f}pp EV={PAYOUT*w.mean()-(1-w.mean()):+.4f}")


frames = []
for sym in ("BTCUSDT", "ETHUSDT"):
    d = load(sym)
    cal = d.filter(pl.col("close_time") < 1767225600000)  # 2025-01-01 UTC
    te = d.filter(pl.col("close_time") >= 1767225600000)
    thr = float(np.quantile(np.abs(cal["ace"].to_numpy()), 1 - N / 1440))
    print(f"[{sym}] cal n={cal.height} thr={thr:.6f} test n={te.height}")
    report(f"{sym} cal all", cal.filter(pl.col("ace").abs() >= thr))
    report(f"{sym} test all", te.filter(pl.col("ace").abs() >= thr))
    # 每日前 N 单：逐 bar 触发，按时间先到先得（因果）
    hit = (te.filter(pl.col("ace").abs() >= thr)
              .with_columns(pl.col("close_time").rank(method="ordinal").over("day").alias("rk"))
              .filter(pl.col("rk") <= N))
    report(f"{sym} test daily-cap{N}", hit)
    report(f"{sym} test daily-cap{N} WD", hit.filter(pl.col("wd") <= 5))
    frames.append(hit.with_columns(pl.lit(sym).alias("sym")))

comb = pl.concat(frames)
report("COMBINED daily-cap14", comb)
report("COMBINED daily-cap14 WD", comb.filter(pl.col("wd") <= 5))
