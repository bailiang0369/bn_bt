"""faiss ace 因果滚动日内阈值：每分钟只用今天已出现的 bar 重算 top-N 门槛。

与固定阈值区别：门槛随当天已观测分数动态收紧/放松，理论上更接近日内 top-N 上界。
"""
from pathlib import Path

import numpy as np
import polars as pl

OUT = Path(r"e:/codebuddy_wrokspace/bn_bt/.codebuddy/artifacts")
N = 7
PAYOUT = 0.8
START = 1767225600000  # 2026-01-01 UTC


def causal_select(df):
    days = np.unique(df["day"].to_numpy())
    rows = []
    for day in days:
        g = df.filter(pl.col("day") == day).sort("close_time")
        a = np.abs(g["ace"].to_numpy())
        ace = g["ace"].to_numpy()
        y = g["target"].to_numpy()
        wd = g["wd"].to_numpy()[0]
        used = 0
        for i in range(len(a)):
            if used >= N:
                break
            m = i + 1
            if m <= N:
                continue
            q = 1 - N / m
            thr = float(np.quantile(a[:m], q))
            if a[i] >= thr:
                rows.append((int(ace[i]), int(y[i]), int(wd)))
                used += 1
    return rows


def report(tag, rows):
    if not rows:
        print(tag, "n=0"); return
    a = np.array([r[0] for r in rows]); y = np.array([r[1] for r in rows])
    w = (np.where(a > 0, 1, 0) == y)
    print(f"{tag}: n={len(w)} per_day={len(w)/310:.1f} win={w.mean()*100:.2f}% "
          f"EV={PAYOUT*w.mean()-(1-w.mean()):+.4f}")


all_rows = []
for sym in ("BTCUSDT", "ETHUSDT"):
    d = pl.read_parquet(OUT / f"faiss_scores_{sym}.parquet")
    d = d.with_columns(
        pl.from_epoch(pl.col("close_time"), time_unit="ms").dt.strftime("%Y-%m-%d").alias("day"))
    d = d.filter(pl.col("close_time") >= START)
    rows = causal_select(d)
    report(f"{sym} ALL", rows)
    report(f"{sym} WD", [r for r in rows if r[2] <= 5])
    all_rows += rows
report("COMBINED ALL", all_rows)
report("COMBINED WD", [r for r in all_rows if r[2] <= 5])
