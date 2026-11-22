"""修正 faiss_daily_quantile.py 的 int(ace) 方向截断 bug。

原实现第 35 行 `rows.append((int(ace[i]), ...))`：ace 绝对值 <1（实测 max 0.82），
int() 对 100% 行=0，第 44 行 np.where(a>0,1,0) 恒判「跌」→ 52.80% 是「全押跌」的准确率，
不是因果滚动阈值的真实表现。这里保留原始 ace 判方向，其余逻辑完全不变。
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
                rows.append((float(ace[i]), int(y[i]), int(wd)))  # FIX: 保留浮点 ace
                used += 1
    return rows


def report(tag, rows, n_days):
    if not rows:
        print(tag, "n=0")
        return
    a = np.array([r[0] for r in rows])
    y = np.array([r[1] for r in rows])
    w = (np.where(a > 0, 1, 0) == y)
    wd = np.array([r[2] for r in rows]) <= 5
    se = np.sqrt(w.mean() * (1 - w.mean()) / max(len(w) / 15, 1))
    extra = ""
    if wd.sum() >= 20:
        extra = f" WD_win={w[wd].mean()*100:.2f}%(n={wd.sum()})"
    print(f"{tag}: n={len(w)} per_day={len(w)/n_days:.1f} win={w.mean()*100:.2f}% "
          f"SE={se*100:.2f}pp EV={PAYOUT*w.mean()-(1-w.mean()):+.4f}{extra}")


all_rows = []
for sym in ("BTCUSDT", "ETHUSDT"):
    d = pl.read_parquet(OUT / f"faiss_scores_{sym}.parquet")
    d = d.with_columns(
        pl.from_epoch(pl.col("close_time"), time_unit="ms").dt.strftime("%Y-%m-%d").alias("day"))
    d = d.filter(pl.col("close_time") >= START)
    n_days = d["day"].n_unique()
    rows = causal_select(d)
    report(f"{sym} ALL", rows, n_days)
    report(f"{sym} WD", [r for r in rows if r[2] <= 5], n_days)
    all_rows += rows
report("COMBINED ALL", all_rows, n_days)
report("COMBINED WD", [r for r in all_rows if r[2] <= 5], n_days)
