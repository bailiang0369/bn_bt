"""多标的 faiss 信号每日 top-N 上界诊断（含工作日子集）。

上界：按自然日取 |ace| 最大的 N 单（日内排名，含前视，只用于判断是否有空间）。
若上界都到不了 60%，因果版就不必做；若上界明显 >60%，再改成严格因果的日内动态阈值。
"""
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(r"e:/codebuddy_wrokspace/bn_bt")
sys.path.insert(0, str(ROOT))
from morph import morph as m  # noqa: E402

KL = Path(r"e:/codebuddy_wrokspace/bn_hisdata/data/vision")
OUT = ROOT / ".codebuddy/artifacts"
N_PER_DAY = 7
PAYOUT = 0.8


def local_load(symbol):
    files = sorted(KL.glob(f"{symbol}_1m_klines_*.parquet"))
    df = pl.concat([pl.read_parquet(f) for f in files], how="vertical_relaxed")
    return df.select("close_time", "open", "high", "low", "close").sort("close_time")


m.load_klines = local_load
frames = []
for sym in ("BTCUSDT", "ETHUSDT"):
    r = m.run_experiment(sym, split_date="2025-01-01", horizon=15)
    d = (r["test_final"].select("close_time", "ace", "target")
         .with_columns(
             pl.lit(sym).alias("sym"),
             pl.from_epoch(pl.col("close_time"), time_unit="ms").dt.weekday().alias("wd"),
             (pl.from_epoch(pl.col("close_time"), time_unit="ms")
              .dt.strftime("%Y-%m-%d")).alias("day")))
    frames.append(d)
    d.write_parquet(OUT / f"faiss_scores_{sym}.parquet")
    print(f"[save] {sym} {d.height:,}", flush=True)

all_d = pl.concat(frames)


def daily_top(df, n, only_wd=False):
    x = df.filter(pl.col("wd") <= 5) if only_wd else df
    sel = (x.with_columns(pl.col("ace").abs().alias("_a"))
             .sort(["day", "_a"], descending=[False, True])
             .group_by("day").head(n))
    ace = sel["ace"].to_numpy()
    y = sel["target"].to_numpy()
    wd = sel["wd"].to_numpy()
    win = (np.where(ace > 0, 1, 0) == y)
    nd = len(np.unique(sel["day"].to_numpy()))
    return len(win), len(win) / nd, win.mean(), (wd >= 6).mean()


for sym in ("BTCUSDT", "ETHUSDT"):
    for tag, sub in (("ALL", all_d.filter(pl.col("sym") == sym)),
                     ("WD", all_d.filter((pl.col("sym") == sym) & (pl.col("wd") <= 5)))):
        n, pd_, wr, wes = daily_top(sub, N_PER_DAY)
        print(f"{sym} {tag} top{N_PER_DAY}/day: n={n} per_day={pd_:.1f} "
              f"win={wr*100:.2f}% weekend_share={wes*100:.1f}%")

n, pd_, wr, wes = daily_top(all_d, N_PER_DAY * 2)
print(f"COMBINED top{N_PER_DAY*2}/day: n={n} per_day={pd_:.1f} win={wr*100:.2f}%")
n, pd_, wr, wes = daily_top(all_d, N_PER_DAY * 2, only_wd=True)
print(f"COMBINED-WD top{N_PER_DAY*2}/day: n={n} per_day={pd_:.1f} win={wr*100:.2f}%")
