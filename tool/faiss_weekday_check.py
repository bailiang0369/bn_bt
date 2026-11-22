"""复用 morph/morph.py 的 faiss 聚类流程，检查信号是否集中在周末。

只读：不改 morph.py，也不改任何基线配置。数据用本地 bn_hisdata 的 BTC 1m K 线。
"""
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(r"e:/codebuddy_wrokspace/bn_bt")
sys.path.insert(0, str(ROOT))
from morph import morph as m  # noqa: E402

KL = Path(r"e:/codebuddy_wrokspace/bn_hisdata/data/vision")


def local_load(symbol):
    files = sorted(KL.glob(f"{symbol}_1m_klines_*.parquet"))
    df = pl.concat([pl.read_parquet(f) for f in files], how="vertical_relaxed")
    return df.select("close_time", "open", "high", "low", "close").sort("close_time")


m.load_klines = local_load
r = m.run_experiment("BTCUSDT", split_date="2025-01-01", horizon=15)
d = r["test_final"].with_columns(
    pl.from_epoch(pl.col("close_time"), time_unit="ms").dt.weekday().alias("wd")
)
print(f"\n[faiss] test n={d.height} horizon=15 base={r['base']:.4f}")

for cov in (0.01, 0.05, 0.25):
    k = int(d.height * cov)
    if k < 1:
        continue
    sub = (d.with_columns(pl.col("ace").abs().alias("_a"))
             .sort("_a", descending=True).head(k))
    wd = sub["wd"].to_numpy()
    we = wd >= 6
    pred = (sub["ace"].to_numpy() > 0).astype(int)
    y = sub["target"].to_numpy()
    win = (pred == y)
    print(f"cov={cov:.0%} n={k} 周末占比={we.mean()*100:.1f}% "
          f"周末胜率={win[we].mean()*100:.2f}% (n={we.sum()}) "
          f"工作日胜率={win[~we].mean()*100:.2f}% (n={(~we).sum()})")
