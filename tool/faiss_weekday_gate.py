"""faiss ace 分数作为工作日 h3 二级门控：val 定规则，test 只评估。

faiss 聚类/edge 只用 2025 前数据，分数对 2026 是样本外；不改任何基线文件。
"""
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(r"e:/codebuddy_wrokspace/bn_bt")
sys.path.insert(0, str(ROOT))
from morph import morph as m  # noqa: E402

KL = Path(r"e:/codebuddy_wrokspace/bn_hisdata/data/vision")
PRED = ROOT / ".codebuddy/artifacts/h3gate/h3_gate_pred.parquet"


def local_load(symbol):
    files = sorted(KL.glob(f"{symbol}_1m_klines_*.parquet"))
    df = pl.concat([pl.read_parquet(f) for f in files], how="vertical_relaxed")
    return df.select("close_time", "open", "high", "low", "close").sort("close_time")


m.load_klines = local_load
r = m.run_experiment("BTCUSDT", split_date="2025-01-01", horizon=15)
fa = (r["test_final"].select("close_time", "ace")
      .with_columns(pl.from_epoch(pl.col("close_time"), time_unit="ms").dt.weekday().alias("wd")))
print("[faiss] scores", fa.height, flush=True)

p = pl.read_parquet(PRED).filter(pl.col("split") == "time")
p = p.with_columns(pl.from_epoch(pl.col("close_time"), time_unit="ms").dt.weekday().alias("wd"))
d = p.join(fa, on="close_time", how="inner").filter(pl.col("wd") <= 5)
va, te = (d.filter(pl.col("segment") == s) for s in ("val", "test"))

s = va["score"].to_numpy() - .5
thr = float(np.quantile(np.abs(s), 1 - 15 / 1440))
print("base thr", round(thr, 5), "val/test", va.height, te.height)


def stat(tag, df, idx):
    ss = df["score"].to_numpy() - .5
    y = df["target"].to_numpy()
    if len(idx) == 0:
        print(tag, "n=0"); return
    w = (np.where(ss[idx] > 0, 1, 0) == y[idx])
    nd = len(np.unique(df["close_time"].to_numpy() // 86_400_000))
    se = np.sqrt(w.mean() * (1 - w.mean()) / max(len(w) / 10, 1))
    print(f"{tag}: n={len(w)} per_day={len(w)/nd:.1f} win={w.mean()*100:.2f}% "
          f"SE={se*100:.2f}pp EV={0.8*w.mean()-(1-w.mean()):+.4f}")


for tag, df in (("val", va), ("test", te)):
    stat(f"{tag} base", df, np.where(np.abs(df["score"].to_numpy() - .5) >= thr)[0])

# 规则 1：faiss ace 方向与 h3 方向一致 / 相反。只用 val 选规则。
agree = {}
for name, sign in (("agree", 1), ("oppose", -1)):
    mv = (np.sign(va["score"].to_numpy() - .5) == sign * np.sign(va["ace"].to_numpy()))
    idx = np.where((np.abs(va["score"].to_numpy() - .5) >= thr) & mv)[0]
    agree[name] = sign
    stat(f"val faiss-{name}", va, idx)
# 简单比较：直接算两档 val 胜率
vals = {}
for name, sign in agree.items():
    idx = np.where((np.abs(va["score"].to_numpy() - .5) >= thr) &
                   (np.sign(va["score"].to_numpy() - .5) == sign * np.sign(va["ace"].to_numpy())))[0]
    ss = va["score"].to_numpy() - .5
    vals[name] = (np.where(ss[idx] > 0, 1, 0) == va["target"].to_numpy()[idx]).mean() if len(idx) else 0
best = max(vals, key=vals.get)
print("val 选中规则", best, vals)
for name, sign in agree.items():
    mt = (np.sign(te["score"].to_numpy() - .5) == sign * np.sign(te["ace"].to_numpy()))
    stat(f"test faiss-{name}", te, np.where((np.abs(te["score"].to_numpy() - .5) >= thr) & mt)[0])

# 规则 2：faiss |ace| 分位筛到 12 单/天，阈值只在 val 的候选里定
vc = np.abs(va["ace"].to_numpy())
vbase = np.abs(va["score"].to_numpy() - .5) >= thr
aq = float(np.quantile(vc[vbase], 1 - 12 / 1440)) if vbase.sum() > 30 else np.nan
print("faiss ace thr", round(aq, 6) if np.isfinite(aq) else None)
for tag, df in (("val", va), ("test", te)):
    idx = np.where((np.abs(df["score"].to_numpy() - .5) >= thr) &
                   (np.abs(df["ace"].to_numpy()) >= aq))[0]
    stat(f"{tag} faiss-|ace| top", df, idx)
