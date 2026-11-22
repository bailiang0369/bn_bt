"""工作日 ETH 共振过滤：周末 h3 冻结，只用 val 定规则，test 评估。"""
import numpy as np
import polars as pl
from pathlib import Path

ROOT = Path(r"e:/codebuddy_wrokspace/bn_bt")
KL = Path(r"e:/codebuddy_wrokspace/bn_hisdata/data/vision")
PRED = ROOT / ".codebuddy/artifacts/h3gate/h3_gate_pred.parquet"
TPD = 15
PAYOUT = 0.8

eth = pl.read_parquet(KL / "ETHUSDT_1m_klines_2026.parquet").select("close_time", "close").sort("close_time")
for k in (3, 10, 15, 60):
    eth = eth.with_columns((pl.col("close") - pl.col("close").shift(k)).alias(f"d{k}"))

p = pl.read_parquet(PRED).filter(pl.col("split") == "time")
p = p.with_columns(pl.from_epoch(pl.col("close_time"), time_unit="ms").dt.weekday().alias("wd"))
d = p.join(eth, on="close_time", how="inner").filter(pl.col("wd") <= 5)
va, te = (d.filter(pl.col("segment") == s) for s in ("val", "test"))

s = va["score"].to_numpy() - .5
thr = float(np.quantile(np.abs(s), 1 - TPD / 1440))
print("base thr", round(thr, 5))


def stat(tag, df, mask):
    ss = df["score"].to_numpy() - .5
    y = df["target"].to_numpy()
    idx = np.where((np.abs(ss) >= thr) & mask)[0]
    if len(idx) == 0:
        print(tag, "n=0"); return
    w = (np.where(ss[idx] > 0, 1, 0) == y[idx])
    nd = len(np.unique(df["close_time"].to_numpy() // 86_400_000))
    se = np.sqrt(w.mean() * (1 - w.mean()) / max(len(w) / 10, 1))
    print(f"{tag}: n={len(w)} per_day={len(w)/nd:.1f} win={w.mean()*100:.2f}% "
          f"SE={se*100:.2f}pp EV={PAYOUT*w.mean()-(1-w.mean()):+.4f}")


for k in (3, 10, 15, 60):
    col = f"d{k}"
    # val 先看方向一致 / 相反两档；规则只看 val，不在 test 选
    for name, sign in (("agree", 1), ("oppose", -1)):
        mv = (np.sign(va["score"].to_numpy() - .5) == sign * np.sign(va[col].to_numpy()))
        mt = (np.sign(te["score"].to_numpy() - .5) == sign * np.sign(te[col].to_numpy()))
        stat(f"val {name} eth{k}", va, mv)
        stat(f"test {name} eth{k}", te, mt)
