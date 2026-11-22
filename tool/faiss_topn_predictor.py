"""训练因果模型预测「这根 bar 最终会不会进当天 |ace| top-N」。

训练标签用当天结束后才可得的 top-N 成员（只作 label），特征只用 bar 当时可见信息。
2025 训/es，2026 test；test 用预测概率阈值 + 每日先到先得 N 单，严格因果。
"""
import os
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

ROOT = Path(r"e:/codebuddy_wrokspace/bn_bt")
sys.path.insert(0, str(ROOT))
from morph import kline_long as kl  # noqa: E402

KL = Path(r"e:/codebuddy_wrokspace/bn_hisdata/data/vision")
OUT = ROOT / ".codebuddy/artifacts"
N = 7
PAYOUT = 0.8


def features(sym):
    files = sorted(KL.glob(f"{sym}_1m_klines_202[56].parquet"))
    raw = pl.concat([pl.read_parquet(f) for f in files], how="vertical_relaxed")
    raw = raw.unique(subset=["close_time"], keep="last").sort("close_time")
    k, cols = kl.build(raw)
    return k.select(["close_time"] + cols)


frames = []
for sym in ("BTCUSDT", "ETHUSDT"):
    fa = pl.read_parquet(OUT / f"faiss_scores_{sym}.parquet")
    fa = fa.with_columns(
        pl.from_epoch(pl.col("close_time"), time_unit="ms").dt.strftime("%Y-%m-%d").alias("day"),
        pl.from_epoch(pl.col("close_time"), time_unit="ms").dt.hour().alias("hr"),
        pl.from_epoch(pl.col("close_time"), time_unit="ms").dt.minute().alias("mi"))
    fa = fa.with_columns(
        (pl.col("hr") * 2 * np.pi / 24).sin().alias("hr_s"),
        (pl.col("hr") * 2 * np.pi / 24).cos().alias("hr_c"),
        pl.lit(sym).alias("sym"))
    kf = features(sym)
    d = fa.join(kf, on="close_time", how="inner")
    # 当天 top-N 标签：只用历史值算，但作为 label 不进特征
    d = d.with_columns(
        pl.col("ace").abs().rank(method="ordinal", descending=True).over("day").alias("rk"))
    d = d.with_columns((pl.col("rk") <= N).cast(pl.Int8).alias("top"))
    frames.append(d)
    print(sym, d.shape, "top rate", d["top"].mean(), flush=True)

d = pl.concat(frames, how="vertical_relaxed")
feat_cols = ["ace", "hr_s", "hr_c"] + [c for c in d.columns if c.startswith("lg_")]
cal_days = sorted(d.filter(pl.col("close_time") < 1767225600000)["day"].unique().to_list())
tr_days = set(cal_days[:int(len(cal_days) * 0.8)])
es_days = set(cal_days[int(len(cal_days) * 0.8):])
tr = d.filter(pl.col("day").is_in(list(tr_days)))
es = d.filter(pl.col("day").is_in(list(es_days)))
te = d.filter(pl.col("close_time") >= 1767225600000)
print("train/es/test", tr.height, es.height, te.height)

Xtr, ytr = tr.select(feat_cols).to_numpy().astype(np.float32), tr["top"].to_numpy()
Xes, yes = es.select(feat_cols).to_numpy().astype(np.float32), es["top"].to_numpy()
ds_tr = lgb.Dataset(Xtr, ytr, feature_name=feat_cols)
ds_es = lgb.Dataset(Xes, yes, reference=ds_tr)
mdl = lgb.train(dict(objective="binary", metric="auc", learning_rate=0.05, num_leaves=31,
                     min_child_samples=500, feature_fraction=0.8, bagging_fraction=0.8,
                     bagging_freq=1, verbose=-1, num_threads=os.cpu_count() or 4, seed=42),
                ds_tr, num_boost_round=400, valid_sets=[ds_es],
                callbacks=[lgb.early_stopping(30, verbose=False)])
print("best_iter", mdl.best_iteration)

pe = mdl.predict(te.select(feat_cols).to_numpy().astype(np.float32), num_iteration=mdl.best_iteration)
te = te.with_columns(pl.Series("pred", pe))
# test 阈值：用 es 的 pred 取 top-N/天（因果，es 在 test 之前）
pes = mdl.predict(es.select(feat_cols).to_numpy().astype(np.float32), num_iteration=mdl.best_iteration)
es = es.with_columns(pl.Series("pred", pes))
thr = float(np.quantile(es["pred"].to_numpy(), 1 - N * es["sym"].n_unique() / 1440))
print("pred thr", round(thr, 6))


def report(tag, df):
    if df.height == 0:
        print(tag, "n=0"); return
    a = df["ace"].to_numpy()
    y = df["target"].to_numpy()
    w = (np.where(a > 0, 1, 0) == y)
    nd = len(np.unique(df["day"].to_numpy()))
    print(f"{tag}: n={len(w)} per_day={len(w)/nd:.1f} win={w.mean()*100:.2f}% "
          f"EV={PAYOUT*w.mean()-(1-w.mean()):+.4f}")


hit = (te.filter(pl.col("pred") >= thr)
          .with_columns(pl.col("close_time").rank(method="ordinal").over(["day", "sym"]).alias("rk"))
          .filter(pl.col("rk") <= N))
report("test selected ALL", hit)
report("test selected WD", hit.filter(pl.col("wd") <= 5))
report("test oracle top7 WD", te.filter((pl.col("rk") <= N) & (pl.col("wd") <= 5)))
