"""工作日二级门控：周末 h3 冻结，只用 val 训 meta，test 只评估。

思路：现役 h3 分数给候选，K 线特征给独立上下文；meta 预测「这一单是否会赢」，
在 test 上把 wd 候选单按 meta 概率取 top 档，目标 10~15 单/天且胜率高于同期基线。
纯本地，不改任何基线文件。
"""
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl
import lightgbm as lgb

ROOT = Path(r"e:/codebuddy_wrokspace/bn_bt")
sys.path.insert(0, str(ROOT))
from morph import kline_long as kl  # noqa: E402

PRED = ROOT / ".codebuddy/artifacts/h3gate/h3_gate_pred.parquet"
KL = Path(r"e:/codebuddy_wrokspace/bn_hisdata/data/vision")
THR_TPD = 15
TARGET_TPD = 12
PAYOUT = 0.8


def log(*a):
    print(*a, flush=True)


# ---- 1. 本地 K 线特征（2025+ 给 1440 根预热）----
files = sorted(KL.glob("BTCUSDT_1m_klines_202[56].parquet"))
raw = pl.concat([pl.read_parquet(f) for f in files], how="vertical_relaxed")
raw = raw.unique(subset=["close_time"], keep="last").sort("close_time")
kdf, kcols = kl.build(raw)
keep = ["close_time"] + kcols
kdf = kdf.select(keep)
log("K 线特征", kdf.shape)

# ---- 2. h3 分数 + weekday ----
p = pl.read_parquet(PRED).filter(pl.col("split") == "time")
p = p.with_columns(pl.from_epoch(pl.col("close_time"), time_unit="ms").dt.weekday().alias("wd"))
d = p.join(kdf, on="close_time", how="inner").filter(pl.col("wd") <= 5).sort("close_time")
va, te = (d.filter(pl.col("segment") == s) for s in ("val", "test"))
log("weekday val/test", va.height, te.height)

for df in (va, te):
    pass
va = va.with_columns([
    (pl.col("score") - 0.5).alias("s"),
    (pl.col("score") - 0.5).abs().alias("abs_s"),
    ((pl.col("target") == 1).cast(pl.Int8)).alias("up"),
])
te = te.with_columns([
    (pl.col("score") - 0.5).alias("s"),
    (pl.col("score") - 0.5).abs().alias("abs_s"),
    ((pl.col("target") == 1).cast(pl.Int8)).alias("up"),
])
meta_cols = ["s", "abs_s"] + kcols

# ---- 3. 候选阈值只在 val 定 ----
thr = float(np.quantile(np.abs(va["s"].to_numpy()), 1 - THR_TPD / 1440))
log("base thr", round(thr, 5))

# ---- 4. meta 训练：label = 该 bar 若按 base 方向下注是否会赢 ----
# val 前半训练、后半早停，不碰 test
vdays = np.unique(va["close_time"].to_numpy() // 86_400_000)
cut = int(vdays[len(vdays) * 2 // 3])
mtr = va["close_time"].to_numpy() // 86_400_000 < cut
mes = ~mtr
Xtr = va.filter(pl.col("close_time").is_in(va.filter(mtr)["close_time"].to_numpy())) if False else va
# polars mask via numpy index
va_np = va.to_numpy()
mask_tr = (va["close_time"].to_numpy() // 86_400_000 < cut)
mask_es = ~mask_tr
Xtr = va.select(meta_cols).to_numpy()[mask_tr].astype(np.float32)
Xes = va.select(meta_cols).to_numpy()[mask_es].astype(np.float32)
dir_tr = np.where(va["s"].to_numpy()[mask_tr] > 0, 1, 0)
dir_es = np.where(va["s"].to_numpy()[mask_es] > 0, 1, 0)
ytr = (dir_tr == va["up"].to_numpy()[mask_tr]).astype(np.int8)
yes = (dir_es == va["up"].to_numpy()[mask_es]).astype(np.int8)
log("meta train/es", Xtr.shape, Xes.shape, "base win rate", round(float(ytr.mean()), 4), round(float(yes.mean()), 4))

params = dict(objective="binary", metric="auc", learning_rate=0.05, num_leaves=15,
              min_child_samples=500, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=1, verbose=-1, num_threads=os.cpu_count() or 4, seed=42)
ds_tr = lgb.Dataset(Xtr, ytr, feature_name=meta_cols)
ds_es = lgb.Dataset(Xes, yes, reference=ds_tr)
mdl = lgb.train(params, ds_tr, num_boost_round=600,
                valid_sets=[ds_es], callbacks=[lgb.early_stopping(40, verbose=False)])
log("meta best_iter", mdl.best_iteration)

# ---- 5. meta 阈值：val 后半的候选单上按 tpd=12 定 ----
vp = mdl.predict(va.select(meta_cols).to_numpy().astype(np.float32), num_iteration=mdl.best_iteration)
vs = va["s"].to_numpy()
vy = va["up"].to_numpy()
vcand = np.abs(vs) >= thr
vmask = vcand & mask_es
meta_thr = float(np.quantile(vp[vmask], 1 - TARGET_TPD / 1440)) if vmask.sum() > 30 else np.nan
vw = ((np.where(vs[vmask] > 0, 1, 0) == vy[vmask]))
log("val candidates", int(vcand.sum()), "val meta thr", round(meta_thr, 5) if np.isfinite(meta_thr) else None,
    "val filtered n", int((vp[vmask] >= meta_thr).sum()),
    "val filtered win", round(float(vw[vp[vmask] >= meta_thr].mean()), 4) if np.isfinite(meta_thr) else None)

# ---- 6. test 只评估 ----
tp = mdl.predict(te.select(meta_cols).to_numpy().astype(np.float32), num_iteration=mdl.best_iteration)
ts = te["s"].to_numpy()
ty = te["up"].to_numpy()
tcand = np.abs(ts) >= thr
tdays = len(np.unique(te["close_time"].to_numpy() // 86_400_000))


def report(tag, idx):
    if len(idx) == 0:
        log(tag, "n=0"); return
    w = (np.where(ts[idx] > 0, 1, 0) == ty[idx])
    neff = max(len(w) / 10, 1)
    se = np.sqrt(w.mean() * (1 - w.mean()) / neff)
    log(f"{tag}: n={len(w)} per_day={len(w)/tdays:.1f} win={w.mean()*100:.2f}% "
        f"SE={se*100:.2f}pp EV={PAYOUT*w.mean()-(1-w.mean()):+.4f}")


report("test base(cand)", np.where(tcand)[0])
if np.isfinite(meta_thr):
    report("test meta filtered", np.where(tcand & (tp >= meta_thr))[0])
# 同单量基线：test 上按 base |s| 取前 TARGET_TPD 单/天（仅诊断，不用于选参）
k = int(TARGET_TPD * tdays)
top = np.argsort(-np.abs(ts))[:k]
report("test base topN(diagnostic)", top)
