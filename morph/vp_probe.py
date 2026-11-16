#!/usr/bin/env python3
"""VP（量价分布）特征在**强正则化基线**下的重测（新文件）。

为什么值得重测（2026-11-16）
----------------------------
`实验详录.md 篇五` 第 12 轮结论：63+VP = AUC 0.5280、VP 单独 0.5209
→ 判为「无效（信息已被覆盖）」。但注意 0.5280 相对当年的 0.5269 基线**其实有
+0.0011**，只是不显著。而那一轮用的正是 `num_leaves=63 / min_child_samples=100`
的过拟合配置——低信噪比特征在噪声淹没下最容易被判死。

现在 AUC 基线已抬到 0.532~0.534（reg2），重新测一次才有意义。

因果性
------
直接复用 `kline_vp.build_vp`，并强制先跑 `vp_leak_check()`。该检查抓到过
`pos // anchor` 的锚点错位（AUC 从 0.5277 虚增到 0.7658）——不通过就退出。

对齐方式
--------
VP 在**原始 K 线**上算（需要 volume，用 um_buy_vol+um_sell_vol 合成），再按
close_time join 回标签表。不靠「取前 N 行」对齐，避免中间有缺 bar 时整体错位。

四个变体（同一 walk-forward、同一超参、同一 seeds，完全配对）：
  base   49 列
  vp     15 列（只用 VP，判断它是否自带信息）
  both   49+15
  vpadd  49 列 + 仅 vp_poc/vp_imb（4~6 列，最可能非共线的两族）

env
---
  HORIZON=10 WIN_DAYS=90 SEEDS=42,43,44 ROUNDS=2000
  START_MONTH=2026-01 END_MONTH=2026-10 PAY=0.80
  VP_WINS=240,960,1440 VP_ANCHOR=10 VP_TICK=2.0 VP_SELFCHECK=1
"""
from __future__ import annotations

import datetime as dt
import os
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from morph import kline_long as kl  # noqa: E402
from morph.h10_signal import add_flow  # noqa: E402
from morph.kline_vp import build_vp, vp_leak_check  # noqa: E402
from morph.two_legs import add_target, matrix  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

import lightgbm as lgb  # noqa: E402

HORIZON = int(os.environ.get("HORIZON", "10"))
WIN_DAYS = int(os.environ.get("WIN_DAYS", "90"))
SEEDS = [int(s) for s in os.environ.get("SEEDS", "42,43,44").split(",") if s.strip()]
ROUNDS = int(os.environ.get("ROUNDS", "2000"))
ES_FRAC = float(os.environ.get("ES_FRAC", "0.05"))
THR_FRAC = float(os.environ.get("THR_FRAC", "0.10"))
START_MONTH = os.environ.get("START_MONTH", "2026-01")
END_MONTH = os.environ.get("END_MONTH", "2026-10")
PAY = float(os.environ.get("PAY", "0.80"))
BARS_PER_DAY = int(os.environ.get("BARS_PER_DAY", "1440"))
TPD_LIST = [float(x) for x in os.environ.get("TPD_LIST", "2,5,8,15,30").split(",") if x.strip()]
PRIMARY_TPD = float(os.environ.get("PRIMARY_TPD", "8"))
LGB_REG = dict(num_leaves=15, min_child_samples=3000, feature_fraction=.5, lambda_l2=100)
T = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T:7.1f}s] {msg}", flush=True)


def fit(Xtr, ytr, cols, Xes, yes, seed):
    params = dict(objective="binary", metric="auc", boosting_type="gbdt",
                  learning_rate=.05, bagging_fraction=.9, bagging_freq=1,
                  verbose=-1, seed=seed, num_threads=os.cpu_count() or 4)
    params.update(LGB_REG)
    ds = lgb.Dataset(Xtr, ytr, feature_name=cols)
    va = lgb.Dataset(Xes, yes, reference=ds)
    return lgb.train(params, ds, num_boost_round=ROUNDS, valid_sets=[ds, va],
                     callbacks=[lgb.early_stopping(50, verbose=False)])


def main() -> None:
    if os.environ.get("VP_SELFCHECK", "1") == "1" and not vp_leak_check():
        raise SystemExit("[vp] 泄露自检失败，拒绝继续")

    raw = kl.load_raw()
    if "volume" not in raw.columns:
        raw = raw.with_columns(
            (pl.col("um_buy_vol").cast(pl.Float64)
             + pl.col("um_sell_vol").cast(pl.Float64)).alias("volume"))
    vp, vpcols = build_vp(raw)
    log(f"VP {len(vpcols)} 列：{vpcols}")

    df, feats = kl.build(raw)
    df, feats = add_flow(df, feats)
    f_base = list(feats)

    # 按 close_time 精确对齐（VP 在原始行上算，标签表会砍掉尾部 H 行）
    vp_pl = pl.DataFrame({c: vp[:, i] for i, c in enumerate(vpcols)}).with_columns(
        pl.Series("close_time", raw["close_time"].to_numpy()))
    df = add_target(df, HORIZON).filter(
        pl.col("target").is_not_null() & pl.col("close").is_not_null()).sort("close_time")
    df = df.join(vp_pl, on="close_time", how="left")
    df = df.with_columns(
        pl.col("close_time").cast(pl.Datetime("ms")).dt.strftime("%Y-%m").alias("ym"))
    miss = int(df[vpcols[0]].null_count())
    log(f"特征 base={len(f_base)} vp={len(vpcols)} | 可用 {df.height:,} 行 | "
        f"VP 未对齐 {miss:,} 行 | H={HORIZON} | 窗 {WIN_DAYS}d")

    f_poc = [c for c in vpcols if c.startswith(("vp_poc_", "vp_imb_"))]
    fsets = {
        "base": f_base,
        "vp": vpcols,
        "both": f_base + vpcols,
        "vpadd": f_base + f_poc,
    }
    feats_all = f_base + vpcols
    X_all = matrix(df, feats_all)
    y_all = df["target"].to_numpy().astype(int)
    ct = df["close_time"].cast(pl.Int64).to_numpy()
    ym = df["ym"].to_numpy()

    months = [m for m in sorted(set(ym.tolist())) if START_MONTH <= m <= END_MONTH]
    splits = []
    for m in months:
        m_start = int(dt.datetime.strptime(m + "-01", "%Y-%m-%d")
                      .replace(tzinfo=dt.timezone.utc).timestamp() * 1000)
        lo = m_start - WIN_DAYS * 86_400_000
        i_tr = np.where((ct < m_start) & (ct >= lo))[0]
        i_te = np.where(ym == m)[0]
        if len(i_tr) < 100_000 or len(i_te) < 1_000:
            log(f"[{m}] 样本不足，跳过")
            continue
        h = len(i_tr)
        c1 = int(h * (1 - ES_FRAC - THR_FRAC))
        c2 = int(h * (1 - THR_FRAC))
        splits.append((m, i_tr[:c1], i_tr[c1:c2], i_tr[c2:], i_te))
    log(f"walk-forward {len(splits)} 窗口：{splits[0][0]}~{splits[-1][0]}")

    rows = []
    for name, cols in fsets.items():
        ci = [feats_all.index(c) for c in cols]
        X = X_all[:, ci]
        per = {t: dict(s=[], y=[], nd=[]) for t in TPD_LIST}
        ps, ys = [], []
        for m, i_tr, i_es, i_th, i_te in splits:
            p_th = np.zeros(len(i_th))
            p_te = np.zeros(len(i_te))
            for seed in SEEDS:
                mdl = fit(X[i_tr], y_all[i_tr], cols, X[i_es], y_all[i_es], seed)
                p_th += mdl.predict(X[i_th], num_iteration=mdl.best_iteration) / len(SEEDS)
                p_te += mdl.predict(X[i_te], num_iteration=mdl.best_iteration) / len(SEEDS)
            ps.append(p_te)
            ys.append(y_all[i_te])
            nd = len(np.unique(ct[i_te] // 86_400_000))
            s = p_te - .5
            for t in TPD_LIST:
                thr = float(np.quantile(np.abs(p_th - .5), 1 - t / BARS_PER_DAY))
                hit = np.abs(s) >= thr
                if hit.sum() < 30:
                    continue
                per[t]["s"].append(s[hit])
                per[t]["y"].append(y_all[i_te][hit])
                per[t]["nd"].append(nd)
        auc = float(roc_auc_score(np.concatenate(ys), np.concatenate(ps)))
        rows.append((name, auc, per, len(cols)))
        log(f"  {name:<7} 列数={len(cols):<3} OOS AUC={auc:.4f}")

    # ---- 汇总 ----
    base = next(r for r in rows if r[0] == "base")
    print(f"\n{'变体':<8}{'列数':>5}{'AUC':>9}{'ΔAUC':>9}"
          + "".join(f"{'tpd' + format(t, 'g') + ' 单/天 胜率 EV':>26}" for t in TPD_LIST))
    for name, auc, per, nc in rows:
        cells = ""
        for t in TPD_LIST:
            if not per[t]["y"]:
                cells += f"{'-':>26}"
                continue
            s = np.concatenate(per[t]["s"])
            y = np.concatenate(per[t]["y"])
            nd = float(np.sum(per[t]["nd"]))
            wr = float(((s > 0).astype(int) == y).mean())
            cells += f"{len(y) / nd:>10.1f}{wr * 100:>6.1f}%{PAY * wr - (1 - wr):>+7.3f}"
        print(f"{name:<8}{nc:>5d}{auc:>9.4f}{auc - base[1]:>+9.4f}{cells}")
    log("DONE")


if __name__ == "__main__":
    main()
