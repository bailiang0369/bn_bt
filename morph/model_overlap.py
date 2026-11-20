#!/usr/bin/env python3
"""LGB vs XGB 高置信度预测重合度分析。

核心问题：两个模型在 test 月上选出的高置信度样本是否高度重合？
- 重合度高 → 两模型看到的是同一批信号，XGB 输是自然的（模型不是瓶颈）
- 重合度低 → XGB 有独立判断，值得进一步调参/融合

每个 test 月：
  1. 同一 train/es/thr 窗，分别训 LGB(base) 和 XGB(指定 preset 或 tuned params)
  2. 各自在 thr 段定 tpd=8 阈值，取 test 月的高置信度样本
  3. 计算：
     - 分数 Pearson / Spearman 相关
     - top tpd 样本的 Jaccard 重合度
     - 三者（LGB独有 / XGB独有 / 重合）各自的实际胜率
     - 两模型方向判断一致率

env: XGB_PRESET / XGB_PARAMS(json) / MONTHS / TPD / ROUNDS
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import pearsonr, spearmanr
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from morph import ob_kline as ok
from morph import h10_signal as hs
from morph.two_legs import add_target, fit as fit_lgb, matrix, XGB_PRESETS, _XGBWrapper

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

SEEDS = [42, 43, 44, 45, 46]
TPD = int(os.environ.get("TPD", "8"))
ROUNDS = int(os.environ.get("ROUNDS", "2000"))
MONTHS = os.environ.get("MONTHS", "2026-01,2026-02,2026-03,2026-04,2026-05").split(",")
XGB_PRESET = os.environ.get("XGB_PRESET", "reg2")
XGB_PARAMS = os.environ.get("XGB_PARAMS", "")
T = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T:7.1f}s] {msg}", flush=True)


def fit_xgb(Xtr, ytr, Xes, yes, seed, params):
    m = XGBClassifier(objective="binary:logistic", eval_metric="auc", tree_method="hist",
                      n_estimators=ROUNDS, early_stopping_rounds=50,
                      random_state=seed, n_jobs=os.cpu_count() or 4, verbosity=0, **params)
    m.fit(Xtr, ytr, eval_set=[(Xes, yes)], verbose=False)
    return m


def main() -> None:
    # 数据 + 特征
    a = ok._assets(ok.KLINE_TAG)
    names = [f"BTCUSDT_1m_klines_{y}.parquet" for y in range(2020, 2027)]
    k = pl.concat([pl.read_parquet(ok.fetch(ok.KLINE_TAG, n),
                                   columns=["close_time", "open", "high", "low", "close",
                                            "um_buy_vol", "um_sell_vol", "um_count"])
                   for n in names if n in a])
    df, feats = hs.kl.build(k)
    df, feats = hs.add_flow(df, feats)
    df = df.with_columns(pl.col("close_time").cast(pl.Datetime("ms")).dt.strftime("%Y-%m").alias("ym"))
    tagged = add_target(df, 10).filter(pl.col("target").is_not_null()).sort("close_time")
    log(f"tagged {tagged.height:,} rows, {len(feats)} feats")

    xgb_params = json.loads(XGB_PARAMS) if XGB_PARAMS else dict(XGB_PRESETS[XGB_PRESET])
    log(f"XGB params ({XGB_PRESET}): {xgb_params}")

    # 聚合统计
    all_pl, all_px, all_y = [], [], []
    for m in MONTHS:
        t0 = int(dt.datetime.strptime(m + "-01", "%Y-%m-%d").replace(
            tzinfo=dt.timezone.utc).timestamp() * 1000)
        lo = t0 - 1460 * 86_400_000
        pre = tagged.filter((pl.col("close_time") < t0) & (pl.col("close_time") >= lo))
        te = tagged.filter(pl.col("ym") == m)
        n = pre.height
        c1, c2 = int(n * 0.85), int(n * 0.95)
        tr, es, thr = pre[:c1], pre[c1:c2], pre[c2:]
        Xtr, ytr = matrix(tr, feats), tr["target"].to_numpy().astype(int)
        Xes, yes = matrix(es, feats), es["target"].to_numpy().astype(int)
        Xth = matrix(thr, feats)
        Xte = matrix(te, feats)
        yte = te["target"].to_numpy().astype(np.int8)

        pth_l, pte_l = np.zeros(thr.height), np.zeros(te.height)
        pth_x, pte_x = np.zeros(thr.height), np.zeros(te.height)
        for seed in SEEDS:
            ml = fit_lgb(Xtr, ytr, feats, Xes, yes, seed=seed,
                         params_override=hs.LGB_PARAMS)
            pth_l += ml.predict(Xth, num_iteration=ml.best_iteration) / len(SEEDS)
            pte_l += ml.predict(Xte, num_iteration=ml.best_iteration) / len(SEEDS)
            mx = fit_xgb(Xtr, ytr, Xes, yes, seed, xgb_params)
            k = int(mx.best_iteration)
            pth_x += mx.predict_proba(Xth, iteration_range=(0, k+1))[:, 1] / len(SEEDS)
            pte_x += mx.predict_proba(Xte, iteration_range=(0, k+1))[:, 1] / len(SEEDS)
        all_pl.append(pte_l); all_px.append(pte_x); all_y.append(yte)
        log(f"[{m}] done")

    pl_, px_, y_ = np.concatenate(all_pl), np.concatenate(all_px), np.concatenate(all_y)
    pear = pearsonr(pl_, px_)[0]
    spear = spearmanr(pl_, px_)[0]
    thr_l = np.quantile(np.abs(pl_ - .5), 1 - TPD / 1440)  # 近似：用 test 分布（仅诊断，非实盘）
    thr_x = np.quantile(np.abs(px_ - .5), 1 - TPD / 1440)
    hl = np.abs(pl_ - .5) >= thr_l
    hx = np.abs(px_ - .5) >= thr_x
    inter = hl & hx
    l_only = hl & ~hx
    x_only = hx & ~hl
    union = hl | hx
    jac = inter.sum() / union.sum() if union.sum() else 0
    agree = ((pl_ > .5) == (px_ > .5)).mean()
    log(f"\n=== score correlation: Pearson={pear:.4f} Spearman={spear:.4f} ===")
    log(f"direction agreement (all rows): {agree*100:.2f}%")
    log(f"high-conf hits: LGB={hl.sum()} XGB={hx.sum()} inter={inter.sum()} "
        f"LGB-only={l_only.sum()} XGB-only={x_only.sum()} Jaccard={jac*100:.2f}%")
    for name, mask in (("LGB-only", l_only), ("XGB-only", x_only), ("both", inter)):
        if mask.sum() == 0:
            continue
        wl = ((pl_[mask] > .5).astype(int) == y_[mask]).mean()
        wx = ((px_[mask] > .5).astype(int) == y_[mask]).mean()
        log(f"  {name} n={mask.sum()}: LGB win={wl*100:.2f}% XGB win={wx*100:.2f}%")
    log("DONE")


if __name__ == "__main__":
    main()
