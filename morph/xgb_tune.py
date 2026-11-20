#!/usr/bin/env python3
"""XGBoost 随机参数搜索（多参数组合，非单参数档位）。

在调参窗 2026-01~03 上评估 30 组随机参数，按 test AUC + tpd8 尾部胜率排序。
每组：单 seed 42、ROUNDS=1000、train/es/thr 三段切、阈值只在 thr 段定（因果）。
评估口径与项目现役完全一致；唯一变量是 XGB 超参组合。

输出 top 10 参数组合（按 AUC 和按尾部胜率各一组），供后续全量验证。
"""
from __future__ import annotations

import datetime as dt
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from morph import ob_kline as ok
from morph import h10_signal as hs
from morph.two_legs import add_target, matrix

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

N_TRIALS = int(os.environ.get("N_TRIALS", "30"))
SEED = 42
TPD = 8
ROUNDS = 1000
T = time.time()

PARAM_SPACE = {
    "eta": [0.01, 0.02, 0.03, 0.05, 0.08, 0.1],
    "max_depth": [3, 4, 5, 6, 7, 8],
    "min_child_weight": [1, 5, 10, 30, 50, 100, 200],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "reg_lambda": [0, 1, 5, 10, 20, 50, 100],
}


def log(msg: str) -> None:
    print(f"[{time.time() - T:7.1f}s] {msg}", flush=True)


def main() -> None:
    # 数据
    a = ok._assets(ok.KLINE_TAG)
    names = [f"BTCUSDT_1m_klines_{y}.parquet" for y in range(2020, 2027)]
    k = pl.concat([pl.read_parquet(ok.fetch(ok.KLINE_TAG, n),
                                   columns=["close_time", "open", "high", "low", "close",
                                            "um_buy_vol", "um_sell_vol", "um_count"])
                   for n in names if n in a])
    # 复用 h10_signal 的特征构造（build + flow）
    raw = k.clone()
    df, feats = hs.kl.build(raw)
    df, feats = hs.add_flow(df, feats)
    df = df.with_columns(pl.col("close_time").cast(pl.Datetime("ms")).dt.strftime("%Y-%m").alias("ym"))
    tagged = add_target(df, 10).filter(pl.col("target").is_not_null()).sort("close_time")
    log(f"tagged {tagged.height:,} rows, {len(feats)} feats")

    months = ["2026-01", "2026-02", "2026-03"]
    folds = []
    for m in months:
        t0 = int(dt.datetime.strptime(m + "-01", "%Y-%m-%d").replace(
            tzinfo=dt.timezone.utc).timestamp() * 1000)
        lo = t0 - 1460 * 86_400_000
        pre = tagged.filter((pl.col("close_time") < t0) & (pl.col("close_time") >= lo))
        te = tagged.filter(pl.col("ym") == m)
        n = pre.height
        c1, c2 = int(n * 0.85), int(n * 0.95)
        tr, es, thr = pre[:c1], pre[c1:c2], pre[c2:]
        folds.append(dict(
            Xtr=matrix(tr, feats), ytr=tr["target"].to_numpy().astype(int),
            Xes=matrix(es, feats), yes=es["target"].to_numpy().astype(int),
            Xth=matrix(thr, feats),
            Xte=matrix(te, feats), yte=te["target"].to_numpy().astype(np.int8)))
    log(f"built {len(folds)} folds")

    rng = random.Random(20261120)
    trials = []
    for i in range(N_TRIALS):
        p = {k: rng.choice(v) for k, v in PARAM_SPACE.items()}
        aucs, wins = [], []
        for f in folds:
            mdl = XGBClassifier(objective="binary:logistic", eval_metric="auc",
                                tree_method="hist", n_estimators=ROUNDS,
                                early_stopping_rounds=50, random_state=SEED,
                                n_jobs=os.cpu_count() or 4, verbosity=0, **p)
            mdl.fit(f["Xtr"], f["ytr"], eval_set=[(f["Xes"], f["yes"])], verbose=False)
            k_best = int(mdl.best_iteration)
            pth = mdl.predict_proba(f["Xth"], iteration_range=(0, k_best + 1))[:, 1]
            pte = mdl.predict_proba(f["Xte"], iteration_range=(0, k_best + 1))[:, 1]
            aucs.append(roc_auc_score(f["yte"], pte))
            thr_v = float(np.quantile(np.abs(pth - .5), 1 - TPD / 1440))
            hit = np.abs(pte - .5) >= thr_v
            if hit.sum() >= 30:
                wins.append(float(((pte[hit] - .5 > 0).astype(int) == f["yte"][hit]).mean()))
        auc, win = float(np.mean(aucs)), float(np.mean(wins))
        trials.append(dict(params=p, auc=round(auc, 5), win=round(win, 5),
                           score=round(auc + win, 5)))
        if (i + 1) % 5 == 0:
            log(f"  trial {i+1}/{N_TRIALS}: auc={auc:.4f} win={win*100:.2f}% best_so_far={max(t['score'] for t in trials):.4f}")

    trials.sort(key=lambda t: t["score"], reverse=True)
    by_auc = sorted(trials, key=lambda t: t["auc"], reverse=True)[:10]
    by_win = sorted(trials, key=lambda t: t["win"], reverse=True)[:10]
    print("\n=== TOP 10 by AUC ===")
    for t in by_auc:
        print(json.dumps(t, ensure_ascii=False))
    print("\n=== TOP 10 by tail win@8 ===")
    for t in by_win:
        print(json.dumps(t, ensure_ascii=False))
    print(f"\nBEST overall: {json.dumps(trials[0], ensure_ascii=False)}")
    log("DONE")


if __name__ == "__main__":
    main()
