"""口径对照：把今天的模型（single / wf365）在 static_val 与 roll_1d/3d 下同时评一遍。

动机
----
用户记得的 0.64~0.65 出自 `§16/§27` 的 **static_val（全局 top-K，K=15×天数）** 口径；
`§28` 换成滚动标准化后同一组预测只有 0.6167，并给出证据：静态 top-K 只覆盖 60/96 天、
单天最多 180 单（目标 15/天）。两边都没错，是**口径不同**。

本脚本把两种口径放在同一张表里，并加上"覆盖天 / 单天最多"两列自证：
  model ∈ {single(单次训练), wf365(1 年窗 30 天重训)}，off=0，5 seeds，K=15×天数
输出：
  每种口径的 topK、按天 top15、覆盖天数、单天下单峰值、AUC
"""
import os
import sys
import time

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lightgbm as lgb  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from orderbook.znorm_eval import (BASE, DAILY_K, H, OB_REG, complete_days,  # noqa: E402
                                 daily_topk_eval, day_index, global_topk_eval,
                                 load, log, rolling_z)

DAY = 86_400_000
BLOCK = 30


def fit(X, y, tr, cols, pred, seeds, rounds):
    acc = np.zeros(len(pred))
    for sd in seeds:
        q = dict(BASE, **OB_REG, seed=sd, bagging_seed=sd, feature_fraction_seed=sd,
                 data_random_seed=sd)
        m = lgb.train(q, lgb.Dataset(X[tr], y[tr], feature_name=cols), num_boost_round=rounds)
        acc += m.predict(X[pred])
    return (acc / len(seeds)).astype(np.float32)


def fit_es(X, y, tr, va_idx, cols, pred, seeds):
    """与 znorm_eval.py 一致：val 上 early stopping（历史 0.6410 的来源配置）。"""
    tr = tr[tr + H < int(va_idx[0])]
    acc = np.zeros(len(pred))
    iters = []
    for sd in seeds:
        q = dict(BASE, **OB_REG, seed=sd, bagging_seed=sd, feature_fraction_seed=sd,
                 data_random_seed=sd)
        m = lgb.train(q, lgb.Dataset(X[tr], y[tr], feature_name=cols), num_boost_round=600,
                      valid_sets=[lgb.Dataset(X[va_idx], y[va_idx], feature_name=cols)],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        iters.append(int(m.best_iteration or 600))
        acc += m.predict(X[pred], num_iteration=m.best_iteration)
    return (acc / len(seeds)).astype(np.float32), iters


def detail(z, y, dd, k_per_day):
    """返回 topK 准确率、覆盖天数、单天最大下单数。"""
    n_days = len(np.unique(dd))
    K = min(int(k_per_day * n_days), len(y))
    idx = np.argsort(-np.abs(z))[:K]
    acc = float(((z[idx] > 0).astype(int) == y[idx]).mean())
    dd_sel = dd[idx]
    cover = len(np.unique(dd_sel))
    peak = int(np.bincount(dd_sel - dd_sel.min()).max()) if len(dd_sel) else 0
    return acc, cover, n_days, peak


def main():
    offs = [int(x) for x in os.environ.get("OFFS", "0").split(",")]
    seeds = [int(x) for x in os.environ.get("SEEDS", "42,43,44,45,46").split(",")]
    rounds = int(os.environ.get("ROUNDS", "60"))
    wf_win = int(os.environ.get("WF_WIN", "365"))
    t0 = time.time()

    df = load()
    ct = df["close_time"].to_numpy()
    y = df["target"].to_numpy().astype(np.int8)
    cols = [c for c in df.columns if c not in ("target", "close_time")]
    X = df.select(cols).to_numpy().astype(np.float32)
    del df
    L = X.shape[0]
    k1_0, k2_0 = int(L * 0.64), int(L * 0.80)

    for off in offs:
        k1, k2 = k1_0 - off * 1440, k2_0 - off * 1440
        val_start, test_start = int(ct[k1]), int(ct[k2])
        va_m = (ct >= val_start) & (ct < test_start)
        te_m = ct >= test_start
        te_idx = np.where(te_m)[0]
        dd = day_index(ct[te_m])
        mt = complete_days(dd, 1200)

        s_single = np.zeros(L)
        s_single[te_idx] = fit(X, y, np.arange(0, k1 - H), cols, te_idx, seeds, rounds)
        log(f"[off={off}] single 完成 {time.time()-t0:.0f}s")

        s_wf = np.zeros(L)
        if os.environ.get("WF", "1") == "1":
            edges = [int(ct[te_idx[0]])]
            while edges[-1] < int(ct[te_idx[-1]]):
                edges.append(edges[-1] + BLOCK * DAY)
            for b0, b1 in zip(edges[:-1], edges[1:]):
                pred = np.where((ct >= b0) & (ct < b1))[0]
                if len(pred) < 100:
                    continue
                lo = int(ct[pred[0]]) - wf_win * DAY
                trb = np.where((ct >= lo) & (ct < int(ct[pred[0]])))[0]
                trb = trb[trb + H < int(pred[0])]
                if len(trb) < 50000:
                    continue
                s_wf[pred] = fit(X, y, trb, cols, pred, seeds, rounds)
            log(f"[off={off}] wf{wf_win} 完成 {time.time()-t0:.0f}s")

        log(f"\n{'模型':<9}{'口径':<11}{'topK':>8}{'按天':>8}{'覆盖天':>8}{'单天峰值':>9}{'AUC':>8}")
        models = [("single", s_single), ("wf365", s_wf)]
        if os.environ.get("ES", "1") == "1":
            s_es, iters = fit_es(X, y, np.arange(0, k1 - H), np.where(va_m)[0], cols,
                                 te_idx, seeds)
            models.append(("earlystop", s_es))
            log(f"[off={off}] earlystop 完成 {time.time()-t0:.0f}s best_iter={iters}")
        for name, s in models:
            for tag, z in (("static_val", (s - s[va_m].mean()) / (s[va_m].std() + 1e-9)),
                           ("roll_1d", rolling_z(s, 1440)),
                           ("roll_3d", rolling_z(s, 4320))):
                acc, cover, nd, peak = detail(z[te_m], y[te_m], dd, DAILY_K)
                daily = float(daily_topk_eval(z[te_m], y[te_m], dd, mt, DAILY_K).mean())
                auc = float(roc_auc_score(y[te_m], z[te_m]))
                log(f"{name:<9}{tag:<11}{acc:>8.4f}{daily:>8.4f}{cover:>5}/{nd:<3}{peak:>9}{auc:>8.4f}")
    log(f"[done] {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
