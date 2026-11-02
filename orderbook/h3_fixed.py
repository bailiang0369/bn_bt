"""3min 攻坚 4：固定轮数替代早停（修集成不稳定）。

Run Q 暴露的问题：off=120 那段的全天按天 top15 只有 0.5531，而 5 个 seed 的
best_iteration 是 [137, 21, 2, 89, 55] —— **有 seed 只训了 2 轮就"早停"**，
集成直接被近似随机的模型拖低。根因是 val 只有 30 天，AUC 曲线噪声大。

本脚本对比：固定轮数 60/100/150（不做早停） vs 早停，在 3 个 test 段上按天 top15。

用法：
    python -m orderbook.h3_fixed
"""
import os
import sys
import argparse
from pathlib import Path

import numpy as np
import polars as pl
import lightgbm as lgb

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from orderbook.fusion_test import (
    load_orderbook, day_index, complete_days, daily_topk_eval, acc_at_cov)

REG = dict(objective="binary", metric="auc", learning_rate=0.05, num_leaves=31,
           feature_fraction=0.6, bagging_fraction=0.9, bagging_freq=1,
           min_child_samples=500, lambda_l2=5.0, verbose=-1)


def train_fixed(X, y, tr, va, cols, seeds, rounds):
    probs = np.zeros(len(y), dtype=np.float64)
    for sd in seeds:
        params = dict(REG, seed=sd, bagging_seed=sd,
                      feature_fraction_seed=sd, data_random_seed=sd)
        ds = lgb.Dataset(X[tr], y[tr], feature_name=cols)
        m = lgb.train(params, ds, num_boost_round=rounds)
        probs += m.predict(X)
    return (probs / len(seeds)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description="固定轮数 vs 早停")
    ap.add_argument("--seeds", default=os.environ.get("KAGGLE_SEEDS", "42,43,44"))
    ap.add_argument("--rounds-list", default="60,100,150")
    ap.add_argument("--daily-k", type=int, default=15)
    ap.add_argument("--h", type=int, default=3)
    ap.add_argument("--offsets", default=os.environ.get("KAGGLE_TEST_OFFSETS", "0,60,120"))
    a = ap.parse_args()

    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    rlist = [int(s) for s in a.rounds_list.split(",") if s.strip()]
    offsets = [int(s) for s in a.offsets.split(",") if s.strip()]
    h = a.h
    ob0 = load_orderbook()

    print(f"{'前移':>5}{'轮数':>6}{'按天top15':>11}{'全局1%':>9}{'AUC':>8}")
    best = (-1, None)
    for off in offsets:
        obx = ob0.head(ob0.height - off * 1440) if off > 0 else ob0
        ob = obx.with_columns((pl.col("close").shift(-h) > pl.col("close"))
                              .cast(pl.Int8).alias("target")).filter(pl.col("target").is_not_null())
        L = ob.height
        k1, k2 = int(L * 0.64), int(L * 0.80)
        tr, va, te = slice(0, k1 - h), slice(k1, k2 - h), slice(k2, L)
        ct = ob["close_time"].to_numpy()
        y = ob["target"].to_numpy().astype(np.int8)
        cols = [c for c in ob.columns if c not in ("target", "close_time")]
        X = ob.select(cols).to_numpy().astype(np.float32)
        d_te = day_index(ct[te])
        m_te = complete_days(d_te)
        from sklearn.metrics import roc_auc_score
        for rounds in rlist:
            p = train_fixed(X, y, tr, va, cols, seeds, rounds)
            m, s = float(p[tr].mean()), float(p[tr].std() + 1e-9)
            z = (p - m) / s
            dk = float(daily_topk_eval(z[te], y[te], d_te, m_te, a.daily_k).mean())
            a1, _ = acc_at_cov(z[te], y[te], 0.01)
            auc = float(roc_auc_score(y[te], z[te]))
            print(f"{off:>5}{rounds:>6}{dk:>11.4f}{a1:>9.4f}{auc:>8.4f}")
            if dk > best[0]:
                best = (dk, f"off={off} rounds={rounds}")
    print(f"\n[结论] 最优 {best[1]} 按天top{a.daily_k}={best[0]:.4f}（目标 0.65）")


if __name__ == "__main__":
    main()
