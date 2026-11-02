"""3min 攻坚 2：时段分层 + 3 分钟累积订单流特征。

背景（Run O）：一致性打分/连续加权在 val 与 test 上不一致，属于噪声级改动。
本模块换两个更有信息的方向：
  1) 时段分层：不同时段的盘口信号强度差异很大。用 **val** 统计各时段优势做分数乘子
     （test 只用于报告），把 15 单/天压到强时段。
  2) 3 分钟累积订单流：现有特征多是 1m/5m/15m 的 z-score，缺"正好 3 分钟"的累积量；
     补 rolling 3 的均值/求和/差分（obi、主动买失衡、冲击、价格位移）。

所有评估都是"按天 top15"（日均 15 单口径）。

用法：
    python -m orderbook.h3_time
"""
import os
import sys
import argparse
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from orderbook.fusion_test import (
    load_orderbook, add_derived_features, train_lgb_avg, acc_at_cov,
    day_index, complete_days, daily_topk_eval)

REG = dict(num_leaves=31, min_child_samples=500, feature_fraction=0.6, lambda_l2=5.0)
CUM_COLS = ["ob_f_obi_total", "atk_f_imb", "ob_f_impact_imb",
            "ob_f_ret_1m_bps", "atk_f_def_gap"]


def add_cum3(df):
    """补"正好 3 分钟"的累积量/斜率。"""
    exprs, names = [], []
    for c in CUM_COLS:
        if c not in df.columns:
            continue
        exprs += [pl.col(c).rolling_sum(3).alias(f"cum3_{c}_sum"),
                  pl.col(c).rolling_mean(3).alias(f"cum3_{c}_mean"),
                  (pl.col(c) - pl.col(c).shift(3)).alias(f"cum3_{c}_d3")]
        names += [f"cum3_{c}_sum", f"cum3_{c}_mean", f"cum3_{c}_d3"]
    out = df.with_columns(exprs) if exprs else df
    print(f"[cum3] +{len(names)} 列: {names}")
    return out, names


def hour_of(ct):
    return ((ct % 86_400_000) // 3_600_000).astype(int)


def seg_of(hour):
    return (hour // 6).astype(int)   # 4 段，每段 6 小时（UTC）


def eval_cov(score, y, day_id, mask, k):
    return float(daily_topk_eval(score, y, day_id, mask, k).mean())


def main():
    ap = argparse.ArgumentParser(description="3min 时段分层 + 累积特征")
    ap.add_argument("--seeds", default=os.environ.get("KAGGLE_SEEDS", "42,43,44,45,46"))
    ap.add_argument("--rounds", type=int, default=1000)
    ap.add_argument("--daily-k", type=int, default=15)
    ap.add_argument("--h", type=int, default=3)
    a = ap.parse_args()

    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    h = a.h

    ob = load_orderbook()
    ob = ob.with_columns((pl.col("close").shift(-h) > pl.col("close"))
                         .cast(pl.Int8).alias("target"))
    ob = ob.filter(pl.col("target").is_not_null())
    L = ob.height
    k1, k2 = int(L * 0.64), int(L * 0.80)
    tr, va, te = slice(0, k1 - h), slice(k1, k2 - h), slice(k2, L)
    ct = ob["close_time"].to_numpy()
    y = ob["target"].to_numpy().astype(np.int8)
    print(f"[ob] rows={L} h={h} train[0:{k1 - h}] val[{k1}:{k2 - h}] test[{k2}:{L}]")

    obd = add_derived_features(ob)
    obc, cum_cols = add_cum3(obd)
    cols = [c for c in obc.columns if c not in ("target", "close_time")]
    X = obc.select(cols).to_numpy().astype(np.float32)
    print(f"[feats] total={len(cols)}")

    print(f"[train] {len(seeds)} seeds, reg")
    p, iters = train_lgb_avg(X, y, tr, va, cols, seeds, a.rounds, None, REG)
    m, s = float(p[tr].mean()), float(p[tr].std() + 1e-9)
    z = (p - m) / s
    print(f"[train] iters={iters}")

    dh_t = day_index(ct[te])
    mt = complete_days(dh_t)
    dv = day_index(ct[va])
    mv = complete_days(dv)
    hour_t = hour_of(ct[te])
    seg_t = seg_of(hour_t)

    base_te = eval_cov(z[te], y[te], dh_t, mt, a.daily_k)
    print(f"\n[base] test 按天top{a.daily_k} = {base_te:.4f}")

    # ---- 时段诊断：test 上按天 top-k 选中样本的准确率（分时段）----
    S, Y, D = z[te], y[te], dh_t
    sel = np.zeros(len(S), dtype=bool)
    for u in np.unique(D[mt]):
        msk = D == u
        kk = min(a.daily_k, int(msk.sum()))
        idx = np.flatnonzero(msk)[np.argsort(-np.abs(S[msk]))[:kk]]
        sel[idx] = True
    print(f"{'时段(UTC)':<12}{'选中单数':>9}{'准确率':>9}")
    for s0 in range(4):
        msk = sel & (seg_t == s0)
        if msk.sum() > 0:
            acc = float(((S[msk] > 0).astype(int) == Y[msk]).mean())
            print(f"{s0 * 6:02d}-{s0 * 6 + 6:02d}{'':<6}{int(msk.sum()):>9}{acc:>9.4f}")

    # ---- 用 val 统计的时段优势做乘子（test 不参与） ----
    hour_v = hour_of(ct[va])
    seg_v = seg_of(hour_v)
    w = np.ones(4)
    for s0 in range(4):
        msk = seg_v == s0
        if msk.sum() > 200:
            acc0 = float(((z[va][msk] > 0).astype(int) == y[va][msk]).mean())
            w[s0] = float(np.clip(1.0 + (acc0 - 0.5) / 0.10, 0.7, 1.3))
            print(f"[val] 时段 {s0 * 6:02d}-{s0 * 6 + 6:02d} acc={acc0:.4f} → 权重 {w[s0]:.2f}")
    z_w = z[te] * w[seg_t]
    t_w = eval_cov(z_w, y[te], dh_t, mt, a.daily_k)
    print(f"[时段加权] test 按天top{a.daily_k} = {t_w:.4f}（base {base_te:.4f}，{t_w - base_te:+.4f}）")

    auc = roc_auc_score(y[te], z[te])
    a1, _ = acc_at_cov(z[te], y[te], 0.01)
    print(f"\n[AUC] {auc:.4f}  全局1%={a1:.4f}")
    print(f"[结论] base={base_te:.4f} 时段加权={t_w:.4f}（目标 0.65）")


if __name__ == "__main__":
    main()
