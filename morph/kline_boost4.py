"""第四轮：评估端的中心偏移调优（不重训模型）。

前三轮证明特征/集成都到顶了：63 特征 LGB、5~8 seed 平均，static topK ≈ 0.61，
AUC 卡在 0.527。而 topK 的定义里有一个从未调过的自由参数：

    score = z - mu，取 |score| 最大的 K 个

mu 一直被当成 0（即 val 均值）。但"哪些样本值得下注"本质是一个选择性偏差问题：
如果概率分布偏斜，mu 前移/后移会改变被选中的集合；单边（只做多/只做空）也可能
优于对称。mu 与方向都在 val 上选，test 只做评估，不存在泄漏。

同时报告 test 内的 oracle 上界，用来看这条路还能榨出多少。
"""
import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lightgbm as lgb  # noqa: E402

from morph.kline_boost import (DAILY_K, OB_END, PARAMS, TEST_START, VAL_START,  # noqa: E402
                               build_features, day_index, load_klines, log, zscore)

MODES = ("abs", "long", "short")


def pick_score(z, mu, mode):
    s = z - mu
    if mode == "abs":
        return np.abs(s)
    return s if mode == "long" else -s


def acc_at(z, y, sel, k):
    idx = np.argsort(-sel)[:min(k, len(sel))]
    return float(((z[idx] > 0).astype(np.int8) == y[idx]).mean())


def daily_acc(z, y, day, sel, k):
    out = []
    for d in np.unique(day):
        m = day == d
        out.append(acc_at(z[m], y[m], sel[m], k))
    return np.array(out)


def main():
    seeds = [int(x) for x in os.environ.get("SEEDS", "42,43,44,45,46").split(",")]
    rounds = int(os.environ.get("ROUNDS", "60"))

    raw = load_klines().filter(pl.col("close_time") <= OB_END)
    d = (raw.with_columns(pl.col("close").shift(-3).alias("_fc"))
         .with_columns((pl.col("_fc") > pl.col("close")).cast(pl.Int8).alias("target"))
         .drop("_fc").filter(pl.col("target").is_not_null()))
    ft, cols = build_features(d)
    ct = ft["close_time"].to_numpy()
    y = ft["target"].to_numpy().astype(np.int8)
    X = ft.select(cols).to_numpy().astype(np.float32)
    del ft, d

    tr = np.where(ct < VAL_START)[0]
    va = np.where((ct >= VAL_START) & (ct < TEST_START))[0]
    tr = tr[tr + 3 < int(va[0])]
    te = np.where(ct >= TEST_START)[0]
    log(f"[split] train={len(tr)} val={len(va)} test={len(te)}")

    acc = np.zeros(len(y))
    for sd in seeds:
        p = dict(PARAMS, seed=sd, bagging_seed=sd, feature_fraction_seed=sd,
                 data_random_seed=sd)
        m = lgb.train(p, lgb.Dataset(X[tr], y[tr], feature_name=cols),
                      num_boost_round=rounds)
        acc += m.predict(X)
    prob = acc / len(seeds)
    z = zscore(prob, prob[va])
    del X

    dva, dte = day_index(ct[va]), day_index(ct[te])
    kva, kte = DAILY_K * len(np.unique(dva)), DAILY_K * len(np.unique(dte))

    base_va = acc_at(z[va], y[va], pick_score(z[va], 0.0, "abs"), kva)
    base_te = acc_at(z[te], y[te], pick_score(z[te], 0.0, "abs"), kte)
    log(f"[base] mu=0 abs: val={base_va:.4f} test={base_te:.4f}")

    best = (-1.0, 0.0, "abs")
    grid = np.linspace(0.05, 0.95, 91)
    for q in grid:
        mu = float(np.quantile(z[va], q))
        for mode in MODES:
            a = acc_at(z[va], y[va], pick_score(z[va], mu, mode), kva)
            if a > best[0]:
                best = (a, mu, mode)
    a_va, mu, mode = best
    sel_te = pick_score(z[te], mu, mode)
    a_te = acc_at(z[te], y[te], sel_te, kte)
    log(f"[tuned] val 最优 q={grid[int(np.argmin(np.abs(np.quantile(z[va], grid) - mu)))]:.2f} "
        f"mu={mu:+.4f} mode={mode} -> val={a_va:.4f} test={a_te:.4f}")

    for mode2 in MODES:
        for mu2, tag in ((0.0, "mu0"), (mu, "mu*")):
            s = pick_score(z[te], mu2, mode2)
            log(f"  test {mode2:<5} {tag:<4} global={acc_at(z[te], y[te], s, kte):.4f} "
                f"按天={daily_acc(z[te], y[te], dte, s, DAILY_K).mean():.4f}")

    orc = (-1.0, 0.0, "abs")
    for mu2 in np.linspace(np.quantile(z[te], 0.05), np.quantile(z[te], 0.95), 91):
        for mode2 in MODES:
            a = acc_at(z[te], y[te], pick_score(z[te], mu2, mode2), kte)
            if a > orc[0]:
                orc = (a, float(mu2), mode2)
    log(f"[oracle] test 内最优 mu={orc[1]:+.4f} mode={orc[2]} -> {orc[0]:.4f} "
        f"(上界，实际不可用)")

    te_sel = pick_score(z[te], mu, mode)
    per_day = daily_acc(z[te], y[te], dte, te_sel, DAILY_K)
    log(f"[final] val选参 mu={mu:+.4f} {mode} | test global={acc_at(z[te], y[te], te_sel, kte):.4f} "
        f"按天={per_day.mean():.4f}±{per_day.std(ddof=1):.4f} 天={len(per_day)}")
    log(f"[final] 对照 mu=0 abs | test global={base_te:.4f} "
        f"按天={daily_acc(z[te], y[te], dte, pick_score(z[te], 0.0, 'abs'), DAILY_K).mean():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
