"""盘口腿（生产腿）上的 regime 分裂专模 A/B —— 借鉴项最终裁决。

背景：`morph/borrow_ab.py` 在 K 线腿上测出，外部参考（termbin f8mx）的可借鉴项里
只有「按自相关/布林扩张分流训两个专模」略为正（static +0.65pp / roll +1.05pp，
P(Δ>0)=0.81，CI 跨 0），其余（方向步进包、极浅树）都是噪声。
K 线腿不是生产腿；本脚本在**盘口腿**（150 列微观特征，roll_1d 0.6167 / roll_3d 0.6229）
上做同一实验，判据不变：

  base    : 现有 150 列 + OB_REG + 60 轮 + 5 seeds（对照，应复现 0.6167 / 0.6229）
  auto    : +1m 收益 15/60 阶自相关 + 布林扩张比（3 列）
  regime  : 仅训练方式变（base 特征按 autocorr_15>0.05 或 bbexp>1.15 分流训两个专模再路由）
  rauto   : regime + auto 特征

口径与 znorm_eval 完全一致（64/80 切分、off=0/60/120、static_val/roll_1d/roll_3d、
K=15×天数），不引入任何新口径，方便与 §28 的数字直接对比。
"""
import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lightgbm as lgb  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from orderbook.znorm_eval import (BASE, DAILY_K, H, OB_REG, complete_days,  # noqa: E402
                                 daily_topk_eval, day_index, global_topk_eval,
                                 load, log, rolling_z)

DROP = ("target", "close_time")


def add_state(df):
    """只用 close 的因果状态量：1m 收益一阶自相关（15/60）+ 布林宽度扩张比。"""
    c = pl.col("close")
    r = c / c.shift(1) - 1.0
    rs = r.shift(1)
    ex = []
    for w in (15, 60):
        cov = (r * rs).rolling_mean(w) - r.rolling_mean(w) * rs.rolling_mean(w)
        sd = r.rolling_std(w) * rs.rolling_std(w) + 1e-12
        ex.append((cov / sd).clip(-1.0, 1.0).cast(pl.Float32).alias(f"obx_ac{w}"))
    ma, sd20 = c.rolling_mean(20), c.rolling_std(20)
    width = 4.0 * sd20 / (ma + 1e-9)
    ex.append((width / (width.rolling_mean(60) + 1e-9)).clip(0.0, 10.0)
              .cast(pl.Float32).alias("obx_bbexp"))
    return df.with_columns(ex)


def fit(X, y, tr, cols, seeds, rounds):
    p = np.zeros(X.shape[0])
    for sd in seeds:
        q = dict(BASE, **OB_REG, seed=sd, bagging_seed=sd,
                 feature_fraction_seed=sd, data_random_seed=sd)
        p += lgb.train(q, lgb.Dataset(X[tr], y[tr], feature_name=cols),
                       num_boost_round=rounds).predict(X)
    return (p / len(seeds)).astype(np.float32)


def main():
    seeds = [int(x) for x in os.environ.get("SEEDS", "42,43,44,45,46").split(",")]
    rounds = int(os.environ.get("ROUNDS", "60"))
    offs = [int(x) for x in os.environ.get("OFFS", "0,60,120").split(",")]
    variants = [x.strip() for x in os.environ.get(
        "VARIANTS", "base,auto,regime,rauto").split(",") if x.strip()]

    df = add_state(load())
    ct = df["close_time"].to_numpy()
    y = df["target"].to_numpy().astype(np.int8)
    base_cols = [c for c in df.columns if c not in DROP and not c.startswith("obx_")]
    auto_cols = [c for c in df.columns if c.startswith("obx_")]
    cols = base_cols + auto_cols
    X = df.select(cols).to_numpy().astype(np.float32)
    del df
    L = X.shape[0]
    k1_0, k2_0 = int(L * 0.64), int(L * 0.80)
    ac15 = X[:, cols.index("obx_ac15")]
    bbexp = X[:, cols.index("obx_bbexp")]
    is_trend = (ac15 > 0.05) | (bbexp > 1.15)
    log(f"[data] rows={L} feats={len(cols)} (base {len(base_cols)} + state {len(auto_cols)}) "
        f"trend 占比={is_trend.mean():.3f}")

    rows = {}
    for off in offs:
        k1, k2 = k1_0 - off * 1440, k2_0 - off * 1440
        test_start = int(ct[k2])
        tr_m = ct < int(ct[k1])
        va_m = (ct >= int(ct[k1])) & (ct < test_start)
        te_m = ct >= test_start
        tr = np.where(tr_m)[0]
        tr = tr[tr + H < int(ct[k2])]
        te_idx = np.where(te_m)[0]
        dte = day_index(ct[te_m])
        mt = complete_days(dte, 1200)
        log(f"\n===== off={off} train={len(tr)} val={int(va_m.sum())} "
            f"test={int(te_m.sum())} 天={len(np.unique(dte))} 基准={y[te_m].mean():.4f} =====")

        def report(tag, s):
            zs = (s - s[va_m].mean()) / (s[va_m].std() + 1e-9)
            r = {"auc": float(roc_auc_score(y[te_m], zs[te_m])),
                 "static": global_topk_eval(zs[te_m], y[te_m], dte, DAILY_K),
                 "roll1d": global_topk_eval(rolling_z(s, 1440)[te_m], y[te_m], dte, DAILY_K),
                 "roll3d": global_topk_eval(rolling_z(s, 4320)[te_m], y[te_m], dte, DAILY_K),
                 "daily": float(daily_topk_eval(zs[te_m], y[te_m], dte, mt, DAILY_K).mean())}
            rows[(off, tag)] = r
            log(f"[{tag:<7}] static={r['static']:.4f} roll1d={r['roll1d']:.4f} "
                f"roll3d={r['roll3d']:.4f} AUC={r['auc']:.4f} 按天={r['daily']:.4f}")

        def routed(sub_cols):
            idx = [cols.index(c) for c in sub_cols]
            s = np.zeros(L)
            for mask, n in ((is_trend, int(is_trend[tr].sum())), (~is_trend, int((~is_trend)[tr].sum()))):
                trs = tr[mask[tr]]
                if n < 20000:
                    log(f"    [regime] 子集仅 {n} 行，整体回退")
                    trs = tr
                s += np.where(mask, fit(X[:, idx], y, trs, sub_cols, seeds, rounds), 0.0)
            return s.astype(np.float32)

        bidx = [cols.index(c) for c in base_cols]
        if "base" in variants:
            report("base", fit(X[:, bidx], y, tr, base_cols, seeds, rounds))
        if "auto" in variants:
            report("auto", fit(X, y, tr, cols, seeds, rounds))
        if "regime" in variants:
            report("regime", routed(base_cols))
        if "rauto" in variants:
            report("rauto", routed(cols))

    log(f"\n{'变体':<8}{'off':>5}{'static':>9}{'roll1d':>9}{'roll3d':>9}{'AUC':>8}{'按天':>8}")
    for (off, tag), r in rows.items():
        log(f"{tag:<8}{off:>5}{r['static']:>9.4f}{r['roll1d']:>9.4f}"
            f"{r['roll3d']:>9.4f}{r['auc']:>8.4f}{r['daily']:>8.4f}")
    log(f"\n{'变体':<8}{'roll1d均值':>11}{'roll3d均值':>11}{'roll3d最差':>11}{'AUC均值':>9}{'Δroll1d':>10}")
    b1 = np.mean([rows[(o, "base")]["roll1d"] for o in offs if (o, "base") in rows])
    for tag in variants:
        rs = [rows[(o, tag)] for o in offs if (o, tag) in rows]
        if not rs:
            continue
        m1 = float(np.mean([r["roll1d"] for r in rs]))
        m3 = float(np.mean([r["roll3d"] for r in rs]))
        w3 = float(np.min([r["roll3d"] for r in rs]))
        ma = float(np.mean([r["auc"] for r in rs]))
        if tag == "base":
            b1 = m1
        log(f"{tag:<8}{m1:>11.4f}{m3:>11.4f}{w3:>11.4f}{ma:>9.4f}"
            f"{(m1-b1)*100:>+9.2f}pp")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
