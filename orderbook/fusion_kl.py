"""K 线腿 × 盘口腿 融合（3min，按天 15 单）。

两条腿用**不同的训练数据集**（这正是用户指出的关键）：
  - K 线腿：2020-01 → val_start（5.5 年，3.5M 行，63 个量价特征）→ 样本量是盘口的 5 倍；
  - 盘口腿：2025-07 → val_start（15 个月，658k 行，150 个盘口特征）；
  - 两腿都在 orderbook 的 val/test 时间边界上评估（按 close_time 精确 join）。

单腿水平（Run S / Run K）：K 线 0.6087、盘口 0.608~0.615，量级相同但信号源不同，融合有空间。

用法：
    python -m orderbook.fusion_kl
"""
import os
import sys
import argparse
from pathlib import Path

import numpy as np
import polars as pl
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from morph.kline_lgb import load_klines, PARAMS as KL_PARAMS
from morph.kline_features import build_features
from orderbook.fusion_test import (
    load_orderbook, add_target, train_lgb_avg, acc_at_cov,
    day_index, complete_days, daily_topk_eval)

OB_REG = dict(num_leaves=31, min_child_samples=500, feature_fraction=0.6, lambda_l2=5.0)
WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]


def train_predict(X, y, tr, cols, seeds, rounds, pred_idx, params, weight=None):
    out = np.zeros(len(pred_idx), dtype=np.float64)
    for sd in seeds:
        p = dict(params, seed=sd, bagging_seed=sd,
                 feature_fraction_seed=sd, data_random_seed=sd)
        ds = lgb.Dataset(X[tr], y[tr], feature_name=cols,
                         weight=None if weight is None else weight[tr])
        m = lgb.train(p, ds, num_boost_round=rounds)
        out += m.predict(X[pred_idx])
    return (out / len(seeds)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description="K线 × 盘口 融合（3min）")
    ap.add_argument("--h", type=int, default=int(os.environ.get("KAGGLE_HORIZON", "3")))
    ap.add_argument("--kline-seeds", default=os.environ.get("KAGGLE_KSEEDS", "42,43,44"))
    ap.add_argument("--ob-seeds", default=os.environ.get("KAGGLE_OSEEDS", "42,43,44,45,46"))
    ap.add_argument("--kline-stride", type=int, default=2)
    ap.add_argument("--rounds", type=int, default=60)
    ap.add_argument("--daily-k", type=int, default=15)
    ap.add_argument("--save-pred", default=os.environ.get("KAGGLE_FUSION_PRED"))
    a = ap.parse_args()

    h = a.h
    kseeds = [int(s) for s in a.kline_seeds.split(",") if s.strip()]
    oseeds = [int(s) for s in a.ob_seeds.split(",") if s.strip()]

    # ---------- 1) 盘口腿：它决定统一的时间边界 ----------
    ob = add_target(load_orderbook(), h).filter(pl.col("target").is_not_null())
    L = len(ob)
    k1, k2 = int(L * 0.64), int(L * 0.80)
    ob_ct = ob["close_time"].to_numpy()
    val_start, test_start, ob_end = int(ob_ct[k1]), int(ob_ct[k2]), int(ob_ct[-1])
    print(f"[ob] rows={L} val_start={val_start} test_start={test_start} end={ob_end}")

    ob_cols = [c for c in ob.columns if c not in ("target", "close_time")]
    Xo = ob.select(ob_cols).to_numpy().astype(np.float32)
    yo = ob["target"].to_numpy().astype(np.int8)
    tro, vao, teo = slice(0, k1 - h), slice(k1, k2 - h), slice(k2, L)
    p_ob, ob_iters = train_lgb_avg(Xo, yo, tro, vao, ob_cols, oseeds, a.rounds, None, OB_REG)
    print(f"[ob] 训练完成 iters={ob_iters}")

    # ---------- 2) K 线腿：2020 → val_start ----------
    # 取到 2026-10，避免 K 线末尾比 orderbook 早 3 周导致 join 丢样本（缺月会自动跳过）
    kl = load_klines(end=(2026, 10))
    kl = kl.filter(pl.col("close_time") <= ob_end)
    kl = add_target(kl, h).filter(pl.col("target").is_not_null())
    print(f"[kl] 时间 {int(kl['close_time'][0])} ~ {int(kl['close_time'][-1])} "
          f"（orderbook 末尾 {ob_end}）")
    kf, kcols = build_features(kl)
    Xk = kf.select(kcols).to_numpy().astype(np.float32)
    yk = kf["target"].to_numpy().astype(np.int8)
    kct = kf["close_time"].to_numpy()
    print(f"[kl] rows={len(yk)} feats={len(kcols)}")

    trk = np.where(kct + h * 60_000 < val_start)[0][::a.kline_stride]
    vak = np.where((kct >= val_start) & (kct + h * 60_000 < test_start))[0]
    pred_idx = np.where(kct >= val_start)[0]
    print(f"[kl] train={len(trk)} val={len(vak)} pred={len(pred_idx)}")
    p_kl = train_predict(Xk, yk, trk, kcols, kseeds, a.rounds, pred_idx, KL_PARAMS)
    kct_pred = kct[pred_idx]

    # ---------- 3) 按 close_time 精确 join（K 线 +1ms 对齐盘口） ----------
    merged = (pl.DataFrame({"close_time": ob_ct, "target": yo, "p_ob": p_ob})
              .join(pl.DataFrame({"close_time": kct_pred + 1, "p_kl": p_kl}),
                    on="close_time", how="inner")
              .sort("close_time"))
    print(f"[join] 交集行数={merged.height}（盘口 {L} 行）")
    mct = merged["close_time"].to_numpy()
    yy = merged["target"].to_numpy().astype(np.int8)
    zo = merged["p_ob"].to_numpy()
    zk = merged["p_kl"].to_numpy()

    dyn = day_index(mct)
    m_val = (mct >= val_start) & (mct < test_start)
    m_te = (mct >= test_start)
    dv, dte = dyn[m_val], dyn[m_te]
    mv, mt = complete_days(dv), complete_days(dte)

    def zs(x, ref):
        return (x - x[ref].mean()) / (x[ref].std() + 1e-9)

    zo_s, zk_s = zs(zo, m_val), zs(zk, m_val)

    def dacc(s, m, cm):
        """m=区间掩码（全长），cm=该区间内的完整天掩码（已按 m 取子集）。"""
        idx = np.where(m)[0]
        return float(daily_topk_eval(s[idx], yy[idx], dyn[idx], cm, a.daily_k).mean())

    # ---------- 4) 全权重网格：val 与 test 同时打（历史教训：val 只有 30 天，选出的 w 不稳）----------
    print(f"\n[权重网格] 按天 top{a.daily_k}（val {len(np.unique(dv[mv]))} 天 / "
          f"test {len(np.unique(dte[mt]))} 天）")
    print(f"{'w':>7}{'val':>10}{'test':>10}")
    grid = []
    for w in WEIGHTS:
        s = zo_s + w * zk_s
        av, at = dacc(s, m_val, mv), dacc(s, m_te, mt)
        grid.append((w, av, at))
        print(f"{w:>7.2f}{av:>10.4f}{at:>10.4f}")
    best_w = max(grid, key=lambda g: g[1])[0]
    test_best_w = max(grid, key=lambda g: g[2])[0]
    print(f"[val] 选出 w={best_w}（{dict((g[0], g[1]) for g in grid)[best_w]:.4f}）")
    print(f"[诊断] test 上最优 w={test_best_w}（仅诊断，不可用于决策）")

    # ---------- 5) test 三口径 ----------
    print(f"\n[test] n={int(m_te.sum())} 基准={yy[m_te].mean():.4f} "
          f"天数={len(np.unique(dte))}")
    print(f"{'口径':<14}{'AUC':>9}{'全局1%':>10}{'按天top' + str(a.daily_k):>12}")
    rows = []
    for name, s in [("盘口腿", zo_s), ("K线腿", zk_s),
                    (f"融合w={best_w}", zo_s + best_w * zk_s)]:
        auc = float(roc_auc_score(yy[m_te], s[m_te]))
        a1, _ = acc_at_cov(s[m_te], yy[m_te], 0.01)
        dk = dacc(s, m_te, mt)
        rows.append((name, auc, a1, dk))
        print(f"{name:<14}{auc:>9.4f}{a1:>10.4f}{dk:>12.4f}")

    # LR stacker（val 拟合 2 特征，test 评估）
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(np.column_stack([zo_s[m_val], zk_s[m_val]]), yy[m_val])
    s_lr = lr.decision_function(np.column_stack([zo_s, zk_s]))
    auc = float(roc_auc_score(yy[m_te], s_lr[m_te]))
    a1, _ = acc_at_cov(s_lr[m_te], yy[m_te], 0.01)
    dk = dacc(s_lr, m_te, mt)
    rows.append(("LR stacker", auc, a1, dk))
    print(f"{'LR stacker':<14}{auc:>9.4f}{a1:>10.4f}{dk:>12.4f}"
          f"  coef=ob {lr.coef_[0][0]:+.3f} / kl {lr.coef_[0][1]:+.3f}")

    corr = float(np.corrcoef(zo_s[m_te], zk_s[m_te])[0, 1])
    print(f"\n[相关性] 两腿 test 分数相关={corr:.3f}（越低融合空间越大）")

    # 同向/反向分组：判断融合是靠“共识压噪”还是“互补补位”
    agree = np.sign(zo_s[m_te]) == np.sign(zk_s[m_te])
    for tag, mm in [("同向", agree), ("反向", ~agree)]:
        if mm.sum():
            pr = (zo_s[m_te][mm] > 0).astype(int)
            print(f"[{tag}] 占比={mm.mean():.3f} 盘口腿ACC={float((pr == yy[m_te][mm]).mean()):.4f}")

    # 逐日分布（目标按“每天 15 单”衡量，必须看稳定性）
    print(f"\n[逐日] {'腿':<12}{'天数':>6}{'日均':>9}{'中位':>9}{'p10':>8}{'最差':>8}{'≥65%天':>9}")
    for name, s in [("盘口腿", zo_s), ("K线腿", zk_s),
                    (f"融合w={best_w}", zo_s + best_w * zk_s), ("LR stacker", s_lr)]:
        accs = daily_topk_eval(s[m_te], yy[m_te], dte, mt, a.daily_k)
        print(f"{name:<12}{len(accs):>6}{accs.mean():>9.4f}{np.median(accs):>9.4f}"
              f"{np.percentile(accs, 10):>8.4f}{accs.min():>8.4f}{(accs >= 0.65).mean():>9.3f}")

    best = max(rows, key=lambda r: r[3])
    print(f"\n[结论] 最优 {best[0]} 按天top{a.daily_k}={best[3]:.4f}（目标 0.65）")

    if a.save_pred:
        pl.DataFrame({"close_time": mct[m_te], "target": yy[m_te],
                      "z_ob": zo_s[m_te], "z_kl": zk_s[m_te]}).write_parquet(a.save_pred)
        print(f"[save] {a.save_pred}")


if __name__ == "__main__":
    main()
