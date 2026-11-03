"""K 线腿 × 盘口腿 融合（3min，日均 15 单 = 全局 top-K）。

评估口径（用户 2026-11-03 明确）：日均 15 单指【测试集总单量 / 天数】≈15，
即全段按 |score| 排序取前 K = 15 × 天数（`global_topk_eval`）；
不是"每天必须凑满 15 单"（`daily_topk_eval`，作为对照口径同时打印）。
两者差 2~3pp，历史 Run I~R 的结论都是后一种口径下得出的，需按新口径重读。

两条腿用**不同的训练数据集**（用户指出的关键）：
  - K 线腿：2020-01 → val_start（5.5 年，3.5M 行，63 个量价特征）；
  - 盘口腿：2025-07 → val_start（15 个月，687k 行，150 个盘口特征）；
  - 两腿都在 orderbook 的 val/test 时间边界上评估（按 close_time 精确 join，+1ms）。

用法：
    python -m orderbook.fusion_kl                       # 单段（off=0）
    python -m orderbook.fusion_kl --offsets 0,60,120    # 时间外推稳定性复核
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
    load_orderbook, add_target, train_lgb_avg,
    day_index, complete_days, daily_topk_eval, global_topk_eval)

OB_REG = dict(num_leaves=31, min_child_samples=500, feature_fraction=0.6, lambda_l2=5.0)
# 主口径是全局 top-K 排序，小权重才有意义（K 线腿弱于盘口腿，权重 ∝ 强度）
WEIGHTS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]


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
    ap.add_argument("--daily-k", type=int, default=15, help="目标日均单量（K = 该值 × 天数），主口径")
    ap.add_argument("--offsets", default=os.environ.get("KAGGLE_OFFSETS", "0"),
                    help="逗号分隔的多段偏移（天），用于时间外推稳定性复核")
    _wf = os.environ.get("KAGGLE_W_FIXED")
    ap.add_argument("--w-fixed", type=float, default=(float(_wf) if _wf else None),
                    help="固定融合权重（跳过 val 选择）")
    ap.add_argument("--save-pred", default=os.environ.get("KAGGLE_FUSION_PRED"))
    a = ap.parse_args()

    h = a.h
    kseeds = [int(s) for s in a.kline_seeds.split(",") if s.strip()]
    oseeds = [int(s) for s in a.ob_seeds.split(",") if s.strip()]
    offsets = [int(x) for x in a.offsets.split(",") if x.strip()]

    # ---------- 1) 盘口腿：决定统一时间边界（一次性加载） ----------
    ob = add_target(load_orderbook(), h).filter(pl.col("target").is_not_null())
    L = len(ob)
    ob_ct = ob["close_time"].to_numpy()
    ob_end = int(ob_ct[-1])
    k1_0, k2_0 = int(L * 0.64), int(L * 0.80)
    print(f"[ob] rows={L} end={ob_end}")

    ob_cols = [c for c in ob.columns if c not in ("target", "close_time")]
    Xo = ob.select(ob_cols).to_numpy().astype(np.float32)
    yo = ob["target"].to_numpy().astype(np.int8)

    # ---------- 2) K 线腿：2020 → ob_end（一次性特征） ----------
    kl = load_klines(end=(2026, 10))
    kl = kl.filter(pl.col("close_time") <= ob_end)
    kl = add_target(kl, h).filter(pl.col("target").is_not_null())
    print(f"[kl] 时间 {int(kl['close_time'][0])} ~ {int(kl['close_time'][-1])}（orderbook 末尾 {ob_end}）")
    kf, kcols = build_features(kl)
    Xk = kf.select(kcols).to_numpy().astype(np.float32)
    yk = kf["target"].to_numpy().astype(np.int8)
    kct = kf["close_time"].to_numpy()
    print(f"[kl] rows={len(yk)} feats={len(kcols)}")

    def run_off(off):
        k1, k2 = k1_0 - off * 1440, k2_0 - off * 1440
        assert k1 - h > 0, f"off-days={off} 太大（k1={k1}）"
        val_start, test_start = int(ob_ct[k1]), int(ob_ct[k2])
        print(f"\n========== [off={off}] val_start={val_start} test_start={test_start} ==========")

        # 盘口腿
        tro, vao = slice(0, k1 - h), slice(k1, k2 - h)
        p_ob, ob_iters = train_lgb_avg(Xo, yo, tro, vao, ob_cols, oseeds, a.rounds, None, OB_REG)
        print(f"[ob] iters={ob_iters}")

        # K 线腿
        trk = np.where(kct + h * 60_000 < val_start)[0][::a.kline_stride]
        pred_idx = np.where(kct >= val_start)[0]
        p_kl = train_predict(Xk, yk, trk, kcols, kseeds, a.rounds, pred_idx, KL_PARAMS)
        kct_pred = kct[pred_idx]
        print(f"[kl] train={len(trk)} pred={len(pred_idx)}")

        # join
        merged = (pl.DataFrame({"close_time": ob_ct, "target": yo, "p_ob": p_ob})
                  .join(pl.DataFrame({"close_time": kct_pred + 1, "p_kl": p_kl}),
                        on="close_time", how="inner").sort("close_time"))
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

        def gacc(s, m):
            idx = np.where(m)[0]
            return global_topk_eval(s[idx], yy[idx], dyn[idx], a.daily_k)

        def dacc(s, m, cm):
            idx = np.where(m)[0]
            return float(daily_topk_eval(s[idx], yy[idx], dyn[idx], cm, a.daily_k).mean())

        # 权重网格（主口径 = 全局 top-K）
        n_val_days = len(np.unique(dv))
        n_te_days = len(np.unique(dte))
        print(f"\n[权重网格] 全局 top{a.daily_k}（val {n_val_days} 天 / test {n_te_days} 天）")
        print(f"{'w':>7}{'val':>10}{'test':>10}")
        grid = []
        for w in WEIGHTS:
            s = zo_s + w * zk_s
            av, at = gacc(s, m_val), gacc(s, m_te)
            grid.append((w, av, at))
            print(f"{w:>7.2f}{av:>10.4f}{at:>10.4f}")
        if a.w_fixed is not None:
            best_w = a.w_fixed
            print(f"[w] 固定为 {best_w}")
        else:
            best_w = max(grid, key=lambda g: g[1])[0]
            print(f"[val] 选出 w={best_w}（{dict((g[0], g[1]) for g in grid)[best_w]:.4f}）")
        print(f"[诊断] test 上最优 w={max(grid, key=lambda g: g[2])[0]}（仅诊断，不可用于决策）")

        # test 明细
        print(f"\n[test] n={int(m_te.sum())} 基准={yy[m_te].mean():.4f} 天数={n_te_days} "
              f"全局K={a.daily_k * n_te_days}")
        print(f"{'口径':<14}{'AUC':>9}{'全局topK':>10}{'按天topK':>10}")
        rows = []
        for name, s in [("盘口腿", zo_s), ("K线腿", zk_s), (f"融合w={best_w}", zo_s + best_w * zk_s)]:
            auc = float(roc_auc_score(yy[m_te], s[m_te]))
            gk, dk = gacc(s, m_te), dacc(s, m_te, mt)
            rows.append((name, auc, gk, dk))
            print(f"{name:<14}{auc:>9.4f}{gk:>10.4f}{dk:>10.4f}")

        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(max_iter=1000)
        lr.fit(np.column_stack([zo_s[m_val], zk_s[m_val]]), yy[m_val])
        s_lr = lr.decision_function(np.column_stack([zo_s, zk_s]))
        auc = float(roc_auc_score(yy[m_te], s_lr[m_te]))
        gk, dk = gacc(s_lr, m_te), dacc(s_lr, m_te, mt)
        rows.append(("LR stacker", auc, gk, dk))
        print(f"{'LR stacker':<14}{auc:>9.4f}{gk:>10.4f}{dk:>10.4f}"
              f"  coef=ob {lr.coef_[0][0]:+.3f} / kl {lr.coef_[0][1]:+.3f}")

        corr = float(np.corrcoef(zo_s[m_te], zk_s[m_te])[0, 1])
        print(f"\n[相关性] 两腿 test 分数相关={corr:.3f}")

        for name, s in [("盘口腿", zo_s), (f"融合w={best_w}", zo_s + best_w * zk_s)]:
            idx = np.argsort(-np.abs(s[m_te]))[:a.daily_k * n_te_days]
            dd = dte[idx]
            u, c = np.unique(dd, return_counts=True)
            print(f"[集中度] {name}: 覆盖 {len(u)}/{n_te_days} 天，单/天 {c.min()}~{c.max()}（均 {c.mean():.1f}）")

        print(f"\n[对照口径·按天top{a.daily_k}] {'腿':<12}{'天数':>6}{'日均':>9}{'中位':>9}{'p10':>8}{'最差':>8}{'≥65%天':>9}")
        for name, s in [("盘口腿", zo_s), ("K线腿", zk_s), (f"融合w={best_w}", zo_s + best_w * zk_s), ("LR stacker", s_lr)]:
            accs = daily_topk_eval(s[m_te], yy[m_te], dte, mt, a.daily_k)
            print(f"{name:<12}{len(accs):>6}{accs.mean():>9.4f}{np.median(accs):>9.4f}"
                  f"{np.percentile(accs, 10):>8.4f}{accs.min():>8.4f}{(accs >= 0.65).mean():>9.3f}")

        best = max(rows, key=lambda r: r[2])
        print(f"\n[结论 off={off}] 最优 {best[0]} 全局top{a.daily_k}={best[2]:.4f}"
              f"（目标 0.65 → {'达成' if best[2] >= 0.65 else '未达成'}）")

        if a.save_pred:
            p = a.save_pred.replace(".parquet", f"_off{off}.parquet")
            pl.DataFrame({"close_time": mct[m_te], "target": yy[m_te],
                          "z_ob": zo_s[m_te], "z_kl": zk_s[m_te]}).write_parquet(p)
            print(f"[save] {p}")

    for off in offsets:
        run_off(off)


if __name__ == "__main__":
    main()
