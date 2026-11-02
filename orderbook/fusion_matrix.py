"""3min orderbook 变体矩阵：一次 kernel 内跑多组配置，快速定位有效方向。

为什么不含 faiss 腿：Run E/F/G 已确认 3min 上 w=0（纯 orderbook）最好，
给 faiss 任何正权重都会伤 test（val 只有 30 天，选 w 不稳）。

每组变体都报三个口径：
  AUC / 全局 top1%（14.4 单/天）/ 按天 top-k（真实『日均 15 单』）

用法：
    python -m orderbook.fusion_matrix
    python -m orderbook.fusion_matrix --variants base,derived,minret2
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

from orderbook.fusion_test import (
    load_orderbook, add_target, add_derived_features, train_lgb_avg,
    acc_at_cov, day_index, complete_days, daily_topk_eval)

DAY_MS = 86_400_000

VARIANTS = {
    "base":     dict(x="base", decay=0, minret=0, extra=None, sort="p"),
    "derived":  dict(x="der", decay=0, minret=0, extra=None, sort="p"),
    "decay120": dict(x="base", decay=120, minret=0, extra=None, sort="p"),
    "decay240": dict(x="base", decay=240, minret=0, extra=None, sort="p"),
    "minret2":  dict(x="base", decay=0, minret=2, extra=None, sort="p"),
    "minret5":  dict(x="base", decay=0, minret=5, extra=None, sort="p"),
    "reg":      dict(x="base", decay=0, minret=0, sort="p",
                     extra=dict(num_leaves=31, min_child_samples=500,
                                feature_fraction=0.6, lambda_l2=5.0)),
    "amp":      dict(x="base", decay=0, minret=0, extra=None, sort="amp"),
}


def train_lgb_reg_avg(X, y_ret, tr, va, cols, seeds, rounds, weight=None, alpha=1e-4):
    """回归头：预测未来 horizon 收益幅度（huber 抗极端值），多 seed 平均。"""
    preds = np.zeros(len(y_ret), dtype=np.float64)
    for sd in seeds:
        params = dict(objective="huber", alpha=alpha, metric="mae",
                      learning_rate=0.05, num_leaves=31, feature_fraction=0.8,
                      bagging_fraction=0.8, bagging_freq=1, min_child_samples=300,
                      verbose=-1, seed=sd, bagging_seed=sd,
                      feature_fraction_seed=sd, data_random_seed=sd)
        ds = lgb.Dataset(X[tr], y_ret[tr], feature_name=cols,
                         weight=None if weight is None else weight[tr])
        vs = lgb.Dataset(X[va], y_ret[va], reference=ds,
                         weight=None if weight is None else weight[va])
        m = lgb.train(params, ds, num_boost_round=rounds, valid_sets=[ds, vs],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        preds += m.predict(X, num_iteration=m.best_iteration)
    return (preds / len(seeds)).astype(np.float32)


def metrics(score, y, day_id, mask, k):
    a1, _ = acc_at_cov(score, y, 0.01)
    d = float(daily_topk_eval(score, y, day_id, mask, k).mean())
    return float(roc_auc_score(y, score)), a1, d


def main():
    ap = argparse.ArgumentParser(description="3min orderbook 变体矩阵")
    ap.add_argument("--horizon", type=int, default=int(os.environ.get("KAGGLE_HORIZON", "3")))
    ap.add_argument("--rounds", type=int, default=1000)
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--daily-k", type=int, default=15)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--save-pred", default=os.environ.get("KAGGLE_FUSION_PRED"))
    a = ap.parse_args()

    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    h = a.horizon

    ob = load_orderbook()
    ob = add_target(ob, h).drop_nulls()
    L = len(ob)
    k1, k2 = int(L * 0.64), int(L * 0.80)
    tr, va, te = slice(0, k1 - h), slice(k1, k2 - h), slice(k2, L)
    ct = ob["close_time"].to_numpy()
    T_end = int(ct[k1 - 1 - h])
    print(f"[ob] rows={L} h={h} train[0:{k1 - h}] val[{k1}:{k2 - h}] test[{k2}:{L}] purge={h}")

    y = ob["target"].to_numpy().astype(np.int8)
    close = ob["close"].to_numpy().astype(np.float64)
    fwd = np.zeros(L)
    fwd[:-h] = close[h:] / close[:-h] - 1.0

    base_cols = [c for c in ob.columns if c not in ("target", "close_time")]
    Xb = ob.select(base_cols).to_numpy().astype(np.float32)
    obd = add_derived_features(ob)
    der_cols = [c for c in obd.columns if c not in ("target", "close_time")]
    Xd = obd.select(der_cols).to_numpy().astype(np.float32)
    print(f"[feats] base={Xb.shape[1]} derived={Xd.shape[1]}")

    day_te = day_index(ct[te])
    m_te = complete_days(day_te)
    n_full = len(np.unique(day_te[m_te]))
    print(f"[eval] test 完整天={n_full}  按天 top{a.daily_k} → 总 {n_full * a.daily_k} 单")

    rows = []
    preds = {}
    for name in [v.strip() for v in a.variants.split(",") if v.strip()]:
        cfg = VARIANTS[name]
        X, cols = (Xb, base_cols) if cfg["x"] == "base" else (Xd, der_cols)
        weight = None
        if cfg["decay"] > 0:
            age = (T_end - ct) / DAY_MS
            weight = np.power(0.5, np.clip(age, 0, None) / cfg["decay"])
        if cfg["minret"] > 0:
            keep = (np.abs(fwd) * 1e4 >= cfg["minret"]).astype(np.float64)
            weight = keep if weight is None else weight * keep
            print(f"[{name}] 保留 |fwd|>={cfg['minret']}bp: train {keep[tr].mean():.3f}")

        p_all, iters = train_lgb_avg(X, y, tr, va, cols, seeds, a.rounds, weight, cfg["extra"])
        score = p_all[te] - 0.5
        if cfg["sort"] == "amp":
            r_all = train_lgb_reg_avg(X, fwd, tr, va, cols, seeds, a.rounds, weight)
            score = (p_all[te] - 0.5) * np.abs(r_all[te])
            preds[f"{name}_ret"] = r_all[te]
        auc, a1, d15 = metrics(score, y[te], day_te, m_te, a.daily_k)
        rows.append((name, auc, a1, d15))
        preds[name] = score
        print(f"[{name}] AUC={auc:.4f} 全局1%={a1:.4f} 按天top{a.daily_k}={d15:.4f} iters={iters}")

    print("\n" + "=" * 72)
    print(f"{'变体':<12}{'AUC':>9}{'全局1%':>10}{'按天top' + str(a.daily_k):>12}")
    print("-" * 72)
    for name, auc, a1, d15 in sorted(rows, key=lambda r: -r[3]):
        print(f"{name:<12}{auc:>9.4f}{a1:>10.4f}{d15:>12.4f}")
    best = max(rows, key=lambda r: r[3])
    print(f"\n[结论] 最优按天口径: {best[0]} = {best[3]:.4f}（目标 0.65）")

    if a.save_pred and preds:
        out = pl.DataFrame({"close_time": ct[te], "target": y[te], **preds})
        out.write_parquet(a.save_pred)
        print(f"[save] {a.save_pred} rows={out.height}")


if __name__ == "__main__":
    main()
