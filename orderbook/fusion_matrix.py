"""3min/多 horizon orderbook 变体矩阵：一次 kernel 内跑多组配置，快速定位有效方向。

为什么不含 faiss 腿：Run E/F/G 已确认 3min 上 w=0（纯 orderbook）最好，
给 faiss 任何正权重都会伤 test（val 只有 30 天，选 w 不稳）。

每组变体都报三个口径：
  AUC / 全局 top1%（14.4 单/天）/ 按天 top-k（真实『日均 15 单』）

用法：
    python -m orderbook.fusion_matrix                                  # 默认 horizon=3，全部变体
    python -m orderbook.fusion_matrix --horizons 1,2,3,5 --variants base,minret2
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
    # 叠加候选（矩阵第一轮里单项最好的组合）
    "combo":    dict(x="der", decay=0, minret=2, sort="p",
                     extra=dict(num_leaves=31, min_child_samples=500,
                                feature_fraction=0.6, lambda_l2=5.0)),
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


def daily_topk_ret(score, fwd_bp, day_id, mask, k):
    """按天 top-k 选中样本的平均收益（bp，已按分数符号取方向）。"""
    s, r, d = score[mask], fwd_bp[mask], day_id[mask]
    rets = []
    for u in np.unique(d):
        m = d == u
        kk = min(k, int(m.sum()))
        idx = np.argsort(-np.abs(s[m]))[:kk]
        rets.append(float((np.sign(s[m][idx]) * r[m][idx]).mean()))
    return float(np.mean(rets))


def smooth(score, w):
    """时序平滑（只用过去 w 个 bar），用于压制单 bar 噪声。"""
    c = np.concatenate([[0.0], np.cumsum(score, dtype=np.float64)])
    out = (c[w:] - c[:-w]) / w
    return np.concatenate([np.full(w - 1, score[0], dtype=np.float64), out])


def metrics(score, y, day_id, mask, k, fwd_bp=None):
    a1, _ = acc_at_cov(score, y, 0.01)
    accs = daily_topk_eval(score, y, day_id, mask, k)
    d = float(accs.mean())
    ret = daily_topk_ret(score, fwd_bp, day_id, mask, k) if fwd_bp is not None else float("nan")
    return float(roc_auc_score(y, score)), a1, d, ret


def run_horizon(ob0, h, names, seeds, rounds, daily_k, save_pred=None, test_offset_days=0):
    if test_offset_days > 0:
        # 把时间轴整体前移 N 天：test 变成"更早的 92 天"，训练也相应截止（严格前向）
        ob0 = ob0.head(ob0.height - test_offset_days * 1440)
    ob = add_target(ob0, h).drop_nulls()
    L = len(ob)
    k1, k2 = int(L * 0.64), int(L * 0.80)
    tr, va, te = slice(0, k1 - h), slice(k1, k2 - h), slice(k2, L)
    ct = ob["close_time"].to_numpy()
    T_end = int(ct[k1 - 1 - h])
    y = ob["target"].to_numpy().astype(np.int8)
    close = ob["close"].to_numpy().astype(np.float64)
    fwd = np.zeros(L)
    fwd[:-h] = close[h:] / close[:-h] - 1.0

    base_cols = [c for c in ob.columns if c not in ("target", "close_time")]
    Xb = ob.select(base_cols).to_numpy().astype(np.float32)
    obd = add_derived_features(ob)
    der_cols = [c for c in obd.columns if c not in ("target", "close_time")]
    Xd = obd.select(der_cols).to_numpy().astype(np.float32)

    day_te = day_index(ct[te])
    m_te = complete_days(day_te)
    n_full = len(np.unique(day_te[m_te]))
    fwd_bp = fwd[te] * 1e4
    print(f"\n[h={h}] rows={L} train[0:{k1 - h}] val[{k1}:{k2 - h}] test[{k2}:{L}] "
          f"purge={h} 完整天={n_full} 按天top{daily_k}→{n_full * daily_k} 单  "
          f"feats base={Xb.shape[1]} derived={Xd.shape[1]}")

    rows, preds = [], {}
    for name in names:
        cfg = VARIANTS[name]
        X, cols = (Xb, base_cols) if cfg["x"] == "base" else (Xd, der_cols)
        weight = None
        if cfg["decay"] > 0:
            age = (T_end - ct) / DAY_MS
            weight = np.power(0.5, np.clip(age, 0, None) / cfg["decay"])
        if cfg["minret"] > 0:
            keep = (np.abs(fwd) * 1e4 >= cfg["minret"]).astype(np.float64)
            weight = keep if weight is None else weight * keep

        p_all, iters = train_lgb_avg(X, y, tr, va, cols, seeds, rounds, weight, cfg["extra"])
        score = p_all[te] - 0.5
        if cfg["sort"] == "amp":
            r_all = train_lgb_reg_avg(X, fwd, tr, va, cols, seeds, rounds, weight)
            score = (p_all[te] - 0.5) * np.abs(r_all[te])
        auc, a1, dk, ret = metrics(score, y[te], day_te, m_te, daily_k, fwd_bp)
        rows.append((name, auc, a1, dk, ret))
        preds[name] = score
        print(f"  [{name}] AUC={auc:.4f} 全局1%={a1:.4f} 按天top{daily_k}={dk:.4f} "
              f"平均{ret:.2f}bp iters={iters}")
        # 时序平滑后处理（不重训，只是把单 bar 噪声压掉）
        for w in (3, 5):
            s2 = smooth(score, w)
            auc2, a12, dk2, ret2 = metrics(s2, y[te], day_te, m_te, daily_k, fwd_bp)
            rows.append((f"{name}+s{w}", auc2, a12, dk2, ret2))
            preds[f"{name}_s{w}"] = s2

    print(f"  {'变体':<12}{'AUC':>9}{'全局1%':>10}{'按天top' + str(daily_k):>12}{'平均bp':>9}")
    for name, auc, a1, dk, ret in sorted(rows, key=lambda r: -r[3]):
        print(f"  {name:<12}{auc:>9.4f}{a1:>10.4f}{dk:>12.4f}{ret:>9.2f}")
    best = max(rows, key=lambda r: r[3])
    print(f"  [h={h}] 最优按天口径: {best[0]} = {best[3]:.4f}（平均 {best[4]:.2f}bp）")

    if save_pred:
        out = pl.DataFrame({"close_time": ct[te], "target": y[te],
                            "horizon": [h] * len(ct[te]), **preds})
        out.write_parquet(save_pred)
        print(f"  [save] {save_pred}")
    return rows


def main():
    ap = argparse.ArgumentParser(description="orderbook 变体矩阵（多 horizon）")
    ap.add_argument("--horizons", default=os.environ.get("KAGGLE_HORIZONS", "3"))
    ap.add_argument("--rounds", type=int, default=1000)
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--daily-k", type=int, default=15)
    ap.add_argument("--variants",
                    default=os.environ.get("KAGGLE_VARIANTS", ",".join(VARIANTS)))
    ap.add_argument("--save-pred", default=os.environ.get("KAGGLE_FUSION_PRED"))
    ap.add_argument("--test-offsets",
                    default=os.environ.get("KAGGLE_TEST_OFFSETS",
                                           os.environ.get("KAGGLE_TEST_OFFSET", "0")),
                    help="时间轴整体前移的天数列表（0=原始 test），用于样本外复核")
    a = ap.parse_args()

    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    horizons = [int(s) for s in a.horizons.split(",") if s.strip()]
    offsets = [int(s) for s in a.test_offsets.split(",") if s.strip()]
    names = [v.strip() for v in a.variants.split(",") if v.strip() and v.strip() in VARIANTS]

    ob0 = load_orderbook()
    print(f"[ob] rows={ob0.height} horizons={horizons} variants={names}")

    all_rows = {}
    for h in horizons:
        for off in offsets:
            save = None
            if a.save_pred:
                stem = a.save_pred.rsplit(".parquet", 1)[0]
                save = f"{stem}_h{h}_off{off}.parquet"
            # 不同 horizon 的 drop_nulls 行数不同，派生特征不跨 horizon 复用（成本仅几秒）
            all_rows[(h, off)] = run_horizon(ob0, h, names, seeds, a.rounds, a.daily_k,
                                             save, off)

    print("\n" + "=" * 78)
    print("汇总（按天 top%d）" % a.daily_k)
    print(f"{'h':>4}{'前移天':>8}{'最优变体':>14}{'按天':>10}{'全局1%':>10}{'平均bp':>9}")
    for (h, off), rows in all_rows.items():
        best = max(rows, key=lambda r: r[3])
        print(f"{h:>4}{off:>8}{best[0]:>14}{best[3]:>10.4f}{best[2]:>10.4f}{best[4]:>9.2f}")
    overall = max(((k, r) for k, rows in all_rows.items() for r in rows), key=lambda x: x[1][3])
    print(f"\n[结论] 全局最优: horizon={overall[0][0]} 前移={overall[0][1]}天 "
          f"变体={overall[1][0]} 按天={overall[1][3]:.4f}（目标 0.65）")


if __name__ == "__main__":
    main()
