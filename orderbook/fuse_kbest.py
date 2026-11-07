"""K 线腿 × 盘口腿融合 + 频次-准确率曲线（第 2 步，也是 65% 目标的落点）。

为什么单独写一个
----------------
§28 的融合结论是「+0.2~0.3pp，按天口径为负」，但那次用的是 **static val z** 且没给频次曲线。
本次：
  1. 两腿各自训练、各自 rolling z（1d/3d，严格因果）；
  2. 在 close_time 上精确对齐（K 线 close_time 比盘口早 1ms，需 +1ms）；
  3. 报 w 网格 × K 网格（日均 5/8/10/12/15/20 单）的准确率，三个窗口 off=0/60/120；
  4. 输出两腿相关性（判断还有没有互补空间）。

判据
  - 融合有用：同一 K 下，三窗均值比 max(单腿) 高 ≥0.5pp，且最差窗不退化；
  - 达标：某个 (w, K) 上三窗均值 ≥0.65 且最差窗 ≥0.62，才敢说"稳定 65"。

环境变量：OFFS / OSEEDS / KSEEDS / ROUNDS / KL_SUBSET(all|flow_shape|flow_only|no_mom) / OUT
产物：两腿原始分数 + target + close_time（parquet），便于离线换口径复算。
"""
import os
import sys
import time

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lightgbm as lgb  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from morph.kline_ablate import group_of, subset  # noqa: E402
from morph.kline_boost import (OB_END, PARAMS as KL_PARAMS, TEST_START, VAL_START,  # noqa: E402
                               build_features, complete_days, day_index,
                               global_topk_eval, load_klines, log, make_target,
                               roll_z, zscore)
from orderbook.znorm_eval import BASE, H, OB_REG, load  # noqa: E402

DAY = 86_400_000
W_GRID = [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
K_GRID = [20, 15, 12, 10, 8, 5]


def fit(X, y, tr, cols, seeds, rounds, params):
    acc = np.zeros(X.shape[0], dtype=np.float64)
    for sd in seeds:
        q = dict(params, seed=sd, bagging_seed=sd, feature_fraction_seed=sd, data_random_seed=sd)
        acc += lgb.train(q, lgb.Dataset(X[tr], y[tr], feature_name=cols),
                         num_boost_round=rounds).predict(X)
    return (acc / len(seeds)).astype(np.float32)


def curve(score, y, dd, k_per_day):
    """全局 top-K（|score| 排序），K = k_per_day × 天数。"""
    n_days = len(np.unique(dd))
    k = min(int(k_per_day * n_days), len(y))
    idx = np.argsort(-np.abs(score))[:k]
    return float(((score[idx] > 0).astype(int) == y[idx]).mean())


def main():
    offs = [int(x) for x in os.environ.get("OFFS", "0,60,120").split(",")]
    oseeds = [int(x) for x in os.environ.get("OSEEDS", "42,43,44,45,46").split(",")]
    kseeds = [int(x) for x in os.environ.get("KSEEDS", "42,43").split(",")]
    rounds = int(os.environ.get("ROUNDS", "60"))
    kl_subset = os.environ.get("KL_SUBSET", "all")
    out = os.environ.get("OUT", "")
    t0 = time.time()

    # ---- 盘口腿 ----
    ob = load()
    ct_ob = ob["close_time"].to_numpy()
    y_ob = ob["target"].to_numpy().astype(np.int8)
    ob_cols = [c for c in ob.columns if c not in ("target", "close_time")]
    X_ob = ob.select(ob_cols).to_numpy().astype(np.float32)
    del ob
    L_ob = X_ob.shape[0]
    k1_0, k2_0 = int(L_ob * 0.64), int(L_ob * 0.80)
    log(f"[ob] rows={L_ob} feats={len(ob_cols)}")

    # ---- K 线腿 ----
    raw = load_klines().filter(pl.col("close_time") <= OB_END)
    ft, kl_all = build_features(make_target(raw, H))
    ct_kl = ft["close_time"].to_numpy().astype(np.int64)
    y_kl = ft["target"].to_numpy().astype(np.int8)
    if kl_subset == "all":
        kl_cols = kl_all
    elif kl_subset == "no_mom":
        kl_cols = [c for c in kl_all if group_of(c) != "MOM"]
    else:
        groups = {"flow_shape": {"FLOW", "SHAPE"}, "flow_only": {"FLOW"},
                  "shape_only": {"SHAPE"}, "vol_shape": {"VOL", "SHAPE"}}[kl_subset]
        kl_cols = subset(kl_all, groups)
    X_kl = ft.select(kl_cols).to_numpy().astype(np.float32)
    del ft, raw
    log(f"[kl] rows={X_kl.shape[0]} feats={len(kl_cols)} subset={kl_subset}")

    store = []
    rows = {}
    for off in offs:
        k1, k2 = k1_0 - off * 1440, k2_0 - off * 1440
        test_start = int(ct_ob[k2])
        val_start = int(ct_ob[k1])
        tr_ob = np.arange(0, k1 - H)
        va_ob = (ct_ob >= val_start) & (ct_ob < test_start)
        te_ob = (ct_ob >= test_start)

        tr_kl = np.where(ct_kl < val_start)[0]
        va_kl = (ct_kl >= val_start) & (ct_kl < test_start)
        tr_kl = tr_kl[tr_kl + H < int(np.where(va_kl)[0][0])]

        s_ob = fit(X_ob, y_ob, tr_ob, ob_cols, oseeds, rounds, dict(BASE, **OB_REG))
        s_kl = fit(X_kl, y_kl, tr_kl, kl_cols, kseeds, rounds, KL_PARAMS)
        log(f"[off={off}] 训练完成 ({time.time()-t0:.0f}s)  test={int(te_ob.sum())} 天={len(np.unique(day_index(ct_ob[te_ob])))}")

        # K 线分数对齐到盘口网格（close_time +1ms 精确匹配）
        pos = np.searchsorted(ct_kl + 1, ct_ob)
        pos = np.clip(pos, 0, len(ct_kl) - 1)
        match = (ct_kl[pos] + 1) == ct_ob
        s_kl_on_ob = np.where(match, s_kl[pos], 0.0)
        log(f"    对齐命中率={match.mean():.4f}")

        z_ob1, z_ob3 = roll_z(s_ob, 1440), roll_z(s_ob, 4320)
        z_kl1, z_kl3 = roll_z(s_kl_on_ob, 1440), roll_z(s_kl_on_ob, 4320)
        pair = np.corrcoef(z_ob3[te_ob], z_kl3[te_ob])[0, 1]
        dd = day_index(ct_ob[te_ob])
        log(f"    两腿相关(roll3d,test)={pair:.3f}  "
            f"AUC ob={roc_auc_score(y_ob[te_ob], z_ob3[te_ob]):.4f} "
            f"kl={roc_auc_score(y_ob[te_ob], z_kl3[te_ob]):.4f}")

        for scale, zo, zk in (("1d", z_ob1, z_kl1), ("3d", z_ob3, z_kl3)):
            for w in W_GRID:
                s = zo + w * zk
                key = (scale, w)
                rows.setdefault(key, {})
                for K in K_GRID:
                    rows[key][(off, K)] = curve(s[te_ob], y_ob[te_ob], dd, K)
        store.append((off, te_ob, s_ob, s_kl_on_ob, y_ob, ct_ob))

    # ---- 汇总 ----
    for scale in ("1d", "3d"):
        log(f"\n===== 口径 roll_{scale}：三窗均值准确率（行=w，列=日均单量）=====")
        log(f"{'w':>5}" + "".join(f"{('K%d' % k):>9}" for k in K_GRID))
        for w in W_GRID:
            rs = rows[(scale, w)]
            line = f"{w:>5.2f}"
            for K in K_GRID:
                vals = [rs[(o, K)] for o in offs if (o, K) in rs]
                line += f"{np.mean(vals):>9.4f}"
            log(line)

        log(f"\n----- roll_{scale}：各 (w,K) 的最差窗（判断稳定性）-----")
        log(f"{'w':>5}" + "".join(f"{('K%d' % k):>9}" for k in K_GRID))
        for w in W_GRID:
            rs = rows[(scale, w)]
            line = f"{w:>5.2f}"
            for K in K_GRID:
                vals = [rs[(o, K)] for o in offs if (o, K) in rs]
                line += f"{np.min(vals):>9.4f}"
            log(line)

    if out:
        try:
            frames = []
            for off, te, a, b, y, c in store:
                frames.append(pl.DataFrame({"off": off, "close_time": c[te],
                                            "target": y[te], "s_ob": a[te], "s_kl": b[te]}))
            pl.concat(frames).write_parquet(out)
            log(f"[save] {out}")
        except Exception as e:
            log(f"[save] 失败 {e}")
    log(f"[done] {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
