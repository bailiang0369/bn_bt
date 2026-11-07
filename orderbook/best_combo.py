"""把所有**已单独验证有效**的改动合成一次，给出诚实口径下的最好配置。

已单独验证有效的三项（各自 3 窗均值）：
  §32 K 线腿去动量（no_mom）      ：roll3d +0.81pp
  §34 盘口腿 wf365 滚动重训       ：roll1d +1.1pp，最差窗 0.5809→0.5975
  §33 两腿融合 w=0.2~0.3 @K=8~12  ：+0.5~1.3pp，最差窗 +1.7pp

本脚本组合：盘口腿 = wf365 滚动重训；K 线腿 = no_mom 单次训练；融合 = w_ob + w·w_kl，
再扫 K（日均单量 5/8/10/12/15）。对照仍报"单次训练盘口腿"作为基线。

产出：一张 (w, K) × (均值/最差窗) 表 —— 这就是我们目前能给的最优操作点。
"""
import os
import sys
import time

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lightgbm as lgb  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from morph.kline_ablate import group_of  # noqa: E402
from morph.kline_boost import (OB_END, PARAMS as KL_P, build_features, day_index,  # noqa: E402
                               load_klines, log, make_target, roll_z)
from orderbook.znorm_eval import BASE, H, OB_REG, load  # noqa: E402

DAY = 86_400_000
BLOCK = 30
W_GRID = [0.0, 0.2, 0.3, 0.5]
K_GRID = [15, 12, 10, 8, 5]


def fit(df_X, y, tr, cols, pred, seeds, rounds, params):
    acc = np.zeros(len(pred))
    for sd in seeds:
        q = dict(params, seed=sd, bagging_seed=sd, feature_fraction_seed=sd, data_random_seed=sd)
        m = lgb.train(q, lgb.Dataset(df_X[tr], y[tr], feature_name=cols),
                      num_boost_round=rounds)
        acc += m.predict(df_X[pred])
    return (acc / len(seeds)).astype(np.float32)


def curve(score, y, dd, k_per_day):
    n_days = len(np.unique(dd))
    k = min(int(k_per_day * n_days), len(y))
    idx = np.argsort(-np.abs(score))[:k]
    return float(((score[idx] > 0).astype(int) == y[idx]).mean())


def main():
    offs = [int(x) for x in os.environ.get("OFFS", "0,60,120").split(",")]
    oseeds = [int(x) for x in os.environ.get("OSEEDS", "42,43,44").split(",")]
    kseeds = [int(x) for x in os.environ.get("KSEEDS", "42,43").split(",")]
    rounds = int(os.environ.get("ROUNDS", "60"))
    wf_win = int(os.environ.get("WF_WIN", "365"))
    t0 = time.time()

    ob = load()
    ct = ob["close_time"].to_numpy()
    y = ob["target"].to_numpy().astype(np.int8)
    ob_cols = [c for c in ob.columns if c not in ("target", "close_time")]
    X_ob = ob.select(ob_cols).to_numpy().astype(np.float32)
    del ob
    L = X_ob.shape[0]
    k1_0, k2_0 = int(L * 0.64), int(L * 0.80)

    raw = load_klines().filter(pl.col("close_time") <= OB_END)
    ft, kl_all = build_features(make_target(raw, H))
    ct_kl = ft["close_time"].to_numpy().astype(np.int64)
    y_kl = ft["target"].to_numpy().astype(np.int8)
    kl_cols = [c for c in kl_all if group_of(c) != "MOM"]
    X_kl = ft.select(kl_cols).to_numpy().astype(np.float32)
    del ft, raw
    log(f"[data] ob={L}×{len(ob_cols)}  kl={X_kl.shape[0]}×{len(kl_cols)}")

    rows = {}
    for off in offs:
        k1, k2 = k1_0 - off * 1440, k2_0 - off * 1440
        val_start, test_start = int(ct[k1]), int(ct[k2])
        te_m = ct >= test_start
        te_idx = np.where(te_m)[0]
        dd = day_index(ct[te_m])

        # --- 盘口腿 A：单次训练（基线） ---
        t = time.time()
        s_single = np.zeros(L)
        tr = np.arange(0, k1 - H)
        s_single[te_idx] = fit(X_ob, y, tr, ob_cols, te_idx, oseeds, rounds, dict(BASE, **OB_REG))
        log(f"[off={off}] 盘口单次训练 {time.time()-t:.0f}s")

        # --- 盘口腿 B：wf 滚动重训 ---
        t = time.time()
        s_wf = np.zeros(L)
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
            s_wf[pred] = fit(X_ob, y, trb, ob_cols, pred, oseeds, rounds, dict(BASE, **OB_REG))
        log(f"[off={off}] 盘口 wf{wf_win} {time.time()-t:.0f}s ({len(edges)-1} 块)")

        # --- K 线腿 ---
        t = time.time()
        tr_kl = np.where(ct_kl < val_start)[0]
        va_kl = (ct_kl >= val_start) & (ct_kl < test_start)
        tr_kl = tr_kl[tr_kl + H < int(np.where(va_kl)[0][0])]
        s_kl = fit(X_kl, y_kl, tr_kl, kl_cols, np.arange(X_kl.shape[0]), kseeds, rounds, KL_P)
        best = None
        for shift in (0, 1, -1):
            pos = np.clip(np.searchsorted(ct_kl + shift, ct), 0, len(ct_kl) - 1)
            m = (ct_kl[pos] + shift) == ct
            if best is None or m.mean() > best[0]:
                best = (m.mean(), pos, m)
        hit, pos, match = best
        s_kl_ob = np.where(match, s_kl[pos], 0.0)
        log(f"[off={off}] K线训练 {time.time()-t:.0f}s 对齐={hit:.4f}")

        zo1, zo3 = roll_z(s_single, 1440), roll_z(s_single, 4320)
        zw1, zw3 = roll_z(s_wf, 1440), roll_z(s_wf, 4320)
        zk1, zk3 = roll_z(s_kl_ob, 1440), roll_z(s_kl_ob, 4320)
        log(f"    AUC single={roc_auc_score(y[te_m], zo3[te_m]):.4f} "
            f"wf={roc_auc_score(y[te_m], zw3[te_m]):.4f}")

        for tag, a1, a3 in (("single", zo1, zo3), ("wf", zw1, zw3)):
            for w in W_GRID:
                for scale, (x, k) in (("1d", (a1, zk1)), ("3d", (a3, zk3))):
                    s = x + w * k
                    for K in K_GRID:
                        rows[(off, tag, scale, w, K)] = curve(s[te_m], y[te_m], dd, K)

    for tag in ("single", "wf"):
        for scale in ("1d", "3d"):
            log(f"\n===== {tag} / roll_{scale}：三窗均值（行=w，列=日均单量）=====")
            log(f"{'w':>5}" + "".join(f"{('K%d' % k):>9}" for k in K_GRID))
            for w in W_GRID:
                line = f"{w:>5.2f}"
                for K in K_GRID:
                    line += f"{np.mean([rows[(o, tag, scale, w, K)] for o in offs]):>9.4f}"
                log(line)
            log(f"----- {tag} / roll_{scale}：最差窗 -----")
            log(f"{'w':>5}" + "".join(f"{('K%d' % k):>9}" for k in K_GRID))
            for w in W_GRID:
                line = f"{w:>5.2f}"
                for K in K_GRID:
                    line += f"{np.min([rows[(o, tag, scale, w, K)] for o in offs]):>9.4f}"
                log(line)
    log(f"[done] {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
