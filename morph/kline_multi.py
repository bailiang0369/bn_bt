"""第九轮：多 horizon 一致性（不是分数相加，是互相验证）。

第七轮做过 h=1/2/3 分数相加，那是错的（把不同 horizon 的尺度混在一起，
且 roll 引入了未来 bar）。本轮换成"一致性加权"：

  对 h ∈ {1,2,3,5,10} 各训一个模型，都在同一时刻 t 判断"之后 h 分钟方向"；
  agreement = |Σ sign(z_h)| / 5，衡量 5 个时间尺度是否指向同一方向；
  最终置信度 = mean(z_h) × (0.5 + 0.5 × agreement)，方向仍由 mean(z_h) 定。

逻辑：如果 1 分钟和 10 分钟同向，说明不是单根噪声，topK 的高分样本应更可靠。
这改的是"选哪些样本"，不改特征，所以能直接检验 AUC 之外的一致性假设。
"""
import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lightgbm as lgb  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from morph.kline_boost import (DAILY_K, OB_END, PARAMS, TEST_START, VAL_START,  # noqa: E402
                               build_features, complete_days, daily_topk_eval,
                               day_index, global_topk_eval, load_klines, log, zscore)

HS = tuple(int(x) for x in os.environ.get("HS", "1,2,3,5,10").split(","))


def evaluate(tag, s, ct, y, va, rows, min_bars=1200):
    te = np.where(ct >= TEST_START)[0]
    z = zscore(s, s[va])
    dd = day_index(ct[te])
    r = {"static": global_topk_eval(s[te], y[te], dd, DAILY_K),
         "auc": float(roc_auc_score(y[te], z[te])),
         "daily": float(daily_topk_eval(s[te], y[te], dd,
                                        complete_days(dd, min_bars), DAILY_K).mean())}
    edges = np.linspace(int(ct[te[0]]), int(ct[te[-1]]) + 1, 5).astype(np.int64)
    wv = []
    for i in range(4):
        m = (ct[te] >= edges[i]) & (ct[te] < edges[i + 1])
        idx = te[m]
        if len(idx):
            wv.append(global_topk_eval(s[idx], y[idx], day_index(ct[idx]), DAILY_K))
    r["w4"], r["w4sd"] = float(np.mean(wv)), float(np.std(wv, ddof=1))
    r["wins"] = [round(x, 4) for x in wv]
    rows[tag] = r
    log(f"[{tag}] static={r['static']:.4f} AUC={r['auc']:.4f} 按天={r['daily']:.4f} "
        f"窗口4={r['w4']:.4f}±{r['w4sd']:.4f} 明细={r['wins']}")
    return r


def main():
    seeds = [int(x) for x in os.environ.get("SEEDS", "42,43").split(",")]
    rounds = int(os.environ.get("ROUNDS", "60"))

    raw = load_klines().filter(pl.col("close_time") <= OB_END)
    ft, cols = build_features(raw)
    nmax = raw.height - max(HS)          # 所有 horizon 共用同一个行网格
    X = ft.head(nmax).select(cols).to_numpy().astype(np.float32)
    ct = ft.head(nmax)["close_time"].to_numpy()
    close = raw["close"].to_numpy().astype(np.float64)
    del ft
    log(f"[data] rows={nmax} cols={len(cols)}")

    tr = np.where(ct < VAL_START)[0]
    va = np.where((ct >= VAL_START) & (ct < TEST_START))[0]
    tr = tr[tr + 10 < int(va[0])]

    zs = {}
    for h in HS:
        y = (close[h:h + nmax] > close[:nmax]).astype(np.int8)
        acc = np.zeros(nmax)
        for sd in seeds:
            p = dict(PARAMS, seed=sd, bagging_seed=sd, feature_fraction_seed=sd,
                     data_random_seed=sd)
            acc += lgb.train(p, lgb.Dataset(X[tr], y[tr], feature_name=cols),
                             num_boost_round=rounds).predict(X)
        ph = acc / len(seeds)
        zs[h] = zscore(ph, ph[va])
        log(f"[h={h}] done")

    y3 = (close[3:3 + nmax] > close[:nmax]).astype(np.int8)

    rows = {}
    z3 = zs[3]
    evaluate("h3_only", z3, ct, y3, va, rows)
    evaluate("mean_all", np.mean([zs[h] for h in HS], axis=0), ct, y3, va, rows)

    near = [h for h in HS if h <= 3]
    if len(near) > 1:
        evaluate("mean_near", np.mean([zs[h] for h in near], axis=0), ct, y3, va, rows)
    far = [h for h in HS if h >= 3]
    if len(far) > 1:
        evaluate("mean_far", np.mean([zs[h] for h in far], axis=0), ct, y3, va, rows)

    signs = np.array([np.sign(zs[h]) for h in HS])
    agree = np.abs(signs.sum(axis=0)) / len(HS)
    evaluate("mean_agree", np.mean([zs[h] for h in HS], axis=0) * (0.5 + 0.5 * agree),
             ct, y3, va, rows)

    log(f"\n{'变体':<12}{'static':>9}{'AUC':>8}{'按天':>8}{'窗口4':>8}{'4段sd':>8}")
    for k, r in rows.items():
        log(f"{k:<12}{r['static']:>9.4f}{r['auc']:>8.4f}{r['daily']:>8.4f}"
            f"{r['w4']:>8.4f}{r['w4sd']:>8.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
