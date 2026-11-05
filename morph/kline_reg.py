"""第七轮：换损失函数。分类学"方向概率"，回归直接学"3 分钟收益幅度"。

理由：topK 评估要的是 |score| 最大的那 1% 样本的准确率。分类器的概率极端值
不一定对应真实的大幅波动；回归器直接预测收益（bps），|pred| 天然就是"这笔
波动有多大"的置信度，方向取 sign(pred)。两者的 topK 排序逻辑不同，值得对比。

同时测时间衰减权重（近期样本权重高），对抗 2020~2026 的 regime 漂移。

变体（各 2 seed）：
  clf          基线分类（对照）
  reg          回归 L2，标签 = 3min 收益(bps)
  clf_decay    分类 + 半衰期 180 天权重
  reg_decay    回归 + 半衰期 180 天权重
  comb         分类方向 × 回归幅度（z 相乘）
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
                               day_index, global_topk_eval, load_klines, log,
                               make_target, zscore)

DAY = 86_400_000
TAU = 180 * DAY


def fit_scores(X, label, tr, cols, seeds, rounds, objective, weight=None):
    acc = np.zeros(X.shape[0])
    for sd in seeds:
        p = dict(PARAMS, objective=objective,
                 metric="l2" if objective == "regression" else "auc",
                 seed=sd, bagging_seed=sd, feature_fraction_seed=sd, data_random_seed=sd)
        ds = lgb.Dataset(X[tr], label[tr], weight=None if weight is None else weight[tr],
                         feature_name=cols)
        acc += lgb.train(p, ds, num_boost_round=rounds).predict(X)
    return acc / len(seeds)


def evaluate(tag, s, ct, y, rows):
    """s 的符号即方向，|s| 即置信度（不做中心化，回归输出必须保留原始零点）。"""
    va = np.where((ct >= VAL_START) & (ct < TEST_START))[0]
    te = np.where(ct >= TEST_START)[0]
    va = va[va + 3 < int(te[0])]
    dd = day_index(ct[te])
    r = {"static": global_topk_eval(s[te], y[te], dd, DAILY_K),
         "auc": float(roc_auc_score(y[te], s[te])),
         "daily": float(daily_topk_eval(s[te], y[te], dd,
                                        complete_days(dd, 1200), DAILY_K).mean())}
    edges = np.linspace(int(ct[te[0]]), int(ct[te[-1]]) + 1, 5).astype(np.int64)
    w = []
    for i in range(4):
        m = (ct[te] >= edges[i]) & (ct[te] < edges[i + 1])
        idx = te[m]
        if len(idx):
            w.append(global_topk_eval(s[idx], y[idx], day_index(ct[idx]), DAILY_K))
    r["w4"], r["w4sd"] = float(np.mean(w)), float(np.std(w, ddof=1))
    r["va_static"] = global_topk_eval(s[va], y[va], day_index(ct[va]), DAILY_K)
    rows[tag] = r
    log(f"[{tag}] static={r['static']:.4f} (val {r['va_static']:.4f}) "
        f"AUC={r['auc']:.4f} 按天={r['daily']:.4f} 窗口4={r['w4']:.4f}±{r['w4sd']:.4f}")
    return r


def main():
    seeds = [int(x) for x in os.environ.get("SEEDS", "42,43").split(",")]
    rounds = int(os.environ.get("ROUNDS", "60"))

    raw = load_klines().filter(pl.col("close_time") <= OB_END)
    d = (raw.with_columns(pl.col("close").shift(-3).alias("_fc"))
         .with_columns(((pl.col("_fc") / pl.col("close") - 1.0) * 1e4).alias("fret_bps"))
         .with_columns((pl.col("_fc") > pl.col("close")).cast(pl.Int8).alias("target"))
         .drop("_fc").filter(pl.col("target").is_not_null()))
    ft, cols = build_features(d)
    ct = ft["close_time"].to_numpy()
    y = ft["target"].to_numpy().astype(np.int8)
    fret = np.nan_to_num(ft["fret_bps"].to_numpy())
    X = ft.select(cols).to_numpy().astype(np.float32)
    del ft, d
    log(f"[data] rows={X.shape[0]} cols={len(cols)}")

    tr = np.where(ct < VAL_START)[0]
    va = np.where((ct >= VAL_START) & (ct < TEST_START))[0]
    tr = tr[tr + 3 < int(va[0])]
    w = np.exp(-(ct[int(tr[-1])] - ct) / TAU)
    log(f"[weight] 衰减权重 最新={w[tr].max():.3f} 最旧={w[tr].min():.4f} "
        f"均值={w[tr].mean():.3f}")

    rows, store = {}, {}
    store["clf"] = fit_scores(X, y, tr, cols, seeds, rounds, "binary")
    evaluate("clf", store["clf"], ct, y, rows)

    store["reg"] = fit_scores(X, fret, tr, cols, seeds, rounds, "regression")
    evaluate("reg", store["reg"], ct, y, rows)

    store["clf_decay"] = fit_scores(X, y, tr, cols, seeds, rounds, "binary", weight=w)
    evaluate("clf_decay", store["clf_decay"], ct, y, rows)

    store["reg_decay"] = fit_scores(X, fret, tr, cols, seeds, rounds, "regression", weight=w)
    evaluate("reg_decay", store["reg_decay"], ct, y, rows)

    # 分类定方向、回归定幅度：置信度相乘，符号沿用分类
    zc = zscore(store["clf"], store["clf"][va])
    mag = np.abs(store["reg"]) / (np.abs(store["reg"][va]).mean() + 1e-9)
    comb = np.sign(zc) * np.abs(zc) * mag
    evaluate("comb", comb, ct, y, rows)

    log(f"\n{'变体':<12}{'static':>9}{'val':>9}{'AUC':>8}{'按天':>8}{'窗口4':>8}{'4段sd':>8}")
    for k, r in rows.items():
        log(f"{k:<12}{r['static']:>9.4f}{r['va_static']:>9.4f}{r['auc']:>8.4f}"
            f"{r['daily']:>8.4f}{r['w4']:>8.4f}{r['w4sd']:>8.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
