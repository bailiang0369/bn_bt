"""第八轮：把「形态」本身变成特征，而不是让模型从统计量里猜形态。

依据：morph/morph.py 的 faiss 形态聚类只用价格窗口形状 + stoch 就到 60+，
说明窗口形状里有 63 个统计特征没表达出来的信息。前几轮试过"形态子集"
（只挑已有特征）和"CNN 直接吃形态"（弱、过拟合），但没试过把形态量化成
离散簇、再用簇的历史胜率去喂 LGB —— 这是形态与统计真正的结合点。

做法（严格时间隔离，无未来信息）：
  1. 窗口形状：最近 60 根 1m 的 close，按 6 根采样成 10 个点，除以当前 close
     得到相对路径（bps），只描述形状不含量纲；
  2. MiniBatchKMeans(64) 只在 train 段抽样拟合，然后给全部样本打簇号；
  3. 簇的历史胜率做**前向编码**：train 内部切成 5 段，第 k 段只用前 k-1 段
     统计（第 1 段用全局 prior），val/test 用完整 train 统计 —— 避免把样本
     自己的标签喂给自己；
  4. 把这 2 个特征（簇胜率、簇样本量）追加到 63 特征上重训 LGB，与基线对比。
"""
import os
import sys

import numpy as np
import polars as pl
from numpy.lib.stride_tricks import sliding_window_view

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lightgbm as lgb  # noqa: E402
from sklearn.cluster import MiniBatchKMeans  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from morph.kline_boost import (DAILY_K, OB_END, PARAMS, TEST_START, VAL_START,  # noqa: E402
                               build_features, complete_days, daily_topk_eval,
                               day_index, global_topk_eval, load_klines, log,
                               make_target, zscore)

W = 60
STEP = 6
NCL = 64
PRIOR_N = 50.0


def shape_paths(close, n):
    win = sliding_window_view(close, W)          # (n-W+1, W)
    samp = np.ascontiguousarray(win[:, ::STEP][:, :10])
    path = (samp / samp[:, -1:] - 1.0) * 1e4
    out = np.zeros((n, path.shape[1]), dtype=np.float32)
    out[W - 1:] = path
    return out


def cluster_ids(paths, tr, seeds=0):
    km = MiniBatchKMeans(n_clusters=NCL, batch_size=8192, n_init=3, random_state=seeds)
    pool = tr[::max(1, len(tr) // 200_000)]
    km.fit(paths[pool])
    return km.predict(paths)


def forward_encode(cl, y, ct, tr, va, te, n_seg=5):
    """前向编码：train 分 n_seg 段，第 k 段只用前 k-1 段统计；val/test 用完整 train。"""
    prior = float(y[tr].mean())
    rate = np.full(len(y), prior, dtype=np.float32)
    cntf = np.zeros(len(y), dtype=np.float32)
    bounds = np.linspace(0, len(tr), n_seg + 1).astype(int)
    for k in range(n_seg):
        seg = tr[bounds[k]:bounds[k + 1]]
        hist = tr[:bounds[k]] if k > 0 else tr[:0]
        if len(hist) < 10_000:
            rate[seg] = prior
            cntf[seg] = 0.0
            continue
        cnt = np.bincount(cl[hist], minlength=NCL).astype(np.float64)
        pos = np.bincount(cl[hist], weights=y[hist].astype(np.float64), minlength=NCL)
        r = (pos + prior * PRIOR_N) / (cnt + PRIOR_N)
        rate[seg] = r[cl[seg]].astype(np.float32)
        cntf[seg] = np.log1p(cnt)[cl[seg]].astype(np.float32)
    cnt = np.bincount(cl[tr], minlength=NCL).astype(np.float64)
    pos = np.bincount(cl[tr], weights=y[tr].astype(np.float64), minlength=NCL)
    r = (pos + prior * PRIOR_N) / (cnt + PRIOR_N)
    for idx in (va, te):
        rate[idx] = r[cl[idx]].astype(np.float32)
        cntf[idx] = np.log1p(cnt)[cl[idx]].astype(np.float32)
    return rate, cntf


def fit(X, y, tr, cols, seeds, rounds):
    acc = np.zeros(len(y))
    for sd in seeds:
        p = dict(PARAMS, seed=sd, bagging_seed=sd, feature_fraction_seed=sd,
                 data_random_seed=sd)
        acc += lgb.train(p, lgb.Dataset(X[tr], y[tr], feature_name=cols),
                         num_boost_round=rounds).predict(X)
    return acc / len(seeds)


def evaluate(tag, prob, ct, y, va, rows):
    te = np.where(ct >= TEST_START)[0]
    z = zscore(prob, prob[va])
    dd = day_index(ct[te])
    r = {"static": global_topk_eval(z[te], y[te], dd, DAILY_K),
         "auc": float(roc_auc_score(y[te], z[te])),
         "daily": float(daily_topk_eval(z[te], y[te], dd,
                                        complete_days(dd, 1200), DAILY_K).mean())}
    edges = np.linspace(int(ct[te[0]]), int(ct[te[-1]]) + 1, 5).astype(np.int64)
    wv = []
    for i in range(4):
        m = (ct[te] >= edges[i]) & (ct[te] < edges[i + 1])
        idx = te[m]
        if len(idx):
            wv.append(global_topk_eval(z[idx], y[idx], day_index(ct[idx]), DAILY_K))
    r["w4"], r["w4sd"] = float(np.mean(wv)), float(np.std(wv, ddof=1))
    rows[tag] = r
    log(f"[{tag}] static={r['static']:.4f} AUC={r['auc']:.4f} 按天={r['daily']:.4f} "
        f"窗口4={r['w4']:.4f}±{r['w4sd']:.4f}")
    return r


def main():
    seeds = [int(x) for x in os.environ.get("SEEDS", "42,43,44").split(",")]
    rounds = int(os.environ.get("ROUNDS", "60"))

    raw = load_klines().filter(pl.col("close_time") <= OB_END)
    d = make_target(raw, 3)
    ft, cols = build_features(d)
    ct = ft["close_time"].to_numpy()
    y = ft["target"].to_numpy().astype(np.int8)
    close = ft["close"].to_numpy().astype(np.float64)
    n = len(y)
    log(f"[data] rows={n} cols={len(cols)}")

    tr = np.where(ct < VAL_START)[0]
    va = np.where((ct >= VAL_START) & (ct < TEST_START))[0]
    te = np.where(ct >= TEST_START)[0]
    tr = tr[tr + 3 < int(va[0])]
    va = va[va + 3 < int(te[0])]

    paths = shape_paths(close, n)
    log(f"[shape] W={W} step={STEP} dim={paths.shape[1]}")
    cl = cluster_ids(paths, tr)
    log(f"[cluster] 簇数={len(np.unique(cl[tr]))} 最大簇占比="
        f"{np.bincount(cl[tr]).max() / len(tr):.3f}")
    rate, cntf = forward_encode(cl, y, ct, tr, va, te)
    log(f"[encode] 簇胜率 train 范围={rate[tr].min():.3f}~{rate[tr].max():.3f}")

    X0 = ft.select(cols).to_numpy().astype(np.float32)
    X1 = np.hstack([X0, rate.reshape(-1, 1), cntf.reshape(-1, 1)])
    del X0, ft

    rows = {}
    evaluate("base63", fit(X1[:, :len(cols)], y, tr, cols, seeds, rounds), ct, y, va, rows)
    evaluate("shape65", fit(X1, y, tr, cols + ["shape_rate", "shape_cnt"],
                            seeds, rounds), ct, y, va, rows)

    log(f"\n{'变体':<10}{'static':>9}{'AUC':>8}{'按天':>8}{'窗口4':>8}{'4段sd':>8}")
    for k, r in rows.items():
        log(f"{k:<10}{r['static']:>9.4f}{r['auc']:>8.4f}{r['daily']:>8.4f}"
            f"{r['w4']:>8.4f}{r['w4sd']:>8.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
