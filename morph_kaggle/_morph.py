"""形态聚类存在性证明（本地 Python 版，对应 faiss v6）。

仅用 close（价格形态）+ stoch，多尺度定长窗口聚类，train-only 算簇 edge，
ace_score = 各尺度 edge 之和；按 |ace_score| 取前 X% 覆盖率报准确率。
修复原 v6 的 dict 合并 bug：价格簇 / stoch 簇用不同前缀，可分别与合并评估。
"""
import gc
import numpy as np
import polars as pl
import faiss
from datetime import datetime

HYBRID_CONFIG = {
    'v_short_10': {'window': 10, 'n_clusters': 100, 'repeat': 0},
    'short_15':   {'window': 15, 'n_clusters': 120, 'repeat': 10},
    'short_30':   {'window': 30, 'n_clusters': 150, 'repeat': 15},
    'mid_60':     {'window': 60, 'n_clusters': 300, 'repeat': 15},
    'long_120':   {'window': 120, 'n_clusters': 600, 'repeat': 15},
}


def load_df():
    df = pl.read_parquet("data/BTCUSDT.parquet").sort("close_time")
    return df


def add_target(df, horizon=15):
    # 1 分钟 bar，horizon 根后 = horizon 分钟后的涨跌方向
    return df.with_columns(
        (pl.col("close").shift(-horizon) > pl.col("close")).cast(pl.Int8).alias("target")
    )


def stoch(df, period):
    denom = pl.col("high").rolling_max(period) - pl.col("low").rolling_min(period)
    k = ((pl.col("close") - pl.col("low").rolling_min(period)) /
         pl.when(denom == 0).then(1e-9).otherwise(denom)) * 100.0
    out = df.select(k).to_numpy().reshape(-1)
    return np.nan_to_num(out, nan=50.0).astype(np.float32)


def get_feats(arr, w, r_count, end_k=3, is_indicator=False):
    shape = (len(arr) - w + 1, w)
    strides = (arr.strides[0], arr.strides[0])
    windows = np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)
    if is_indicator:
        norm = windows / 100.0
    else:
        wmin = windows.min(1, keepdims=True); wmax = windows.max(1, keepdims=True)
        norm = (windows - wmin) / np.where((wmax - wmin) == 0, 1e-9, wmax - wmin)
    feat = np.hstack([norm] + [norm[:, -end_k:]] * r_count) if r_count > 0 else norm
    return np.ascontiguousarray(feat, dtype="float32")


def run_clustering(tr_data, te_data, config, prefix, is_indicator=False):
    tr_ids, te_ids = {}, {}
    for name, cfg in config.items():
        w, n_c, r = cfg["window"], cfg["n_clusters"], cfg["repeat"]
        tr_f = get_feats(tr_data, w, r, is_indicator=is_indicator)
        te_f = get_feats(te_data, w, r, is_indicator=is_indicator)
        km = faiss.Kmeans(tr_f.shape[1], n_c, niter=40)
        km.train(tr_f)
        _, tr_i = km.index.search(tr_f, 1)
        _, te_i = km.index.search(te_f, 1)
        tr_ids[f"{prefix}_cluster_{name}"] = np.concatenate([np.full(w - 1, -1, np.int32), tr_i.flatten()])
        te_ids[f"{prefix}_cluster_{name}"] = np.concatenate([np.full(w - 1, -1, np.int32), te_i.flatten()])
        del tr_f, te_f; gc.collect()
    return tr_ids, te_ids


def add_edges(train_final, test_final, cluster_cols):
    for col in cluster_cols:
        mapping = (train_final.filter(pl.col(col) != -1)
                   .group_by(col).agg((pl.col("target").mean() - 0.5).alias(f"edge_{col}")))
        test_final = test_final.join(mapping, on=col, how="left")
    return test_final


def report(df, score_col, title):
    clean = df.filter(pl.col("target").is_not_null())
    total = len(clean)
    base = clean.select(pl.col("target").mean()).item()
    print(f"\n--- {title} ---  n={total}  基准(target=1)={base:.2%}")
    for cov in [0.01, 0.02, 0.05, 0.10, 0.25]:
        cnt = int(total * cov)
        if cnt <= 0:
            continue
        sub = (clean.with_columns(pl.col(score_col).abs().alias("_a"))
               .sort("_a", descending=True).head(cnt))
        acc = sub.select(
            pl.when(pl.col(score_col) > 0).then(pl.col("target") == 1)
            .otherwise(pl.col("target") == 0).mean()).item()
        print(f"  {cov:>5.0%}  覆盖{cnt:>7d}  准确率{acc:7.2%}")


def main(horizon=15, split_date=None):
    df = add_target(load_df(), horizon)
    n = len(df)
    if split_date is None:
        k = int(n * 0.8)
        tag = "80/20 索引切分"
    else:
        end_ms = int(datetime.strptime(split_date, "%Y-%m-%d").timestamp() * 1000)
        k = int((df["close_time"].to_numpy() < end_ms).sum())
        tag = f"日期切分 train<{split_date} / test>={split_date}"
    print(f"[split] {tag}  train={k}  test={n - k}")
    train_df, test_df = df.head(k), df.tail(n - k)
    tr_close = train_df["close"].to_numpy().astype(np.float32)
    te_close = test_df["close"].to_numpy().astype(np.float32)

    tr_p, te_p = run_clustering(tr_close, te_close, HYBRID_CONFIG, "p", False)
    stoch29 = stoch(df, 29)
    stoch69 = stoch(df, 69)
    tr_s, te_s = run_clustering(stoch29[:k].astype(np.float32), stoch29[k:].astype(np.float32),
                                HYBRID_CONFIG, "s", True)

    all_ids = {**tr_p, **tr_s}
    all_te = {**te_p, **te_s}
    train_final = train_df.with_columns([pl.Series(k, v) for k, v in all_ids.items()])
    test_final = test_df.with_columns([pl.Series(k, v) for k, v in all_te.items()])

    cluster_cols = [c for c in train_final.columns if "cluster_" in c]
    test_final = add_edges(train_final, test_final, cluster_cols)
    edge_cols = [c for c in test_final.columns if "edge_" in c]
    p_edges = [c for c in edge_cols if c.startswith("edge_p_")]
    s_edges = [c for c in edge_cols if c.startswith("edge_s_")]
    test_final = test_final.with_columns(
        ace_p=pl.sum_horizontal([pl.col(c).fill_null(0) for c in p_edges]),
        ace_s=pl.sum_horizontal([pl.col(c).fill_null(0) for c in s_edges]),
        ace=pl.sum_horizontal([pl.col(c).fill_null(0) for c in edge_cols]),
    )
    print(f"[config] horizon={horizon}min  价格簇={len(p_edges)}  stoch簇={len(s_edges)}  (原v6只用到其中一组)")
    report(test_final, "ace_p", "价格形态(close) only")
    report(test_final, "ace_s", "stoch only")
    report(test_final, "ace", "close + stoch 合并")


if __name__ == "__main__":
    main(15)
