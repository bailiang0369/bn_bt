"""价格形态聚类存在性证明（统一版，对应 faiss v6）。

仅用 close（价格形态）+ stoch，多尺度定长窗口聚类；聚类只在训练集 fit，
训练集算每簇历史胜率 edge，ace_score = 各尺度 edge 之和；
按 |ace_score| 取前 X% 覆盖率报准确率。

修复原 v6 的 dict 合并 bug：价格簇 / stoch 簇用不同前缀（p_ / s_），可分别与合并评估。

数据：用 download_klines.py 从公开 Binance Vision 拉取的 1m K 线月度 parquet：
    data/um/monthly/klines/{SYMBOL}/1m/*.parquet
（BTCUSDT / ETHUSDT 均可，symbol 参数切换）。

用法：
    python morph.py --symbol ETHUSDT                 # 单资产 80/20
    python morph.py --symbol BTCUSDT --split-date 2025-01-01   # 严格前向
    python morph.py --compare                         # BTC vs ETH 全面对比
"""
import gc
import sys
import os
import argparse
import numpy as np
import polars as pl
import faiss
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
KLINE_DIR = ROOT / "data" / "um" / "monthly" / "klines"

HYBRID_CONFIG = {
    'v_short_10': {'window': 10, 'n_clusters': 100, 'repeat': 0},
    'short_15':   {'window': 15, 'n_clusters': 120, 'repeat': 10},
    'short_30':   {'window': 30, 'n_clusters': 150, 'repeat': 15},
    'mid_60':     {'window': 60, 'n_clusters': 300, 'repeat': 15},
    'long_120':   {'window': 120, 'n_clusters': 600, 'repeat': 15},
}


def load_klines(symbol: str) -> pl.DataFrame:
    base = KLINE_DIR / symbol / "1m"
    files = sorted(base.glob("*.parquet"))
    if not files:
        raise SystemExit(
            f"[load_klines] 在 {base} 没找到 {symbol} 的 K 线 parquet。\n"
            f"先运行: python download_klines.py --market um --symbol {symbol} "
            f"--category klines --interval 1m --granularity monthly "
            f"--start-date 2020-01-01 --end-date 2026-09-30 -j 8")
    print(f"[load_klines] {symbol}: {len(files)} 个月文件 "
          f"{files[0].name} .. {files[-1].name}")
    df = pl.concat([pl.read_parquet(f) for f in files], how="vertical_relaxed")
    return df.select("close_time", "open", "high", "low", "close").sort("close_time")


def add_target(df: pl.DataFrame, horizon: int = 15) -> pl.DataFrame:
    # 1 分钟 bar，shift(-horizon) 即 horizon 分钟后的涨跌方向
    return df.with_columns(
        (pl.col("close").shift(-horizon) > pl.col("close")).cast(pl.Int8).alias("target")
    )


def stoch(df: pl.DataFrame, period: int) -> np.ndarray:
    denom = pl.col("high").rolling_max(period) - pl.col("low").rolling_min(period)
    k = ((pl.col("close") - pl.col("low").rolling_min(period)) /
         pl.when(denom == 0).then(1e-9).otherwise(denom)) * 100.0
    out = df.select(k).to_numpy().reshape(-1)
    return np.nan_to_num(out, nan=50.0).astype(np.float32)


def get_feats(arr: np.ndarray, w: int, r_count: int, end_k: int = 3,
              is_indicator: bool = False) -> np.ndarray:
    arr = np.ascontiguousarray(arr, dtype=np.float32).ravel()
    shape = (len(arr) - w + 1, w)
    strides = (arr.strides[0], arr.strides[0])
    windows = np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)
    if is_indicator:
        norm = windows / 100.0
    else:
        wmin = windows.min(1, keepdims=True)
        wmax = windows.max(1, keepdims=True)
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
        tr_ids[f"{prefix}_cluster_{name}"] = np.concatenate(
            [np.full(w - 1, -1, np.int32), tr_i.flatten()])
        te_ids[f"{prefix}_cluster_{name}"] = np.concatenate(
            [np.full(w - 1, -1, np.int32), te_i.flatten()])
        del tr_f, te_f
        gc.collect()
    return tr_ids, te_ids


def add_edges(train_final, test_final, cluster_cols):
    for col in cluster_cols:
        mapping = (train_final.filter(pl.col(col) != -1)
                   .group_by(col)
                   .agg((pl.col("target").mean() - 0.5).alias(f"edge_{col}")))
        test_final = test_final.join(mapping, on=col, how="left")
    return test_final


def run_experiment(symbol: str, split_date=None, horizon: int = 15) -> dict:
    df = add_target(load_klines(symbol), horizon)
    n = len(df)
    if split_date is None:
        k = int(n * 0.8)
        tag = "80/20"
    else:
        end_ms = int(datetime.strptime(split_date, "%Y-%m-%d").timestamp() * 1000)
        k = int((df["close_time"].to_numpy().ravel() < end_ms).sum())
        tag = f"train<{split_date}"
    print(f"[split] {symbol} {tag}  train={k}  test={n - k}")
    train_df, test_df = df.head(k), df.tail(n - k)
    tr_close = train_df["close"].to_numpy().ravel().astype(np.float32)
    te_close = test_df["close"].to_numpy().ravel().astype(np.float32)

    tr_p, te_p = run_clustering(tr_close, te_close, HYBRID_CONFIG, "p", False)
    stoch29 = stoch(df, 29)
    stoch69 = stoch(df, 69)
    tr_s, te_s = run_clustering(stoch29[:k].astype(np.float32),
                                stoch29[k:].astype(np.float32),
                                HYBRID_CONFIG, "s", True)

    all_ids = {**tr_p, **tr_s}
    all_te = {**te_p, **te_s}
    train_final = train_df.with_columns([pl.Series(kk, v) for kk, v in all_ids.items()])
    test_final = test_df.with_columns([pl.Series(kk, v) for kk, v in all_te.items()])

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

    clean = test_final.filter(pl.col("target").is_not_null())
    base = clean.select(pl.col("target").mean()).item()

    def acc_of(score_col, cov):
        cnt = int(len(clean) * cov)
        if cnt <= 0:
            return None
        sub = (clean.with_columns(pl.col(score_col).abs().alias("_a"))
               .sort("_a", descending=True).head(cnt))
        return sub.select(
            pl.when(pl.col(score_col) > 0).then(pl.col("target") == 1)
            .otherwise(pl.col("target") == 0).mean()).item()

    rows = {}
    for label, col in [("close only", "ace_p"),
                      ("stoch only", "ace_s"),
                      ("close+stoch", "ace")]:
        rows[label] = {cov: acc_of(col, cov)
                       for cov in [0.01, 0.02, 0.05, 0.10, 0.25]}
    return {"symbol": symbol, "tag": tag, "n_test": len(clean),
            "base": base, "rows": rows, "test_final": clean}


def print_detail(r: dict):
    print(f"\n=== {r['symbol']} [{r['tag']}]  n_test={r['n_test']}  "
          f"基准(target=1)={r['base']:.2%} ===")
    for label in ["close only", "stoch only", "close+stoch"]:
        print(f"--- {label} ---")
        for cov in [0.01, 0.02, 0.05, 0.10, 0.25]:
            acc = r["rows"][label][cov]
            if acc is None:
                continue
            cnt = int(r["n_test"] * cov)
            print(f"  {cov:>5.0%}  覆盖{cnt:>8d}  准确率{acc:7.2%}")


def compare(symbols, split_date=None):
    modes = [(None, "80/20"), ("2025-01-01", "forward(25-)")]
    print("\n" + "=" * 86)
    print("形态聚类全面对比  horizon=15min  （覆盖率对应 |ace_score| 排序的前 X% 最自信样本）")
    print("=" * 86)
    hdr = (f"{'资产':<10}{'切分':<13}{'基准':>8}{'close@1%':>10}"
           f"{'close@5%':>10}{'close@25%':>11}{'comb@1%':>10}{'comb@25%':>10}")
    print(hdr)
    for sym in symbols:
        for sd, tag in modes:
            r = run_experiment(sym, sd)
            c = r["rows"]["close only"]
            m = r["rows"]["close+stoch"]
            print(f"{sym:<10}{tag:<13}{r['base']:>7.2%}"
                  f"{c[0.01]:>10.2%}{c[0.05]:>10.2%}{c[0.25]:>11.2%}"
                  f"{m[0.01]:>10.2%}{m[0.25]:>10.2%}")
        print("-" * 86)
    # 详细分覆盖率（便于核对）
    for sym in symbols:
        for sd, tag in modes:
            print_detail(run_experiment(sym, sd))


def main():
    ap = argparse.ArgumentParser(description="价格形态聚类存在性证明")
    ap.add_argument("--symbol", default="BTCUSDT", help="BTCUSDT / ETHUSDT")
    ap.add_argument("--split-date", default=None,
                    help="严格前向切分：训练<该日期，测试>=该日期（如 2025-01-01）")
    ap.add_argument("--compare", action="store_true",
                    help="BTC vs ETH 全面对比（80/20 + 前向）")
    ap.add_argument("--horizon", type=int, default=15)
    args = ap.parse_args()

    if args.compare:
        compare(["BTCUSDT", "ETHUSDT"], args.split_date)
    else:
        r = run_experiment(args.symbol, args.split_date, args.horizon)
        print_detail(r)


if __name__ == "__main__":
    main()
