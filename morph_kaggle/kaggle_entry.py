"""Kaggle 入口（CPU-only，不使用 GPU；单文件自包含，Kaggle 只上传 code_file）。

内容：
  1) pip 安装 faiss-cpu / polars / numpy / requests（必须在第三方 import 之前）；
  2) 从公开 Binance Vision 下载 ETH/BTC 2020+ 1 分钟 K 线（futures/um，月度 zip，需联网）；
  3) 价格形态聚类存在性证明（close + stoch，多尺度定长窗口，train-only 算簇 edge）；
  4) 在 Kaggle 同环境计算 BTCUSDT，与 ETHUSDT 做对比；
  5)【重采样实验】把 1 分钟聚成 2/3/5 分钟，验证：
       A) 其他时间框架的形态是否也有 edge；
       B) 把多框架样本池化进训练集，是否提升 1 分钟测试的准确率。

注：ETH/BTC 的 1m 数据均落 dataset；首次下载后，后续 kernel 直接挂载读取，
  不再重复联网下载（脚本用 glob 自动发现 /kaggle/input/**/*_1m.parquet）。
"""
import sys
import io
import gc
import datetime as _dt
import subprocess
import os
import glob

# Kaggle 默认不预装 faiss-cpu / polars，必须在任何第三方 import 之前先装好
print(">>> 安装依赖: faiss-cpu polars numpy requests")
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "faiss-cpu", "polars", "numpy", "requests"], check=False)

import numpy as np
import polars as pl
import faiss
import requests
import zipfile


# ===================== 1) K 线下载（仅 klines，futures/um，月度） =====================
def download_klines(symbol, start_date="2020-01-01", end_date="2026-09-30"):
    BASE = "https://data.binance.vision/data/futures/um/monthly/klines"
    start = _dt.date.fromisoformat(start_date)
    end = _dt.date.fromisoformat(end_date)
    y, m = start.year, start.month
    cols = ["open_time", "open", "high", "low", "close", "volume", "close_time",
            "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]
    frames = []
    while (y, m) <= (end.year, end.month):
        period = f"{y:04d}-{m:02d}"
        url = f"{BASE}/{symbol}/1m/{symbol}-1m-{period}.zip"
        try:
            r = requests.get(url, timeout=60)
            if r.status_code != 200:
                print(f"  [dl] {symbol} {period} 跳过 (HTTP {r.status_code})")
            else:
                z = zipfile.ZipFile(io.BytesIO(r.content))
                csv_name = [n for n in z.namelist() if n.endswith(".csv")][0]
                raw = io.BytesIO(z.read(csv_name))
                first = raw.readline().decode("utf-8", "replace")
                raw.seek(0)
                if first.split(",")[0].strip() == "open_time":
                    df = pl.read_csv(raw, has_header=True)
                else:
                    df = pl.read_csv(raw, has_header=False, new_columns=cols)
                df = df.select("close_time", "open", "high", "low", "close", "volume", "taker_buy_volume")
                frames.append(df)
                print(f"  [dl] {symbol} {period} ok ({df.height} 行)")
        except Exception as e:
            print(f"  [dl] {symbol} {period} 出错: {e}")
        m += 1
        if m > 12:
            m, y = 1, y + 1
    total = sum(f.height for f in frames)
    print(f"[dl] {symbol} 合计 {total} 行")
    return pl.concat(frames, how="vertical_relaxed").sort("close_time")


def prepare_data(symbols):
    """优先从已挂载的 dataset 读取各资产的 1 分钟 K 线；缺失的则下载并存盘，
    供后续据此创建 dataset、避免每次重下。返回 {symbol: df}。"""
    out = {}
    for sym in symbols:
        key = sym.lower()
        cands = glob.glob(f"/kaggle/input/**/{key}_1m.parquet", recursive=True)
        if cands:
            p = sorted(cands)[0]
            df = pl.read_parquet(p)
            if "volume" not in df.columns:
                print(f">>> dataset 缺少 volume 列，转下载 {sym} ...")
                df = download_klines(sym)
                try:
                    df.write_parquet(f"/kaggle/working/{key}_1m.parquet")
                    print(f">>> 已保存 /kaggle/working/{key}_1m.parquet")
                except Exception as e:
                    print(f">>> 保存 {key}_1m.parquet 失败（可忽略）: {e}")
            else:
                print(f">>> 从 dataset 载入 {sym}: {p}")
            out[sym] = df
        else:
            print(f">>> 未挂载 dataset，下载 {sym} 2020+ 1m K 线（公开 Binance Vision，需联网）")
            df = download_klines(sym)
            out[sym] = df
            try:
                df.write_parquet(f"/kaggle/working/{key}_1m.parquet")
                print(f">>> 已保存 /kaggle/working/{key}_1m.parquet")
            except Exception as e:
                print(f">>> 保存 {key}_1m.parquet 失败（可忽略）: {e}")
    return out


# ===================== 2) 形态聚类管线（与本地 morph.py 完全一致） =====================
HYBRID_CONFIG = {
    'v_short_10': {'window': 10, 'n_clusters': 100, 'repeat': 0},
    'short_15':   {'window': 15, 'n_clusters': 120, 'repeat': 10},
    'short_30':   {'window': 30, 'n_clusters': 150, 'repeat': 15},
    'mid_60':     {'window': 60, 'n_clusters': 300, 'repeat': 15},
    'long_120':   {'window': 120, 'n_clusters': 600, 'repeat': 15},
}
COVERAGES = [0.01, 0.02, 0.05, 0.10, 0.25]


def add_target(df, horizon=15):
    return df.with_columns(
        (pl.col("close").shift(-horizon) > pl.col("close")).cast(pl.Int8).alias("target"))


def stoch(df, period):
    denom = pl.col("high").rolling_max(period) - pl.col("low").rolling_min(period)
    k = ((pl.col("close") - pl.col("low").rolling_min(period)) /
         pl.when(denom == 0).then(1e-9).otherwise(denom)) * 100.0
    return np.nan_to_num(df.select(k).to_numpy().reshape(-1), nan=50.0).astype(np.float32)


def cvd_signal(df):
    """累计净主动买压力(CVD) 1D 信号，与 df 行对齐。
    signed_i = 2*taker_buy_volume_i - volume_i (= 净taker买 - 净taker卖)，
    再 cumsum 得到净流向积分路径；喂给 get_feats 做窗内 min-max 形状通道。"""
    vol = df["volume"].to_numpy().ravel().astype(np.float32)
    tb = df["taker_buy_volume"].to_numpy().ravel().astype(np.float32)
    signed = 2.0 * tb - vol
    return np.cumsum(signed).astype(np.float32)


def get_feats(arr, w, r_count, end_k=3, is_indicator=False):
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


def split_df(df, split_date=None, horizon=15):
    n = len(df)
    if split_date is None:
        k = int(n * 0.8)
        tag = "80/20"
    else:
        end_ms = int(_dt.datetime.strptime(split_date, "%Y-%m-%d").timestamp() * 1000)
        k = int((df["close_time"].to_numpy().ravel() < end_ms).sum())
        tag = f"train<{split_date}"
    return df.head(k), df.tail(n - k), tag, k


def eval_rows(clean, n_test):
    base = clean.select(pl.col("target").mean()).item()

    def acc_of(score_col, cov):
        cnt = int(n_test * cov)
        if cnt <= 0:
            return None
        sub = (clean.with_columns(pl.col(score_col).abs().alias("_a"))
               .sort("_a", descending=True).head(cnt))
        return sub.select(pl.when(pl.col(score_col) > 0).then(pl.col("target") == 1)
                         .otherwise(pl.col("target") == 0).mean()).item()

    rows = {}
    for label, col in [("close only", "ace_p"),
                      ("stoch only", "ace_s"),
                      ("close+stoch", "ace_ps"),
                      ("close+stoch+cvd", "ace")]:
        rows[label] = {cov: acc_of(col, cov) for cov in COVERAGES}
    return base, rows


def run_experiment(df, symbol, split_date=None, horizon=15, label=None, verbose=True):
    df = add_target(df, horizon)
    train_df, test_df, tag, k = split_df(df, split_date, horizon)
    if verbose:
        print(f"\n[split] {label or symbol} {tag}  train={k}  test={len(df) - k}")
    tr_close = train_df["close"].to_numpy().ravel().astype(np.float32)
    te_close = test_df["close"].to_numpy().ravel().astype(np.float32)
    tr_p, te_p = run_clustering(tr_close, te_close, HYBRID_CONFIG, "p", False)
    stoch29 = stoch(df, 29)
    stoch69 = stoch(df, 69)
    tr_s, te_s = run_clustering(stoch29[:k].astype(np.float32),
                                stoch29[k:].astype(np.float32), HYBRID_CONFIG, "s", True)
    cvd = cvd_signal(df)
    tr_c, te_c = run_clustering(cvd[:k].astype(np.float32),
                                cvd[k:].astype(np.float32), HYBRID_CONFIG, "c", False)
    all_ids = {**tr_p, **tr_s, **tr_c}
    all_te = {**te_p, **te_s, **te_c}
    train_final = train_df.with_columns([pl.Series(kk, v) for kk, v in all_ids.items()])
    test_final = test_df.with_columns([pl.Series(kk, v) for kk, v in all_te.items()])
    cluster_cols = [c for c in train_final.columns if "cluster_" in c]
    test_final = add_edges(train_final, test_final, cluster_cols)
    edge_cols = [c for c in test_final.columns if "edge_" in c]
    p_edges = [c for c in edge_cols if c.startswith("edge_p_")]
    s_edges = [c for c in edge_cols if c.startswith("edge_s_")]
    c_edges = [c for c in edge_cols if c.startswith("edge_c_")]
    test_final = test_final.with_columns(
        ace_p=pl.sum_horizontal([pl.col(c).fill_null(0) for c in p_edges]),
        ace_s=pl.sum_horizontal([pl.col(c).fill_null(0) for c in s_edges]),
        ace_ps=pl.sum_horizontal([pl.col(c).fill_null(0) for c in (p_edges + s_edges)]),
        ace=pl.sum_horizontal([pl.col(c).fill_null(0) for c in (p_edges + s_edges + c_edges)]))
    clean = test_final.filter(pl.col("target").is_not_null())
    base, rows = eval_rows(clean, len(clean))
    return {"symbol": symbol, "tag": tag, "n_test": len(clean),
            "base": base, "rows": rows}


# ===================== 3) 重采样（1 分钟 -> k 分钟） =====================
def resample(df, k):
    """把 1 分钟 K 线按索引聚成 k 分钟：open=首, high=最大, low=最小, close=末。"""
    if k == 1:
        return df
    d = df.with_columns((pl.arange(0, pl.len()) // k).alias("bucket"))
    out = (d.group_by("bucket", maintain_order=True)
             .agg(pl.col("open").first(), pl.col("high").max(),
                  pl.col("low").min(), pl.col("close").last(),
                  pl.col("close_time").last(),
                  pl.col("volume").sum(), pl.col("taker_buy_volume").sum())
             .drop("bucket")
             .sort("close_time"))
    return out


# ===================== 4) 池化训练实验（多框架样本混入训练集） =====================
def run_pooled(df1m, scales=(1, 2, 3, 5), horizon=15):
    """把多个时间框架的训练样本池化进同一个聚类空间，只在 1 分钟测试集上评估。
    用于验证：把聚合数据加入训练集，是否提升 1 分钟测试的准确率。"""
    prep = {}
    for k in scales:
        rk = resample(df1m, k)
        d = add_target(rk, horizon)
        n = len(d)
        cut = int(n * 0.8)
        st = stoch(rk, 29)
        prep[k] = (d.head(cut), d.tail(n - cut), st)

    def _channel(prep, scales, use_stoch):
        edge_maps, te_ids = {}, {}
        for name, cfg in HYBRID_CONFIG.items():
            w, r, nc = cfg["window"], cfg["repeat"], cfg["n_clusters"]
            Xtr, ytr, te1 = [], [], None
            for k in scales:
                tr, te, st = prep[k]
                if use_stoch:
                    tr_arr = st[:len(tr)].astype(np.float32)
                    te_arr = st[len(tr):].astype(np.float32)
                else:
                    tr_arr = tr["close"].to_numpy().ravel().astype(np.float32)
                    te_arr = te["close"].to_numpy().ravel().astype(np.float32)
                Xtr.append(get_feats(tr_arr, w, r, is_indicator=use_stoch))
                # ytr 必须与 get_feats 输出对齐：get_feats 丢掉每尺度前 w-1 行
                ytr.append(tr["target"].to_numpy().ravel().astype(np.int8)[w - 1:])
                if k == 1:
                    te1 = get_feats(te_arr, w, r, is_indicator=use_stoch)
            Xtr = np.concatenate(Xtr, 0)
            ytr = np.concatenate(ytr, 0)
            assert Xtr.shape[0] == len(ytr), f"len mismatch Xtr={Xtr.shape[0]} ytr={len(ytr)}"
            km = faiss.Kmeans(Xtr.shape[1], nc, niter=40)
            km.train(Xtr)
            _, Itr = km.index.search(Xtr, 1)
            _, Ite = km.index.search(te1, 1)
            m = {}
            for cid in np.unique(Itr.flatten()):
                mask = Itr.flatten() == cid
                m[int(cid)] = float(ytr[mask].mean()) - 0.5 if mask.sum() > 0 else 0.0
            edge_maps[name] = m
            te_ids[name] = np.concatenate([np.full(w - 1, -1, np.int32), Ite.flatten()])
        return edge_maps, te_ids

    em_p, ti_p = _channel(prep, scales, False)
    em_s, ti_s = _channel(prep, scales, True)

    te1_df = prep[1][1]
    for name in HYBRID_CONFIG:
        vals = np.array([em_p[name].get(int(c), 0.0) for c in ti_p[name]], dtype=np.float32)
        te1_df = te1_df.with_columns(pl.Series(f"edge_p_{name}", vals))
        vals = np.array([em_s[name].get(int(c), 0.0) for c in ti_s[name]], dtype=np.float32)
        te1_df = te1_df.with_columns(pl.Series(f"edge_s_{name}", vals))
    edge_cols = [c for c in te1_df.columns if "edge_" in c]
    p_edges = [c for c in edge_cols if c.startswith("edge_p_")]
    s_edges = [c for c in edge_cols if c.startswith("edge_s_")]
    te1_df = te1_df.with_columns(
        ace_p=pl.sum_horizontal([pl.col(c).fill_null(0) for c in p_edges]),
        ace_s=pl.sum_horizontal([pl.col(c).fill_null(0) for c in s_edges]),
        ace_ps=pl.sum_horizontal([pl.col(c).fill_null(0) for c in (p_edges + s_edges)]),
        ace=pl.sum_horizontal([pl.col(c).fill_null(0) for c in edge_cols]))
    clean = te1_df.filter(pl.col("target").is_not_null())
    base, rows = eval_rows(clean, len(clean))
    return {"symbol": "ETH", "tag": "pooled", "n_test": len(clean),
            "base": base, "rows": rows}


# ===================== 5) BTC vs ETH 同环境对比（均在 Kaggle 计算） =====================
def print_comparison(eth_results, btc_results):
    print("\n" + "=" * 92)
    print("BTC vs ETH 形态聚类对比（同环境 Kaggle 计算，horizon=15min）")
    print("=" * 92)
    hdr = (f"{'资产':<10}{'切分':<13}{'基准':>8}{'close@1%':>10}{'close@5%':>10}"
           f"{'close@25%':>11}{'comb@1%':>10}{'comb@25%':>10}")
    print(hdr)
    for tag in ["80/20", "forward(25-)"]:
        for nm, res in [("ETHUSDT", eth_results), ("BTCUSDT", btc_results)]:
            r = res[tag]
            c = r["rows"]["close only"]; m = r["rows"]["close+stoch"]
            print(f"{nm:<10}{tag:<13}{r['base']:>7.2%}{c[0.01]:>10.2%}{c[0.05]:>10.2%}"
                  f"{c[0.25]:>11.2%}{m[0.01]:>10.2%}{m[0.25]:>10.2%}")
        print("-" * 92)


def analyze_resample(df1m, symbol):
    scales = [1, 2, 3, 5]
    print("\n" + "=" * 70)
    print(f"重采样多时间框架形态测试（{symbol}，80/20，horizon=15 根）")
    print("=" * 70)
    hdr = (f"{'周期':<6}{'基准':>8}{'close@1%':>10}{'close@5%':>10}"
           f"{'close@25%':>11}{'comb@1%':>10}{'comb@25%':>10}")
    print(hdr)
    res = {}
    for k in scales:
        r = run_experiment(resample(df1m, k), symbol, None, label=f"{k}m", verbose=False)
        res[k] = r
        c = r["rows"]["close only"]; m = r["rows"]["close+stoch"]
        print(f"{k}m{'':<2}{r['base']:>7.2%}{c[0.01]:>10.2%}{c[0.05]:>10.2%}"
              f"{c[0.25]:>11.2%}{m[0.01]:>10.2%}{m[0.25]:>10.2%}")
    rp = run_pooled(df1m, scales, 15)
    c1 = res[1]["rows"]["close+stoch"]
    cp = rp["rows"]["close+stoch"]
    print("-" * 70)
    print(f"池化训练(1+2+3+5m -> 1m测试): 基准{rp['base']:.2%}  "
          f"comb@1%={cp[0.01]:.2%}  comb@25%={cp[0.25]:.2%}")
    print(f"1m 单独基线:                    "
          f"comb@1%={c1[0.01]:.2%}  comb@25%={c1[0.25]:.2%}")
    delta1 = cp[0.01] - c1[0.01]
    delta25 = cp[0.25] - c1[0.25]
    verdict = "池化>=基线 → 聚合数据有助（稳定/提升）" if (delta1 >= 0 and delta25 >= 0) \
        else "池化未胜出 → 聚合多为冗余/稀释（符合'派生数据非独立'判断）"
    print(f"差值: comb@1% {delta1:+.2%}   comb@25% {delta25:+.2%}")
    print("结论:", verdict)


def main():
    print("\n>>> 准备 ETHUSDT 2020+ 1m 数据（含 volume/taker_buy_volume，优先 dataset）")
    data = prepare_data(["ETHUSDT"])
    df_eth = data["ETHUSDT"]
    # 1) 1 分钟 headline（单币种试验：ETH）+ CVD 通道对照
    eth_results = {}
    for sd, tag in [(None, "80/20"), ("2025-01-01", "forward(25-)")]:
        eth_results[tag] = run_experiment(df_eth, "ETHUSDT", sd, label=f"ETH {tag}")
        r = eth_results[tag]
        print(f"\n=== ETHUSDT [{r['tag']}] n_test={r['n_test']} 基准={r['base']:.2%} ===")
        for label in ["close only", "stoch only", "close+stoch", "close+stoch+cvd"]:
            print(f"--- {label} ---")
            for cov in COVERAGES:
                acc = r["rows"][label][cov]
                if acc is None:
                    continue
                cnt = int(r["n_test"] * cov)
                print(f"  {cov:>5.0%}  覆盖{cnt:>8d}  准确率{acc:7.2%}")
    # 2) 重采样多时间框架实验（用 ETH，同样含 CVD 对照）
    analyze_resample(df_eth, "ETHUSDT")
    print("\n>>> 完成。")


if __name__ == "__main__":
    main()
