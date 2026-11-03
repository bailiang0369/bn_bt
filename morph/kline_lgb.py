"""3min 方向模型（纯 K 线，5.5 年数据）——把没用上的 4.4 倍数据利用起来。

背景：orderbook 只有 2025-07 起的 15 个月（658,712 行），3min 已到 0.60~0.62 上限；
而 1m K 线有 2020-01 起的 3,549,600 行，此前 faiss/morph 只用 close + stoch 两个通道。
本模块用 60+ 个 K 线特征（动量/波动/量能/形态/时间）在全部历史上训练，验证单腿水平。

数据：Binance Vision futures/um 月度 1m klines（含 volume / taker_buy_volume），
首次运行下载 81 个月（约 400MB），之后从 /kaggle/working 缓存读取。

用法：
    python -m morph.kline_lgb
    python -m morph.kline_lgb --offsets 0,60,120 --seeds 42,43,44
"""
import io
import os
import sys
import glob
import argparse
import zipfile
from pathlib import Path

import numpy as np
import polars as pl
import lightgbm as lgb
import requests
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from morph.kline_features import build_features
from orderbook.fusion_test import day_index, complete_days, daily_topk_eval, acc_at_cov

CACHE = Path("/kaggle/working/btcusdt_1m_full.parquet") if os.name != "nt" \
    else ROOT / "data" / "raw" / "btcusdt_1m_full.parquet"
CSV_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
            "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]
PARAMS = dict(objective="binary", metric="auc", learning_rate=0.05, num_leaves=63,
              feature_fraction=0.7, bagging_fraction=0.9, bagging_freq=1,
              min_child_samples=300, lambda_l2=1.0, verbose=-1)


def load_klines(start=(2020, 1), end=(2026, 9), symbol="BTCUSDT"):
    if CACHE.exists():
        print(f"[cache] {CACHE}")
        return pl.read_parquet(CACHE).sort("close_time")
    hits = glob.glob("/kaggle/input/**/btcusdt_1m_full.parquet", recursive=True)
    if hits:
        print(f"[dataset] {hits[0]}")
        return pl.read_parquet(hits[0]).sort("close_time")

    base = "https://data.binance.vision/data/futures/um/monthly/klines"
    frames, y, m = [], start[0], start[1]
    while (y, m) <= end:
        period = f"{y:04d}-{m:02d}"
        url = f"{base}/{symbol}/1m/{symbol}-1m-{period}.zip"
        try:
            r = requests.get(url, timeout=90)
            if r.status_code != 200:
                print(f"  [dl] {period} HTTP {r.status_code}")
            else:
                z = zipfile.ZipFile(io.BytesIO(r.content))
                name = [n for n in z.namelist() if n.endswith(".csv")][0]
                raw = io.BytesIO(z.read(name))
                head = raw.readline().decode("utf-8", "replace")
                raw.seek(0)
                if head.split(",")[0].strip() == "open_time":
                    d = pl.read_csv(raw, has_header=True)
                else:
                    d = pl.read_csv(raw, has_header=False, new_columns=CSV_COLS)
                frames.append(d.select("close_time", "open", "high", "low", "close",
                                       "volume", "taker_buy_volume"))
                if m in (1, 7):
                    print(f"  [dl] {period} ok ({d.height})")
        except Exception as e:
            print(f"  [dl] {period} 出错: {e}")
        m += 1
        if m > 12:
            m, y = 1, y + 1
    df = pl.concat(frames, how="vertical_relaxed").sort("close_time")
    print(f"[dl] 合计 {df.height} 行 {df['close_time'][0]} ~ {df['close_time'][-1]}")
    try:
        df.write_parquet(CACHE)
        print(f"[cache] saved {CACHE}")
    except Exception as e:
        print(f"[cache] save failed: {e}")
    return df


def train_fixed(X, y, tr, va, cols, seeds, rounds):
    probs = np.zeros(len(y), dtype=np.float64)
    last = None
    for sd in seeds:
        params = dict(PARAMS, seed=sd, bagging_seed=sd,
                      feature_fraction_seed=sd, data_random_seed=sd)
        ds = lgb.Dataset(X[tr], y[tr], feature_name=cols)
        m = lgb.train(params, ds, num_boost_round=rounds)
        probs += m.predict(X)
        last = m
    return (probs / len(seeds)).astype(np.float32), last


def main():
    ap = argparse.ArgumentParser(description="3min 纯 K 线特征模型")
    ap.add_argument("--horizon", type=int, default=int(os.environ.get("KAGGLE_HORIZON", "3")))
    ap.add_argument("--seeds", default=os.environ.get("KAGGLE_SEEDS", "42,43,44"))
    ap.add_argument("--rounds", type=int, default=60)
    ap.add_argument("--stride", type=int, default=2, help="训练集抽样步长")
    ap.add_argument("--daily-k", type=int, default=15)
    ap.add_argument("--test-days", type=int, default=92)
    ap.add_argument("--val-days", type=int, default=30)
    ap.add_argument("--offsets", default=os.environ.get("KAGGLE_TEST_OFFSETS", "0"))
    a = ap.parse_args()

    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    offsets = [int(s) for s in a.offsets.split(",") if s.strip()]
    h = a.horizon

    df = load_klines()
    print(f"[kline] rows={df.height}")
    df = df.with_columns((pl.col("close").shift(-h) > pl.col("close"))
                         .cast(pl.Int8).alias("target"))
    df = df.filter(pl.col("target").is_not_null())

    featured, cols = build_features(df)
    X = featured.select(cols).to_numpy().astype(np.float32)
    y = featured["target"].to_numpy().astype(np.int8)
    ct = featured["close_time"].to_numpy()
    print(f"[feats] {len(cols)} 列  X={X.shape}")

    day_ms = 86_400_000
    end = int(ct[-1])
    for off in offsets:
        end_off = end - off * day_ms
        test_start = end_off - a.test_days * day_ms
        val_start = test_start - a.val_days * day_ms
        tr = np.where(ct < val_start)[0]
        va = np.where((ct >= val_start) & (ct < test_start))[0]
        te = np.where((ct >= test_start) & (ct <= end_off))[0]
        if len(va) == 0 or len(te) == 0:
            print(f"[off={off}] 段为空，跳过")
            continue
        tr = tr[tr + h < int(va[0])][::a.stride]
        va = va[va + h < int(te[0])]
        import time as _t
        fmt = lambda ms: _t.strftime("%Y-%m-%d", _t.gmtime(ms / 1000))
        print(f"\n[off={off}] train={len(tr)} val={len(va)} test={len(te)} "
              f"test {fmt(ct[te[0]])} ~ {fmt(ct[te[-1]])}")

        p, m = train_fixed(X, y, tr, va, cols, seeds, a.rounds)
        mu, sd = float(p[tr].mean()), float(p[tr].std() + 1e-9)
        z = (p - mu) / sd
        d_te = day_index(ct[te])
        m_te = complete_days(d_te)
        auc = float(roc_auc_score(y[te], z[te]))
        acc1, _ = acc_at_cov(z[te], y[te], 0.01)
        dk = float(daily_topk_eval(z[te], y[te], d_te, m_te, a.daily_k).mean())
        print(f"[off={off}] AUC={auc:.4f} 全局1%={acc1:.4f} 按天top{a.daily_k}={dk:.4f} "
              f"({len(np.unique(d_te))} 天)")

        if off == 0:
            imp = sorted(zip(cols, m.feature_importance("gain")),
                         key=lambda x: -x[1])[:20]
            print("[top20 gain] " + ", ".join(f"{n}:{v:.0f}" for n, v in imp))

    print("\n[说明] 这是纯 K 线单腿；能到 0.58+ 就有融合价值（orderbook 腿 0.60~0.62）。")


if __name__ == "__main__":
    main()
