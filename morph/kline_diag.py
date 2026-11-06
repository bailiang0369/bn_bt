"""K 线腿诊断（Kaggle）：为什么 5.5 年 / 3.5M 行只值 0.52 AUC。

与盘口腿用**完全相同的时间边界**（BTC orderbook 的 64/80 切分点）：
  val_start  = 1777704120000
  test_start = 1784298120000
主口径 global_topk_eval（K=15×天数），z-score 用 **val**（与盘口腿一致）。

诊断三件事：
1) 标签尺度：h=3/5/10/15/30 各自 AUC 与 topK —— 判断瓶颈是"3 分钟本身太随机"
   还是特征不行；
2) 标准化口径：train vs val z-score 差多少；
3) 特征有效性：gain 重要性 top20 与零增益特征数。
"""
import glob
import io
import os
import subprocess
import sys
import zipfile

import numpy as np
import polars as pl

try:
    import lightgbm as lgb
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "lightgbm"], check=True)
    import lightgbm as lgb

from sklearn.metrics import roc_auc_score

REPO = "https://github.com/bailiang0369/bn_bt"
ROOT = "/tmp/bn_bt"
if not os.path.isdir(ROOT):
    subprocess.run(["git", "clone", "--depth=1", REPO, ROOT], check=True)
sys.path.insert(0, ROOT)
from morph.kline_features import build_features  # noqa: E402

CACHE = "/kaggle/working/btcusdt_1m_full.parquet"
CSV_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
            "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]
PARAMS = dict(objective="binary", metric="auc", learning_rate=0.05, num_leaves=63,
              feature_fraction=0.7, bagging_fraction=0.9, bagging_freq=1,
              min_child_samples=300, lambda_l2=1.0, verbose=-1)
VAL_START = 1777704120000   # BTC orderbook ct[int(0.64*L)]
TEST_START = 1784298120000  # BTC orderbook ct[int(0.80*L)]
OB_END = 1792540620000
DAILY_K = 15


def log(m):
    print(m, flush=True)


def load_klines(start=(2020, 1), end=(2026, 9), symbol="BTCUSDT"):
    if os.path.exists(CACHE):
        log(f"[cache] {CACHE}")
        return pl.read_parquet(CACHE).sort("close_time")
    hits = glob.glob("/kaggle/input/**/btcusdt_1m_full.parquet", recursive=True)
    if hits:
        log(f"[dataset] {hits[0]}")
        return pl.read_parquet(hits[0]).sort("close_time")

    base = "https://data.binance.vision/data/futures/um/monthly/klines"
    frames, y, m = [], start[0], start[1]
    while (y, m) <= end:
        period = f"{y:04d}-{m:02d}"
        url = f"{base}/{symbol}/1m/{symbol}-1m-{period}.zip"
        try:
            import requests
            r = requests.get(url, timeout=120)
            if r.status_code != 200:
                log(f"  [dl] {period} HTTP {r.status_code}")
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
                    log(f"  [dl] {period} ok ({d.height})")
        except Exception as e:
            log(f"  [dl] {period} 出错: {e}")
        m += 1
        if m > 12:
            m, y = 1, y + 1
    df = pl.concat(frames, how="vertical_relaxed").sort("close_time")
    log(f"[dl] 合计 {df.height} 行 {int(df['close_time'][0])} ~ {int(df['close_time'][-1])}")
    try:
        df.write_parquet(CACHE)
        log("[cache] saved")
    except Exception as e:
        log(f"[cache] save failed: {e}")
    return df


# --- 复制自 orderbook/fusion_test.py ---
def day_index(ct_ms):
    return (np.asarray(ct_ms) // 86_400_000).astype(np.int64)


def complete_days(day_id, min_bars=1200):
    u, c = np.unique(day_id, return_counts=True)
    return np.isin(day_id, u[c >= min_bars])


def global_topk_eval(score, y, day_id, k_per_day):
    n_days = len(np.unique(day_id))
    k = min(int(k_per_day) * n_days, len(y))
    idx = np.argsort(-np.abs(score))[:k]
    return float(((score[idx] > 0).astype(int) == y[idx]).mean())


def daily_topk_eval(score, y, day_id, mask, k):
    s, yy, d = score[mask], y[mask], day_id[mask]
    accs = []
    for u in np.unique(d):
        m = d == u
        kk = min(k, int(m.sum()))
        idx = np.argsort(-np.abs(s[m]))[:kk]
        accs.append(float(((s[m][idx] > 0).astype(int) == yy[m][idx]).mean()))
    return np.array(accs, dtype=np.float64)


def main():
    hs = [int(x) for x in os.environ.get("HS", "3,5,10,15,30").split(",")]
    seeds = [int(x) for x in os.environ.get("SEEDS", "42").split(",")]
    rounds = int(os.environ.get("ROUNDS", "60"))
    stride = int(os.environ.get("STRIDE", "1"))

    df = load_klines().filter(pl.col("close_time") <= OB_END)
    log(f"[kline] rows={df.height} 时间 {int(df['close_time'][0])} ~ {int(df['close_time'][-1])}")

    featured, cols = build_features(df)
    ct = featured["close_time"].to_numpy()
    close = featured["close"].to_numpy().astype(np.float64)
    day = day_index(ct)
    X = featured.select(cols).to_numpy().astype(np.float32)
    log(f"[feats] {len(cols)} 列 X={X.shape}")
    del featured, df

    for h in hs:
        y = np.zeros(len(close), dtype=np.int8)
        y[:-h] = (close[h:] > close[:-h]).astype(np.int8)
        valid = np.arange(len(close)) < len(close) - h

        tr = np.where(valid & (ct < VAL_START))[0]
        va = np.where(valid & (ct >= VAL_START) & (ct < TEST_START))[0]
        te = np.where(valid & (ct >= TEST_START))[0]
        tr = tr[tr + h < int(va[0])][::stride]
        va = va[va + h < int(te[0])]

        probs = np.zeros(len(y), dtype=np.float64)
        last = None
        for sd in seeds:
            p = dict(PARAMS, seed=sd, bagging_seed=sd,
                     feature_fraction_seed=sd, data_random_seed=sd)
            ds = lgb.Dataset(X[tr], y[tr], feature_name=cols)
            last = lgb.train(p, ds, num_boost_round=rounds)
            probs += last.predict(X)
        p = (probs / len(seeds)).astype(np.float32)

        z_val = (p - p[va].mean()) / (p[va].std() + 1e-9)
        z_tr = (p - p[tr].mean()) / (p[tr].std() + 1e-9)
        dte = day[te]
        mt = complete_days(dte)
        n_te = len(np.unique(dte))
        k = DAILY_K * n_te
        auc = float(roc_auc_score(y[te], z_val[te]))
        gk = global_topk_eval(z_val[te], y[te], dte, DAILY_K)
        dk = float(daily_topk_eval(z_val[te], y[te], dte, mt, DAILY_K).mean())
        gk_tr = global_topk_eval(z_tr[te], y[te], dte, DAILY_K)
        base = float(y[te].mean())
        log(f"\n[h={h}] train={len(tr)} test={len(te)} 天数={n_te} K={k} 基准={base:.4f}")
        log(f"[h={h}] AUC={auc:.4f}  topK(val z)={gk:.4f}  按天={dk:.4f}  "
            f"topK(train z)={gk_tr:.4f}")

        imp = sorted(zip(cols, last.feature_importance("gain")),
                     key=lambda x: -x[1])
        zero = sum(1 for _, v in imp if v <= 0)
        log(f"[h={h}] gain top12: " + ", ".join(f"{n}:{v:.0f}" for n, v in imp[:12]))
        log(f"[h={h}] 零增益特征 {zero}/{len(cols)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
