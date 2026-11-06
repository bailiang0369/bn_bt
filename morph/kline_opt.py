"""K 线腿优化（Kaggle）：针对 §26 诊断的三处问题。

诊断发现：
  - 数据只下到 2026-09，比 orderbook 少 20 天 -> 融合 join 丢样本；
  - h=10 标签的 topK(0.6509) 反超 h=3(0.6070)，说明 3 分钟尺度噪声大；
  - k_boll_60 gain 一家独大（约为第二名的 5 倍），模型接近单因子。

对应三个变体（评测标签**统一固定 h=3**，与盘口腿可融合）：
  v0 h3 单模型（基线）
  v1 h10 单模型，直接用它的分数评 h=3 标签
  v2 h3+h5+h10 三模型 z-score 等权平均
  v3 单特征 k_boll_60（验证是否退化为单因子）
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
VAL_START = 1777704120000
TEST_START = 1784298120000
OB_END = 1792540620000
DAILY_K = 15


def log(m):
    print(m, flush=True)


def load_klines(start=(2020, 1), end=(2026, 10), symbol="BTCUSDT"):
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
    seeds = [int(x) for x in os.environ.get("SEEDS", "42,43").split(",")]
    rounds = int(os.environ.get("ROUNDS", "60"))
    hs_ens = [int(x) for x in os.environ.get("ENS_HS", "3,5,10").split(",")]
    max_h = max(hs_ens)

    df = load_klines().filter(pl.col("close_time") <= OB_END)
    log(f"[kline] rows={df.height} 时间 {int(df['close_time'][0])} ~ {int(df['close_time'][-1])}")

    featured, cols = build_features(df)
    ct = featured["close_time"].to_numpy()
    close = featured["close"].to_numpy().astype(np.float64)
    n = len(close)
    day = day_index(ct)
    X = featured.select(cols).to_numpy().astype(np.float32)
    log(f"[feats] {len(cols)} 列 X={X.shape}")
    del featured, df

    # 各 horizon 的标签；评测统一用 h=3
    ys = {}
    for h in set(hs_ens) | {3}:
        t = np.zeros(n, dtype=np.int8)
        t[:-h] = (close[h:] > close[:-h]).astype(np.int8)
        ys[h] = t
    valid = np.arange(n) < n - max_h

    tr = np.where(valid & (ct < VAL_START))[0]
    va = np.where(valid & (ct >= VAL_START) & (ct < TEST_START))[0]
    te = np.where(valid & (ct >= TEST_START))[0]
    tr = tr[tr + max_h < int(va[0])]
    va = va[va + max_h < int(te[0])]
    dte = day[te]
    mt = complete_days(dte)
    n_te = len(np.unique(dte))
    y3 = ys[3]
    log(f"[split] train={len(tr)} val={len(va)} test={len(te)} 天数={n_te} "
        f"K={DAILY_K*n_te} 基准={y3[te].mean():.4f}")
    log(f"[split] test 时间 {int(ct[te[0]])} ~ {int(ct[te[-1]])}")
    import time as _t
    fmt = lambda ms: _t.strftime("%Y-%m-%d", _t.gmtime(ms / 1000))
    log(f"[split] test {fmt(ct[te[0]])} ~ {fmt(ct[te[-1]])}")

    pred_idx = np.concatenate([va, te])
    nv, nt = len(va), len(te)
    X_tr = X[tr]
    X_pr = X[pred_idx]

    def fit(sub, ylab):
        """sub=None 用全部特征；否则是特征下标列表。只在 val+test 上预测。"""
        ftr = X_tr if sub is None else X_tr[:, sub]
        fpr = X_pr if sub is None else X_pr[:, sub]
        fname = cols if sub is None else [cols[i] for i in sub]
        probs = np.zeros(len(pred_idx), dtype=np.float64)
        for sd in seeds:
            p = dict(PARAMS, seed=sd, bagging_seed=sd,
                     feature_fraction_seed=sd, data_random_seed=sd)
            m = lgb.train(p, lgb.Dataset(ftr, ylab[tr], feature_name=fname),
                          num_boost_round=rounds)
            probs += m.predict(fpr)
        return (probs / len(seeds)).astype(np.float32)

    def zs(p):
        return (p - p[:nv].mean()) / (p[:nv].std() + 1e-9)

    label_variants = {}
    for h in hs_ens:
        label_variants[h] = zs(fit(None, ys[h]))
        log(f"[fit] h={h} 完成")

    # v3 单特征 k_boll_60
    z1 = None
    if "k_boll_60" in cols:
        z1 = zs(fit([cols.index("k_boll_60")], y3))
        log("[fit] k_boll_60 单特征完成")

    cands = {f"v0 h{min(hs_ens)} 单模型": label_variants[min(hs_ens)]}
    if max(hs_ens) != min(hs_ens):
        cands[f"v1 h{max(hs_ens)} 单模型"] = label_variants[max(hs_ens)]
    cands[f"v2 h{'+'.join(map(str, hs_ens))} 集成"] = np.mean(
        [label_variants[h] for h in hs_ens], axis=0)
    if z1 is not None:
        cands["v3 仅 k_boll_60"] = z1

    log(f"\n{'变体（评测标签固定 h=3）':<24}{'AUC':>8}{'topK':>8}{'按天':>8}{'≥65%天':>9}")
    for name, z in cands.items():
        zt = z[nv:]
        auc = float(roc_auc_score(y3[te], zt))
        gk = global_topk_eval(zt, y3[te], dte, DAILY_K)
        accs = daily_topk_eval(zt, y3[te], dte, mt, DAILY_K)
        log(f"{name:<24}{auc:>8.4f}{gk:>8.4f}{accs.mean():>8.4f}"
            f"{float((accs >= 0.65).mean()):>9.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
