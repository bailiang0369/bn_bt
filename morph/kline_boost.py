"""纯 K 线预测提升（Kaggle）。

背景：此前纯 K 线 63 特征 LGB 只有 topK 0.6049(静态)/0.5694(roll_1d)。
但 `morph/morph.py` 的 faiss 形态聚类（只用价格窗口形状 + stoch）能到 60+，
说明**形态本身**是有效信号，63 个技术指标反而把形状拆散成了统计量。

按三个方向做，统一时间切分（val_start/test_start 取自 orderbook 的 64/80）与评估口径：
  v0  1m 基线：63 特征，预测 3 分钟后
  v1  3m 重采样：1m -> 3m bar，在新 bar 上重算特征，预测**下一根**
  v2  1m + 符号累加：sign(ret) 滚动净和 + 连续同号游程（点数图视角）
  v3  1m 形态子集：只留价格形态类，去掉量能/时间
  v4  v1+v2 堆叠：3m 与 1m 符号版的 z-score 平均
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


def _repo_root():
    """优先复用已有仓库（bootstrap 已 clone 到 /kaggle/working/bn_bt），最后才 clone。"""
    here = os.path.dirname(os.path.abspath(__file__))
    for c in (os.path.dirname(here), "/kaggle/working/bn_bt", "/tmp/bn_bt"):
        if os.path.isfile(os.path.join(c, "morph", "kline_features.py")):
            return c
    subprocess.run(["git", "clone", "--depth=1", REPO, "/tmp/bn_bt"], check=True)
    return "/tmp/bn_bt"


ROOT = _repo_root()
sys.path.insert(0, ROOT)
from morph.kline_features import build_features  # noqa: E402

CACHE = "/kaggle/working/btcusdt_1m_full.parquet"
# Kaggle dataset byh850515/eth-btc-1m-2020 里的文件名是 btcusdt_1m.parquet（含 volume/taker）
KLINE_NAMES = ("btcusdt_1m_full.parquet", "btcusdt_1m.parquet", "btc_1m.parquet")
CSV_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
            "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]
PARAMS = dict(objective="binary", metric="auc", learning_rate=0.05, num_leaves=63,
              feature_fraction=0.7, bagging_fraction=0.9, bagging_freq=1,
              min_child_samples=300, lambda_l2=1.0, verbose=-1)
VAL_START = 1777704120000
TEST_START = 1784298120000
OB_END = 1792540620000
DAILY_K = 15
SHAPE_PREFIX = ("k_ret_", "k_body", "k_upper", "k_lower", "k_gap", "k_stoch_",
                "k_boll_", "k_pos_", "k_rsi_", "k_streak")
SIGN_COLS = ([f"sgn_sum_{w}" for w in (5, 10, 30, 60, 120)]
             + [f"sgn_bal_{w}" for w in (5, 10, 30, 60, 120)]
             + ["run_len", "run_sgn"])


def log(m):
    print(m, flush=True)


def load_klines(start=(2020, 1), end=(2026, 10), symbol="BTCUSDT"):
    """优先用 Kaggle dataset / 本地已有 parquet；全部缺失才联网下载。"""
    cands = [CACHE, os.path.join(ROOT, "data", "raw", "btcusdt_1m_full.parquet")]
    for n in KLINE_NAMES:
        cands += sorted(glob.glob(f"/kaggle/input/**/{n}", recursive=True))
    for p in cands:
        if os.path.exists(p):
            d = pl.read_parquet(p).sort("close_time")
            log(f"[data] {p} rows={d.height} cols={d.columns}")
            if "volume" not in d.columns:
                log("[data] 警告：无 volume/taker_buy_volume，量能特征将缺失")
            return d
    log("[data] 未找到现成 parquet，开始下载 Binance Vision 月度 zip")
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
                d = (pl.read_csv(raw, has_header=True) if head.split(",")[0].strip() == "open_time"
                     else pl.read_csv(raw, has_header=False, new_columns=CSV_COLS))
                frames.append(d.select("close_time", "open", "high", "low", "close",
                                       "volume", "taker_buy_volume"))
        except Exception as e:
            log(f"  [dl] {period} 出错: {e}")
        m += 1
        if m > 12:
            m, y = 1, y + 1
    df = pl.concat(frames, how="vertical_relaxed").sort("close_time")
    log(f"[dl] 合计 {df.height} 行")
    try:
        df.write_parquet(CACHE)
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


def roll_z(x, win):
    s = pl.Series("x", x)
    mp = max(win // 4, 30)
    mu = s.rolling_mean(win, min_samples=mp).shift(1)
    sd = s.rolling_std(win, min_samples=mp).shift(1)
    return ((s - mu) / (sd + 1e-9)).fill_nan(0.0).fill_null(0.0).to_numpy()


def resample_3m(df):
    t0 = int(df["close_time"][0])
    return (df.with_columns(((pl.col("close_time") - t0) // 180_000).alias("_g"))
            .group_by("_g")
            .agg(pl.col("close_time").last(), pl.col("open").first(), pl.col("high").max(),
                 pl.col("low").min(), pl.col("close").last(),
                 pl.col("volume").sum(), pl.col("taker_buy_volume").sum())
            .sort("close_time"))


def make_target(d, h):
    d = d.with_columns(pl.col("close").shift(-h).alias("_fc"))
    return (d.with_columns((pl.col("_fc") > pl.col("close")).cast(pl.Int8).alias("target"))
            .drop("_fc").filter(pl.col("target").is_not_null()))


def add_sign_feats(d):
    """+1/-1 累加视角：符号滚动净和、同号游程长度（点数图/游程）。"""
    r = pl.col("close") / pl.col("close").shift(1) - 1.0
    sgn = (pl.when(r > 0).then(1).when(r < 0).then(-1).otherwise(0)).cast(pl.Int8)
    d = d.with_columns(sgn.alias("_sgn"))
    exprs = []
    for w in (5, 10, 30, 60, 120):
        exprs.append(pl.col("_sgn").rolling_sum(w).cast(pl.Float32).alias(f"sgn_sum_{w}"))
        exprs.append((pl.col("_sgn").rolling_sum(w) / w).cast(pl.Float32).alias(f"sgn_bal_{w}"))
    d = d.with_columns(exprs)
    chg = (pl.col("_sgn") != pl.col("_sgn").shift(1)).cast(pl.Int32).cum_sum()
    d = d.with_columns(chg.alias("_grp"))
    d = d.with_columns((pl.int_range(pl.len()).over("_grp") + 1).cast(pl.Float32).alias("run_len"))
    d = d.with_columns((pl.col("_sgn") * pl.col("run_len")).cast(pl.Float32).alias("run_sgn"))
    return d.drop(["_sgn", "_grp"])


def zscore(x, ref):
    return (x - ref.mean()) / (ref.std() + 1e-9)


def align_to(src_ct, src, dst_ct):
    """把 src（按 src_ct 递增）前向填充到 dst_ct 网格上。"""
    if len(src_ct) == 0:
        return np.zeros(len(dst_ct))
    idx = np.clip(np.searchsorted(src_ct, dst_ct, side="right") - 1, 0, len(src_ct) - 1)
    return np.asarray(src, dtype=np.float64)[idx]


def score_eval(tag, s, ct, y, h, min_bars, bpd, nfeat, results):
    """统一评估：静态 / 1日滚动 / 3日滚动 z-score 的全局 topK、AUC、按天 topK。"""
    va = np.where((ct >= VAL_START) & (ct < TEST_START))[0]
    te = np.where(ct >= TEST_START)[0]
    va = va[va + h < int(te[0])]
    zs = zscore(s, s[va])
    dyn = day_index(ct)
    dte = dyn[te]
    mt = complete_days(dte, min_bars=min_bars)
    row = {"feats": nfeat, "n": len(te), "days": len(np.unique(dte)),
           "base": float(y[te].mean())}
    for nm, z in [("static", zs), ("roll1d", roll_z(s, bpd)), ("roll3d", roll_z(s, 3 * bpd))]:
        row[nm] = global_topk_eval(z[te], y[te], dte, DAILY_K)
    row["auc"] = float(roc_auc_score(y[te], zs[te]))
    row["daily"] = float(daily_topk_eval(zs[te], y[te], dte, mt, DAILY_K).mean())
    results[tag] = row
    log(f"[{tag}] feats={nfeat} n={row['n']} 天={row['days']} 基准={row['base']:.4f} | "
        f"static={row['static']:.4f} roll1d={row['roll1d']:.4f} "
        f"roll3d={row['roll3d']:.4f} AUC={row['auc']:.4f} 按天={row['daily']:.4f}")
    return row


def main():
    seeds = [int(x) for x in os.environ.get("SEEDS", "42,43").split(",")]
    rounds = int(os.environ.get("ROUNDS", "60"))
    only = [x for x in os.environ.get("ONLY", "").split(",") if x]
    want = (lambda t: (not only) or t in only)

    raw = load_klines().filter(pl.col("close_time") <= OB_END)
    log(f"[1m] rows={raw.height} {int(raw['close_time'][0])}~{int(raw['close_time'][-1])}")
    raw3 = resample_3m(raw)
    log(f"[3m] rows={raw3.height}")

    cache_feat = {}

    def feats(key, d, h):
        k = f"{key}_h{h}"
        if k not in cache_feat:
            cache_feat[k] = build_features(make_target(d, h))
        return cache_feat[k]

    results, store = {}, {}

    def run(tag, ft, cols, h, min_bars, bpd):
        ct = ft["close_time"].to_numpy()
        y = ft["target"].to_numpy().astype(np.int8)
        X = ft.select(cols).to_numpy().astype(np.float32)
        tr = np.where(ct < VAL_START)[0]
        va = np.where((ct >= VAL_START) & (ct < TEST_START))[0]
        te = np.where(ct >= TEST_START)[0]
        tr = tr[tr + h < int(va[0])]
        probs = np.zeros(len(y))
        for sd in seeds:
            p = dict(PARAMS, seed=sd, bagging_seed=sd,
                     feature_fraction_seed=sd, data_random_seed=sd)
            m = lgb.train(p, lgb.Dataset(X[tr], y[tr], feature_name=cols),
                          num_boost_round=rounds)
            probs += m.predict(X)
        s = (probs / len(seeds)).astype(np.float32)
        score_eval(tag, s, ct, y, h, min_bars, bpd, len(cols), results)
        store[tag] = (s, ct, y, h, min_bars, bpd)
        return s, ct

    def sign_frame():
        return add_sign_feats(ft1).select(["close_time", "target"] + c1 + SIGN_COLS)

    V0, V1, V2, V3, V4 = "v0 1m_63f", "v1 3m_63f_next", "v2 1m_signacc", "v3 1m_shape", \
        "v4 3m+1m_sign"
    need1 = {"v0", "v2", "v3", "v4"}
    ft1, c1 = feats("1m", raw, 3) if (not only or need1 & set(only)) else (None, None)
    ft3, c3 = feats("3m", raw3, 1) if (not only or {"v1", "v4"} & set(only)) else (None, None)

    if want("v0"):
        run(V0, ft1, c1, 3, 1200, 1440)

    if want("v1"):
        run(V1, ft3, c3, 1, 400, 480)

    if want("v2"):
        run(V2, sign_frame(), c1 + SIGN_COLS, 3, 1200, 1440)

    if want("v3"):
        run(V3, ft1, [c for c in c1 if c.startswith(SHAPE_PREFIX)], 3, 1200, 1440)

    if want("v4"):
        if V1 not in store:
            run(V1, ft3, c3, 1, 400, 480)
        if V2 not in store:
            run(V2, sign_frame(), c1 + SIGN_COLS, 3, 1200, 1440)
        s3, ct3 = store[V1][:2]
        s1, ct1 = store[V2][:2]
        y3, h3 = store[V1][2], store[V1][3]
        # 3m 网格上对齐：1m 分数前向填充，两侧在 val 上各自 z-score 后等权平均
        s1a = align_to(ct1, s1, ct3)
        ref = (ct3 >= VAL_START) & (ct3 < TEST_START)
        mix = 0.5 * zscore(s3, s3[ref]) + 0.5 * zscore(s1a, s1a[ref])
        nf = f"{len(c3)}+{len(c1) + len(SIGN_COLS)}"
        score_eval(V4, mix, ct3, y3, h3, 400, 480, nf, results)

    log(f"\n{'变体':<18}{'特征':>6}{'static':>9}{'roll1d':>9}{'roll3d':>9}"
        f"{'AUC':>8}{'按天':>8}")
    for k, r in results.items():
        log(f"{k:<18}{r['feats']:>6}{r['static']:>9.4f}{r['roll1d']:>9.4f}"
            f"{r['roll3d']:>9.4f}{r['auc']:>8.4f}{r['daily']:>8.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
