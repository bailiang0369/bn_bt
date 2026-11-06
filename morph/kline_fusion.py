"""修复 K 线腿数据缺口后重跑融合（Kaggle）。

背景（§26/§27 诊断）：
  - `fusion_kl` 里 `load_klines(end=(2026,9))` 让 K 线止于 2026-09-30，
    比 orderbook 少 20 天，join 时直接丢样本；本脚本补齐到 2026-10。
  - 两腿用**同一时间边界**（orderbook 的 64%/80% 切分），同一 val z-score，
    同一主口径 global_topk_eval（K=15×天数）。

输出：盘口腿 / K 线腿 / 各权重融合 的 val 与 test，权重只用 val 选。
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
OB_REG = dict(num_leaves=31, min_child_samples=500, feature_fraction=0.6, lambda_l2=5.0)
KL_PARAMS = dict(objective="binary", metric="auc", learning_rate=0.05, num_leaves=63,
                 feature_fraction=0.7, bagging_fraction=0.9, bagging_freq=1,
                 min_child_samples=300, lambda_l2=1.0, verbose=-1)
H = 3
DAILY_K = 15
WEIGHTS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]


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
        except Exception as e:
            log(f"  [dl] {period} 出错: {e}")
        m += 1
        if m > 12:
            m, y = 1, y + 1
    df = pl.concat(frames, how="vertical_relaxed").sort("close_time")
    log(f"[dl] 合计 {df.height} 行 {int(df['close_time'][0])} ~ {int(df['close_time'][-1])}")
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
    """滚动 z-score：每根 bar 只用它之前 win 根（shift(1)），模拟实盘。"""
    s = pl.Series("x", x)
    mp = max(win // 4, 30)
    mu = s.rolling_mean(win, min_samples=mp).shift(1)
    sd = s.rolling_std(win, min_samples=mp).shift(1)
    return ((s - mu) / (sd + 1e-9)).fill_nan(0.0).fill_null(0.0).to_numpy()


def find_ob():
    exact = "BTCUSDT.parquet"
    hits = [h for h in glob.glob(f"/kaggle/input/**/{exact}", recursive=True)
            if os.path.basename(h) == exact]
    if not hits:
        raise FileNotFoundError(exact)
    log(f"[ob] {hits[0]}")
    return hits[0]


def main():
    oseeds = [int(x) for x in os.environ.get("OSEEDS", "42,43,44,45,46").split(",")]
    kseeds = [int(x) for x in os.environ.get("KSEEDS", "42,43,44").split(",")]
    rounds = int(os.environ.get("ROUNDS", "60"))
    stride = int(os.environ.get("KSTRIDE", "1"))

    # ---------- 盘口腿 ----------
    ob = pl.read_parquet(find_ob()).sort("close_time")
    ob = ob.with_columns(pl.col("close").shift(-H).alias("_fc"))
    ob = ob.with_columns((pl.col("_fc") > pl.col("close")).cast(pl.Int8).alias("target")).drop("_fc")
    ob = ob.filter(pl.col("target").is_not_null())
    L = ob.height
    ob_ct = ob["close_time"].to_numpy()
    ob_end = int(ob_ct[-1])
    k1, k2 = int(L * 0.64), int(L * 0.80)
    val_start, test_start = int(ob_ct[k1]), int(ob_ct[k2])
    ob_cols = [c for c in ob.columns if c not in ("target", "close_time")]
    yo = ob["target"].to_numpy().astype(np.int8)
    Xo = ob.select(ob_cols).to_numpy().astype(np.float32)
    log(f"[ob] rows={L} feats={len(ob_cols)} val_start={val_start} test_start={test_start}")
    del ob

    tro, vao = slice(0, k1 - H), slice(k1, k2 - H)
    p_ob = np.zeros(L, dtype=np.float64)
    for sd in oseeds:
        p = dict(objective="binary", metric="auc", learning_rate=0.05, num_leaves=63,
                 feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1,
                 min_child_samples=100, verbose=-1,
                 seed=sd, bagging_seed=sd, feature_fraction_seed=sd, data_random_seed=sd)
        p.update(OB_REG)
        ds = lgb.Dataset(Xo[tro], yo[tro], feature_name=ob_cols)
        vs = lgb.Dataset(Xo[vao], yo[vao], reference=ds)
        m = lgb.train(p, ds, num_boost_round=rounds, valid_sets=[ds, vs],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        p_ob += m.predict(Xo)
    p_ob = (p_ob / len(oseeds)).astype(np.float32)
    log("[ob] done")
    del Xo

    # ---------- K 线腿 ----------
    kl = load_klines().filter(pl.col("close_time") <= ob_end)
    kl = kl.with_columns(pl.col("close").shift(-H).alias("_fc"))
    kl = kl.with_columns((pl.col("_fc") > pl.col("close")).cast(pl.Int8).alias("target")).drop("_fc")
    kl = kl.filter(pl.col("target").is_not_null())
    kf, kcols = build_features(kl)
    kct = kf["close_time"].to_numpy()
    yk = kf["target"].to_numpy().astype(np.int8)
    Xk = kf.select(kcols).to_numpy().astype(np.float32)
    log(f"[kl] rows={len(yk)} feats={len(kcols)} 到 {int(kct[-1])}（ob_end={ob_end}）")
    del kf, kl

    trk = np.where(kct + H * 60_000 < val_start)[0][::stride]
    pred_idx = np.where(kct >= val_start)[0]
    log(f"[kl] train={len(trk)} pred={len(pred_idx)}")
    Xk_tr, Xk_pr = Xk[trk], Xk[pred_idx]
    p_kl = np.zeros(len(pred_idx), dtype=np.float64)
    for sd in kseeds:
        p = dict(KL_PARAMS, seed=sd, bagging_seed=sd,
                 feature_fraction_seed=sd, data_random_seed=sd)
        m = lgb.train(p, lgb.Dataset(Xk_tr, yk[trk], feature_name=kcols),
                      num_boost_round=rounds)
        p_kl += m.predict(Xk_pr)
    p_kl = (p_kl / len(kseeds)).astype(np.float32)
    kct_pred = kct[pred_idx]
    log("[kl] done")
    del Xk, Xk_tr, Xk_pr

    # ---------- join 到同一批时间戳 ----------
    # K 线 bar 的 close_time 比 orderbook 早 1ms（...999999 vs ...000000），对齐需 +1
    merged = (pl.DataFrame({"close_time": ob_ct, "target": yo, "p_ob": p_ob})
              .join(pl.DataFrame({"close_time": kct_pred + 1, "p_kl": p_kl}),
                    on="close_time", how="inner").sort("close_time"))
    log(f"[join] 交集={merged.height}（盘口 {L} 行，丢失 {L - merged.height}）")
    mct = merged["close_time"].to_numpy()
    yy = merged["target"].to_numpy().astype(np.int8)
    zo = merged["p_ob"].to_numpy().astype(np.float64)
    zk = merged["p_kl"].to_numpy().astype(np.float64)

    dyn = day_index(mct)
    m_val = (mct >= val_start) & (mct < test_start)
    m_te = mct >= test_start
    log(f"[join] val={int(m_val.sum())} test={int(m_te.sum())}")
    if m_te.sum() == 0:
        raise SystemExit("[join] test 段为空，时间戳对齐有问题")
    dv, dte = dyn[m_val], dyn[m_te]
    mt = complete_days(dte)
    n_te = len(np.unique(dte))

    def zs(x):
        return (x - x[m_val].mean()) / (x[m_val].std() + 1e-9)

    def gacc(s, m):
        idx = np.where(m)[0]
        return global_topk_eval(s[idx], yy[idx], dyn[idx], DAILY_K)

    def stat(s):
        idx = np.argsort(-np.abs(s[m_te]))[:DAILY_K * n_te]
        u, c = np.unique(dte[idx], return_counts=True)
        accs = daily_topk_eval(s[m_te], yy[m_te], dte, mt, DAILY_K)
        return (float(roc_auc_score(yy[m_te], s[m_te])), gacc(s, m_te),
                float(accs.mean()), len(u), int(c.max()))

    modes = [("static_val(旧)", None)] + [(f"roll_{w}({w/1440:.0f}d)", w)
                                         for w in (1440, 4320, 10080)]
    log(f"[test] n={int(m_te.sum())} 天数={n_te} K={DAILY_K*n_te} "
        f"基准={yy[m_te].mean():.4f}")
    log(f"\n{'口径':<14}{'腿':<6}{'最优w':>7}{'AUC':>8}{'topK':>8}{'按天':>8}"
        f"{'覆盖天':>8}{'单天max':>9}")
    for name, win in modes:
        if win is None:
            zo_s, zk_s = zs(zo), zs(zk)
        else:
            zo_s, zk_s = roll_z(zo, win), roll_z(zk, win)
        grid = [(wv, gacc(zo_s + wv * zk_s, m_val), gacc(zo_s + wv * zk_s, m_te))
                for wv in WEIGHTS]
        bw = max(grid, key=lambda g: g[1])[0]
        for leg, s in [("盘口", zo_s), ("K线", zk_s),
                       (f"融合", zo_s + bw * zk_s)]:
            auc, gk, dk, cov, mx = stat(s)
            wshow = "-" if leg != "融合" else f"{bw:.2f}"
            log(f"{name:<14}{leg:<6}{wshow:>7}{auc:>8.4f}{gk:>8.4f}{dk:>8.4f}"
                f"{cov:>8}{mx:>9}")
        log(f"    └ val 网格 w→val: " +
            " ".join(f"{wv:g}:{av:.4f}" for wv, av, _ in grid))
        corr = float(np.corrcoef(zo_s[m_te], zk_s[m_te])[0, 1])
        log(f"    └ 两腿 test 相关={corr:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
