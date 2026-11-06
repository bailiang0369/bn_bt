"""盘口腿 topK 定向优化（Kaggle，CPU）。

动机：主口径 = 按 |score| 取 top 1% 再判方向，而现有 150 列全是**静态水平量**
（当前深度/挂单/价差），没有一行刻画"订单簿怎么变"。ETH 那边的消融显示更细的
盘口特征值 +0.91pp，故这里给 BTC 补动态特征：
  - d1：一阶差分 x - shift(1)（订单簿流量）
  - rz：滚动 z-score (x - roll_mean(60)) / roll_std(60)（当前水平相对近期常态）
另外顺带验证降复杂度（num_leaves 31 -> 15）与多 seed。

口径与 fusion_kl 完全一致：h=3，k1=64%/k2=80%，末尾 purge h，60 轮，val z-score，
主口径 global_topk_eval（K=15×test 天数）。**用 val 选，test 只报**。
"""
import argparse
import glob
import os
import sys

import numpy as np
import polars as pl

try:
    import lightgbm as lgb
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "lightgbm"], check=True)
    import lightgbm as lgb

from sklearn.metrics import roc_auc_score

H = 3
DAILY_K = 15
OB_REG = dict(num_leaves=31, min_child_samples=500, feature_fraction=0.6, lambda_l2=5.0)
BASE = dict(objective="binary", metric="auc", learning_rate=0.05, num_leaves=63,
            feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1,
            min_child_samples=100, verbose=-1)


def log(m):
    print(m, flush=True)


# --- 复制自 orderbook/fusion_test.py（该模块顶层 import faiss，这里不想装）---
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


def find(sym):
    exact = f"{sym}.parquet"
    hits = [h for h in glob.glob(f"/kaggle/input/**/{exact}", recursive=True)
            if os.path.basename(h) == exact]
    if not hits:
        raise FileNotFoundError(sym)
    log(f"[data] {hits[0]}")
    return hits[0]


def load():
    df = pl.read_parquet(find("BTCUSDT")).sort("close_time")
    df = df.with_columns(pl.col("close").shift(-H).alias("_fc"))
    df = df.with_columns(((pl.col("_fc") / pl.col("close") - 1) * 1e4).alias("fwd"))
    df = (df.with_columns((pl.col("fwd") > 0).cast(pl.Int8).alias("target"))
          .drop("_fc").filter(pl.col("fwd").is_not_null()))
    return df


def add_dyn(df, base_cols, kind, win=60, chunk=40):
    """订单簿动态特征。kind: d1 / rz / both。分块 with_columns，避免一次性展开 300 列。"""
    names = []
    for i in range(0, len(base_cols), chunk):
        exprs = []
        for c in base_cols[i:i + chunk]:
            if kind in ("d1", "both"):
                exprs.append((pl.col(c) - pl.col(c).shift(1)).cast(pl.Float32).alias(f"d1_{c}"))
                names.append(f"d1_{c}")
            if kind in ("rz", "both"):
                mu = pl.col(c).rolling_mean(win)
                sd = pl.col(c).rolling_std(win)
                exprs.append(((pl.col(c) - mu) / (sd + 1e-9)).cast(pl.Float32).alias(f"rz_{c}"))
                names.append(f"rz_{c}")
        df = df.with_columns(exprs)
    return df.select(names)


def train(X, y, tr, va, cols, seeds, rounds, params):
    probs = np.zeros(len(y), dtype=np.float64)
    iters = []
    for sd in seeds:
        p = dict(params, seed=sd, bagging_seed=sd, feature_fraction_seed=sd, data_random_seed=sd)
        ds = lgb.Dataset(X[tr], y[tr], feature_name=cols)
        vs = lgb.Dataset(X[va], y[va], reference=ds)
        m = lgb.train(p, ds, num_boost_round=rounds, valid_sets=[ds, vs],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        probs += m.predict(X, num_iteration=m.best_iteration)
        iters.append(int(m.best_iteration))
    return (probs / len(seeds)).astype(np.float32), iters


def bucket_stats(z, y, mask, n=8):
    """val 上按 |z| 分桶，统计 z>0 / z<0 两种决策的实际 P(y=1)。"""
    az, zz, yy = np.abs(z[mask]), z[mask], y[mask]
    edges = np.quantile(az, np.linspace(0, 1, n + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    stats = []
    for i in range(n):
        m = (az >= edges[i]) & (az < edges[i + 1])
        mp, mn = m & (zz > 0), m & (zz < 0)
        stats.append((i, int(m.sum()),
                      float(yy[mp].mean()) if mp.sum() >= 50 else np.nan,
                      float(yy[mn].mean()) if mn.sum() >= 50 else np.nan))
    return edges, stats


def apply_flip(z, edges, fp, fn):
    b = np.clip(np.searchsorted(edges[1:-1], np.abs(z), side="right"), 0, len(fp) - 1)
    out = z.copy()
    pos = z > 0
    out[pos & fp[b]] *= -1
    out[~pos & fn[b]] *= -1
    return out


def quota_acc(z, y, k_total, ratio_dn):
    """固定总量 k_total，按 ratio_dn 分配做空配额，多空各自按置信度取前若干。

    做多 = z>0（越正越自信），做空 = z<0（越负越自信）；合并准确率按单量加权。
    """
    zp, yp = z[z > 0], y[z > 0]
    zn, yn = z[z < 0], y[z < 0]
    kd = min(int(round(k_total * ratio_dn)), len(zn))
    ku = min(k_total - kd, len(zp))
    op = np.argsort(-zp)[:ku]
    on = np.argsort(zn)[:kd]
    hits = int((yp[op] == 1).sum()) + int((yn[on] == 0).sum())
    return float(hits / max(ku + kd, 1))


def diag(z, y, m_val, m_te, dv, dte):
    """诊断 + 后处理：主口径只看 top-K，检查方向偏差与多空不对称。"""
    k = DAILY_K * len(np.unique(dte))
    idx = np.argsort(-np.abs(z[m_te]))[:k]
    zt, yt = z[m_te][idx], y[m_te][idx]
    up, dn = zt > 0, zt < 0
    log(f"  [方向] topK 做多 n={int(up.sum())} acc={float((yt[up] == 1).mean()):.4f} | "
        f"做空 n={int(dn.sum())} acc={float((yt[dn] == 0).mean()):.4f}")
    edges, stats = bucket_stats(z, y, m_val)
    fp = np.array([s[2] < 0.5 for s in stats])
    fn = np.array([s[3] > 0.5 for s in stats])
    zc = apply_flip(z, edges, fp, fn)
    base = global_topk_eval(z[m_te], y[m_te], dte, DAILY_K)
    cal = global_topk_eval(zc[m_te], y[m_te], dte, DAILY_K)
    log(f"  [校准] 需翻转 多={list(np.where(fp)[0])} 空={list(np.where(fn)[0])} "
        f"test topK {base:.4f} -> {cal:.4f} ({cal - base:+.4f})")

    kv = DAILY_K * len(np.unique(dv))
    log(f"  [配额] val 选 ratio_dn（val K={kv} / test K={k}）")
    log(f"          {'ratio_dn':<10}{'val':>9}{'test':>9}")
    best_r, best_v = 0.5, -1.0
    for r in np.linspace(0, 1, 11):
        av = quota_acc(z[m_val], y[m_val], kv, r)
        at = quota_acc(z[m_te], y[m_te], k, r)
        if av > best_v:
            best_v, best_r = av, r
        log(f"          {r:<10.1f}{av:>9.4f}{at:>9.4f}")
    log(f"  [配额] val 选 ratio_dn={best_r:.1f} -> test "
        f"{quota_acc(z[m_te], y[m_te], k, best_r):.4f}  "
        f"(全混排 {base:.4f}, 纯做空 {quota_acc(z[m_te], y[m_te], k, 1.0):.4f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offs", default=os.environ.get("OFFS", "0"))
    ap.add_argument("--seeds", default=os.environ.get("SEEDS", "42,43,44,45,46"))
    ap.add_argument("--rounds", type=int, default=int(os.environ.get("ROUNDS", "60")))
    ap.add_argument("--only", default=os.environ.get("ONLY", ""))
    ap.add_argument("--diag", type=int, default=int(os.environ.get("DIAG", "0")),
                    help="1=输出方向诊断与 val 分桶校准对比")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    offs = [int(x) for x in a.offs.split(",")]

    df = load()
    L = len(df)
    ct = df["close_time"].to_numpy()
    y = df["target"].to_numpy().astype(np.int8)
    static = [c for c in df.columns if c not in ("target", "close_time", "fwd")]
    log(f"[data] rows={L} static={len(static)} 时间 {int(ct[0])}~{int(ct[-1])}")

    XS = df.select(static).to_numpy().astype(np.float32)

    cache = {}

    def build(kind):
        """按需构建特征矩阵并缓存；多 offset 复用，避免重复算 300 列。"""
        if kind in cache:
            return cache[kind]
        if kind == "static":
            cache[kind] = (static, XS)
            return cache[kind]
        d = add_dyn(df, static, kind)
        new = list(d.columns)
        X = np.hstack([XS, d.to_numpy().astype(np.float32)])
        del d
        log(f"[feat] {kind}: +{len(new)} 列 -> {X.shape[1]}")
        cache[kind] = (static + new, X)
        return cache[kind]

    variants = {
        "v0_static": ("static 150  lv31", "static", dict(BASE, **OB_REG)),
        "v1_d1": ("static+d1   lv31", "d1", dict(BASE, **OB_REG)),
        "v2_rz": ("static+rz   lv31", "rz", dict(BASE, **OB_REG)),
        "v3_both": ("static+d1+rz lv31", "both", dict(BASE, **OB_REG)),
        "v4_static_lv15": ("static 150  lv15", "static",
                           dict(BASE, **dict(OB_REG, num_leaves=15))),
        "v5_ens": ("static+d1 ens9", "d1", None),
    }
    pick = [s for s in a.only.split(",") if s] or list(variants)

    k1_0, k2_0 = int(L * 0.64), int(L * 0.80)
    dyn_day = day_index(ct)

    for off in offs:
        k1, k2 = k1_0 - off * 1440, k2_0 - off * 1440
        val_start, test_start = int(ct[k1]), int(ct[k2])
        tr, va = slice(0, k1 - H), slice(k1, k2 - H)
        m_val = (ct >= val_start) & (ct < test_start)
        m_te = ct >= test_start
        dv, dte = dyn_day[m_val], dyn_day[m_te]
        mt = complete_days(dte)
        n_te = len(np.unique(dte))
        log(f"\n===== off={off} test={int(m_te.sum())} 天数={n_te} K={DAILY_K*n_te} "
            f"基准={y[m_te].mean():.4f} =====")
        log(f"{'变体':<22}{'val':>8}{'test':>8}{'AUC':>8}{'按天':>8}{'iters':>16}")
        for name in pick:
            label, sk, params = variants[name]
            cols, X = build(sk)
            if params is None:
                # 架构集成：三个 num_leaves 各 3 seeds 共 9 模型，z-score 后等权平均。
                # 不依赖 val 选参，纯降方差。
                zs, iters = [], []
                for lv in (15, 31, 63):
                    s, it = train(X, y, tr, va, cols, seeds[:3], a.rounds,
                                  dict(BASE, **dict(OB_REG, num_leaves=lv)))
                    zs.append((s - s[m_val].mean()) / (s[m_val].std() + 1e-9))
                    iters += it
                z = np.mean(zs, axis=0)
            else:
                s, iters = train(X, y, tr, va, cols, seeds, a.rounds, params)
                z = (s - s[m_val].mean()) / (s[m_val].std() + 1e-9)
            auc = float(roc_auc_score(y[m_te], z[m_te]))
            gv = global_topk_eval(z[m_val], y[m_val], dv, DAILY_K)
            gk = global_topk_eval(z[m_te], y[m_te], dte, DAILY_K)
            dk = float(daily_topk_eval(z[m_te], y[m_te], dte, mt, DAILY_K).mean())
            log(f"{label:<22}{gv:>8.4f}{gk:>8.4f}{auc:>8.4f}{dk:>8.4f}"
                f"{str(iters):>16}")
            if a.diag:
                diag(z, y, m_val, m_te, dv, dte)
            del X, z
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
