"""标准化口径修正：滚动 z-score vs 静态 z-score（Kaggle）。

此前所有评估都用 `(p - p[val].mean()) / p[val].std()`：
  - 用整段 val 的固定均值/方差去缩放 test，假设分数分布跨期不变；
  - 实盘做不到"用未来一段的统计量"，且分数分布随市场状态漂移时，
    |z| 会系统性失真，进而让 top-K 向某几段集中。

本脚本对**同一组预测**比较多种标准化口径，只训练一次，避免训练噪声干扰对比：
  static_val / static_train / roll_{1d,3d,7d}（滚动窗口，shift(1) 只用过去）
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


def find(sym="BTCUSDT"):
    exact = f"{sym}.parquet"
    hits = [h for h in glob.glob(f"/kaggle/input/**/{exact}", recursive=True)
            if os.path.basename(h) == exact]
    if not hits:
        raise FileNotFoundError(exact)
    log(f"[data] {hits[0]}")
    return hits[0]


def load():
    df = pl.read_parquet(find()).sort("close_time")
    df = df.with_columns(pl.col("close").shift(-H).alias("_fc"))
    df = (df.with_columns((pl.col("_fc") > pl.col("close")).cast(pl.Int8).alias("target"))
          .drop("_fc").filter(pl.col("target").is_not_null()))
    return df


def rolling_z(p, win):
    """对每根 bar 用**它之前** win 根做标准化；窗口不足时回退到文档内已有历史。"""
    s = pl.Series("p", p)
    mp = max(win // 4, 30)
    mu = s.rolling_mean(win, min_periods=mp).shift(1)
    sd = s.rolling_std(win, min_periods=mp).shift(1)
    z = (s - mu) / (sd + 1e-9)
    return z.fill_nan(0.0).fill_null(0.0).to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offs", default=os.environ.get("OFFS", "0"))
    ap.add_argument("--seeds", default=os.environ.get("SEEDS", "42,43,44,45,46"))
    ap.add_argument("--rounds", type=int, default=int(os.environ.get("ROUNDS", "60")))
    ap.add_argument("--wins", default=os.environ.get("WINS", "1440,4320,10080"))
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split(",")]
    offs = [int(x) for x in a.offs.split(",")]
    wins = [int(x) for x in a.wins.split(",")]

    df = load()
    L = df.height
    ct = df["close_time"].to_numpy()
    y = df["target"].to_numpy().astype(np.int8)
    cols = [c for c in df.columns if c not in ("target", "close_time")]
    X = df.select(cols).to_numpy().astype(np.float32)
    del df
    log(f"[data] rows={L} feats={len(cols)}")

    k1_0, k2_0 = int(L * 0.64), int(L * 0.80)
    dyn = day_index(ct)

    for off in offs:
        k1, k2 = k1_0 - off * 1440, k2_0 - off * 1440
        val_start, test_start = int(ct[k1]), int(ct[k2])
        tr, va = slice(0, k1 - H), slice(k1, k2 - H)
        m_val = (ct >= val_start) & (ct < test_start)
        m_te = ct >= test_start
        dte = dyn[m_te]
        mt = complete_days(dte)
        n_te = len(np.unique(dte))
        K = DAILY_K * n_te

        # 只训练一次，后面所有标准化口径复用同一组原始预测
        p = np.zeros(L, dtype=np.float64)
        for sd in seeds:
            q = dict(BASE, **OB_REG, seed=sd, bagging_seed=sd,
                     feature_fraction_seed=sd, data_random_seed=sd)
            ds = lgb.Dataset(X[tr], y[tr], feature_name=cols)
            vs = lgb.Dataset(X[va], y[va], reference=ds)
            m = lgb.train(q, ds, num_boost_round=a.rounds, valid_sets=[ds, vs],
                          callbacks=[lgb.early_stopping(50, verbose=False)])
            p += m.predict(X)
        p = (p / len(seeds)).astype(np.float32)
        log(f"\n===== off={off} test={int(m_te.sum())} 天数={n_te} K={K} "
            f"基准={y[m_te].mean():.4f} =====")

        cands = {
            "static_val(旧口径)": (p - p[m_val].mean()) / (p[m_val].std() + 1e-9),
            "static_train": (p - p[tr].mean()) / (p[tr].std() + 1e-9),
        }
        for w in wins:
            cands[f"roll_{w}({w/1440:.0f}d)"] = rolling_z(p, w)

        log(f"{'口径':<20}{'AUC':>8}{'topK':>8}{'按天':>8}{'覆盖天':>8}"
            f"{'单天min':>9}{'单天max':>9}")
        for name, s in cands.items():
            auc = float(roc_auc_score(y[m_te], s[m_te]))
            gk = global_topk_eval(s[m_te], y[m_te], dte, DAILY_K)
            accs = daily_topk_eval(s[m_te], y[m_te], dte, mt, DAILY_K)
            idx = np.argsort(-np.abs(s[m_te]))[:K]
            u, c = np.unique(dte[idx], return_counts=True)
            log(f"{name:<20}{auc:>8.4f}{gk:>8.4f}{accs.mean():>8.4f}"
                f"{len(u):>8}{int(c.min()):>9}{int(c.max()):>9}")
        del p
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
