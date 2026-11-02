"""融合评估（多 seed 集成 + 权重调优版）。

基础 = orderbook（2025-07~2026-10，数据少，只能当基础）：
  1) 按时间切 train/val/test；取训练集末尾 T_end；
  2) faiss 形态聚类用【K 线 2020→T_end】全量训练，且【多 seed 平均 edge】；
  3) orderbook 模型单独训练，且【多 seed 平均概率】；
  4) 两个模型各自预测，在低覆盖率（top 1%/2%/5%/10%/25%）上评估组合；
  5) 组合权重 w 在 val 上选（不碰 test），再报 test。

对齐：K 线 close_time +1ms 与 orderbook close_time 对齐（Binance close=open+59999，
orderbook close=open+60000），±2s 最近邻兜底。
归一化：形态窗口逐窗 min-max（滚动），非全局。
防泄漏：聚类只在 kline<=T_end 上 fit；edge 只在 orderbook 训练行上算（训练行 LOO）；w 只在 val 选。
"""
import sys
import os
import glob
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import polars as pl
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import faiss

from morph.morph import get_feats, stoch, HYBRID_CONFIG, add_target

SEED = 42
COVS = [0.01, 0.02, 0.05, 0.10, 0.25]
WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]


def _find(names):
    hits = []
    for n in names:
        if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
            hits += glob.glob(f"/kaggle/input/**/{n}", recursive=True)
        loc = ROOT / "data" / n
        if loc.exists():
            hits.append(str(loc))
    return sorted(set(hits))


def load_orderbook():
    ps = _find(["BTCUSDT.parquet"])
    if not ps:
        raise FileNotFoundError("找不到 orderbook BTCUSDT.parquet")
    print(f"[orderbook] {ps[0]}")
    return pl.read_parquet(ps[0]).sort("close_time")


def load_kline(path=None):
    if path:
        print(f"[kline] {path}")
        return pl.read_parquet(path).sort("close_time")
    ps = _find(["btcusdt_1m.parquet", "btc_1m.parquet"])
    if not ps:
        raise FileNotFoundError("找不到 K 线 btcusdt_1m.parquet")
    print(f"[kline] {ps[0]}")
    return pl.read_parquet(ps[0]).sort("close_time")


def cluster_ids_kline(kline, t_end_idx, arr, w, r, nc, is_indicator, seed):
    feat = get_feats(arr, w, r, is_indicator=is_indicator)
    n = len(arr)
    ids = np.full(n, -1, dtype=np.int32)
    tr_n = (t_end_idx + 1) - (w - 1)
    if tr_n <= 0:
        return ids
    km = faiss.Kmeans(feat.shape[1], nc, niter=40, seed=seed)
    km.train(np.ascontiguousarray(feat[:tr_n]))
    _, I = km.index.search(feat, 1)
    ids[w - 1:] = I.flatten()
    return ids


def edge_loo(ob_ids, target, k1):
    L = len(ob_ids)
    edge = np.zeros(L, dtype=np.float64)
    cnt, ssum = {}, {}
    for i in range(k1):
        c = int(ob_ids[i])
        if c < 0:
            continue
        cnt[c] = cnt.get(c, 0) + 1
        ssum[c] = ssum.get(c, 0.0) + float(target[i])
    for i in range(L):
        c = int(ob_ids[i])
        if c < 0:
            continue
        if i < k1:
            n = cnt.get(c, 0) - 1
            if n > 0:
                edge[i] = (ssum[c] - float(target[i])) / n - 0.5
        else:
            n = cnt.get(c, 0)
            if n > 0:
                edge[i] = ssum[c] / n - 0.5
    return edge


def acc_at_cov(score, y, cov):
    n = len(y)
    k = max(1, int(n * cov))
    idx = np.argsort(-np.abs(score))[:k]
    pred = (score[idx] > 0).astype(int)
    return float((pred == y[idx]).mean()), k


def train_lgb_avg(X, y, tr, va, cols, seeds, rounds):
    probs = np.zeros(len(y), dtype=np.float64)
    iters = []
    for sd in seeds:
        params = dict(objective="binary", metric="auc", learning_rate=0.05, num_leaves=63,
                      feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1,
                      min_child_samples=100, verbose=-1,
                      seed=sd, bagging_seed=sd, feature_fraction_seed=sd, data_random_seed=sd)
        ds = lgb.Dataset(X[tr], y[tr], feature_name=cols)
        vs = lgb.Dataset(X[va], y[va], reference=ds)
        m = lgb.train(params, ds, num_boost_round=rounds, valid_sets=[ds, vs],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        probs += m.predict(X, num_iteration=m.best_iteration)
        iters.append(int(m.best_iteration))
    return (probs / len(seeds)).astype(np.float32), iters


def main():
    ap = argparse.ArgumentParser(description="orderbook x 形态 融合（多 seed + 权重调优）")
    ap.add_argument("--horizon", type=int, default=int(os.environ.get("KAGGLE_HORIZON", "3")))
    ap.add_argument("--rounds", type=int, default=1000)
    ap.add_argument("--rows", type=int, default=0, help="烟测：只取前 N 行，0=全量")
    ap.add_argument("--kline-file", default=None, help="烟测：指定 K 线 parquet")
    ap.add_argument("--lgb-seeds", default="42,43,44,45,46")
    ap.add_argument("--faiss-seeds", default="42,43,44")
    a = ap.parse_args()

    lgb_seeds = [int(x) for x in a.lgb_seeds.split(",") if x.strip()]
    faiss_seeds = [int(x) for x in a.faiss_seeds.split(",") if x.strip()]

    ob = load_orderbook()
    kl = load_kline(a.kline_file)
    if a.rows:
        ob = ob.head(a.rows)
        kl = kl.head(a.rows)
    print(f"[kline] rows={kl.height} cols={kl.columns}")
    print(f"[seeds] lgb={lgb_seeds}  faiss={faiss_seeds}")

    ob = add_target(ob, a.horizon).drop_nulls()
    L = len(ob)
    k1, k2 = int(L * 0.64), int(L * 0.80)
    tr, va, te = slice(0, k1), slice(k1, k2), slice(k2, L)
    ob_ct = ob["close_time"].to_numpy()
    T_end = int(ob_ct[k1 - 1])
    print(f"[ob] rows={L} split train[0:{k1}] val[{k1}:{k2}] test[{k2}:{L}]  T_end(ms)={T_end}")

    kl_ct = kl["close_time"].to_numpy() + 1
    assert np.all(np.diff(kl_ct) >= 0), "kline close_time 非升序"
    pos = np.clip(np.searchsorted(kl_ct, ob_ct), 1, len(kl_ct) - 1)
    left, right = kl_ct[pos - 1], kl_ct[pos]
    nearest = np.where(np.abs(ob_ct - left) <= np.abs(ob_ct - right), pos - 1, pos)
    diff = ob_ct - kl_ct[nearest]
    valid = np.abs(diff) <= 2000
    t_end_idx = int(np.searchsorted(kl_ct, T_end, side="right") - 1)
    print(f"[align] match_rate={valid.mean():.4f}  |diff|max={int(np.abs(diff).max())}ms  "
          f"kline_train_end_idx={t_end_idx}")

    target = ob["target"].to_numpy().astype(int)
    close = kl["close"].to_numpy().astype(np.float32)
    st = stoch(kl, 29)

    p_edges, s_edges = [], []
    for ch, arr, ind, bucket in [("p", close, False, p_edges), ("s", st, True, s_edges)]:
        for name, cfg in HYBRID_CONFIG.items():
            acc = np.zeros(L, dtype=np.float64)
            for sd in faiss_seeds:
                ids = cluster_ids_kline(kl, t_end_idx, arr, cfg["window"], cfg["repeat"],
                                        cfg["n_clusters"], ind, sd)
                ob_ids = np.where(valid, ids[nearest], -1)
                acc += edge_loo(ob_ids, target, k1)
            bucket.append((acc / len(faiss_seeds)).astype(np.float32))
    ace_p = np.sum(np.vstack(p_edges), axis=0).astype(np.float32)
    ace_s = np.sum(np.vstack(s_edges), axis=0).astype(np.float32)
    ace = (ace_p + ace_s).astype(np.float32)

    base_cols = [c for c in ob.columns if c not in ("target", "close_time")]
    X = ob.select(base_cols).to_numpy().astype(np.float32)
    y = target
    p_all, iters = train_lgb_avg(X, y, tr, va, base_cols, lgb_seeds, a.rounds)

    def mean_std(ref):
        return float(ref.mean()), float(ref.std() + 1e-9)

    m_ace, s_ace = mean_std(ace[tr])
    m_ob, s_ob = mean_std((p_all[tr] - 0.5))

    def zc(score, m, s):
        return (score - m) / s

    z_ace_va = zc(ace[va], m_ace, s_ace)
    z_ob_va = zc((p_all[va] - 0.5), m_ob, s_ob)
    y_va = y[va]

    best_w, best_val = 1.0, -1.0
    for w in WEIGHTS:
        acc1, _ = acc_at_cov(z_ob_va + w * z_ace_va, y_va, 0.01)
        if acc1 > best_val:
            best_val, best_w = acc1, w
    print(f"[val] 按 top1% ACC 选出的权重 w={best_w}（val@1%={best_val:.4f}）")

    y_te = y[te]
    ace_te = ace[te]
    ob_te = (p_all[te] - 0.5)
    z_ace_te = zc(ace_te, m_ace, s_ace)
    z_ob_te = zc(ob_te, m_ob, s_ob)
    combined = z_ob_te + best_w * z_ace_te

    base = float(y_te.mean())
    print("\n" + "=" * 78)
    print(f"[test] n={len(y_te)} 基准={base:.4f} orderbook_seed_iters={iters}  w={best_w}")
    print(f"[单独AUC] faiss={roc_auc_score(y_te, ace_te):.4f}  orderbook={roc_auc_score(y_te, p_all[te]):.4f}")
    print("=" * 78)
    print(f"{'cov':>6}{'n':>9}{'faiss':>10}{'orderbook':>12}{'combined':>12}")
    for cov in COVS:
        af, _ = acc_at_cov(ace_te, y_te, cov)
        ao, _ = acc_at_cov(ob_te, y_te, cov)
        ac, n = acc_at_cov(combined, y_te, cov)
        print(f"{cov:>6.0%}{n:>9}{af:>10.4f}{ao:>12.4f}{ac:>12.4f}")

    agree = np.sign(ace_te) == np.sign(ob_te)
    pred_up = (ace_te > 0).astype(int)
    print(f"\n[agree] 同向占比={agree.mean():.3f}  同向ACC={float((pred_up[agree] == y_te[agree]).mean()):.4f}")


if __name__ == "__main__":
    main()
