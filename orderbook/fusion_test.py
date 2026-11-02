"""融合 A/B 测试：orderbook 特征 vs orderbook + faiss 形态簇特征。

目标统一为 t+3 涨跌（把 morph 原来的 15min 改成 3min，与 orderbook 对齐）。
防泄漏：
  - 形态聚类（faiss Kmeans）只在【训练集】fit，再对全量做 assign；
  - 簇 edge = 簇内训练集 target 均值 - 0.5，仅用训练行计算；
  - 训练行的 edge 用 leave-one-out（扣掉自身），避免模型在训练集上"记住"自己的标签；
  - 形态归一化是逐窗 min-max（滚动），非全局归一化。
"""
import sys
import os
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import polars as pl
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, accuracy_score
import faiss

from morph.morph import get_feats, stoch, HYBRID_CONFIG, add_target

DATASET_SLUG = "btcusdt-orderbook-features"
SEED = 42


def load_data():
    local = ROOT / "data" / "BTCUSDT.parquet"
    kaggle = Path(f"/kaggle/input/{DATASET_SLUG}/BTCUSDT.parquet")
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") and kaggle.exists():
        p = kaggle
    elif local.exists():
        p = local
    else:
        p = kaggle
    print(f"[load] {p}")
    return pl.read_parquet(str(p)).sort("close_time")


def cluster_ids(arr, w, r, n_clusters, k_train, is_indicator):
    """在训练集上前 w 窗口特征上 fit Kmeans，再对全量 assign 簇 id（前 w-1 行设 -1）。"""
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    feat = get_feats(arr, w, r, is_indicator=is_indicator)  # 行 j <-> 原序列 t=j+w-1
    L = len(arr)
    ids = np.full(L, -1, dtype=np.int32)
    tr_n = k_train - (w - 1)
    if tr_n <= 0:
        return ids
    km = faiss.Kmeans(feat.shape[1], n_clusters, niter=40, seed=SEED)
    km.train(np.ascontiguousarray(feat[:tr_n]))
    _, I = km.index.search(feat, 1)
    ids[w - 1:] = I.flatten()
    return ids


def cluster_edge(ids, target, k_train):
    """训练行 leave-one-out edge，非训练行用训练集簇均值 edge，无簇用 0。"""
    edge = np.zeros(len(ids), dtype=np.float32)
    cnt, ssum = {}, {}
    for i in range(k_train):
        c = int(ids[i])
        if c < 0:
            continue
        cnt[c] = cnt.get(c, 0) + 1
        ssum[c] = ssum.get(c, 0.0) + float(target[i])
    for i in range(len(ids)):
        c = int(ids[i])
        if c < 0:
            continue
        if i < k_train:
            n = cnt.get(c, 0) - 1
            if n > 0:
                edge[i] = (ssum[c] - float(target[i])) / n - 0.5
        else:
            n = cnt.get(c, 0)
            if n > 0:
                edge[i] = ssum[c] / n - 0.5
    return edge


def build_morph_edges(df, target, k_train):
    """返回各尺度 edge 数组 + 两个通道的合并 ace。"""
    close = df["close"].to_numpy().ravel().astype(np.float32)
    st = stoch(df, 29)
    p_edges, s_edges = [], []
    for ch, arr, ind, bucket in [("p", close, False, p_edges), ("s", st, True, s_edges)]:
        for name, cfg in HYBRID_CONFIG.items():
            ids = cluster_ids(arr, cfg["window"], cfg["repeat"], cfg["n_clusters"], k_train, ind)
            bucket.append(cluster_edge(ids, target, k_train))
    ace_p = np.sum(np.vstack(p_edges), axis=0).astype(np.float32)
    ace_s = np.sum(np.vstack(s_edges), axis=0).astype(np.float32)
    return p_edges, s_edges, ace_p, ace_s


def eval_lgb(X, y, cols, tr, va, te, rounds):
    train_set = lgb.Dataset(X[tr], y[tr], feature_name=cols)
    val_set = lgb.Dataset(X[va], y[va], reference=train_set)
    params = dict(objective="binary", metric="auc", learning_rate=0.05, num_leaves=63,
                  feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1,
                  min_child_samples=100, verbose=-1, seed=SEED)
    model = lgb.train(params, train_set, num_boost_round=rounds,
                      valid_sets=[train_set, val_set],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
    best = model.best_iteration
    out = {}
    for name, sl in [("train", tr), ("val", va), ("test", te)]:
        p = model.predict(X[sl], num_iteration=best)
        out[name] = dict(auc=float(roc_auc_score(y[sl], p)),
                         acc=float(accuracy_score(y[sl], (p >= 0.5).astype(int))))
    # 诚实选择性：阈值在 val 定（最自信 1%），原样套 test
    pv = model.predict(X[va], num_iteration=best)
    pt = model.predict(X[te], num_iteration=best)
    T = np.quantile(np.abs(pv - 0.5), 0.99)
    sel = np.abs(pt - 0.5) >= T
    if sel.sum():
        out["test"]["sel_acc"] = float(accuracy_score(y[te][sel], (pt[sel] >= 0.5).astype(int)))
        out["test"]["sel_cov"] = float(sel.mean())
    out["best_iter"] = int(best)
    out["val_auc"] = float(roc_auc_score(y[va], pv))
    return out


def main():
    ap = argparse.ArgumentParser(description="orderbook x 形态 融合 A/B 测试")
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--rows", type=int, default=0, help="只取前 N 行做烟测，0=全量")
    ap.add_argument("--rounds", type=int, default=1000)
    a = ap.parse_args()

    df = load_data()
    if a.rows:
        df = df.head(a.rows)
    df = add_target(df, a.horizon)
    df = df.drop_nulls()
    L = len(df)
    print(f"[data] 清洗后 {L} 行, 列数 {len(df.columns)}")

    k1 = int(L * 0.64)
    k2 = int(L * 0.80)
    tr, va, te = slice(0, k1), slice(k1, k2), slice(k2, L)
    print(f"[split] train[0:{k1}] val[{k1}:{k2}] test[{k2}:{L}]")

    target = df["target"].to_numpy().astype(int)
    base_cols = [c for c in df.columns if c not in ("target", "close_time")]
    X_base = df.select(base_cols).to_numpy().astype(np.float32)

    p_edges, s_edges, ace_p, ace_s = build_morph_edges(df, target, k1)
    names = ([f"edge_p_{n}" for n in HYBRID_CONFIG]
             + [f"edge_s_{n}" for n in HYBRID_CONFIG] + ["ace_p", "ace_s"])
    morph_mat = np.vstack(p_edges + s_edges + [ace_p, ace_s]).T.astype(np.float32)
    X_fused = np.hstack([X_base, morph_mat]).astype(np.float32)
    fused_cols = base_cols + names

    print("\n[1/2] baseline: 纯 orderbook 特征")
    r_base = eval_lgb(X_base, target, base_cols, tr, va, te, a.rounds)
    print("[2/2] fused: orderbook + 形态簇特征")
    r_fuse = eval_lgb(X_fused, target, fused_cols, tr, va, te, a.rounds)

    ace = (ace_p + ace_s)
    print(f"[diag] best_iter: baseline={r_base['best_iter']}  fused={r_fuse['best_iter']}")
    print(f"[diag] val_AUC:    baseline={r_base['val_auc']:.4f}  fused={r_fuse['val_auc']:.4f}")
    print(f"[diag] 形态 standalone test_AUC: ace_p={roc_auc_score(target[te], ace_p[te]):.4f}"
          f"  ace_s={roc_auc_score(target[te], ace_s[te]):.4f}"
          f"  ace={roc_auc_score(target[te], ace[te]):.4f}")

    print("\n" + "=" * 66)
    print(f"融合 A/B 测试  horizon={a.horizon}min  样本={L}")
    print("=" * 66)
    print(f"{'':<10}{'test_AUC':>12}{'test_ACC':>12}{'sel_ACC':>12}{'sel_cov':>10}")
    for nm, r in [("baseline", r_base), ("fused", r_fuse)]:
        t = r["test"]
        print(f"{nm:<10}{t['auc']:>12.4f}{t['acc']:>12.4f}"
              f"{t.get('sel_acc', float('nan')):>12.4f}{t.get('sel_cov', float('nan')):>10.3f}")
    d_auc = r_fuse["test"]["auc"] - r_base["test"]["auc"]
    d_acc = r_fuse["test"]["acc"] - r_base["test"]["acc"]
    print(f"\n差值: test_AUC {d_auc:+.4f}   test_ACC {d_acc:+.4f}")
    print("结论:",
          "融合有增量 -> 形态相对 orderbook 正交、可加信息"
          if d_auc > 0.002 else
          "融合无显著增量 -> 形态信息可能已被 orderbook 覆盖")


if __name__ == "__main__":
    main()
