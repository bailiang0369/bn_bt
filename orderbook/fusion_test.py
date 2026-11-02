"""融合评估（正确版）。

基础 = orderbook（2025-07~2026-10，数据少，只能当基础）：
  1) 按时间切 train/val/test；
  2) 取训练集末尾时间点 T_end；
  3) faiss 形态聚类用【K 线 2020 → T_end】的全量数据训练（历史长、簇更稳）；
  4) orderbook 模型单独在 orderbook 训练集上训练；
  5) 两个模型各自预测，再在低覆盖率（top 1%/2%/5%/10%/25%）上评估组合，
     并打印每个覆盖率的样本量（不是两个预测直接 AND，那样量太大没筛选意义）。

对齐：K 线 close_time = Binance 的 open+59999ms，orderbook = open+60000ms，
     故 K 线 close_time 统一 +1（毫秒）后与 orderbook close_time 对齐；
     并加 ±2s 最近邻兜底 + 打印匹配率诊断。
归一化：形态窗口是逐窗 min-max（滚动），非全局。
防泄漏：聚类只在 kline <= T_end 上 fit；簇 edge 只在 orderbook 训练行上算（训练行 leave-one-out）。
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
from sklearn.metrics import roc_auc_score, accuracy_score
import faiss

from morph.morph import get_feats, stoch, HYBRID_CONFIG, add_target

SEED = 42
COVS = [0.01, 0.02, 0.05, 0.10, 0.25]


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


def cluster_ids_kline(kline, t_end_idx, arr, w, r, nc, is_indicator):
    """在 kline[0..t_end_idx] 的窗口特征上 fit Kmeans，再对全 kline 赋簇 id。"""
    feat = get_feats(arr, w, r, is_indicator=is_indicator)
    n = len(arr)
    ids = np.full(n, -1, dtype=np.int32)
    tr_n = (t_end_idx + 1) - (w - 1)
    if tr_n <= 0:
        return ids
    km = faiss.Kmeans(feat.shape[1], nc, niter=40, seed=SEED)
    km.train(np.ascontiguousarray(feat[:tr_n]))
    _, I = km.index.search(feat, 1)
    ids[w - 1:] = I.flatten()
    return ids


def edge_loo(ob_ids, target, k1):
    """训练行 leave-one-out edge；非训练行用训练集簇均值；无簇 0。"""
    L = len(ob_ids)
    edge = np.zeros(L, dtype=np.float32)
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


def main():
    ap = argparse.ArgumentParser(description="orderbook x 形态 融合评估（正确版）")
    ap.add_argument("--horizon", type=int, default=int(os.environ.get("KAGGLE_HORIZON", "3")))
    ap.add_argument("--rounds", type=int, default=1000)
    ap.add_argument("--rows", type=int, default=0, help="烟测：只取前 N 行，0=全量")
    ap.add_argument("--kline-file", default=None, help="烟测：指定 K 线 parquet（可指向 orderbook 自身）")
    a = ap.parse_args()

    ob = load_orderbook()
    kl = load_kline(a.kline_file)
    if a.rows:
        ob = ob.head(a.rows)
        kl = kl.head(a.rows)
    print(f"[kline] rows={kl.height} cols={kl.columns}")

    ob = add_target(ob, a.horizon).drop_nulls()
    L = len(ob)
    k1, k2 = int(L * 0.64), int(L * 0.80)
    tr, va, te = slice(0, k1), slice(k1, k2), slice(k2, L)
    ob_ct = ob["close_time"].to_numpy()
    T_end = int(ob_ct[k1 - 1])
    print(f"[ob] rows={L} split train[0:{k1}] val[{k1}:{k2}] test[{k2}:{L}]  T_end(ms)={T_end}")

    # K 线 close_time +1（毫秒）后与 orderbook 对齐；最近邻 ±2s 兜底
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
            ids = cluster_ids_kline(kl, t_end_idx, arr, cfg["window"], cfg["repeat"],
                                    cfg["n_clusters"], ind)
            ob_ids = np.where(valid, ids[nearest], -1)
            bucket.append(edge_loo(ob_ids, target, k1))
    ace_p = np.sum(np.vstack(p_edges), axis=0).astype(np.float32)
    ace_s = np.sum(np.vstack(s_edges), axis=0).astype(np.float32)
    ace = (ace_p + ace_s).astype(np.float32)

    # orderbook 模型（单独训练）
    base_cols = [c for c in ob.columns if c not in ("target", "close_time")]
    X = ob.select(base_cols).to_numpy().astype(np.float32)
    y = target
    train_set = lgb.Dataset(X[tr], y[tr], feature_name=base_cols)
    val_set = lgb.Dataset(X[va], y[va], reference=train_set)
    params = dict(objective="binary", metric="auc", learning_rate=0.05, num_leaves=63,
                  feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1,
                  min_child_samples=100, verbose=-1, seed=SEED)
    model = lgb.train(params, train_set, num_boost_round=a.rounds,
                      valid_sets=[train_set, val_set],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
    p_all = model.predict(X, num_iteration=model.best_iteration)

    y_te = y[te]
    p_te = p_all[te]
    ace_te = ace[te]
    ob_score = p_te - 0.5

    def z(s, ref):
        return (s - ref.mean()) / (ref.std() + 1e-9)

    z_ace = z(ace_te, ace[tr])
    z_ob = z(ob_score, (p_all[tr] - 0.5))
    combined = z_ace + z_ob

    base = float(y_te.mean())
    print("\n" + "=" * 74)
    print(f"[test] n={len(y_te)}  基准(target=1)={base:.4f}  orderbook_best_iter={model.best_iteration}")
    print(f"[单独AUC] faiss={roc_auc_score(y_te, ace_te):.4f}  orderbook={roc_auc_score(y_te, p_te):.4f}")
    print("=" * 74)
    print(f"{'cov':>6}{'n':>9}{'faiss':>10}{'orderbook':>12}{'combined':>12}")
    for cov in COVS:
        af, _ = acc_at_cov(ace_te, y_te, cov)
        ao, _ = acc_at_cov(ob_score, y_te, cov)
        ac, n = acc_at_cov(combined, y_te, cov)
        print(f"{cov:>6.0%}{n:>9}{af:>10.4f}{ao:>12.4f}{ac:>12.4f}")

    agree = np.sign(ace_te) == np.sign(ob_score)
    print(f"\n[agree] 两模型同向占比={agree.mean():.3f}（直接AND的样本量，太大无筛选意义）")
    if agree.sum():
        pred_up = (ace_te[agree] > 0).astype(int)
        print(f"[agree] 同向ACC={float((pred_up == y_te[agree]).mean()):.4f}")


if __name__ == "__main__":
    main()
