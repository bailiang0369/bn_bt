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
    on_kaggle = bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE")) or os.path.isdir("/kaggle/input")
    for n in names:
        if on_kaggle:
            hits += glob.glob(f"/kaggle/input/**/{n}", recursive=True)
        for loc in (ROOT / "data" / n, ROOT / "data" / "raw" / n):
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


def kline_edge_lut(ids, kl_t, t_end_idx, nc, horizon=0):
    """在【K线 2020→T_end-horizon】上算每个簇的 edge = mean(target)-0.5，返回查表数组。
    purge：末尾 horizon 个样本的 target 是 t+h 的价格，会跨进 val 段。"""
    valid = (ids >= 0) & (kl_t >= 0)
    valid[max(0, t_end_idx + 1 - horizon):] = False
    codes = ids[valid].astype(np.int64)
    tv = kl_t[valid].astype(np.float64)
    sums = np.bincount(codes, weights=tv, minlength=nc)
    counts = np.bincount(codes, minlength=nc)
    return np.where(counts > 0, sums / np.maximum(counts, 1) - 0.5, 0.0).astype(np.float32)


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
    ap.add_argument("--save-pred", default=os.environ.get("KAGGLE_FUSION_PRED"),
                    help="输出逐行 pred parquet（val+test），供离线阈值/组合分析")
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
    # purge：训练/验证末尾 h 个样本的标签是 t+h 的价格，会跨进下一段
    h = a.horizon
    tr, va, te = slice(0, k1 - h), slice(k1, k2 - h), slice(k2, L)
    ob_ct = ob["close_time"].to_numpy()
    T_end = int(ob_ct[k1 - 1 - h])
    print(f"[ob] rows={L} split train[0:{k1 - h}] val[{k1}:{k2 - h}] test[{k2}:{L}]  "
          f"T_end(ms)={T_end}  purge={h}")

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
    # faiss 的 edge 也必须在【K线 2020→T_end】上算（K线自身 t+15 target），与 orderbook 目标无关
    kl_t = add_target(kl, a.horizon)["target"].fill_null(-1).to_numpy().astype(np.int32)
    ob_close = ob["close"].to_numpy()
    kl_close_aligned = kl["close"].to_numpy()[nearest]
    cc = np.corrcoef(ob_close[valid], kl_close_aligned[valid])[0, 1]
    print(f"[align] orderbook vs kline close 相关系数={cc:.6f}")

    p_edges, s_edges = [], []
    for ch, arr, ind, bucket in [("p", close, False, p_edges), ("s", st, True, s_edges)]:
        for name, cfg in HYBRID_CONFIG.items():
            acc = np.zeros(L, dtype=np.float64)
            for sd in faiss_seeds:
                ids = cluster_ids_kline(kl, t_end_idx, arr, cfg["window"], cfg["repeat"],
                                        cfg["n_clusters"], ind, sd)
                lut = kline_edge_lut(ids, kl_t, t_end_idx, cfg["n_clusters"], a.horizon)
                ob_ids = np.where(valid, ids[nearest], -1)
                ob_edge = np.zeros(L, dtype=np.float64)
                m = ob_ids >= 0
                ob_edge[m] = lut[ob_ids[m]]
                acc += ob_edge
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

    y_va, y_te = y[va], y[te]
    z_ace_va = zc(ace[va], m_ace, s_ace)
    z_ob_va = zc((p_all[va] - 0.5), m_ob, s_ob)

    # 单独腿的零点：0（概率 0.5）还是 train 均值校准。在 val 上选，不碰 test。
    cand = {"zero": acc_at_cov(p_all[va] - 0.5, y_va, 0.01)[0],
            "calib": acc_at_cov(z_ob_va, y_va, 0.01)[0]}
    ob_mode = max(cand, key=cand.get)
    print(f"[val] orderbook 零点口径 zero={cand['zero']:.4f} calib={cand['calib']:.4f} → 采用 {ob_mode}")

    # 组合权重：val 上按 1%/2%/5% 平均 ACC 选（单点 top1% 噪声太大）
    best_w, best_val = 1.0, -1.0
    for w in WEIGHTS:
        s = z_ob_va + w * z_ace_va
        v = float(np.mean([acc_at_cov(s, y_va, c)[0] for c in (0.01, 0.02, 0.05)]))
        print(f"  [val] w={w:<5.2f} acc(1/2/5%均值)={v:.4f}")
        if v > best_val:
            best_val, best_w = v, w
    print(f"[val] 选出 w={best_w}（val 1/2/5% 均值={best_val:.4f}）")

    z_ob_te = zc((p_all[te] - 0.5), m_ob, s_ob)
    z_ace_te = zc(ace[te], m_ace, s_ace)

    # LR stacker（val 训练、test 评估，只做候选）
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(np.column_stack([z_ob_va, z_ace_va]), y_va)
    s_lr_te = lr.decision_function(np.column_stack([z_ob_te, z_ace_te]))

    ace_te = ace[te]
    ob_te = (p_all[te] - 0.5) if ob_mode == "zero" else z_ob_te
    combined = z_ob_te + best_w * z_ace_te

    base = float(y_te.mean())
    days = len(y_te) / 1440.0
    print("\n" + "=" * 84)
    print(f"[test] n={len(y_te)} 基准={base:.4f} orderbook_seed_iters={iters}  w={best_w}  ob_mode={ob_mode}")
    print(f"[单独AUC] faiss={roc_auc_score(y_te, ace_te):.4f}  orderbook={roc_auc_score(y_te, p_all[te]):.4f}")
    print("=" * 84)
    print(f"{'cov':>6}{'n':>9}{'单/天':>8}{'faiss':>10}{'orderbook':>12}{'combined':>11}{'LR':>9}")
    n1 = 0
    for cov in COVS:
        af, _ = acc_at_cov(ace_te, y_te, cov)
        ao, _ = acc_at_cov(ob_te, y_te, cov)
        ac, n = acc_at_cov(combined, y_te, cov)
        al, _ = acc_at_cov(s_lr_te, y_te, cov)
        if cov == 0.01:
            n1 = n
        print(f"{cov:>6.0%}{n:>9}{n / days:>8.1f}{af:>10.4f}{ao:>12.4f}{ac:>11.4f}{al:>9.4f}")
    best1 = max(acc_at_cov(x, y_te, 0.01)[0] for x in (ace_te, ob_te, combined, s_lr_te))
    print(f"[目标] 1% 覆盖（{n1 / days:.1f} 单/天）最高={best1:.4f}  "
          f"短期 0.65 → {'达成' if best1 >= 0.65 else '未达成'}  "
          f"长期 0.70 → {'达成' if best1 >= 0.70 else '未达成'}")

    agree = np.sign(ace_te) == np.sign(ob_te)
    pred_up = (ace_te > 0).astype(int)
    print(f"[agree] 同向占比={agree.mean():.3f}  同向ACC={float((pred_up[agree] == y_te[agree]).mean()):.4f}")

    # 逐日稳定性（目标口径 = 每天 ~1% 的单量）
    ct_te = ob_ct[te]
    day_id = (ct_te // 86_400_000).astype(np.int64)

    def daily(score, cov, name):
        k = max(1, int(len(y_te) * cov))
        idx = np.argsort(-np.abs(score))[:k]
        d, yv, sv = day_id[idx], y_te[idx], score[idx]
        accs = np.array([float(((sv[d == u] > 0).astype(int) == yv[d == u]).mean())
                         for u in np.unique(d)])
        print(f"[daily] {name}@{cov:.0%}: 天数={len(accs)} 单/天={k / len(np.unique(day_id)):.1f} "
              f"日均ACC={accs.mean():.4f} 中位={np.median(accs):.4f} "
              f"最差10%={np.percentile(accs, 10):.4f} 最差={accs.min():.4f}")

    daily(combined, 0.01, "combined")
    daily(ob_te, 0.01, "orderbook")

    if a.save_pred:
        out = pl.concat([
            pl.DataFrame({"split": ["val"] * len(y_va), "close_time": ob_ct[va],
                          "target": y_va.astype(np.int8), "p_ob": p_all[va],
                          "ace": ace[va], "z_ob": z_ob_va, "z_ace": z_ace_va}),
            pl.DataFrame({"split": ["test"] * len(y_te), "close_time": ct_te,
                          "target": y_te.astype(np.int8), "p_ob": p_all[te],
                          "ace": ace_te, "z_ob": z_ob_te, "z_ace": z_ace_te}),
        ])
        out.write_parquet(a.save_pred)
        print(f"[save] {a.save_pred} rows={out.height}")


if __name__ == "__main__":
    main()
