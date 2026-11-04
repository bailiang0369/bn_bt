"""第二轮：先量化噪声，再谈提升。

上一轮 v0-v4 的 topK 差异都在 1pt 量级，而 global topK 只取 1% 的极端样本，
同一天内高度相关，方差远大于 1pt。不做噪声基线就无法区分"提升"和"抖动"。

本轮统一口径（2026-07-17 ~ 2026-10-21，96 天，K=15/天）：
  base       v0：63 特征，5 个 seed **各自**评估，得到 seed 间 sd
  win4       同一份 base 分数切成 4 个 24 天窗口，得到窗口间波动
  v5         形态 26 + 符号 12 = 38 特征（去掉量能/时间/动量的冗余统计）
  b5/b10     训练集剔除 |3min 收益| < 5/10 bps 的样本（标签净化）
  v5b5       v5 + 5bps 净化

标签净化的理由：3 分钟收益的绝对值小于几个 bp 时方向本质是噪声，这些样本在
训练里贡献的是梯度噪音，在 test 里却占 topK 名额（模型对它们给出大 |score| 时
纯属运气）。剔除后模型应更专注于真实方向。
"""
import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lightgbm as lgb  # noqa: E402

from morph.kline_boost import (DAILY_K, OB_END, PARAMS, SHAPE_PREFIX, SIGN_COLS,  # noqa: E402
                               TEST_START, VAL_START, add_sign_feats, build_features,
                               day_index, global_topk_eval, load_klines, log,
                               score_eval, zscore)


def make_all(raw, h):
    d = (raw.with_columns(pl.col("close").shift(-h).alias("_fc"))
         .with_columns((pl.col("_fc") / pl.col("close") - 1.0).alias("_fret"))
         .with_columns((pl.col("_fc") > pl.col("close")).cast(pl.Int8).alias("target"))
         .drop("_fc").filter(pl.col("target").is_not_null()))
    ft, cols = build_features(d)
    return add_sign_feats(ft), cols


def fit(ft, cols, h, seeds, rounds, thr_bps=0.0):
    ct = ft["close_time"].to_numpy()
    y = ft["target"].to_numpy().astype(np.int8)
    fret = ft["_fret"].to_numpy()
    X = ft.select(cols).to_numpy().astype(np.float32)
    tr = np.where(ct < VAL_START)[0]
    va = np.where((ct >= VAL_START) & (ct < TEST_START))[0]
    tr = tr[tr + h < int(va[0])]
    if thr_bps > 0:
        keep = np.abs(fret[tr]) >= thr_bps * 1e-4
        log(f"  净化 thr={thr_bps}bps：train {len(tr)} -> {int(keep.sum())}")
        tr = tr[keep]
    per_seed = []
    for sd in seeds:
        p = dict(PARAMS, seed=sd, bagging_seed=sd, feature_fraction_seed=sd,
                 data_random_seed=sd)
        m = lgb.train(p, lgb.Dataset(X[tr], y[tr], feature_name=cols),
                      num_boost_round=rounds)
        per_seed.append(m.predict(X))
    return per_seed, ct, y, fret


def window_eval(z, y, ct, te, n_win=4):
    edges = np.linspace(int(ct[te[0]]), int(ct[te[-1]]) + 1, n_win + 1).astype(np.int64)
    out = []
    for i in range(n_win):
        m = (ct[te] >= edges[i]) & (ct[te] < edges[i + 1])
        idx = te[m]
        if len(idx) == 0:
            continue
        dd = day_index(ct[idx])
        out.append(global_topk_eval(z[idx], y[idx], dd, DAILY_K))
    return np.array(out)


def subset_eval(z, y, ct, te, fret, thr_bps):
    idx = te[np.abs(fret[te]) >= thr_bps * 1e-4]
    dd = day_index(ct[idx])
    return global_topk_eval(z[idx], y[idx], dd, DAILY_K)


def main():
    seeds5 = [int(x) for x in os.environ.get("SEEDS", "42,43,44,45,46").split(",")]
    seeds2 = seeds5[:2]
    rounds = int(os.environ.get("ROUNDS", "60"))
    only = [x for x in os.environ.get("ONLY", "").split(",") if x]
    want = (lambda t: (not only) or t in only)

    raw = load_klines().filter(pl.col("close_time") <= OB_END)
    log(f"[1m] rows={raw.height} 量能={'volume' in raw.columns}")
    ft, cols = make_all(raw, 3)
    shape = [c for c in cols if c.startswith(SHAPE_PREFIX)]
    v5 = shape + SIGN_COLS
    log(f"[cols] 全量={len(cols)} 形态={len(shape)} v5={len(v5)}")

    rows, store = {}, {}

    def evaluate(tag, per_seed, ct, y, fret, nfeat):
        p = np.mean(per_seed, axis=0)
        r = score_eval(tag, p, ct, y, 3, 1200, 1440, nfeat, rows)
        te = np.where(ct >= TEST_START)[0]
        z = zscore(p, p[(ct >= VAL_START) & (ct < TEST_START)])
        w = window_eval(z, y, ct, te)
        r["w4"], r["w4sd"] = float(w.mean()), float(w.std(ddof=1))
        r["big5"] = subset_eval(z, y, ct, te, fret, 5)
        r["big10"] = subset_eval(z, y, ct, te, fret, 10)
        log(f"   窗口4段={r['w4']:.4f}±{r['w4sd']:.4f} | |ret|>=5bps 子集={r['big5']:.4f} "
            f">=10bps={r['big10']:.4f}")
        store[tag] = p
        return r

    if want("base") or want("noise"):
        log("[base] 63 特征 × 5 seed（各自评估，量化 seed 波动）")
        ps, ct, y, fret = fit(ft, cols, 3, seeds5, rounds)
        evaluate("base", ps, ct, y, fret, len(cols))
        st = []
        for i, p in enumerate(ps):
            z = zscore(p, p[(ct >= VAL_START) & (ct < TEST_START)])
            te = np.where(ct >= TEST_START)[0]
            dd = day_index(ct[te])
            st.append(global_topk_eval(z[te], y[te], dd, DAILY_K))
        st = np.array(st)
        log(f"[noise] seed static: {np.round(st, 4).tolist()} "
            f"mean={st.mean():.4f} sd={st.std(ddof=1):.4f}")

    if want("v5"):
        log("[v5] 形态 + 符号 38 特征")
        ps, ct, y, fret = fit(ft, v5, 3, seeds2, rounds)
        evaluate("v5", ps, ct, y, fret, len(v5))

    if want("b5"):
        log("[b5] 63 特征 + 5bps 净化")
        ps, ct, y, fret = fit(ft, cols, 3, seeds2, rounds, thr_bps=5)
        evaluate("b5", ps, ct, y, fret, len(cols))

    if want("b10"):
        log("[b10] 63 特征 + 10bps 净化")
        ps, ct, y, fret = fit(ft, cols, 3, seeds2, rounds, thr_bps=10)
        evaluate("b10", ps, ct, y, fret, len(cols))

    if want("v5b5"):
        log("[v5b5] v5 + 5bps 净化")
        ps, ct, y, fret = fit(ft, v5, 3, seeds2, rounds, thr_bps=5)
        evaluate("v5b5", ps, ct, y, fret, len(v5))

    if "base" in store:
        for k, p in store.items():
            if k != "base":
                c = float(np.corrcoef(store["base"], p)[0, 1])
                log(f"[corr] base vs {k} = {c:.4f}")

    log(f"\n{'变体':<10}{'特征':>6}{'static':>9}{'roll1d':>9}{'roll3d':>9}{'AUC':>8}"
        f"{'按天':>8}{'窗口4':>8}{'4段sd':>8}{'>5bps':>8}{'>10bps':>8}")
    for k, r in rows.items():
        log(f"{k:<10}{r['feats']:>6}{r['static']:>9.4f}{r['roll1d']:>9.4f}"
            f"{r['roll3d']:>9.4f}{r['auc']:>8.4f}{r['daily']:>8.4f}{r['w4']:>8.4f}"
            f"{r['w4sd']:>8.4f}{r['big5']:>8.4f}{r['big10']:>8.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
