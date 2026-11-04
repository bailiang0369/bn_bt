"""第三轮：只试三个可能真正带来增益的方向，其余已被前两轮否掉。

前两轮的硬结论（96 天 test，K=15/天）：
  - seed 间 sd = 0.47pt，窗口间 sd = 2.9pt → 1pt 以内不可区分；
  - 3m 重采样、形态子集、符号叠加、硬剔除小波动样本，全部显著变差；
  - AUC 卡在 0.527，说明特征侧没有新信息被漏掉。

因此本轮只保留三个"改变集成/训练分布"的方向：
  s8    8 seed 平均（base 是 5 seed 的 0.6125，看多 seed 是否还有边际）
  w5/w10 软加权 w = clip(|3min ret| / q, 0, 1)：硬剔除会毁掉全样本排序，
         软加权只在尾部提权重，应避免 b10 那种灾难
  h123  h=1/2/3 三个模型融合：1min 模型在 t..t+2 的分数之和近似 t->t+3 的走向
  div   多样性 bagging（num_leaves 31/127、feature_fraction 0.5/0.9）+ base 平均
"""
import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lightgbm as lgb  # noqa: E402

from morph.kline_boost import (DAILY_K, OB_END, PARAMS, TEST_START, VAL_START,  # noqa: E402
                               build_features, day_index, global_topk_eval, load_klines,
                               log, score_eval, zscore)


def target_of(raw, h):
    return (raw.select(
        pl.col("close_time"),
        (pl.col("close").shift(-h) / pl.col("close") - 1.0).alias("fret"),
        (pl.col("close").shift(-h) > pl.col("close")).cast(pl.Int8).alias("target"),
    ).filter(pl.col("target").is_not_null()))


def train_one(X, y, tr, cols, sd, rounds, over=None, w=None):
    p = dict(PARAMS, seed=sd, bagging_seed=sd, feature_fraction_seed=sd, data_random_seed=sd)
    if over:
        p.update(over)
    ds = lgb.Dataset(X[tr], y[tr], weight=None if w is None else w[tr], feature_name=cols)
    return lgb.train(p, ds, num_boost_round=rounds).predict(X)


def main():
    seeds = [int(x) for x in os.environ.get("SEEDS", "42,43,44,45,46,47,48,49").split(",")]
    rounds = int(os.environ.get("ROUNDS", "60"))
    only = [x for x in os.environ.get("ONLY", "").split(",") if x]
    want = (lambda t: (not only) or t in only)

    raw = load_klines().filter(pl.col("close_time") <= OB_END)
    log(f"[1m] rows={raw.height}")
    ft_all, cols = build_features(raw)
    t3 = target_of(raw, 3)
    n3 = t3.height
    t1 = target_of(raw, 1)
    nmax = t1.height
    ct = t3["close_time"].to_numpy()
    y = t3["target"].to_numpy().astype(np.int8)
    fret = t3["fret"].to_numpy()
    X = ft_all.head(nmax).select(cols).to_numpy().astype(np.float32)
    del ft_all

    tr = np.where(ct < VAL_START)[0]
    va = np.where((ct >= VAL_START) & (ct < TEST_START))[0]
    tr = tr[tr + 3 < int(va[0])]
    te = np.where(ct >= TEST_START)[0]
    log(f"[split] train={len(tr)} val={len(va)} test={len(te)} "
        f"天数={len(np.unique(day_index(ct[te])))}")

    rows, store = {}, {}

    def report(tag, p, nfeat):
        r = score_eval(tag, p, ct, y, 3, 1200, 1440, nfeat, rows)
        z = zscore(p, p[va])
        idx = te[np.abs(fret[te]) >= 5e-4]
        r["big5"] = global_topk_eval(z[idx], y[idx], day_index(ct[idx]), DAILY_K)
        return r

    if want("s8"):
        log(f"[s8] 63 特征 × {len(seeds)} seed")
        acc = np.zeros(n3)
        for sd in seeds:
            acc += train_one(X, y, tr, cols, sd, rounds)
        store["s8"] = acc / len(seeds)
        report("s8", store["s8"], len(cols))

    if want("w5") or want("w10"):
        for q in (5, 10):
            tag = f"w{q}"
            if not want(tag):
                continue
            w = np.clip(np.abs(fret) / (q * 1e-4), 0.0, 1.0)
            log(f"[{tag}] 软加权 q={q}bps 平均权重={w[tr].mean():.3f}")
            acc = np.zeros(n3)
            for sd in seeds[:2]:
                acc += train_one(X, y, tr, cols, sd, rounds, w=w)
            store[tag] = acc / 2
            report(tag, store[tag], len(cols))

    if want("h123"):
        log("[h123] h=1/2/3 融合")
        parts = []
        for h in (1, 2, 3):
            if h == 3 and "s8" in store:
                ph = store["s8"].copy()
            else:
                th = target_of(raw, h)
                nh = th.height
                yh = th["target"].to_numpy().astype(np.int8)
                acc = np.zeros(nh)
                for sd in seeds[:2]:
                    acc += train_one(X[:nh], yh, tr, cols, sd, rounds)
                ph = (acc / 2)[:n3]
            if h == 1:
                ph = ph + np.roll(ph, -1) + np.roll(ph, -2)
            elif h == 2:
                ph = ph + np.roll(ph, -1)
            parts.append(zscore(ph, ph[va]))
            log(f"  h={h} done")
        store["h123"] = np.mean(parts, axis=0)
        report("h123", store["h123"], len(cols))

    if want("div"):
        log("[div] 多样性 bagging")
        acc = np.zeros(n3)
        cfgs = [dict(num_leaves=31, feature_fraction=0.5),
                dict(num_leaves=127, feature_fraction=0.9)]
        for c in cfgs:
            for sd in seeds[:2]:
                acc += train_one(X, y, tr, cols, sd, rounds, over=c)
        store["div"] = acc / (2 * len(cfgs))
        report("div", store["div"], len(cols))
        if "s8" in store:
            mix = 0.5 * zscore(store["s8"], store["s8"][va]) + \
                0.5 * zscore(store["div"], store["div"][va])
            report("div+s8", mix, len(cols))

    log(f"\n{'变体':<8}{'特征':>6}{'static':>9}{'roll1d':>9}{'roll3d':>9}{'AUC':>8}"
        f"{'按天':>8}{'>5bps':>8}")
    for k, r in rows.items():
        log(f"{k:<8}{r['feats']:>6}{r['static']:>9.4f}{r['roll1d']:>9.4f}"
            f"{r['roll3d']:>9.4f}{r['auc']:>8.4f}{r['daily']:>8.4f}{r['big5']:>8.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
