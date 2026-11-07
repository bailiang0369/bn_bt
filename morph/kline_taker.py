"""taker 主动成交多空量比（Binance metrics，5 分钟粒度）作为 K 线腿新信息。

来由
----
`morph/newinfo_diag.py` 单特征体检发现：`sum_taker_long_short_vol_ratio` 单特征 AUC≈0.68
（其余 13 个仓位/资金费率特征全在 0.50），是唯一有信息量的新数据。
但它可能来自 5 分钟桶的**桶内前视**（create_time 是窗口起点时，t 处能看到 t~t+5m 的成交）。

所以本实验同时跑 lag=0 与 lag=5min 两个口径：
  - lag5 是**可上线**的诚实口径（只用上一个完整 5 分钟桶）；
  - lag0 与 lag5 的差值 = 前视带来的虚高，用来量化"看着很香其实是偷看"。

变体（都是 h=3、3 窗 off=0/60/120、2 seeds × 60 轮）
  base    54 列 K 线（no_mom，§32 最优）+ 全历史
  recent  54 列 K 线，但只用"有 metrics 的时段"训练（控制样本量变量）
  lag5    54 + 8 列（taker/OI/大户，滞后 5 分钟），只用有数据时段训练
  lag0    同 lag5，但滞后 0（对比前视幅度）
  takonly 只用 8 列新特征（滞后 5 分钟）

判据：lag5 相对 recent 的 roll3d 三窗均值 ≥ +0.5pp 且最差窗不退化。
"""
import os
import sys
import time

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lightgbm as lgb  # noqa: E402

from morph.borrow_ab import score_eval  # noqa: E402
from morph.kline_ablate import group_of  # noqa: E402
from morph.kline_boost import (OB_END, PARAMS, TEST_START, VAL_START,  # noqa: E402
                               build_features, load_klines, log, make_target)
from morph.kline_newinfo import build, load_kind  # noqa: E402

H = 3
DAY = 86_400_000
TAK = ["t_tak", "t_tak_z", "t_tak_ma3", "t_tak_d",
       "t_dok", "t_doi_1h", "t_oi_z", "t_lsr_z"]


def metrics_frame(m):
    """在 5 分钟网格上构造特征（全部向后看）。"""
    return m.with_columns([
        pl.col("sum_taker_long_short_vol_ratio").cast(pl.Float64).alias("tak"),
        pl.col("sum_open_interest").cast(pl.Float64).log1p().alias("oi"),
        pl.col("count_toptrader_long_short_ratio").cast(pl.Float64).alias("lsr"),
    ]).with_columns([
        ((pl.col("tak") - pl.col("tak").rolling_mean(288))
         / (pl.col("tak").rolling_std(288) + 1e-12)).alias("tak_z"),
        pl.col("tak").rolling_mean(3).alias("tak_ma3"),
        (pl.col("tak") - pl.col("tak").shift(1)).alias("tak_d"),
        (pl.col("oi") / pl.col("oi").shift(1) - 1.0).alias("dok"),
        (pl.col("oi") / pl.col("oi").shift(12) - 1.0).alias("doi_1h"),
        ((pl.col("oi") - pl.col("oi").rolling_mean(288))
         / (pl.col("oi").rolling_std(288) + 1e-12)).alias("oi_z"),
        ((pl.col("lsr") - pl.col("lsr").rolling_mean(288))
         / (pl.col("lsr").rolling_std(288) + 1e-12)).alias("lsr_z"),
    ])


def asof(ts, vals, query):
    ts = np.asarray(ts, dtype=np.int64)
    if len(ts) == 0:
        return np.full(len(query), np.nan)
    idx = np.searchsorted(ts, query, side="right") - 1
    v = np.asarray(vals, dtype=np.float64)
    return np.where(idx >= 0, v[np.clip(idx, 0, len(v) - 1)], np.nan)


def fit_scores(X, y, tr, cols, seeds, rounds):
    acc = np.zeros(X.shape[0], dtype=np.float64)
    for sd in seeds:
        q = dict(PARAMS, seed=sd, bagging_seed=sd, feature_fraction_seed=sd, data_random_seed=sd)
        acc += lgb.train(q, lgb.Dataset(X[tr], y[tr], feature_name=cols),
                         num_boost_round=rounds).predict(X)
    return (acc / len(seeds)).astype(np.float32)


def main():
    seeds = [int(x) for x in os.environ.get("SEEDS", "42,43").split(",")]
    rounds = int(os.environ.get("ROUNDS", "60"))
    stride = int(os.environ.get("TRAIN_STRIDE", "1"))
    offs = [int(x) for x in os.environ.get("OFFSETS", "0,60,120").split(",")]
    want = [x.strip() for x in os.environ.get(
        "VARIANTS", "base,recent,lag5,lag0,takonly").split(",") if x.strip()]
    t0 = time.time()

    raw = load_klines().filter(pl.col("close_time") <= OB_END)
    ft, base_cols = build_features(make_target(raw, H))
    kl_cols = [c for c in base_cols if group_of(c) != "MOM"]
    ct = ft["close_time"].to_numpy()
    y = ft["target"].to_numpy().astype(np.int8)

    m = build("metrics", load_kind("metrics"))
    if m is None:
        log("[fatal] metrics 无数据")
        return 1
    sp = metrics_frame(m)
    t = sp["t"].to_numpy()
    log(f"[metrics] n={len(t)} {pl.Series(t[:1]).cast(pl.Datetime('ms')).to_list()} ~ "
        f"{pl.Series(t[-1:]).cast(pl.Datetime('ms')).to_list()}")

    feats = {}
    for lag in (0, 5):
        q = ct - lag * 60_000
        for name, src in (("t_tak", "tak"), ("t_tak_z", "tak_z"), ("t_tak_ma3", "tak_ma3"),
                          ("t_tak_d", "tak_d"), ("t_dok", "dok"), ("t_doi_1h", "doi_1h"),
                          ("t_oi_z", "oi_z"), ("t_lsr_z", "lsr_z")):
            feats[(lag, name)] = asof(t, sp[src].to_numpy(), q).astype(np.float32)

    ft = ft.with_columns([pl.Series(f"l0_{k}", v) for (lag, k), v in feats.items() if lag == 0]
                         + [pl.Series(k, v) for (lag, k), v in feats.items() if lag == 5])
    log(f"[data] rows={ct.size} kl57={len(kl_cols)} taker覆盖={np.isfinite(feats[(5,'t_tak')]).mean():.2%}")

    cols_l5 = kl_cols + TAK
    X_l5 = ft.select(cols_l5).to_numpy().astype(np.float32)
    X_kl = ft.select(kl_cols).to_numpy().astype(np.float32)
    X_l0 = ft.select(kl_cols + [f"l0_{c}" for c in TAK]).to_numpy().astype(np.float32)
    X_tk = ft.select(TAK).to_numpy().astype(np.float32)
    has = np.isfinite(feats[(5, "t_tak")])
    del ft, raw

    test_len = OB_END - TEST_START
    rows = {}
    for off in offs:
        test_end = OB_END - off * DAY
        test_start = test_end - test_len
        val_start = VAL_START - off * DAY
        va_m = (ct >= val_start) & (ct < test_start)
        te_m = (ct >= test_start) & (ct < test_end)
        va_idx = np.where(va_m)[0]
        if len(va_idx) == 0 or te_m.sum() == 0:
            continue
        tr_full = np.where(ct < val_start)[0]
        tr_full = tr_full[tr_full + H < int(va_idx[0])][::stride]
        tr_rec = tr_full[has[tr_full]]
        log(f"\n===== off={off} train_full={len(tr_full)} train_recent={len(tr_rec)} "
            f"test={int(te_m.sum())} 测试段taker覆盖={has[te_m].mean():.2%} =====")

        plan = {"base": (X_kl, kl_cols, tr_full),
                "recent": (X_kl, kl_cols, tr_rec),
                "lag5": (X_l5, cols_l5, tr_rec),
                "lag0": (X_l0, kl_cols + [f"l0_{c}" for c in TAK], tr_rec),
                "takonly": (X_tk, TAK, tr_rec)}
        for tag in want:
            X, cols, tr = plan[tag]
            s = fit_scores(X, y, tr, cols, seeds, rounds)
            r = score_eval(s, ct, y, va_m, te_m)
            rows[(off, tag)] = r
            log(f"[{tag:<8} {len(cols):>2}f] static={r['static']:.4f} roll1d={r['roll1d']:.4f} "
                f"roll3d={r['roll3d']:.4f} AUC={r['auc']:.4f} 按天={r['daily']:.4f}")

    log(f"\n{'变体':<9}{'roll3d均值':>11}{'roll3d最差':>11}{'roll1d均值':>11}{'AUC均值':>9}{'按天均值':>10}")
    for tag in want:
        rs = [rows[(o, tag)] for o in offs if (o, tag) in rows]
        if not rs:
            continue
        log(f"{tag:<9}{np.mean([r['roll3d'] for r in rs]):>11.4f}"
            f"{np.min([r['roll3d'] for r in rs]):>11.4f}"
            f"{np.mean([r['roll1d'] for r in rs]):>11.4f}"
            f"{np.mean([r['auc'] for r in rs]):>9.4f}"
            f"{np.mean([r['daily'] for r in rs]):>10.4f}")
    log(f"[done] {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
