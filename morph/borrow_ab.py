"""外部参考（termbin f8mx）可借鉴项 A/B —— 全部按我们的口径复核。

来源：https://termbin.com/f8mx（base64 zip，解压到 E:\\codebuddy_wrokspace\\f8mx_extract）。
审计结论：其绝对准确率大量来自「test 集定阈值 / 无 val 训练内定分位 / 评估脚本是 mock」，
数字不可采信；但有三处**思路**我们没系统测过，拿到我们自己的 harness 上做单变量 A/B：

  base      : 现有 63 列 K 线特征（对照，应与 boost 系列同量级）
  dir       : +方向步进包（其 §15）：bar 实体方向累加 cs 的 EMA 偏离 + cs 上的 Stoch69 + 阳线占比
  auto      : +状态量（其 §11）：1m 收益 15/60 阶自相关、布林带宽度扩张比
  diraut    : dir + auto 全加
  regime    : 其 §11 的「分裂专模」：按 autocorr_15>0.05 或 bb_exp>1.15 因果分流，训两个专模路由预测
  shallow   : 其 §17 的极浅树（depth3 / leaves8 / mcs100），轮数给到 200 补偿容量

评估口径（与 fusion/znorm 系列一致，不看单点）：
  - 三个时间窗 off=0/60/120（整体前移，test 段**等长** 96 天，避免窗口长度差异）；
  - static / roll_1d / roll_3d 三种标准化，主看 roll_3d 的三窗均值与最差窗；
  - 附 AUC 与「按天 top15」。
判据：roll3d 三窗均值 +≥0.5pp 且最差窗不退化，才算可借鉴；单窗 <1.5pp 视为噪声（K=1440 时 SE≈1.27pp）。

环境变量：OFFSETS / SEEDS / ROUNDS / TRAIN_STRIDE / VARIANTS / OUT
本地烟测：python -m morph.borrow_ab --synthetic --rows 200000
"""
import argparse
import os
import sys
import time

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lightgbm as lgb  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from morph.kline_boost import (DAILY_K, OB_END, PARAMS, TEST_START, VAL_START,  # noqa: E402
                               build_features, complete_days, daily_topk_eval,
                               day_index, global_topk_eval, load_klines, log,
                               make_target, roll_z, zscore)

H = 3
DAY = 86_400_000
TEST_LEN = OB_END - TEST_START          # off=0 时的 test 段长度（≈96 天）
DIR_COLS = ["b_csdev_5", "b_csdev_10", "b_csdev_20",
            "b_csst_k", "b_csst_d", "b_csst_diff",
            "b_up5", "b_up20"]
AUTO_COLS = ["b_ac15", "b_ac60", "b_bbexp"]
BORROW = DIR_COLS + AUTO_COLS
SHALLOW = dict(num_leaves=8, max_depth=3, min_child_samples=100,
               feature_fraction=0.8, learning_rate=0.05, lambda_l2=1.0)


def add_borrow(d):
    """方向步进（纯方向、幅度不变）+ 状态量。全部 rolling/EMA，严格因果。"""
    c, o, h, l = pl.col("close"), pl.col("open"), pl.col("high"), pl.col("low")
    dirb = (pl.when(c > o).then(1.0).when(c < o).then(-1.0).otherwise(0.0))
    d = d.with_columns(dirb.cast(pl.Float64).alias("_dirb"))
    cs = pl.col("_dirb").cum_sum()
    ex = []
    for span in (5, 10, 20):
        ema = cs.ewm_mean(span=span, adjust=False)
        ex.append((cs - ema).cast(pl.Float32).alias(f"b_csdev_{span}"))
    lo69, hi69 = cs.rolling_min(69), cs.rolling_max(69)
    k = ((cs - lo69) / (hi69 - lo69 + 1e-9) * 100.0).cast(pl.Float64)
    kd = k.rolling_mean(3)
    ex += [k.cast(pl.Float32).alias("b_csst_k"),
           kd.cast(pl.Float32).alias("b_csst_d"),
           (k - kd).cast(pl.Float32).alias("b_csst_diff")]
    up = (pl.col("_dirb") > 0).cast(pl.Float64)
    ex += [up.rolling_mean(5).cast(pl.Float32).alias("b_up5"),
           up.rolling_mean(20).cast(pl.Float32).alias("b_up20")]
    # 1m 收益一阶自相关（15 / 60 根滚动）
    r = c / c.shift(1) - 1.0
    rs = r.shift(1)
    for w in (15, 60):
        cov = (r * rs).rolling_mean(w) - r.rolling_mean(w) * rs.rolling_mean(w)
        sd = r.rolling_std(w) * rs.rolling_std(w) + 1e-12
        ex.append((cov / sd).clip(-1.0, 1.0).cast(pl.Float32).alias(f"b_ac{w}"))
    # 布林带宽度扩张比（宽度 / 过去 60 根平均宽度）
    ma, sd20 = c.rolling_mean(20), c.rolling_std(20)
    width = 4.0 * sd20 / (ma + 1e-9)
    ex.append((width / (width.rolling_mean(60) + 1e-9)).clip(0.0, 10.0)
              .cast(pl.Float32).alias("b_bbexp"))
    return d.with_columns(ex).drop("_dirb")


def fit_score(X, y, tr, cols, seeds, rounds, params=None, extra_w=None):
    p = dict(PARAMS, **(params or {}))
    acc = np.zeros(X.shape[0], dtype=np.float64)
    for sd in seeds:
        q = dict(p, seed=sd, bagging_seed=sd, feature_fraction_seed=sd, data_random_seed=sd)
        ds = lgb.Dataset(X[tr], y[tr], feature_name=cols,
                         weight=None if extra_w is None else extra_w)
        acc += lgb.train(q, ds, num_boost_round=rounds).predict(X)
    return (acc / len(seeds)).astype(np.float32)


def score_eval(s, ct, y, va_m, te_m):
    zs = zscore(s, s[va_m])
    dte = day_index(ct[te_m])
    mt = complete_days(dte, 1200)
    r = {"n": int(te_m.sum()), "days": len(np.unique(dte)), "base": float(y[te_m].mean())}
    for nm, z in (("static", zs), ("roll1d", roll_z(s, 1440)), ("roll3d", roll_z(s, 4320))):
        r[nm] = global_topk_eval(z[te_m], y[te_m], dte, DAILY_K)
    r["auc"] = float(roc_auc_score(y[te_m], zs[te_m]))
    r["daily"] = float(daily_topk_eval(zs[te_m], y[te_m], dte, mt, DAILY_K).mean())
    return r


def synthetic(n=200_000, seed=0):
    rng = np.random.default_rng(seed)
    ret = rng.normal(0, 3e-4, n)
    close = 30000.0 * np.exp(np.cumsum(ret))
    op = np.r_[close[0], close[:-1]]
    hi = np.maximum(op, close) * (1 + np.abs(rng.normal(0, 1.5e-4, n)))
    lo = np.minimum(op, close) * (1 - np.abs(rng.normal(0, 1.5e-4, n)))
    vol = rng.lognormal(0, 1.0, n)
    return pl.DataFrame({
        "close_time": np.arange(n, dtype=np.int64) * 60_000 + 60_000,
        "open": op, "high": hi, "low": lo, "close": close,
        "volume": vol, "taker_buy_volume": vol * rng.uniform(0.3, 0.7, n),
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offsets", default=os.environ.get("OFFSETS", "0,60,120"))
    ap.add_argument("--seeds", default=os.environ.get("SEEDS", "42,43"))
    ap.add_argument("--rounds", type=int, default=int(os.environ.get("ROUNDS", "60")))
    ap.add_argument("--stride", type=int, default=int(os.environ.get("TRAIN_STRIDE", "2")))
    ap.add_argument("--variants", default=os.environ.get(
        "VARIANTS", "base,dir,auto,diraut,regime,shallow"))
    ap.add_argument("--out", default=os.environ.get("OUT", "/kaggle/working/borrow_pred.parquet"))
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--rows", type=int, default=200_000)
    a = ap.parse_args()

    seeds = [int(x) for x in a.seeds.split(",")]
    offs = [int(x) for x in a.offsets.split(",")]
    want = [x.strip() for x in a.variants.split(",") if x.strip()]
    t0 = time.time()

    if a.synthetic:
        global VAL_START, TEST_START, OB_END
        raw = synthetic(a.rows)
        n = raw.height
        VAL_START = int(raw["close_time"][int(n * 0.70)])
        TEST_START = int(raw["close_time"][int(n * 0.80)])
        OB_END = int(raw["close_time"][-1]) + 60_000
        log(f"[synthetic] rows={n}")
    else:
        raw = load_klines().filter(pl.col("close_time") <= OB_END)

    ft, base_cols = build_features(make_target(raw, H))
    ft = add_borrow(ft)
    ct = ft["close_time"].to_numpy()
    y = ft["target"].to_numpy().astype(np.int8)
    cols = base_cols + BORROW
    X = ft.select(cols).to_numpy().astype(np.float32)
    del ft, raw
    log(f"[data] rows={X.shape[0]} feats={len(cols)} (base {len(base_cols)} + borrow {len(BORROW)}) "
        f"{int(ct[0])}~{int(ct[-1])}")

    test_len = (OB_END - TEST_START) if not a.synthetic else (int(ct[-1]) - TEST_START + 60_000)
    is_trend = ((X[:, cols.index("b_ac15")] > 0.05) |
                (X[:, cols.index("b_bbexp")] > 1.15))
    log(f"[regime] trend 占比={is_trend.mean():.3f}")

    results, dump = {}, []
    for off in offs:
        test_end = OB_END - off * DAY
        test_start = test_end - test_len
        val_start = VAL_START - off * DAY
        va_m = (ct >= val_start) & (ct < test_start)
        te_m = (ct >= test_start) & (ct < test_end)
        tr = np.where(ct < val_start)[0]
        va_idx = np.where(va_m)[0]
        if len(va_idx) == 0 or te_m.sum() == 0:
            log(f"[off={off}] 跳过：空区间（synthetic 只支持 off=0）")
            continue
        tr = tr[tr + H < int(va_idx[0])]
        tr_s = tr[::a.stride]
        log(f"\n===== off={off} train={len(tr)} (stride{a.stride}->{len(tr_s)}) "
            f"val={int(va_m.sum())} test={int(te_m.sum())} 天={len(np.unique(day_index(ct[te_m])))} "
            f"基准={y[te_m].mean():.4f} =====")

        def run(tag, sub, rounds=None, params=None):
            if tag not in want:
                return None
            t = time.time()
            s = fit_score(X[:, [cols.index(c) for c in sub]], y, tr_s, sub,
                          seeds, rounds or a.rounds, params)
            r = score_eval(s, ct, y, va_m, te_m)
            results[(off, tag)] = r
            log(f"[{tag:<8}] static={r['static']:.4f} roll1d={r['roll1d']:.4f} "
                f"roll3d={r['roll3d']:.4f} AUC={r['auc']:.4f} 按天={r['daily']:.4f} "
                f"({time.time()-t:.0f}s)")
            dump.append((off, tag, s, te_m))
            return s

        run("base", base_cols)
        run("dir", base_cols + DIR_COLS)
        run("auto", base_cols + AUTO_COLS)
        run("diraut", cols)

        if "regime" in want:
            t = time.time()
            s = np.zeros(X.shape[0], dtype=np.float64)
            idx = np.arange(len(cols))
            for flag, mask in ((True, is_trend), (False, ~is_trend)):
                trs = tr_s[mask[tr_s]]
                if len(trs) < 5000:
                    log(f"[regime] {'trend' if flag else 'range'} 样本不足 {len(trs)}，整体回退")
                    trs = tr_s
                s += np.where(mask, fit_score(X[:, idx], y, trs, cols, seeds, a.rounds), 0.0)
            r = score_eval(s.astype(np.float32), ct, y, va_m, te_m)
            results[(off, "regime")] = r
            log(f"[regime  ] static={r['static']:.4f} roll1d={r['roll1d']:.4f} "
                f"roll3d={r['roll3d']:.4f} AUC={r['auc']:.4f} 按天={r['daily']:.4f} "
                f"({time.time()-t:.0f}s)")
            dump.append((off, "regime", s.astype(np.float32), te_m))

        run("shallow", base_cols, rounds=200, params=SHALLOW)

        if a.synthetic:
            break

    # ---- 汇总：单窗表 + 跨窗均值/最差窗 ----
    log("\n" + "=" * 92)
    log(f"{'变体':<10}{'off':>5}{'static':>9}{'roll1d':>9}{'roll3d':>9}{'AUC':>8}{'按天':>8}")
    for (off, tag), r in results.items():
        log(f"{tag:<10}{off:>5}{r['static']:>9.4f}{r['roll1d']:>9.4f}"
            f"{r['roll3d']:>9.4f}{r['auc']:>8.4f}{r['daily']:>8.4f}")

    log("\n" + "=" * 92)
    log(f"{'变体':<10}{'roll3d均值':>12}{'roll3d最差':>12}{'roll1d均值':>12}"
        f"{'AUC均值':>10}{'按天均值':>10}{'Δroll3d':>10}")
    tags = sorted({t for _, t in results}, key=lambda t: want.index(t) if t in want else 99)
    base_mean = None
    for tag in tags:
        rs = [r for (o, t), r in results.items() if t == tag]
        if not rs:
            continue
        m3 = float(np.mean([r["roll3d"] for r in rs]))
        w3 = float(np.min([r["roll3d"] for r in rs]))
        m1 = float(np.mean([r["roll1d"] for r in rs]))
        ma = float(np.mean([r["auc"] for r in rs]))
        md = float(np.mean([r["daily"] for r in rs]))
        if tag == "base":
            base_mean = m3
        d = "" if base_mean is None or tag == "base" else f"{(m3-base_mean)*100:+.2f}pp"
        log(f"{tag:<10}{m3:>12.4f}{w3:>12.4f}{m1:>12.4f}{ma:>10.4f}{md:>10.4f}{d:>10}")

    if not a.synthetic and a.out:
        try:
            frames = []
            for off, tag, s, te_m in dump:
                sub = pl.DataFrame({"close_time": ct[te_m], "off": off, "tag": tag,
                                    "score": s[te_m], "target": y[te_m].astype(np.int8)})
                frames.append(sub)
            pl.concat(frames).write_parquet(a.out)
            log(f"\n[save] {a.out}")
        except Exception as e:
            log(f"[save] 失败: {e}")

    log(f"\n[done] {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
