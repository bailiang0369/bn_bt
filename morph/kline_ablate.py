"""K 线腿特征组消融：先把"独立信息在哪一类特征上"问清楚（第 1 步：单腿提升）。

动机
----
63 列特征一股脑喂给 LGB，AUC 只有 0.530，roll_3d topK 0.584。
而 §28 指出 K 线腿与盘口腿相关 0.515，重叠的是 `k_ret_*`（价格动量）。
所以先做单变量归因，回答两个问题：
  1. 哪一组特征真的携带方向信息（去掉它 AUC 掉不掉）？
  2. 剔掉价格动量后，K 线腿还剩多少（这决定融合还有没有空间）？

特征分组（按 morph/kline_features.py 的命名）
  MOM   k_ret_*
  VOL   k_vol_* / k_absret_* / k_atr_* / k_pos_*
  FLOW  log_vol / vr_* / tbr_* / tbrs_* / net_* / vpin_*
  SHAPE k_body/k_upper/k_lower/k_gap / k_rsi_* / k_boll_* / k_stoch_* / k_streak*
  TIME  hour_* / dow_* / us_session

变体（h=3，三个窗口 off=0/60/120）
  all        63 列（对照）
  no_mom     去掉 MOM（预期掉一点，但可能与盘口去相关）
  flow_shape 只留 FLOW + SHAPE（无动量、无时间）
  flow_only  只留 FLOW（纯筹码/量能，最可能与盘口互补）
  shape_only 只留 SHAPE
  vol_shape  VOL + SHAPE

判据（预先写死，防止自己在 test 上挑）
  单腿可用 = roll3d 三窗均值 ≥ all + 0.5pp，且 roll1d 均值不退化、最差窗不退化。
  即使不过线，只要 `flow_shape`/`flow_only` 的 AUC 不崩（≥0.525），仍保留给融合阶段做去相关腿。

产物：所有变体的 test 段分数落盘（close_time/off/tag/score/target），供融合阶段直接复用。
环境变量：OFFSETS / SEEDS / ROUNDS / TRAIN_STRIDE / VARIANTS / OUT
"""
import os
import sys
import time

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lightgbm as lgb  # noqa: E402

from morph.borrow_ab import score_eval  # noqa: E402
from morph.kline_boost import (OB_END, PARAMS, TEST_START, VAL_START,  # noqa: E402
                               build_features, load_klines, log, make_target)

H = 3
DAY = 86_400_000

MOM = ("k_ret_",)
VOL = ("k_vol_", "k_absret_", "k_atr_", "k_pos_")
FLOW = ("log_vol", "vr_", "tbr_", "tbrs_", "net_", "vpin_")
SHAPE = ("k_body", "k_upper", "k_lower", "k_gap", "k_rsi_", "k_boll_", "k_stoch_", "k_streak")
TIME = ("hour_", "dow_", "us_session")


def group_of(c):
    for name, pres in (("MOM", MOM), ("VOL", VOL), ("FLOW", FLOW),
                       ("SHAPE", SHAPE), ("TIME", TIME)):
        if c.startswith(pres):
            return name
    return "OTHER"


def subset(cols, groups):
    return [c for c in cols if group_of(c) in groups]


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
    stride = int(os.environ.get("TRAIN_STRIDE", "2"))
    offs = [int(x) for x in os.environ.get("OFFSETS", "0,60,120").split(",")]
    want = [x.strip() for x in os.environ.get(
        "VARIANTS", "all,no_mom,flow_shape,flow_only,shape_only,vol_shape").split(",") if x.strip()]
    out = os.environ.get("OUT", "")

    raw = load_klines().filter(pl.col("close_time") <= OB_END)
    ft, cols = build_features(make_target(raw, H))
    ct = ft["close_time"].to_numpy()
    y = ft["target"].to_numpy().astype(np.int8)
    X_all = ft.select(cols).to_numpy().astype(np.float32)
    del ft, raw
    gmap = {c: group_of(c) for c in cols}
    cnt = {g: sum(1 for c in cols if gmap[c] == g) for g in ("MOM", "VOL", "FLOW", "SHAPE", "TIME", "OTHER")}
    log(f"[data] rows={X_all.shape[0]} feats={len(cols)} 分组={cnt}")

    sets = {
        "all": cols,
        "no_mom": subset(cols, {"VOL", "FLOW", "SHAPE", "TIME", "OTHER"}),
        "flow_shape": subset(cols, {"FLOW", "SHAPE"}),
        "flow_only": subset(cols, {"FLOW"}),
        "shape_only": subset(cols, {"SHAPE"}),
        "vol_shape": subset(cols, {"VOL", "SHAPE"}),
    }
    for k, v in sets.items():
        log(f"  {k:<11} {len(v):>3} 列")

    test_len = OB_END - TEST_START
    rows, dump = {}, []
    t0 = time.time()
    for off in offs:
        test_end = OB_END - off * DAY
        test_start = test_end - test_len
        val_start = VAL_START - off * DAY
        va_m = (ct >= val_start) & (ct < test_start)
        te_m = (ct >= test_start) & (ct < test_end)
        va_idx = np.where(va_m)[0]
        if len(va_idx) == 0 or te_m.sum() == 0:
            continue
        tr0 = np.where(ct < val_start)[0]
        tr = tr0[tr0 + H < int(va_idx[0])][::stride]
        log(f"\n===== off={off} train={len(tr)} val={int(va_m.sum())} test={int(te_m.sum())} =====")
        for tag in want:
            sub = sets[tag]
            idx = [cols.index(c) for c in sub]
            s = fit_scores(X_all[:, idx], y, tr, sub, seeds, rounds)
            r = score_eval(s, ct, y, va_m, te_m)
            rows[(off, tag)] = r
            log(f"[{tag:<11} {len(sub):>2}f] static={r['static']:.4f} roll1d={r['roll1d']:.4f} "
                f"roll3d={r['roll3d']:.4f} AUC={r['auc']:.4f} 按天={r['daily']:.4f}")
            dump.append((off, tag, s[te_m], ct[te_m], y[te_m]))

    log(f"\n{'变体':<12}{'off':>5}{'static':>9}{'roll1d':>9}{'roll3d':>9}{'AUC':>8}{'按天':>8}")
    for (off, tag), r in rows.items():
        log(f"{tag:<12}{off:>5}{r['static']:>9.4f}{r['roll1d']:>9.4f}"
            f"{r['roll3d']:>9.4f}{r['auc']:>8.4f}{r['daily']:>8.4f}")

    log(f"\n{'变体':<12}{'roll3d均值':>11}{'roll3d最差':>11}{'roll1d均值':>11}"
        f"{'AUC均值':>9}{'按天均值':>10}{'Δroll3d':>10}")
    base3 = np.mean([rows[(o, "all")]["roll3d"] for o in offs if (o, "all") in rows]) \
        if any((o, "all") in rows for o in offs) else None
    for tag in want:
        rs = [rows[(o, tag)] for o in offs if (o, tag) in rows]
        if not rs:
            continue
        m3 = float(np.mean([r["roll3d"] for r in rs]))
        d = "" if base3 is None or tag == "all" else f"{(m3-base3)*100:+.2f}pp"
        log(f"{tag:<12}{m3:>11.4f}{np.min([r['roll3d'] for r in rs]):>11.4f}"
            f"{np.mean([r['roll1d'] for r in rs]):>11.4f}"
            f"{np.mean([r['auc'] for r in rs]):>9.4f}"
            f"{np.mean([r['daily'] for r in rs]):>10.4f}{d:>10}")

    if out:
        try:
            pl.concat([pl.DataFrame({"off": off, "tag": tag, "close_time": c,
                                     "score": s, "target": t})
                       for off, tag, s, c, t in dump]).write_parquet(out)
            log(f"[save] {out}")
        except Exception as e:
            log(f"[save] 失败 {e}")
    log(f"[done] {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
