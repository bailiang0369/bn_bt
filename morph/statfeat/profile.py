#!/usr/bin/env python3
"""统计特征**画像归档**（不淘汰）。

与 validate.py 的区别：validate 是「能否用于 h=3/15 事件合约」的准入门，
本模块是知识库建设——每个特征在 3 分钟到 1 周共 8 个 horizon、3 段时间、
多个标的上的统计属性全部记录，并标注它适配哪个周期/场景。
用户 2026-11-11：举例只是例子，长周期统计未必用于二元事件合约，
可能用于长周期现货/合约交易；「尽可能多做总结，未来总有用处」。

输出长表 profile.csv，每行 = (标的, 特征, horizon)，列含
全段/分段 IC、符号一致性、档位差、分布矩、自相关、与旧特征相关性、因果性、场景。
矩阵化：特征秩矩阵只算一次，8 个 horizon 只做 8 次矩阵点积。

    python -m morph.statfeat.profile --synthetic            # 本地烟测
    python -m morph.statfeat.profile --step 5 --out /kaggle/working/profile_btc.csv
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl

try:
    from .features import REGISTRY
    from .validate import baseline_feats, causality_ok, load_data, max_base_corr
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from morph.statfeat.features import REGISTRY
    from morph.statfeat.validate import baseline_feats, causality_ok, load_data, max_base_corr

from scipy.stats import rankdata, skew, kurtosis

HORIZONS = (3, 5, 15, 30, 60, 240, 1440, 10080)
N_SEG = 3


def log(msg: str) -> None:
    print(msg, flush=True)


def scenario(h: int) -> str:
    if h <= 60:
        return "event_3_60m"
    if h <= 1440:
        return "intraday_4h_1d"
    return "swing_1w"


def rank_matrix(x: np.ndarray) -> np.ndarray:
    """逐列 rankdata 后标准化；NaN 保持 NaN（冷启动点）。"""
    out = np.full(x.shape, np.nan, dtype=np.float64)
    for j in range(x.shape[1]):
        c = x[:, j]
        ok = np.isfinite(c)
        if ok.sum() < 100:
            continue
        out[ok, j] = rankdata(c[ok])
    return out


def describe(x: np.ndarray) -> tuple[float, ...]:
    """分布形态 + 水平统计。用户举例的现象本身就是描述统计，必须归档。"""
    v = x[np.isfinite(x)]
    if len(v) < 1000:
        return (0.0,) * 7
    ac = float(np.corrcoef(v[:-1], v[1:])[0, 1])
    return (float(skew(v)), float(kurtosis(v)),
            ac if np.isfinite(ac) else 0.0,
            float(np.mean(v)), float(np.median(v)),
            float(np.quantile(v, 0.1)), float(np.quantile(v, 0.9)))


def main() -> None:
    ap = argparse.ArgumentParser(description="统计特征多 horizon 画像归档")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--rows", type=int, default=200000)
    ap.add_argument("--step", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2 * 7 * 1440)
    ap.add_argument("--symbol", default=os.environ.get("SYMBOL", "BTCUSDT"))
    ap.add_argument("--check-causal", action="store_true", default=True)
    ap.add_argument("--out", default=os.environ.get("STATPROF_OUT"))
    a = ap.parse_args(list(sys.argv[1:]) + os.environ.get("STATPROF_ARGS", "").split())

    df = load_data(a.synthetic, a.rows)
    n = df.height
    log(f"[data] symbol={a.symbol} rows={n:,} cols={len(df.columns)}")
    spec_list = list(REGISTRY)
    names = [s.name for s in spec_list]
    log(f"[feat] {len(names)} 个候选（含 {sum(1 for s in spec_list if not s.expect_pass)} 个负控）")

    F = np.column_stack([s.fn(df) for s in spec_list]).astype(np.float32)
    log(f"[matrix] {F.shape}")

    warm = min(a.warmup, n // 3) if a.synthetic else a.warmup
    # 个别标的可能缺 spot_close 等列 → 该特征全 NaN，不能让它把评估点清空
    usable = np.where(np.isfinite(F).mean(axis=0) > 0.5)[0]
    log(f"[matrix] 有效列 {len(usable)}/{F.shape[1]}"
        + (f"  全空列: {[names[j] for j in range(F.shape[1]) if j not in set(usable.tolist())]}"
           if len(usable) < F.shape[1] else ""))
    common = np.isfinite(F[:, usable]).all(axis=1)
    cand = np.where(common)[0]
    cand = cand[(cand >= warm) & (cand < n - max(HORIZONS) - 1)]
    log(f"[eval] 候选点={len(cand):,}  warmup={warm:,}  公共有效={common.mean():.3f}")

    c = df["close"].to_numpy().astype(np.float64)
    base = baseline_feats(df)
    idx_desc = cand[::a.step]
    descs = [describe(F[idx_desc, j]) for j in range(len(names))]
    corr_base = [max_base_corr(F[:, j], base, idx_desc) for j in range(len(names))]
    causal = [causality_ok(df, s.fn) if a.check_causal else True for s in spec_list]

    rows = []
    for h in HORIZONS:
        # 关键：长 horizon 标签重叠会把有效样本数夸大几个数量级，产生伪 IC。
        # 每个 horizon 用 step>=h 的非重叠抽样，n_eval 即独立样本数。
        step_h = max(a.step, h)
        idx = cand[::step_h]
        Fs = F[idx]
        R = rank_matrix(Fs)
        R -= np.nanmean(R, axis=0)
        R /= (np.nanstd(R, axis=0) + 1e-9)
        segs = np.array_split(np.arange(len(idx)), N_SEG)
        log(f"[h={h:>5}m] 非重叠评估点={len(idx):,} (step={step_h})")
        ret = np.full(n, np.nan)
        ret[:-h] = c[h:] / c[:-h] - 1
        rv = ret[idx]
        rr = rankdata(rv)
        rr = (rr - rr.mean()) / (rr.std() + 1e-9)
        ic_all = np.nanmean(R * rr[:, None], axis=0)
        # CVD 水平值 = 时间趋势累积量，长 horizon 与未来收益天然共线。
        # 用它当基线，记录扣除趋势后的超额 IC，避免把伪相关写进知识库。
        ctrl_j = names.index("CTRL_cvd_level") if "CTRL_cvd_level" in names else -1
        ctrl_ic = float(ic_all[ctrl_j]) if ctrl_j >= 0 else 0.0
        ic_seg = []
        for sg in segs:
            rs = rankdata(rv[sg])
            rs = (rs - rs.mean()) / (rs.std() + 1e-9)
            ic_seg.append(np.nanmean(R[sg] * rs[:, None], axis=0))
        yv = (rv > 0).astype(np.float64)
        for j, name in enumerate(names):
            x = Fs[:, j]
            q = np.nanquantile(x, np.array([0.1, 0.9]))
            lo, hi = yv[x <= q[0]], yv[x >= q[1]]
            gap = float(hi.mean() - lo.mean()) if len(lo) and len(hi) else 0.0
            ics = [float(v[j]) for v in ic_seg]
            same = all(v > 0 for v in ics) or all(v < 0 for v in ics)
            rows.append(dict(symbol=a.symbol, name=name, family=spec_list[j].family,
                             horizon=h, scenario=scenario(h),
                             ic_all=float(ic_all[j]), ic_s1=ics[0], ic_s2=ics[1], ic_s3=ics[2],
                             sign_consistent=bool(same), gap_pp=gap * 100,
                             n_eval=len(idx), ctrl_ic=ctrl_ic,
                             ic_excess=float(ic_all[j]) - ctrl_ic,
                             skew=descs[j][0], kurt=descs[j][1], ac1=descs[j][2],
                             mean=descs[j][3], median=descs[j][4],
                             p10=descs[j][5], p90=descs[j][6],
                             corr_base=corr_base[j][0], corr_with=corr_base[j][1],
                             causal=bool(causal[j]), is_control=not spec_list[j].expect_pass,
                             hypothesis=spec_list[j].hypothesis))

    out = pl.DataFrame(rows)
    log("")
    log("=== 每个特征 |IC| 最大的 horizon（画像摘要，非判定） ===")
    log(f"{'name':<26}{'family':<16}{'h(min)':>8}{'IC':>9}{'n':>8}{'3段同号':>8}"
        f"{'档差pp':>8}{'r旧':>7}{'因果':>5}  场景")
    for name in names:
        sub = out.filter((pl.col("name") == name) & pl.col("ic_all").is_finite())
        if sub.height == 0:
            log(f"{name:<26}（无足够独立样本，跳过）")
            continue
        r = sub.sort(pl.col("ic_all").abs(), descending=True).row(0, named=True)
        flag = "*" if r["n_eval"] < 200 else " "
        log(f"{name:<26}{r['family']:<16}{r['horizon']:>8}{r['ic_all']:>9.4f}"
            f"{str(r['n_eval']) + flag:>8}{'Y' if r['sign_consistent'] else 'N':>8}{r['gap_pp']:>8.2f}"
            f"{r['corr_base']:>7.2f}{'Y' if r['causal'] else 'N':>5}  {r['scenario']}")
    log("* = 独立样本 <200，IC 与档差仅定性参考")

    log("")
    log("=== 趋势基线：CTRL_cvd_level 各 horizon 的 IC（长周期伪相关参照） ===")
    for h in HORIZONS:
        r = out.filter((pl.col("name") == "CTRL_cvd_level")
                       & (pl.col("horizon") == h)).row(0, named=True)
        log(f"  h={h:>5}m  ctrl_ic={r['ic_all']:+.4f}  n={r['n_eval']}")

    log("")
    log("=== 用户举例现象的描述统计（mean / median / p10 / p90 / 偏度 / 一阶自相关） ===")
    for nm in ("stoch_below50_frac7d", "down_frac7d", "cvd_slope_corr7d",
               "cvdu_fit_r2_1d", "cvd_down_run", "cvdu_slope_z7d",
               "cvd_slope_spread1d", "vol_ratio_spot_um"):
        r = out.filter(pl.col("name") == nm).row(0, named=True)
        log(f"  {nm:<26} mean={r['mean']:>9.3f} med={r['median']:>9.3f}"
            f" p10={r['p10']:>9.3f} p90={r['p90']:>9.3f}"
            f" skew={r['skew']:>7.2f} ac1={r['ac1']:>6.2f}")

    log("")
    log("=== 扣除趋势基线后 |超额IC| 前 10（长 horizon 注意 n） ===")
    top = (out.filter((~pl.col("is_control")) & pl.col("ic_excess").is_finite())
           .sort(pl.col("ic_excess").abs(), descending=True).head(10))
    for r in top.iter_rows(named=True):
        log(f"  {r['name']:<26} h={r['horizon']:>5}m excess={r['ic_excess']:+.4f}"
            f" raw={r['ic_all']:+.4f} n={r['n_eval']} 同号={r['sign_consistent']}")

    log("")
    log("=== 各周期场景中 |IC| 排名前 8（负控除外） ===")
    for sc in ("event_3_60m", "intraday_4h_1d", "swing_1w"):
        top = (out.filter((pl.col("scenario") == sc) & (~pl.col("is_control"))
                          & pl.col("ic_all").is_finite())
               .sort(pl.col("ic_all").abs(), descending=True).head(8))
        log(f"[{sc}] " + ", ".join(
            f"{r['name']}@{r['horizon']}m:{r['ic_all']:+.3f}"
            for r in top.iter_rows(named=True)))

    log("")
    log("=== 负控自检（应无显著、跨段不同号） ===")
    ctl = (out.filter(pl.col("is_control") & pl.col("ic_all").is_finite())
           .sort(pl.col("ic_all").abs(), descending=True).head(4))
    for r in ctl.iter_rows(named=True):
        log(f"  {r['name']}@{r['horizon']}m IC={r['ic_all']:+.4f} 同号={r['sign_consistent']}")

    if a.out:
        out.write_csv(a.out)
        log(f"[save] {a.out} rows={out.height}")
    log("DONE")


if __name__ == "__main__":
    main()
