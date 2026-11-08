#!/usr/bin/env python3
"""h=3 专用宽表尝试：K 线特征 + 订单簿特征 + 跨域（成交↔挂单）特征。

为什么单独一个文件
------------------
`two_legs.py` 是「两腿分开训练 + 预测层融合」的基线（h=3 订单簿 65.49%）。这里的宽表
是**另一条技术路线**，只在 h=3 上试（其它 horizon 订单簿腿已证实衰减到 53% 以下）。
放独立文件 + 独立 kernel，是为了任何一次失败尝试都**不影响**基线，随时可回退。

实验记录：根目录 `h3宽表实验记录.md`，每次跑完把 `EXP=` 那行抄进去（只追加不改历史）。

跨域特征口径
------------
直接照 `bn_hisdata/features/ob_kline_cross.py` 复算，键都是 `close_time`：
    atk_f_imb / atk_f_def_gap / atk_f_absorb_buy / atk_f_absorb_sell   （合约）
    atk_s_*                                                            （现货）
        主动成交（攻）× 当前挂单（守）：flow 与 stock 的关系
    ob_amp_obi_015/035/080
        在「k × 近 5 分钟振幅（30m 均值）」距离处的买卖不平衡，k = 0.15/0.35/0.8
        （对应 3/5 分钟涨跌只看近端盘口）；深度按 vol8 分箱，插值前用
        `ob_f_sigma_bps` 把 bps 阈值换成 σ 倍数。

env
---
    FEATS=k+ob+cross   特征集：k / ob / k+ob / k+ob+cross（默认最后一个）
    HORIZON=3         固定 3 分钟（本文件只服务这个 horizon）
    TARGET_TPD=15     评估口径：日均 15 单
    EXP=W1            实验编号，只用于日志/记录
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

try:
    from morph import ob_kline as ok
    from morph.two_legs import TARGET_TPD, add_target, fit, matrix, report
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from morph import ob_kline as ok
    from morph.two_legs import TARGET_TPD, add_target, fit, matrix, report

FEATS = os.environ.get("FEATS", "k+ob+cross").lower()
HORIZON = int(os.environ.get("HORIZON", "3"))
EXP = os.environ.get("EXP", "W?")
TRAIN_FRAC, VAL_FRAC = 0.64, 0.16
T = time.time()

DEPTH_BINS = (("v0", 0.0, 1.0), ("v1", 1.0, 2.0), ("v2", 2.0, 4.0), ("v3", 4.0, 8.0))
DEPTH_TAIL = ("v4", 8.0)
BAND_BPS = 30.0
AMP_KS = (0.15, 0.35, 0.8)
CROSS_PREFIX = ("atk_", "ob_amp_obi_")


def log(msg: str) -> None:
    print(f"[{time.time() - T:7.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------- 跨域特征
def _amplitude(kw: pl.DataFrame) -> pl.DataFrame:
    """近 5 分钟振幅（high-low）/close，再取 30 分钟滚动均值（bps）。"""
    ts = pl.from_epoch(pl.col("close_time"), time_unit="ms")
    rng = (kw.with_columns(ts.alias("_ts"))
             .rolling(index_column="_ts", period="5m")
             .agg((pl.col("high").max() - pl.col("low").min()).alias("_amp5"))
             .drop("_ts"))
    kw = pl.concat([kw, rng], how="horizontal").with_columns(
        (pl.col("_amp5") / pl.col("close") * 1e4).alias("amp_5m_bps"))
    mean = (kw.with_columns(ts.alias("_ts"))
              .rolling(index_column="_ts", period="30m")
              .agg(pl.col("amp_5m_bps").mean().alias("amp_mean_30m_bps"))
              .drop("_ts"))
    return pl.concat([kw, mean], how="horizontal").drop("_amp5")


def _cum(x: pl.Expr, side: str) -> pl.Expr:
    """距离 x(bps) 内的累计挂单量：先把 x 换算成 σ 倍数，再在 v0..v4 上线性插值。"""
    xs = x / pl.col("ob_f_sigma_bps")
    tail, tail_lo = DEPTH_TAIL
    acc = pl.when(xs >= tail_lo).then(pl.col(f"ob_f_q_{side}_{tail}")).otherwise(0.0)
    for bn, lo, hi in DEPTH_BINS:
        q = pl.col(f"ob_f_q_{side}_{bn}")
        acc = (acc
               + pl.when(xs >= hi).then(q).otherwise(0.0)
               + pl.when((xs > lo) & (xs < hi)).then(q * (xs - lo) / (hi - lo))
               .otherwise(0.0))
    return pl.when(pl.col("ob_f_sigma_bps").is_null()).then(None).otherwise(acc)


def add_cross(df: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    """成交 ↔ 挂单的跨域列（口径与 bn_hisdata/features/ob_kline_cross.py 一致）。"""
    cols: list[str] = []
    amp = pl.col("amp_mean_30m_bps")
    for k in AMP_KS:
        tag = f"{round(k * 100):03d}"
        thr = (amp * k).clip(upper_bound=BAND_BPS)
        cb, ca = _cum(thr, "bid"), _cum(thr, "ask")
        df = df.with_columns(
            pl.when((cb + ca) > 0).then((cb - ca) / (cb + ca)).otherwise(None)
            .cast(pl.Float32).alias(f"ob_amp_obi_{tag}"))
        cols.append(f"ob_amp_obi_{tag}")

    if {"um_buy_vol", "um_sell_vol"}.issubset(df.columns):
        b, s = pl.col("um_buy_vol"), pl.col("um_sell_vol")
        qb, qa = pl.col("ob_f_q_bid"), pl.col("ob_f_q_ask")
        df = df.with_columns([((b - s) / (b + s)).alias("atk_f_imb"),
                              ((b - s) / (b + s) - (qb - qa) / (qb + qa)).alias("atk_f_def_gap"),
                              (b / qa).alias("atk_f_absorb_buy"),
                              (s / qb).alias("atk_f_absorb_sell")])
        cols += ["atk_f_imb", "atk_f_def_gap", "atk_f_absorb_buy", "atk_f_absorb_sell"]
    if {"spot_buy_vol", "spot_sell_vol", "ob_s_q_bid", "ob_s_q_ask"}.issubset(df.columns):
        b, s = pl.col("spot_buy_vol"), pl.col("spot_sell_vol")
        qb, qa = pl.col("ob_s_q_bid"), pl.col("ob_s_q_ask")
        df = df.with_columns([((b - s) / (b + s)).alias("atk_s_imb"),
                              ((b - s) / (b + s) - (qb - qa) / (qb + qa)).alias("atk_s_def_gap"),
                              (b / qa).alias("atk_s_absorb_buy"),
                              (s / qb).alias("atk_s_absorb_sell")])
        cols += ["atk_s_imb", "atk_s_def_gap", "atk_s_absorb_buy", "atk_s_absorb_sell"]

    # 除零会造 inf：不参与统计也不报错，只静默污染模型
    num = [c for c in df.columns if df[c].dtype in (pl.Float32, pl.Float64)]
    df = df.with_columns([pl.when(pl.col(c).is_finite()).then(pl.col(c))
                          .otherwise(None).alias(c) for c in num])
    return df, cols


# ---------------------------------------------------------------- 主流程
def main() -> None:
    ob = ok.load_ob()
    obcols = [c for c in ob.columns if c != "close_time"]
    lo, hi = int(ob["close_time"].min()), int(ob["close_time"].max())

    kf, kcols = ok.load_kline_features(keep_raw=True)
    # 振幅只需订单簿窗口（+ 35 分钟预热）：在 350 万行上做 5m/30m 滚动纯属浪费
    kw = kf.filter((pl.col("close_time") >= lo - 35 * 60_000) & (pl.col("close_time") <= hi))
    amp = _amplitude(kw).select("close_time", "amp_mean_30m_bps")
    log(f"振幅: {amp.height:,} 行（{ok._iso(lo)} 前 35min 起）")

    raw_cols = [c for c in ("um_buy_vol", "um_sell_vol", "spot_buy_vol", "spot_sell_vol")
                if c in kf.columns]
    wide = (ob.join(kf.select("close_time", "close", *kcols, *raw_cols),
                    on="close_time", how="left")
              .join(amp, on="close_time", how="left").sort("close_time"))

    use_ob = "ob" in FEATS.split("+")
    use_cross = "cross" in FEATS.split("+")
    cols = list(kcols) + (obcols if use_ob else [])
    cross_cols: list[str] = []
    if use_cross:
        wide, cross_cols = add_cross(wide)
        cols += cross_cols
    log(f"[{EXP}] 特征集 {FEATS}: {len(cols)} 列（K线 {len(kcols)} / "
        f"订单簿 {len(obcols) if use_ob else 0} / 跨域 {len(cross_cols)}）")

    wide = add_target(wide, HORIZON)
    wide = wide.filter(pl.col("target").is_not_null() & pl.col("close").is_not_null())
    n = wide.height
    tr1, va1 = int(n * TRAIN_FRAC), int(n * (TRAIN_FRAC + VAL_FRAC))
    log(f"[{EXP}] 可用 {n:,} 行 | train[0:{tr1}] val[{tr1}:{va1}] test[{va1}:{n}]")

    X = matrix(wide, cols)
    y = wide["target"].to_numpy().astype(int)
    days = (wide["close_time"].to_numpy() // 86_400_000)

    model = fit(X[:tr1], y[:tr1], cols, X[tr1:va1], y[tr1:va1])
    p_va = model.predict(X[tr1:va1], num_iteration=model.best_iteration)
    p_te = model.predict(X[va1:], num_iteration=model.best_iteration)
    log(f"[{EXP}] best_iter={model.best_iteration}")
    w_va = report(f"{EXP} val", p_va - 0.5, y[tr1:va1], days[tr1:va1])
    w_te = report(f"{EXP} test", p_te - 0.5, y[va1:], days[va1:])

    imp = np.array(model.feature_importance("gain"))
    order = np.argsort(imp)[::-1]
    log(f"[{EXP}] Top15: " + ", ".join(cols[i] for i in order[:15]))
    for tag, mask in (("K线", np.array([c in kcols for c in cols])),
                      ("订单簿", np.array([c in obcols for c in cols])),
                      ("跨域", np.array([c in cross_cols for c in cols]))):
        if mask.any():
            log(f"[{EXP}] 组 {tag}: {int(mask.sum())} 列 gain 占比 "
                f"{imp[mask].sum() / imp.sum() * 100:.1f}%")
    log(f"[{EXP}] SUMMARY feats={FEATS} cols={len(cols)} rows={n} "
        f"val={w_va * 100:.2f}% test={w_te * 100:.2f}% (@日均{TARGET_TPD}单)")
    log("DONE")


if __name__ == "__main__":
    main()
