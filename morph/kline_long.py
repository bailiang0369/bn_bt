#!/usr/bin/env python3
"""L1：长周期（15/30/60 分钟）K 线特征 + walk-forward 实盘口径评估。

为什么不用盘口：h=15 上 184 列盘口 AUC 只有 0.5165、@日均15单 50.9%（W2），
盘口在长周期没有信息。长周期的信号来源应是**趋势持续性 / 波动结构 / 量价 / 季节性**。

特征集（约 50 列，全部只用 1m K 线，无前视）
--------------------------------------------
1. 多尺度动量：log 收益 5/15/30/60/120/240/480/1440 分钟（bps）
2. 均线结构：close/SMA - 1、SMA 自身斜率（15/60/240/1440）
3. 波动结构：各尺度 RV（1m 收益的滚动标准差）+ RV 期限结构比值
4. 量价：成交量 Z-score、taker 买入占比及其多尺度均值、量价同向度
5. 位置/突破：过去 n 分钟的 (close-min)/(max-min)
6. 效率比：|净变动| / 路径长度（Kaufman ER），趋势 vs 震荡
7. 季节性：小时/星期的 sin-cos、距 8h 资金费结算的分钟数
8. 跨市场（若当侧 K 线含 spot 价格）：perp-spot 基差及其变化

评估口径与 W1 完全一致（逐月 walk-forward、早停用窗末 10%、4 年滚动窗口），
主判据仍是 `@日均15单`（15×天数），并同报 `每天top15` 与 AUC。

env: HORIZON / SEEDS / ROUNDS / WIN_DAYS / START_MONTH / END_MONTH / SYMBOL / KLINE_FROM
"""
from __future__ import annotations

import datetime as dt
import os
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

try:
    from morph import h3_gate_eval as ge
    from morph import ob_kline as ok
    from morph.two_legs import TARGET_TPD, add_target, fit, matrix
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from morph import h3_gate_eval as ge
    from morph import ob_kline as ok
    from morph.two_legs import TARGET_TPD, add_target, fit, matrix

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

HORIZON = int(os.environ.get("HORIZON", "15"))
SEEDS = [int(s) for s in os.environ.get("SEEDS", "42").split(",") if s.strip()]
WIN_DAYS = int(os.environ.get("WIN_DAYS", "1460"))
ES_FRAC = float(os.environ.get("ES_FRAC", "0.1"))
START_MONTH = os.environ.get("START_MONTH", "2026-01")
END_MONTH = os.environ.get("END_MONTH", "2026-10")
SYMBOL = os.environ.get("SYMBOL", "BTCUSDT")
KLINE_FROM = int(os.environ.get("KLINE_FROM", "2020"))
OUT = os.environ.get("OUT", "/kaggle/working/kline_long.csv")
T = time.time()
MI = 60_000


def log(msg: str) -> None:
    print(f"[{time.time() - T:7.1f}s] {msg}", flush=True)


def pick(cols, cands):
    for c in cands:
        if c in cols:
            return c
    return None


def load_raw() -> pl.DataFrame:
    a = ok._assets(ok.KLINE_TAG)
    names = [f"{SYMBOL}_1m_klines_{y}.parquet" for y in range(KLINE_FROM, 2027)]
    have = [n for n in names if n in a]
    if not have:
        raise SystemExit(f"{ok.KLINE_TAG} 里没有 {SYMBOL} K 线")
    df = pl.concat([pl.read_parquet(ok.fetch(ok.KLINE_TAG, n)) for n in have],
                   how="vertical_relaxed")
    df = df.unique(subset=["close_time"], keep="last").sort("close_time")
    log(f"K 线原始 {df.height:,} 行 | 列={df.columns}")
    return df


def build(df: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    cols = df.columns
    c = pl.col("close").cast(pl.Float64)
    r1 = (c / c.shift(1)).log() * 1e4
    e: list[pl.Expr] = []

    for k in (5, 15, 30, 60, 120, 240, 480, 1440):
        e.append(((c / c.shift(k)).log() * 1e4).alias(f"lg_ret{k}"))

    for n in (15, 60, 240, 1440):
        ma = c.rolling_mean(n, min_samples=max(2, n // 2))
        e.append(((c / ma - 1) * 1e4).alias(f"lg_madev{n}"))
        e.append(((ma / ma.shift(n) - 1) * 1e4).alias(f"lg_maslope{n}"))
        e.append(r1.rolling_std(n, min_samples=max(3, n // 2)).alias(f"lg_rv{n}"))

    vol_name = pick(cols, ("volume", "um_volume", "spot_volume"))
    tb_name = pick(cols, ("taker_buy_volume", "um_buy_vol", "um_taker_buy_vol"))
    if vol_name:
        v = pl.col(vol_name).cast(pl.Float64)
        vm = v.rolling_mean(1440, min_samples=720)
        vs = v.rolling_std(1440, min_samples=720)
        vz = (v - vm) / (vs + 1e-9)
        e.append(vz.alias("lg_volz"))
        e.append((v / (vm + 1e-9)).alias("lg_volratio"))
        e.append((r1.sign() * vz).rolling_mean(60, min_samples=30).alias("lg_volprice_60"))
        if tb_name:
            tr = pl.col(tb_name).cast(pl.Float64) / (v + 1e-9)
            e.append(tr.alias("lg_taker"))
            for n in (15, 60, 240):
                e.append(tr.rolling_mean(n, min_samples=max(2, n // 2)).alias(f"lg_taker{n}"))

    for n in (60, 240, 1440):
        lo = c.rolling_min(n, min_samples=max(2, n // 2))
        hi = c.rolling_max(n, min_samples=max(2, n // 2))
        e.append(((c - lo) / (hi - lo + 1e-9)).alias(f"lg_pos{n}"))

    abs_r1 = r1.abs()
    for n in (60, 240):
        e.append((r1.rolling_sum(n).abs() / (abs_r1.rolling_sum(n) + 1e-9)).alias(f"lg_er{n}"))

    ct = pl.col("close_time")
    hour = ((ct // 3_600_000) % 24).cast(pl.Float64)
    dow = ((ct // 86_400_000) % 7).cast(pl.Float64)
    e += [(hour * (2 * np.pi / 24)).sin().alias("lg_hour_sin"),
          (hour * (2 * np.pi / 24)).cos().alias("lg_hour_cos"),
          (dow * (2 * np.pi / 7)).sin().alias("lg_dow_sin"),
          (dow * (2 * np.pi / 7)).cos().alias("lg_dow_cos"),
          (((ct // MI) % 1440) % 480).cast(pl.Float64).alias("lg_funding_phase")]

    if "spot_close" in cols:
        basis = (c / pl.col("spot_close").cast(pl.Float64) - 1) * 1e4
        e += [basis.alias("lg_basis"),
              (basis - basis.shift(60)).alias("lg_basis_chg60"),
              basis.rolling_mean(60, min_samples=30).alias("lg_basis_ma60")]

    out = df.with_columns(e)
    e2: list[pl.Expr] = []
    for n in (15, 60, 240, 1440):
        if f"lg_rv{n}" in out.columns:
            e2.append((pl.col(f"lg_ret{n}") / (pl.col(f"lg_rv{n}") + 1e-9)).alias(f"lg_zret{n}"))
    for a_, b_ in ((15, 60), (60, 240), (240, 1440)):
        e2.append((pl.col(f"lg_rv{a_}") / (pl.col(f"lg_rv{b_}") + 1e-9)).alias(f"lg_rvr{a_}_{b_}"))
    out = out.with_columns(e2)
    feats = [x for x in out.columns if x.startswith("lg_")]
    return out, feats


def acc(s, y):
    return float(((s > 0).astype(int) == y).mean())


def topk(s, y, k):
    k = min(max(1, k), len(y))
    idx = np.argsort(-np.abs(s))[:k]
    return acc(s[idx], y[idx])


def daily_stats(s, y, days, n):
    w = [topk(s[days == d], y[days == d], n) for d in np.unique(days)]
    a = np.array(w)
    return float(a.mean()), float(a.std())


def main() -> None:
    raw = load_raw()
    df, feats = build(raw)
    df = add_target(df, HORIZON).filter(
        pl.col("target").is_not_null() & pl.col("close").is_not_null()).sort("close_time")
    df = df.with_columns(
        pl.col("close_time").cast(pl.Datetime("ms")).dt.strftime("%Y-%m").alias("ym"))
    log(f"特征 {len(feats)} 列 | 可用 {df.height:,} 行 | {df['ym'].min()} ~ {df['ym'].max()}")

    months = [m for m in sorted(df["ym"].unique().to_list()) if START_MONTH <= m <= END_MONTH]
    log(f"walk-forward：{len(months)} 窗口（{months[0]}~{months[-1]}）| H={HORIZON}"
        f" | 滚动窗 {WIN_DAYS} 天 | seeds={SEEDS}")

    rows, all_s, all_y, all_d = [], [], [], []
    for m in months:
        m_start = int(dt.datetime.strptime(m + "-01", "%Y-%m-%d")
                      .replace(tzinfo=dt.timezone.utc).timestamp() * 1000)
        lo = m_start - WIN_DAYS * 86_400_000
        tr_all = df.filter((pl.col("close_time") < m_start) & (pl.col("close_time") >= lo))
        te = df.filter(pl.col("ym") == m)
        if tr_all.height < 50_000 or te.height < 1000:
            log(f"[{m}] 样本不足（{tr_all.height:,}/{te.height:,}），跳过")
            continue
        cut = int(tr_all.height * (1 - ES_FRAC))
        tr, es = tr_all[:cut], tr_all[cut:]
        Xtr, ytr = matrix(tr, feats), tr["target"].to_numpy().astype(int)
        Xes, yes = matrix(es, feats), es["target"].to_numpy().astype(int)
        ps = []
        for seed in SEEDS:
            mdl = fit(Xtr, ytr, feats, Xes, yes, seed=seed)
            ps.append(mdl.predict(matrix(te, feats)))
        p = np.mean(ps, axis=0)
        s, y = p - .5, te["target"].to_numpy().astype(np.int8)
        days = ge.day_id(te["close_time"].to_numpy())
        nd = len(np.unique(days))
        a15 = topk(s, y, TARGET_TPD * nd)
        dm, ds = daily_stats(s, y, days, TARGET_TPD)
        rows.append(dict(ym=m, n=te.height, days=nd, auc=float(roc_auc_score(y, p)),
                         a15=a15, daily_mean=dm, daily_std=ds, train=tr_all.height))
        all_s.append(s)
        all_y.append(y)
        all_d.append(days)
        log(f"[{m}] train={tr_all.height:,} test={te.height:,}({nd}天) AUC={rows[-1]['auc']:.4f}"
            f" @日均{TARGET_TPD}单={a15 * 100:.2f}% 每天top15={dm * 100:.2f}%(±{ds * 100:.1f}pp)")

    if not rows:
        raise SystemExit("无窗口完成")

    s, y, d = (np.concatenate(x) for x in (all_s, all_y, all_d))
    nd = len(np.unique(d))
    oos = topk(s, y, TARGET_TPD * nd)
    dm, ds = daily_stats(s, y, d, TARGET_TPD)
    a = np.array([r["a15"] for r in rows])
    seq = ", ".join(f"{r['ym']}:{r['a15'] * 100:.1f}%" for r in rows)
    log(f"===== 全 OOS：{len(y):,} 行 / {nd} 天 | AUC={roc_auc_score(y, s + .5):.4f}"
        f" | @日均{TARGET_TPD}单={oos * 100:.2f}% | 每天top15={dm * 100:.2f}%(±{ds * 100:.1f}pp)")
    log(f"===== 月度：{seq}")
    log(f"===== 分布：mean={a.mean() * 100:.2f}% std={a.std() * 100:.2f}pp"
        f" min={a.min() * 100:.2f}% max={a.max() * 100:.2f}%"
        f" ≥65%={int((a >= .65).sum())}/{len(a)}")
    if OUT:
        pl.DataFrame(rows).write_csv(OUT)
        log(f"明细已落盘: {OUT}")
    log("DONE")


if __name__ == "__main__":
    main()
