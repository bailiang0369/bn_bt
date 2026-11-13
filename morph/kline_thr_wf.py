#!/usr/bin/env python3
"""长周期 K 线 walk-forward：固定阈值触发口径（不控单量）vs 强制每天N单。

背景（2026-11-13）：离线发现「强制@40单/天」会把低置信单硬塞进来、压低胜率；
改用现役 h=3 同款「val 定 |score| 分位阈值 → test 逐 bar 触发、不控单量」后，
ETH 2024-2025 上 30m 可到 62%、60m 62.6%。本脚本在 **BTC 2020+ 全量** 上做
逐月 walk-forward 复核：每月 train(730d) → val(30d，定阈值) → test(该月)，
report 每月各 val 分位(top1/2/3/5/10%) 的 test 触发率/日均单/胜率/EV。

基础 K 尺度与 horizon 匹配（用户 2026-11-13）：预测 30/60m 用 1m 和 5m 两种基础，
特征全部在该尺度上定义；不用会重绘的 ZigZag。

env: SYMBOL / START_MONTH / END_MONTH / WIN_DAYS / VAL_DAYS / SEEDS / KLINE_FROM
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
    from morph import ob_kline as ok
except ModuleNotFoundError:  # 本地直跑
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from morph import ob_kline as ok

try:
    import lightgbm as lgb
except Exception as e:  # pragma: no cover
    raise SystemExit(f"lightgbm 缺失: {e}")

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

SYMBOL = os.environ.get("SYMBOL", "BTCUSDT")
START_MONTH = os.environ.get("START_MONTH", "2026-01")
END_MONTH = os.environ.get("END_MONTH", "2026-10")
WIN_DAYS = int(os.environ.get("WIN_DAYS", "730"))
VAL_DAYS = int(os.environ.get("VAL_DAYS", "30"))
SEEDS = [int(s) for s in os.environ.get("SEEDS", "42,43,44").split(",") if s.strip()]
KLINE_FROM = int(os.environ.get("KLINE_FROM", "2020"))
PAYOUT = 0.8
TPDS = (1, 2, 3, 5, 10)
# (基础K分钟, horizon分钟)
CONFIGS = [(1, 30), (5, 30), (1, 60), (5, 60)]
MI = 60_000
T = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T:7.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------- 数据
def load_raw() -> pl.DataFrame:
    a = ok._assets(ok.KLINE_TAG)
    names = [f"{SYMBOL}_1m_klines_{y}.parquet" for y in range(KLINE_FROM, 2027)]
    have = [n for n in names if n in a]
    if not have:
        raise SystemExit(f"{ok.KLINE_TAG} 里没有 {SYMBOL} K 线")
    df = pl.concat([pl.read_parquet(ok.fetch(ok.KLINE_TAG, n)) for n in have],
                   how="vertical_relaxed")
    df = df.unique(subset=["close_time"], keep="last").sort("close_time")
    df = df.with_columns(
        (pl.col("um_buy_vol") + pl.col("um_sell_vol")).alias("volume"),
        pl.col("um_buy_vol").alias("taker_buy"))
    log(f"K 线 {df.height:,} 行 | {df.columns}")
    return df


def resample(df: pl.DataFrame, minutes: int) -> pl.DataFrame:
    ms = minutes * MI
    if minutes == 1:
        return df
    return (df.with_columns(_b=(pl.col("close_time") // ms))
            .group_by("_b")
            .agg(close_time=pl.col("close_time").last(),
                 open=pl.col("open").first(), high=pl.col("high").max(),
                 low=pl.col("low").min(), close=pl.col("close").last(),
                 volume=pl.col("volume").sum(), taker_buy=pl.col("taker_buy").sum(),
                 cnt=pl.len())
            .filter(pl.col("cnt") == minutes)
            .sort("close_time").drop("cnt"))


# ---------------------------------------------------------------- 特征
def base_feats(df: pl.DataFrame, minutes: int, tag: str) -> list[pl.Expr]:
    c = pl.col("close").cast(pl.Float64)
    r1 = (c / c.shift(1)).log() * 1e4
    e: list[pl.Expr] = []

    def nb(m: int) -> int:
        return max(1, int(round(m / minutes)))

    for m in (5, 15, 30, 60, 120, 240, 480, 1440):
        e.append(((c / c.shift(nb(m))).log() * 1e4).alias(f"{tag}_ret{m}"))
    for m in (15, 60, 240, 1440):
        n = nb(m)
        ma = c.rolling_mean(n, min_samples=max(2, n // 2))
        e.append(((c / ma - 1) * 1e4).alias(f"{tag}_madev{m}"))
        e.append(((ma / ma.shift(n) - 1) * 1e4).alias(f"{tag}_maslope{m}"))
        e.append(r1.rolling_std(n, min_samples=max(3, n // 2)).alias(f"{tag}_rv{m}"))
    v = pl.col("volume").cast(pl.Float64)
    vm = v.rolling_mean(nb(1440), min_samples=max(2, nb(720)))
    vs = v.rolling_std(nb(1440), min_samples=max(2, nb(720)))
    vz = (v - vm) / (vs + 1e-9)
    e.append(vz.alias(f"{tag}_volz"))
    e.append((v / (vm + 1e-9)).alias(f"{tag}_volratio"))
    tr = pl.col("taker_buy").cast(pl.Float64) / (v + 1e-9)
    e.append(tr.alias(f"{tag}_taker"))
    for m in (15, 60, 240):
        e.append(tr.rolling_mean(nb(m), min_samples=max(2, nb(m) // 2)).alias(f"{tag}_taker{m}"))
    for m in (60, 240, 1440):
        n = nb(m)
        cmin = c.rolling_min(n, min_samples=max(2, n // 2))
        cmax = c.rolling_max(n, min_samples=max(2, n // 2))
        e.append(((c - cmin) / (cmax - cmin + 1e-9)).alias(f"{tag}_pos{m}"))
    ar = r1.abs()
    for m in (60, 240):
        n = nb(m)
        e.append((r1.rolling_sum(n).abs() / (ar.rolling_sum(n) + 1e-9)).alias(f"{tag}_er{m}"))
    for a, b in ((15, 60), (60, 240), (240, 1440)):
        e.append((r1.rolling_std(nb(a), min_samples=max(3, nb(a) // 2))
                  / (r1.rolling_std(nb(b), min_samples=max(3, nb(b) // 2)) + 1e-9))
                 .alias(f"{tag}_rvr{a}_{b}"))
    ct = pl.col("close_time")
    hour = ((ct // 3_600_000) % 24).cast(pl.Float64)
    dow = ((ct // 86_400_000) % 7).cast(pl.Float64)
    e += [(hour * (2 * np.pi / 24)).sin().alias(f"{tag}_hsin"),
          (hour * (2 * np.pi / 24)).cos().alias(f"{tag}_hcos"),
          (dow * (2 * np.pi / 7)).sin().alias(f"{tag}_wsin"),
          (dow * (2 * np.pi / 7)).cos().alias(f"{tag}_wcos")]
    return e


def shape_feats(df: pl.DataFrame, minutes: int, tag: str) -> list[pl.Expr]:
    c = pl.col("close").cast(pl.Float64)
    o = pl.col("open").cast(pl.Float64)
    h = pl.col("high").cast(pl.Float64)
    lo = pl.col("low").cast(pl.Float64)
    r1 = (c / c.shift(1)).log() * 1e4
    rng = (h - lo).clip(1e-9, None)
    e: list[pl.Expr] = []

    def nb(m: int) -> int:
        return max(1, int(round(m / minutes)))

    e.append(((c - o) / rng).alias(f"{tag}_body"))
    e.append(((h - pl.max_horizontal(o, c)) / rng).alias(f"{tag}_upper"))
    e.append(((pl.min_horizontal(o, c) - lo) / rng).alias(f"{tag}_lower"))
    up = (r1 > 0).cast(pl.Float64)
    for m in (15, 30, 60):
        e.append(up.rolling_mean(nb(m), min_samples=max(2, nb(m) // 2)).alias(f"{tag}_upratio{m}"))
    ms = (5, 15, 30, 60, 120, 240)
    agree = (c / c.shift(nb(ms[0])) - 1).sign()
    for m in ms[1:]:
        agree = agree + (c / c.shift(nb(m)) - 1).sign()
    e.append((agree / len(ms)).alias(f"{tag}_trend_agree"))
    e.append(((r1.sign() * (c / c.shift(nb(240)) - 1).sign()) < 0)
             .cast(pl.Float64).alias(f"{tag}_counter"))
    for m in (60, 240, 1440):
        n = nb(m)
        cmin = c.rolling_min(n, min_samples=max(2, n // 2))
        cmax = c.rolling_max(n, min_samples=max(2, n // 2))
        e.append(((c - cmin) / (cmax - cmin + 1e-9)).alias(f"{tag}_rpos{m}"))
    for a, b in ((15, 60), (60, 240)):
        e.append((((c / c.shift(nb(a)) - 1) - (c / c.shift(nb(b)) - 1)) * 1e4)
                 .alias(f"{tag}_acc{a}_{b}"))
    return e


def build(df: pl.DataFrame, minutes: int, horizon: int, tag: str):
    h_bars = max(1, int(round(horizon / minutes)))
    d = df.with_columns(
        (pl.col("close").shift(-h_bars) > pl.col("close")).cast(pl.Int8).alias("target"))
    d = d.filter(pl.col("target").is_not_null())
    d = d.with_columns(base_feats(d, minutes, tag) + shape_feats(d, minutes, tag))
    feats = [x for x in d.columns if x.startswith(tag + "_")]
    d = d.with_columns(
        pl.col("close_time").cast(pl.Datetime("ms")).dt.strftime("%Y-%m").alias("ym"))
    return d.sort("close_time"), feats


# ---------------------------------------------------------------- walk-forward
def topk_acc(s, y, k):
    k = min(max(1, int(k)), len(y))
    idx = np.argsort(-np.abs(s))[:k]
    return float(((s[idx] > 0).astype(int) == y[idx]).mean())


def run_config(df1m: pl.DataFrame, minutes: int, horizon: int) -> dict:
    tag = f"m{minutes}h{horizon}"
    base = df1m if minutes == 1 else resample(df1m, minutes)
    d, feats = build(base, minutes, horizon, tag)
    log(f"[{tag}] base={minutes}m h={horizon}m | 可用 {d.height:,} 行 | 特征 {len(feats)} 列")

    months = [m for m in sorted(d["ym"].unique().to_list()) if START_MONTH <= m <= END_MONTH]
    rows = []
    for m in months:
        m_start = int(dt.datetime.strptime(m + "-01", "%Y-%m-%d")
                      .replace(tzinfo=dt.timezone.utc).timestamp() * 1000)
        tr_lo = m_start - WIN_DAYS * 86_400_000
        va_lo = m_start - VAL_DAYS * 86_400_000
        tr = d.filter((pl.col("close_time") >= tr_lo) & (pl.col("close_time") < va_lo))
        va = d.filter((pl.col("close_time") >= va_lo) & (pl.col("close_time") < m_start))
        te = d.filter(pl.col("ym") == m)
        if tr.height < 50_000 or va.height < 1000 or te.height < 500:
            continue
        cut = int(tr.height * 0.9)
        Xtr = tr[:cut].select(feats).to_numpy().astype(np.float32)
        ytr = tr["target"].to_numpy()[:cut].astype(int)
        Xes = tr[cut:].select(feats).to_numpy().astype(np.float32)
        yes = tr[cut:]["target"].to_numpy().astype(int)
        Xva = va.select(feats).to_numpy().astype(np.float32)
        Xte = te.select(feats).to_numpy().astype(np.float32)
        ps_v, ps_t = [], []
        for seed in SEEDS:
            mdl = lgb.LGBMClassifier(
                n_estimators=400, learning_rate=0.05, num_leaves=31, subsample=0.8,
                colsample_bytree=0.8, min_child_samples=100, reg_lambda=1.0,
                random_state=seed, n_jobs=-1, verbose=-1)
            mdl.fit(Xtr, ytr, eval_set=[(Xes, yes)],
                    callbacks=[lgb.early_stopping(50, verbose=False)])
            ps_v.append(mdl.predict_proba(Xva)[:, 1])
            ps_t.append(mdl.predict_proba(Xte)[:, 1])
        sv = np.mean(ps_v, axis=0) - .5
        st = np.mean(ps_t, axis=0) - .5
        yv = va["target"].to_numpy().astype(np.int8)
        yt = te["target"].to_numpy().astype(np.int8)
        av, at = np.abs(sv), np.abs(st)
        nd = len(np.unique(te["close_time"].to_numpy() // 86_400_000))
        auc_t = float(roc_auc_score(yt, st + .5))
        row = dict(ym=m, n=te.height, nd=nd, auc=auc_t)
        for q in TPDS:
            thr = float(np.quantile(av, 1 - q / 100))
            hit = at >= thr
            nh = int(hit.sum())
            if nh < 20:
                row[f"p{q}"] = (np.nan, np.nan, np.nan)
                continue
            acc = float(((st[hit] > 0).astype(int) == yt[hit]).mean())
            row[f"p{q}"] = (acc, nh / nd, (1 + PAYOUT) * acc - 1)
        rows.append(row)
        cells = "  ".join(
            f"p{q}:{row[f'p{q}'][0] * 100:5.2f}%/{row[f'p{q}'][1]:5.1f}"
            for q in TPDS if not np.isnan(row[f'p{q}'][0]))
        log(f"  [{m}] n={te.height:,} AUC={auc_t:.4f} | {cells}")

    # 汇总：各分位跨月的加权胜率（按 test 单量加权）+ 月度范围
    print(f"\n===== [{tag}] 汇总（{len(rows)} 个月）=====")
    print(f"{'分位':>6}{'总单量':>9}{'日均':>7}{'加权胜率':>10}{'EV':>8}{'月min':>8}{'月max':>8}{'月std':>8}")
    for q in TPDS:
        accs = [(r[f"p{q}"][0], r[f"p{q}"][1] * r["nd"]) for r in rows
                if not np.isnan(r[f"p{q}"][0])]
        if not accs:
            continue
        tot = sum(w for _, w in accs)
        wacc = sum(a * w for a, w in accs) / tot
        arr = np.array([a for a, _ in accs])
        nd_all = sum(r["nd"] for r in rows if not np.isnan(r[f"p{q}"][0]))
        print(f"{q:>5}%{int(tot):>9,}{tot / max(nd_all, 1):>7.1f}{wacc * 100:>9.2f}%"
              f"{(1 + PAYOUT) * wacc - 1:>+8.3f}{arr.min() * 100:>7.2f}%{arr.max() * 100:>7.2f}%"
              f"{arr.std() * 100:>7.2f}%")
    print()
    return rows


def main() -> None:
    raw = load_raw()
    log(f"配置：{CONFIGS} | 月份 {START_MONTH}~{END_MONTH} | seeds={SEEDS}")
    for minutes, horizon in CONFIGS:
        run_config(raw, minutes, horizon)
    log("DONE")


if __name__ == "__main__":
    main()
