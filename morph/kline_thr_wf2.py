#!/usr/bin/env python3
"""长周期 K 线 walk-forward v2：目标是**提升胜率**，不是提升单量。

v1（kline_thr_wf.py）已证实：纯方向模型 + |score| 分位阈值，10 个月上最好只有
m5h60 1% 的 61.37%（其余配置 55~57%，贴着 55.56% 的平衡线）。v2 在**模型不变**
的前提下，把「训练一次、评估多种过滤条件」做成网格实验，一次跑完回答：
哪些过滤条件真的能把胜率推上去。

新增 / 改进：
1. 现货特征（v1 的 resample 把 spot_* 列全丢了）：永续-现货基差 basis 及其 z、
   现货 taker 买占比、现货/永续 taker 背离、现货领先收益等；
2. 过滤网格：单边（只多/只空）、信号持续性（连续 2/3 根同向）、
   趋势一致（对齐 240m 动量）、逆势、波动区间（rv60 高/低）、趋势效率（er60 高）、
   以及上述组合；
3. 去重叠（cooldown = horizon）口径：重叠样本会让显著性虚高，单列一行；
4. 多周期共振：同基础尺度下 30/60/120/240m 分数同向才算信号；
5. 打印无条件基准上涨率 base_rate（v1 缺失，导致无法判断 EV 的绝对水平）。

env: SYMBOL / START_MONTH / END_MONTH / WIN_DAYS / VAL_DAYS / SEEDS / KLINE_FROM
"""
from __future__ import annotations

import datetime as dt
import gc
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
START_MONTH = os.environ.get("START_MONTH", "2024-01")
END_MONTH = os.environ.get("END_MONTH", "2026-10")
WIN_DAYS = int(os.environ.get("WIN_DAYS", "730"))
VAL_DAYS = int(os.environ.get("VAL_DAYS", "30"))
SEEDS = [int(s) for s in os.environ.get("SEEDS", "42,43,44").split(",") if s.strip()]
KLINE_FROM = int(os.environ.get("KLINE_FROM", "2020"))

PAYOUT = 0.8
BREAKEVEN = 1.0 / (1.0 + PAYOUT)      # 55.56%：低于此线 EV 必为负
Q_REF = (0.5, 1, 2, 3)               # base 变体的分位参考
Q_VAR = (1, 2)                       # 变体网格评估用的分位
MIN_N = 10                           # 单月单量低于此值不参与月度统计（q=1% 组合过滤后单量很小）
CONFIGS = [(5, 30), (5, 60), (5, 120), (5, 240)]   # 5m 基础；v1 的 1m 基础无用，砍掉
CONSENSUS = [((5, 30), (5, 60)), ((5, 60), (5, 120)), ((5, 60), (5, 240))]
CTX_KEYS = ("rv60", "ret240", "ret60", "volz", "er60")
MI = 60_000
T = time.time()

VARIANTS = [
    ("base", ()), ("long", ("L",)), ("short", ("S",)),
    ("p3", ("p3",)), ("trend", ("tr",)), ("rev", ("rv",)),
    ("rvhi", ("rvh",)), ("rvlo", ("rvl",)), ("erhi", ("erh",)),
    ("trend+p3", ("tr", "p3")), ("long+p3", ("L", "p3")), ("short+p3", ("S", "p3")),
    ("trend+rvhi", ("tr", "rvh")), ("p3+rvhi", ("p3", "rvh")),
]

SCORES: dict[str, dict] = {}    # tag -> {ym: dict(ct, st, y, thr)}


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
    """聚合到 minutes 周期；**保留 spot_* 列**（v1 在这里丢掉了现货信息）。"""
    ms = minutes * MI
    if minutes == 1:
        return df
    agg = [pl.col("close_time").last(), pl.col("open").first(), pl.col("high").max(),
           pl.col("low").min(), pl.col("close").last(),
           pl.col("volume").sum(), pl.col("taker_buy").sum(), pl.len().alias("cnt")]
    for c in ("spot_open", "spot_high", "spot_low", "spot_close",
              "spot_buy_vol", "spot_sell_vol"):
        if c in df.columns:
            agg.append(pl.col(c).first() if c == "spot_open" else
                       pl.col(c).last() if c == "spot_close" else
                       pl.col(c).max() if c == "spot_high" else
                       pl.col(c).min() if c == "spot_low" else
                       pl.col(c).sum())
    return (df.with_columns(_b=(pl.col("close_time") // ms))
            .group_by("_b").agg(agg).filter(pl.col("cnt") == minutes)
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
    e.append(((v - vm) / (vs + 1e-9)).alias(f"{tag}_volz"))
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


def spot_feats(df: pl.DataFrame, minutes: int, tag: str) -> list[pl.Expr]:
    """现货维度：基差 + 现货主动买卖 + 现货领先。v1 完全没有这些信息。"""
    if "spot_close" not in df.columns:
        return []
    c = pl.col("close").cast(pl.Float64)
    sc = pl.col("spot_close").cast(pl.Float64)
    pv = pl.col("volume").cast(pl.Float64)
    sbv = pl.col("spot_buy_vol").cast(pl.Float64)
    ssv = pl.col("spot_sell_vol").cast(pl.Float64)
    sv = sbv + ssv
    e: list[pl.Expr] = []

    def nb(m: int) -> int:
        return max(1, int(round(m / minutes)))

    b = (c - sc) / sc * 1e4
    bm = b.rolling_mean(nb(240), min_samples=max(2, nb(120)))
    bs = b.rolling_std(nb(240), min_samples=max(2, nb(120)))
    e.append(b.alias(f"{tag}_sp_basis"))
    e.append(((b - bm) / (bs + 1e-9)).alias(f"{tag}_sp_basisz"))
    e.append((b - b.shift(1)).alias(f"{tag}_sp_basisd"))
    e.append(b.rolling_std(nb(60), min_samples=max(2, nb(30))).alias(f"{tag}_sp_basisrv"))
    st_ = sbv / (sv + 1e-9)
    pt_ = pl.col("taker_buy").cast(pl.Float64) / (pv + 1e-9)
    e.append(st_.alias(f"{tag}_sp_staker"))
    e.append((pt_ - st_).alias(f"{tag}_sp_tdiv"))
    for m in (15, 60):
        e.append((pt_ - st_).rolling_mean(nb(m), min_samples=max(2, nb(m) // 2))
                 .alias(f"{tag}_sp_tdiv{m}"))
    e.append((sv / (pv + 1e-9)).alias(f"{tag}_sp_volr"))
    for m in (5, 15, 60):
        e.append(((sc / sc.shift(nb(m))).log() * 1e4).alias(f"{tag}_sp_sret{m}"))
    e.append((((sc / sc.shift(nb(15))).log() - (c / c.shift(nb(15))).log()) * 1e4)
             .alias(f"{tag}_sp_lead15"))
    e.append((((sc / sc.shift(nb(60))).log() - (c / c.shift(nb(60))).log()) * 1e4)
             .alias(f"{tag}_sp_lead60"))
    return e


def build(df: pl.DataFrame, minutes: int, horizon: int, tag: str):
    h_bars = max(1, int(round(horizon / minutes)))
    d = df.with_columns(
        (pl.col("close").shift(-h_bars) > pl.col("close")).cast(pl.Int8).alias("target"))
    d = d.filter(pl.col("target").is_not_null())
    d = d.with_columns(base_feats(d, minutes, tag) + shape_feats(d, minutes, tag)
                       + spot_feats(d, minutes, tag))
    feats = [x for x in d.columns if x.startswith(tag + "_")]
    d = d.with_columns(
        pl.col("close_time").cast(pl.Datetime("ms")).dt.strftime("%Y-%m").alias("ym"))
    return d.sort("close_time"), feats


# ---------------------------------------------------------------- 过滤网格
def _components(s: np.ndarray, ctx: dict, med: dict) -> dict:
    sg = np.sign(s)
    n = len(s)
    p3 = np.zeros(n, bool)
    if n > 2:
        p3[2:] = (sg[2:] == sg[1:-1]) & (sg[1:-1] == sg[:-2])
    tr = np.sign(ctx["ret240"])
    return {
        "L": sg > 0, "S": sg < 0, "p3": p3,
        "tr": sg == tr, "rv": sg == -tr,
        "rvh": ctx["rv60"] >= med["rv60"], "rvl": ctx["rv60"] < med["rv60"],
        "erh": ctx["er60"] >= med["er60"],
    }


def _pick(mask: np.ndarray, cooldown: int = 0) -> np.ndarray:
    idx = np.flatnonzero(mask)
    if cooldown <= 1:
        return idx
    keep, last = [], -(10 ** 9)
    for i in idx:
        if i - last >= cooldown:
            keep.append(i)
            last = i
    return np.asarray(keep, dtype=int)


def _win(s: np.ndarray, y: np.ndarray, idx: np.ndarray) -> float:
    return float(((s[idx] > 0) == (y[idx] == 1)).mean())


def _agg(rows: list[dict], nd_all: int) -> dict | None:
    rows = [r for r in rows if r and r["n"] >= MIN_N]
    if not rows:
        return None
    tot = sum(r["n"] for r in rows)
    w = sum(r["win"] * r["n"] for r in rows) / tot
    arr = np.array([r["win"] for r in rows])
    return dict(n=tot, per_day=tot / max(nd_all, 1), win=w,
                months_ok=int((arr > BREAKEVEN).sum()), nmon=len(arr),
                mn=float(arr.min()), sd=float(arr.std()))


def _fmt(a: dict | None) -> str:
    if a is None:
        return f"{'-':>10}{'-':>8}{'-':>7}{'-':>9}{'-':>8}{'-':>8}{'-':>8}{'-':>7}"
    return (f"{a['n']:>10,}{a['per_day']:>8.1f}{a['win'] * 100:>8.2f}%"
            f"{(1 + PAYOUT) * a['win'] - 1:>+8.3f}"
            f"{a['months_ok']:>5}/{a['nmon']:<2}{a['mn'] * 100:>7.2f}%{a['sd'] * 100:>6.2f}%")


HDR = (f"{'variant':>10}{'n':>8}{'/day':>7}{'win%':>9}{'EV':>8}"
       f"{'mon':>8}{'min%':>8}{'std%':>7}")


# ---------------------------------------------------------------- walk-forward
def train_eval(d, feats, months, tag, base_arrs):
    """对给定特征集跑完整 walk-forward，返回按月记录。"""
    out = []
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
        ct = te["close_time"].to_numpy()
        nd = len(np.unique(ct // 86_400_000))
        ctx, med = {}, {}
        for k in CTX_KEYS:
            col = f"{tag}_{k}"
            ctx[k] = te[col].to_numpy().astype(np.float64)
            med[k] = float(np.nanmedian(va[col].to_numpy().astype(np.float64)))
        out.append(dict(
            ym=m, nd=nd, n=te.height, st=st, sv=sv, y=yt, ct=ct, ctx=ctx, med=med,
            auc=float(roc_auc_score(yt, st + .5)),
            auc_v=float(roc_auc_score(yv, sv + .5)),
            base_rate=float(yt.mean()),
            thr={q: float(np.quantile(np.abs(sv), 1 - q / 100)) for q in Q_REF}))
        base_arrs.append(out[-1])
    return out


def report(tag: str, fset: str, recs: list[dict], months: int) -> dict:
    nd_all = sum(r["nd"] for r in recs)
    auc = np.mean([r["auc"] for r in recs])
    aucv = np.mean([r["auc_v"] for r in recs])
    br = float(np.average([r["base_rate"] for r in recs],
                          weights=[r["n"] for r in recs]))
    log(f"[{tag}] 特征集={fset:<4} 月={len(recs)} AUC(test)={auc:.4f} "
        f"AUC(val)={aucv:.4f} base_rate={br * 100:.2f}%")
    print(f"{HDR}")
    for q in Q_VAR:
        print(f"  --- 分位 q={q}% ---")
        for name, keys in VARIANTS:
            rows = []
            for r in recs:
                hit = np.abs(r["st"]) >= r["thr"][q]
                comp = _components(r["st"], r["ctx"], r["med"]) if keys else None
                m = hit.copy()
                for k in keys:
                    m &= comp[k]
                idx = _pick(m)
                rows.append(None if len(idx) < MIN_N else
                            dict(n=len(idx), win=_win(r["st"], r["y"], idx)))
            print(f"{name:>10}{_fmt(_agg(rows, nd_all))}")
    # 去重叠口径：base 与 trend+p3（cooldown = horizon）
    h_bars = recs[0].get("h_bars") if recs and "h_bars" in recs[0] else None
    print(f"  --- 去重叠（cooldown=horizon={h_bars} 根）---")
    for name, keys in (("base", ()), ("trend+p3", ("tr", "p3")), ("long", ("L",))):
        rows2 = []
        for r in recs:
            hit = np.abs(r["st"]) >= r["thr"][1]
            comp = _components(r["st"], r["ctx"], r["med"]) if keys else None
            m = hit.copy()
            for k in keys:
                m &= comp[k]
            idx = _pick(m, cooldown=h_bars or 1)
            rows2.append(None if len(idx) < MIN_N else
                         dict(n=len(idx), win=_win(r["st"], r["y"], idx)))
        print(f"{name + '@noov':>14}{_fmt(_agg(rows2, nd_all))}")
    print()


def run_config(df1m: pl.DataFrame, minutes: int, horizon: int) -> None:
    tag = f"m{minutes}h{horizon}"
    base = df1m if minutes == 1 else resample(df1m, minutes)
    d, feats_all = build(base, minutes, horizon, tag)
    feats_base = [c for c in feats_all if "_sp_" not in c]
    feats_spot = [c for c in feats_all if "_sp_" in c]
    log(f"[{tag}] 可用 {d.height:,} 行 | 特征 base={len(feats_base)} spot={len(feats_spot)}")
    months = [m for m in sorted(d["ym"].unique().to_list())
              if START_MONTH <= m <= END_MONTH]
    h_bars = max(1, int(round(horizon / minutes)))
    for fset, feats in (("base", feats_base), ("spot", feats_all)):
        keep: list[dict] = []
        recs = train_eval(d, feats, months, tag, keep)
        for r in recs:
            r["h_bars"] = h_bars
        if fset == "base":
            SCORES[tag] = {r["ym"]: r for r in recs}
        report(tag, fset, recs, len(months))
    del d
    gc.collect()


def run_consensus() -> None:
    log("多周期共振（同基础尺度，分数同向且各自过 q 分位）")
    print(f"{'pair':>16}{'q':>4}{HDR}")
    for t1, t2 in CONSENSUS:
        k1, k2 = f"m{t1[0]}h{t1[1]}", f"m{t2[0]}h{t2[1]}"
        if k1 not in SCORES or k2 not in SCORES:
            continue
        for q in Q_VAR:
            rows = []
            for ym, r1 in SCORES[k1].items():
                r2 = SCORES[k2].get(ym)
                if r2 is None:
                    continue
                # 两边都按 close_time 升序：用 r2 的时间在 r1 里定位同一根 bar
                pos = np.clip(np.searchsorted(r1["ct"], r2["ct"]), 0, len(r1["ct"]) - 1)
                keep = r1["ct"][pos] == r2["ct"]
                if not keep.any():
                    continue
                s1 = r1["st"][pos][keep]
                s2 = r2["st"][keep]
                y = r2["y"][keep]
                m = ((np.abs(s1) >= r1["thr"][q]) & (np.abs(s2) >= r2["thr"][q])
                     & (np.sign(s1) == np.sign(s2)))
                idx = _pick(m)
                rows.append(None if len(idx) < MIN_N else
                            dict(n=len(idx), win=_win(s1, y, idx)))
            nd_all = sum(r["nd"] for r in SCORES[k1].values())
            print(f"{k1 + '&' + k2:>16}{q:>3}%{_fmt(_agg(rows, nd_all))}")
    print()


def main() -> None:
    raw = load_raw()
    log(f"配置：{CONFIGS} | 月份 {START_MONTH}~{END_MONTH} | seeds={SEEDS} "
        f"| 平衡线={BREAKEVEN * 100:.2f}%")
    for minutes, horizon in CONFIGS:
        run_config(raw, minutes, horizon)
    run_consensus()
    log("DONE")


if __name__ == "__main__":
    main()
