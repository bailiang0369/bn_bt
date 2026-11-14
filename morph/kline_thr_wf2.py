#!/usr/bin/env python3
"""长周期 K 线 walk-forward v2：目标是**提升胜率**，不是提升单量。

v1（kline_thr_wf.py）：纯方向模型 + |score| 分位阈值，10 个月(2026-01~10)上
m5h60 1% = 61.37%，其余配置 55~57%。v2 把「训练一次、评估多种过滤条件」做成
网格，34 个月(2024-01~2026-10)复核并逐项排除无效方向。

v2 硬结论（已实测，勿重复投入）：
- 现货基差 / 现货主动买卖 / 现货领先（14 个特征）：AUC 无增量（±0.0005），
  h60 q1 胜率 +1.7pp 但 h240 全面变差 → 噪声，剔除；
- 顺趋势过滤（对齐 240m 动量）：**系统性有害**（h120 51.90%、h240 38.62%）；
  反倒是逆势(rev) 稳定略优于 base → 模型 alpha 在动量反转端；
- 信号持续性 p3（连续 3 根同向）：与 base 无差异（<0.2pp），无效；
- 高波动过滤 rvhi：系统性有害；低波动 rvlo 胜率最高但月度方差爆炸（std 20%+）；
- 长周期空头端基本无效（h120 short 41.49%）→ 只做多(long) 稳定优于双向；
- 唯一一致正向的几何过滤：erhi（60 根趋势效率高于 val 中位数），h120 上
  +1.8pp（62.23% vs 60.44%，26/33 月过线）；
- horizon 与胜率正相关：h30 57.9 → h60 59.5 → h120 60.4（q2 base 口径）。

v3 本轮唯一新增的信息源：**跨品种 ETH**（BTC 单品种已榨干）。
ETHUSDT 1m 与 BTC 同源同 release，按 close_time 对齐后加：ETH 多尺度动量、
ETH 领先/滞后 BTC 的收益差、BTC-ETH 滚动相关、可解释残差动量、两币波动比。
另加 q=0.5%（更极端阈值）与 long+erhi 组合。

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
AUX_SYMBOL = os.environ.get("AUX_SYMBOL", "ETHUSDT")   # 跨品种信息源

PAYOUT = 0.8
BREAKEVEN = 1.0 / (1.0 + PAYOUT)      # 55.56%：低于此线 EV 必为负
Q_REF = (0.5, 1, 2, 3)               # base 变体的分位参考
Q_VAR = (0.5, 1, 2)                  # 变体网格评估用的分位（新增 q=0.5%）
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
    ("long+erhi", ("L", "erh")),
]

SCORES: dict[str, dict] = {}    # tag -> {ym: dict(ct, st, y, thr)}


def log(msg: str) -> None:
    print(f"[{time.time() - T:7.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------- 数据
def load_raw(sym: str = None, light: bool = False) -> pl.DataFrame:
    """light=True 只保留 close_time/close（给跨品种辅助标的用，省内存）。"""
    sym = sym or SYMBOL
    a = ok._assets(ok.KLINE_TAG)
    names = [f"{sym}_1m_klines_{y}.parquet" for y in range(KLINE_FROM, 2027)]
    have = [n for n in names if n in a]
    if not have:
        raise SystemExit(f"{ok.KLINE_TAG} 里没有 {sym} K 线")
    cols = ["close_time", "close"] if light else None
    df = pl.concat([pl.read_parquet(ok.fetch(ok.KLINE_TAG, n), columns=cols)
                    for n in have], how="vertical_relaxed")
    df = df.unique(subset=["close_time"], keep="last").sort("close_time")
    if not light:
        df = df.with_columns(
            (pl.col("um_buy_vol") + pl.col("um_sell_vol")).alias("volume"),
            pl.col("um_buy_vol").alias("taker_buy"))
    log(f"{sym} K 线 {df.height:,} 行 | {len(df.columns)} 列")
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


def resample_close(df: pl.DataFrame, minutes: int) -> pl.DataFrame:
    """辅助标的（只有 close_time/close 两列）的轻量重采样。"""
    if minutes == 1:
        return df.select("close_time", "close")
    ms = minutes * MI
    return (df.with_columns(_b=(pl.col("close_time") // ms))
            .group_by("_b")
            .agg(pl.col("close_time").last(), pl.col("close").last(),
                 pl.len().alias("cnt"))
            .filter(pl.col("cnt") == minutes)
            .sort("close_time").drop("cnt"))


def xasset_feats(df: pl.DataFrame, minutes: int, tag: str) -> list[pl.Expr]:
    """跨品种（ETH）维度。BTC 单品种已榨干（现货基差/主动买卖实测无增量），
    唯一还没用过的独立信息是另一只主流币的同期行为。"""
    if "eth_close" not in df.columns:
        return []
    c = pl.col("close").cast(pl.Float64)
    ec = pl.col("eth_close").cast(pl.Float64)
    rb = (c / c.shift(1)).log()
    re_ = (ec / ec.shift(1)).log()
    e: list[pl.Expr] = []

    def nb(m: int) -> int:
        return max(1, int(round(m / minutes)))

    for m in (5, 15, 60, 240):
        n = nb(m)
        e.append(((ec / ec.shift(n)).log() * 1e4).alias(f"{tag}_xa_eret{m}"))
        # BTC 相对 ETH 的强弱（同一尺度收益差）
        e.append((((c / c.shift(n)).log() - (ec / ec.shift(n)).log()) * 1e4)
                 .alias(f"{tag}_xa_rs{m}"))
    # ETH 领先 BTC：短尺度 ETH 已走、BTC 未走的部分
    e.append(((ec / ec.shift(nb(5))).log() - (c / c.shift(nb(5))).log())
             .mul(1e4).alias(f"{tag}_xa_lead5"))
    e.append(((ec / ec.shift(nb(15))).log() - (c / c.shift(nb(15))).log())
             .mul(1e4).alias(f"{tag}_xa_lead15"))
    # 滚动相关与协方差结构（用 240 根）
    n = nb(240)
    mb = rb.rolling_mean(n, min_samples=max(2, n // 2))
    me = re_.rolling_mean(n, min_samples=max(2, n // 2))
    cov = ((rb - mb) * (re_ - me)).rolling_mean(n, min_samples=max(2, n // 2))
    vb = (rb - mb).pow(2).rolling_mean(n, min_samples=max(2, n // 2))
    ve = (re_ - me).pow(2).rolling_mean(n, min_samples=max(2, n // 2))
    e.append((cov / (vb * ve).sqrt() + 1e-9).alias(f"{tag}_xa_corr240"))
    # 用 ETH 解释 BTC 后的残差动量：BTC 自身未被大盘解释的部分
    beta = cov / (ve + 1e-12)
    e.append((beta * 1e4).alias(f"{tag}_xa_beta240"))
    e.append(((rb - beta * re_) * 1e4).alias(f"{tag}_xa_resid"))
    e.append(((rb - beta * re_).rolling_sum(nb(60)) * 1e4).alias(f"{tag}_xa_residm60"))
    # 两币波动比：谁在主导行情
    e.append(((rb.rolling_std(nb(60), min_samples=max(3, nb(30))) + 1e-9)
              / (re_.rolling_std(nb(60), min_samples=max(3, nb(30))) + 1e-9))
             .alias(f"{tag}_xa_rvr"))
    return e


def build(df: pl.DataFrame, minutes: int, horizon: int, tag: str):
    h_bars = max(1, int(round(horizon / minutes)))
    d = df.with_columns(
        (pl.col("close").shift(-h_bars) > pl.col("close")).cast(pl.Int8).alias("target"))
    d = d.filter(pl.col("target").is_not_null())
    d = d.with_columns(base_feats(d, minutes, tag) + shape_feats(d, minutes, tag)
                       + xasset_feats(d, minutes, tag))
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


def run_config(df1m: pl.DataFrame, aux1m: pl.DataFrame | None,
               minutes: int, horizon: int) -> None:
    tag = f"m{minutes}h{horizon}"
    base = df1m if minutes == 1 else resample(df1m, minutes)
    if aux1m is not None:
        ax = resample_close(aux1m, minutes).select(
            pl.col("close_time"), pl.col("close").alias("eth_close"))
        base = base.join(ax, on="close_time", how="left")
    d, feats_all = build(base, minutes, horizon, tag)
    feats_base = [c for c in feats_all if "_xa_" not in c]
    feats_xa = [c for c in feats_all if "_xa_" in c]
    log(f"[{tag}] 可用 {d.height:,} 行 | 特征 base={len(feats_base)} xasset={len(feats_xa)}")
    months = [m for m in sorted(d["ym"].unique().to_list())
              if START_MONTH <= m <= END_MONTH]
    h_bars = max(1, int(round(horizon / minutes)))
    for fset, feats in (("base", feats_base), ("xasset", feats_all)):
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
    aux = None
    if AUX_SYMBOL and AUX_SYMBOL != SYMBOL:
        try:
            aux = load_raw(AUX_SYMBOL, light=True)
        except SystemExit as e:
            log(f"[!] 跨品种数据不可用，退回单品种：{e}")
    log(f"配置：{CONFIGS} | 月份 {START_MONTH}~{END_MONTH} | seeds={SEEDS} "
        f"| 平衡线={BREAKEVEN * 100:.2f}% | aux={AUX_SYMBOL if aux is not None else '-'}")
    for minutes, horizon in CONFIGS:
        run_config(raw, aux, minutes, horizon)
    run_consensus()
    log("DONE")


if __name__ == "__main__":
    main()
