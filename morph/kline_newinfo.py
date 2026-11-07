"""K 线腿引入**非价格**新信息：资金费率 / 持仓量 / 大户多空比 / taker 多空量比（第 1 步续）。

理由
----
K 线 63 列全是从 OHLCV 派生的，AUC 0.530，且与盘口腿相关 0.515（都是价格动量）。
要抬 AUC 只能引入**不同信息源**。Binance Vision 有现成的、体积很小的两类数据：
  futures/um/{monthly,daily}/fundingRate  —— 每 8 小时结算的资金费率（多空拥挤度）
  futures/um/{monthly,daily}/metrics      —— 每日持仓量、大户账户/持仓多空比、taker 买卖量比

二者都不是价格派生量，是"仓位/成本"维度，与 1m K 线的动量维度正交。

特征（全部 as-of 回填，只用已发生的数据）
  n_fund_last / n_fund_ma3 / n_fund_z30 / n_fund_7d
  n_oi_log / n_doi_1d / n_doi_7d / n_oi_px_div
  n_lsr_cnt / n_lsr_sum / n_lsr_cnt_z30 / n_lsr_sum_z30
  n_taker_lsr / n_taker_lsr_z30

变体（h=3）
  base  63 列（对照）
  new   63 + 14 列新信息
  only  只留 14 列（看这批数据自身的信息量）

判据：roll3d 三窗均值 ≥ base + 0.5pp 且最差窗不退化。
注意：metrics 可能 2021 年末才有，早期为 NaN（LGB 原生处理），存在"NaN≈早期"的隐含标记，
故只看相对增量；上线前需按有数据时段重训。
"""
import glob
import io
import os
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lightgbm as lgb  # noqa: E402
import requests  # noqa: E402

from morph.borrow_ab import score_eval  # noqa: E402
from morph.kline_boost import (OB_END, PARAMS, TEST_START, VAL_START,  # noqa: E402
                               build_features, load_klines, log, make_target)

H = 3
DAY = 86_400_000
BASE = "https://data.binance.vision/data/futures/um"
CACHE = "/tmp/newinfo"
NEW = ["n_fund_last", "n_fund_ma3", "n_fund_z30", "n_fund_7d",
       "n_oi_log", "n_doi_1d", "n_doi_7d", "n_oi_px_div",
       "n_lsr_cnt", "n_lsr_sum", "n_lsr_cnt_z30", "n_lsr_sum_z30",
       "n_taker_lsr", "n_taker_lsr_z30"]
COLS = {
    "fundingRate": ["calc_time", "funding_interval_hours", "last_funding_rate"],
    "metrics": ["create_time", "symbol", "sum_open_interest", "sum_open_interest_value",
                "count_toptrader_long_short_ratio", "sum_toptrader_long_short_ratio",
                "count_long_short_ratio", "sum_taker_long_short_vol_ratio"],
}


def grab(kind, symbol, period, daily):
    key = f"{kind}_{period}_{'d' if daily else 'm'}"
    p = os.path.join(CACHE, key + ".parquet")
    if os.path.exists(p):
        try:
            return pl.read_parquet(p)
        except Exception:
            pass
    url = f"{BASE}/{'daily' if daily else 'monthly'}/{kind}/{symbol}/{symbol}-{kind}-{period}.zip"
    try:
        r = requests.get(url, timeout=40)
        if r.status_code != 200:
            return None
        raw = io.BytesIO(zipfile.ZipFile(io.BytesIO(r.content))
                         .read([n for n in zipfile.ZipFile(io.BytesIO(r.content)).namelist()
                                if n.endswith(".csv")][0]))
        head = raw.readline().decode("utf-8", "replace")
        raw.seek(0)
        d = (pl.read_csv(raw) if head.split(",")[0].strip() in ("calc_time", "create_time")
             else pl.read_csv(raw, has_header=False, new_columns=COLS[kind]))
        d.write_parquet(p)
        return d
    except Exception:
        return None


def periods_monthly(a=(2020, 1), b=(2026, 10)):
    y, m = a
    out = []
    while (y, m) <= b:
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m > 12:
            y, m = y + 1, 1
    return out


def periods_daily(a=(2025, 7), b=(2026, 10)):
    out = []
    d0 = np.datetime64(f"{a[0]:04d}-{a[1]:02d}-01")
    d1 = np.datetime64(f"{b[0]:04d}-{b[1]:02d}-28")
    while d0 <= d1:
        out.append(str(d0))
        d0 = d0 + np.timedelta64(1, "D")
    return out


def load_kind(kind, symbol="BTCUSDT"):
    os.makedirs(CACHE, exist_ok=True)
    probe = grab(kind, symbol, "2026-10", False)
    if probe is not None and probe.height:
        log(f"[new] {kind} 月度可用")
        per = periods_monthly((2021, 1) if kind == "metrics" else (2020, 1), (2026, 10))
        fn = lambda p: grab(kind, symbol, p, False)  # noqa: E731
    else:
        log(f"[new] {kind} 月度不可用 → 走日度（2025-07~2026-10，与盘口数据同段）")
        per = periods_daily()
        fn = lambda p: grab(kind, symbol, p, True)  # noqa: E731
    out = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        for d in ex.map(fn, per):
            if d is not None and d.height:
                out.append(d)
    log(f"[new] {kind} 拉到 {len(out)}/{len(per)} 个文件")
    return pl.concat(out, how="vertical_relaxed") if out else None


def build(kind, d):
    if d is None:
        return None
    tcol = "calc_time" if kind == "fundingRate" else "create_time"
    d = d.with_columns(pl.col(tcol).cast(pl.Int64).alias("t")).sort("t").unique(subset=["t"])
    if kind == "fundingRate":
        return d.select("t", pl.col("last_funding_rate").cast(pl.Float64).alias("fund"))
    keep = [c for c in ("sum_open_interest", "sum_open_interest_value",
                        "count_toptrader_long_short_ratio", "sum_toptrader_long_short_ratio",
                        "count_long_short_ratio", "sum_taker_long_short_vol_ratio")
            if c in d.columns]
    return d.select(["t"] + [pl.col(c).cast(pl.Float64) for c in keep])


def asof(ts, vals, ct):
    ts = np.asarray(ts, dtype=np.int64)
    if len(ts) == 0:
        return np.full(len(ct), np.nan)
    v = np.asarray(vals, dtype=np.float64)
    idx = np.searchsorted(ts, ct, side="right") - 1
    return np.where(idx >= 0, v[np.clip(idx, 0, len(v) - 1)], np.nan)


def add_new(ct, close, f, m):
    out = {}
    if f is not None and f.height > 1:
        sp = f.with_columns([
            pl.col("fund").rolling_mean(3).alias("ma3"),
            pl.col("fund").rolling_sum(8).alias("s8"),
            ((pl.col("fund") - pl.col("fund").rolling_mean(90))
             / (pl.col("fund").rolling_std(90) + 1e-12)).alias("z90")])
        t = sp["t"].to_numpy()
        for name, col in (("n_fund_last", "fund"), ("n_fund_ma3", "ma3"),
                          ("n_fund_z30", "z90"), ("n_fund_7d", "s8")):
            out[name] = asof(t, sp[col].to_numpy(), ct)
    if m is not None and m.height > 1:
        t = m["t"].to_numpy()
        g = lambda c: (m[c].to_numpy() if c in m.columns else np.full(m.height, np.nan))  # noqa: E731
        sp = pl.DataFrame({"oi": g("sum_open_interest"),
                           "lsr_c": g("count_toptrader_long_short_ratio"),
                           "lsr_s": g("sum_toptrader_long_short_ratio"),
                           "tak": g("sum_taker_long_short_vol_ratio")})
        sp = sp.with_columns([
            pl.col("oi").log1p().alias("oi_log"),
            (pl.col("oi") / pl.col("oi").shift(1) - 1.0).alias("doi_1d"),
            (pl.col("oi") / pl.col("oi").shift(7) - 1.0).alias("doi_7d"),
            ((pl.col("lsr_c") - pl.col("lsr_c").rolling_mean(30))
             / (pl.col("lsr_c").rolling_std(30) + 1e-12)).alias("lsr_c_z"),
            ((pl.col("lsr_s") - pl.col("lsr_s").rolling_mean(30))
             / (pl.col("lsr_s").rolling_std(30) + 1e-12)).alias("lsr_s_z"),
            ((pl.col("tak") - pl.col("tak").rolling_mean(30))
             / (pl.col("tak").rolling_std(30) + 1e-12)).alias("tak_z")])
        for name, col in (("n_oi_log", "oi_log"), ("n_doi_1d", "doi_1d"), ("n_doi_7d", "doi_7d"),
                          ("n_lsr_cnt", "lsr_c"), ("n_lsr_sum", "lsr_s"),
                          ("n_lsr_cnt_z30", "lsr_c_z"), ("n_lsr_sum_z30", "lsr_s_z"),
                          ("n_taker_lsr", "tak"), ("n_taker_lsr_z30", "tak_z")):
            out[name] = asof(t, sp[col].to_numpy(), ct)
        ret1d = np.full(len(ct), np.nan)
        nxt = np.searchsorted(ct, ct + DAY, side="left")
        ok = nxt < len(ct)
        ret1d[ok] = close[nxt[ok]] / close[ok] - 1.0
        out["n_oi_px_div"] = out.get("n_doi_7d", np.full(len(ct), np.nan)) - ret1d
    return {c: np.nan_to_num(out.get(c, np.full(len(ct), np.nan)), nan=np.nan).astype(np.float32)
            for c in NEW}


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
    t0 = time.time()

    raw = load_klines().filter(pl.col("close_time") <= OB_END)
    ft, base_cols = build_features(make_target(raw, H))
    ct = ft["close_time"].to_numpy()
    close = ft["close"].to_numpy().astype(np.float64)
    y = ft["target"].to_numpy().astype(np.int8)

    f = build("fundingRate", load_kind("fundingRate"))
    m = build("metrics", load_kind("metrics"))
    feats = add_new(ct, close, f, m)
    ft = ft.with_columns([pl.Series(k, v) for k, v in feats.items()])
    cols = base_cols + NEW
    X = ft.select(cols).to_numpy().astype(np.float32)
    del ft, raw
    cover = float(np.isfinite(X[:, cols.index("n_oi_log")]).mean())
    log(f"[data] rows={X.shape[0]} feats={len(cols)} 持仓量覆盖率={cover:.2%}")

    sets = {"base": base_cols, "new": cols, "only": NEW}
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
        tr = np.where(ct < val_start)[0]
        tr = tr[tr + H < int(va_idx[0])][::stride]
        log(f"\n===== off={off} train={len(tr)} test={int(te_m.sum())} =====")
        for tag, sub in sets.items():
            idx = [cols.index(c) for c in sub]
            s = fit_scores(X[:, idx], y, tr, sub, seeds, rounds)
            r = score_eval(s, ct, y, va_m, te_m)
            rows[(off, tag)] = r
            log(f"[{tag:<5} {len(sub):>2}f] static={r['static']:.4f} roll1d={r['roll1d']:.4f} "
                f"roll3d={r['roll3d']:.4f} AUC={r['auc']:.4f} 按天={r['daily']:.4f}")

    log(f"\n{'变体':<6}{'off':>5}{'static':>9}{'roll1d':>9}{'roll3d':>9}{'AUC':>8}{'按天':>8}")
    for (off, tag), r in rows.items():
        log(f"{tag:<6}{off:>5}{r['static']:>9.4f}{r['roll1d']:>9.4f}"
            f"{r['roll3d']:>9.4f}{r['auc']:>8.4f}{r['daily']:>8.4f}")

    log(f"\n{'变体':<6}{'roll3d均值':>11}{'roll3d最差':>11}{'roll1d均值':>11}{'AUC均值':>9}{'Δroll3d':>10}")
    b3 = np.mean([rows[(o, "base")]["roll3d"] for o in offs if (o, "base") in rows])
    for tag in sets:
        rs = [rows[(o, tag)] for o in offs if (o, tag) in rows]
        if not rs:
            continue
        m3 = float(np.mean([r["roll3d"] for r in rs]))
        log(f"{tag:<6}{m3:>11.4f}{np.min([r['roll3d'] for r in rs]):>11.4f}"
            f"{np.mean([r['roll1d'] for r in rs]):>11.4f}"
            f"{np.mean([r['auc'] for r in rs]):>9.4f}{(m3-b3)*100:>+9.2f}pp")
    log(f"[done] {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
