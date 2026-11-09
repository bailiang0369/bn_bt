#!/usr/bin/env python3
"""D0：ETH 为什么比 BTC 差 —— 同窗口、同特征、同口径的机制诊断。

只输出统计到 stdout（外加一个小 json），**不落大数据产物、不需要下载到本地**。
ETH 订单簿来自 `kernel_sources` 挂载的 ethbuild output，BTC 订单簿与两侧 K 线
走 `ob-dataset-vol8` / `vision-base` 的 asset API。

四个候选机制 + 对应指标
----------------------
1. 盘口分辨率：ETH 的 tick/价格比 BTC 粗（0.01/3000 vs 0.1/100000）→ 同样 5 档
   覆盖更宽的 bps 区间。指标：`ob_f_spread_bps` / `ob_f_vwap_*_v0_bps` 分位，
   以及「中位 |r3| / 中位 spread」这一信噪比。
2. 特征退化：分母趋零的 bps 列（`ob_spread_ratio_f_s` 已有 inf）在 ETH 上更严重。
   指标：每列 inf 行数、近常数列数、全 NaN 行数、单列唯一值数。
3. 单变量信息整体更少：184 列逐列 test AUC 的分布（中位/均值/>0.53 计数），
   以及跨标的逐列 AUC 的相关系数、top10 强列的重叠度。
4. 只有极端分不稳：模型分数 top1%/5%/25% 的**局部 AUC**、按天 top15 胜率的
   均值/标准差/正天数比例，以及 val→test 的落差。

env: ROUNDS / TARGET_TPD / SEED / ETH_OB / OUT
"""
from __future__ import annotations

import glob
import json
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
    from morph.h3_gate import split_wide
    from morph.two_legs import (TARGET_TPD, TRAIN_FRAC, VAL_FRAC, acc_at_cov,
                                add_target, fit, matrix)
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from morph import h3_gate_eval as ge
    from morph import ob_kline as ok
    from morph.h3_gate import split_wide
    from morph.two_legs import (TARGET_TPD, TRAIN_FRAC, VAL_FRAC, acc_at_cov,
                                add_target, fit, matrix)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

HORIZON = 3
KLINE_YEARS = (2025, 2026)
SEED = int(os.environ.get("SEED", "42"))
ETH_OB = os.environ.get("ETH_OB", "")
OUT = os.environ.get("OUT", "/kaggle/working/eth_diag.json")
T = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T:7.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------- 加载
def load_eth_ob() -> pl.DataFrame:
    p = ETH_OB
    if not p:
        hits = sorted(glob.glob("/kaggle/input/**/ob_ETHUSDT.parquet", recursive=True))
        if not hits:
            raise SystemExit("找不到 ob_ETHUSDT.parquet（ethbuild output 未挂载？）")
        p = hits[0]
    log(f"[data] ETH {p}")
    return (pl.read_parquet(p).unique(subset=["close_time"], keep="last")
            .sort("close_time"))


def load_btc_ob() -> pl.DataFrame:
    a = ok._assets(ok.OB_TAG)
    whole = "ob_BTCUSDT.parquet"
    if whole in a:
        log(f"[data] BTC {whole}（整包）")
        df = pl.read_parquet(ok.fetch(ok.OB_TAG, whole))
    else:
        ms = sorted(n[len("ob_BTCUSDT_"):-len(".parquet")] for n in a
                    if n.startswith("ob_BTCUSDT_") and n.endswith(".parquet"))
        if not ms:
            raise SystemExit(f"{ok.OB_TAG} 里没有 ob_BTCUSDT*")
        log(f"[data] BTC 月度分片 {len(ms)} 个（{ms[0]} ~ {ms[-1]}）")
        df = pl.concat([pl.read_parquet(ok.fetch(ok.OB_TAG, f"ob_BTCUSDT_{m}.parquet"))
                        for m in ms], how="vertical_relaxed")
    return df.unique(subset=["close_time"], keep="last").sort("close_time")


def load_close(sym: str) -> pl.DataFrame:
    a = ok._assets(ok.KLINE_TAG)
    have = [y for y in KLINE_YEARS if f"{sym}_1m_klines_{y}.parquet" in a]
    if not have:
        raise SystemExit(f"{ok.KLINE_TAG} 里没有 {sym} 的 K 线")
    df = pl.concat([pl.read_parquet(ok.fetch(ok.KLINE_TAG, f"{sym}_1m_klines_{y}.parquet"))
                    .select("close_time", "close") for y in have])
    return (df.unique(subset=["close_time"], keep="last").sort("close_time")
              .filter(pl.col("close").is_not_null() & (pl.col("close") > 0)))


# ---------------------------------------------------------------- 口径
def local_auc(s: np.ndarray, y: np.ndarray, cov: float) -> float:
    k = max(2, int(len(y) * cov))
    idx = np.argsort(-np.abs(s))[:k]
    if y[idx].min() == y[idx].max():
        return float("nan")
    return float(roc_auc_score(y[idx], s[idx]))


def daily_topk_win(s: np.ndarray, y: np.ndarray, days: np.ndarray, k: int = 15) -> dict:
    wins = []
    for d in np.unique(days):
        m = days == d
        kk = min(k, int(m.sum()))
        ss, yy = s[m], y[m]
        idx = np.argsort(-np.abs(ss))[:kk]
        wins.append(float(((ss[idx] > 0).astype(int) == yy[idx]).mean()))
    w = np.array(wins)
    return dict(days=int(len(w)), mean=float(w.mean()), std=float(w.std()),
                pos_frac=float((w > 0.5).mean()), p10=float(np.quantile(w, .10)),
                p90=float(np.quantile(w, .90)))


def tiers(s: np.ndarray, y: np.ndarray, days: np.ndarray, tpd: int) -> dict:
    k = min(tpd * len(np.unique(days)), len(y))
    return {"@15": acc_at_cov(s, y, k / len(y)), "@1%": acc_at_cov(s, y, .01),
            "@5%": acc_at_cov(s, y, .05), "@25%": acc_at_cov(s, y, .25)}


def analyze(sym: str, ob: pl.DataFrame, t_tr_end: int, t_va_end: int) -> dict:
    obcols = [c for c in ob.columns if c != "close_time"]
    close = load_close(sym)
    wide = ob.join(close, on="close_time", how="left").sort("close_time")
    wide = add_target(wide, HORIZON).filter(
        pl.col("target").is_not_null() & pl.col("close").is_not_null())

    c = wide["close"].to_numpy().astype(np.float64)
    r3 = np.full(len(c), np.nan)
    r3[:-HORIZON] = (c[HORIZON:] / c[:-HORIZON] - 1.0) * 1e4
    wide = wide.with_columns(pl.Series("r3", r3))

    tr, va, te = (split_wide(wide, "time", t_tr_end, t_va_end))

    y_te = te["target"].to_numpy().astype(np.int8)
    d_te = ge.day_id(te["close_time"].to_numpy())
    ar3 = np.abs(te["r3"].to_numpy().astype(np.float64))
    ar3 = ar3[np.isfinite(ar3)]
    nm = lambda a: float(np.nanmedian(a))  # noqa: E731

    scale = dict(
        abs_r3_med_bps=nm(ar3), abs_r3_p25_bps=float(np.nanquantile(ar3, .25)),
        abs_r3_p75_bps=float(np.nanquantile(ar3, .75)),
        f_spread_med_bps=nm(te["ob_f_spread_bps"].to_numpy().astype(np.float64)),
        s_spread_med_bps=nm(te["ob_s_spread_bps"].to_numpy().astype(np.float64)),
        f_vwap_v0_med_bps=nm(np.abs(te["ob_f_vwap_bid_v0_bps"].to_numpy().astype(np.float64))),
        basis_med_bps=nm(te["ob_basis_bps"].to_numpy().astype(np.float64)),
        f_impact_med=nm(te["ob_f_impact_bid_bps_per_btc"].to_numpy().astype(np.float64)),
    )
    scale["snr_r3_over_spread"] = scale["abs_r3_med_bps"] / max(scale["f_spread_med_bps"], 1e-9)

    X = matrix(te, obcols).astype(np.float64)
    fin = np.isfinite(X)
    n_inf = np.isinf(X).sum(axis=0)
    col_nan = (~fin).mean(axis=0)
    col_std = np.array([np.nanstd(np.where(fin[:, i], X[:, i], np.nan))
                        if fin[:, i].any() else np.nan for i in range(len(obcols))])
    col_nun = np.array([len(np.unique(X[fin[:, i], i])) if fin[:, i].any() else 0
                        for i in range(len(obcols))])
    degrade = dict(
        cols_with_inf=int((n_inf > 0).sum()),
        inf_rows_total=int(n_inf.sum()),
        worst_inf=[(obcols[i], int(n_inf[i])) for i in np.argsort(-n_inf)[:6] if n_inf[i] > 0],
        cols_all_nan_test=int((col_nan > .999).sum()),
        cols_near_const=int(((col_std < 1e-12) | (col_nun <= 2)).sum()),
        row_all_nan_test=int((~fin).all(axis=1).sum()),
    )

    col_auc = {}
    for i, name in enumerate(obcols):
        m = fin[:, i]
        if m.sum() < 50_000 or len(np.unique(X[m, i])) < 3:
            continue
        a = roc_auc_score(y_te[m], X[m, i])
        col_auc[name] = float(max(a, 1.0 - a))
    ca = np.array(list(col_auc.values()))
    uni = dict(
        cols_scored=int(len(ca)), auc_median=float(np.median(ca)), auc_mean=float(ca.mean()),
        n_gt_053=int((ca > .53).sum()), n_gt_055=int((ca > .55).sum()),
        top10=sorted(col_auc.items(), key=lambda kv: -kv[1])[:10],
    )

    Xtr, ytr = matrix(tr, obcols), tr["target"].to_numpy().astype(int)
    Xva, yva = matrix(va, obcols), va["target"].to_numpy().astype(int)
    m = fit(Xtr, ytr, obcols, Xva, yva, seed=SEED)
    p_va, p_te = m.predict(Xva), m.predict(matrix(te, obcols))
    log(f"[{sym}] best_iter={m.best_iteration}")

    s_va, s_te = p_va - .5, p_te - .5
    y_va = va["target"].to_numpy().astype(np.int8)
    d_va = ge.day_id(va["close_time"].to_numpy())
    res = dict(
        symbol=sym, rows=int(wide.height),
        n_train=int(tr.height), n_val=int(va.height), n_test=int(te.height),
        target_mean_train=float(tr["target"].mean()), target_mean_val=float(y_va.mean()),
        target_mean_test=float(y_te.mean()), best_iter=int(m.best_iteration),
        auc_val=float(roc_auc_score(y_va, p_va)), auc_test=float(roc_auc_score(y_te, p_te)),
        tiers_val=tiers(s_va, y_va, d_va, TARGET_TPD),
        tiers_test=tiers(s_te, y_te, d_te, TARGET_TPD),
        daily_test=daily_topk_win(s_te, y_te, d_te),
        local_auc_test={f"top{c:.2f}": local_auc(s_te, y_te, c) for c in (.01, .05, .25)},
        scale=scale, degrade=degrade, univariate=uni,
    )
    res["_col_auc"] = col_auc
    for k, v in (("val", res["tiers_val"]), ("test", res["tiers_test"])):
        log(f"[{sym} {k}] " + " | ".join(f"{a}={b * 100:.2f}%" for a, b in v.items()))
    log(f"[{sym}] AUC val={res['auc_val']:.4f} test={res['auc_test']:.4f}"
        f" | local {json.dumps({k: round(v, 4) for k, v in res['local_auc_test'].items()})}")
    log(f"[{sym}] 尺度 " + " ".join(f"{a}={b:.4f}" for a, b in scale.items()))
    log(f"[{sym}] 退化 {json.dumps(degrade, ensure_ascii=False)}")
    log(f"[{sym}] 单变量 median={uni['auc_median']:.4f} mean={uni['auc_mean']:.4f}"
        f" >0.53={uni['n_gt_053']}/{uni['cols_scored']} >0.55={uni['n_gt_055']}")
    log(f"[{sym}] top10 " + ", ".join(f"{c}:{v:.4f}" for c, v in uni["top10"]))
    log(f"[{sym}] 按天top15 mean={res['daily_test']['mean'] * 100:.2f}%"
        f" std={res['daily_test']['std'] * 100:.2f}pp"
        f" 正天={res['daily_test']['pos_frac'] * 100:.0f}%"
        f" p10={res['daily_test']['p10'] * 100:.1f}% p90={res['daily_test']['p90'] * 100:.1f}%")
    return res


def compare(eth: dict, btc: dict) -> None:
    a, b = eth["_col_auc"], btc["_col_auc"]
    common = sorted(set(a) & set(b))
    if not common:
        log("跨标的：没有共同的单变量列（异常）")
        return
    va = np.array([a[c] for c in common])
    vb = np.array([b[c] for c in common])
    ta = {c for c, _ in eth["univariate"]["top10"]}
    tb = {c for c, _ in btc["univariate"]["top10"]}
    log(f"跨标的：共同列 {len(common)} | 逐列AUC相关={np.corrcoef(va, vb)[0, 1]:.3f}"
        f" | ETH median={np.median(va):.4f} mean={va.mean():.4f}"
        f" | BTC median={np.median(vb):.4f} mean={vb.mean():.4f}")
    log(f"跨标的：top10 重叠 {len(ta & tb)}/10"
        f" | 仅BTC强 {sorted(tb - ta)[:6]} | 仅ETH强 {sorted(ta - tb)[:6]}")
    worse = sorted(((c, a[c] - b[c]) for c in common), key=lambda kv: kv[1])[:8]
    better = sorted(((c, a[c] - b[c]) for c in common), key=lambda kv: -kv[1])[:8]
    log("跨标的：ETH 相对最弱 " + ", ".join(f"{c}:{d:+.4f}" for c, d in worse))
    log("跨标的：ETH 相对最强 " + ", ".join(f"{c}:{d:+.4f}" for c, d in better))
    for k in ("abs_r3_med_bps", "f_spread_med_bps", "snr_r3_over_spread",
              "f_vwap_v0_med_bps", "basis_med_bps", "f_impact_med"):
        log(f"跨标的：{k}  ETH={eth['scale'][k]:.4f}  BTC={btc['scale'][k]:.4f}"
            f"  比值={eth['scale'][k] / btc['scale'][k] if btc['scale'][k] else float('nan'):.2f}")
    log(f"跨标的：val→test @25%  ETH={100 * (eth['tiers_test']['@25%'] - eth['tiers_val']['@25%']):+.2f}pp"
        f"  BTC={100 * (btc['tiers_test']['@25%'] - btc['tiers_val']['@25%']):+.2f}pp")
    log(f"跨标的：val→test @15   ETH={100 * (eth['tiers_test']['@15'] - eth['tiers_val']['@15']):+.2f}pp"
        f"  BTC={100 * (btc['tiers_test']['@15'] - btc['tiers_val']['@15']):+.2f}pp")


def main() -> None:
    eth_ob = load_eth_ob()
    n = eth_ob.height
    t_tr_end = int(eth_ob["close_time"][int(n * TRAIN_FRAC) - 1])
    t_va_end = int(eth_ob["close_time"][int(n * (TRAIN_FRAC + VAL_FRAC)) - 1])
    log(f"统一时间边界（按 ETH 分位）：T_end={ok._iso(t_tr_end)} | val 末={ok._iso(t_va_end)}")

    out = {}
    out["ETHUSDT"] = analyze("ETHUSDT", eth_ob, t_tr_end, t_va_end)
    del eth_ob

    btc_ob = load_btc_ob()
    out["BTCUSDT"] = analyze("BTCUSDT", btc_ob, t_tr_end, t_va_end)

    compare(out["ETHUSDT"], out["BTCUSDT"])
    for r in out.values():
        r.pop("_col_auc", None)
    if OUT:
        Path(OUT).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        log(f"统计已落盘: {OUT}")
    log("DONE")


if __name__ == "__main__":
    main()
