#!/usr/bin/env python3
"""h=10 K 线工作日专模：只在 Mon-Fri 训练/定阈/下单，周末 h3 完全冻结。

与 h10_signal 生产口径的区别：单一切分（64/16/20）代替逐月 walk-forward，
用来回答「工作日单独训练能否学到方向」；样本仍取最后 WIN_DAYS 天，
特征在完整连续 K 线上算好后再筛星期，标签先算再筛。

产物：pred parquet + report json，不在 kernel 内挑最优参数。
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import polars as pl

try:
    from morph import h10_signal as hs
    from morph import kline_long as kl
    from morph.two_legs import add_target, fit, matrix
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from morph import h10_signal as hs
    from morph import kline_long as kl
    from morph.two_legs import add_target, fit, matrix

HORIZON = int(os.environ.get("HORIZON", "10"))
WIN_DAYS = int(os.environ.get("WIN_DAYS", "1460"))
SEEDS = [int(s) for s in os.environ.get("SEEDS", "42,43,44,45,46").split(",") if s.strip()]
ROUNDS = int(os.environ.get("ROUNDS", "2000"))
TPD_WD = int(os.environ.get("TPD_WD", "12"))
EXP = os.environ.get("EXP", "WD1")
WORK = Path(os.environ.get("KAGGLE_WORKING_DIR", "/kaggle/working"))
PAYOUT = 0.8
TRAIN_FRAC, VAL_FRAC = 0.64, 0.16
T = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T:7.1f}s] {msg}", flush=True)


def metrics(p: np.ndarray, y: np.ndarray, ct: np.ndarray, thr: float) -> dict:
    s = p - 0.5
    idx = np.where(np.abs(s) >= thr)[0]
    if len(idx) == 0:
        return {"n": 0}
    d = np.where(s[idx] > 0, 1, 0)
    w = (d == y[idx]).astype(np.int8)
    days = ct // 86_400_000
    nd = len(np.unique(days))
    nt = len(np.unique(days[idx]))
    ym = (ct[idx] // 1000).astype("datetime64[s]").astype("datetime64[M]")
    mv = np.array([w[ym == m].mean() for m in np.unique(ym)])
    total = float(w.mean())
    neff = len(w) / HORIZON
    se = float(np.sqrt(total * (1 - total) / max(neff, 1)))
    cs = np.concatenate([[0], np.cumsum(w)])
    r100 = ((cs[100:] - cs[:-100]) / 100 if len(w) > 100 else np.array([total]))
    return dict(n=int(len(w)), days=int(nd), traded_days=int(nt),
                per_day=float(len(w) / max(nd, 1)),
                per_traded_day=float(len(w) / max(nt, 1)),
                win=total, se=se, month_min=float(mv.min()),
                month_std=float(mv.std()), months=int(len(mv)),
                roll100_min=float(r100.min()),
                ev=float((1 + PAYOUT) * total - 1),
                breakeven=float(1 / (1 + PAYOUT)))


def main() -> None:
    raw = kl.load_raw()
    df, feats = kl.build(raw)
    df, feats = hs.add_flow(df, feats)
    df = add_target(df, HORIZON).filter(pl.col("target").is_not_null()).sort("close_time")
    ct_all = df["close_time"].to_numpy()
    cut = int(ct_all.max()) - WIN_DAYS * 86_400_000
    wd = pl.from_epoch(pl.col("close_time"), time_unit="ms").dt.weekday()
    df = df.with_columns(wd.alias("wd")).filter(
        (pl.col("close_time") >= cut) & (pl.col("wd") <= 5)).sort("close_time")
    n = df.height
    tr1, va1 = int(n * TRAIN_FRAC), int(n * (TRAIN_FRAC + VAL_FRAC))
    log(f"[{EXP}] 工作日样本 {n:,} 行（窗 {WIN_DAYS} 天）| "
        f"train={tr1:,} val={va1-tr1:,} test={n-va1:,} | 特征={len(feats)} seeds={SEEDS}")

    X = matrix(df, feats)
    y = df["target"].to_numpy().astype(int)
    ct = df["close_time"].to_numpy()

    p = np.zeros(n, dtype=np.float64)
    for seed in SEEDS:
        m = fit(X[:tr1], y[:tr1], feats, X[tr1:va1], y[tr1:va1], seed=seed)
        p[tr1:va1] += m.predict(X[tr1:va1], num_iteration=m.best_iteration) / len(SEEDS)
        p[va1:] += m.predict(X[va1:], num_iteration=m.best_iteration) / len(SEEDS)
        acc_va = ((p[tr1:va1] > 0.5).astype(int) == y[tr1:va1]).mean()
        log(f"[{EXP}] seed={seed} best_iter={m.best_iteration} val_acc={acc_va:.4f}")
    p[:tr1] = np.nan

    vdays = np.unique(ct[tr1:va1] // 86_400_000)
    cal = np.isin(ct[tr1:va1] // 86_400_000, vdays[:len(vdays) // 2])
    thr = float(np.quantile(np.abs(p[tr1:va1][cal] - 0.5), 1 - TPD_WD / 1440))
    log(f"[{EXP}] 阈值={thr:.5f}（val 前半 {len(vdays)//2} 工作日，目标 {TPD_WD} 单/天）")

    seg = np.full(n, "train", dtype=object)
    seg[tr1:va1] = "val"
    seg[va1:] = "test"
    out = pl.DataFrame(dict(close_time=ct, wd=df["wd"].to_numpy(),
                            target=y, p=p, segment=seg))
    out = out.filter(pl.col("segment") != "train")
    out.write_parquet(WORK / f"h10_weekday_{EXP}_pred.parquet")

    report = dict(config=dict(horizon=HORIZON, window_days=WIN_DAYS, tpd_wd=TPD_WD,
                              thr=thr, seeds=SEEDS, rounds=ROUNDS, payout=PAYOUT),
                  metrics={})
    for tag in ("val2", "test"):
        sub = out.filter(pl.col("segment") == tag)
        if tag == "val2":
            sub = sub.filter(~pl.col("close_time").is_in(ct[tr1:va1][cal]))
        r = metrics(sub["p"].to_numpy(), sub["target"].to_numpy(),
                    sub["close_time"].to_numpy(), thr)
        report["metrics"][tag] = r
        log(f"[{EXP}] {tag}: n={r.get('n')} 日均={r.get('per_day', 0):.1f} "
            f"胜率={r.get('win', 0) * 100:.2f}%±{r.get('se', 0) * 100:.2f}pp "
            f"EV={r.get('ev', 0):+.4f} 月min={r.get('month_min', 0) * 100:.2f}%")
    (WORK / f"h10_weekday_{EXP}_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"[{EXP}] DONE -> {WORK}/h10_weekday_{EXP}_pred.parquet")


if __name__ == "__main__":
    main()
