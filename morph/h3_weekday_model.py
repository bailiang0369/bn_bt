#!/usr/bin/env python3
"""h=3 工作日专模：只在 Mon-Fri 训练/定阈/下单，周末模型完全冻结。

目标（用户 2026-11-22）：周末保持现役 h3 的 0.05806 阈值与单量/胜率不变；
工作日另训一个模型，在保持 10~15 单/天的前提下提高胜率。

口径：
- 数据：订单簿 184 列快照 + K 线 close（只用来算 target）。
- 样本：只保留 UTC 周一~周五；按时间 64/16/20 切 train/val/test，段界 purge 3 bar。
- 训练：5 seeds 概率平均，LightGBM binary，early stopping 只用 val。
- 阈值：只在 val 前半的工作日分数上按 tpd_wd/1440 分位定；val 后半与 test 只评估。
- 判据：test 工作日 10~15 单/天、胜率显著高于现役基线、月 min 不低于平衡线。

产物：pred parquet + report json（供本地配对检验，不在 kernel 内选优）。
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import polars as pl

try:
    from morph import ob_kline as ok
    from morph.two_legs import add_target, fit, matrix
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from morph import ob_kline as ok
    from morph.two_legs import add_target, fit, matrix

HORIZON = int(os.environ.get("HORIZON", "3"))
SEEDS = [int(s) for s in os.environ.get("SEEDS", "42,43,44,45,46").split(",") if s.strip()]
ROUNDS = int(os.environ.get("ROUNDS", "2000"))
TPD_WD = int(os.environ.get("TPD_WD", "12"))
EXP = os.environ.get("EXP", "WD")
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
    ob = ok.load_ob()
    obcols = [c for c in ob.columns if c != "close_time"]
    lo, hi = int(ob["close_time"].min()), int(ob["close_time"].max())
    kf, _ = ok.load_kline_features(keep_raw=False)
    px = kf.select("close_time", "close").filter(
        (pl.col("close_time") >= lo - 130 * 60_000) & (pl.col("close_time") <= hi))
    wide = (ob.join(px, on="close_time", how="left").sort("close_time"))
    wide = add_target(wide, HORIZON).filter(
        pl.col("target").is_not_null() & pl.col("close").is_not_null())

    wd = (pl.from_epoch(pl.col("close_time"), time_unit="ms").dt.weekday())
    wide = wide.with_columns(wd.alias("wd")).filter(pl.col("wd") <= 5).sort("close_time")
    n = wide.height
    tr1, va1 = int(n * TRAIN_FRAC), int(n * (TRAIN_FRAC + VAL_FRAC))
    log(f"[{EXP}] 工作日样本 {n:,} 行 | train={tr1:,} val={va1-tr1:,} test={n-va1:,} "
        f"| 特征={len(obcols)} seeds={SEEDS}")

    X = matrix(wide, obcols)
    y = wide["target"].to_numpy().astype(int)
    ct = wide["close_time"].to_numpy()

    p = np.zeros(n, dtype=np.float64)
    for seed in SEEDS:
        m = fit(X[:tr1], y[:tr1], obcols, X[tr1:va1], y[tr1:va1], seed=seed)
        p[tr1:va1] += m.predict(X[tr1:va1], num_iteration=m.best_iteration) / len(SEEDS)
        p[va1:] += m.predict(X[va1:], num_iteration=m.best_iteration) / len(SEEDS)
        acc_va = ((p[tr1:va1] > 0.5).astype(int) == y[tr1:va1]).mean()
        log(f"[{EXP}] seed={seed} best_iter={m.best_iteration} val_acc={acc_va:.4f}")
    p[:tr1] = np.nan

    vdays = np.unique(ct[tr1:va1] // 86_400_000)
    cut = vdays[:len(vdays) // 2]
    cal = np.isin(ct[tr1:va1] // 86_400_000, cut)
    thr = float(np.quantile(np.abs(p[tr1:va1][cal] - 0.5), 1 - TPD_WD / 1440))
    log(f"[{EXP}] 阈值={thr:.5f}（val 前半 {len(cut)} 工作日，目标 {TPD_WD} 单/天）")

    seg = np.full(n, "train", dtype=object)
    seg[tr1:va1] = "val"
    seg[va1:] = "test"
    out = pl.DataFrame(dict(close_time=ct, wd=wide["wd"].to_numpy(),
                            target=y, p=p, segment=seg))
    out = out.filter(pl.col("segment") != "train")
    out.write_parquet(WORK / f"h3_weekday_{EXP}_pred.parquet")

    report = dict(config=dict(horizon=HORIZON, tpd_wd=TPD_WD, thr=thr,
                              seeds=SEEDS, rounds=ROUNDS, payout=PAYOUT),
                  metrics={})
    for tag, a, b in (("cal", tr1, tr1 + int(np.sum(cal))),
                      ("val2", tr1, va1), ("test", va1, n)):
        if tag == "cal":
            continue
        sub = out.filter(pl.col("segment") == tag)
        if tag == "val2":
            sub = sub.filter(~pl.col("close_time").is_in(ct[tr1:va1][cal]))
        r = metrics(sub["p"].to_numpy(), sub["target"].to_numpy(),
                    sub["close_time"].to_numpy(), thr)
        report["metrics"][tag] = r
        log(f"[{EXP}] {tag}: n={r.get('n')} 日均={r.get('per_day', 0):.1f} "
            f"胜率={r.get('win', 0) * 100:.2f}%±{r.get('se', 0) * 100:.2f}pp "
            f"EV={r.get('ev', 0):+.4f} 月min={r.get('month_min', 0) * 100:.2f}%")
    (WORK / f"h3_weekday_{EXP}_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"[{EXP}] DONE -> {WORK}/h3_weekday_{EXP}_pred.parquet")


if __name__ == "__main__":
    main()
