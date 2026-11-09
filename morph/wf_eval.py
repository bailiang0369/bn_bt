#!/usr/bin/env python3
"""W1：walk-forward 评估（实盘口径，替代固定 64/16/20 的单一 test 数字）。

问题：固定划分下 `test @日均15单 66.9%` 与 `val 60.2%` 差 6.7pp。
这 6.7pp 里既有「时段差异」（必然存在），也有「模型陈旧」（实盘会重训消掉），
固定划分无法区分。本脚本按实盘做法逐月滚动：

    对 2026-01 ~ 2026-10 的每个月 M：
        train  = close_time < M 月 1 日（扩展窗口，越往后数据越多）
        early-stop = train 的最后 ES_FRAC（默认 20%，模拟「用最近数据定轮数」）
        预测   = 该月全部行，SEEDS 个模型概率平均
    最后把 10 个月的 OOS 预测拼起来，给整体口径。

判据仍按业务口径：
  - `@日均15单` = 该月按 |score| 取 15×天数 单（选择性下单，允许某些天 0 单）
  - `每天top15` = 强制每天满 15 单的均值（全天候下限）
两者同报，差值即「集中能力」。

env: SEEDS / ROUNDS / ES_FRAC / START_MONTH / END_MONTH / OUT / OB_MONTHS
"""
from __future__ import annotations

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

HORIZON = 3
SEEDS = [int(s) for s in os.environ.get("SEEDS", "42,43,44,45,46").split(",") if s.strip()]
ES_FRAC = float(os.environ.get("ES_FRAC", "0.2"))
START_MONTH = os.environ.get("START_MONTH", "2026-01")
END_MONTH = os.environ.get("END_MONTH", "2026-10")
OUT = os.environ.get("OUT", "/kaggle/working/wf_monthly.csv")
T = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T:7.1f}s] {msg}", flush=True)


def acc(s: np.ndarray, y: np.ndarray) -> float:
    return float(((s > 0).astype(int) == y).mean())


def topk_acc(s: np.ndarray, y: np.ndarray, k: int) -> float:
    k = min(k, len(y))
    idx = np.argsort(-np.abs(s))[:k]
    return acc(s[idx], y[idx])


def daily_mean(s: np.ndarray, y: np.ndarray, days: np.ndarray, n: int) -> tuple[float, float]:
    w = []
    for d in np.unique(days):
        m = days == d
        w.append(topk_acc(s[m], y[m], min(n, int(m.sum()))))
    a = np.array(w)
    return float(a.mean()), float(a.std())


def main() -> None:
    ob = ok.load_ob()
    obcols = [c for c in ob.columns if c != "close_time"]
    kfeat, _ = ok.load_kline_features(int(os.environ.get("KLINE_FROM", "2025")), keep_raw=False)
    wide = (ob.join(kfeat.select("close_time", "close"), on="close_time", how="left")
              .sort("close_time"))
    wide = add_target(wide, HORIZON).filter(
        pl.col("target").is_not_null() & pl.col("close").is_not_null())
    wide = wide.with_columns(
        pl.col("close_time").cast(pl.Datetime("ms")).dt.strftime("%Y-%m").alias("ym"))
    log(f"订单簿 {wide.height:,} 行 | 特征 {len(obcols)} 列 | 月份 {wide['ym'].min()} ~ {wide['ym'].max()}")

    months = [m for m in sorted(wide["ym"].unique().to_list())
              if START_MONTH <= m <= END_MONTH]
    log(f"walk-forward 窗口：{len(months)} 个（{months[0]} ~ {months[-1]}）| seeds={SEEDS}")
    if not months:
        raise SystemExit("窗口为空，检查 START_MONTH/END_MONTH")

    rows, all_s, all_y, all_d = [], [], [], []
    for m in months:
        tr_all = wide.filter(pl.col("ym") < m)
        te = wide.filter(pl.col("ym") == m)
        if tr_all.height < 50_000 or te.height < 1000:
            log(f"[{m}] 样本不足（train {tr_all.height:,} / test {te.height:,}），跳过")
            continue
        cut = int(tr_all.height * (1 - ES_FRAC))
        tr, es = tr_all[:cut], tr_all[cut:]
        Xtr, ytr = matrix(tr, obcols), tr["target"].to_numpy().astype(int)
        Xes, yes = matrix(es, obcols), es["target"].to_numpy().astype(int)
        ps = []
        for seed in SEEDS:
            mdl = fit(Xtr, ytr, obcols, Xes, yes, seed=seed)
            ps.append(mdl.predict(matrix(te, obcols)))
        p = np.mean(ps, axis=0)
        s, y = p - .5, te["target"].to_numpy().astype(np.int8)
        days = ge.day_id(te["close_time"].to_numpy())
        nd = len(np.unique(days))
        a15 = topk_acc(s, y, TARGET_TPD * nd)
        dm, ds = daily_mean(s, y, days, TARGET_TPD)
        rows.append(dict(ym=m, n=te.height, days=nd, auc=float(roc_auc_score(y, p)),
                         a15=a15, k=TARGET_TPD * nd, daily_mean=dm, daily_std=ds,
                         train=tr_all.height))
        all_s.append(s)
        all_y.append(y)
        all_d.append(days)
        log(f"[{m}] train={tr_all.height:,} test={te.height:,}({nd}天) AUC={rows[-1]['auc']:.4f}"
            f" @日均{TARGET_TPD}单={a15 * 100:.2f}%(K={TARGET_TPD * nd})"
            f" 每天top15={dm * 100:.2f}%(±{ds * 100:.1f}pp)")

    if not rows:
        raise SystemExit("没有任何窗口完成")

    s = np.concatenate(all_s)
    y = np.concatenate(all_y)
    d = np.concatenate(all_d)
    nd = len(np.unique(d))
    oos = topk_acc(s, y, TARGET_TPD * nd)
    dm, ds = daily_mean(s, y, d, TARGET_TPD)
    a = np.array([r["a15"] for r in rows])
    log(f"===== 全 OOS 拼接：{len(y):,} 行 / {nd} 天 | AUC={roc_auc_score(y, s + .5):.4f}"
        f" | @日均{TARGET_TPD}单={oos * 100:.2f}%（K={TARGET_TPD * nd}）"
        f" | 每天top15={dm * 100:.2f}%(±{ds * 100:.1f}pp)")
    seq = ", ".join(f"{r['ym']}:{r['a15'] * 100:.1f}%" for r in rows)
    log(f"===== 月度 @日均{TARGET_TPD}单 序列：{seq}")
    log(f"===== 月度分布：mean={a.mean() * 100:.2f}% std={a.std() * 100:.2f}pp"
        f" min={a.min() * 100:.2f}%({rows[int(np.argmin(a))]['ym']})"
        f" max={a.max() * 100:.2f}%  正月={int((a > .5).sum())}/{len(a)}"
        f"  ≥65%={int((a >= .65).sum())}/{len(a)}")
    if OUT:
        pl.DataFrame(rows).write_csv(OUT)
        log(f"月度明细已落盘: {OUT}")
    log("DONE")


if __name__ == "__main__":
    main()
