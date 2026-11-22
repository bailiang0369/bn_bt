#!/usr/bin/env python3
"""新阶段：h10 好/坏月的全特征漂移诊断（只读，不改策略）。

标签来源：历史 Kaggle 执行日志（h10sig / h10sigA / h10sigB / h10sigC）的逐月胜率，
写在 MONTHLY_WIN 里，带 provenance。不再预先冻结特征：对全部 49 列做扫描。

严格因果：用于监控的漂移量取"月初之前 30 天"的 trailing 窗口，
与当月训练参考分布（train 段）比较；另算"当月全月"的分布仅作描述，不用于门控。

统计：每个特征算 mean_z / median_z / std_ratio / PSI，与次月胜率做 Spearman，
置换检验 + BH-FDR 校正。探索结果只能生成假设，不能直接上线。
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from morph import kline_long as kl  # noqa: E402
from live_sim.h10_live_sim import add_flow, features  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

# provenance: kaggle kernels logs byh850515/btcusdt-h10-causal-signal(/a/b/c)
# 行格式 [ym] ... AUC=... 阈值=... X.X单/天 胜率=YY.YY%
MONTHLY_WIN = {
    "2026-01": 68.10, "2026-02": 55.74, "2026-03": 60.66, "2026-04": 68.56,
    "2026-05": 57.53, "2026-06": 67.18, "2026-07": 64.47, "2026-08": 68.36,
    "2026-09": 66.47, "2026-10": 60.54, "2026-11": 71.43,
}
WIN_BY_CONFIG = {
    "base": [68.10, 55.74, 60.66, 68.56, 57.53, 67.18, 64.47, 68.36, 66.47, 60.54, 71.43],
    "A": [60.77, 52.96, 64.58, 69.84, 55.44, 70.97, 67.83, 70.32, 64.43, 65.11],
    "B": [57.47, 57.19, 60.38, 66.67, 48.06, 66.67, 54.62, 72.58, 58.73, 64.58],
}
WORK = Path(os.environ.get("LIVE_SIM_OUT", "/kaggle/working/live_sim"))
MS_DAY = 86_400_000
PRE_DAYS = 30
WIN_DAYS = 1460
ES_FRAC, THR_FRAC = 0.05, 0.10
N_PERM = 5000
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def psi(a: np.ndarray, b: np.ndarray, bins: int = 20) -> float:
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 50 or len(b) < 50:
        return float("nan")
    edges = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    pa = np.histogram(a, bins=edges)[0] / len(a)
    pb = np.histogram(b, bins=edges)[0] / len(b)
    pa, pb = np.clip(pa, 1e-6, None), np.clip(pb, 1e-6, None)
    return float(np.sum((pa - pb) * np.log(pa / pb)))


def summary(x: np.ndarray) -> dict:
    x = x[np.isfinite(x)]
    return dict(mean=float(np.mean(x)), std=float(np.std(x)),
                median=float(np.median(x)), n=int(len(x)))


def drift(ref: dict, cur: dict, ref_arr: np.ndarray, cur_arr: np.ndarray) -> dict:
    sd = ref["std"] or 1e-9
    return dict(mean_z=(cur["mean"] - ref["mean"]) / sd,
                median_z=(cur["median"] - ref["median"]) / sd,
                std_ratio=cur["std"] / sd,
                psi=psi(cur_arr, ref_arr))


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 4 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def perm_p(x: np.ndarray, y: np.ndarray, observed: float, rng: np.random.Generator) -> float:
    if not np.isfinite(observed):
        return float("nan")
    rx = rankdata(x)
    hits = 0
    for _ in range(N_PERM):
        r = spearman(rx, rng.permutation(y))
        if np.isfinite(r) and abs(r) >= abs(observed) - 1e-12:
            hits += 1
    return (hits + 1) / (N_PERM + 1)


def bh_fdr(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    ok = np.isfinite(p)
    out = np.full(len(p), np.nan)
    idx = np.where(ok)[0]
    if not len(idx):
        return out
    pp = p[idx]
    order = np.argsort(pp)
    q = pp[order] * len(pp) / (np.arange(len(pp)) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    tmp = np.empty_like(q)
    tmp[order] = np.clip(q, 0, 1)
    out[idx] = tmp
    return out


def main() -> None:
    os.makedirs(WORK, exist_ok=True)
    raw = kl.load_raw()
    raw_feat, feats = features(raw)
    raw_feat = raw_feat.sort("close_time")
    ct_all = raw_feat["close_time"].to_numpy()
    months = sorted(MONTHLY_WIN)
    wins = np.array([MONTHLY_WIN[m] for m in months], dtype=float)
    log(f"标签 {len(months)} 个月：{months[0]}~{months[-1]} | 特征 {len(feats)} 列")

    pre_rows, full_rows = [], []
    for m in months:
        ms = int(dt.datetime.strptime(m + "-01", "%Y-%m-%d")
                 .replace(tzinfo=dt.timezone.utc).timestamp() * 1000)
        me = int((dt.datetime.strptime(m + "-01", "%Y-%m-%d")
                  .replace(tzinfo=dt.timezone.utc) + dt.timedelta(days=32))
                 .replace(day=1).timestamp() * 1000)
        lo = ms - WIN_DAYS * MS_DAY
        tr_all = raw_feat.filter((pl.col("close_time") < ms) & (pl.col("close_time") >= lo))
        c1 = int(tr_all.height * (1 - ES_FRAC - THR_FRAC))
        ref = tr_all[:c1]
        pre = raw_feat.filter((pl.col("close_time") >= ms - PRE_DAYS * MS_DAY)
                              & (pl.col("close_time") < ms))
        full = raw_feat.filter((pl.col("close_time") >= ms) & (pl.col("close_time") < me))
        if min(ref.height, pre.height, full.height) < 1000:
            log(f"[{m}] 样本不足，跳过 ({ref.height}/{pre.height}/{full.height})")
            continue
        a_ref = {f: ref[f].to_numpy() for f in feats}
        a_pre = {f: pre[f].to_numpy() for f in feats}
        a_full = {f: full[f].to_numpy() for f in feats}
        s_ref = {f: summary(a_ref[f]) for f in feats}
        row_pre, row_full = {}, {}
        for f in feats:
            row_pre[f] = drift(s_ref[f], summary(a_pre[f]), a_ref[f], a_pre[f])
            row_full[f] = drift(s_ref[f], summary(a_full[f]), a_ref[f], a_full[f])
        pre_rows.append(dict(ym=m, win=MONTHLY_WIN[m], drift=row_pre))
        full_rows.append(dict(ym=m, win=MONTHLY_WIN[m], drift=row_full))
        log(f"[{m}] 胜率={MONTHLY_WIN[m]:.2f}% ref={ref.height:,} pre={pre.height:,} full={full.height:,}")

    rng = np.random.default_rng(20261122)
    results = []
    for f in feats:
        for metric in ("mean_z", "median_z", "std_ratio", "psi"):
            v = np.array([r["drift"][f][metric] for r in pre_rows], dtype=float)
            if np.count_nonzero(np.isfinite(v)) < 6:
                continue
            rho = spearman(v, wins)
            p = perm_p(v, wins, rho, rng)
            results.append(dict(feature=f, metric=metric, rho=rho, p=p,
                                values=[None if not np.isfinite(x) else round(float(x), 5)
                                        for x in v]))
    qs = bh_fdr(np.array([r["p"] for r in results]))
    for r, q in zip(results, qs):
        r["q"] = None if not np.isfinite(q) else float(q)
    results.sort(key=lambda x: (x["p"], x["feature"]))

    report = dict(config=dict(pre_days=PRE_DAYS, win_days=WIN_DAYS, n_perm=N_PERM,
                              label_source="kaggle h10sig/A/B/C monthly logs",
                              months=months, wins=wins.tolist()),
                  n_features=len(feats), results=results,
                  pre_rows=pre_rows, full_rows=full_rows,
                  win_by_config=WIN_BY_CONFIG)
    (WORK / "h10_drift_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + "=" * 92)
    print(f"全特征漂移扫描：{len(feats)} 列 × 4 指标 × {len(months)} 月，置换 {N_PERM} 次，BH-FDR")
    print("feature metric rho p q | 按月份顺序的漂移值")
    for r in results[:20]:
        print(f"{r['feature']:<18}{r['metric']:<10}{r['rho']:+.2f} {r['p']:.4f} "
              f"{r['q'] if r['q'] is not None else float('nan'):.3f} | "
              + " ".join("—" if v is None else f"{v:+.2f}" for v in r["values"][:6]))
    sig = [r for r in results if r["q"] is not None and r["q"] < 0.10]
    print(f"FDR<10% 的特征数：{len(sig)}（ exploratory only，不能直接上线）")
    print(f"报告：{WORK}/h10_drift_report.json")
    print("=" * 92)


if __name__ == "__main__":
    main()
