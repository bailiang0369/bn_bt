#!/usr/bin/env python3
"""超参随机搜索：目标不是 AUC，是「目标单量档位下的 EV」（新文件）。

为什么不用 optuna（2026-11-16 决定）
------------------------------------
搜索空间 6 维，网格要 3^6=729 次不可行；随机搜索在 6 维问题上与 TPE 的差距很小，
而 Kaggle 镜像不保证有 optuna。自己写 20 行采样更可控、结果可复现（固定 RNG）。

必须纠正的方法论缺陷
--------------------
上一轮 `auc_probe` 用 test 的 AUC 选超参，属于 in-sample 选择（靠 2025 样本外救回）。
这里严格分段：
  选参段  2026-01 ~ 2026-06        只看 EV，不看 AUC
  报告段  2026-07 ~ 2026-10、2025 全年   完全不参与选参

目标函数
--------
业务关心「每天 N 单时的期望」，不是全局排序。所以每个 trial 对
tpd ∈ {12,15,18,20} 各算一遍实际单量与 EV，取「实际单量落在
TPD_LO~TPD_HI」且 EV 最高的档位作为得分；都不落进区间就按距离惩罚。
这样避免搜出「AUC 高但分数尾部极厚、单量失控」的参数——上一轮的 base
在 2025 年实际单量是目标的 1.6~3.8 倍，正是这个问题。

env
---
  HORIZON=10 WIN_DAYS=90 PAY=0.80 N_TRIALS=30 KLINE_FROM=2020
  SEARCH_SEEDS=42,43  REPORT_SEEDS=42,43,44  TPD_LO=14 TPD_HI=22
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from morph import kline_long as kl  # noqa: E402
from morph.h10_signal import add_flow  # noqa: E402
from morph.two_legs import add_target, matrix  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

import lightgbm as lgb  # noqa: E402

HORIZON = int(os.environ.get("HORIZON", "10"))
WIN_DAYS = int(os.environ.get("WIN_DAYS", "90"))
ROUNDS = int(os.environ.get("ROUNDS", "2000"))
ES_FRAC = float(os.environ.get("ES_FRAC", "0.05"))
THR_FRAC = float(os.environ.get("THR_FRAC", "0.10"))
PAY = float(os.environ.get("PAY", "0.80"))
BARS_PER_DAY = int(os.environ.get("BARS_PER_DAY", "1440"))
N_TRIALS = int(os.environ.get("N_TRIALS", "30"))
SEARCH_SEEDS = [int(s) for s in os.environ.get("SEARCH_SEEDS", "42,43").split(",") if s.strip()]
REPORT_SEEDS = [int(s) for s in os.environ.get("REPORT_SEEDS", "42,43,44").split(",") if s.strip()]
TPD_GRID = [float(x) for x in os.environ.get("TPD_GRID", "12,15,18,20").split(",") if x.strip()]
TPD_LO = float(os.environ.get("TPD_LO", "14"))
TPD_HI = float(os.environ.get("TPD_HI", "22"))
SEARCH_RANGE = tuple(os.environ.get("SEARCH_RANGE", "2026-01,2026-06").split(","))
REPORT_RANGES = [
    tuple(os.environ.get("REPORT_RANGE_1", "2026-07,2026-10").split(",")),
    tuple(os.environ.get("REPORT_RANGE_2", "2025-01,2025-12").split(",")),
]
T = time.time()

# reg2 是上一轮在 2026-01~10 上选出的，这里作为对照之一
REG2 = dict(num_leaves=15, min_child_samples=3000, feature_fraction=.5, lambda_l2=100)
DEFAULT = dict(learning_rate=.05, num_leaves=63, feature_fraction=.9,
               bagging_fraction=.9, lambda_l2=0, min_child_samples=100)


def log(msg: str) -> None:
    print(f"[{time.time() - T:7.1f}s] {msg}", flush=True)


def fit(Xtr, ytr, cols, Xes, yes, seed, extra):
    params = dict(objective="binary", metric="auc", boosting_type="gbdt",
                  bagging_freq=1, verbose=-1, seed=seed, num_threads=os.cpu_count() or 4)
    params.update(DEFAULT)
    params.update(extra)
    ds = lgb.Dataset(Xtr, ytr, feature_name=cols)
    va = lgb.Dataset(Xes, yes, reference=ds)
    return lgb.train(params, ds, num_boost_round=ROUNDS, valid_sets=[ds, va],
                     callbacks=[lgb.early_stopping(50, verbose=False)])


def sample(rng: np.random.Generator) -> dict:
    """对数均匀采样：min_child_samples 与 lambda_l2 跨数量级，必须用 log 尺度。"""
    return dict(
        num_leaves=int(rng.choice([7, 15, 31])),
        min_child_samples=int(np.exp(rng.uniform(np.log(200), np.log(8000)))),
        feature_fraction=float(rng.uniform(.30, .90)),
        lambda_l2=float(np.exp(rng.uniform(np.log(1), np.log(500)))),
        learning_rate=float(rng.uniform(.02, .08)),
        bagging_fraction=float(rng.uniform(.70, 1.0)),
    )


def fmt(p: dict) -> str:
    return (f"leaves={p['num_leaves']} mcs={p['min_child_samples']} "
            f"ff={p['feature_fraction']:.2f} l2={p['lambda_l2']:.0f} "
            f"lr={p['learning_rate']:.3f} bag={p['bagging_fraction']:.2f}")


def main() -> None:
    raw = kl.load_raw()
    df, feats = kl.build(raw)
    df, feats = add_flow(df, feats)
    df = add_target(df, HORIZON).filter(
        pl.col("target").is_not_null() & pl.col("close").is_not_null()).sort("close_time")
    df = df.with_columns(
        pl.col("close_time").cast(pl.Datetime("ms")).dt.strftime("%Y-%m").alias("ym"))
    X = matrix(df, feats)
    y_all = df["target"].to_numpy().astype(int)
    ct = df["close_time"].cast(pl.Int64).to_numpy()
    ym = df["ym"].to_numpy()
    log(f"特征 {len(feats)} 列 | 可用 {df.height:,} 行 | H={HORIZON} | 窗 {WIN_DAYS}d")

    # ---- 全部月份的切分只算一次，并把特征矩阵切片缓存下来 ----
    # 30 trials 会重复用同一批月，若每次都做 fancy indexing 会白烧一倍时间。
    all_m = sorted(set(ym.tolist()))
    SPL: dict[str, tuple] = {}
    for m in all_m:
        if not ("2024-10" <= m <= "2026-10"):
            continue
        m_start = int(dt.datetime.strptime(m + "-01", "%Y-%m-%d")
                      .replace(tzinfo=dt.timezone.utc).timestamp() * 1000)
        lo = m_start - WIN_DAYS * 86_400_000
        i_tr = np.where((ct < m_start) & (ct >= lo))[0]
        i_te = np.where(ym == m)[0]
        if len(i_tr) < 100_000 or len(i_te) < 1_000:
            continue
        h = len(i_tr)
        c1 = int(h * (1 - ES_FRAC - THR_FRAC))
        c2 = int(h * (1 - THR_FRAC))
        SPL[m] = (X[i_tr[:c1]], X[i_tr[c1:c2]], X[i_tr[c2:]], X[i_te],
                  y_all[i_tr[:c1]], y_all[i_tr[c1:c2]], y_all[i_te],
                  len(np.unique(ct[i_te] // 86_400_000)))
    log(f"可用窗口 {len(SPL)} 个月：{min(SPL)} ~ {max(SPL)}（特征切片已缓存）")

    def months_in(a: str, b: str) -> list[str]:
        return [m for m in sorted(SPL) if a <= m <= b]

    def evaluate(params: dict, mlist: list[str], seeds: list[int]) -> dict:
        """返回各 tpd 档的 (实际单量, 胜率, EV) 与 OOS AUC。"""
        per = []
        ps, ys, nd_all = [], [], 0
        for m in mlist:
            Xtr, Xes, Xth, Xte, ytr, yes, y_te, nd = SPL[m]
            p_th = np.zeros(len(Xth))
            p_te = np.zeros(len(Xte))
            for seed in seeds:
                mdl = fit(Xtr, ytr, feats, Xes, yes, seed, params)
                p_th += mdl.predict(Xth, num_iteration=mdl.best_iteration) / len(seeds)
                p_te += mdl.predict(Xte, num_iteration=mdl.best_iteration) / len(seeds)
            per.append((p_th, p_te, y_te, nd))
            ps.append(p_te)
            ys.append(y_te)
            nd_all += nd
        out = []
        for tpd in TPD_GRID:
            sv, yv, nds = [], [], 0
            for p_th, p_te, y, nd in per:
                thr = float(np.quantile(np.abs(p_th - .5), 1 - tpd / BARS_PER_DAY))
                s = p_te - .5
                hit = np.abs(s) >= thr
                if hit.sum() < 30:
                    continue
                sv.append(s[hit])
                yv.append(y[hit])
                nds += nd          # 只统计真正有单的月份，否则单/天被低估
            if not sv:
                continue
            s = np.concatenate(sv)
            y = np.concatenate(yv)
            wr = float(((s > 0).astype(int) == y).mean())
            out.append(dict(tpd=tpd, act=len(y) / max(nds, 1), wr=wr,
                            ev=PAY * wr - (1 - wr)))
        return dict(tpds=out, nd=nd_all,
                    auc=float(roc_auc_score(np.concatenate(ys), np.concatenate(ps))))

    def score(res: dict) -> tuple[float, float, bool]:
        cand = [r for r in res["tpds"] if TPD_LO <= r["act"] <= TPD_HI]
        if cand:
            b = max(cand, key=lambda r: r["ev"])
            return b["ev"], b["act"], True
        b = min(res["tpds"], key=lambda r: min(abs(r["act"] - TPD_LO), abs(r["act"] - TPD_HI)))
        return b["ev"] - .01 * abs(b["act"] - np.clip(b["act"], TPD_LO, TPD_HI)), b["act"], False

    # ---- Phase 1：选参段随机搜索 ----
    sl = months_in(*SEARCH_RANGE)
    log(f"===== 选参段 {sl[0]}~{sl[-1]}（{len(sl)} 月）| {N_TRIALS} trials "
        f"| 目标：实际单量 {TPD_LO:g}~{TPD_HI:g}/天 时最大化 EV =====")
    rng = np.random.default_rng(20261116)
    trials = []
    for i in range(N_TRIALS):
        p = sample(rng)
        res = evaluate(p, sl, SEARCH_SEEDS)
        sc, act, ok = score(res)
        trials.append(dict(params=p, score=sc, act=act, ok=ok, auc=res["auc"],
                           tpd=[(r["tpd"], r["act"], r["wr"], r["ev"]) for r in res["tpds"]]))
        log(f"  trial {i + 1:2d}/{N_TRIALS} EV={sc:+.3f} act={act:5.1f} "
            f"in={'Y' if ok else 'N'} AUC={res['auc']:.4f} | {fmt(p)}")
    trials.sort(key=lambda t: -t["score"])
    best = trials[0]
    log(f"最优 trial：EV={best['score']:+.3f} act={best['act']:.1f}/天 | {fmt(best['params'])}")

    # ---- Phase 2：报告段（完全不参与选参）----
    cands = [("base", {}), ("reg2", REG2), ("best", best["params"])]
    for a, b in REPORT_RANGES:
        ml = months_in(a, b)
        if not ml:
            continue
        log(f"===== 报告段 {ml[0]}~{ml[-1]}（{len(ml)} 月，未参与选参）=====")
        print(f"{'配置':<8}{'AUC':>9}" + "".join(f"{'tpd' + format(t, 'g'):>22}" for t in TPD_GRID))
        print(f"{'':<8}{'':>9}" + "".join(f"{'单/天  胜率    EV':>22}" for _ in TPD_GRID))
        for tag, p in cands:
            res = evaluate(p, ml, REPORT_SEEDS)
            cells = ""
            for r in res["tpds"]:
                cells += f"{r['act']:>7.1f}{r['wr'] * 100:>6.1f}%{r['ev']:>+7.3f}"
            print(f"{tag:<8}{res['auc']:>9.4f}{cells}")
    log("DONE")


def fmt(p: dict) -> str:
    return (f"leaves={p['num_leaves']} mcs={p['min_child_samples']} "
            f"ff={p['feature_fraction']:.2f} l2={p['lambda_l2']:.0f} "
            f"lr={p['learning_rate']:.3f} bag={p['bagging_fraction']:.2f}")


if __name__ == "__main__":
    main()
