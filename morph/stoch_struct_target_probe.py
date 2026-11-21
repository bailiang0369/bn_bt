#!/usr/bin/env python3
"""Stoch 确认点 → 下一段属性 的预测力诊断（不交易、不对撞 h10）。

回答用户 2026-11-21 的问题：结构特征不该押 t+10 方向，而该在**每个确认点**预测
"到达下一个确认点所需的持续时间 / 能达到的最高最低点 / 振幅"。
本 kernel 只诊断预测力，对标 §21 的 MACD 段长 log-L 回归 ρ=0.280：

  目标1  duration   = log(下一确认时刻 − 当前确认时刻)，回归 Spearman ρ（对标 §21）
  目标2  amplitude  = |下一极值 − 当前价| / 当前价 × 1e4，回归 Spearman ρ
  目标3  reach_dir  = 下一极值在当前价上方=1 / 下方=0，二分类 AUC

口径：train = 2020-01~2025-12 的确认点，test = 2026-01~10；只用确认时刻已有的特征。
 LightGBM 固定小配置，单调对照，不做超参搜索。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from morph import kline_long as kl  # noqa: E402
from morph.stoch_struct import FEAT_NAMES, build_struct, log, struct_leak_check  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

HORIZON = int(os.environ.get("HORIZON", "10"))
K_PERIOD = int(os.environ.get("K_PERIOD", "80"))
TEST_START = int(os.environ.get("TEST_START", "2026-01"))


def fit_predict(Xtr, ytr, Xte, kind):
    p = dict(objective="regression" if kind == "reg" else "binary",
             metric="l2" if kind == "reg" else "auc",
             learning_rate=.05, num_leaves=15, min_child_samples=100,
             feature_fraction=.8, bagging_fraction=.8, bagging_freq=1,
             verbose=-1, num_threads=os.cpu_count() or 4, seed=42)
    m = lgb.train(p, lgb.Dataset(Xtr, ytr), num_boost_round=300)
    return m.predict(Xte)


def score(name, yte, pred, kind):
    if kind == "reg":
        rho = float(spearmanr(yte, pred).statistic)
        print(f"  {name:<12} Spearman ρ={rho:+.4f}  (§21 MACD 段长基线 ρ=0.280)")
    else:
        a = float(roc_auc_score(yte, pred))
        print(f"  {name:<12} AUC={a:.4f}  |AUC-0.5|={abs(a-0.5):.4f}")


def main() -> None:
    if os.environ.get("SC_SELFCHECK", "1") == "1":
        if not struct_leak_check(k_period=K_PERIOD):
            raise SystemExit("[sc-target] 泄露自检失败")

    raw = kl.load_raw()
    X, names, (ptype, pidx, px) = build_struct(raw, k_period=K_PERIOD, return_points=True)
    ct = raw["close_time"].to_numpy()
    close = raw["close"].to_numpy().astype(np.float64)
    high = raw["high"].to_numpy().astype(np.float64)
    low = raw["low"].to_numpy().astype(np.float64)
    n = len(ct)

    # 段边界 = ptype 交替（峰→谷 / 谷→峰）；同类极值替换不算新段
    ev = np.where((np.diff(ptype) != 0))[0] + 1
    ev = ev[pidx[ev] >= 0]
    log(f"段边界（ptype 交替）{len(ev):,} 个 | 平均间隔 {n / max(len(ev), 1):.0f} 根")

    # 逐事件构造目标：用"确认时刻 t0→t1"（段是两次确认之间的时间），而非极值点位置
    rows = []
    for j in range(len(ev) - 1):
        t0, t1 = int(ev[j]), int(ev[j + 1])
        next_px = float(px[t1])     # 下一确认点的极值价位
        c0 = close[t0]
        test_flag = ct[t0] >= int(np.datetime64(TEST_START + "-01", "D").astype("datetime64[ms]").astype(np.int64))
        rows.append(dict(
            t=t0, test=test_flag,
            duration=np.log(max(1, t1 - t0)),                       # log 段长（对标 §21）
            amplitude=abs(next_px - c0) / max(c0, 1e-9) * 1e4,      # 到下一极值的振幅 bp
            reach_dir=1.0 if (next_px > c0) else 0.0,               # 下一极值在现价上方=1
        ))
    d = pl.DataFrame(rows)
    E = d.height
    Te = int(d["test"].sum())
    log(f"可用事件对 {E:,} | train {E - Te:,} / test {Te:,}")

    Xev = X[ev[:E]]
    tr = ~d["test"].to_numpy()
    te = d["test"].to_numpy()

    targets = [("duration", "reg"), ("amplitude", "reg"), ("reach_dir", "cls")]
    print("\n[确认点 → 下一段属性] test 预测力：")
    for name, kind in targets:
        y = d[name].to_numpy().astype(np.float64)
        if len(np.unique(y[te])) < 2:
            print(f"  {name:<12} test 单值，跳过"); continue
        pred = fit_predict(Xev[tr], y[tr], Xev[te], kind)
        score(name, y[te], pred, kind)

    # 朴素基线：用历史均值/多数类，给出"可预测"的地板
    print("\n[基线] 常数预测：")
    for name, kind in targets:
        y = d[name].to_numpy().astype(np.float64)
        yhat = np.full(te.sum(), y[tr].mean())
        if kind == "reg":
            rho = float(spearmanr(y[te], yhat).statistic) if np.std(yhat) > 0 else 0.0
            print(f"  {name:<12} 常数ρ≈{rho:.3f}（train均值 {y[tr].mean():.3f}）")
        else:
            print(f"  {name:<12} train正类占比 {y[tr].mean():.3f}（AUC 应≈0.5）")
    log("DONE")


if __name__ == "__main__":
    main()
