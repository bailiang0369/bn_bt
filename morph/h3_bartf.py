#!/usr/bin/env python3
"""bar 尺度 × 特征域 的等单量对照（h=10 复刻 + 隔离实验）。

动机（2026-11-15）
------------------
外部 notebook（tmp/new_b.ipynb）报 h=10 约 58.3%@12.7 单/天，我们订单簿 1m bar h=10
只有 53.68%@48.6 单/天，差 4.6pp。它的实现有三处不能采信（阈值取 test 分位、9×9 网格
在 test 上挑最优、三分类 barrier 多除了一次 close 导致中间类只剩 6%），但它有一个我们
从来没用过、且物理上说得通的差异：**bar 尺度**。

h=10 用 1m bar 是「预测 10 步」，相邻样本的标签重叠 10 根，特征还停在 1 分钟信噪比上；
用 5m bar 只有「预测 2 步」，重叠 2 根，等于先把噪声沿时间压掉一半。我们全部历史实验
（订单簿 / K线 / 融合 / 宽表）都是 1m bar，这一维从未验证。

本文件只做一件事：同一份 1m 数据，换 bar 尺度重采样，看 h=10 的胜率是否随之上升。
纪律照我们的，不照它的：阈值只在 val 定、test 只评估、多 seed 概率平均、
输出单量档网格 + 月度分布（跨方案比较必须等单量，§18）。

env
---
  FEATS=ob|k            特征域：ob=184 列订单簿，k=63 列 K 线特征
  BAR=1                 重采样周期（分钟），1/3/5/10/15
  HORIZON_MIN=10        预测 horizon（分钟），实际步数 = HORIZON_MIN / BAR
  TPD_LIST=120,...      单量档网格（名义），每档各自在 val 定 |p-0.5| 分位阈值
  SEEDS=42,...          seed 列表（≥5 个，单 seed 的 ±1pp 判不出东西）
  ROUNDS / KLINE_FROM / SYMBOL
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from morph import ob_kline as ok  # noqa: E402

try:
    import lightgbm as lgb
except Exception as e:  # pragma: no cover
    raise SystemExit(f"lightgbm 缺失: {e}")

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

FEATS = os.environ.get("FEATS", "k").lower()
BAR = int(os.environ.get("BAR", "1"))
# 逗号列表：同一份 1m 数据只加载一次，按不同 bar 尺度重采样（重采样本身几乎不耗时）
BARS = [int(x) for x in os.environ.get("BARS", str(BAR)).split(",") if x.strip()]
HORIZON_MIN = int(os.environ.get("HORIZON_MIN", "10"))
TPD_LIST = [float(x) for x in os.environ.get("TPD_LIST", "120,60,30,15,10,5,2").split(",") if x.strip()]
SEEDS = [int(s) for s in os.environ.get("SEEDS", "42,43,44,45,46").split(",") if s.strip()]
ROUNDS = int(os.environ.get("ROUNDS", "2000"))
KLINE_FROM = int(os.environ.get("KLINE_FROM", "2020"))
TRAIN_FRAC, VAL_FRAC = 0.64, 0.16
PAYOUT = 0.808
BARS_PER_DAY = 1440 // BAR
BAR_TAG = ""        # 打印前缀，main 里按当前 (FEATS, bar) 设定
T = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T:7.1f}s] {msg}", flush=True)


def params(seed: int) -> dict:
    return dict(objective="binary", metric="auc", learning_rate=0.05, num_leaves=63,
                feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1,
                min_child_samples=100, verbose=-1,
                seed=seed, bagging_seed=seed, feature_fraction_seed=seed,
                data_random_seed=seed)


def resample(df: pl.DataFrame, by: int, cols: list[str]) -> pl.DataFrame:
    """1m -> by 分钟。特征取桶内最后一个 1m 快照（与外部 notebook 的 last 口径一致）。"""
    if by == 1:
        return df
    agg = [pl.col(c).last() for c in cols] + [
        pl.col("close").last(), pl.col("close_time").last()]
    return (df.with_columns(pl.from_epoch("close_time", time_unit="ms").alias("ts"))
              .sort("ts")
              .group_by_dynamic("ts", every=f"{by}m")
              .agg(agg)
              .drop("ts")
              .sort("close_time"))


def run_bar(raw: pl.DataFrame, cols: list[str], bar: int) -> None:
    step = HORIZON_MIN // bar
    if step * bar != HORIZON_MIN:
        log(f"  跳过 BAR={bar}：不能整除 HORIZON_MIN={HORIZON_MIN}")
        return
    tight = resample(raw, bar, cols)
    close = tight["close"].to_numpy().astype(np.float64)
    ct = tight["close_time"].to_numpy()
    n = tight.height - step
    y = (close[step:] > close[:-step]).astype(np.int8)
    days = ct[:n] // 86_400_000
    X = tight.select(cols).to_numpy()[:n].astype(np.float32)
    X[~np.isfinite(X)] = np.nan
    del tight

    tr1, va1 = int(n * TRAIN_FRAC), int(n * (TRAIN_FRAC + VAL_FRAC))
    log(f"FEATS={FEATS} BAR={bar}m step={step} 列={len(cols)} | 样本 {n:,} 行 | "
        f"seeds={len(SEEDS)} | 标签上涨占比={y.mean() * 100:.2f}%")

    pv = np.zeros(va1 - tr1)
    pt = np.zeros(n - va1)
    its = []
    for sd in SEEDS:
        ds = lgb.Dataset(X[:tr1], y[:tr1], feature_name=cols)
        vs = lgb.Dataset(X[tr1:va1], y[tr1:va1], reference=ds)
        m = lgb.train(params(sd), ds, num_boost_round=ROUNDS, valid_sets=[ds, vs],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        it = int(m.best_iteration)
        pv += m.predict(X[tr1:va1], num_iteration=it) / len(SEEDS)
        pt += m.predict(X[va1:n], num_iteration=it) / len(SEEDS)
        its.append(it)
    del X
    yva, yte = y[tr1:va1], y[va1:n]
    auc_v = float(roc_auc_score(yva, pv))
    auc_t = float(roc_auc_score(yte, pt))
    log(f"  AUC(val)={auc_v:.4f} AUC(test)={auc_t:.4f} best_iter={int(np.mean(its))}")

    s = pt - .5
    days_te = days[va1:n]
    nd_te = len(np.unique(days_te))
    for tpd in TPD_LIST:
        thr = float(np.quantile(np.abs(pv - .5), 1 - tpd / BARS_PER_DAY))
        hit = np.abs(s) >= thr
        nb = int(hit.sum())
        if nb < 50:
            continue
        ww = ((s[hit] > 0).astype(int) == yte[hit]).astype(np.int8)
        win = float(ww.mean())
        # 标签重叠 step 根：有效样本数按 n/step 折算，SE 才不虚低
        se = float(np.sqrt(win * (1 - win) / max(nb / step, 2)))
        dd = days_te[hit]
        mons = [float(ww[dd == u].mean()) for u in np.unique(dd) if (dd == u).sum() >= 50]
        print(f"{BAR_TAG} {tpd:>6.1f}{thr:>9.5f}{nb / nd_te:>7.1f}{nb:>7d}"
              f"{win * 100:>7.2f}%{se * 100:>6.2f}%{PAYOUT * win - (1 - win):>+10.3f}"
              f"{(min(mons) if mons else win) * 100:>7.2f}%", flush=True)


def main() -> None:
    global BAR_TAG
    if FEATS == "ob":
        raw = ok.load_ob().sort("close_time")
        kf, _ = ok.load_kline_features(y0=KLINE_FROM, keep_raw=True)
        raw = (raw.join(kf.select("close_time", "close"), on="close_time", how="left")
                  .filter(pl.col("close").is_not_null()).sort("close_time"))
        cols = [c for c in raw.columns if c not in ("close_time", "target", "close")]
    elif FEATS == "k":
        raw, cols = ok.load_kline_features(y0=KLINE_FROM, keep_raw=True)
    else:
        raise SystemExit(f"FEATS 只能是 ob / k，收到 {FEATS}")
    log(f"原始 1m: {raw.height:,} 行 × {len(cols)} 特征 | bars={BARS}")
    print(f"\n{'特征域/bar':>11}{'名义tpd':>8}{'阈值':>9}{'单/天':>7}{'n':>7}"
          f"{'胜率':>8}{'SE*':>7}{'EV@0.808':>10}{'月min':>8}")
    for bar in BARS:
        BAR_TAG = f"{FEATS}@{bar}m".rjust(11)
        run_bar(raw, cols, bar)
    log("DONE")


if __name__ == "__main__":
    main()
