#!/usr/bin/env python3
"""h=3 提 AUC：给订单簿特征加时序动态（差分 / 滚动），验证是否突破 AUC 0.543。

动机（2026-11-14）：现有 184 列订单簿特征是 **t 时刻的 1 分钟截面**，
`ob_amp_obi_030` 之类是振幅档而非时间窗。预测 3 分钟方向却只喂单个快照，
盘口的动态（挂单增减、撤单、队列位移）完全没进模型。

已排除的其它路（勿重复）：现货基差、跨品种 ETH、OI（项目实测 |IC|<=0.025）、
aggTrades（末窗 IC 单调减弱）、模型复杂度（24 组 AUC 极差 0.002）、
标签净化（val 涨 test 不动）、val 并入训练（AUC +0.0008）、滚动分位阈值（全崩）。

特征集：
  base       184 列原始快照（对照）
  d1         base + lag1 差分（184 列）→ 368
  d1m3       base + lag1 差分 + 3 分钟滚动均值（+368）→ 552
  d1m3s3     base + lag1 差分 + 滚动均值 + 滚动标准差 → 736

口径同现役：64/16/20、5 seeds、val 早停、|p-0.5| 在 val 的 98.958 分位选单、
赔率 0.808。判据：AUC(test) 与 AUC(val) 必须同向提升。

env: SEEDS / ROUNDS / TPD / SETS
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

HORIZON = int(os.environ.get("HORIZON", "3"))
SEEDS = [int(s) for s in os.environ.get("SEEDS", "42,43,44,45,46").split(",") if s.strip()]
ROUNDS = int(os.environ.get("ROUNDS", "2000"))
TPD = int(os.environ.get("TPD", "15"))
TRAIN_FRAC, VAL_FRAC = 0.64, 0.16
PAYOUT = 0.808
BARS_PER_DAY = 1440
QUANT = 1 - TPD / BARS_PER_DAY
SETS = [s.strip() for s in os.environ.get("SETS", "base,d1,d1m3").split(",") if s.strip()]
T = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T:7.1f}s] {msg}", flush=True)


def params(seed: int) -> dict:
    return dict(objective="binary", metric="auc", learning_rate=0.05, num_leaves=63,
                feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1,
                min_child_samples=100, verbose=-1,
                seed=seed, bagging_seed=seed, feature_fraction_seed=seed,
                data_random_seed=seed)


def build_feats(w: pl.DataFrame, cols: list[str], mode: str) -> tuple[pl.DataFrame, list[str]]:
    """按 mode 追加时序动态列。只用 past（shift/rolling 均为向后看）。"""
    if mode == "base":
        return w, cols
    ex = []
    for c in cols:
        s = pl.col(c).cast(pl.Float32)
        if mode in ("d1", "d1m3", "d1m3s3"):
            ex.append((s - s.shift(1)).alias(f"{c}__d1"))
        if mode in ("d1m3", "d1m3s3"):
            ex.append(s.rolling_mean(3, min_samples=2).alias(f"{c}__m3"))
        if mode == "d1m3s3":
            ex.append(s.rolling_std(3, min_samples=2).alias(f"{c}__s3"))
    w = w.with_columns(ex)
    new = [c for c in w.columns if "__" in c]
    return w, cols + new


def main() -> None:
    ob = ok.load_ob().sort("close_time")
    cols0 = [c for c in ob.columns if c not in ("close_time", "target", "close")]
    kf, _ = ok.load_kline_features(keep_raw=True)
    w0 = (ob.join(kf.select("close_time", "close"), on="close_time", how="left")
          .sort("close_time").filter(pl.col("close").is_not_null()))
    close = w0["close"].to_numpy().astype(np.float64)
    ct = w0["close_time"].to_numpy()
    h = HORIZON
    n0 = w0.height
    log(f"订单簿 {ob.height:,} 行 × {len(cols0)} 特征 | 带 close {n0:,} 行")

    hdr = (f"{'特征集':>10}{'列数':>7}{'轮数':>6}{'AUCval':>8}{'AUCtest':>9}"
           f"{'单/天':>7}{'胜率':>8}{'EV':>8}{'月min':>8}")
    print("\n" + hdr)
    rows = []
    for mode in SETS:
        w, cols = build_feats(w0, cols0, mode)
        n = w.height - h
        y = (close[h:] > close[:-h]).astype(np.int8)
        days = ct[:n] // 86_400_000
        X = w.select(cols).to_numpy()[:n].astype(np.float32)
        X[~np.isfinite(X)] = np.nan
        tr1, va1 = int(n * TRAIN_FRAC), int(n * (TRAIN_FRAC + VAL_FRAC))
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
        yva, yte = y[tr1:va1], y[va1:n]
        auc_v = float(roc_auc_score(yva, pv))
        auc_t = float(roc_auc_score(yte, pt))
        thr = float(np.quantile(np.abs(pv - .5), QUANT))
        s = pt - .5
        hit = np.abs(s) >= thr
        nb = int(hit.sum())
        nd_te = len(np.unique(days[va1:n]))
        if nb < 100:
            log(f"  [{mode}] 单量太少 {nb}")
            continue
        ww = ((s[hit] > 0).astype(int) == yte[hit]).astype(np.int8)
        win = float(ww.mean())
        dd = days[va1:n][hit]
        mons = [float(ww[dd == u].mean()) for u in np.unique(dd) if (dd == u).sum() >= 50]
        print(f"{mode:>10}{len(cols):>7}{int(np.mean(its)):>6}{auc_v:>8.4f}"
              f"{auc_t:>9.4f}{nb / nd_te:>7.1f}{win * 100:>7.2f}%"
              f"{PAYOUT * win - (1 - win):>+8.3f}"
              f"{(min(mons) if mons else win) * 100:>7.2f}%")
        rows.append((mode, len(cols), auc_v, auc_t, win, nb / nd_te))
        del X, w
    if rows:
        b = [r for r in rows if r[0] == "base"]
        if b:
            best = max(rows, key=lambda r: r[3])
            print(f"\n[最优 AUC(test)] {best[0]} cols={best[1]} AUC={best[3]:.4f} "
                  f"胜率={best[4] * 100:.2f}% 单/天={best[5]:.1f} → "
                  f"ΔAUC={best[3] - b[0][3]:+.4f} Δ胜率={(best[4] - b[0][4]) * 100:+.2f}pp")
    log("DONE")


if __name__ == "__main__":
    main()
