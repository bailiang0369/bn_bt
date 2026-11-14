#!/usr/bin/env python3
"""h=3 A/B：把验证集并入训练集重训，是否提升 AUC / 同单量胜率。

背景（2026-11-14）：现役 64/16/20 切分，val 那 16% 只用于早停和定阈值就丢弃，
训练样本偏少。h=3 信噪比极低（AUC≈0.54），样本量对结果敏感。

A（对照，= 现役口径）
    train = [0, 64%)  val = [64%, 80%)  test = [80%, 100%)
    早停用 val，按 best_iteration 预测 test
B（改法）
    train = [0, 80%)（= train + val 合并），轮数固定 = A 各 seed 早停轮数的均值
    预测 test

阈值（保证单量可比）：两者都取 |p-0.5| 在**各自 val 段**的同一分位
（1 - TPD/1440，TPD=15 → 98.958%）。注意 B 的 val 已进训练集，阈值是样本内
取的、对 B 有利；若 B 仍无增益，结论更硬。

评估：AUC(test)、@日均 TPD 单的胜率 / EV(@0.808) / 最差月 / 100 单滚动最差。

env: HORIZON / SEEDS / ROUNDS / TPD
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
TRAIN_FRAC = 0.64
VAL_FRAC = 0.16
PAYOUT = 0.808
BARS_PER_DAY = 1440
QUANT = 1 - TPD / BARS_PER_DAY
T = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T:7.1f}s] {msg}", flush=True)


def params(seed: int) -> dict:
    return dict(objective="binary", metric="auc", learning_rate=0.05, num_leaves=63,
                feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1,
                min_child_samples=100, verbose=-1,
                seed=seed, bagging_seed=seed, feature_fraction_seed=seed,
                data_random_seed=seed)


def evaluate(p, y, days, thr, tag):
    s = p - 0.5
    hit = np.abs(s) >= thr
    n = int(hit.sum())
    if n < 100:
        log(f"  {tag}: 单量太少 n={n}")
        return None
    ww = ((s[hit] > 0).astype(int) == y[hit]).astype(np.int8)
    win = float(ww.mean())
    nd = len(np.unique(days))
    dd = days[hit]
    mons = [float(ww[dd == u].mean()) for u in np.unique(dd) if (dd == u).sum() >= 50]
    roll = (float(np.convolve(ww, np.ones(100) / 100, "valid").min())
            if len(ww) >= 100 else win)
    return dict(n=n, per_day=n / max(nd, 1), win=win,
                ev=PAYOUT * win - (1 - win),
                mn=(min(mons) if mons else win), roll=roll,
                auc=float(roc_auc_score(y, s + .5)))


def main() -> None:
    ob = ok.load_ob().sort("close_time")
    cols = [c for c in ob.columns if c not in ("close_time", "target", "close")]
    kf, _ = ok.load_kline_features(keep_raw=True)
    w = (ob.join(kf.select("close_time", "close"), on="close_time", how="left")
         .sort("close_time").filter(pl.col("close").is_not_null()))
    n0 = w.height
    log(f"订单簿 {ob.height:,} 行 → 带 close {n0:,} 行 | 特征 {len(cols)} 列")

    close = w["close"].to_numpy().astype(np.float64)
    ct = w["close_time"].to_numpy()
    h = HORIZON
    n = n0 - h
    y = (close[h:] > close[:-h]).astype(np.int8)
    days = ct[:n] // 86_400_000
    X = w.select(cols).to_numpy()[:n].astype(np.float32)
    X[~np.isfinite(X)] = np.nan
    log(f"可用 {n:,} 行 | 目标上涨率 {y.mean() * 100:.2f}%")

    tr1 = int(n * TRAIN_FRAC)
    va1 = int(n * (TRAIN_FRAC + VAL_FRAC))
    log(f"切分 A: train[0:{tr1:,}] val[{tr1:,}:{va1:,}] test[{va1:,}:{n:,}]")

    # ---------------- A：现役口径
    pA = np.zeros(n, dtype=np.float64)
    itA = []
    for sd in SEEDS:
        ds = lgb.Dataset(X[:tr1], y[:tr1], feature_name=cols)
        vs = lgb.Dataset(X[tr1:va1], y[tr1:va1], reference=ds)
        m = lgb.train(params(sd), ds, num_boost_round=ROUNDS, valid_sets=[ds, vs],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        pA += m.predict(X, num_iteration=m.best_iteration) / len(SEEDS)
        itA.append(int(m.best_iteration))
        log(f"  [A] seed={sd} best_iter={m.best_iteration}")
    log(f"[A] 早停轮数 {itA} 均值 {np.mean(itA):.0f}")

    # ---------------- B：train+val 重训，轮数固定
    nB = max(10, int(round(float(np.mean(itA)))))
    pB = np.zeros(n, dtype=np.float64)
    for sd in SEEDS:
        ds = lgb.Dataset(X[:va1], y[:va1], feature_name=cols)
        m = lgb.train(params(sd), ds, num_boost_round=nB)
        pB += m.predict(X) / len(SEEDS)
        log(f"  [B] seed={sd} rounds={nB}")
    log(f"[B] 训练集 {va1:,} 行（+{va1 - tr1:,}）轮数固定 {nB}")

    thrA = float(np.quantile(np.abs(pA[tr1:va1] - .5), QUANT))
    thrB = float(np.quantile(np.abs(pB[tr1:va1] - .5), QUANT))
    log(f"阈值(同分位 {QUANT:.5f}) A={thrA:.5f} B={thrB:.5f}")

    te = slice(va1, n)
    va = slice(tr1, va1)
    log(f"[A] AUC val={roc_auc_score(y[va], pA[va]):.4f}(样本外) "
        f"test={roc_auc_score(y[te], pA[te]):.4f}")
    log(f"[B] AUC val={roc_auc_score(y[va], pB[va]):.4f}(样本内,不可比) "
        f"test={roc_auc_score(y[te], pB[te]):.4f}")

    rA = evaluate(pA[te], y[te], days[te], thrA, "A")
    rB = evaluate(pB[te], y[te], days[te], thrB, "B")
    hdr = (f"{'方案':>6}{'单量':>8}{'单/天':>8}{'胜率':>9}{'EV':>8}"
           f"{'月min':>8}{'100单滚动min':>13}{'AUC':>8}")
    print("\n" + hdr)
    for name, r in (("A 现役", rA), ("B 重训", rB)):
        if not r:
            continue
        print(f"{name:>6}{r['n']:>8,}{r['per_day']:>8.1f}{r['win'] * 100:>8.2f}%"
              f"{r['ev']:>+8.3f}{r['mn'] * 100:>7.2f}%{r['roll'] * 100:>12.2f}%"
              f"{r['auc']:>8.4f}")
    if rA and rB:
        print(f"\n[B−A] 胜率 {(rB['win'] - rA['win']) * 100:+.2f}pp | "
              f"EV {rB['ev'] - rA['ev']:+.3f} | 月min {(rB['mn'] - rA['mn']) * 100:+.2f}pp | "
              f"单/天 {rB['per_day'] - rA['per_day']:+.1f}")
    log("DONE")


if __name__ == "__main__":
    main()
