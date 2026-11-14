#!/usr/bin/env python3
"""h=3 提 AUC 扫描：模型复杂度 + 标签净化。

动机（2026-11-14）：A/B 实验暴露「早停轮数只有 24~49、均值 36 棵」——184 个
订单簿特征 × 55 万样本，模型只用 36 棵树就停止改善。要么信号就这么多，
要么 `num_leaves=63 / min_child_samples=100` 的配置在 36 棵树后就开始拟合噪声。
文档里「AUC 0.54 是 1m bar 物理上限」的判断从未在这个维度上验证过。

两组变量：
1. 复杂度/正则：num_leaves × min_child_samples × learning_rate
2. 标签净化：3 分钟收益 |ret| < EPS 的样本本质随机（价格几乎没动），
   训练与早停时剔除，预测仍用全量。EPS ∈ {0, 0.5bp, 1bp}

固定口径：切分 64/16/20（与现役一致）、5m→3m 不适用、AUC 报 val(样本外) 与
test、选单用 |p-0.5| 在 val 的 98.958 分位（= 日均 15 单）、赔率 0.808。

env: SEEDS(粗扫用 42,43,44) / ROUNDS / EPS_LIST / COMBOS
"""
from __future__ import annotations

import itertools
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
SEEDS = [int(s) for s in os.environ.get("SEEDS", "42,43,44").split(",") if s.strip()]
ROUNDS = int(os.environ.get("ROUNDS", "2000"))
TPD = int(os.environ.get("TPD", "15"))
TRAIN_FRAC, VAL_FRAC = 0.64, 0.16
PAYOUT = 0.808
BARS_PER_DAY = 1440
QUANT = 1 - TPD / BARS_PER_DAY
EPS_LIST = [float(x) for x in os.environ.get("EPS_LIST", "0,0.00005,0.0001").split(",")]
T = time.time()

# (num_leaves, min_child_samples, learning_rate)
COMBOS = [
    (63, 100, 0.05),     # 现役
    (31, 100, 0.05),
    (15, 100, 0.05),
    (31, 500, 0.05),
    (15, 500, 0.05),
    (15, 2000, 0.05),
    (31, 500, 0.02),
    (15, 1000, 0.02),
]


def log(msg: str) -> None:
    print(f"[{time.time() - T:7.1f}s] {msg}", flush=True)


def params(seed: int, nl: int, mcs: int, lr: float) -> dict:
    return dict(objective="binary", metric="auc", learning_rate=lr, num_leaves=nl,
                feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1,
                min_child_samples=mcs, verbose=-1,
                seed=seed, bagging_seed=seed, feature_fraction_seed=seed,
                data_random_seed=seed)


def main() -> None:
    ob = ok.load_ob().sort("close_time")
    cols = [c for c in ob.columns if c not in ("close_time", "target", "close")]
    kf, _ = ok.load_kline_features(keep_raw=True)
    w = (ob.join(kf.select("close_time", "close"), on="close_time", how="left")
         .sort("close_time").filter(pl.col("close").is_not_null()))
    n0 = w.height
    close = w["close"].to_numpy().astype(np.float64)
    ct = w["close_time"].to_numpy()
    h = HORIZON
    n = n0 - h
    y = (close[h:] > close[:-h]).astype(np.int8)
    fwd = np.zeros(n)
    fwd[:] = close[h:] / close[:n] - 1.0
    days = ct[:n] // 86_400_000
    X = w.select(cols).to_numpy()[:n].astype(np.float32)
    X[~np.isfinite(X)] = np.nan
    tr1, va1 = int(n * TRAIN_FRAC), int(n * (TRAIN_FRAC + VAL_FRAC))
    te = slice(va1, n)
    va = slice(tr1, va1)
    nd_te = len(np.unique(days[te]))
    log(f"数据 {n:,} 行 × {len(cols)} 特征 | train={tr1:,} val={va1 - tr1:,} "
        f"test={n - va1:,} ({nd_te} 天) | 目标上涨率 {y.mean() * 100:.2f}%")

    hdr = (f"{'组合':>16}{'eps':>7}{'轮数':>6}{'AUCval':>8}{'AUCtest':>9}"
           f"{'单/天':>7}{'胜率':>8}{'EV':>8}{'月min':>8}")
    print("\n" + hdr)
    rows = []
    for (nl, mcs, lr), eps in itertools.product(COMBOS, EPS_LIST):
        m_tr = np.abs(fwd[:tr1]) >= eps
        m_va = np.abs(fwd[tr1:va1]) >= eps
        pv = np.zeros(va1 - tr1)
        pt = np.zeros(n - va1)
        its = []
        for sd in SEEDS:
            prm = params(sd, nl, mcs, lr)
            ds = lgb.Dataset(X[:tr1][m_tr], y[:tr1][m_tr], feature_name=cols)
            vs = lgb.Dataset(X[tr1:va1][m_va], y[tr1:va1][m_va], reference=ds)
            m = lgb.train(prm, ds, num_boost_round=ROUNDS, valid_sets=[ds, vs],
                          callbacks=[lgb.early_stopping(50, verbose=False)])
            it = int(m.best_iteration)
            pv += m.predict(X[tr1:va1], num_iteration=it) / len(SEEDS)
            pt += m.predict(X[va1:n], num_iteration=it) / len(SEEDS)
            its.append(it)
        auc_v = float(roc_auc_score(y[va][m_va], pv[m_va]))
        auc_t = float(roc_auc_score(y[te], pt))
        thr = float(np.quantile(np.abs(pv[m_va] - .5), QUANT))
        s = pt - .5
        hit = np.abs(s) >= thr
        nbet = int(hit.sum())
        if nbet < 100:
            continue
        ww = ((s[hit] > 0).astype(int) == y[te][hit]).astype(np.int8)
        win = float(ww.mean())
        dd = days[te][hit]
        mons = [float(ww[dd == u].mean()) for u in np.unique(dd) if (dd == u).sum() >= 50]
        tag = f"nl{nl}/m{mcs}/{lr}"
        print(f"{tag:>16}{eps * 1e4:>7.1f}{int(np.mean(its)):>6}{auc_v:>8.4f}"
              f"{auc_t:>9.4f}{nbet / nd_te:>7.1f}{win * 100:>7.2f}%"
              f"{PAYOUT * win - (1 - win):>+8.3f}"
              f"{(min(mons) if mons else win) * 100:>7.2f}%")
        rows.append((tag, eps, auc_t, win, nbet / nd_te))

    if rows:
        base = [r for r in rows if r[0] == "nl63/m100/0.05" and r[1] == 0.0]
        b = base[0] if base else None
        best = max(rows, key=lambda r: r[2])
        print(f"\n[最优 AUC(test)] {best[0]} eps={best[1] * 1e4:.1f}bp "
              f"AUC={best[2]:.4f} 胜率={best[3] * 100:.2f}% 单/天={best[4]:.1f}")
        if b:
            print(f"[现役基线] AUC={b[2]:.4f} 胜率={b[3] * 100:.2f}% → "
                  f"ΔAUC={best[2] - b[2]:+.4f} Δ胜率={(best[3] - b[3]) * 100:+.2f}pp")
    log("DONE")


if __name__ == "__main__":
    main()
