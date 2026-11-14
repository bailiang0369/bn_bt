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

horizon 阶梯（2026-11-14 加）：`HORIZONS=3,5,10,15` 一次跑完，同口径同 harness。
用途是 §29.5 的落地决策——现役信号是 h=3，但币安最短只有 10m（赔率 0.8，平衡线
55.56%），必须先测 h=10 在固定阈值口径下的胜率，才知道币安能不能当主力平台。
SETS=base 时每一档都是「184 列纯订单簿」的纯 horizon 效应，不含特征干扰。

env: SEEDS / ROUNDS / TPD / SETS / HORIZON(单值) / HORIZONS(逗号列表，优先)
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
# 逗号列表一次跑多档 horizon（同一次运行、同一份数据，避免跨 run 的 seed/数据漂移）
HORIZONS = [int(x) for x in os.environ.get("HORIZONS", str(HORIZON)).split(",") if x.strip()]
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


def run_one(w0: pl.DataFrame, cols0: list[str], close: np.ndarray, ct: np.ndarray,
            mode: str, hz: int) -> dict | None:
    """单组（特征集 × horizon）：训练 5 seed、val 定阈值、test 评估。"""
    w, cols = build_feats(w0, cols0, mode)
    n = w.height - hz
    y = (close[hz:] > close[:-hz]).astype(np.int8)
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
    del X, w
    yva, yte = y[tr1:va1], y[va1:n]
    auc_v = float(roc_auc_score(yva, pv))
    auc_t = float(roc_auc_score(yte, pt))
    thr = float(np.quantile(np.abs(pv - .5), QUANT))
    s = pt - .5
    hit = np.abs(s) >= thr
    nb = int(hit.sum())
    if nb < 100:
        log(f"  [{mode}@h{hz}] 单量太少 {nb}")
        return None
    ww = ((s[hit] > 0).astype(int) == yte[hit]).astype(np.int8)
    win = float(ww.mean())
    dd = days[va1:n][hit]
    mons = [float(ww[dd == u].mean()) for u in np.unique(dd) if (dd == u).sum() >= 50]
    nd_te = len(np.unique(days[va1:n]))
    return dict(tag=f"{mode}@h{hz}", mode=mode, hz=hz, cols=len(cols), it=int(np.mean(its)),
                auc_v=auc_v, auc_t=auc_t, per_day=nb / nd_te, win=win,
                ev=PAYOUT * win - (1 - win), monmin=min(mons) if mons else win)


def main() -> None:
    ob = ok.load_ob().sort("close_time")
    cols0 = [c for c in ob.columns if c not in ("close_time", "target", "close")]
    kf, _ = ok.load_kline_features(keep_raw=True)
    w0 = (ob.join(kf.select("close_time", "close"), on="close_time", how="left")
          .sort("close_time").filter(pl.col("close").is_not_null()))
    close = w0["close"].to_numpy().astype(np.float64)
    ct = w0["close_time"].to_numpy()
    log(f"订单簿 {ob.height:,} 行 × {len(cols0)} 特征 | 带 close {w0.height:,} 行 | "
        f"horizons={HORIZONS} sets={SETS} seeds={len(SEEDS)}")

    hdr = (f"{'特征集@h':>12}{'列数':>7}{'轮数':>6}{'AUCval':>8}{'AUCtest':>9}"
           f"{'单/天':>7}{'胜率':>8}{'EV':>8}{'月min':>8}")
    print("\n" + hdr)
    rows = []
    for hz in HORIZONS:
        for mode in SETS:
            r = run_one(w0, cols0, close, ct, mode, hz)
            if r is None:
                continue
            rows.append(r)
            print(f"{r['tag']:>12}{r['cols']:>7}{r['it']:>6}{r['auc_v']:>8.4f}"
                  f"{r['auc_t']:>9.4f}{r['per_day']:>7.1f}{r['win'] * 100:>7.2f}%"
                  f"{r['ev']:>+8.3f}{r['monmin'] * 100:>7.2f}%")

    # 同一 horizon 内比特征集（base 为基准）；跨 horizon 只看 base 的阶梯（平台选择用）
    for hz in HORIZONS:
        same = [r for r in rows if r["hz"] == hz]
        b = [r for r in same if r["mode"] == "base"]
        if not b:
            continue
        best = max(same, key=lambda r: r["auc_t"])
        print(f"[h={hz}] 最优 {best['tag']} cols={best['cols']} AUC={best['auc_t']:.4f} "
              f"胜率={best['win'] * 100:.2f}% → "
              f"ΔAUC={best['auc_t'] - b[0]['auc_t']:+.4f} "
              f"Δ胜率={(best['win'] - b[0]['win']) * 100:+.2f}pp")
    lad = sorted([r for r in rows if r["mode"] == "base"], key=lambda r: r["hz"])
    if len(lad) > 1:
        print("\n[horizon 阶梯] 184 列纯订单簿、val 定阈值、test 评估；平衡线 0.808→55.31% / 0.80→55.56%")
        print(f"{'h':>4}{'AUCval':>8}{'AUCtest':>9}{'单/天':>7}{'胜率':>8}"
              f"{'EV@0.808':>10}{'EV@0.80':>9}{'月min':>8}")
        for r in lad:
            print(f"{r['hz']:>4}{r['auc_v']:>8.4f}{r['auc_t']:>9.4f}{r['per_day']:>7.1f}"
                  f"{r['win'] * 100:>7.2f}%{r['ev']:>+10.3f}"
                  f"{0.8 * r['win'] - (1 - r['win']):>+9.3f}{r['monmin'] * 100:>7.2f}%")
    log("DONE")


if __name__ == "__main__":
    main()
