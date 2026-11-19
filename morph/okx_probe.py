#!/usr/bin/env python3
"""方向 E1（跨所订单簿）：OKX 订单簿特征对币安 h=3 方向的独立预测力。

假设：OKX 盘口失衡可能领先于币安价格（跨所套利者先在 OKX 下单）。
判据（预登记）：test AUC > 0.54 且 @tpd15 胜率 > 58%（对标币安盘口腿 0.542 / 62.85%）。
达不到即关闭整个跨所方向。

数据：
  OKX 特征：bn_hisdata Release `okx-features` 或 Kaggle dataset（本轮先用 Kaggle 产物）
  币安标签：vision-base BTCUSDT 1m close，target = close[t+3] > close[t]
  对齐：两边都是 close_time（UTC 毫秒），OKX close_time = bucket + 60000。

时间切分（单月简化版，严格因果）：
  前 70% train / 15% es / 15% thr / 最后 7 天 test（互不重叠）

env: OKX_FEAT, SYMBOL, SEEDS, ROUNDS, HORIZON=3, OUT
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
from morph import ob_kline as ok
from morph.two_legs import fit

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

H = int(os.environ.get("HORIZON", "3"))
SEEDS = [int(s) for s in os.environ.get("SEEDS", "42,43,44,45,46").split(",")]
ROUNDS = int(os.environ.get("ROUNDS", "1000"))
TPD = int(os.environ.get("TPD", "15"))
PAYOUT = float(os.environ.get("PAYOUT", "0.808"))
TEST_DAYS = int(os.environ.get("TEST_DAYS", "7"))
T = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T:7.1f}s] {msg}", flush=True)


def main() -> None:
    # 1. 加载 OKX 特征
    feat_path = os.environ.get("OKX_FEAT", "")
    if not feat_path:
        raise SystemExit("env OKX_FEAT required")
    okx = pl.read_parquet(feat_path)
    cols = [c for c in okx.columns if c.startswith("okx_")
            and not any(c.startswith(p) for p in
                        ("okx_ob_n_", "okx_ob_levels_ratio", "okx_ob_qmean",
                         "okx_ob_qcv", "okx_ob_qmax_ratio"))]
    log(f"OKX features: {okx.height:,} rows, {len(cols)} usable cols (count-type dropped)")

    # 2. 加载币安 1m K 线，算 h=3 标签
    sym = os.environ.get("SYMBOL", "BTCUSDT")
    a = ok._assets(ok.KLINE_TAG)
    names = [f"{sym}_1m_klines_{y}.parquet" for y in range(2025, 2027)]
    have = [n for n in names if n in a]
    k = pl.concat([pl.read_parquet(ok.fetch(ok.KLINE_TAG, n),
                                   columns=["close_time", "close"]) for n in have])
    k = k.unique(subset="close_time").sort("close_time")
    # target[t] = close[t+H] > close[t]，左移 H 行
    k = k.with_columns(pl.col("close").shift(-H).alias("_fut"))
    k = k.with_columns((pl.col("_fut") > pl.col("close")).cast(pl.Int8).alias("target"))
    k = k.filter(pl.col("target").is_not_null()).select(["close_time", "target"])

    # 3. 对齐 join（两边 close_time 同语义）
    df = k.join(okx.select(["close_time"] + cols), on="close_time", how="inner").sort("close_time")
    log(f"joined: {df.height:,} rows, range "
        f"{df['close_time'].min()} ~ {df['close_time'].max()}")

    # 4. 时间切分：最后 TEST_DAYS 天 test；前面 70/15/15
    max_t = df["close_time"].max()
    test_start = max_t - TEST_DAYS * 86_400_000
    pre = df.filter(pl.col("close_time") < test_start)
    te = df.filter(pl.col("close_time") >= test_start)
    n = pre.height
    tr = pre[:int(n * 0.70)]
    es = pre[int(n * 0.70):int(n * 0.85)]
    thr = pre[int(n * 0.85):]
    log(f"split: train={tr.height:,} es={es.height:,} thr={thr.height:,} test={te.height:,}")

    Xtr, ytr = tr.select(cols).to_numpy().astype(np.float32), tr["target"].to_numpy().astype(int)
    Xes, yes = es.select(cols).to_numpy().astype(np.float32), es["target"].to_numpy().astype(int)
    Xte = te.select(cols).to_numpy().astype(np.float32)
    yte = te["target"].to_numpy().astype(np.int8)
    aucs, wins = [], []
    for seed in SEEDS:
        mdl = fit(Xtr, ytr, cols, Xes, yes, seed=seed,
                  params_override={"learning_rate": 0.03, "num_leaves": 15,
                                   "min_child_samples": 500, "lambda_l2": 20})
        p = mdl.predict(Xte, num_iteration=mdl.best_iteration)
        aucs.append(roc_auc_score(yte, p))
        # thr 段定阈值（需重训一个模型预测 thr，简化：用 es 段的 p 分布近似）
        p_es = mdl.predict(Xes, num_iteration=mdl.best_iteration)
        thr_v = float(np.quantile(np.abs(p_es - .5), 1 - TPD / 1440))
        hit = np.abs(p - .5) >= thr_v
        if hit.sum() >= 30:
            wins.append(float(((p[hit] - .5 > 0).astype(int) == yte[hit]).mean()))
        del mdl
    auc = float(np.mean(aucs))
    win = float(np.mean(wins)) if wins else float("nan")
    ev = PAYOUT * win - (1 - win) if wins else float("nan")
    print(f"\n[E1] OKX→Binance h={H}: AUC={auc:.4f} | @{TPD}/天 win={win*100:.2f}% "
          f"EV@{PAYOUT:g}={ev:+.3f}")
    print(f"  判据: AUC>0.54 且 win>58% → "
          f"{'PASS' if (auc > .54 and win > .58) else 'FAIL'}")
    log("DONE")


if __name__ == "__main__":
    main()
