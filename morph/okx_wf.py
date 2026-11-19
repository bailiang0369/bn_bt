#!/usr/bin/env python3
"""跨所订单簿严格 walk-forward 验证（决定 E1 59.70% 是否为过拟合）。

口径与项目现役 h10_signal 完全一致：
  - 逐月 test，滚动窗 WIN_DAYS 天，窗内 train/es/thr/test 四段
  - 模型只训 train，早停用 es，阈值只在 thr 段定（严格因果，不用 es in-sample）
  - 逐月胜率 + 月度 min + 全期 EV + SE（n/H 修正）
  - 标签：币安 h=3 方向；特征：OKX 90 列 L2（剔除档数类）

判据（预登记，沿用项目 §4）：
  主：tpd=15 全期胜率 > 58% 且 EV@0.808 > 0
  次：破 50% 月数 ≤ 2/4，月 min > 52%
  任一不满足 → 跨所方向关闭；全满足 → 进 E2 融合增量实验
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
from morph import ob_kline as ok
from morph import h3_gate_eval as ge
from morph.two_legs import fit
from morph.okx_l2 import _reindex, build_l2

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

H = int(os.environ.get("HORIZON", "3"))
SEEDS = [int(s) for s in os.environ.get("SEEDS", "42,43,44,45,46").split(",")]
ROUNDS = int(os.environ.get("ROUNDS", "1000"))
TPD = int(os.environ.get("TPD", "15"))
PAYOUT = float(os.environ.get("PAYOUT", "0.808"))
WIN_DAYS = int(os.environ.get("WIN_DAYS", "60"))
TEST_DAYS = int(os.environ.get("TEST_DAYS", "7"))
T = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T:7.1f}s] {msg}", flush=True)


def main() -> None:
    # OKX 特征（多月）
    paths = sorted(Path("/kaggle/input").rglob("*okx_BTC-USDT_*.parquet"))
    if not paths:
        raise SystemExit("no OKX feature parquet")
    log(f"loading {len(paths)} files")
    raw = pl.concat([pl.read_parquet(p) for p in paths]).unique(
        subset="close_time", keep="any").sort("close_time")
    raw = _reindex(raw.rename({c: c.removeprefix("okx_") for c in raw.columns
                                if c != "close_time"}))
    raw = build_l2(raw)
    cols = [c for c in raw.columns if c not in ("close_time", "ob_missing")
            and not any(c.startswith(p) for p in
                        ("ob_n_", "ob_levels_ratio", "ob_qmean", "ob_qcv",
                         "ob_qmax_ratio", "ob_vol_bucket"))]
    log(f"OKX {raw.height:,} rows x {len(cols)} cols")

    # 币安标签
    a = ok._assets(ok.KLINE_TAG)
    names = [f"BTCUSDT_1m_klines_{y}.parquet" for y in range(2025, 2027)]
    k = pl.concat([pl.read_parquet(ok.fetch(ok.KLINE_TAG, n),
                                   columns=["close_time", "close"]) for n in names if n in a])
    k = k.unique(subset="close_time").sort("close_time")
    k = k.with_columns(pl.col("close").shift(-H).alias("_fut"))
    k = k.with_columns((pl.col("_fut") > pl.col("close")).cast(pl.Int8).alias("target"))
    k = k.filter(pl.col("target").is_not_null()).select(["close_time", "target"])

    df = k.join(raw.select(["close_time"] + cols), on="close_time", how="inner").sort("close_time")
    df = df.with_columns(pl.col("close_time").cast(pl.Datetime("ms")).dt.strftime("%Y-%m-%d").alias("day"))
    log(f"joined {df.height:,} rows")

    days = sorted(df["day"].unique().to_list())
    test_days = days[-TEST_DAYS:]
    all_s, all_y, all_d, mons = [], [], [], []
    for td in test_days:
        t0 = int(dt.datetime.strptime(td, "%Y-%m-%d").replace(
            tzinfo=dt.timezone.utc).timestamp() * 1000)
        lo = t0 - WIN_DAYS * 86_400_000
        pre = df.filter((pl.col("close_time") < t0) & (pl.col("close_time") >= lo))
        te = df.filter(pl.col("day") == td)
        n = pre.height
        if n < 1000:
            log(f"[{td}] insufficient pre={n}")
            continue
        tr, es, thr = pre[:int(n * .85)], pre[int(n * .85):int(n * .95)], pre[int(n * .95):]
        Xtr, ytr = tr.select(cols).to_numpy().astype(np.float32), tr["target"].to_numpy().astype(int)
        Xes, yes = es.select(cols).to_numpy().astype(np.float32), es["target"].to_numpy().astype(int)
        Xth = thr.select(cols).to_numpy().astype(np.float32)
        Xte = te.select(cols).to_numpy().astype(np.float32)
        pth = np.zeros(thr.height)
        pte = np.zeros(te.height)
        for seed in SEEDS:
            mdl = fit(Xtr, ytr, cols, Xes, yes, seed=seed,
                      params_override={"learning_rate": .03, "num_leaves": 15,
                                       "min_child_samples": 500, "lambda_l2": 20})
            pth += mdl.predict(Xth, num_iteration=mdl.best_iteration) / len(SEEDS)
            pte += mdl.predict(Xte, num_iteration=mdl.best_iteration) / len(SEEDS)
        y = te["target"].to_numpy().astype(np.int8)
        thr_v = float(np.quantile(np.abs(pth - .5), 1 - TPD / 1440))
        hit = np.abs(pte - .5) >= thr_v
        auc = roc_auc_score(y, pte)
        if hit.sum() >= 30:
            w = ((pte[hit] - .5 > 0).astype(int) == y[hit]).astype(np.int8)
            win = float(w.mean())
            all_s.append(pte[hit]); all_y.append(y[hit]); all_d.append(te["close_time"].to_numpy()[hit])
            mons.append(win)
            log(f"[{td}] AUC={auc:.4f} hit={hit.sum()} win={win*100:.2f}%")
        else:
            log(f"[{td}] AUC={auc:.4f} too few hits")
    if not all_y:
        raise SystemExit("no valid test days")
    sv = np.concatenate(all_s); yv = np.concatenate(all_y)
    w = ((sv > .5).astype(int) == yv).astype(np.int8)
    win = float(w.mean()); ev = PAYOUT * win - (1 - win)
    nd = sum(len(np.unique(d // 86_400_000)) for d in all_d)
    se = float(np.sqrt(win * (1 - win) / max(len(w) / H, 2)))
    mon = np.array(mons)
    print(f"\n[E1-WF] n={len(w)} {len(w)/nd:.1f}单/天 胜率={win*100:.2f}% "
          f"EV@{PAYOUT:g}={ev:+.3f} SE={se*100:.2f}%")
    print(f"  月min={mon.min()*100:.2f}% 破50%月={int((mon<.5).sum())}/{len(mon)}")
    ok_main = win > .58 and ev > 0
    ok_sub = (mon < .5).sum() <= 2 and mon.min() > .52
    print(f"  判据: 主(胜率>58%&EV>0)={'PASS' if ok_main else 'FAIL'} "
          f"次(月min>52%&破50<=2)={'PASS' if ok_sub else 'FAIL'}")
    log("DONE")


if __name__ == "__main__":
    main()
