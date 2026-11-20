#!/usr/bin/env python3
"""LGB + XGB 分数层 ensemble 验证（同口径全量）。

在现役 h10 全量口径下（10 窗 × 5 seed、train/es/thr/test、阈值只在 thr 段定），
把 LGB 和 XGB 的 test 分数按权重融合，对比：
  - w=1.0（纯 LGB，现役基线 63.32%）
  - w=0.7 / 0.5 / 0.3（不同融合权重）
  - w=0.0（纯 XGB）

分数必须先各自标准化（z-score），因为两模型分数尺度不同。
z-score 用 thr 段的均值和 std（因果），test 用同样的转换。

判据：融合后 tpd8 胜率 > 63.32% 且月 min 不低于 55.74% 才算有效。
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
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from morph import ob_kline as ok
from morph import h10_signal as hs
from morph.two_legs import add_target, fit as fit_lgb, matrix, XGB_PRESETS

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

SEEDS = [42, 43, 44, 45, 46]
ROUNDS = int(os.environ.get("ROUNDS", "2000"))
WEIGHTS = [float(w) for w in os.environ.get("WEIGHTS", "1.0,0.7,0.5,0.3,0.0").split(",")]
TPD_LIST = [int(x) for x in os.environ.get("TPD_LIST", "2,5,8,15,30").split(",")]
PRIMARY_TPD = int(os.environ.get("PRIMARY_TPD", "8"))
PAYOUT = float(os.environ.get("PAYOUT", "0.80"))
XGB_PRESET = os.environ.get("XGB_PRESET", "tuned")
T = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T:7.1f}s] {msg}", flush=True)


def main() -> None:
    # 数据 + 49 列特征
    a = ok._assets(ok.KLINE_TAG)
    names = [f"BTCUSDT_1m_klines_{y}.parquet" for y in range(2020, 2027)]
    k = pl.concat([pl.read_parquet(ok.fetch(ok.KLINE_TAG, n),
                                   columns=["close_time", "open", "high", "low", "close",
                                            "um_buy_vol", "um_sell_vol", "um_count"])
                   for n in names if n in a])
    df, feats = hs.kl.build(k)
    df, feats = hs.add_flow(df, feats)
    df = df.with_columns(pl.col("close_time").cast(pl.Datetime("ms")).dt.strftime("%Y-%m").alias("ym"))
    tagged = add_target(df, 10).filter(pl.col("target").is_not_null()).sort("close_time")
    log(f"tagged {tagged.height:,} rows, {len(feats)} feats")

    xgb_params = dict(XGB_PRESETS[XGB_PRESET])
    log(f"XGB preset={XGB_PRESET}: {xgb_params}")
    log(f"ensemble weights (LGB): {WEIGHTS}")

    months = [m for m in sorted(df["ym"].unique().to_list()) if "2026-01" <= m <= "2026-10"]
    per_w = {w: dict(s=[], y=[], d=[], mon=[]) for w in WEIGHTS}
    for m in months:
        t0 = int(dt.datetime.strptime(m + "-01", "%Y-%m-%d").replace(
            tzinfo=dt.timezone.utc).timestamp() * 1000)
        lo = t0 - 1460 * 86_400_000
        pre = tagged.filter((pl.col("close_time") < t0) & (pl.col("close_time") >= lo))
        te = tagged.filter(pl.col("ym") == m)
        n = pre.height
        c1, c2 = int(n * 0.85), int(n * 0.95)
        tr, es, thr = pre[:c1], pre[c1:c2], pre[c2:]
        Xtr, ytr = matrix(tr, feats), tr["target"].to_numpy().astype(int)
        Xes, yes = matrix(es, feats), es["target"].to_numpy().astype(int)
        Xth = matrix(thr, feats)
        Xte = matrix(te, feats)
        yte = te["target"].to_numpy().astype(np.int8)
        ct = te["close_time"].to_numpy()

        pth_l = np.zeros(thr.height); pte_l = np.zeros(te.height)
        pth_x = np.zeros(thr.height); pte_x = np.zeros(te.height)
        for seed in SEEDS:
            ml = fit_lgb(Xtr, ytr, feats, Xes, yes, seed=seed,
                         params_override=hs.LGB_PARAMS)
            pth_l += ml.predict(Xth, num_iteration=ml.best_iteration) / len(SEEDS)
            pte_l += ml.predict(Xte, num_iteration=ml.best_iteration) / len(SEEDS)
            mx = XGBClassifier(objective="binary:logistic", eval_metric="auc",
                               tree_method="hist", n_estimators=ROUNDS,
                               early_stopping_rounds=50, random_state=seed,
                               n_jobs=os.cpu_count() or 4, verbosity=0, **xgb_params)
            mx.fit(Xtr, ytr, eval_set=[(Xes, yes)], verbose=False)
            kb = int(mx.best_iteration)
            pth_x += mx.predict_proba(Xth, iteration_range=(0, kb+1))[:, 1] / len(SEEDS)
            pte_x += mx.predict_proba(Xte, iteration_range=(0, kb+1))[:, 1] / len(SEEDS)

        # thr 段各自 z-score（因果），test 用同样转换
        mu_l, sd_l = pth_l.mean(), pth_l.std() + 1e-12
        mu_x, sd_x = pth_x.mean(), pth_x.std() + 1e-12
        zth_l = (pth_l - mu_l) / sd_l; zte_l = (pte_l - mu_l) / sd_l
        zth_x = (pth_x - mu_x) / sd_x; zte_x = (pte_x - mu_x) / sd_x
        days = np.unique(ct // 86_400_000)

        for w in WEIGHTS:
            zth = w * zth_l + (1 - w) * zth_x
            zte = w * zte_l + (1 - w) * zte_x
            for tpd in TPD_LIST:
                thr_v = float(np.quantile(np.abs(zth), 1 - tpd / 1440))
                hit = np.abs(zte) >= thr_v
                if hit.sum() < 30:
                    continue
                win = float(((zte[hit] > 0).astype(int) == yte[hit]).mean())
                per_w[w]["s"].append(zte[hit]); per_w[w]["y"].append(yte[hit])
                per_w[w]["d"].append(ct[hit]); per_w[w]["mon"].append(win)
        log(f"[{m}] done")

    print(f"\n{'w(LGB)':>8}{'n':>7}{'单/天':>8}{'胜率':>9}{'EV@0.8':>9}{'月min':>8}{'破50':>7}{'AUC':>8}")
    nd_days = len(set(int(d) for w in WEIGHTS for d in per_w[w]["d"]))
    for w in WEIGHTS:
        p = per_w[w]
        if not p["y"]:
            continue
        s = np.concatenate(p["s"]); y = np.concatenate(p["y"])
        win = float(((s > 0).astype(int) == y).mean())
        ev = PAYOUT * win - (1 - win)
        mon = np.array(p["mon"])
        # AUC 近似（用所有 hit 的分数；全样本 AUC 另算更准，这里只看 hit 区）
        nd = len(np.unique(np.concatenate(p["d"]) // 86_400_000))
        print(f"{w:>8.2f}{len(y):>7d}{len(y)/nd:>8.1f}{win*100:>8.2f}%"
              f"{ev:>+9.3f}{mon.min()*100:>7.2f}%{int((mon<.5).sum()):>5}/{len(mon)}")
    print(f"\n基线（LGB w=1.0）应为 63.32% @7.4/天 EV+0.140 月min 55.74%")
    log("DONE")


if __name__ == "__main__":
    main()
