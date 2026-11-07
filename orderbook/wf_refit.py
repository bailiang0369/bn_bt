"""盘口腿 walk-forward 滚动重训：验证"模型太旧"是不是 0.62 卡住的主因。

背景
----
我们全部实验都是**一次训练、长期外推**（训练窗最长 6 年，val 只有 30 天）。
症状：时好时坏（off=0 的 0.6167、off=120 的 0.5531）、阈值/参数在 30 天 val 上选什么都会翻车。
如果模型漂移是主因，那么"只用最近 W 天数据、每 B 天重训一次"应当显著更稳。

方案
  对每个测试窗（off=0/60/120）：
    - baseline：单次训练（val_start 之前全部数据），5 seeds —— 与 §28/§30 同口径；
    - wf-W-B  ：把测试窗按 B=30 天切块，每块只用"块起点前 W 天"的数据重训（W=180 / 365）。
  两者都用同样 60 轮、OB_REG、5→3 seeds，打分后统一 rolling z。
指标：roll_1d / roll_3d topK（K=15/天，另附 K=8）、AUC、按天。
判据：wf 相对 baseline 的 roll3d 三窗均值 ≥ +0.5pp，且最差窗不退化，才认为"漂移是主因"。
"""
import os
import sys
import time

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lightgbm as lgb  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from orderbook.znorm_eval import (BASE, DAILY_K, H, OB_REG, complete_days,  # noqa: E402
                                 daily_topk_eval, day_index, global_topk_eval,
                                 load, log, rolling_z)

DAY = 86_400_000
BLOCK = 30


def fit(X, y, tr, cols, seeds, rounds, predict_idx):
    acc = np.zeros(len(predict_idx), dtype=np.float64)
    for sd in seeds:
        q = dict(BASE, **OB_REG, seed=sd, bagging_seed=sd, feature_fraction_seed=sd,
                 data_random_seed=sd)
        m = lgb.train(q, lgb.Dataset(X[tr], y[tr], feature_name=cols), num_boost_round=rounds)
        acc += m.predict(X[predict_idx])
    return (acc / len(seeds)).astype(np.float32)


def main():
    offs = [int(x) for x in os.environ.get("OFFS", "0,60,120").split(",")]
    seeds = [int(x) for x in os.environ.get("SEEDS", "42,43,44").split(",")]
    seeds_base = [int(x) for x in os.environ.get("BASE_SEEDS", "42,43,44,45,46").split(",")]
    rounds = int(os.environ.get("ROUNDS", "60"))
    wins = [int(x) for x in os.environ.get("WINS", "180,365").split(",")]
    t0 = time.time()

    df = load()
    ct = df["close_time"].to_numpy()
    y = df["target"].to_numpy().astype(np.int8)
    cols = [c for c in df.columns if c not in ("target", "close_time")]
    X = df.select(cols).to_numpy().astype(np.float32)
    del df
    L = X.shape[0]
    k1_0, k2_0 = int(L * 0.64), int(L * 0.80)
    log(f"[data] rows={L} feats={len(cols)} block={BLOCK}d wins={wins}")

    rows = {}
    for off in offs:
        k1, k2 = k1_0 - off * 1440, k2_0 - off * 1440
        val_start, test_start = int(ct[k1]), int(ct[k2])
        tr_all = np.arange(0, k1 - H)
        va_m = (ct >= val_start) & (ct < test_start)
        te_m = ct >= test_start
        te_idx = np.where(te_m)[0]
        dte = day_index(ct[te_m])
        mt = complete_days(dte, 1200)
        log(f"\n===== off={off} test={len(te_idx)} 天={len(np.unique(dte))} "
            f"{int(ct[te_idx[0]])}~{int(ct[te_idx[-1]])} =====")

        def report(tag, s):
            zs = (s - s[va_m].mean()) / (s[va_m].std() + 1e-9)
            r = {"auc": float(roc_auc_score(y[te_m], zs[te_m])),
                 "k15": global_topk_eval(zs[te_m], y[te_m], dte, DAILY_K),
                 "k8": global_topk_eval(zs[te_m], y[te_m], dte, 8),
                 "roll1d": global_topk_eval(rolling_z(s, 1440)[te_m], y[te_m], dte, DAILY_K),
                 "roll3d": global_topk_eval(rolling_z(s, 4320)[te_m], y[te_m], dte, DAILY_K),
                 "daily": float(daily_topk_eval(zs[te_m], y[te_m], dte, mt, DAILY_K).mean())}
            rows[(off, tag)] = r
            log(f"[{tag:<12}] k15={r['k15']:.4f} k8={r['k8']:.4f} roll1d={r['roll1d']:.4f} "
                f"roll3d={r['roll3d']:.4f} AUC={r['auc']:.4f} 按天={r['daily']:.4f}")

        t = time.time()
        s = np.zeros(L)
        s[te_idx] = fit(X, y, tr_all, cols, seeds_base, rounds, te_idx)
        report("baseline", s)
        log(f"    baseline {time.time()-t:.0f}s")

        for W in wins:
            t = time.time()
            s = np.zeros(L)
            edges = [int(ct[te_idx[0]])]
            while edges[-1] < int(ct[te_idx[-1]]):
                edges.append(edges[-1] + BLOCK * DAY)
            for b0, b1 in zip(edges[:-1], edges[1:]):
                pred = np.where((ct >= b0) & (ct < b1))[0]
                if len(pred) < 100:
                    continue
                lo = int(ct[pred[0]]) - W * DAY
                tr = np.where((ct >= lo) & (ct < int(ct[pred[0]])))[0]
                tr = tr[tr + H < int(pred[0])]
                if len(tr) < 50000:
                    log(f"    wf{W} 块起点 {int(b0)} 训练行不足({len(tr)})，跳过")
                    continue
                s[pred] = fit(X, y, tr, cols, seeds, rounds, pred)
            report(f"wf{W}", s)
            log(f"    wf{W} {time.time()-t:.0f}s")

    log(f"\n{'变体':<13}{'off':>5}{'k15':>8}{'k8':>8}{'roll1d':>9}{'roll3d':>9}{'AUC':>8}{'按天':>8}")
    for (off, tag), r in rows.items():
        log(f"{tag:<13}{off:>5}{r['k15']:>8.4f}{r['k8']:>8.4f}{r['roll1d']:>9.4f}"
            f"{r['roll3d']:>9.4f}{r['auc']:>8.4f}{r['daily']:>8.4f}")

    tags = ["baseline"] + [f"wf{w}" for w in wins]
    log(f"\n{'变体':<13}{'roll3d均值':>11}{'roll3d最差':>11}{'roll1d均值':>11}{'AUC均值':>9}{'Δroll3d':>10}")
    b3 = np.mean([rows[(o, "baseline")]["roll3d"] for o in offs])
    for tag in tags:
        rs = [rows[(o, tag)] for o in offs if (o, tag) in rows]
        if not rs:
            continue
        m3 = float(np.mean([r["roll3d"] for r in rs]))
        log(f"{tag:<13}{m3:>11.4f}{np.min([r['roll3d'] for r in rs]):>11.4f}"
            f"{np.mean([r['roll1d'] for r in rs]):>11.4f}"
            f"{np.mean([r['auc'] for r in rs]):>9.4f}{(m3-b3)*100:>+9.2f}pp")
    log(f"[done] {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
