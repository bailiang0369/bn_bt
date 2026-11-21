#!/usr/bin/env python3
"""Stoch 确认点 → "真反转"二元预测的诊断（用户 2026-11-21 修正口径）。

目标定义（非重言）
------------------
起点 = 结构点确认时刻（stoch 刚进入超买/超卖块）：
  - 若刚确认谷（进入超买，sc_ptype=-1）：在 stoch **重新上穿 80** 之前，
    会不会**先下穿 20**（到达超卖）？是=1（真反转），否=0（假突破）。
  - 若刚确认峰（进入超卖，sc_ptype=+1）：在 stoch **重新下穿 20** 之前，
    会不会**先上穿 80**？是=1，否=0。

这不是"峰→谷交替"的重言：超买后可能真跌到超卖，也可能假突破后重回超买，
结果由市场动态决定，ptype 单列无法推出。

诊断（不训练对撞、几分钟）
-------------------------
  1) label 分布（真反转占比，不能一边倒）
  2) ptype-only AUC（必须 ≈0.5，否则仍是重言）
  3) 全 17 特征 AUC + 与 ptype-only 的差
  4) train(2020-25) / test(2026) 时间切分

env: K_PERIOD TEST_START MAX_WINDOW SC_SELFCHECK
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from morph import kline_long as kl  # noqa: E402
from morph.stoch_struct import FEAT_NAMES, build_struct, log, struct_leak_check  # noqa: E402,F401

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

K_PERIOD = int(os.environ.get("K_PERIOD", "80"))
TEST_START = os.environ.get("TEST_START", "2026-01")
MAX_WINDOW = int(os.environ.get("MAX_WINDOW", "10080"))   # 超时视为未反转（7 天）


def fit_predict(Xtr, ytr, Xte):
    p = dict(objective="binary", metric="auc", learning_rate=.05, num_leaves=15,
             min_child_samples=200, feature_fraction=.8, bagging_fraction=.8,
             bagging_freq=1, verbose=-1, num_threads=os.cpu_count() or 4, seed=42)
    return lgb.train(p, lgb.Dataset(Xtr, ytr), num_boost_round=250).predict(Xte)


def main() -> None:
    if os.environ.get("SC_SELFCHECK", "1") == "1":
        if not struct_leak_check(k_period=K_PERIOD):
            raise SystemExit("[rev] 泄露自检失败")

    raw = kl.load_raw()
    X, names, (ptype, pidx, px) = build_struct(raw, k_period=K_PERIOD, return_points=True)
    sk = X[:, FEAT_NAMES.index("sc_stoch_k")].astype(np.float64)
    ct = raw["close_time"].to_numpy()
    n = len(sk)

    # stoch 穿越事件（sk 在 warmup 为 NaN，比较自然为 False）
    cross_up = np.where((sk[:-1] <= 80) & (sk[1:] > 80))[0] + 1    # 进入超买
    cross_dn = np.where((sk[:-1] >= 20) & (sk[1:] < 20))[0] + 1    # 进入超卖

    # 确认点事件 = ptype 交替处
    ev = np.where(np.diff(ptype) != 0)[0] + 1
    ev = ev[pidx[ev] >= 0]
    log(f"确认点 {len(ev):,} | 上穿80 {len(cross_up):,} | 下穿20 {len(cross_dn):,}")

    rows = []
    for t0 in ev:
        t0 = int(t0)
        is_trough_conf = ptype[t0] == -1        # 刚确认谷 → 进入超买
        target = cross_dn if is_trough_conf else cross_up    # 真反转要到达的另一极端
        reentry = cross_up if is_trough_conf else cross_dn   # 假突破=重回原极端
        jt = int(np.searchsorted(target, t0, side="right"))
        jr = int(np.searchsorted(reentry, t0, side="right"))
        t_target = int(target[jt]) if jt < len(target) else n + MAX_WINDOW
        t_reentry = int(reentry[jr]) if jr < len(reentry) else n + MAX_WINDOW
        if min(t_target, t_reentry) - t0 > MAX_WINDOW:
            continue                            # 超时：既没反转也没重回，剔除
        label = 1.0 if t_target < t_reentry else 0.0
        rows.append(dict(t=t0, label=label))

    d = pl.DataFrame(rows)
    E = d.height
    test_ms = int(np.datetime64(TEST_START + "-01", "D").astype("datetime64[ms]").astype(np.int64))
    # 时间切分：按每段起点的时间；用 ct[t0]
    tv = np.array([ct[int(r["t"])] >= test_ms for r in d.to_dicts()])
    tr, te = ~tv, tv
    log(f"有效确认点（窗口内结局）{E:,} | train {tr.sum():,} / test {te.sum():,}")

    y = d["label"].to_numpy()
    print(f"\n[label 分布] 真反转占比 train={y[tr].mean():.3f} test={y[te].mean():.3f}"
          f"（一边倒则目标无意义）")

    Xev = X[np.array([r["t"] for r in d.to_dicts()])]
    pt_col = Xev[:, FEAT_NAMES.index("sc_ptype")].reshape(-1, 1)

    if y[te].std() < 1e-9:
        print("test label 单值，无法算 AUC"); log("DONE"); return

    a_pt = float(roc_auc_score(y[te], fit_predict(pt_col[tr], y[tr], pt_col[te])))
    a_all = float(roc_auc_score(y[te], fit_predict(Xev[tr], y[tr], Xev[te])))
    print(f"[test AUC] 全特征={a_all:.4f}  ptype-only={a_pt:.4f}  增量={a_all - a_pt:+.4f}")
    print("  判据：ptype-only≈0.5 证明非重言；全特征须 >0.55 且增量 >0.03 才有交易价值")
    log("DONE")


if __name__ == "__main__":
    main()
