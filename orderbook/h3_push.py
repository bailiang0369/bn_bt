"""3min 专项攻坚（事件合约最小周期 = 3min，1min 无意义）。

当前 3min 上限（按天 15 单）≈ 0.615，距 65% 差 3.5pp。本模块一次跑多组候选：

A. 打分组合（不重训，复用 h=1/2/3 三个模型）
   1) p3                      —— 基线
   2) 0.3·z1 + 0.7·z3         —— 加权
   3) consensus               —— 三个 horizon 同向才给分，且取最保守的那个
   4) consensus 的两两版（1&3 / 2&3）
B. 训练加权
   5) w_absret                —— sample weight ∝ |未来 3min 收益|（硬过滤 minret2 的连续版）

所有候选都在 val 与 test 上同时报"按天 top-k"，只有在 val 上同向提升的才可信。

用法：
    python -m orderbook.h3_push
"""
import os
import sys
import argparse
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from orderbook.fusion_test import (
    load_orderbook, train_lgb_avg, acc_at_cov, day_index, complete_days, daily_topk_eval)

REG = dict(num_leaves=31, min_child_samples=500, feature_fraction=0.6, lambda_l2=5.0)


def consensus(zlist):
    """多 horizon 同向才给分，取最保守（绝对值最小）的那个；不同向给 0。"""
    Z = np.vstack(zlist)
    out = np.zeros(Z.shape[1], dtype=np.float64)
    pos = np.all(Z > 0, axis=0)
    neg = np.all(Z < 0, axis=0)
    out[pos] = Z.min(axis=0)[pos]
    out[neg] = Z.max(axis=0)[neg]
    return out


def evaluate(score, y, day_id, mask, k):
    acc = float(daily_topk_eval(score, y, day_id, mask, k).mean())
    a1, _ = acc_at_cov(score, y, 0.01)
    return acc, a1


def main():
    ap = argparse.ArgumentParser(description="3min 专项攻坚")
    ap.add_argument("--seeds", default=os.environ.get("KAGGLE_SEEDS", "42,43,44,45,46"))
    ap.add_argument("--rounds", type=int, default=1000)
    ap.add_argument("--daily-k", type=int, default=15)
    ap.add_argument("--main-h", type=int, default=3, help="结算周期（事件合约=3min）")
    a = ap.parse_args()

    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    hs = [1, 2, a.main_h]
    hmax = max(hs)

    ob = load_orderbook()
    L = ob.height
    ct = ob["close_time"].to_numpy()
    close = ob["close"].to_numpy().astype(np.float64)
    k1, k2 = int(L * 0.64), int(L * 0.80)
    tr, va = slice(0, k1 - hmax), slice(k1, k2 - hmax)
    te = slice(k2, L - hmax)
    print(f"[ob] rows={L} horizons={hs} purge={hmax} "
          f"train[0:{k1 - hmax}] val[{k1}:{k2 - hmax}] test[{k2}:{L - hmax}]")

    def y_of(h):
        y = np.zeros(L, dtype=np.int8)
        y[:-h] = (close[h:] > close[:-h]).astype(np.int8)
        return y

    cols = [c for c in ob.columns if c not in ("target", "close_time")]
    X = ob.select(cols).to_numpy().astype(np.float32)
    fwd = np.zeros(L)
    fwd[:-a.main_h] = close[a.main_h:] / close[:-a.main_h] - 1.0
    # 连续加权：|未来收益| 越大越可预测（minret2 硬过滤的平滑版），限制在 [0.2, 3]
    w_abs = np.clip(np.abs(fwd) * 1e4 / 2.0, 0.2, 3.0)

    ps, zs = {}, {}
    for h in hs:
        y = y_of(h)
        print(f"[h={h}] 训练（{len(seeds)} seeds, reg）")
        p, it = train_lgb_avg(X, y, tr, va, cols, seeds, a.rounds, None, REG)
        m, s = float(p[tr].mean()), float(p[tr].std() + 1e-9)
        ps[h], zs[h] = p, (p - m) / s
        print(f"[h={h}] iters={it}")

    # 加权版（只对主 horizon 训练一次）
    print(f"[h={a.main_h}] 训练（|fwd| 连续加权）")
    p_w, it_w = train_lgb_avg(X, y_of(a.main_h), tr, va, cols, seeds, a.rounds, w_abs, REG)
    m, s = float(p_w[tr].mean()), float(p_w[tr].std() + 1e-9)
    z_w = (p_w - m) / s
    print(f"[h={a.main_h}] iters={it_w}")

    cands = {
        "p3": zs[a.main_h],
        "w13": 0.3 * zs[1] + 0.7 * zs[a.main_h],
        "w23": 0.3 * zs[2] + 0.7 * zs[a.main_h],
        "cons3": consensus([zs[1], zs[2], zs[a.main_h]]),
        "cons13": consensus([zs[1], zs[a.main_h]]),
        "cons23": consensus([zs[2], zs[a.main_h]]),
        "absret": z_w,
        "absret_w13": 0.3 * zs[1] + 0.7 * z_w,
    }

    y_main = y_of(a.main_h)
    dv, mv = day_index(ct[va]), complete_days(day_index(ct[va]))
    dt, mt = day_index(ct[te]), complete_days(day_index(ct[te]))

    print(f"\n{'候选':<12}{'val按天':>10}{'val全局1%':>11}{'test按天':>11}{'test全局1%':>12}")
    rows = []
    for name, s in cands.items():
        v, va1 = evaluate(s[va], y_main[va], dv, mv, a.daily_k)
        t, ta1 = evaluate(s[te], y_main[te], dt, mt, a.daily_k)
        rows.append((name, v, va1, t, ta1))
        print(f"{name:<12}{v:>10.4f}{va1:>11.4f}{t:>11.4f}{ta1:>12.4f}")

    # 只有 val 与 test 同时最优才算可信
    best_v = max(rows, key=lambda r: r[1])
    best_t = max(rows, key=lambda r: r[3])
    print(f"\n[val 最优] {best_v[0]} = {best_v[1]:.4f}（test {best_v[3]:.4f}）")
    print(f"[test 最优] {best_t[0]} = {best_t[3]:.4f}（val {best_t[1]:.4f}）")
    print(f"[目标] 按天 top{a.daily_k} ≥ 0.65 → "
          f"{'达成' if best_t[3] >= 0.65 else '未达成'}")

    if os.environ.get("KAGGLE_FUSION_PRED"):
        out = pl.DataFrame({
            "close_time": ct[te], "target": y_main[te],
            **{f"z_h{h}": zs[h][te] for h in hs},
            "z_absret": z_w[te],
            "cons3": cands["cons3"][te],
        })
        out.write_parquet(os.environ["KAGGLE_FUSION_PRED"])
        print(f"[save] {os.environ['KAGGLE_FUSION_PRED']}")


if __name__ == "__main__":
    main()
