"""多 horizon 分数混合（h=1/2/3）——用短周期的强信号救长周期。

动机（Run J/K）：h=1 的按天 top15 到 0.695，h=3 只有 0.609。但 h=3 的收益窗口
"包含"了 h=1 的方向（3 分钟上涨通常先有 1 分钟上涨压力），所以把 h=1/2/3 的
预测分数线性混合，理论上能给 h=3 补信息。

做法：
  1) 统一行集（不做 drop_nulls，用 max(h) 做 purge），保证三个 horizon 索引对齐；
  2) 每个 h 单独训练 LightGBM（多 seed 平均）；
  3) val 上用『按天 top-k』口径网格搜索混合权重（不碰 test）；
  4) test 上同时报告单腿与混合。

用法：
    python -m orderbook.blend_h
    python -m orderbook.blend_h --horizons 1,2,3 --extra reg
"""
import os
import sys
import argparse
import itertools
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from orderbook.fusion_test import (
    load_orderbook, train_lgb_avg, acc_at_cov, day_index, complete_days, daily_topk_eval)

GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
REG_PARAMS = dict(num_leaves=31, min_child_samples=500, feature_fraction=0.6, lambda_l2=5.0)


def main():
    ap = argparse.ArgumentParser(description="多 horizon 分数混合")
    ap.add_argument("--horizons", default=os.environ.get("KAGGLE_HORIZONS", "1,2,3"))
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--rounds", type=int, default=1000)
    ap.add_argument("--daily-k", type=int, default=15)
    ap.add_argument("--extra", default=os.environ.get("KAGGLE_BLEND_EXTRA", ""),
                    help="base 或 reg（强正则）")
    ap.add_argument("--save-pred", default=os.environ.get("KAGGLE_FUSION_PRED"))
    a = ap.parse_args()

    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    hs = [int(s) for s in a.horizons.split(",") if s.strip()]
    extra = REG_PARAMS if a.extra == "reg" else None

    ob = load_orderbook()
    L = ob.height
    ct = ob["close_time"].to_numpy()
    close = ob["close"].to_numpy().astype(np.float64)
    hmax = max(hs)
    k1, k2 = int(L * 0.64), int(L * 0.80)
    tr, va = slice(0, k1 - hmax), slice(k1, k2 - hmax)
    te = slice(k2, L - hmax)          # 末端 hmax 行没有完整未来标签
    print(f"[ob] rows={L} horizons={hs} purge={hmax} "
          f"train[0:{k1 - hmax}] val[{k1}:{k2 - hmax}] test[{k2}:{L - hmax}]")

    cols = [c for c in ob.columns if c not in ("target", "close_time")]
    X = ob.select(cols).to_numpy().astype(np.float32)

    def y_of(h):
        y = np.zeros(L, dtype=np.int8)
        y[:-h] = (close[h:] > close[:-h]).astype(np.int8)
        return y

    ps, zs = {}, {}
    for h in hs:
        y = y_of(h)
        print(f"[h={h}] 训练中（{len(seeds)} seeds，extra={a.extra or 'base'}）")
        p, iters = train_lgb_avg(X, y, tr, va, cols, seeds, a.rounds, None, extra)
        m, s = float(p[tr].mean()), float(p[tr].std() + 1e-9)
        ps[h] = p
        zs[h] = (p - m) / s
        day_te_h = day_index(ct[te])
        auc = roc_auc_score(y[te], p[te])
        a1, _ = acc_at_cov(p[te] - 0.5, y[te], 0.01)
        d15 = float(daily_topk_eval(p[te] - 0.5, y[te], day_te_h,
                                    complete_days(day_te_h), a.daily_k).mean())
        print(f"[h={h}] test AUC={auc:.4f} 全局1%={a1:.4f} 按天top{a.daily_k}={d15:.4f} iters={iters}")

    day_te = day_index(ct[te])
    m_te = complete_days(day_te)
    z_te = {h: zs[h][te] for h in hs}
    z_va = {h: zs[h][va] for h in hs}
    dv = day_index(ct[va])
    mv = complete_days(dv)
    # 混合评估统一用"主 horizon"（最大 h）的标签
    y_va = y_of(hmax)[va]
    y_main_te = y_of(hmax)[te]

    combos = [c for c in itertools.product(GRID, repeat=len(hs)) if sum(c) > 0]
    best, best_acc = None, -1.0
    for c in combos:
        s = sum(w * z_va[h] for w, h in zip(c, hs))
        acc = float(daily_topk_eval(s, y_va, dv, mv, a.daily_k).mean())
        if acc > best_acc:
            best_acc, best = acc, c
    print(f"\n[blend] val 选出权重 {dict(zip(hs, best))} 按天top{a.daily_k}={best_acc:.4f}")

    print(f"\n{'组合':<28}{'AUC':>9}{'全局1%':>10}{'按天top' + str(a.daily_k):>12}")
    for name, s in [(f"h{h}", z_te[h]) for h in hs]:
        auc = roc_auc_score(y_main_te, s)
        a1, _ = acc_at_cov(s, y_main_te, 0.01)
        d = float(daily_topk_eval(s, y_main_te, day_te, m_te, a.daily_k).mean())
        print(f"{name:<28}{auc:>9.4f}{a1:>10.4f}{d:>12.4f}")

    s_best = sum(w * z_te[h] for w, h in zip(best, hs))
    auc = roc_auc_score(y_main_te, s_best)
    a1, _ = acc_at_cov(s_best, y_main_te, 0.01)
    d = float(daily_topk_eval(s_best, y_main_te, day_te, m_te, a.daily_k).mean())
    print(f"{'blend' + str(dict(zip(hs, best))):<28}{auc:>9.4f}{a1:>10.4f}{d:>12.4f}")
    print(f"\n[结论] 混合按天 top{a.daily_k} = {d:.4f}（目标 0.65，主 horizon={hmax}）")

    if a.save_pred:
        out = pl.DataFrame({"close_time": ct[te], "target": y_main_te,
                            **{f"p_h{h}": ps[h][te] for h in hs},
                            "blend": s_best})
        out.write_parquet(a.save_pred)
        print(f"[save] {a.save_pred}")


if __name__ == "__main__":
    main()
