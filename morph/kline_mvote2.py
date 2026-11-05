"""第十六轮：多数票目标（修正版）。第 15 轮的 mv_ups 有 off-by-one：
窗口取成了 up[t..t+n-1]（含 t 时刻已知的那根、漏掉最远的 t+n），标签被塞进了
"已知的部分答案"，这也是当时 own-label AUC 虚高到 0.72 的原因。本轮全部重跑：

  1) `mv_selfcheck()`：向量化实现 vs 逐根实现，必须逐元素相等（先过这关再谈结论）；
  2) 多数票目标（未来 n 根里涨的根数 > k）：5>3、5>4、10>5、10>6、10>7；
     评估标签仍然固定 h=3（与定稿、与订单簿腿同口径）；
  3) **形态表**：不做模型，直接看"当前连涨/连跌长度"这个最朴素的形态，
     对多数票标签、对 h=3 标签的条件概率各是多少 —— 这直接回答
     "形态能不能预测未来 N 根多数票 / 能不能预测 3 分钟方向"。

其余口径与第 15 轮一致：base = 5 horizon × 5 seed 的 val z 平均；融合权重只报曲线。
"""
import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from morph.kline_boost import (OB_END, TEST_START, VAL_START,  # noqa: E402
                               build_features, load_klines, log)
from morph.kline_multi import HS, evaluate  # noqa: E402
from morph.kline_probe import mv_selfcheck, mv_ups  # noqa: E402
from morph.kline_vp import fit_z  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

MVS = tuple(tuple(int(x) for x in s.split(":"))
            for s in os.environ.get("MVS", "5:3,5:4,10:5,10:6,10:7").split(","))


def runs(close):
    """带符号的连涨/连跌长度：正=连涨根数，负=连跌根数（t 时刻已知）。"""
    up = np.zeros(len(close), dtype=np.int8)
    up[1:] = close[1:] > close[:-1]
    r = np.ones(len(close), dtype=np.int32)
    for i in range(2, len(close)):
        r[i] = r[i - 1] + 1 if up[i] == up[i - 1] else 1
    return (r * np.where(up > 0, 1, -1)).astype(np.int32)


def pattern_table(r, y3, labels, mask, title):
    log(f"\n[{title}] 形态（连涨/连跌长度）→ 条件概率")
    log(f"  {'形态':<10}{'样本':>9}" + "".join(f"{k:>10}" for k in ["h3涨", *labels]))
    for sgn, name in ((1, "连涨"), (-1, "连跌")):
        for L in (1, 2, 3, 4, 5):
            m = mask & (r == sgn * L) if L < 5 else mask & (sgn * r >= 5)
            n = int(m.sum())
            if n < 500:
                continue
            cells = [f"{y3[m].mean():>10.4f}"] + [f"{labels[k][m].mean():>10.4f}"
                                                 for k in labels]
            log(f"  {name}{L}根{'':<4}{n:>9}" + "".join(cells))


def main():
    seeds = [int(x) for x in os.environ.get("SEEDS", "42,43,44,45,46").split(",")]
    mv_seeds = [int(x) for x in os.environ.get("MV_SEEDS", "42,43,44").split(",")]
    rounds = int(os.environ.get("ROUNDS", "60"))
    max_h = max(HS)
    mv_selfcheck()
    if max(n for n, _ in MVS) > max_h:
        raise SystemExit(f"[mv] n 最大 {max(n for n, _ in MVS)} 超过行网格余量 {max_h}")

    raw = load_klines().filter(pl.col("close_time") <= OB_END)
    ft, cols = build_features(raw)
    nmax = raw.height - max_h
    ct = ft.head(nmax)["close_time"].to_numpy()
    close = raw["close"].to_numpy().astype(np.float64)
    log(f"[data] rows={nmax} cols={len(cols)} MVS={MVS}")

    tr = np.where(ct < VAL_START)[0]
    va = np.where((ct >= VAL_START) & (ct < TEST_START))[0]
    te = np.where(ct >= TEST_START)[0]
    va = va[va + max_h < int(te[0])]
    tr = tr[tr + max_h < int(va[0])]
    pr = np.concatenate([va, te])
    nv = len(va)
    y3 = (close[3:3 + nmax] > close[:nmax]).astype(np.int8)
    log(f"[split] train={len(tr)} val={len(va)} test={len(te)} h3 基准={y3[te].mean():.4f}")

    X = ft.head(nmax).select(cols).to_numpy().astype(np.float32)
    Xtr, Xpr = X[tr], X[pr]
    del X

    rows, own = {}, {}
    full = np.zeros(nmax)
    log("\n=== 对照: 定稿 LGB 腿（5 horizon × 5 seed）===")
    zs = {}
    for h in HS:
        y = (close[h:h + nmax] > close[:nmax]).astype(np.int8)
        zs[h], _ = fit_z(Xtr, Xpr, y, tr, va, nv, seeds, rounds, cols)
        log(f"[base h={h}] done")
    base_z = np.mean([zs[h] for h in HS], axis=0)
    full[pr] = base_z
    evaluate("base5s", full, ct, y3, va, rows)

    log("\n=== 多数票目标（修正标签；评估标签固定 h=3）===")
    mv_z, labels = {}, {}
    for n, k in MVS:
        ymv = (mv_ups(close, n, nmax) > k).astype(np.int8)
        tag = f"mv{n}>{k}"
        labels[tag] = ymv
        log(f"[{tag}] 正类占比 train={ymv[tr].mean():.4f} val={ymv[va].mean():.4f} "
            f"test={ymv[te].mean():.4f}")
        z, _ = fit_z(Xtr, Xpr, ymv, tr, va, nv, mv_seeds, rounds, cols)
        mv_z[tag] = z
        own[tag] = float(roc_auc_score(ymv[te], z[nv:]))        # pr = [va, te]
        full[pr] = z
        evaluate(f"{tag}@h3", full, ct, y3, va, rows)
        for w in (0.2, 0.4):
            full[pr] = base_z + w * z
            evaluate(f"base+{w}{tag}", full, ct, y3, va, rows)
        log(f"[{tag}] own-label AUC={own[tag]:.4f}（test）")

    mz = np.mean(list(mv_z.values()), axis=0)
    full[pr] = mz
    evaluate("mv_mean@h3", full, ct, y3, va, rows)
    for w in (0.2, 0.4):
        full[pr] = base_z + w * mz
        evaluate(f"base+{w}mv_mean", full, ct, y3, va, rows)

    # 一致性用法（不改特征、只改选哪些样本）：多个多数票视角同向时加权
    agree = np.abs(np.mean([np.sign(v) for v in mv_z.values()], axis=0))
    log(f"[agree] 5 个多数票视角全同向占比={float((agree > 0.99).mean()):.3f} "
        f"全反向占比={float((agree < 1e-6).mean()):.3f}")
    full[pr] = base_z * (0.5 + 0.5 * agree)
    evaluate("base×agree_mv", full, ct, y3, va, rows)

    # ---- 形态表：不做模型，直接看连涨连跌长度对两类标签的条件概率 ----
    r = runs(close)[:nmax]
    pick = {k: labels[k] for k in list(labels)[:3]}
    mask_te = np.zeros(nmax, dtype=bool)
    mask_te[te] = True
    pattern_table(r, y3, pick, mask_te, "test")
    mask_va = np.zeros(nmax, dtype=bool)
    mask_va[va] = True
    pattern_table(r, y3, pick, mask_va, "val（只作确认，不选参）")

    log(f"\n多数票 own-label AUC（测试集，仅诊断难度）：")
    for k, v in own.items():
        log(f"  {k:<10}{v:.4f}")
    log(f"\n{'变体':<20}{'static':>9}{'AUC':>8}{'按天':>8}{'窗口4':>8}{'4段sd':>8}")
    for k, r2 in rows.items():
        log(f"{k:<20}{r2['static']:>9.4f}{r2['auc']:>8.4f}{r2['daily']:>8.4f}"
            f"{r2['w4']:>8.4f}{r2['w4sd']:>8.4f}  明细={r2['wins']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
