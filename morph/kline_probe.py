"""第十五轮：复核三条"新视角"思路（3m 重采样 / 符号化 / 多数票目标）。

用户提的三条里，前两条第 1 轮已经测过（v1 3m 重采样、v2 符号累加、v4 堆叠），
本轮只补第 1 轮**没做干净**的部分 + 第三条（从未测过）：

【A】3m 视角的公平对比（回答"3m 到底有没有独立排序能力"）
  第 1 轮把 3m 分数前向填回 1m 网格再算 topK，那会把 1/3 的样本密度缺陷混进来
  （每 3 根 1m 才有 1 个新分数）。本轮改成**同一批行、同一个标签、两个特征集**：
    - 只在 3m bar 收盘的那些 1m 行上评估（占 1/3 的样本）；
    - 3m 腿：特征在 3m 网格上重算，目标是"下一根完整 3m bar"方向；
    - 1m 腿：定稿 LGB 的分数在这些边界行上取值，标签同样是 close[t+3]>close[t]；
    - 两者在同一 3m 网格上按同一口径评估，AUC 直接可比。
  另附一行 `3m->1m(ffill)`：把 3m 分数前向填回 1m 网格（复现第 1 轮的 1/3 密度损失）。
  对齐说明（用户关心的点）：resample_3m 按 (close_time - t0)//180000 分组，t0 是首根
  1m 的收盘时刻（2020-01-01 00:01），所以每组正好是 3 根 clock 对齐的完整 1m bar，
  组的 close_time = 该组最后一根的 close_time → 特征只用完整 bar，目标 shift(-1)
  = 下一根完整 3m bar，不存在"用未完成 bar"或"拿上一根 bar 当目标"的问题。

【B】多数票目标（未来 n 根里上涨根数 > k，即"连开"）
  评估标签一律固定 h=3（和定稿、和订单簿腿同一口径）。多数票模型只当打分函数；
  另给 `@own`（在它自己标签上评估）作诊断，但**不可跨标签比较**（基准不同）。

【C】符号化视角：第 1 轮 v2 已实测（sign 累加特征加到 63 列上 → static 0.6014 vs
  0.6069，AUC 不变），本轮不重复。

对照 base5s：第 14 轮定稿的 LGB 腿（5 horizon × 5 seed z 平均），顺带做跨 kernel 复现检查。
"""
import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lightgbm as lgb  # noqa: E402

from morph.kline_boost import (OB_END, PARAMS, TEST_START, VAL_START,  # noqa: E402
                               build_features, load_klines, log, resample_3m,
                               zscore)
from morph.kline_multi import HS, evaluate  # noqa: E402
from morph.kline_vp import fit_z  # noqa: E402

MVS = tuple(tuple(int(x) for x in s.split(":"))
            for s in os.environ.get("MVS", "5:3,10:6,10:5").split(","))


def mv_ups(close, n, nmax):
    """sum_{j=1..n} [close[t+j] > close[t+j-1]]，t = 0..nmax-1（只用未来 n 根，无泄漏）。"""
    up = np.zeros(len(close), dtype=np.int32)
    up[1:] = (close[1:] > close[:-1])
    c = np.concatenate(([0], np.cumsum(up)))
    return (c[n:n + nmax] - c[:nmax]).astype(np.int32)


def fit_plain(Xtr, Xpr, y, tr, va, seeds, rounds, names):
    """普通多 seed 平均 → 在 val 上 z-score（返回与 Xpr 等长的数组）。"""
    acc = np.zeros(len(Xpr))
    for sd in seeds:
        p = dict(PARAMS, seed=sd, bagging_seed=sd, feature_fraction_seed=sd,
                 data_random_seed=sd)
        acc += lgb.train(p, lgb.Dataset(Xtr, y[tr], feature_name=names),
                         num_boost_round=rounds).predict(Xpr)
    ph = acc / len(seeds)
    return zscore(ph, ph[:len(va)])


def main():
    seeds = [int(x) for x in os.environ.get("SEEDS", "42,43,44,45,46").split(",")]
    mv_seeds = [int(x) for x in os.environ.get("MV_SEEDS", "42,43,44").split(",")]
    rounds = int(os.environ.get("ROUNDS", "60"))
    max_h = max(HS)
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

    rows = {}
    full = np.zeros(nmax)
    log("=== 对照: 定稿 LGB 腿（5 horizon × 5 seed）===")
    zs = {}
    for h in HS:
        y = (close[h:h + nmax] > close[:nmax]).astype(np.int8)
        zs[h], _ = fit_z(Xtr, Xpr, y, tr, va, nv, seeds, rounds, cols)
        log(f"[base h={h}] done")
    base_z = np.mean([zs[h] for h in HS], axis=0)
    full[pr] = base_z
    evaluate("base5s", full, ct, y3, va, rows)

    # ---------------- A: 3m 视角的公平对比 ----------------
    log("\n=== A: 3m 重采样（同一批边界行、同一标签）===")
    raw3 = resample_3m(raw)
    ft3, cols3 = build_features(raw3)
    close3 = raw3["close"].to_numpy().astype(np.float64)
    nmax3 = raw3.height - 1
    X3 = ft3.head(nmax3).select(cols3).to_numpy().astype(np.float32)
    ct3 = ft3.head(nmax3)["close_time"].to_numpy()
    y3m = (close3[1:1 + nmax3] > close3[:nmax3]).astype(np.int8)
    tr3 = np.where(ct3 < VAL_START)[0]
    va3 = np.where((ct3 >= VAL_START) & (ct3 < TEST_START))[0]
    te3 = np.where(ct3 >= TEST_START)[0]
    va3 = va3[va3 + 1 < int(te3[0])]          # 标签不跨切分边界
    tr3 = tr3[tr3 + 1 < int(va3[0])]
    pr3 = np.concatenate([va3, te3])
    log(f"[3m] rows={nmax3} train={len(tr3)} val={len(va3)} test={len(te3)} "
        f"3m涨占比={y3m[te3].mean():.4f}")

    s3_grid = np.zeros(nmax3)
    s3_grid[pr3] = fit_plain(X3[tr3], X3[pr3], y3m, tr3, va3, mv_seeds, rounds, cols3)
    evaluate("3m@3m_grid", s3_grid, ct3, y3m, va3, rows, min_bars=400)
    del ft3, X3

    # 1m 定稿分数落在 3m 边界行上（同一批行、同一标签）
    idx_b = np.clip(np.searchsorted(ct, ct3, side="left"), 0, nmax - 1)
    ok = np.flatnonzero(ct[idx_b] == ct3)
    same = y3[idx_b[ok]] == y3m[ok]
    log(f"[3m] 边界行匹配 {len(ok)}/{len(ct3)}；1m(t+3 根) 与 3m(下一根 3m bar) "
        f"标签一致率 {float(same.mean()):.4f}（不一致仅来自分钟缺口）")
    ok_s = ok[same]                      # 严格同标签子集 → 真·头对头
    ct_s, y_s = ct3[ok_s], y3m[ok_s]
    va_s = np.flatnonzero((ct_s >= VAL_START) & (ct_s < TEST_START))
    evaluate("3m@h2h", s3_grid[ok_s], ct_s, y_s, va_s, rows, min_bars=400)
    evaluate("1m@h2h", full[idx_b[ok_s]], ct_s, y_s, va_s, rows, min_bars=400)

    # 3m 分数前向填回 1m 网格（复现第 1 轮的 1/3 密度损失）
    idx_ok = idx_b[ok]
    j = np.clip(np.searchsorted(idx_ok, np.arange(nmax), side="right") - 1, 0, len(ok) - 1)
    s3_ff = s3_grid[ok][j]
    evaluate("3m->1m(ffill)", s3_ff, ct, y3, va, rows)

    # ---------------- B: 多数票目标 ----------------
    log("\n=== B: 多数票目标（评估标签固定 h=3）===")
    mv_z = {}
    for n, k in MVS:
        ups = mv_ups(close, n, nmax)
        ymv = (ups > k).astype(np.int8)
        tag = f"mv{n}>{k}"
        log(f"[{tag}] 正类占比 train={ymv[tr].mean():.4f} val={ymv[va].mean():.4f} "
            f"test={ymv[te].mean():.4f}")
        z, _ = fit_z(Xtr, Xpr, ymv, tr, va, nv, mv_seeds, rounds, cols)
        mv_z[tag] = z
        full[pr] = z
        evaluate(f"{tag}@own", full, ct, ymv, va, rows)
        evaluate(f"{tag}@h3", full, ct, y3, va, rows)
        for w in (0.3, 0.6):
            full[pr] = base_z + w * z
            evaluate(f"base+{w}{tag}", full, ct, y3, va, rows)
        log(f"[{tag}] done")

    if len(mv_z) > 1:
        mz = np.mean(list(mv_z.values()), axis=0)
        full[pr] = mz
        evaluate("mv_mean@h3", full, ct, y3, va, rows)
        full[pr] = base_z + 0.4 * mz
        evaluate("base+0.4mv_mean", full, ct, y3, va, rows)

    log(f"\n{'变体':<20}{'static':>9}{'AUC':>8}{'按天':>8}{'窗口4':>8}{'4段sd':>8}")
    for k, r in rows.items():
        log(f"{k:<20}{r['static']:>9.4f}{r['auc']:>8.4f}{r['daily']:>8.4f}"
            f"{r['w4']:>8.4f}{r['w4sd']:>8.4f}  明细={r['wins']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
