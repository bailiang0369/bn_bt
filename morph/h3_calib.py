#!/usr/bin/env python3
"""h=3 现役信号的**概率校准**体检（零训练成本，本地可跑）。

为什么这件事比换模型更值得做
----------------------------
`h3_signal.py` 的阈值 0.05806 是「|p-0.5| 的 98.96 分位」，这里隐含一个从未验证的
假设：**score 就是真实概率**。如果模型系统性过度自信（score=0.65 时实际只有 0.60），
那么 p=0.558 的触发点对应的真实概率可能已经跌破 55.56% 的盈亏平衡——
阈值本身在系统性地透支期望。

本脚本只回答三件事
------------------
1. score 校准吗？分箱看「平均预测 p」vs「实际命中率」，两个时间段各做一份。
   val 与 test 的校准曲线如果一致 → 不需要校准；如果 test 系统性低于 val → 漂移。
2. 校准能改善多少？在 val 上学等频分箱映射（isotonic 的简化版），test 上算
   Brier / log-loss 的前后对比，以及 Brier Skill Score。
3. **触发单上到底差多少**：现役触发的单，预测 p 的均值 vs 实际胜率。
   这是唯一能判断「阈值是否透支期望」的数字。

判据
----
- 校准良好：分箱残差（预测-实际）绝对值 < 1pp，且单调。
- 过度自信：高分区预测显著高于实际 → 阈值应上调。
- 保守：低分区预测显著低于实际 → 阈值可下调。

用法
----
    python -m morph.h3_calib
    python -m morph.h3_calib --pred .codebuddy/artifacts/h3gate/h3_gate_pred.parquet
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import polars as pl

from morph.h3_signal import calibrate, emit, evaluate

N_BINS = 20
MIN_BIN = 500


def log(m: str) -> None:
    print(m, flush=True)


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def logloss(p: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def calib_table(p: np.ndarray, y: np.ndarray, n_bins: int = N_BINS) -> list[dict]:
    """等频分箱：每箱看平均预测概率 vs 实际命中率。"""
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    bid = np.digitize(p, edges[1:-1])
    rows = []
    for b in range(n_bins):
        m = bid == b
        if m.sum() < MIN_BIN:
            continue
        rows.append(dict(bin=b, n=int(m.sum()), p_mean=float(p[m].mean()),
                         y_rate=float(y[m].mean()),
                         resid=float(p[m].mean() - y[m].mean())))
    return rows


def fit_isotonic(p: np.ndarray, y: np.ndarray, n_bins: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """等频分箱 + 单调化（PAVA 的简化版）：返回 (bin 中心, 命中率) 单调序列。"""
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    bid = np.digitize(p, edges[1:-1])
    xs, ys, ws = [], [], []
    for b in range(n_bins):
        m = bid == b
        if m.sum() < 200:
            continue
        xs.append(float(np.mean(p[m])))
        ys.append(float(np.mean(y[m])))
        ws.append(int(m.sum()))
    xs, ys, ws = np.array(xs), np.array(ys), np.array(ws, dtype=np.float64)
    # PAVA（加权单调不减）
    ys_iso = ys.copy()
    w = ws.copy()
    i = 0
    while i < len(ys_iso) - 1:
        if ys_iso[i] > ys_iso[i + 1]:
            s = (ys_iso[i] * w[i] + ys_iso[i + 1] * w[i + 1]) / (w[i] + w[i + 1])
            ys_iso[i] = ys_iso[i + 1] = s
            w[i] = w[i + 1] = w[i] + w[i + 1]
            i = max(i - 1, 0)
        else:
            i += 1
    return xs, ys_iso


def apply_iso(p: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    return np.interp(p, xs, ys)


def main() -> None:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    ap = argparse.ArgumentParser(description="h=3 现役信号概率校准体检")
    ap.add_argument("--pred", default=".codebuddy/artifacts/h3gate/h3_gate_pred.parquet")
    ap.add_argument("--payout", type=float, default=0.8)
    a = ap.parse_args()

    d = pl.read_parquet(a.pred).filter(pl.col("split") == "time").sort("close_time")
    va = d.filter(pl.col("segment") == "val")
    te = d.filter(pl.col("segment") == "test")
    vp, vy = va["score"].to_numpy(), va["target"].to_numpy().astype(np.int8)
    tp, ty = te["score"].to_numpy(), te["target"].to_numpy().astype(np.int8)

    log("=" * 88)
    log(f"pred={a.pred}")
    log(f"val {len(vp):,} 行 (base {vy.mean() * 100:.2f}%) | "
        f"test {len(tp):,} 行 (base {ty.mean() * 100:.2f}%)")
    log(f"score 分布: val [{vp.min():.4f},{vp.max():.4f}] 均值 {vp.mean():.4f} | "
        f"test [{tp.min():.4f},{tp.max():.4f}] 均值 {tp.mean():.4f}")
    log("=" * 88)

    # ---------- 1. 校准曲线
    for tag, p, y in (("VAL", vp, vy), ("TEST", tp, ty)):
        log("")
        log(f"[{tag}] 等频 {N_BINS} 箱校准曲线")
        log(f"  {'bin':>3}{'n':>8}{'预测p':>9}{'实际':>9}{'残差':>9}")
        rows = calib_table(p, y)
        for r in rows:
            flag = "  <-- 过度自信" if r["resid"] > 0.015 else (
                "  <-- 偏保守" if r["resid"] < -0.015 else "")
            log(f"  {r['bin']:>3}{r['n']:>8}{r['p_mean'] * 100:>8.2f}%"
                f"{r['y_rate'] * 100:>8.2f}%{r['resid'] * 100:>+8.2f}pp{flag}")
        log(f"  Brier={brier(p, y):.6f}  base-Brier={y.mean() * (1 - y.mean()):.6f}  "
            f"BSS={1 - brier(p, y) / (y.mean() * (1 - y.mean())):+.4f}  "
            f"logloss={logloss(p, y):.6f}")

    # ---------- 2. val 上学映射，test 上验证
    log("")
    log("[校准映射] val 学等频分箱 + PAVA 单调化 → test 应用")
    xs, ys = fit_isotonic(vp, vy)
    tp_cal = apply_iso(tp, xs, ys)
    b0, b1 = brier(tp, ty), brier(tp_cal, ty)
    l0, l1 = logloss(tp, ty), logloss(tp_cal, ty)
    log(f"  test Brier : {b0:.6f} → {b1:.6f}  ({'改善' if b1 < b0 else '变差'} {abs(b1 - b0) / b0 * 100:.2f}%)")
    log(f"  test logloss: {l0:.6f} → {l1:.6f}")
    log(f"  score 均值平移: {tp.mean():.4f} → {tp_cal.mean():.4f}")
    log("  映射采样（原始 p → 校准 p）:")
    for q in (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99):
        x = float(np.quantile(tp, q))
        log(f"    p={x:.4f} → {float(apply_iso(np.array([x]), xs, ys)[0]):.4f}")

    # ---------- 3. 触发单上差多少（关键）
    log("")
    log("[关键] 现役阈值触发单上的「预测 p」vs「实际胜率」")
    ud = np.unique(va["day"].to_numpy())
    cal = np.isin(va["day"].to_numpy(), ud[: len(ud) // 2])
    thr = calibrate(vp, cal)
    for tag, p, y in (("VAL后半", vp[~cal], vy[~cal]), ("TEST", tp, ty)):
        dirs = emit(p, thr)
        idx = np.where(dirs != 0)[0]
        if len(idx) < 50:
            continue
        # 模型对自己这一侧的概率：做多取 p，做空取 1-p
        p_side = np.where(dirs[idx] > 0, p[idx], 1 - p[idx])
        win = ((dirs[idx] > 0).astype(int) == y[idx])
        log(f"  {tag:<8} n={len(idx):>5}  预测(p_side)均值={p_side.mean() * 100:.2f}%  "
            f"实际胜率={win.mean() * 100:.2f}%  **残差={(p_side.mean() - win.mean()) * 100:+.2f}pp**")
        # 按 p_side 分档
        qs = np.quantile(p_side, [0.33, 0.66])
        for k, lab in enumerate(("低", "中", "高")):
            lo = -np.inf if k == 0 else qs[k - 1]
            hi = np.inf if k == 2 else qs[k]
            m = (p_side >= lo) & (p_side < hi)
            if m.sum() < 30:
                continue
            log(f"      {lab}档 n={int(m.sum()):>5} 预测={p_side[m].mean() * 100:.2f}% "
                f"实际={win[m].mean() * 100:.2f}% 残差={(p_side[m].mean() - win[m].mean()) * 100:+.2f}pp")

    # ---------- 4. 用校准概率重定阈值，看 EV 变化
    log("")
    log("[反事实] 若用校准后的概率重新按 98.96 分位定阈值（val 前半→test）")
    vp_cal = apply_iso(vp, xs, ys)
    tp_cal = apply_iso(tp, xs, ys)
    thr_cal = calibrate(vp_cal, cal)
    log(f"  原阈值 {thr:.5f} → 校准后阈值 {thr_cal:.5f}")
    for tag, p, thr_x in (("原 score", tp, thr), ("校准后", tp_cal, thr_cal)):
        dirs = emit(p, thr_x)
        try:
            r = evaluate(te["close_time"].to_numpy(), ty, dirs, a.payout)
        except SystemExit:
            log(f"  {tag}: 触发单不足")
            continue
        log(f"  {tag:<9} n={r['n']:>5} 日均={r['per_day']:.1f} 总体={r['total'] * 100:.2f}% "
            f"EV={r['ev']:+.3f} 滚动100min={r['roll_min'] * 100:.1f}%")
    log("=" * 88)


if __name__ == "__main__":
    main()
