#!/usr/bin/env python3
"""滚动分位阈值（rolling quantile threshold）实验 —— 2026-11-11。

动机
----
现役信号 `h3_signal.py` 用**固定绝对阈值** THR=0.05806（val 前半 38 天的 98.96 分位）。
test 段分数分布漂移 → 触发量从名义 15 单/天 膨胀到 40.6 单/天（2.8% 覆盖），
选择性被稀释，总体 = 62.85%；而评估口径 @日均15单（1.04% 覆盖）= 66.94%。

本实验把阈值换成**滚动分位**：
    第 d 天的阈值 = 过去 LOOKBACK 天（**不含 d**）的 |p-0.5| 的 (1 - tpd/1440) 分位
纯因果：只用 d 之前已收盘的数据，不含日内排名 / 全期分位 / 未来数据，
因此可直接用于自动执行（与「每天固定 top15」的事后挑单有本质区别）。

判据（沿用 `h3宽表实验记录.md` 记录规则 §4）
-------------------------------------------
主判据：按天块 bootstrap 的两条线总体胜率差值 CI95 是否不跨 0。
次判据：滚动 100 单最差、破 50% 窗口占比、月度 min —— 稳定性是否改善。
参照：固定阈值 62.85%；全期 top-K 66.94%（有前视，仅作上界参考，不是可达值）。

不做参数搜索：LOOKBACK=30 是先验选择（≈一个月，覆盖一个行情周期），
14 / 60 与 tpd 10/20 仅作敏感性展示，**不作为选参依据**。

用法
----
    python -m morph.h3_rollthr
    python -m morph.h3_rollthr --pred <path> --target-tpd 15 --lookback 30
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import polars as pl

from morph.h3_signal import BARS_PER_DAY, ROLL_N, TPD, calibrate, emit, evaluate, show

MS_DAY = 86_400_000


# ---------------------------------------------------------------- 阈值


def day_thresholds(days_all: np.ndarray, abs_all: np.ndarray, target_days: np.ndarray,
                   lookback: int, q: float, min_hist_days: int | None = None) -> dict[int, float]:
    """逐日滚动阈值。{day: thr}，thr 只用 [day-lookback, day) 的 |s| 计算。

    days_all / abs_all 需为**按时间升序**的整段历史（val+test 拼起来传）。
    min_hist_days：历史天数不足时返回 nan（当日不下单），避免用太少的样本估分位。
                   默认 max(2, lookback//2)，短 lookback 时放宽，否则窗口永远不够。
    """
    if min_hist_days is None:
        min_hist_days = max(2, lookback // 2)
    order = np.argsort(days_all, kind="stable")
    ds, ab = days_all[order], abs_all[order]
    uniq, start = np.unique(ds, return_index=True)
    end = np.append(start[1:], len(ds))
    out: dict[int, float] = {}
    for d in target_days:
        d = int(d)
        lo = int(np.searchsorted(uniq, d - lookback, side="left"))
        hi = int(np.searchsorted(uniq, d, side="left"))      # 不含 d 当天
        if hi - lo < min_hist_days:
            out[d] = np.nan
            continue
        out[d] = float(np.quantile(ab[start[lo]:end[hi - 1]], q))
    return out


def feedback_threshold(days_all: np.ndarray, abs_all: np.ndarray, target_days: np.ndarray,
                       target_tpd: int, init_thr: float, kp: float = 0.3,
                       lo: float = 0.03, hi: float = 0.15) -> dict[int, float]:
    """反馈控制阈值：用**上一交易日实际触发量**的误差调整今日阈值。纯因果。

    与滚动分位的区别：分位法测的是「分数分布」，反馈法直接测「实际触发率」，
    因此对分数整体平移/缩放都不敏感，能真正把日均压到目标附近。
    kp=0.3 为先验增益，不搜索。
    """
    order = np.argsort(days_all, kind="stable")
    ds, ab = days_all[order], abs_all[order]
    uniq, start = np.unique(ds, return_index=True)
    end = np.append(start[1:], len(ds))
    pos = {int(d): i for i, d in enumerate(uniq)}
    out: dict[int, float] = {}
    prev_thr, prev_n = init_thr, float(target_tpd)
    for d in target_days:
        d = int(d)
        out[d] = float(np.clip(prev_thr, lo, hi))
        i = pos.get(d)
        if i is None:
            continue
        seg = ab[start[i]:end[i]]
        n_hit = int((seg >= out[d]).sum())
        prev_thr = out[d] * (1 + kp * (n_hit / max(target_tpd, 1) - 1.0))
        prev_n = n_hit
    return out


def emit_arr(p: np.ndarray, thr: np.ndarray) -> np.ndarray:
    """逐 bar 阈值版下单方向：0=不动，1=多，-1=空。thr 为 nan 的 bar 一律不下。"""
    s = np.asarray(p, dtype=np.float64) - 0.5
    t = np.asarray(thr, dtype=np.float64)
    d = np.zeros(len(s), dtype=np.int8)
    ok = np.isfinite(t) & (np.abs(s) >= t)
    d[ok] = np.where(s[ok] > 0, 1, -1)
    return d


# ---------------------------------------------------------------- 判据


def block_boot(day: np.ndarray, win: np.ndarray, n_boot: int = 4000,
               seed: int = 0) -> np.ndarray:
    """按天块重采样（保留日内全部单序结构），返回总体胜率的 bootstrap 分布。"""
    uniq, inv = np.unique(day, return_inverse=True)
    inv = np.asarray(inv).ravel()
    sums = np.bincount(inv, weights=win.astype(np.float64), minlength=len(uniq))
    cnts = np.bincount(inv, minlength=len(uniq)).astype(np.float64)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    return sums[pick].sum(1) / cnts[pick].sum(1)


def wins(ct: np.ndarray, y: np.ndarray, direction: np.ndarray):
    idx = np.where(direction != 0)[0]
    w = ((direction[idx] > 0).astype(int) == y[idx]).astype(np.int8)
    return ct[idx], w


def topk_eval(ct: np.ndarray, y: np.ndarray, s: np.ndarray, per_day: int,
              daily: bool = False) -> tuple[int, float]:
    """top-K（**有前视**，仅作上界参照）。

    daily=False：全局 top-K = 在全部 test bar 里挑 |s-0.5| 最大的 k 个（k=per_day×天数），
                 这是最强的前视口径（可跨天挑最强时刻），对应记录里的 66.94%。
    daily=True ：每天各取 |s-0.5| 最大的 per_day 个（日内排名前视，记录里 63~65%）。
    """
    day = ct // MS_DAY
    if daily:
        keep = []
        for d in np.unique(day):
            i = np.where(day == d)[0]
            keep.append(i[np.argsort(-np.abs(s[i] - 0.5))[:min(per_day, len(i))]])
        idx = np.concatenate(keep)
    else:
        k = min(per_day * len(np.unique(day)), len(s))
        idx = np.argsort(-np.abs(s - 0.5))[:k]
    w = ((s[idx] > 0.5).astype(int) == y[idx])
    return len(idx), float(w.mean())


def thr_group_win(ct, y, s, thr_row, direction, n_grp: int = 3) -> list[tuple[float, int, float]]:
    """按当日阈值高低分组的胜率 —— 诊断「弱日强行开单」是否伤质量。"""
    day = ct // MS_DAY
    dthr = {int(d): float(np.nanmedian(thr_row[day == d])) for d in np.unique(day)}
    order = np.array(sorted(dthr, key=lambda k: dthr[k]), dtype=np.int64)
    idx = np.where(direction != 0)[0]
    w = ((direction[idx] > 0).astype(int) == y[idx])
    di = day[idx]
    out = []
    for chunk in np.array_split(order, n_grp):
        m = np.isin(di, chunk)
        if m.sum() == 0:
            continue
        out.append((float(np.median([dthr[k] for k in chunk])), int(m.sum()), float(w[m].mean())))
    return out


# ---------------------------------------------------------------- 主体


def run(pred: str, lookback: int, tpd: int, payout: float, base_thr: float | None) -> None:
    d = pl.read_parquet(pred).filter(pl.col("split") == "time").sort("close_time")
    va = d.filter(pl.col("segment") == "val")
    te = d.filter(pl.col("segment") == "test")

    # ---- 固定阈值基线（复现 62.85%）
    ud = np.unique(va["day"].to_numpy())
    cal = np.isin(va["day"].to_numpy(), ud[: len(ud) // 2])
    thr0 = calibrate(va["score"].to_numpy(), cal) if base_thr is None else base_thr

    vct, vy, vp, vday = (va[c].to_numpy() for c in ("close_time", "target", "score", "day"))
    tct, ty, tp, tday = (te[c].to_numpy() for c in ("close_time", "target", "score", "day"))
    vy = vy.astype(np.int8)
    ty = ty.astype(np.int8)

    print("=" * 96)
    print(f"pred={pred}")
    print(f"val {len(vp):,} 行 / {len(ud)} 天 | test {len(tp):,} 行 / {len(np.unique(tday))} 天")
    print(f"固定阈值 THR={thr0:.5f} | 滚动 lookback={lookback} 天 target={tpd} 单/天 "
          f"(q={1 - tpd / BARS_PER_DAY:.5f}) | payout={payout} 平衡点={1 / (1 + payout) * 100:.2f}%")
    print("=" * 96)

    # ---- val 后半：两条线对照（滚动历史只吃 val 前半，样本不足仅作 sanity）
    vmask = ~cal
    r0 = evaluate(vct[vmask], vy[vmask], emit(vp[vmask], thr0), payout)
    show("VAL后半·固定阈值", r0, payout)

    hist_day = vday[cal]
    hist_abs = np.abs(vp[cal] - 0.5)
    vthr = day_thresholds(hist_day, hist_abs, np.unique(vday[vmask]),
                          lookback, 1 - tpd / BARS_PER_DAY)
    vthr_row = np.array([vthr.get(int(x), np.nan) for x in vday[vmask]])
    vr = evaluate(vct[vmask], vy[vmask], emit_arr(vp[vmask], vthr_row), payout)
    show("VAL后半·滚动分位", vr, payout)

    # ---- test：固定 vs 滚动
    print("-" * 96)
    r_fix = evaluate(tct, ty, emit(tp, thr0), payout)
    show("TEST·固定阈值(现役)", r_fix, payout)

    all_day = np.concatenate([vday, tday])
    all_abs = np.concatenate([np.abs(vp - 0.5), np.abs(tp - 0.5)])
    tthr = day_thresholds(all_day, all_abs, np.unique(tday), lookback, 1 - tpd / BARS_PER_DAY)
    tthr_row = np.array([tthr.get(int(x), np.nan) for x in tday])
    dir_roll = emit_arr(tp, tthr_row)
    r_roll = evaluate(tct, ty, dir_roll, payout)
    show(f"TEST·滚动分位(L={lookback},tpd={tpd})", r_roll, payout)

    print(f"  阈值序列 min={np.nanmin(tthr_row):.5f} p25={np.nanpercentile(tthr_row, 25):.5f} "
          f"中位={np.nanmedian(tthr_row):.5f} p75={np.nanpercentile(tthr_row, 75):.5f} "
          f"max={np.nanmax(tthr_row):.5f}（固定阈值={thr0:.5f}）")

    # ---- 变体 B：反馈控制阈值（直接按上一日实际触发量调阈值）
    fb = feedback_threshold(all_day, all_abs, np.unique(tday), tpd, thr0)
    fb_row = np.array([fb.get(int(x), np.nan) for x in tday])
    dir_fb = emit_arr(tp, fb_row)
    r_fb = evaluate(tct, ty, dir_fb, payout)
    show(f"TEST·反馈控制(tpd={tpd})", r_fb, payout)
    print(f"  阈值序列 min={np.nanmin(fb_row):.5f} 中位={np.nanmedian(fb_row):.5f} "
          f"max={np.nanmax(fb_row):.5f}（固定阈值={thr0:.5f}）")

    # ---- 主判据：按天块 bootstrap 差值
    ctf, wf = wins(tct, ty, emit(tp, thr0))
    bf_bs = block_boot(ctf // MS_DAY, wf)
    for tag, dirs in (("滚动分位", dir_roll), ("反馈控制", dir_fb)):
        ct_x, w_x = wins(tct, ty, dirs)
        bx = block_boot(ct_x // MS_DAY, w_x)
        df = bx - bf_bs
        lo, hi = np.percentile(df, [2.5, 97.5])
        print(f"  [主判据] {tag}-固定 = {(w_x.mean() - wf.mean()) * 100:+.2f}pp "
              f"CI95 [{lo * 100:+.2f}, {hi * 100:+.2f}] → "
              f"{'不跨0，显著' if lo > 0 or hi < 0 else '跨0，不显著'}")

    # ---- 诊断：弱日强行开单是否伤质量
    print("  [诊断] 按当日阈值三分组（低→高）：")
    for t, n, w in thr_group_win(tct, ty, tp, tthr_row, dir_roll):
        print(f"      阈值中位={t:.5f}  n={n:5d}  胜率={w * 100:.2f}%")

    # ---- 敏感性（不作选参依据）
    print("-" * 96)
    print("[敏感性，仅展示，不据此选参]")

    def line(tag: str, dirs: np.ndarray, ct: np.ndarray | None = None,
             y: np.ndarray | None = None) -> None:
        ct = tct if ct is None else ct
        y = ty if y is None else y
        try:
            rr = evaluate(ct, y, dirs, payout)
        except SystemExit:
            print(f"  {tag} | 触发单不足 100，跳过")
            return
        print(f"  {tag} | n={rr['n']:5d} 日均={rr['per_day']:5.1f} "
              f"总体={rr['total'] * 100:.2f}% 月min={rr['month_min'] * 100:.2f}% "
              f"滚动100min={rr['roll_min'] * 100:.1f}% 破50%={rr['roll_bad'] * 100:.1f}% "
              f"连亏={rr['streak']:2d} EV={rr['ev']:+.3f}")

    for lb in (7, 14, 60):
        th = day_thresholds(all_day, all_abs, np.unique(tday), lb, 1 - tpd / BARS_PER_DAY)
        line(f"lookback={lb:2d}", emit_arr(tp, np.array([th.get(int(x), np.nan) for x in tday])))
    for tp_ in (5, 8, 10, 20, 30):
        th = day_thresholds(all_day, all_abs, np.unique(tday), lookback, 1 - tp_ / BARS_PER_DAY)
        line(f"tpd={tp_:2d}     ", emit_arr(tp, np.array([th.get(int(x), np.nan) for x in tday])))

    # ---- 覆盖率-胜率曲线（绝对阈值扫描）。**诊断用，禁止据此在 test 上挑阈值**
    print("-" * 96)
    print("[覆盖率-胜率曲线] 绝对阈值扫描，**诊断用，禁止据此在 test 上挑阈值**")
    print("  阈值     n     日均   覆盖率   总体    月min   滚动100min 破50%  连亏   EV")
    for th0 in (0.050, 0.055, 0.060, 0.065, 0.070, 0.075, 0.080, 0.085, 0.090):
        dirs = emit(tp, th0)
        k = int((dirs != 0).sum())
        if k < 100:
            continue
        rr = evaluate(tct, ty, dirs, payout)
        cov = k / len(tp)
        print(f"  {th0:.3f}  {rr['n']:5d}  {rr['per_day']:5.1f}  {cov * 100:6.2f}%  "
              f"{rr['total'] * 100:6.2f}%  {rr['month_min'] * 100:6.2f}%  "
              f"{rr['roll_min'] * 100:8.1f}%  {rr['roll_bad'] * 100:5.1f}%  "
              f"{rr['streak']:3d}  {rr['ev']:+.3f}")

    # ---- 时间外推：前 48 天定阈值 → 后 48 天验证（这才是真实的部署流程）
    print("-" * 96)
    print(f"[时间外推] 前段定阈值→后段验证（目标 {tpd} 单/天）")
    ds = np.unique(tday)
    mid = ds[len(ds) // 2]
    head = tday <= mid
    tail = ~head
    q = 1 - tpd / BARS_PER_DAY
    thr_h = float(np.quantile(np.abs(tp[head] - 0.5), q))
    print(f"  前段 {head.sum():,} 行 / {int(head.sum() / 1440)} 天 → 定 THR={thr_h:.5f}"
          f"（前段日均 {float((np.abs(tp[head] - 0.5) >= thr_h).sum()) / 48:.1f} 单）")
    line("  后段·该阈值  ", emit(tp[tail], thr_h), tct[tail], ty[tail])
    print(f"  后段·现役阈值 {thr0:.5f}")
    line("  后段·现役    ", emit(tp[tail], thr0), tct[tail], ty[tail])

    # ---- 上界参照（有前视，不可执行；用来校准「因果 vs 前视」的代价）
    print("-" * 96)
    for k in (10, 15, 20):
        n, acc = topk_eval(tct, ty, tp, k, daily=False)
        print(f"  [上界·前视] 全局 top{k}×天数 | n={n:,} 胜率={acc * 100:.2f}%")
        n, acc = topk_eval(tct, ty, tp, k, daily=True)
        print(f"  [上界·前视] 每天 top{k:2d}      | n={n:,} 胜率={acc * 100:.2f}%")

    # ---- placebo（单次 shuffle 的方差 ≈1pp，必须多次平均才是无偏估计）
    rng = np.random.default_rng(0)
    idx = np.where(dir_roll != 0)[0]
    accs = []
    for _ in range(30):
        yp = ty.copy()
        rng.shuffle(yp)
        accs.append(float(((dir_roll[idx] > 0).astype(int) == yp[idx]).mean()))
    print(f"  [PLACEBO] 滚动分位 × 30 次打乱标签：胜率均值={np.mean(accs) * 100:.2f}% "
          f"范围 {np.min(accs) * 100:.2f}~{np.max(accs) * 100:.2f}% "
          f"(标签 base={ty.mean() * 100:.2f}%) — 应≈50%")
    print("=" * 96)


def main() -> None:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")   # Windows 控制台默认 GBK，会编码失败
        except Exception:
            pass
    ap = argparse.ArgumentParser(description="h=3 滚动分位阈值实验（因果、可直接执行）")
    ap.add_argument("--pred", default=".codebuddy/artifacts/h3gate/h3_gate_pred.parquet")
    ap.add_argument("--lookback", type=int, default=30, help="滚动历史天数，先验 30，不搜索")
    ap.add_argument("--target-tpd", type=int, default=TPD, help="名义单量/天，决定分位 q")
    ap.add_argument("--payout", type=float, default=0.8)
    ap.add_argument("--thr", type=float, default=None, help="固定阈值基线，留空则按 val 前半重算")
    a = ap.parse_args()
    run(a.pred, a.lookback, a.target_tpd, a.payout, a.thr)


if __name__ == "__main__":
    main()
