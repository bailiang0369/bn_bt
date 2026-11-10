#!/usr/bin/env python3
"""h=3 事件合约现役自动信号规格（2026-11-10 定稿）。

产品：3 分钟到期事件合约，payout 0.8~0.85，盈亏平衡 1/(1+r) = 55.6% / 54.1%。
模型：纯订单簿 LightGBM 5 seed（`kaggle_launcher.PROJECTS["h3wide"]`, FEATS=ob），
      p = 5 个种子的概率均值，标签 close[t+3] > close[t]。
信号：每根 1m bar 独立判断，|p-0.5| >= THR 才下单，方向 p>0.5 做多、否则做空。
      THR 只在校准段按 |p-0.5| 的 (1-15/1440) 分位定，之后固定不变。
      **不含日内排名、全期分位或任何未来数据**，可直接用于自动执行。

口径（用户 2026-11-10 明确）：滚动判据 = 100 成交单；「日均15单」是全局 top-K
评估口径（仅作对照），不是被控制的每日单量；实盘单量随分布波动，日均只是结果。

校准/验证三段：val 前半（38 天）定阈值 → val 后半选参与下界预算 → test 独立验证。
严禁用 test 调阈值或挑参数。

用法：python -m morph.h3_signal --pred <h3_gate_pred.parquet>
"""
from __future__ import annotations

import argparse
import numpy as np
import polars as pl

THR_DEFAULT = 0.05806
ROLL_N = 100
TPD = 15
BARS_PER_DAY = 1440


def calibrate(p: np.ndarray, calib_mask: np.ndarray, tpd: int = TPD) -> float:
    """阈值只吃校准段的 |p-0.5| 分布；tpd 只是名义目标，实盘单量不由此控制。"""
    return float(np.quantile(np.abs(p[calib_mask] - 0.5), 1 - tpd / BARS_PER_DAY))


def emit(p: np.ndarray, thr: float) -> np.ndarray:
    """0=不下单，1=做多，-1=做空；逐 bar 独立，无跨 bar 依赖。"""
    s = np.asarray(p) - 0.5
    d = np.zeros(len(s), dtype=np.int8)
    hi = np.abs(s) >= thr
    d[hi] = np.where(s[hi] > 0, 1, -1)
    return d


def evaluate(ct: np.ndarray, y: np.ndarray, direction: np.ndarray,
             payout: float = 0.8, roll_n: int = ROLL_N) -> dict:
    """按实际触发的成交单序列评估：总体/月度/滚动100单/连亏/单量分布/每注期望。"""
    idx = np.where(direction != 0)[0]
    if len(idx) < 100:
        raise SystemExit(f"触发单不足：{len(idx)}")
    w = ((direction[idx] > 0).astype(int) == y[idx]).astype(np.int8)
    t, d = ct[idx], ct[idx] // 86_400_000
    nd = len(np.unique(ct // 86_400_000))   # 分母必须是全期自然日，不是有单天数
    _, cnt = np.unique(d, return_counts=True)
    ym = (t // 1000).astype("datetime64[s]").astype("datetime64[M]")
    months = np.unique(ym)
    mv = np.array([w[ym == m].mean() for m in months])
    cs = np.concatenate([[0], np.cumsum(w)])
    r = (cs[roll_n:] - cs[:-roll_n]) / roll_n
    streak = cur = 0
    for v in w:
        cur = cur + 1 if v == 0 else 0
        streak = max(streak, cur)
    tot = float(w.mean())
    return dict(n=len(w), days=nd, per_day=len(w) / nd, order_std=float(cnt.std()),
                order_p90=float(np.percentile(cnt, 90)), order_max=int(cnt.max()),
                traded_days=len(cnt), total=tot,
                se=float(np.sqrt(tot * (1 - tot) / len(w))),
                month_min=float(mv.min()), month_std=float(mv.std()),
                month_bad=int((mv < .5).sum()), months=len(mv),
                roll_min=float(r.min()), roll_bad=float((r < .5).mean()),
                streak=streak, ev=float((1 + payout) * tot - 1),
                breakeven=1 / (1 + payout))


def show(tag: str, r: dict, payout: float) -> None:
    print(f"[{tag}] n={r['n']:,} 日均={r['per_day']:.1f} "
          f"(单量std {r['order_std']:.1f} p90 {r['order_p90']:.0f} max {r['order_max']}, "
          f"有单 {r['traded_days']}/{r['days']} 天)")
    print(f"  总体={r['total'] * 100:.2f}% (SE {r['se'] * 100:.2f}pp) | "
          f"63%:{'过' if r['total'] >= .63 else '未过'} 65%:{'过' if r['total'] >= .65 else '未过'}"
          f" | 盈亏平衡={r['breakeven'] * 100:.2f}%")
    print(f"  月度 min={r['month_min'] * 100:.2f}% std={r['month_std'] * 100:.2f}pp "
          f"破50%={r['month_bad']}/{r['months']} | "
          f"滚动{ROLL_N}单 min={r['roll_min'] * 100:.1f}% 破50%窗口={r['roll_bad'] * 100:.1f}% "
          f"| 连亏={r['streak']}")
    print(f"  每注期望(payout {payout})={r['ev']:+.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="h=3 事件合约自动信号复现与判据体检")
    ap.add_argument("--pred", default=".codebuddy/artifacts/h3gate/h3_gate_pred.parquet")
    ap.add_argument("--thr", type=float, default=THR_DEFAULT,
                    help="留空则用校准段重算；显式给值则固定使用该阈值")
    ap.add_argument("--recalibrate", action="store_true", help="从校准段重算阈值")
    ap.add_argument("--payout", type=float, default=0.8)
    a = ap.parse_args()

    d = pl.read_parquet(a.pred).filter(pl.col("split") == "time")
    va, te = (d.filter(pl.col("segment") == s) for s in ("val", "test"))
    ud = np.unique(va["day"].to_numpy())
    cal = np.isin(va["day"].to_numpy(), ud[:len(ud) // 2])
    thr = calibrate(va["score"].to_numpy(), cal) if a.recalibrate else a.thr
    print(f"阈值 THR={thr:.5f}（校准段 {len(ud) // 2} 天，之后固定）| payout={a.payout}"
          f" | 盈亏平衡={1 / (1 + a.payout) * 100:.2f}%")

    for tag, seg, mask in (("VAL后半", va, ~cal), ("TEST", te, None)):
        ct = seg["close_time"].to_numpy()
        y = seg["target"].to_numpy().astype(np.int8)
        p = seg["score"].to_numpy()
        if mask is not None:
            ct, y, p = ct[mask], y[mask], p[mask]
        show(tag, evaluate(ct, y, emit(p, thr), a.payout), a.payout)

    rng = np.random.default_rng(0)
    yp = te["target"].to_numpy().astype(np.int8).copy()
    rng.shuffle(yp)
    idx = np.where(emit(te["score"].to_numpy(), thr) != 0)[0]
    sig = emit(te["score"].to_numpy(), thr)
    wp = ((sig[idx] > 0).astype(int) == yp[idx]).astype(np.int8)
    print(f"[PLACEBO] 打乱 test 标签后胜率={wp.mean() * 100:.2f}%"
          f" (标签 base={yp.mean() * 100:.2f}%) — 应回落到 base 附近")


if __name__ == "__main__":
    main()
