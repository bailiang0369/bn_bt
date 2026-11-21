#!/usr/bin/env python3
"""结构点起点的 VP（成交密集/稀疏区）特征（用户 2026-11-21 口径）。

口径
----
对每个时刻 t：回看 LOOKBACK 根（默认 150，落在用户指定的 120~180），
在这段里找最高点 argmax(high) 与最低点 argmin(low)，取**时间上离 t 更近**的那个
作为 VP 起点 s（若 s 距 t < MIN_SPAN 根则钳到 t-MIN_SPAN，保证至少 30 根分布）。
VP 直方图 = 从 s 到 t 的成交量按**绝对 log 价格分箱**（固定 tick）；
所有距离/位置都以**当前 bar 的 close** 定位，价格用 log 桶反算成相对 bp。

窗口长度被 LOOKBACK 兜在 120~180 根（≈2~2.5h），不会随 regime 跑到几个月前，
价值区间不失真；起点落在近期真实极值（转折位）而非机械的 t-LOOKBACK。

特征（8 列）
-----------
  av_pos_va     当前价在价值区间(70%量能)内位置：0=下沿 1=上沿 >1 突破 <0 跌破
  av_poc_bp     当前价距 POC 的 bp（相对起始价），正=在 POC 上方
  av_imb        当前价上下方成交量比 (up-dn)/(up+dn)
  av_wide_bp    价值区间宽度 bp
  av_span       起点到当前根数 / LOOKBACK
  av_ret_start  起点到当前收益 bp（方向+幅度）
  av_state      状态离散值 -2..2（跌破/贴下沿/区内/贴上沿/突破）
  av_state_dur  当前状态持续根数（"维持 vs 刚到"）

因果性：起点 s 只由 [t-LOOKBACK+1, t] 的 high/low 决定，直方图只用 [s,t] 的成交量。
改动后先过 avp_leak_check()。
"""
from __future__ import annotations

import time

import numpy as np
import polars as pl

T = time.time()
FEAT_NAMES = ["av_pos_va", "av_poc_bp", "av_imb", "av_wide_bp",
              "av_span", "av_ret_start", "av_state", "av_state_dur"]


def log(msg: str) -> None:
    print(f"[avp {time.time() - T:6.1f}s] {msg}", flush=True)


def _rolling_arg(series: np.ndarray, w: int, mode: str) -> np.ndarray:
    """窗口 [t-w+1, t] 内 argmax/argmin 的绝对索引（单调队列 O(n)）。

    并列时取**最新**（索引最大）的那个——用户口径"取离当前更近的极值点"。
    窗口不足 w 根时返回 -1（该时刻无有效起点）。
    """
    x = np.asarray(series, dtype=np.float64)
    n = len(x)
    out = np.full(n, -1, dtype=np.int64)
    from collections import deque
    dq = deque()  # 存索引，值单调递减(max)/递增(min)
    for i, val in enumerate(x):
        # 弹出窗口外的
        while dq and dq[0] <= i - w:
            dq.popleft()
        # 保持单调；pop 掉更旧且更差(或等于)的，使并列取最新
        if mode == "max":
            while dq and x[dq[-1]] <= val:
                dq.pop()
        else:
            while dq and x[dq[-1]] >= val:
                dq.pop()
        dq.append(i)
        if i >= w - 1:
            out[i] = dq[0]
    return out


def build_avp(df: pl.DataFrame, lookback=150, min_span=30, tick_bp=5.0, cap_bp=300.0):
    """从 vision-base 原始 K 线构造结构点起点 VP；volume 由 um_buy_vol+um_sell_vol 合成。"""
    if "volume" in df.columns:
        v = np.nan_to_num(df["volume"].to_numpy().astype(np.float64))
    else:
        v = np.nan_to_num((df["um_buy_vol"].cast(pl.Float64)
                           + df["um_sell_vol"].cast(pl.Float64)).to_numpy())
    c = df["close"].to_numpy().astype(np.float64)
    h = df["high"].to_numpy().astype(np.float64)
    l = df["low"].to_numpy().astype(np.float64)
    n = len(c)
    pos = np.arange(n)

    # 起点：区间内 max(high)/min(low) 中索引更大（离当前更近）者；太近则钳到 t-min_span
    a_max = _rolling_arg(h, lookback, "max")
    a_min = _rolling_arg(l, lookback, "min")
    start = np.maximum(a_max, a_min)
    start = np.minimum(start, pos - min_span)
    start[start < 0] = 0

    # log 价格分箱（绝对网格，跨时间对齐）；桶 k 的中心 log 价 = k*step
    step = tick_bp * 1e-4
    tp = (h + l + c) / 3.0
    bins = np.floor(np.log(np.maximum(tp, 1e-9)) / step).astype(np.int64)
    logc = np.log(np.maximum(c, 1e-9))
    kernel = np.ones(5, dtype=np.float64) / 5.0

    out = np.zeros((n, 6), dtype=np.float32)  # 前 6 列；state/dur 后补
    # 每根 t 的直方图严格只用 [start[t], t]：不能按 start 分组复用一个跨越未来的直方图，
    # 否则 cut 前、group 未结束的行会吃到 cut 后的 bar（与 §44 的 0.7658 前视同类）。
    for t in range(n):
        s0 = int(start[t])
        base_log = logc[s0]
        out[t, 4] = (t - s0) / float(lookback)
        out[t, 5] = (logc[t] - base_log) * 1e4
        seg_b = bins[s0:t + 1]
        seg_v = v[s0:t + 1]
        if seg_v.sum() <= 0 or seg_b.size < min_span:
            continue
        m = int(seg_b.min())
        hist = np.bincount(seg_b - m, weights=seg_v)
        sm = np.maximum(np.convolve(hist, kernel, mode="same"), 0.0)
        cum = np.concatenate(([0.0], np.cumsum(sm)))
        nb = len(sm)
        order = np.argsort(-sm)
        kk = int(np.searchsorted(np.cumsum(sm[order]), 0.7 * sm.sum())) + 1
        sel = order[:max(1, min(kk, nb))]
        va_lo = (m + int(sel.min())) * step
        va_hi = (m + int(sel.max()) + 1) * step
        poc = (m + int(sm.argmax()) + 0.5) * step
        out[t, 0] = np.clip((logc[t] - va_lo) / max(va_hi - va_lo, 1e-9), -1, 3)
        out[t, 1] = (logc[t] - poc) / step * tick_bp
        cb = int(np.clip(bins[t] - m, 0, nb - 1))
        up = cum[min(cb + 1, nb)] - cum[cb]
        dn = cum[cb] - cum[max(cb - 1, 0)]
        out[t, 2] = (up - dn) / (up + dn + 1e-9)
        out[t, 3] = (va_hi - va_lo) / step * tick_bp

    # state / state_dur（从 av_pos_va 派生，逐行因果）
    pv = out[:, 0]
    state = np.zeros(n, dtype=np.float32)
    state[pv > 1.0] = 2.0
    state[(pv <= 1.0) & (pv > 0.7)] = 1.0
    state[(pv >= -0.3) & (pv <= 0.7)] = 0.0
    state[(pv < -0.3) & (pv >= -1.0)] = -1.0
    state[pv < -1.0] = -2.0
    dur = np.ones(n, dtype=np.float32)
    for i in range(1, n):
        dur[i] = dur[i - 1] + 1 if state[i] == state[i - 1] else 1.0
    full = np.column_stack([out, state, np.minimum(dur, 256.0)]).astype(np.float32)
    log(f"avp built: {full.shape}, 特征 {FEAT_NAMES}")
    return full, list(FEAT_NAMES)


def avp_leak_check(n=20000, cut=17000, lookback=150, min_span=30, seed=0):
    """cut 之后随机化行情，cut 之前输出必须逐位不变。"""
    rng = np.random.default_rng(seed)
    c = 60000 * np.exp(np.cumsum(rng.normal(0, 3e-4, n)))
    o = c * (1 + rng.normal(0, 1e-4, n))
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 2e-4, n)))
    l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 2e-4, n)))
    v = rng.lognormal(3, 1, n)
    df1 = pl.DataFrame({"close": c, "high": h, "low": l, "volume": v})
    x1, _ = build_avp(df1, lookback=lookback, min_span=min_span)
    bad = pl.DataFrame({"close": c[cut:] * 1.3, "high": h[cut:] * 2.0,
                        "low": l[cut:] * 0.5, "volume": v[cut:] * 5.0})
    df2 = pl.concat([df1.head(cut), bad], how="vertical")
    x2, _ = build_avp(df2, lookback=lookback, min_span=min_span)
    bad_n = int((x1[:cut] != x2[:cut]).sum())
    log(f"泄露自检：cut 前不一致元素 {bad_n} -> {'通过' if bad_n == 0 else '失败!'}")
    return bad_n == 0
