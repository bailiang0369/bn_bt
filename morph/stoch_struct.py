#!/usr/bin/env python3
"""Stoch 结构点（带确认时间）的形态特征：逐行确认制，无重绘 / 无前视。

口径（用户 2026-11-21 notebook：stoch极值点计算_final.ipynb）
-------------------------------------------------------------
- Stochastic(k_period, d_smooth, d_period)：stoch_k>OB 记超买块，<OS 记超卖块；
- 当第 i 个状态块出现、且与 i-2 块同号时，在 [block[i-2].last, block[i].first]
  区间取极值确认一个结构点（新超买块→确认波谷，新超卖块→确认波峰），
  `confirmed_at = block[i].first`；**确认之前该点不被认可**。
- 用户 notebook 的 `apply_logic_with_conf` 是全局清洗，会让已确认点被未来更极端点
  事后替换（重绘）。本实现改为**逐行推进**：t 时刻只把 `confirmed_at==t` 的点并入，
  同类型更极端者才替换当前确认点——历史回放与实盘决策一致。

特征（17 列，事件驱动的变长结构，现有 49 列固定窗口 rolling 没有的维度）
-------------------------------------------------------------------------
当前段（最近确认点 P → t）：
  sc_seg_len      段长 / k_period
  sc_ret_bp       当前价相对 P 价位 bp
  sc_slope        ret / 段长
  sc_hi_bp        段内最高价相对 P bp
  sc_lo_bp        段内最低价相对 P bp
  sc_pos          当前价在段内 [lo,hi] 的位置
  sc_ptype        P 类型（1=峰 -1=谷）
  sc_since_conf   距 P 被确认过去多少根 / k_period
历史段（最近 N_HIST 个已确认点之间的段）：
  sc_prev_len     前几段平均长度 / k_period
  sc_prev_absret  前几段平均 |幅度| bp
  sc_prev_slope   前几段平均斜率
  sc_slope_div    当前斜率 − 前段斜率（背离/加速度）
  sc_dir_alt      最近几段方向交替比例
确认状态：
  sc_stoch_k / sc_stoch_d / sc_zone（当前状态 1/-1/0）/ sc_k_gap（当前 k − 确认时 k）
"""
from __future__ import annotations

import time

import numpy as np
import polars as pl

FEAT_NAMES = [
    "sc_seg_len", "sc_ret_bp", "sc_slope", "sc_hi_bp", "sc_lo_bp", "sc_pos",
    "sc_ptype", "sc_since_conf", "sc_prev_len", "sc_prev_absret", "sc_prev_slope",
    "sc_slope_div", "sc_dir_alt", "sc_stoch_k", "sc_stoch_d", "sc_zone", "sc_k_gap",
]
T = time.time()


def log(msg: str) -> None:
    print(f"[sc {time.time() - T:6.1f}s] {msg}", flush=True)


def _stoch(df: pl.DataFrame, k_period: int, d_smooth: int, d_period: int):
    """与 notebook cell1 完全一致的 Stochastic(k,s,d)。"""
    out = df.with_columns([
        pl.col("low").rolling_min(k_period).alias("lowest_low"),
        pl.col("high").rolling_max(k_period).alias("highest_high"),
    ])
    out = out.with_columns([
        (100 * (pl.col("close") - pl.col("lowest_low"))
         / (pl.col("highest_high") - pl.col("lowest_low") + 1e-12)).alias("raw_k")
    ])
    out = out.with_columns([
        pl.col("raw_k").rolling_mean(d_smooth).alias("stoch_k"),
    ])
    out = out.with_columns([
        pl.col("stoch_k").rolling_mean(d_period).alias("stoch_d"),
    ])
    return out


def _candidates(stoch_k: np.ndarray, state: np.ndarray, high: np.ndarray, low: np.ndarray):
    """返回 (idx, point_type, confirmed_at, stoch_k_at_confirm) 数组，按 confirmed_at 排序。"""
    n = len(state)
    tmp = pl.DataFrame({"index": np.arange(n), "state": state}).filter(pl.col("state") != 0)
    if tmp.height == 0:
        return np.empty(0, np.int64), np.empty(0, np.int8), np.empty(0, np.int64), np.empty(0, np.float32)
    tmp = tmp.with_columns(
        (pl.col("state") != pl.col("state").shift(1)).fill_null(True).cum_sum().alias("gid")
    )
    blocks = tmp.group_by("gid").agg([
        pl.col("index").min().alias("first_idx"),
        pl.col("index").max().alias("last_idx"),
        pl.col("state").first().alias("state"),
    ]).sort("gid").to_dicts()

    idxs, types, cats, kvals = [], [], [], []
    for i in range(2, len(blocks)):
        prev, curr = blocks[i - 2], blocks[i]
        if prev["state"] != curr["state"]:
            continue
        s, e = int(prev["last_idx"]), int(curr["first_idx"])
        if e <= s:
            continue
        cat = int(curr["first_idx"])
        if curr["state"] == 1:            # 新波峰块 → 确认上一个波谷
            rel = int(np.argmin(low[s:e + 1]))
            idxs.append(s + rel); types.append(-1)
        else:                              # 新波谷块 → 确认上一个波峰
            rel = int(np.argmax(high[s:e + 1]))
            idxs.append(s + rel); types.append(1)
        cats.append(cat)
        kvals.append(float(stoch_k[cat]) if cat < n else float(stoch_k[-1]))
    if not idxs:
        return np.empty(0, np.int64), np.empty(0, np.int8), np.empty(0, np.int64), np.empty(0, np.float32)
    order = np.argsort(cats)
    return (np.asarray(idxs)[order], np.asarray(types, dtype=np.int8)[order],
            np.asarray(cats)[order], np.asarray(kvals, dtype=np.float32)[order])


def build_struct(df: pl.DataFrame, k_period=80, d_smooth=2, d_period=2,
                 ob=80.0, os=20.0, n_hist=4, return_points=False):
    """从 vision-base 原始 K 线（含 close_time/OHLC）构造 17 列结构特征。

    return_points=True 时额外返回逐行确认点数组 (ptype_arr, pidx_arr, px_arr)：
    每行记录"截至该行最近确认点"的类型/位置/价位（无前视，forward-fill 语义），
    供下一确认点目标（持续时间/极值/振幅）诊断使用。
    """
    d = _stoch(df.select(["close_time", "open", "high", "low", "close"]),
               k_period, d_smooth, d_period)
    sk = d["stoch_k"].to_numpy().astype(np.float64)
    sd = d["stoch_d"].to_numpy().astype(np.float64)
    close = d["close"].to_numpy().astype(np.float64)
    high = d["high"].to_numpy().astype(np.float64)
    low = d["low"].to_numpy().astype(np.float64)
    n = len(close)
    state = np.where(sk > ob, 1, np.where(sk < os, -1, 0)).astype(np.int8)
    ci, ct, cc, ck = _candidates(sk, state, high, low)
    P = len(cc)

    out = np.zeros((n, len(FEAT_NAMES)), dtype=np.float32)
    ptype_arr = np.zeros(n, dtype=np.int8)      # 最近确认点类型（forward-fill）
    pidx_arr = np.full(n, -1, dtype=np.int64)   # 最近确认点位置
    px_arr = np.zeros(n, dtype=np.float64)      # 最近确认点价位
    fresh = np.zeros(n, dtype=bool)             # 该行有新的最近确认点（供 ffill 判定）
    peak_i = trough_i = -1
    peak_px, trough_px = -np.inf, np.inf
    peak_conf = trough_conf = -1
    peak_k = trough_k = 0.0
    last_conf_type = 0     # 最近一次确认的类型（每次确认必翻转）
    hist: list[tuple[int, int, float]] = []   # (idx, type, price) 按确认顺序
    seg_hi, seg_lo, last_P = -np.inf, np.inf, -1
    ptr = 0

    for t in range(n):
        # 并入此刻被确认的点（按 confirmed_at 排序）
        while ptr < P and cc[ptr] == t:
            pidx, ptype, pk = int(ci[ptr]), int(ct[ptr]), float(ck[ptr])
            px = high[pidx] if ptype == 1 else low[pidx]
            # 每次确认都翻转"最近确认点"的类型（确认时间 t 是最新）；
            # "更极端"只决定该类型用哪个点位/价位代表，不决定是否承认这次确认。
            if ptype == 1:
                peak_conf = t
                if px > peak_px:
                    peak_i, peak_px, peak_k = pidx, px, pk
            else:
                trough_conf = t
                if px < trough_px:
                    trough_i, trough_px, trough_k = pidx, px, pk
            last_conf_type = ptype
            cur_is_peak = ptype == 1
            cur_P = peak_i if cur_is_peak else trough_i
            ptype_arr[t] = 1 if cur_is_peak else -1
            pidx_arr[t] = cur_P
            px_arr[t] = peak_px if cur_is_peak else trough_px
            fresh[t] = True
            # 历史点序列：同类更极端则替换，异类追加（真实时的交替结构）
            if hist and hist[-1][1] == ptype:
                if (ptype == 1 and px > hist[-1][2]) or (ptype == -1 and px < hist[-1][2]):
                    hist[-1] = (pidx, ptype, px)
            else:
                hist.append((pidx, ptype, px))
            ptr += 1

        out[t, 13] = sk[t]
        out[t, 14] = sd[t]
        out[t, 15] = state[t]

        # 最近确认点 = 最近一次确认的类型（每次确认必翻转）
        cur_is_peak = last_conf_type == 1
        P_idx = peak_i if cur_is_peak else trough_i
        if P_idx < 0:  # 该类型还没确认过，退回另一个
            cur_is_peak = not cur_is_peak
            P_idx = peak_i if cur_is_peak else trough_i
        if P_idx < 0:
            continue
        P_px = peak_px if cur_is_peak else trough_px
        P_conf = peak_conf if cur_is_peak else trough_conf
        P_kval = peak_k if cur_is_peak else trough_k
        ptype = 1 if cur_is_peak else -1

        if P_idx != last_P:
            seg_hi, seg_lo, last_P = high[P_idx], low[P_idx], P_idx
        seg_hi = max(seg_hi, high[t])
        seg_lo = min(seg_lo, low[t])

        seg_len = max(1, t - P_idx + 1)
        ret_bp = np.log(max(close[t], 1e-9) / max(P_px, 1e-9)) * 1e4
        out[t, 0] = seg_len / k_period
        out[t, 1] = ret_bp
        out[t, 2] = ret_bp / seg_len
        out[t, 3] = np.log(max(seg_hi, 1e-9) / max(P_px, 1e-9)) * 1e4
        out[t, 4] = np.log(max(seg_lo, 1e-9) / max(P_px, 1e-9)) * 1e4
        out[t, 5] = (close[t] - seg_lo) / max(seg_hi - seg_lo, 1e-9)
        out[t, 6] = ptype
        out[t, 7] = (t - P_conf) / k_period
        out[t, 16] = sk[t] - P_kval

        if len(hist) >= 2:
            segs = []
            for j in range(max(1, len(hist) - n_hist), len(hist)):
                a, b = hist[j - 1], hist[j]
                length = max(1, abs(b[0] - a[0]))
                aret = np.log(max(b[2], 1e-9) / max(a[2], 1e-9)) * 1e4
                segs.append((length, aret, np.sign(aret)))
            lens = np.array([x[0] for x in segs], dtype=np.float64)
            rets = np.array([x[1] for x in segs], dtype=np.float64)
            signs = np.array([x[2] for x in segs], dtype=np.float64)
            out[t, 8] = lens.mean() / k_period
            out[t, 9] = np.abs(rets).mean()
            out[t, 10] = (rets / lens).mean()
            out[t, 11] = out[t, 2] - out[t, 10]
            if len(signs) >= 2:
                out[t, 12] = np.mean(signs[1:] != signs[:-1])

    log(f"structure built: {out.shape}, 结构点 {P}, 特征 {len(FEAT_NAMES)}")
    if return_points:
        # forward-fill：只在 fresh 行写入过，铺到后续行
        cp = pt = 0
        ci_ = -1
        for i in range(n):
            if fresh[i]:
                cp, ci_, pt = int(ptype_arr[i]), int(pidx_arr[i]), float(px_arr[i])
            else:
                ptype_arr[i], pidx_arr[i], px_arr[i] = cp, ci_, pt
        return out.astype(np.float32), list(FEAT_NAMES), (ptype_arr, pidx_arr, px_arr)
    return out.astype(np.float32), list(FEAT_NAMES)


def struct_leak_check(n=20000, cut=17000, k_period=80, seed=0):
    """cut 之后随机化行情，cut 之前输出必须逐位不变（含滚动 stoch 的 80 根窗口）。"""
    rng = np.random.default_rng(seed)
    c = 60000 * np.exp(np.cumsum(rng.normal(0, 3e-4, n)))
    o = c * (1 + rng.normal(0, 1e-4, n))
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 2e-4, n)))
    l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 2e-4, n)))
    df1 = pl.DataFrame({"close_time": np.arange(n), "open": o, "high": h, "low": l, "close": c})
    x1, _ = build_struct(df1, k_period=k_period)
    bad = pl.DataFrame({
        "close_time": np.arange(cut, n), "open": o[cut:] * 0.7,
        "high": h[cut:] * 2.0, "low": l[cut:] * 0.5, "close": c[cut:] * 1.3,
    })
    df2 = pl.concat([df1.head(cut), bad], how="vertical")
    x2, _ = build_struct(df2, k_period=k_period)
    a, b = x1[:cut], x2[:cut]
    # NaN != NaN，但两边同为 NaN 不算泄露：逐元素比较时把双 NaN 视为相等
    same = (a == b) | (np.isnan(a) & np.isnan(b))
    bad_n = int((~same).sum())
    log(f"泄露自检：cut 前不一致元素 {bad_n} -> {'通过' if bad_n == 0 else '失败!'}")
    return bad_n == 0
