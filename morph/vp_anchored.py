#!/usr/bin/env python3
"""锚定当前价的 VP（成交密集区 / 稀疏带）特征（用户 2026-11-21 口径）。

与旧版 `kline_vp.build_vp` 的本质区别
-------------------------------------
旧版：4h/16h/24h **固定时间窗**，当前价常落在窗口上/下边缘 → 继续运行方向上
      没有成交量分布，「压力位」看不到（用户指出这是口径错误，§44 已作废）。
本版：**长回溯（2d/7d）+ 全局绝对 log 价格网格**，直方图在锚点处算一次，
      但所有距离/稀疏带都以**当前 bar 的 close** 定位 bucket，不是锚点那根的价格。
      窗口足够长 → 当前价几乎总在分布内部，上下方都能看到密集/稀疏结构。

每窗口 5 列（W=2880/10080，共 10 列）：
  av_res_up_{W}  当前价上方最近「密集区」距离（bp，封顶 300）
  av_res_dn_{W}  下方最近「支撑区」距离（bp）
  av_gap_up_{W}  上方连续「稀疏带」宽度（bp，价格可快速穿越）
  av_gap_dn_{W}  下方稀疏带宽度
  av_imb50_{W}   当前价 ±50bp 的 (上量−下量)/(上量+下量)，局部不平衡

密集/稀疏定义（相对该窗口活动区的中位桶量）：
  dense  = 平滑桶量 > 1.5 × median
  sparse = 平滑桶量 < 0.5 × median

因果性
------
直方图只用 [anchor-W+1, anchor]，anchor ≤ 当前 bar；距离用当前 close 定位桶。
任何改动后必须先过 `avp_leak_check()`：cut 之后随机化行情，cut 之前的输出必须逐位不变。
"""
from __future__ import annotations

import os
import time

import numpy as np
import polars as pl

T = time.time()


def log(msg: str) -> None:
    print(f"[avp {time.time() - T:6.1f}s] {msg}", flush=True)


def _build(close, high, low, volume, windows, tick_bp, anchor, cap_bp):
    c = np.asarray(close, dtype=np.float64)
    h = np.asarray(high, dtype=np.float64)
    l = np.asarray(low, dtype=np.float64)
    v = np.nan_to_num(np.asarray(volume, dtype=np.float64))
    n = len(c)
    step = tick_bp * 1e-4
    cb_cap = int(cap_bp // tick_bp)          # 距离封顶（桶数）
    half = max(1, int(round(50.0 / tick_bp)))  # ±50bp 用几个桶

    tp = (h + l + c) / 3.0
    bins = np.floor(np.log(np.maximum(tp, 1e-9)) / step).astype(np.int64)

    a_idx = np.arange(anchor - 1, n, anchor)  # 锚点 bar 位置
    k_of = (np.arange(n) + 1) // anchor - 1   # 每 bar 用的锚点序号（≤ 自身）
    if len(a_idx) == 0:
        return np.zeros((n, 5 * len(windows)), dtype=np.float32)

    sm_kernel = np.ones(5, dtype=np.float64) / 5.0
    names, cols = [], []
    for W in windows:
        out = np.zeros((n, 5), dtype=np.float32)
        # 分块处理锚点，避免常驻 (n_anchors × n_bins) 的大矩阵
        CHUNK = 256
        for a0 in range(0, len(a_idx), CHUNK):
            a1 = min(a0 + CHUNK, len(a_idx))
            sel = np.where((k_of >= a0) & (k_of < a1))[0]
            if sel.size == 0:
                continue
            for j in range(a0, a1):
                a = int(a_idx[j])
                s = a - W + 1
                if s < 0:
                    continue
                rows = sel[k_of[sel] == j]
                if rows.size == 0:
                    continue
                bb = bins[s:a + 1]
                m = int(bb.min())
                hist = np.bincount(bb - m, weights=v[s:a + 1])
                sm = np.convolve(hist, sm_kernel, mode="same")
                sm = np.maximum(sm, 0.0)
                nz = sm[sm > 0]
                if nz.size < 10:
                    continue
                med = float(np.median(nz))
                idx = np.arange(len(sm), dtype=np.int64)
                dense = np.where(sm > 1.5 * med)[0]
                stop = np.where(sm >= 0.5 * med)[0]  # 稀疏带在此终止

                def _gap(dense_or_stop, direction):
                    """到最近 dense/stop 的桶数；direction=+1 向上，-1 向下。"""
                    d = cb_cap
                    if dense_or_stop.size == 0:
                        return np.full(len(sm), d, dtype=np.float64)
                    q = np.searchsorted(dense_or_stop, idx, side="left" if direction > 0 else "right")
                    if direction > 0:
                        qc = np.clip(q, 0, len(dense_or_stop) - 1)
                        dist = dense_or_stop[qc] - idx
                        dist[q >= len(dense_or_stop)] = d
                    else:
                        qc = np.clip(q - 1, 0, len(dense_or_stop) - 1)
                        dist = idx - dense_or_stop[qc]
                        dist[q == 0] = d
                    return np.clip(dist, 0, d).astype(np.float64)

                du = _gap(dense, +1)      # 上方最近密集区
                dd = _gap(dense, -1)      # 下方最近密集区
                gu = _gap(stop, +1)       # 上方稀疏带宽度（到下一个非稀疏桶）
                gd = _gap(stop, -1)
                cum = np.concatenate(([0.0], np.cumsum(sm)))
                nb = len(sm)

                cur = np.clip(bins[rows] - m, 0, nb - 1)
                up = cum[np.minimum(cur + half, nb)] - cum[cur]
                dn = cum[cur] - cum[np.maximum(cur - half, 0)]
                out[rows, 0] = du[cur] * tick_bp
                out[rows, 1] = dd[cur] * tick_bp
                out[rows, 2] = gu[cur] * tick_bp
                out[rows, 3] = gd[cur] * tick_bp
                out[rows, 4] = (up - dn) / (up + dn + 1e-9)
        cols.append(out)
        names += [f"av_res_up_{W}", f"av_res_dn_{W}", f"av_gap_up_{W}",
                  f"av_gap_dn_{W}", f"av_imb50_{W}"]
        log(f"W={W} 完成（锚点 {len(a_idx):,}，bin={tick_bp}bp，cap={cap_bp}bp）")
    return np.hstack(cols).astype(np.float32), names


def build_avp(df: pl.DataFrame, windows=(2880, 10080), tick_bp=5.0,
              anchor=120, cap_bp=300.0):
    """从 vision-base 原始 K 线构造锚定 VP；volume 由 um_buy_vol+um_sell_vol 合成。"""
    if "volume" in df.columns:
        v = df["volume"].to_numpy()
    else:
        v = (df["um_buy_vol"].cast(pl.Float64) + df["um_sell_vol"].cast(pl.Float64)).to_numpy()
    return _build(df["close"].to_numpy(), df["high"].to_numpy(), df["low"].to_numpy(),
                  v, windows, tick_bp, anchor, cap_bp)


def avp_leak_check(n=20000, cut=17000, windows=(240, 960), anchor=10, seed=0):
    """未来数据自检：cut 之后随机化行情，cut 之前输出必须逐位不变。"""
    cut = (cut // anchor) * anchor + anchor // 2
    rng = np.random.default_rng(seed)
    c = 60000 * np.exp(np.cumsum(rng.normal(0, 3e-4, n)))
    o = c * (1 + rng.normal(0, 1e-4, n))
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 2e-4, n)))
    l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 2e-4, n)))
    v = rng.lognormal(3, 1, n)
    x1, _ = _build(c, h, l, v, windows, 5.0, anchor, 300.0)
    bad_df = pl.DataFrame({"close": c[cut:] * 1.3, "high": h[cut:] * 2.0,
                           "low": l[cut:] * 0.5, "volume": v[cut:] * 5.0})
    # 用同一重建：前半段必须与 x1 的前 cut 行一致（cut 后行情随机化不得影响 cut 前输出）
    full = pl.concat([
        pl.DataFrame({"close": c[:cut], "high": h[:cut], "low": l[:cut], "volume": v[:cut]}),
        bad_df,
    ])
    x2, _ = _build(full["close"].to_numpy(), full["high"].to_numpy(),
                   full["low"].to_numpy(), full["volume"].to_numpy(),
                   windows, 5.0, anchor, 300.0)
    bad = int((x1[:cut] != x2[:cut]).sum())
    log(f"泄露自检：cut 前不一致元素 {bad} -> {'通过' if bad == 0 else '失败!'}")
    return bad == 0
