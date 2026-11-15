#!/usr/bin/env python3
"""逐分桶成交量剖面（Volume Profile）特征 —— **独立模块，不改动任何现役脚本**。

为什么要重做 VP
--------------
第 12 轮 `kline_vp.py` 只做了 15 列汇总标量（POC 距离 / 价值区位置与宽度 / 上下量比 /
VWAP 偏离 × 3 窗口），判定"无效"。2026-11-15 用户指出四个缺口，那 15 列一个都没编码：

1. **不是 POC 的分桶也有阻力** —— 需要上下各 R 桶的局部剖面，而不是只看峰值；
2. **价格与各分桶的相对位置** —— 需要到成交量 top1~top4 分桶的距离；
3. **分桶历史量 vs 近期量 vs 主动买卖** —— 持久量 / 近期量 比值 + 该桶签名成交量；
4. **同一分桶有时撑有时不撑** —— 被触及次数（消化程度）与 delta，条件让模型自己学。

机制上再补一条：阻力通常在剖面**边缘 / 真空区边沿**，不在 POC 峰顶（POC 是筹码充分交换处，
摩擦小、常被穿透）。所以给剖面一阶/二阶差分 + HVN 边缘 / LVN 标记 + 价值区上下沿距离。

因果性（硬性）
--------------
- 价格按 **log 绝对分桶**（桶宽 `VP_RES` bp），不随行漂移；
- 剖面每 `VP_ANCHOR` 根 bar 重算，且**只用已完成的块**（块号 < 当前块）→ 不可能偷未来；
- 滚动和在 padded 缓冲区上**原位增量更新**（只动当前块涉及的几个桶），单块 O(块内桶数)，
  与全局桶数 B 无关；环形缓冲只存每个块的 (桶, 量, delta, 访问) 四个小数组。

产出 45 列：vp8h_* / vp24h_* 各 21 列 + vp_poc_shift / vp_recent24 / vp_recent8。

env: VP_RES=20 VP_ANCHOR=15 VP_W_SHORT=12 VP_W_MID=32 VP_W_LONG=96 VP_R=25
"""
from __future__ import annotations

import os

import numpy as np
import polars as pl

NAMES_BASE = ["dist_poc", "poc_share", "vol_rel", "vol_pm1", "vol_pm2", "grad2",
              "is_lvn", "is_hvn_edge", "dist_top1", "dist_top2", "dist_top3", "dist_top4",
              "top4_share", "va_up", "va_dn", "above_below", "delta_rel", "delta_grad2",
              "delta_ab", "touches_pb", "touches_poc"]


def _block_feats(P, D, C, S, pbs, R: int, res_bps: float) -> np.ndarray:
    """一个块内所有行的 22 列特征；P/D/C/S 是已 pad 过 R 的滚动剖面。"""
    m = pbs.size
    idx = pbs[:, None] + np.arange(2 * R + 1)[None, :]      # 中心落在第 R 列
    L = P[idx]
    Dq = D[idx]
    s = L.sum(1)
    s = np.where(s > 0, s, np.nan)
    mean = s / L.shape[1]
    imax = L.argmax(1)
    c = R
    cm = np.cumsum(L, 1)
    c_at = cm[np.arange(m), imax]
    rows = np.arange(m)

    Ls = L.copy()
    Ls[:, c] = -1.0                          # 当前桶压到最低，top4 天然排除它
    sel = np.argsort(-Ls, axis=1)[:, :4]
    top_dist = (sel - c) * res_bps
    top_share = L[rows[:, None], sel].sum(1) / s

    hi_t = c_at + 0.35 * s
    lo_t = c_at - 0.35 * s
    up = (cm < hi_t[:, None]).sum(1)
    dn = (cm < lo_t[:, None]).sum(1)

    above = s - cm[:, c]
    below = cm[:, c] - L[:, c]

    out = np.empty((m, 22), np.float64)
    out[:, 0] = (imax - c) * res_bps
    out[:, 1] = L.max(1) / s
    out[:, 2] = L[:, c] / mean
    out[:, 3] = (L[:, c + 1] - L[:, c - 1]) / s
    out[:, 4] = (L[:, c + 2] - L[:, c - 2]) / s
    out[:, 5] = (L[:, c + 1] + L[:, c + 2] - L[:, c - 1] - L[:, c - 2]) / s
    out[:, 6] = (L[:, c] < 0.3 * mean).astype(np.float64)
    out[:, 7] = ((L[:, c] > 0.7 * L.max(1))
                 & (np.minimum(L[:, c - 1], L[:, c + 1]) < 0.5 * L[:, c])).astype(np.float64)
    out[:, 8:12] = top_dist
    out[:, 12] = top_share
    out[:, 13] = (up - c) * res_bps
    out[:, 14] = (dn - c) * res_bps
    out[:, 15] = np.log1p(above) - np.log1p(below)
    out[:, 16] = Dq[:, c] / (L[:, c] + 1e-9)
    out[:, 17] = (Dq[:, c + 2] - Dq[:, c - 2]) / mean
    out[:, 18] = (Dq[:, c + 1:].sum(1) - Dq[:, :c + 1].sum(1)) / s
    out[:, 19] = C[pbs + c]
    out[:, 20] = C[pbs + imax]
    out[:, 21] = S[pbs + c] / (P[pbs + c] + 1e-9)
    return out


def add_vp(df: pl.DataFrame, feats: list[str]) -> tuple[pl.DataFrame, list[str]]:
    """在 df 上追加 45 列 VP 结构特征，返回 (新 df, 特征列名)。"""
    res = float(os.environ.get("VP_RES", "20")) * 1e-4      # 桶宽（bp）
    anchor = int(os.environ.get("VP_ANCHOR", "15"))
    w_short = int(os.environ.get("VP_W_SHORT", "12"))        # 3h
    w_mid = int(os.environ.get("VP_W_MID", "32"))            # 8h
    w_long = int(os.environ.get("VP_W_LONG", "96"))          # 24h
    R = int(os.environ.get("VP_R", "25"))                    # 局部 ±R 桶

    n = df.height
    hi_ = df["high"].to_numpy().astype(np.float64)
    lo_ = df["low"].to_numpy().astype(np.float64)
    cl = df["close"].to_numpy().astype(np.float64)
    px = (hi_ + lo_ + cl) / 3.0
    bad = ~np.isfinite(px) | (px <= 0)
    if bad.all():
        return df, feats
    if bad.any():
        px = np.where(bad, np.nanmedian(px[~bad]), px)

    vol_n = next((x for x in ("volume", "um_volume") if x in df.columns), None)
    tb_n = next((x for x in ("taker_buy_volume", "um_taker_buy_vol") if x in df.columns), None)
    v = (df[vol_n].to_numpy().astype(np.float64) if vol_n else np.ones(n))
    tb = (df[tb_n].to_numpy().astype(np.float64) if tb_n else v * 0.5)
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    tb = np.nan_to_num(tb, nan=0.0, posinf=0.0, neginf=0.0)
    dv = (2.0 * tb / np.maximum(v, 1e-12) - 1.0) * v        # 签名成交量

    step = np.log1p(res)
    bb = np.floor(np.log(px) / step).astype(np.int64)
    bb -= bb.min()
    B = int(bb.max()) + 1
    res_bps = res * 1e4

    wins = (w_short, w_mid, w_long)
    Bp = B + 2 * R
    P = {w: np.zeros(Bp) for w in wins}      # 成交量剖面
    Dd = {w: np.zeros(Bp) for w in wins}     # 签名成交量剖面
    Cc = {w: np.zeros(Bp) for w in wins}     # 访问次数剖面
    filled = {w: 0 for w in wins}
    M = max(wins) + 2
    ring: list = [None] * M
    nb = (n + anchor - 1) // anchor

    names: list[str] = ([f"vp8h_{k}" for k in NAMES_BASE]
                        + [f"vp24h_{k}" for k in NAMES_BASE]
                        + ["vp_poc_shift", "vp_recent24", "vp_recent8"])
    out = np.full((n, len(names)), np.nan, np.float32)

    def upd(w: int, kk: int, sign: float) -> None:
        rec = ring[kk % M]
        if rec is None:
            return
        b_a, v_a, d_a, c_a = rec
        j = b_a + R                            # 平移到 padded 坐标
        P[w][j] += sign * v_a
        Dd[w][j] += sign * d_a
        Cc[w][j] += sign * c_a
        P[w][j] = np.maximum(P[w][j], 0.0)     # 增量加减的浮点残差截到非负
        Cc[w][j] = np.maximum(Cc[w][j], 0.0)

    for k in range(nb):
        if k >= 1:
            for w in wins:
                upd(w, k - 1, 1.0)
                filled[w] += 1
        for w in wins:
            old = k - 1 - w
            if old >= 0:
                upd(w, old, -1.0)
                filled[w] -= 1

        i0, i1 = k * anchor, min(n, (k + 1) * anchor)
        bins = bb[i0:i1]
        uniq, inv = np.unique(bins, return_inverse=True)
        ring[k % M] = (uniq, np.bincount(inv, weights=v[i0:i1]),
                       np.bincount(inv, weights=dv[i0:i1]),
                       (np.bincount(inv) > 0).astype(np.float64))

        if filled[w_long] < 2:
            continue
        f8 = _block_feats(P[w_mid], Dd[w_mid], Cc[w_mid], P[w_short], bins, R, res_bps)
        f24 = _block_feats(P[w_long], Dd[w_long], Cc[w_long], P[w_short], bins, R, res_bps)
        out[i0:i1, :21] = f8[:, :21]
        out[i0:i1, 21:42] = f24[:, :21]
        out[i0:i1, 42] = f8[:, 0] - f24[:, 0]
        out[i0:i1, 43] = f24[:, 21]
        out[i0:i1, 44] = f8[:, 21]

    new = df.with_columns([pl.Series(nm, out[:, j]) for j, nm in enumerate(names)])
    return new, feats + names
