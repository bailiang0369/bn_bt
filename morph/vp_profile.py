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
- 滚动和用环形缓冲增量更新：O(n) 时间、常数内存；
- 按块向量化（同一块内 15 行共享同一剖面），避免逐行 Python 开销。

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
    """一个块内所有行的 22 列特征；P/D/C/S 已在两端 pad 了 R 个零。"""
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
    Ls[:, c] = -1.0                          # 把当前桶压到最低，top4 天然排除它
    sel = np.argsort(-Ls, axis=1)[:, :4]
    top_dist = (sel - c) * res_bps
    top_share = L[rows[:, None], sel].sum(1) / s

    hi_t = c_at + 0.35 * s
    lo_t = c_at - 0.35 * s
    up = (cm < hi_t[:, None]).sum(1)
    dn = (cm < lo_t[:, None]).sum(1)

    above = s - cm[:, c]
    below = cm[:, c] - L[:, c]
    touches_pb = C[pbs + c]
    touches_poc = C[pbs + imax]
    recent = S[pbs + c] / (P[pbs + c] + 1e-9)

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
    out[:, 19] = touches_pb
    out[:, 20] = touches_poc
    out[:, 21] = recent
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
    pad = np.zeros(R, np.float64)

    wins = (w_short, w_mid, w_long)
    rv = {w: np.zeros(B, np.float64) for w in wins}
    rd = {w: np.zeros(B, np.float64) for w in wins}
    rc = {w: np.zeros(B, np.float64) for w in wins}
    M = max(wins) + 2
    ring: list = [None] * M
    nb = (n + anchor - 1) // anchor

    names: list[str] = ([f"vp8h_{k}" for k in NAMES_BASE]
                        + [f"vp24h_{k}" for k in NAMES_BASE]
                        + ["vp_poc_shift", "vp_recent24", "vp_recent8"])
    out = np.full((n, len(names)), np.nan, np.float32)

    for k in range(nb):
        if k >= 1:
            rec = ring[(k - 1) % M]
            if rec is not None:
                b_a, v_a, d_a, c_a = rec
                for w in wins:
                    np.add.at(rv[w], b_a, v_a)
                    np.add.at(rd[w], b_a, d_a)
                    np.add.at(rc[w], b_a, c_a)
        for w in wins:
            old = k - 1 - w
            if old >= 0:
                rec = ring[old % M]
                if rec is not None:
                    b_a, v_a, d_a, c_a = rec
                    np.add.at(rv[w], b_a, -v_a)
                    np.add.at(rd[w], b_a, -d_a)
                    np.add.at(rc[w], b_a, -c_a)

        i0, i1 = k * anchor, min(n, (k + 1) * anchor)
        bins = bb[i0:i1]
        uniq, inv = np.unique(bins, return_inverse=True)
        ring[k % M] = (uniq, np.bincount(inv, weights=v[i0:i1]),
                       np.bincount(inv, weights=dv[i0:i1]),
                       (np.bincount(inv) > 0).astype(np.float64))   # 每块内「访问过」=1

        if rv[w_long].sum() <= 0:
            continue                      # 视窗还没填满，留 NaN
        pbs = bins
        # 增量加减会留浮点残差（可能 -1e-18），量类一律截到非负
        Pm = np.concatenate((pad, np.maximum(rv[w_mid], 0.0), pad))
        Pl = np.concatenate((pad, np.maximum(rv[w_long], 0.0), pad))
        Dm = np.concatenate((pad, rd[w_mid], pad))
        Dl = np.concatenate((pad, rd[w_long], pad))
        Cm = np.concatenate((pad, np.maximum(rc[w_mid], 0.0), pad))
        Cl = np.concatenate((pad, np.maximum(rc[w_long], 0.0), pad))
        Sp = np.concatenate((pad, np.maximum(rv[w_short], 0.0), pad))
        f8 = _block_feats(Pm, Dm, Cm, Sp, pbs, R, res_bps)
        f24 = _block_feats(Pl, Dl, Cl, Sp, pbs, R, res_bps)
        out[i0:i1, :21] = f8[:, :21]
        out[i0:i1, 21:42] = f24[:, :21]
        out[i0:i1, 42] = f8[:, 0] - f24[:, 0]
        out[i0:i1, 43] = f24[:, 21]
        out[i0:i1, 44] = f8[:, 21]

    new = df.with_columns([pl.Series(nm, out[:, j]) for j, nm in enumerate(names)])
    return new, feats + names
