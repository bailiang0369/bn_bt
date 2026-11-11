#!/usr/bin/env python3
"""**无假设**条件分桶扫描（hypothesis generator，不产出结论）。

为什么需要它
------------
`features.py` 里的 35 个特征全部是**先有假设、再写公式**：其中 12 个直接对应
用户 2026-11-11 举的三个现象（CVD 斜率突破 / 暴跌后背离 / 滚动分位），
其余 21 个虽分属分布矩、游程、长记忆、CVD 形态、跨市场、极值 6 个家族，
公式依然是手写的、我想到的形态。`profile.py` 只能在**这 35 个既定候选**里排序，
结构上不可能发现第 36 个规律。

本模块换方向：**不写公式，只枚举"条件"**，让数据说出
「在什么条件下，未来 h 分钟上涨概率显著偏离基准」。

方法
----
对每个条件维度 D 的每个桶 b：
    n_b = 桶内非重叠样本数（step = h，避免标签重叠把 n 夸大几个数量级）
    p_b = 桶内上涨比例
    z_b = (p_b - p_base) / sqrt(p_base*(1-p_base)/n_b)
    seg = 3 段时间上 p_b 是否同侧（与 validate.py 同一把尺子）
    q_b = BH-FDR 校正后的 p 值
**只做 1 维枚举**（含少量 2 维展开），2 维全交叉会让检验数爆到 1e4 量级，FDR 后必空。

维度（16 个，全部因果，只用 t 及之前）
------------------------------------
时间：hour(24) / dow(7) / month(12) / min5(12) / dom_bucket(5) /
      fund_h 距 8h 资金费结算(8) / weekend_hour(48)
状态：滚动 30 天经验分位 5 档 —— 已实现波动、成交量、合约 CVD 60m 斜率、
      现货/合约量比、bar 实体占比、**价格在过去 1 天的位置**
交叉：波动率分位 × 上一根方向(10) / CVD 斜率分位 × 上一根方向(10) /
      连续同向长度(1-5)

诚实条款
--------
1. 只**生成假设**，不构成结论；任何桶要经 validate.py 7 关 + 多段配对 A/B 才可入库。
2. 强制报「预期假阳性 = q × 检验数」；显著数不超过它就判该 horizon 无规律。
3. 分桶用 trailing 分位（因果），不用全期分位。

用法
----
    python -m morph.statfeat.scan --synthetic
    python -m morph.statfeat.scan --out /kaggle/working/scan_btc.csv
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl

try:
    from .validate import load_data
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from morph.statfeat.validate import load_data

HORIZONS = (3, 15, 60)
N_SEG = 3
MIN_N = 200
Q_FDR = 0.05
DAY = 1440
WEEK = 7 * DAY


def log(m: str) -> None:
    print(m, flush=True)


# ---------------------------------------------------------------- 因果维度


def _roll_rank5(x: np.ndarray, w: int) -> np.ndarray:
    """trailing 经验分位 → 5 档（0..4）；NaN → -1。因果。"""
    s = pl.Series("x", np.asarray(x, dtype=np.float64))
    r = np.zeros(len(x), dtype=np.int8)
    for q in (0.2, 0.4, 0.6, 0.8):
        qv = s.rolling_quantile(q, window_size=w, min_samples=max(20, w // 2)).to_numpy()
        r += (np.nan_to_num(x, nan=-np.inf) > qv).astype(np.int8)
    r[~np.isfinite(x)] = -1
    return r


def _run_len(ret: np.ndarray) -> np.ndarray:
    """连续同向长度 cap 到 5；无效（ret 为 NaN/0）记 -1，避免被当成独立桶。因果。"""
    n = len(ret)
    out = np.full(n, -1, dtype=np.int8)
    cur = 0
    d_prev = 0
    for i in range(n):
        r = ret[i]
        if not np.isfinite(r) or r == 0:
            cur, d_prev = 0, 0
            continue
        d = 1 if r > 0 else -1
        cur = cur + 1 if d == d_prev else 1
        d_prev = d
        out[i] = min(cur, 5)
    return out


def build_dims(df: pl.DataFrame) -> dict[str, np.ndarray]:
    n = df.height
    t = pl.from_epoch(pl.Series("t", df["close_time"].to_numpy()), time_unit="ms")
    hour = t.dt.hour().to_numpy().astype(np.int16)
    dow = t.dt.weekday().to_numpy().astype(np.int16) - 1      # 0=周一
    day = t.dt.day().to_numpy().astype(np.int16)
    month = (t.dt.month().to_numpy() - 1).astype(np.int16)
    minute5 = (t.dt.minute().to_numpy() // 5).astype(np.int16)

    c = df["close"].to_numpy().astype(np.float64)
    o = df["open"].to_numpy().astype(np.float64) if "open" in df.columns else c.copy()
    hi = df["high"].to_numpy().astype(np.float64) if "high" in df.columns else c.copy()
    lo = df["low"].to_numpy().astype(np.float64) if "low" in df.columns else c.copy()

    ret = np.full(n, np.nan)
    ret[1:] = c[1:] / c[:-1] - 1

    def col(name: str) -> np.ndarray:
        return (df[name].fill_null(0).to_numpy().astype(np.float64)
                if name in df.columns else np.zeros(n))

    um_vol = col("um_buy_vol") + col("um_sell_vol")
    sp_vol = col("spot_buy_vol") + col("spot_sell_vol")
    cvd_u = np.cumsum(col("um_buy_vol") - col("um_sell_vol"))

    rv = pl.Series("x", np.nan_to_num(np.abs(ret))).rolling_mean(
        DAY, min_samples=240).to_numpy()
    vz = pl.Series("x", um_vol).rolling_mean(60, min_samples=20).to_numpy()
    cvd_sl = np.full(n, np.nan)
    cvd_sl[60:] = (cvd_u[60:] - cvd_u[:-60]) / 60
    vr = np.log1p(sp_vol) - np.log1p(um_vol)
    body = (c - o) / (hi - lo + 1e-9)
    pos = pl.Series("c", c).rolling_min(DAY, min_samples=240).to_numpy()
    posmax = pl.Series("c", c).rolling_max(DAY, min_samples=240).to_numpy()
    pos_r = (c - pos) / (posmax - pos + 1e-9)      # 过去 1 天位置 0..1

    rv_q = _roll_rank5(rv, WEEK)
    vol_q = _roll_rank5(vz, WEEK)
    cvd_q = _roll_rank5(cvd_sl, WEEK)
    vratio_q = _roll_rank5(vr, WEEK)
    body_q = _roll_rank5(body, WEEK)
    pos_q = _roll_rank5(pos_r, WEEK)
    run = _run_len(ret)
    up = np.zeros(n, dtype=np.int8)
    up[np.isfinite(ret) & (ret > 0)] = 1

    dims: dict[str, np.ndarray] = {
        "hour": hour,
        "dow": dow,
        "month": month,
        "min5": minute5,
        "dom_bucket": np.digitize(day, [4, 11, 21, 28]).astype(np.int16),
        "fund_h": ((8 - (hour % 8)) % 8).astype(np.int16),
        "weekend_hour": (dow >= 5).astype(np.int16) * 24 + hour,
        "rv_q": rv_q,
        "vol_q": vol_q,
        "cvd_slope_q": cvd_q,
        "volratio_q": vratio_q,
        "body_q": body_q,
        "pos_q": pos_q,
        "run_len": run,
        "rv_x_dir": np.where(rv_q >= 0, rv_q * 2 + up, -1).astype(np.int16),
        "cvd_x_dir": np.where(cvd_q >= 0, cvd_q * 2 + up, -1).astype(np.int16),
    }
    return dims


# ---------------------------------------------------------------- 统计


def bh_fdr(p: np.ndarray) -> np.ndarray:
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    return out


def _p_two(z: float) -> float:
    from math import erfc, sqrt
    return float(erfc(abs(z) / sqrt(2)))


def scan_dim(label: np.ndarray, y: np.ndarray, base_p: float,
             seg_id: np.ndarray) -> list[dict]:
    """向量化：一次 bincount 得到所有桶的计数与和。"""
    ok = (label >= 0) & np.isfinite(y)
    lab = label[ok].astype(np.int64)
    yy = y[ok].astype(np.float64)
    sg = seg_id[ok]
    if len(lab) == 0:
        return []
    nb = int(lab.max()) + 1
    cnt = np.bincount(lab, minlength=nb)
    tot = np.bincount(lab, weights=yy, minlength=nb)
    # 每段
    scnt = np.zeros((N_SEG, nb))
    stot = np.zeros((N_SEG, nb))
    for s in range(N_SEG):
        m = sg == s
        if m.any():
            scnt[s] = np.bincount(lab[m], minlength=nb)
            stot[s] = np.bincount(lab[m], weights=yy[m], minlength=nb)
    rows = []
    for b in range(nb):
        if cnt[b] < MIN_N:
            continue
        p_b = tot[b] / cnt[b]
        se = np.sqrt(base_p * (1 - base_p) / cnt[b])
        z = (p_b - base_p) / (se + 1e-12)
        sp = np.where(scnt[:, b] >= 30, stot[:, b] / np.maximum(scnt[:, b], 1), np.nan)
        v = sp[np.isfinite(sp)]
        rows.append(dict(bucket=b, n=int(cnt[b]), p=float(p_b),
                         gap_pp=float((p_b - base_p) * 100), z=float(z),
                         pval=_p_two(z),
                         sign_consistent=bool(len(v) == N_SEG and
                                              (np.all(v > base_p) or np.all(v < base_p)))))
    return rows


# ---------------------------------------------------------------- 主体


def main() -> None:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    ap = argparse.ArgumentParser(description="无假设条件分桶扫描（只生成假设）")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--rows", type=int, default=200000)
    ap.add_argument("--symbol", default=os.environ.get("SYMBOL", "BTCUSDT"))
    ap.add_argument("--warmup", type=int, default=2 * WEEK)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--out", default=os.environ.get("SCAN_OUT"))
    a = ap.parse_args(list(sys.argv[1:]) + os.environ.get("SCAN_ARGS", "").split())

    df = load_data(a.synthetic, a.rows)
    n = df.height
    log(f"[data] symbol={a.symbol} rows={n:,} cols={len(df.columns)}")
    t = pl.from_epoch(pl.Series("t", df["close_time"].to_numpy()), time_unit="ms")
    log(f"[span] {t.min()} ~ {t.max()}")

    dims = build_dims(df)
    log(f"[dims] {len(dims)} 个: {', '.join(dims)}")

    c = df["close"].to_numpy().astype(np.float64)
    warm = min(a.warmup, n // 3) if a.synthetic else a.warmup
    all_rows = []

    for h in HORIZONS:
        y = np.full(n, np.nan)
        y[:-h] = (c[h:] > c[:-h]).astype(np.float64)
        keep = np.zeros(n, dtype=bool)
        keep[warm:n - h:h] = True
        idx = np.where(keep)[0]
        if len(idx) < 1000:
            log(f"\n=== h={h}m | 独立样本仅 {len(idx):,}，跳过 ===")
            continue
        yk = y[idx]
        base_p = float(np.nanmean(yk))
        seg_id = np.minimum(np.arange(len(idx)) * N_SEG // len(idx), N_SEG - 1)
        log("")
        log(f"=== h={h}m | 独立样本 {len(idx):,} | 基准上涨率 {base_p * 100:.2f}% ===")

        rows_all = []
        for dname, lab in dims.items():
            rr = scan_dim(lab[idx], yk, base_p, seg_id)
            for r in rr:
                r["dim"] = dname
                r["horizon"] = h
                r["symbol"] = a.symbol
            rows_all.extend(rr)

        q = bh_fdr(np.array([r["pval"] for r in rows_all]))
        for r, qq in zip(rows_all, q):
            r["q"] = float(qq)
        all_rows.extend(rows_all)

        exp_fp = Q_FDR * len(rows_all)
        sig = [r for r in rows_all if r["q"] < Q_FDR and r["sign_consistent"]]
        log(f"  检验 {len(rows_all)} 个 | 预期假阳性 {exp_fp:.1f} | "
            f"通过(q<{Q_FDR} 且 3 段同号) {len(sig)}")
        if len(sig) <= exp_fp:
            log("  → 显著数 ≤ 假阳性预期，本 horizon 无可用规律")
        for r in sorted(sig, key=lambda x: -abs(x["gap_pp"]))[: a.top]:
            log(f"    {r['dim']:<14} 桶={r['bucket']:<3} n={r['n']:>7,} "
                f"p={r['p'] * 100:6.2f}% 偏离={r['gap_pp']:+6.2f}pp "
                f"z={r['z']:+5.2f} q={r['q']:.2e}")

    if a.out and all_rows:
        pl.DataFrame(all_rows).write_csv(a.out)
        log("")
        log(f"[out] {a.out}  {len(all_rows)} 行")

    log("")
    log("【提醒】以上只是假设。任何桶在被 validate.py 7 关 + 多段配对 A/B 证实前，"
        "不得写入特征或用于下单。")


if __name__ == "__main__":
    main()
