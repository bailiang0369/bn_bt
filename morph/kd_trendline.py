"""Stoch「指标趋势线突破」：斜率是否决定延续性、到极值要多久

用户的实盘画法（与 K/D 交叉不同）：
  在 Stoch 指标线上连接最近两个峰/谷画出趋势线，指标穿越该线时入场；
  突破斜率越大越可信；突破后持续多久到达极值（80/20）有参考意义。

操作化（全部因果，pivot 需右侧 k 根确认，确认后才可用）：
  Stoch(n=69, k=3, d=3)            通达信口径 SMA(X,N,1)
  pivot high/low：窗口 2k+1 内唯一极值，k=3
  上穿「最近两个 pivot high 的连线」→ 做多
  下穿「最近两个 pivot low  的连线」→ 做空
  slope = (v1-v2)/(t1-t2)  单位：指标点 / bar（区分正负：上升线 vs 下降线）
目标：
  1) 事件数与斜率分布
  2) 固定持有 h 期的胜率（1m: 1/3/5/10；5m: 1/3/5/10），按斜率等频分档
  3) 到极值时间：突破后首次 K>=80（多）/ K<=20（空）的 bar 数，按斜率分档
  4) 对照：普通 K/D 交叉（同口径固定持有）
  5) 分年一致性

数据：ETHUSDT 1m klines（vision-base），Kaggle 端从 GitHub Release 拉，不在本地下载。
env：SYMBOL=ETHUSDT  YEARS=2024,2025,2026  MINS=1,5
"""
import os
import sys
import time

import numpy as np
import polars as pl
import requests

REPO = "bailiang0369/bn_hisdata"
TAG = os.environ.get("TAG", "vision-base")
SYMBOL = os.environ.get("SYMBOL", "ETHUSDT")
YEARS = [int(x) for x in os.environ.get("YEARS", "2024,2025,2026").split(",")]
MINS = [int(x) for x in os.environ.get("MINS", "1,5").split(",")]
SN, SK, SD = 69, 3, 3
PIV_K = 3
HS = (1, 3, 5, 10)
CACHE = "/tmp/kdtrend"


def log(*a):
    print(*a, flush=True)


def fetch(name):
    """GitHub Release asset API（直链 504，API 206），Range 续传。"""
    os.makedirs(CACHE, exist_ok=True)
    dst = os.path.join(CACHE, name)
    if os.path.exists(dst) and os.path.getsize(dst) > 1e6:
        return dst
    h = {"Accept": "application/vnd.github+json"}
    tok = os.environ.get("GH_TOKEN")
    if tok:
        h["Authorization"] = f"Bearer {tok}"
    j = requests.get(f"https://api.github.com/repos/{REPO}/releases/tags/{TAG}",
                     headers=h, timeout=60).json()
    a = [x for x in j["assets"] if x["name"] == name]
    if not a:
        raise RuntimeError(f"asset not found: {name}")
    a = a[0]
    h2 = {"Accept": "application/octet-stream"}
    if tok:
        h2["Authorization"] = f"Bearer {tok}"
    url = f"https://api.github.com/repos/{REPO}/releases/assets/{a['id']}"
    part = dst + ".part"
    for attempt in range(1, 9):
        done = os.path.getsize(part) if os.path.exists(part) else 0
        if done >= a["size"]:
            break
        try:
            r = requests.get(url, headers={**h2, "Range": f"bytes={done}-"}, timeout=120)
            if r.status_code not in (200, 206):
                raise RuntimeError(f"HTTP {r.status_code}")
            with open(part, "ab") as f:
                f.write(r.content)
        except Exception as e:
            log(f"    [dl] {name} attempt {attempt}: {e}")
            time.sleep(2 ** min(attempt, 4))
    os.replace(part, dst)
    return dst


def sma_cn(x, n):
    """通达信 SMA(X,N,1)：Y=(X+(N-1)Y')/N"""
    y = np.empty_like(x, dtype=float)
    prev = np.nan
    for i in range(len(x)):
        xi = x[i]
        if not np.isfinite(xi):
            y[i] = prev
            continue
        prev = xi if not np.isfinite(prev) else (xi + (n - 1) * prev) / n
        y[i] = prev
    return y


def stoch(h, l, c):
    hh = pl.Series(h).rolling_max(SN).to_numpy()
    ll = pl.Series(l).rolling_min(SN).to_numpy()
    k = sma_cn((c - ll) / np.clip(hh - ll, 1e-12, None) * 100, SK)
    d = sma_cn(k, SD)
    return k, d


def pivots(x, k, mode):
    """返回 (pivot 位置, 值)。平台（连续相等）只保留最后一个。"""
    n = len(x)
    w = 2 * k + 1
    sw = np.lib.stride_tricks.sliding_window_view(x, w)
    mid = x[k:n - k]
    if mode == "high":
        is_p = (mid == sw.max(axis=1)) & ((sw == mid[:, None]).sum(axis=1) == 1)
    else:
        is_p = (mid == sw.min(axis=1)) & ((sw == mid[:, None]).sum(axis=1) == 1)
    idx = np.where(is_p & np.isfinite(mid))[0] + k
    return idx, x[idx]


def line_cross(x, piv_t, piv_v, k, hi=400, min_gap=2):
    """指标穿越「最近两个同向 pivot 的连线」。pivot 在 t+k 才确认，无前视。"""
    n = len(x)
    conf = piv_t + k
    pos = np.searchsorted(conf, np.arange(n), side="right") - 1
    i1 = pos
    i2 = pos - 1
    ok = i2 >= 0
    i1c = np.clip(i1, 0, len(piv_t) - 1)
    i2c = np.clip(i2, 0, len(piv_t) - 1)
    t1, v1 = piv_t[i1c], piv_v[i1c]
    t2, v2 = piv_t[i2c], piv_v[i2c]
    gap = t1 - t2
    slope = (v1 - v2) / np.clip(gap, 1, None)
    tt = np.arange(n)
    line = v1 + slope * (tt - t1)
    line_prev = v1 + slope * (tt - 1 - t1)
    same = np.r_[False, pos[1:] == pos[:-1]]
    fresh = (tt - t1) <= hi
    valid = ok & same & fresh & (gap >= min_gap) & np.isfinite(slope)
    xp = np.r_[np.nan, x[:-1]]
    up = valid & (xp <= line_prev) & (x > line)
    dn = valid & (xp >= line_prev) & (x < line)
    return up, dn, slope, line


def resample(df, minutes):
    ms = minutes * 60_000
    return (df.with_columns(b=pl.col("close_time") // ms)
            .group_by("b").agg(ct=pl.col("close_time").last(),
                               high=pl.col("high").max(), low=pl.col("low").min(),
                               close=pl.col("close").last(), cnt=pl.len())
            .sort("b").filter(pl.col("cnt") == minutes))


def stat_cell(win, ret):
    if len(ret) < 30:
        return "      n/a        "
    p = float(win.mean())
    return f"{p*100:5.2f}%/{1.8*p-1:+.3f}"


def analyse(df, minutes):
    c = df["close"].to_numpy()
    h = df["high"].to_numpy()
    l = df["low"].to_numpy()
    ct = df["ct"].to_numpy()
    n = len(c)
    lr = np.r_[np.nan, np.log(c[1:] / c[:-1])]
    K, D = stoch(h, l, c)
    ph_t, ph_v = pivots(K, PIV_K, "high")
    pl_t, pl_v = pivots(K, PIV_K, "low")
    up_peak, _, s_up, _ = line_cross(K, ph_t, ph_v, PIV_K)      # 上破峰连线 → 做多
    _, dn_trough, s_dn, _ = line_cross(K, pl_t, pl_v, PIV_K)    # 下破谷连线 → 做空

    nxt80 = np.minimum.accumulate(np.where(K >= 80, np.arange(n), n)[::-1])[::-1]
    nxt20 = np.minimum.accumulate(np.where(K <= 20, np.arange(n), n)[::-1])[::-1]

    fut = {}
    for hh in HS:
        f = np.full(n, np.nan)
        f[:-hh] = np.log(c[hh:] / c[:-hh])
        fut[hh] = f

    log(f"\n{'='*112}")
    log(f"[{SYMBOL} {minutes}m] Stoch({SN},{SK},{SD})  n={n:,}  "
        f"{pl.from_epoch(pl.Series(ct[:1]), 'ms')[0]} ~ {pl.from_epoch(pl.Series(ct[-1:]), 'ms')[0]}")
    log(f"  pivot: high {len(ph_t):,} / low {len(pl_t):,}   "
        f"突破: 上破峰线 {int(up_peak.sum()):,} / 下破谷线 {int(dn_trough.sum()):,}")

    # ---- 斜率分布 ----
    su = s_up[up_peak]
    sd = s_dn[dn_trough]
    log(f"  上破峰线斜率分位: " + " ".join(f"p{q}={np.percentile(su,q):+.3f}" for q in (10, 50, 90)))
    log(f"  下破谷线斜率分位: " + " "  .join(f"p{q}={np.percentile(sd,q):+.3f}" for q in (10, 50, 90)))

    # ---- 按斜率分档：固定持有 ----
    for name, mask, sgn, side, nxt in (
            ("上破峰线(做多)", up_peak, s_up, +1, nxt80),
            ("下破谷线(做空)", dn_trough, s_dn, -1, nxt20)):
        idx = np.where(mask)[0]
        idx = idx[idx + max(HS) < n - 1]
        if len(idx) < 100:
            log(f"  {name}: 事件不足 {len(idx)}")
            continue
        sv = sgn[idx]
        qs = np.unique(np.percentile(sv, [0, 20, 40, 60, 80, 100]))
        log(f"\n  {name}  事件 {len(idx):,}（按斜率分档，斜率为指标点/bar）")
        log("     斜率档            n    斜率中位   " + "  ".join(f"h={hh}" for hh in HS)
            + "   到极值中位(bar)")
        for i in range(len(qs) - 1):
            m = (sv >= qs[i]) & (sv <= qs[i + 1] if i == len(qs) - 2 else sv < qs[i + 1])
            if m.sum() < 30:
                continue
            cells = []
            for hh in HS:
                r = side * fut[hh][idx[m]] * 1e4
                r = r[np.isfinite(r)]
                cells.append(stat_cell(r > 0, r))
            tt = nxt[idx[m]] - idx[m]
            tt = tt[tt < n]
            log(f"   [{qs[i]:+7.3f},{qs[i+1]:+7.3f}]  {int(m.sum()):5,}   "
                f"{np.median(sv[m]):+7.3f}   " + "  ".join(cells)
                + f"   {np.median(tt) if len(tt) else float('nan'):6.1f}")
        # 全体
        cells = []
        for hh in HS:
            r = side * fut[hh][idx] * 1e4
            r = r[np.isfinite(r)]
            cells.append(stat_cell(r > 0, r))
        tt = nxt[idx] - idx
        tt = tt[tt < n]
        log(f"   {'全体':>16}  {len(idx):5,}   {np.median(sv):+7.3f}   " + "  ".join(cells)
            + f"   {np.median(tt) if len(tt) else float('nan'):6.1f}")

        # 分年
        yr = pl.Series(ct[idx]).cast(pl.Datetime("ms")).dt.year().to_numpy()
        seg = []
        for y in sorted(set(yr)):
            m = yr == y
            r = side * fut[3][idx[m]] * 1e4
            r = r[np.isfinite(r)]
            if len(r) >= 30:
                seg.append(f"{y}:{float((r>0).mean())*100:.1f}%")
        log(f"   分年 h=3 胜率: " + "  ".join(seg))

    # ---- 对照：普通 K/D 交叉 ----
    cu = np.where((K[1:] > D[1:]) & (K[:-1] <= D[:-1]))[0] + 1
    cd = np.where((K[1:] < D[1:]) & (K[:-1] >= D[:-1]))[0] + 1
    log(f"\n  对照 K/D 交叉：金叉 {len(cu):,} / 死叉 {len(cd):,}")
    for name, idx, side in (("金叉做多", cu, +1), ("死叉做空", cd, -1)):
        idx = idx[idx + max(HS) < n - 1]
        cells = []
        for hh in HS:
            r = side * fut[hh][idx] * 1e4
            r = r[np.isfinite(r)]
            cells.append(stat_cell(r > 0, r))
        log(f"   {name:10s} n={len(idx):6,}  " + "  ".join(cells))

    # ---- 到极值时间的整体分布 ----
    for name, nxt, thr in (("到 80", nxt80, 80), ("到 20", nxt20, 20)):
        tt = nxt[up_peak] - np.where(up_peak)[0]
        tt = tt[tt < n]
        if len(tt) > 50:
            log(f"  上破后{name} 的 bar 数: 中位 {np.median(tt):.0f}  p25={np.percentile(tt,25):.0f} "
                f"p75={np.percentile(tt,75):.0f}  ≤5根占 {float((tt<=5).mean())*100:.1f}%")


def main():
    files = [fetch(f"{SYMBOL}_1m_klines_{y}.parquet") for y in YEARS]
    raw = pl.concat([pl.read_parquet(f).select("close_time", "high", "low", "close")
                     for f in files]).sort("close_time")
    log(f"数据 {raw.height:,} 行  {len(files)} 个年度文件  symbol={SYMBOL}")
    for minutes in MINS:
        analyse(resample(raw, minutes), minutes)
    log("\nDONE")


if __name__ == "__main__":
    main()
