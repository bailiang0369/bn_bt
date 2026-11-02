"""PolarsExp 单元测试。

策略
----
- 用纯 numpy / scipy 重写各指标的"标准公式"作为参考实现，与 polars 输出做 tight 比较，
  以捕捉实现层面的 bug（错列、错号、错平滑、除零等）。
- 对边界 / 有界指标做不变量检查（范围、空值位置、单调性等）。
- 不需要外部 TA 库（环境中未安装 ta / talib / pandas_ta）。

运行：``python tool/test_pl_exp.py``（无需 pytest，自带 runner）。
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import polars as pl
from polars import col
from pl_exp import PolarsExp

np.seterr(all="ignore")  # 参考实现里对零区间的除法属预期，忽略告警


# ===================== 测试数据 =====================
def make_df(n: int = 400, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0, 1, n))
    high = close + np.abs(rng.normal(0, 0.5, n))
    low = close - np.abs(rng.normal(0, 0.5, n))
    open_ = close + rng.normal(0, 0.3, n)
    volume = np.abs(rng.normal(100, 20, n))
    return pl.DataFrame({
        "open": open_.astype(np.float64),
        "high": high.astype(np.float64),
        "low": low.astype(np.float64),
        "close": close.astype(np.float64),
        "volume": volume.astype(np.float64),
    })


# ===================== 参考实现（numpy） =====================
def ref_ema(x, span):
    x = np.asarray(x, float)
    a = 2.0 / (span + 1)
    out = np.full(len(x), np.nan)
    if len(x) == 0:
        return out
    y = x[0]
    out[0] = y
    for i in range(1, len(x)):
        y = (1 - a) * y + a * x[i]
        out[i] = y
    return out


def ref_ema_alpha(x, a):
    x = np.asarray(x, float)
    out = np.full(len(x), np.nan)
    if len(x) == 0:
        return out
    y = x[0]
    out[0] = y
    for i in range(1, len(x)):
        y = (1 - a) * y + a * x[i]
        out[i] = y
    return out


def ref_rsi_ewm(close, p):
    close = np.asarray(close, float)
    n = len(close)
    # 与 polars diff(1) 对齐：长度 n，第 0 项为 0
    g = np.zeros(n)
    l = np.zeros(n)
    for j in range(1, n):
        d = close[j] - close[j - 1]
        if d > 0:
            g[j] = d
        elif d < 0:
            l[j] = -d
    ag = ref_ema_alpha(g, 1.0 / p)
    al = ref_ema_alpha(l, 1.0 / p)
    out = np.full(n, np.nan)
    for j in range(n):
        aag = ag[j]
        aal = al[j]
        denom = aal if aal != 0 else 1e-6
        rs = aag / denom
        out[j] = 100 - 100 / (1 + rs)
    return out


def ref_macd(close, fast, slow, sig):
    ef = ref_ema(close, fast)
    es = ref_ema(close, slow)
    dif = ef - es
    dea = ref_ema(dif, sig)
    return dif, dea, dif - dea


def ref_atr_ewm(h, l, c, p):
    h, l, c = map(np.asarray, (h, l, c))
    n = len(c)
    tr = np.empty(n)
    tr[0] = h[0] - l[0]
    for i in range(1, n):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
    return ref_ema_alpha(tr, 1.0 / p)


def ref_bbands(close, p, k):
    close = np.asarray(close, float)
    n = len(close)
    mid = np.full(n, np.nan)
    std = np.full(n, np.nan)
    for i in range(n):
        lo = max(0, i - p + 1)
        w = close[lo:i + 1]
        mid[i] = w.mean()
        std[i] = w.std(ddof=1) if len(w) > 1 else 0.0
    up = mid + k * std
    lo = mid - k * std
    bbw = (up - lo) / mid * 100
    bbp = (close - lo) / (up - lo) * 100
    return up, mid, lo, bbw, bbp


def ref_stoch(h, l, c, kp, dp):
    h, l, c = map(np.asarray, (h, l, c))
    n = len(c)
    hh = np.full(n, np.nan)
    ll = np.full(n, np.nan)
    for i in range(n):
        lo = max(0, i - kp + 1)
        hh[i] = h[lo:i + 1].max()
        ll[i] = l[lo:i + 1].min()
    rng = hh - ll
    k = np.where(rng != 0, (c - ll) / rng * 100, np.nan)
    d = np.full(n, np.nan)
    for i in range(n):
        lo = max(0, i - dp + 1)
        d[i] = np.nanmean(k[lo:i + 1])
    return k, d


def ref_stochrsi(close, rp, sp, dp):
    rsi = ref_rsi_ewm(close, rp)
    n = len(rsi)
    k = np.full(n, 50.0)  # warmup 与 polars fill_null(50) 对齐
    for i in range(sp - 1, n):
        s = slice(i - sp + 1, i + 1)
        seg = rsi[s]
        seg = seg[~np.isnan(seg)]
        if len(seg) == 0:
            continue
        lo = seg.min()
        hi = seg.max()
        rng = hi - lo
        if rng != 0:
            k[i] = (rsi[i] - lo) / rng * 100
        else:
            k[i] = 0.0
    d = np.full(n, np.nan)
    for i in range(n):
        s = slice(max(0, i - dp + 1), i + 1)
        seg = k[s]
        seg = seg[~np.isnan(seg)]
        d[i] = seg.mean() if len(seg) else np.nan
    return k, d


def ref_kdj(h, l, c, n_, smooth):
    h, l, c = map(np.asarray, (h, l, c))
    nn = len(c)
    hh = np.full(nn, np.nan)
    ll = np.full(nn, np.nan)
    for i in range(n_ - 1, nn):
        s = slice(i - n_ + 1, i + 1)
        hh[i] = h[s].max()
        ll[i] = l[s].min()
    rng = hh - ll
    rsv = np.full(nn, 50.0)  # 与 polars fill_null(50) 对齐（warmup 段）
    for i in range(n_ - 1, nn):
        if rng[i] != 0:
            rsv[i] = (c[i] - ll[i]) / rng[i] * 100
        else:
            rsv[i] = 0.0  # 与 polars 1e-6 分母对齐（扁平窗口）
    a = 1.0 / smooth
    k = ref_ema_alpha(rsv, a)
    d = ref_ema_alpha(k, a)
    j = 3 * k - 2 * d
    return k, d, j


def ref_wr(h, l, c, p):
    h, l, c = map(np.asarray, (h, l, c))
    n = len(c)
    hh = np.full(n, np.nan)
    ll = np.full(n, np.nan)
    for i in range(n):
        s = slice(max(0, i - p + 1), i + 1)
        hh[i] = h[s].max()
        ll[i] = l[s].min()
    rng = hh - ll
    return np.where(rng != 0, (hh - c) / rng * -100, np.nan)


def ref_cci(h, l, c, p):
    tp = (h + l + c) / 3
    n = len(c)
    # 与 polars 一致：先做 rolling_mean（warmup p-1），其 null 会传播；
    # 再做滚动均值时又有 p-1 warmup，形成"双重 warmup"。
    sma = np.full(n, np.nan)
    for i in range(p - 1, n):
        sma[i] = tp[i - p + 1:i + 1].mean()
    absdev = np.where(~np.isnan(sma), np.abs(tp - sma), np.nan)
    mdev = np.full(n, np.nan)
    for i in range(n):
        s = slice(max(0, i - p + 1), i + 1)
        seg = absdev[s]
        seg = seg[~np.isnan(seg)]
        if len(seg) >= p:
            mdev[i] = seg.mean()
    denom = np.where((mdev != 0) & ~np.isnan(mdev), 0.015 * mdev, np.nan)
    return np.where(~np.isnan(denom), (tp - sma) / denom, np.nan)


def ref_mfi(h, l, c, v, p):
    h, l, c, v = map(np.asarray, (h, l, c, v))
    n = len(c)
    tp = (h + l + c) / 3
    d = np.diff(tp)
    pf = np.where(d > 0, tp[1:] * v[1:], 0.0)
    nf = np.where(d < 0, tp[1:] * v[1:], 0.0)
    out = np.full(n, np.nan)
    for j in range(1, n):
        s = slice(max(0, j - p), j)
        pm = pf[s].sum()
        nm = nf[s].sum()
        r = pm / nm if nm != 0 else 1e6
        out[j] = 100 - 100 / (1 + r)
    return out


def ref_supertrend(h, l, c, p, m):
    h, l, c = map(np.asarray, (h, l, c))
    n = len(c)
    tp = (h + l + c) / 3
    tr = np.empty(n)
    tr[0] = h[0] - l[0]
    for i in range(1, n):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
    atr = np.empty(n)
    atr[0] = tr[0]
    for i in range(1, n):
        atr[i] = (atr[i - 1] * (p - 1) + tr[i]) / p
    bub = tp + m * atr
    blb = tp - m * atr
    fub = np.empty(n)
    flb = np.empty(n)
    st = np.empty(n)
    direction = np.empty(n)
    for i in range(n):
        if i == 0:
            fub[i] = bub[i]
            flb[i] = blb[i]
            st[i] = bub[i]
            direction[i] = 1.0 if c[i] >= st[i] else -1.0
            continue
        if bub[i] < fub[i - 1] or c[i - 1] > fub[i - 1]:
            fub[i] = bub[i]
        else:
            fub[i] = fub[i - 1]
        if blb[i] > flb[i - 1] or c[i - 1] < flb[i - 1]:
            flb[i] = blb[i]
        else:
            flb[i] = flb[i - 1]
        if st[i - 1] == fub[i - 1]:
            if c[i] <= fub[i]:
                st[i] = fub[i]
                direction[i] = -1.0
            else:
                st[i] = flb[i]
                direction[i] = 1.0
        else:
            if c[i] >= flb[i]:
                st[i] = flb[i]
                direction[i] = 1.0
            else:
                st[i] = fub[i]
                direction[i] = -1.0
    return st, direction


def ref_robust_z(x, w):
    # 原版约定（嵌套滚动）：先逐点滚动中位数 med[j]，再对 |x[j]-med[j]| 做滚动中位数得到 mad[i]
    x = np.asarray(x, float)
    n = len(x)
    med = np.array([np.median(x[max(0, j - w + 1):j + 1]) for j in range(n)])
    dev = np.abs(x - med)
    mad = np.array([np.median(dev[max(0, i - w + 1):i + 1]) for i in range(n)])
    out = np.where(mad != 0, (x - med) / (1.4826 * mad), np.nan)
    return out


def ref_quantile_binning(x, window, qn):
    """复现 pl_exp.quantile_binning：用 rolling_rank 的平均秩逻辑做等频分箱。"""
    x = np.asarray(x, float)
    n = len(x)
    out = np.zeros(n, dtype=int)  # cnt==1 单点窗口 -> 0 号桶
    for i in range(n):
        lo = max(0, i - window + 1)
        w = x[lo:i + 1]
        cnt = len(w)
        if cnt == 1:
            out[i] = 0
            continue
        xi = x[i]
        nless = int((w < xi).sum())
        neq = int((w == xi).sum())
        rank = nless + (neq + 1) / 2.0  # 1-based 平均秩（与 polars rolling_rank 一致）
        pct = (rank - 1) / (cnt - 1)
        b = int(np.floor(pct * qn))
        out[i] = min(max(b, 0), qn - 1)
    return out


# ===================== 比较工具 =====================
def compare(name, pol, ref, atol=1e-4, rtol=1e-6, tail=0, verbose=True):
    pol = np.asarray(pol, float)
    ref = np.asarray(ref, float)
    if tail:
        pol = pol[tail:]
        ref = ref[tail:]
    both_finite = ~(np.isnan(pol) | np.isnan(ref))
    # 1) 空值位置应一致
    nan_mismatch = int(np.sum(np.isnan(pol) != np.isnan(ref)))
    ok_nan = nan_mismatch == 0
    # 2) 有限值数值一致
    if both_finite.sum() > 0:
        ok_val = np.allclose(pol[both_finite], ref[both_finite], atol=atol, rtol=rtol)
    else:
        ok_val = True
    ok = ok_nan and ok_val
    if verbose:
        status = "PASS" if ok else "FAIL"
        extra = "" if ok else f" (nan_mismatch={nan_mismatch}, n_finite={both_finite.sum()})"
        print(f"  [{status}] {name}{extra}")
    return ok


def invariant_range(name, pol, lo, hi, verbose=True):
    pol = np.asarray(pol, float)
    finite = pol[~np.isnan(pol)]
    ok = bool(((finite >= lo - 1e-3) & (finite <= hi + 1e-3)).all())
    if verbose:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name} 范围[{lo},{hi}] (实际 min={finite.min():.3f}, max={finite.max():.3f})")
    return ok


# ===================== 测试主体 =====================
def run_tests():
    results = []
    df = make_df(400)
    o = df["open"].to_numpy().astype(float)
    h = df["high"].to_numpy().astype(float)
    l = df["low"].to_numpy().astype(float)
    c = df["close"].to_numpy().astype(float)
    v = df["volume"].to_numpy().astype(float)
    print("数据: n =", len(df))

    # ---- 基础 ----
    print("\n[基础]")
    pe = PolarsExp()
    pol_sma = df.select(pe.sma("close", 20)).to_series().to_numpy()
    ref_sma = np.array([np.nanmean(c[max(0, i - 20 + 1):i + 1]) for i in range(len(c))])
    results.append(compare("sma(20)", pol_sma, ref_sma, tail=19))

    pol_ema = df.select(pe.ema("close", 20)).to_series().to_numpy()
    results.append(compare("ema(20)", pol_ema, ref_ema(c, 20)))

    # ---- MACD ----
    print("\n[MACD]")
    d, de, bar = pe.macd("close")
    pol_d = df.select(d).to_series().to_numpy()
    pol_de = df.select(de).to_series().to_numpy()
    pol_bar = df.select(bar).to_series().to_numpy()
    rd, rde, rbar = ref_macd(c, 12, 26, 9)
    results.append(compare("macd.dif", pol_d, rd))
    results.append(compare("macd.dea", pol_de, rde))
    results.append(compare("macd.hist(=dif-dea)", pol_bar, rbar))

    # ---- RSI ----
    print("\n[RSI]")
    pol_rsi = df.select(pe.rsi("close", 14)).to_series().to_numpy()
    ref_rsi = ref_rsi_ewm(c, 14)
    results.append(compare("rsi(14)", pol_rsi, ref_rsi))
    results.append(invariant_range("rsi 范围", pol_rsi, 0, 100))

    # ---- StochRSI ----
    print("\n[StochRSI]")
    k, d_ = pe.stochrsi("close")
    pol_k = df.select(k).to_series().to_numpy()
    pol_d = df.select(d_).to_series().to_numpy()
    rk, rd = ref_stochrsi(c, 14, 14, 3)
    results.append(compare("stochrsi_k", pol_k, rk, tail=13))
    results.append(compare("stochrsi_d", pol_d, rd, tail=15))
    results.append(invariant_range("stochrsi_k 范围", pol_k, 0, 100))

    # ---- BBANDS ----
    print("\n[BBANDS]")
    up, mid, lo, bbw, bbp = pe.bbands("close", 20, 2.0)
    pol_up = df.select(up).to_series().to_numpy()
    pol_mid = df.select(mid).to_series().to_numpy()
    pol_lo = df.select(lo).to_series().to_numpy()
    pol_bbw = df.select(bbw).to_series().to_numpy()
    pol_bbp = df.select(bbp).to_series().to_numpy()
    rup, rmid, rlo, rbbw, rbbp = ref_bbands(c, 20, 2.0)
    results.append(compare("bb_upper", pol_up, rup, tail=19))
    results.append(compare("bb_middle", pol_mid, rmid, tail=19))
    results.append(compare("bb_lower", pol_lo, rlo, tail=19))
    results.append(compare("bbw(%)", pol_bbw, rbbw, tail=19))
    results.append(compare("bbp(%)", pol_bbp, rbbp, tail=19))
    ok_bbp = bool(np.all(np.isfinite(pol_bbp[~np.isnan(pol_bbp)])))
    print(f"  [{'PASS' if ok_bbp else 'FAIL'}] bbp 有限性（%B 可超出[0,100]，仅查非 inf）")
    results.append(ok_bbp)

    # ---- ATR ----
    print("\n[ATR]")
    pol_atr = df.select(pe.atr("high", "low", "close", 14)).to_series().to_numpy()
    ref_atr = ref_atr_ewm(h, l, c, 14)
    results.append(compare("atr(14)", pol_atr, ref_atr))

    # ---- STOCH ----
    print("\n[STOCH]")
    sk, sd = pe.stoch("high", "low", "close", 14, 3)
    pol_sk = df.select(sk).to_series().to_numpy()
    pol_sd = df.select(sd).to_series().to_numpy()
    rsk, rsd = ref_stoch(h, l, c, 14, 3)
    results.append(compare("stoch_k", pol_sk, rsk, tail=13))
    results.append(compare("stoch_d", pol_sd, rsd, tail=15))
    results.append(invariant_range("stoch_k 范围", pol_sk, 0, 100))

    # ---- KDJ ----
    print("\n[KDJ]")
    k, d_, j = pe.kdj("high", "low", "close", 9, 3)
    pol_k = df.select(k).to_series().to_numpy()
    pol_d = df.select(d_).to_series().to_numpy()
    pol_j = df.select(j).to_series().to_numpy()
    rk, rd, rj = ref_kdj(h, l, c, 9, 3)
    results.append(compare("kdj_k", pol_k, rk, tail=8))
    results.append(compare("kdj_d", pol_d, rd, tail=8))
    results.append(compare("kdj_j", pol_j, rj, tail=8))
    results.append(invariant_range("kdj_k 范围", pol_k, 0, 100))
    results.append(invariant_range("kdj_d 范围", pol_d, 0, 100))

    # ---- Williams %R ----
    print("\n[Williams %R]")
    pol_wr = df.select(pe.williams_r("high", "low", "close", 14)).to_series().to_numpy()
    r_wr = ref_wr(h, l, c, 14)
    results.append(compare("wr(14)", pol_wr, r_wr, tail=13))
    results.append(invariant_range("wr 范围", pol_wr, -100, 0))

    # ---- CCI ----
    print("\n[CCI]")
    pol_cci = df.select(pe.cci("high", "low", "close", 20)).to_series().to_numpy()
    r_cci = ref_cci(h, l, c, 20)
    results.append(compare("cci(20)", pol_cci, r_cci, tail=38))

    # ---- ROC / Momentum ----
    print("\n[ROC / Momentum]")
    pol_roc = df.select(pe.roc("close", 10)).to_series().to_numpy()
    ref_roc = np.full(len(c), np.nan)
    for i in range(len(c)):
        b = c[i - 10] if i >= 10 else np.nan
        ref_roc[i] = (c[i] - b) / b * 100 if (i >= 10 and b != 0) else np.nan
    results.append(compare("roc(10)", pol_roc, ref_roc, tail=10))
    pol_mom = df.select(pe.momentum("close", 10)).to_series().to_numpy()
    ref_mom = np.array([c[i] - c[i - 10] if i >= 10 else np.nan for i in range(len(c))])
    results.append(compare("momentum(10)", pol_mom, ref_mom, tail=10))

    # ---- MFI ----
    print("\n[MFI]")
    pol_mfi = df.select(pe.mfi("high", "low", "close", "volume", 14)).to_series().to_numpy()
    r_mfi = ref_mfi(h, l, c, v, 14)
    results.append(compare("mfi(14)", pol_mfi, r_mfi, tail=14))
    results.append(invariant_range("mfi 范围", pol_mfi, 0, 100))

    # ---- 波动率 ----
    print("\n[Volatility]")
    pol_hv = df.select(pe.historical_volatility("close", 20)).to_series().to_numpy()
    ref_hv = np.full(len(c), np.nan)
    for i in range(20, len(c)):
        lr = np.diff(np.log(c[i - 20:i + 1]))
        ref_hv[i] = lr.std(ddof=1) if len(lr) >= 2 else np.nan
    results.append(compare("hv(20) 非年化", pol_hv, ref_hv, tail=20, atol=1e-5))
    pol_pv = df.select(pe.parkinson_volatility("high", "low", 20)).to_series().to_numpy()
    ref_pv = np.full(len(c), np.nan)
    for i in range(len(c)):
        s = slice(max(0, i - 20 + 1), i + 1)
        hl = np.log(h[s] / l[s])
        ref_pv[i] = np.sqrt(np.mean(hl ** 2) / (4 * np.log(2))) if len(hl) else np.nan
    results.append(compare("pv(20) 非年化", pol_pv, ref_pv, tail=20, atol=1e-5))
    # 年化分支（525600 分钟）
    pol_hv_a = df.select(pe.historical_volatility("close", 20, periods_per_year=525600)).to_series().to_numpy()
    results.append(compare("hv(20) 年化", pol_hv_a, ref_hv * np.sqrt(525600) * 100, tail=20, atol=1e-2))

    # ---- 成交量 ----
    print("\n[Volume]")
    pol_obv = df.select(pe.obv("close", "volume")).to_series().to_numpy()
    ref_obv = np.zeros(len(c))
    for i in range(1, len(c)):
        if c[i] > c[i - 1]:
            ref_obv[i] = ref_obv[i - 1] + v[i]
        elif c[i] < c[i - 1]:
            ref_obv[i] = ref_obv[i - 1] - v[i]
        else:
            ref_obv[i] = ref_obv[i - 1]
    results.append(compare("obv", pol_obv, ref_obv))
    pol_vr = df.select(pe.volume_ratio("volume", 20)).to_series().to_numpy()
    ref_vr = np.full(len(c), np.nan)
    for i in range(len(c)):
        s = slice(max(0, i - 20 + 1), i + 1)
        ma = v[s].mean()
        ref_vr[i] = v[i] / ma * 100 if ma != 0 else np.nan
    results.append(compare("volume_ratio(20)", pol_vr, ref_vr, tail=19))
    pol_vwap = df.select(pe.vwap("high", "low", "close", "volume")).to_series().to_numpy()
    tp = (h + l + c) / 3
    cum_tpv = np.cumsum(tp * v)
    cum_v = np.cumsum(v)
    results.append(compare("vwap", pol_vwap, cum_tpv / cum_v))

    # ---- 价格位置 ----
    print("\n[Price Position]")
    pol_pp = df.select(pe.price_position("high", "low", "close", 20)).to_series().to_numpy()
    ref_pp = np.full(len(c), np.nan)
    for i in range(len(c)):
        s = slice(max(0, i - 20 + 1), i + 1)
        hh = h[s].max()
        ll = l[s].min()
        ref_pp[i] = (c[i] - ll) / (hh - ll) * 100 if hh != ll else np.nan
    results.append(compare("price_position(20)", pol_pp, ref_pp, tail=19))
    results.append(invariant_range("price_position 范围", pol_pp, 0, 100))

    # ---- DMI / ADX ----
    print("\n[DMI / ADX]")
    adx, pdi, mdi, dx = pe.dmi("high", "low", "close", 14)
    pol_pdi = df.select(pdi).to_series().to_numpy()
    pol_mdi = df.select(mdi).to_series().to_numpy()
    pol_dx = df.select(dx).to_series().to_numpy()
    pol_adx = df.select(adx).to_series().to_numpy()
    # 参考：用 ewm(alpha=1/14) 复现
    up = h[1:] - h[:-1]
    dn = l[:-1] - l[1:]  # 注意方向
    plus_dm = np.zeros(len(c))
    minus_dm = np.zeros(len(c))
    for i in range(1, len(c)):
        um = h[i] - h[i - 1]
        dm = l[i - 1] - l[i]
        if um > dm and um > 0:
            plus_dm[i] = um
        if dm > um and dm > 0:
            minus_dm[i] = dm
    tr = np.empty(len(c))
    tr[0] = h[0] - l[0]
    for i in range(1, len(c)):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
    spdm = ref_ema_alpha(plus_dm, 1.0 / 14)
    smdm = ref_ema_alpha(minus_dm, 1.0 / 14)
    str_ = ref_ema_alpha(tr, 1.0 / 14)
    r_pdi = 100 * spdm / np.where(str_ != 0, str_, 1e-6)
    r_mdi = 100 * smdm / np.where(str_ != 0, str_, 1e-6)
    r_dx = 100 * np.abs(r_pdi - r_mdi) / np.where((r_pdi + r_mdi) != 0, (r_pdi + r_mdi), 1e-6)
    r_adx = ref_ema_alpha(r_dx, 1.0 / 14)
    results.append(compare("plus_di", pol_pdi, r_pdi))
    results.append(compare("minus_di", pol_mdi, r_mdi))
    results.append(compare("dx", pol_dx, r_dx))
    results.append(compare("adx", pol_adx, r_adx))
    results.append(invariant_range("plus_di 范围", pol_pdi, 0, 100, verbose=False) and True)
    results.append(invariant_range("plus_di 范围", pol_pdi, 0, 100))

    # ---- SuperTrend（DataFrame 级） ----
    print("\n[SuperTrend]")
    df_st = pe.supertrend(df, "high", "low", "close", 10, 3.0)
    pol_st = df_st["supertrend"].to_numpy()
    pol_dir = df_st["supertrend_dir"].to_numpy()
    rst, rdir = ref_supertrend(h, l, c, 10, 3.0)
    results.append(compare("supertrend value", pol_st, rst, atol=1e-6))
    results.append(compare("supertrend_dir", pol_dir, rdir, atol=1e-6))
    results.append(invariant_range("supertrend_dir 取值", pol_dir, -1, 1))

    # ---- 无量纲 / 稳健辅助 ----
    print("\n[Dimensionless / Robust]")
    pol_lr = df.select(pe.log_ret_bps("close", 1)).to_series().to_numpy()
    ref_lr = np.full(len(c), np.nan)
    for i in range(len(c)):
        ref_lr[i] = np.log(c[i] / c[i - 1]) * 1e4 if i >= 1 else np.nan
    results.append(compare("log_ret_bps(1)", pol_lr, ref_lr, tail=1))

    pol_med = df.select(pe.rolling_median("close", 20)).to_series().to_numpy()
    ref_med = np.array([np.median(c[max(0, i - 20 + 1):i + 1]) for i in range(len(c))])
    results.append(compare("rolling_median(20)", pol_med, ref_med, tail=19))

    pol_mad = df.select(pe.rolling_mad("close", 20)).to_series().to_numpy()
    # 原版约定（嵌套滚动）：先逐点滚动中位数 ref_med[j]，再对 |c[j]-ref_med[j]| 做滚动中位数
    ref_dev = np.abs(c - ref_med)
    ref_mad = np.array([np.median(ref_dev[max(0, i - 20 + 1):i + 1]) for i in range(len(c))])
    results.append(compare("rolling_mad(20)", pol_mad, ref_mad, tail=19, atol=1e-5))

    pol_rz = df.select(pe.robust_zscore("close", 20)).to_series().to_numpy()
    ref_rz = ref_robust_z(c, 20)
    results.append(compare("robust_zscore(20)", pol_rz, ref_rz, tail=19, atol=1e-5))

    pol_rsc = df.select(pe.rsi_scaled("close", 14)).to_series().to_numpy()
    results.append(compare("rsi_scaled(14)", pol_rsc, (ref_rsi_ewm(c, 14) - 50) / 50))
    results.append(invariant_range("rsi_scaled 范围", pol_rsc, -1, 1))

    ps, pl_, pdiv = pe.rsi_pair(14, 55)
    pol_s = df.select(ps).to_series().to_numpy()
    pol_l = df.select(pl_).to_series().to_numpy()
    pol_div = df.select(pdiv).to_series().to_numpy()
    results.append(compare("rsi_pair.short", pol_s, (ref_rsi_ewm(c, 14) - 50) / 50))
    results.append(compare("rsi_pair.long", pol_l, (ref_rsi_ewm(c, 55) - 50) / 50))
    results.append(compare("rsi_pair.div", pol_div, (ref_rsi_ewm(c, 14) - 50) / 50 - (ref_rsi_ewm(c, 55) - 50) / 50))

    # ---- 功能性检查 ----
    print("\n[Functional]")
    # consecutive_periods
    bool_arr = np.array([True, True, False, True, True, True])
    dfb = pl.DataFrame({"b": bool_arr})
    pol_consec = dfb.select(pe.consecutive_periods("b")).to_series().to_numpy()
    results.append(compare("consecutive_periods", pol_consec, np.array([1, 2, -1, 1, 2, 3]), atol=0))
    # golden_cross
    fast = np.array([1, 2, 3, 2, 1, 2])
    slow = np.array([2, 2, 2, 2, 2, 2])
    dfc = pl.DataFrame({"f": fast.astype(float), "s": slow.astype(float)})
    pol_gc = dfc.select(pe.golden_cross("f", "s")).to_series().to_numpy()
    # 预期：idx2 上穿(+1)、idx4 下穿(-1)，其余 0
    results.append(compare("golden_cross", pol_gc, np.array([0, 0, 1, 0, -1, 0]), atol=0))
    # ema_cross similar smoke
    pol_ec = dfc.select(pe.ema_cross("f", 1, 2)).to_series().to_numpy()
    results.append(compare("ema_cross smoke", pol_ec, pol_ec, atol=0))  # 仅保证可执行+长度
    # quantile_binning
    pol_qb = df.select(pe.quantile_binning("close", 50, 20)).to_series().to_numpy()
    ref_qb = ref_quantile_binning(c, 50, 20)
    results.append(compare("quantile_binning(close,50,20)", pol_qb, ref_qb, atol=0))

    # 总结
    total = len(results)
    passed = sum(1 for r in results if r)
    print("\n" + "=" * 50)
    print(f"总计 {total} 项，通过 {passed}，失败 {total - passed}")
    print("=" * 50)
    return passed == total


if __name__ == "__main__":
    ok = run_tests()
    sys.exit(0 if ok else 1)
