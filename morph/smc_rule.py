#!/usr/bin/env python3
"""SMC 固定规则独立回测（FVG / Order Block / 流动性扫单）——事件合约二元涨跌口径。

与项目现役评估完全对齐（two_legs.add_target）：
  目标 = (close[t+h] > close[t])，二元；赔率 ODDS=0.8 → 盈亏平衡胜率 1/(1+0.8)=55.56%。
  **无手续费、无止损止盈、无滑点**：事件合约就是押方向，不存在挂单成交/中途平仓。
  唯一"成本"是胜负本身，EV = ODDS·p − (1−p)，按每单计。

设计原则：
  1. 规则与参数在跑之前写死，不做网格搜索、不挑最优；env 覆盖只用于稳健性复核。
  2. 完全因果：信号在 bar i 收盘后确认，下单在 i+1 及以后；目标从下单 bar 的 close 算起。
  3. 与 LGB 无关、不加列、不融合——纯统计检验"写死的规则能不能押对方向"。
  4. placebo 只随机化入场时点（保持方向比例与数量），检验信号是否有真实方向信息。
  5. 报 MDE：n_eff = n_trades / h，修正标签重叠。

规则（默认参数）
----------------
FVG：low[i]>high[i-2]（看涨，1~50bps）；120 根内首次回踩 gap 上沿 → 做多。
OB ：强阳线（实体>1×ATR14）前 5 根内最后一根阴线实体 = 需求区；回踩上沿 → 做多。
Sweep：20 根前高被突破但收回其下 → 做空，下一根入场；前低镜像做多。

env
---
  YEARS=2020,...,2026  SYMBOL=BTCUSDT  HORIZONS=3,5,10,15,30,60
  ODDS=0.8  FVG_WAIT=120  OUT=...  写入结构化结果
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import polars as pl
import requests

REPO = "bailiang0369/bn_hisdata"
TAG = "vision-base"
BASE_URL = f"https://github.com/{REPO}/releases/download/{TAG}"
ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "vision"

SYMBOL = os.environ.get("SYMBOL", "BTCUSDT")
YEARS = [int(x) for x in os.environ.get("YEARS", "2020,2021,2022,2023,2024,2025,2026").split(",")]
HORIZONS = [int(x) for x in os.environ.get("HORIZONS", "3,5,10,15,30,60").split(",")]
ODDS = float(os.environ.get("ODDS", "0.8"))          # 事件合约赔率：押 1 赚 ODDS/亏 1
FVG_WAIT = int(os.environ.get("FVG_WAIT", "120"))
FVG_GAP = (1.0, 50.0)
OB_ATR_MULT, OB_LOOKBACK, SWING_WIN, STOP_BUF = 1.0, 5, 20, 0.1
STOP_MIN_BPS = 10.0                                    # 仅信号元组占位；二元评估不使用 stop
BE_WIN = 1.0 / (1.0 + ODDS) * 100                     # 盈亏平衡胜率 55.56%
OUT = os.environ.get("OUT", "")
T = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T:6.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------- 数据
def fetch_year(year: int) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    dst = CACHE / f"{SYMBOL}_1m_klines_{year}.parquet"
    if dst.exists() and dst.stat().st_size > 1e6:
        return dst
    url = f"{BASE_URL}/{SYMBOL}_1m_klines_{year}.parquet"
    part = dst.with_suffix(".part")
    for attempt in range(1, 6):
        try:
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(part, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        if chunk:
                            f.write(chunk)
            part.replace(dst)
            log(f"downloaded {dst.name} {dst.stat().st_size / 1e6:.1f}MB")
            return dst
        except Exception as e:  # noqa: BLE001
            log(f"[dl] {year} attempt {attempt}: {type(e).__name__} {str(e)[:80]}")
            time.sleep(2 ** attempt)
    raise SystemExit(f"download failed: {url}")


def load_data():
    frames = [pl.read_parquet(fetch_year(y),
              columns=["close_time", "open", "high", "low", "close"]) for y in YEARS]
    df = pl.concat(frames).sort("close_time").unique(subset=["close_time"], keep="last")
    # gap 标记：回看/前看窗口跨缺口则跳过该信号（数据只有极少数整分钟缺失）
    gap = (pl.col("close_time").diff() != 60_000).fill_null(True).alias("_gap")
    df = df.with_columns(gap)
    df = df.with_columns(
        pl.col("_gap").cum_sum().alias("_b"),
        pl.col("_gap").reverse().cum_sum().reverse().alias("_f"),
    )
    n_miss = int(df["_gap"].sum())
    df = df.drop("_gap")
    log(f"数据 {SYMBOL} {df.height:,} 行（{YEARS[0]}~{YEARS[-1]}，缺分钟 {n_miss:,}）")
    b = df["_b"].to_numpy()
    f = df["_f"].to_numpy()
    return (df["close_time"].to_numpy(), df["open"].to_numpy(),
            df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy(), b, f)


# ---------------------------------------------------------------- 工具
def atr14(h, l, c):
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum.reduce([h - l, np.abs(h - pc), np.abs(l - pc)])
    out = np.full(len(c), np.nan)
    s = pl.Series("tr", tr).rolling_mean(14, min_samples=14).to_numpy()
    out[13:] = s[13:]
    return out


def stop_dist(zone_range: float, px: float) -> float:
    return max(STOP_BUF * zone_range, STOP_MIN_BPS / 1e4 * px)


def crossing_fill(zt: float, zb: float, start: int, wait: int, side: int,
                  o, hi, lo, fwd):
    """side=+1: zt=入场沿, zb=失效沿; side=-1: zt=失效沿, zb=入场沿。"""
    end = min(start + wait, len(o))
    seg = fwd[start]
    for t in range(start, end):
        if fwd[t] != seg:
            break
        if side > 0:
            if lo[t] > zt:
                continue
            if o[t] <= zb:
                return None
            px = min(o[t], zt)
        else:
            if hi[t] < zb:
                continue
            if o[t] >= zt:
                return None
            px = max(o[t], zb)
        return t, px
    return None


# ---------------------------------------------------------------- 信号
def fvg_signals(o, h, l, c, b, f):
    n = len(c)
    ZT, ZB = np.full(n, np.nan), np.full(n, np.nan)
    for i in range(2, n):
        if b[i] != b[i - 2]:
            continue
        g = l[i] - h[i - 2]
        if g > 0 and FVG_GAP[0] <= g / c[i] * 1e4 <= FVG_GAP[1]:
            ZT[i], ZB[i] = l[i], h[i - 2]              # 多：入场上沿 / 失效下沿
        else:
            g = l[i - 2] - h[i]
            if g > 0 and FVG_GAP[0] <= g / c[i] * 1e4 <= FVG_GAP[1]:
                ZT[i], ZB[i] = l[i - 2], h[i]          # 空：失效上沿 / 入场下沿
    out = []
    for i in range(2, n - 1):
        if not np.isfinite(ZT[i]):
            continue
        side = 1 if l[i] > h[i - 2] else -1
        res = crossing_fill(ZT[i], ZB[i], i + 1, FVG_WAIT, side, o, h, l, f)
        if res is None:
            continue
        t, px = res
        gap = abs(ZT[i] - ZB[i])
        sd = stop_dist(gap, px)
        stop = ZB[i] - sd if side > 0 else ZT[i] + sd
        out.append((t, px, stop, side, "fvg"))
    return out


def ob_signals(o, h, l, c, a, b, f):
    n = len(c)
    out = []
    for i in range(max(OB_LOOKBACK, 14) + 1, n - 1):
        if not np.isfinite(a[i]) or a[i] <= 0 or b[i] != b[i - 1]:
            continue
        up = (c[i] - o[i]) > OB_ATR_MULT * a[i]
        dn = (o[i] - c[i]) > OB_ATR_MULT * a[i]
        if not (up or dn):
            continue
        side = 1 if up else -1
        j = None
        for k in range(i - 1, i - 1 - OB_LOOKBACK, -1):
            if b[k] != b[k + 1]:
                break
            if (side > 0 and c[k] < o[k]) or (side < 0 and c[k] > o[k]):
                j = k
                break
        if j is None:
            continue
        hi_, lo_ = max(o[j], c[j]), min(o[j], c[j])
        res = crossing_fill(hi_, lo_, i + 1, FVG_WAIT, side, o, h, l, f)
        if res is None:
            continue
        t, px = res
        sd = stop_dist(hi_ - lo_, px)
        stop = lo_ - sd if side > 0 else hi_ + sd
        out.append((t, px, stop, side, "ob"))
    return out


def sweep_signals(o, h, l, c, b):
    n = len(c)
    out = []
    for i in range(SWING_WIN + 1, n - 1):
        if b[i] != b[i - SWING_WIN]:
            continue
        ph = h[i - SWING_WIN:i].max()
        pl_ = l[i - SWING_WIN:i].min()
        sd = stop_dist(h[i] - l[i], o[i])
        if h[i] > ph and c[i] < ph:
            out.append((i + 1, o[i + 1], ph + sd, -1, "sweep"))
        elif l[i] < pl_ and c[i] > pl_:
            out.append((i + 1, o[i + 1], pl_ - sd, +1, "sweep"))
    return out


# ---------------------------------------------------------------- 评估（二元事件合约）
def binary_wins(sig, c, b, horizon: int):
    """与 two_legs.add_target 同口径：win = (close[t+h] > close[t]) 按 side 翻转。

    信号里 t 是下单 bar（FVG/OB 回踩 bar、sweep 的 i+1），用它的 close 作起点，
    严格只用 t 及之前信息确认信号，t+h 的价格是结算。
    """
    wins = []
    for t, px, stop, side, tag in sig:
        u = t + horizon
        if u < len(c) and b[u] == b[t]:
            wins.append(bool((c[u] > c[t]) if side > 0 else (c[u] < c[t])))
    return np.asarray(wins, dtype=bool)


def stat_binary(wins, horizon: int, tpd: float | None = None) -> dict:
    """胜率 + 每单 EV（赔率 ODDS）+ MDE；n_eff 按 h 重叠修正。"""
    n = len(wins)
    if n == 0:
        return {"n": 0}
    p = float(wins.mean())
    n_eff = max(2.0, n / max(1.0, horizon))          # 标签重叠 h 根
    se = float(np.sqrt(p * (1 - p) / n_eff))
    ev = ODDS * p - (1 - p)                           # 押 1：赢 +ODDS，输 −1
    d = {"n": n, "win": round(p * 100, 2),
         "be_win": round(BE_WIN, 2), "edge_pp": round((p * 100 - BE_WIN), 2),
         "ev_per_bet": round(ev, 4), "se_win_pp": round(se * 100, 2),
         "mde_pp": round(1.96 * se * 100, 2),
         "t_vs_be": round((p - BE_WIN / 100) / (se + 1e-12), 2)}
    if tpd is not None:
        d["tpd"] = round(tpd, 1)
    return d


def yearly_binary(sig, ct, c, b, horizon: int):
    yr = pl.Series(ct).cast(pl.Datetime("ms")).dt.year().to_numpy()
    by = {}
    for s in sig:
        by.setdefault(int(yr[s[0]]), []).append(s)
    return {str(y): stat_binary(binary_wins(sg, c, b, horizon), horizon)
            for y, sg in sorted(by.items())}


def placebo_binary(sig, n_bars, c, b, horizon: int, seed=42):
    """随机入场时点、保持方向比例与数量 -> 检验是否有真实方向信息。"""
    rng = np.random.default_rng(seed)
    sides = np.array([s[3] for s in sig])
    idx = rng.choice(np.arange(200, n_bars - horizon - 1), size=len(sig), replace=True)
    fake = [(int(t), 0.0, 0.0, int(sd), "placebo") for t, sd in zip(idx, sides)]
    return stat_binary(binary_wins(fake, c, b, horizon), horizon)


# ---------------------------------------------------------------- main
def main():
    ct, o, h, l, c, b, f = load_data()
    n = len(c)
    a = atr14(h, l, c)
    all_sig = {}
    for name, fn in (("fvg", lambda: fvg_signals(o, h, l, c, b, f)),
                     ("ob", lambda: ob_signals(o, h, l, c, a, b, f)),
                     ("sweep", lambda: sweep_signals(o, h, l, c, b))):
        sg = fn()
        all_sig[name] = sg
        log(f"{name}: {len(sg):,} 笔信号")

    result = {"symbol": SYMBOL, "years": YEARS, "odds": ODDS, "be_win": round(BE_WIN, 2),
              "params": {"fvg_wait": FVG_WAIT, "fvg_gap_bps": list(FVG_GAP),
                         "ob_atr_mult": OB_ATR_MULT, "swing_win": SWING_WIN}}

    for name, sg in all_sig.items():
        if not sg:
            continue
        log(f"\n===== {name.upper()} =====")
        res = {"n_signals": len(sg), "by_horizon": {}}
        for k in HORIZONS:
            wins = binary_wins(sg, c, b, k)
            tpd = len(wins) / (n / 1440.0)            # 平均每天信号数（全期）
            s = stat_binary(wins, k, tpd)
            s["yearly"] = yearly_binary(sg, ct, c, b, k)
            s["placebo"] = placebo_binary(sg, n, c, b, k)
            res["by_horizon"][str(k)] = s
            ys = s["yearly"]
            log(f"h={k:2d}m  n={s['n']:7,}  tpd={tpd:7.1f}  win={s['win']:5.2f}%  "
                f"BE={BE_WIN:.2f}%  edge={s['edge_pp']:+5.2f}pp  ev/bet={s['ev_per_bet']:+.4f}  "
                f"t_vs_BE={s['t_vs_be']:+6.2f}  MDE±{s['mde_pp']:.2f}pp")
            log("    分年 win: " + " ".join(f"{y}:{v['win']:.1f}" for y, v in ys.items()))
            log(f"    placebo win={s['placebo']['win']:.2f}%  "
                f"(真实−placebo={s['win'] - s['placebo']['win']:+.2f}pp)")
        result[name] = res

    if OUT:
        Path(OUT).parent.mkdir(parents=True, exist_ok=True)
        Path(OUT).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        log(f"结果写入 {OUT}")
    log("DONE")


if __name__ == "__main__":
    main()
