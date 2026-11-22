#!/usr/bin/env python3
"""波动分层阈值（vol-stratified threshold）诊断 + 可复用核心。

动机（审查结论，非照搬旧总结）
--------------------------------
现役 h3/h10 都用**单一绝对阈值** `|p-0.5| >= thr`。但 corr(|p-0.5|, log vol) = -0.345：
波动越低，模型输出的置信度越大。绝对阈值因此把 85~92% 的成交赶到周末低波时段——
这不是"工作日没信号"，而是"固定阈值在高波的工作日取不到极端分数"。

做法：按**逐 bar 波动**分层（不是 s_t——它是 7 天滚动中位数，按构造抹平星期效应，
详录 §23 已明确；真实逐 bar 波动是 |1m 收益|），每层在自己内部定 |p-0.5| 阈值：
  equal_rate ：每层触发率都 = tpd/1440（总单量不变，只重新分布到各波動层）
  equal_count：每层出单数相同（主动把单量匀到 bar 更少的高波层=工作日，单量最大化）

因果纪律
--------
- 分层边界 edges、各层阈值 thr_l 只用校准段（val 前半）估计，val 后半 / test 只评估。
- 逐 bar 波动 |ret1| 在 bar t 收盘即已知，用于 t 时刻决策、t+h 结算，无前视。

用法
----
本地（秒级，注入本地 BTC K 线的真实逐 bar 波动）：
    python -m morph.volstrat
    VOLWIN=60 NLAYERS=4 TPDS=8,12,15 python -m morph.volstrat
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import polars as pl

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

PRED = os.environ.get("PRED", ".codebuddy/artifacts/h3gate/h3_gate_pred.parquet")
KLINE_GLOB = os.environ.get("KLINE_GLOB", "data/vision/BTCUSDT_1m_klines_*.parquet")
VOLWIN = int(os.environ.get("VOLWIN", "60"))       # 波动窗口（分钟）；60 = §23 的 60m 波幅
NLAYERS = int(os.environ.get("NLAYERS", "3"))
TPDS = [int(x) for x in os.environ.get("TPDS", "10,15,20").split(",")]
BARS_PER_DAY = int(os.environ.get("BARS_PER_DAY", "1440"))
PAYOUT = float(os.environ.get("PAYOUT", "0.8"))


def build_vol_map() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """从本地 BTC 1m K 线算逐 bar 波动，返回 (close_time, vol_inst, vol_win)。"""
    import glob
    fs = sorted(glob.glob(KLINE_GLOB))
    if not fs:
        raise SystemExit(f"没有匹配的 K 线：{KLINE_GLOB}")
    kl = pl.concat([pl.read_parquet(f).select("close_time", "close") for f in fs],
                   how="vertical_relaxed").unique(subset=["close_time"], keep="last").sort("close_time")
    kc = kl["close"].to_numpy().astype(np.float64)
    ct = kl["close_time"].to_numpy()
    r1 = np.r_[0.0, np.abs(np.diff(np.log(kc)) * 1e4)]       # 逐 bar |ret1| (bp)
    # 因果滚动均值：当前 bar 及过去 VOLWIN-1 根，只用已发生数据
    cs = np.concatenate([[0.0], np.cumsum(r1)])
    vol_win = (cs[VOLWIN:] - cs[:-VOLWIN]) / VOLWIN
    vol_win = np.r_[np.full(VOLWIN - 1, np.nan), vol_win]
    return ct, r1, vol_win


def align_vol(ct_target: np.ndarray, ct_src: np.ndarray, vol: np.ndarray) -> np.ndarray:
    """按 close_time 把波动对齐到目标 bar（两者都是 1m bar，一一对应）。"""
    idx = np.searchsorted(ct_src, ct_target)
    out = np.full(len(ct_target), np.nan)
    ok = (idx < len(ct_src))
    out[ok] = vol[idx[ok]]
    ok2 = ok & (ct_src[np.minimum(idx, len(ct_src) - 1)] == ct_target)
    out[~ok2] = np.nan
    return out


def weekday(ct_ms: np.ndarray) -> np.ndarray:
    """1=Mon .. 7=Sun，与 h3_weekday_model 的 dt.weekday() 一致。"""
    return pl.from_epoch(pl.Series(ct_ms), time_unit="ms").dt.weekday().to_numpy()


def fit_layers(a: np.ndarray, vol: np.ndarray, tpd: int, mode: str,
               n_layers: int, n_cal_days: int, edges: np.ndarray) -> np.ndarray:
    """a=|score-.5|, vol=波动；只用校准段。返回每层阈值 thr_l。"""
    lay = np.searchsorted(edges, vol, side="right")
    thr = np.full(n_layers, np.inf)
    for k in range(n_layers):
        mk = (lay == k) & np.isfinite(a) & np.isfinite(vol)
        if mk.sum() < 50:
            continue
        if mode == "equal_rate":
            q = tpd / BARS_PER_DAY
        else:  # equal_count
            q = (tpd * n_cal_days / n_layers) / max(mk.sum(), 1)
            q = min(max(q, 1e-6), 0.5)
        thr[k] = float(np.quantile(a[mk], 1 - q))
    return thr


def apply_layers(a: np.ndarray, vol: np.ndarray, edges: np.ndarray, thr: np.ndarray) -> np.ndarray:
    lay = np.clip(np.searchsorted(edges, vol, side="right"), 0, len(thr) - 1)
    return (a >= thr[lay]) & np.isfinite(vol)


def report(tag: str, ct: np.ndarray, y: np.ndarray, p: np.ndarray,
           hit: np.ndarray, wd: np.ndarray) -> dict:
    if hit.sum() == 0:
        return {}
    pred = (p[hit] > 0.5).astype(int)
    w = (pred == y[hit].astype(int))
    days = len(np.unique(ct // 86_400_000))
    wkd = wd[hit] <= 5
    return dict(tag=tag, n=int(hit.sum()), per_day=hit.sum() / days,
                win=float(w.mean()),
                wd_n=int(wkd.sum()), wd_per_day=wkd.sum() / days,
                wd_win=float(w[wkd].mean()) if wkd.sum() >= 20 else float("nan"),
                we_n=int((~wkd).sum()),
                we_win=float(w[~wkd].mean()) if (~wkd).sum() >= 20 else float("nan"),
                we_share=float((~wkd).mean()))


def fmt(r: dict) -> str:
    if not r:
        return "无成交"
    return (f"日均={r['per_day']:5.1f} 胜率={r['win']*100:5.2f}% "
            f"| 工作日 {r['wd_per_day']:5.1f}/天 {r['wd_win']*100:5.2f}% "
            f"| 周末 {r['we_win']*100:5.2f}% 周末占比={r['we_share']*100:4.1f}%")


def main() -> None:
    d = pl.read_parquet(PRED).filter(pl.col("split") == "time").sort("close_time")
    va, te = (d.filter(pl.col("segment") == s) for s in ("val", "test"))
    ud = np.unique(va["day"].to_numpy())
    cal_mask = np.isin(va["day"].to_numpy(), ud[:len(ud) // 2])
    n_cal_days = len(ud) // 2

    # 注入真实逐 bar 波动
    ct_src, r1, vol_win = build_vol_map()
    for nm, seg in (("val", va), ("test", te)):
        v = align_vol(seg["close_time"].to_numpy(), ct_src, vol_win)
        cov = np.isfinite(v).mean()
        print(f"[{nm}] 波动覆盖 {cov*100:.1f}%")
    ctv = va["close_time"].to_numpy()
    vol_all = align_vol(ctv, ct_src, vol_win)
    vol_cal = vol_all[cal_mask]
    ok = np.isfinite(vol_cal)
    vc = va["score"].to_numpy()[cal_mask][ok]
    volc = vol_cal[ok]
    corr = np.corrcoef(np.abs(vc - 0.5), np.log(volc + 1e-9))[0, 1]
    edges = np.quantile(volc, np.linspace(0, 1, NLAYERS + 1)[1:-1])
    print(f"corr(|p-0.5|, log vol{VOLWIN}) = {corr:+.3f} | {NLAYERS} 层边界 vol={np.round(edges,2)}")
    print(f"校准 {len(vc):,} bar / {n_cal_days} 天 | 赔率 {PAYOUT} 平衡线 {1/(1+PAYOUT)*100:.2f}%")
    print("目标：工作日 10+/天 且 >=60%\n")

    for tpd in TPDS:
        print(f"===== tpd={tpd} =====")
        gthr = float(np.quantile(np.abs(vc - 0.5), 1 - tpd / BARS_PER_DAY))
        thr_r = fit_layers(np.abs(vc - 0.5), volc, tpd, "equal_rate", NLAYERS, n_cal_days, edges)
        thr_c = fit_layers(np.abs(vc - 0.5), volc, tpd, "equal_count", NLAYERS, n_cal_days, edges)
        print(f"  全局thr={gthr:.5f} | equal_rate层thr={np.round(thr_r,5)} | equal_count层thr={np.round(thr_c,5)}")
        for seg_name, seg in (("VAL后半", va), ("TEST", te)):
            ct = seg["close_time"].to_numpy()
            y = seg["target"].to_numpy()
            p = seg["score"].to_numpy()
            vol = align_vol(ct, ct_src, vol_win)
            wd = weekday(ct)
            if seg_name == "VAL后半":
                keep = ~np.isin(seg["day"].to_numpy(), ud[:len(ud)//2])
                ct, y, p, vol, wd = ct[keep], y[keep], p[keep], vol[keep], wd[keep]
            a = np.abs(p - 0.5)
            rs = [
                report(f"{seg_name} 全局", ct, y, p, (a >= gthr) & np.isfinite(vol), wd),
                report(f"{seg_name} 分层equal_rate", ct, y, p, apply_layers(a, vol, edges, thr_r), wd),
                report(f"{seg_name} 分层equal_count", ct, y, p, apply_layers(a, vol, edges, thr_c), wd),
            ]
            for r in rs:
                print(f"  {r['tag']:<24} {fmt(r)}")
        print()


if __name__ == "__main__":
    main()
