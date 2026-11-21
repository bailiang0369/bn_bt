#!/usr/bin/env python3
"""新阶段：h3 工作日独立阈值的因果 replay（独立文件，不改动旧阶段代码）。

阈值只用校准段（val 前半）定：
  global：全部 bar 用 q(|p-.5|, 1 - TPD/1440)
  mixed ：工作日 bar 用 q(校准段工作日 |p-.5|, 1 - TPD_WD/1440)，周末沿用 global

其余口径与现役 h3 一致：逐 bar 独立触发、方向 p>0.5、3 分钟结算、3 个并发槽位。
val 后半与 test 都只评估，不参与定阈值。

env:
  PRED=/kaggle/working/live_sim/h3_pred.parquet  逐行分数（不存在则先跑 h3_gate 生成）
  TPD=15 TPD_WD=15 PAYOUT=0.80 SLOTS=3 OFFSET_MIN=0
  BOOT=100 PLACEBO=50 SPLITS=time SEEDS=42,43,44,45,46
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

WORK = Path(os.environ.get("LIVE_SIM_OUT", "/kaggle/working/live_sim"))
PRED = Path(os.environ.get("PRED", str(WORK / "h3_pred.parquet")))
TPD = int(os.environ.get("TPD", "15"))
TPD_WD = int(os.environ.get("TPD_WD", "15"))
PAYOUT = float(os.environ.get("PAYOUT", "0.80"))
SLOTS = int(os.environ.get("SLOTS", "3"))
OFFSET_MIN = int(os.environ.get("OFFSET_MIN", "0"))
HORIZON = 3
MS_MIN = 60_000
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def weekday(ct_ms: np.ndarray) -> np.ndarray:
    return ((ct_ms // 86_400_000 + 3) % 7 + 1).astype(np.int8)


def ensure_pred() -> None:
    if PRED.exists():
        log(f"复用已有逐行分数 {PRED}")
        return
    os.environ["SPLITS"] = "time"
    os.environ["OUT"] = str(PRED)
    os.environ.setdefault("BOOT", "100")
    os.environ.setdefault("PLACEBO", "50")
    os.environ.setdefault("KLINE_FROM", "2025")
    log("本地无逐行分数，先跑 h3_gate 生成（Kaggle）")
    from morph import h3_gate
    h3_gate.main()


def quantile_thr(p: np.ndarray, mask: np.ndarray, tpd: int) -> float:
    return float(np.quantile(np.abs(p[mask] - .5), 1 - tpd / 1440))


def simulate(ct: np.ndarray, p: np.ndarray, y: np.ndarray, thr_global: float,
             thr_wd: float, segment: str, mode: str) -> dict:
    wd = weekday(ct)
    s = p - .5
    thr_use = np.where(wd <= 5, thr_wd, thr_global) if mode == "mixed" else thr_global
    hit = np.abs(s) >= thr_use
    opens: list[int] = []
    events, wins = [], []
    rejected = 0
    for i in np.flatnonzero(hit):
        dec = int(ct[i]) + OFFSET_MIN * MS_MIN
        opens = [x for x in opens if x > dec]
        if len(opens) >= SLOTS:
            rejected += 1
            accepted, settle = False, dec + HORIZON * MS_MIN
        else:
            accepted = True
            settle = dec + HORIZON * MS_MIN
            opens.append(settle)
            d = 1 if s[i] > 0 else 0
            wins.append(int(d == y[i]))
        events.append(dict(close_time=int(ct[i]), weekday=int(wd[i]), segment=segment,
                           mode=mode, p=float(p[i]), target=int(y[i]),
                           accepted=accepted, reason="" if accepted else "no_slot"))
    w = np.array(wins, dtype=np.int8)
    win = float(w.mean()) if len(w) else float("nan")
    days = (pl.Series([e["close_time"] for e in events if e["accepted"]], dtype=pl.Int64)
            .cast(pl.Datetime("ms")).dt.strftime("%Y-%m-%d").to_numpy()) if len(w) else np.array([])
    cnt = np.unique(days, return_counts=True)[1] if len(days) else np.array([])
    wk = np.array([e["weekday"] for e in events if e["accepted"]], dtype=np.int8)
    by_wd = {}
    for k in range(1, 8):
        mk = wk == k
        by_wd[k] = dict(n=int(mk.sum()), win=float(w[mk].mean()) if mk.any() else None)
    months = (pl.Series([e["close_time"] for e in events if e["accepted"]], dtype=pl.Int64)
              .cast(pl.Datetime("ms")).dt.strftime("%Y-%m").to_numpy()) if len(w) else np.array([])
    mv = []
    for m in np.unique(months):
        mk = months == m
        if mk.sum() >= 20:
            mv.append(float(w[mk].mean()))
    return dict(events=events, wins=wins, n=len(w), win=win,
                ev=PAYOUT * win - (1 - win) if len(w) else float("nan"),
                day_mean=float(cnt.mean()) if len(cnt) else 0.,
                day_std=float(cnt.std()) if len(cnt) else 0.,
                day_min=int(cnt.min()) if len(cnt) else 0,
                day_max=int(cnt.max()) if len(cnt) else 0,
                weekend_share=float((wk >= 6).mean()) if len(wk) else None,
                month_min=min(mv) if mv else None, by_wd=by_wd,
                raw_n=int(hit.sum()), rejected=rejected)


def main() -> None:
    os.makedirs(WORK, exist_ok=True)
    ensure_pred()
    d = pl.read_parquet(PRED).filter(pl.col("split") == "time")
    va, te = (d.filter(pl.col("segment") == s) for s in ("val", "test"))
    ud = np.unique(va["day"].to_numpy())
    cal = np.isin(va["day"].to_numpy(), ud[:len(ud) // 2])
    cal2 = ~cal
    log(f"校准段 {len(ud) // 2} 天 / val 后半 {cal2.sum():,} 行 / test {te.height:,} 行")

    p_cal = va["score"].to_numpy()[cal]
    ct_cal = va["close_time"].to_numpy()[cal]
    wd_cal = weekday(ct_cal)
    thr_global = quantile_thr(p_cal, np.ones(len(p_cal), dtype=bool), TPD)
    thr_wd = quantile_thr(p_cal, wd_cal <= 5, TPD_WD)
    log(f"阈值：global(TPD={TPD})={thr_global:.5f} | weekday(TPD_WD={TPD_WD})={thr_wd:.5f}")

    report = dict(config=dict(tpd=TPD, tpd_wd=TPD_WD, payout=PAYOUT, slots=SLOTS,
                              offset_min=OFFSET_MIN, thr_global=thr_global, thr_wd=thr_wd),
                  segments={})
    all_events = []
    for seg_name, df, mask in (("val2", va, cal2), ("test", te, None)):
        ct = df["close_time"].to_numpy()
        p = df["score"].to_numpy()
        y = df["target"].to_numpy().astype(np.int8)
        if mask is not None:
            ct, p, y = ct[mask], p[mask], y[mask]
        report["segments"][seg_name] = {}
        for mode in ("global", "mixed"):
            r = simulate(ct, p, y, thr_global, thr_wd, seg_name, mode)
            report["segments"][seg_name][mode] = r
            all_events.extend(r["events"])
            log(f"[{seg_name}][{mode}] {r['n']}单/{r['win'] * 100:.2f}% "
                f"EV={r['ev']:+.3f} 日单量={r['day_mean']:.1f}±{r['day_std']:.1f} "
                f"周末占比={r['weekend_share'] * 100:.1f}% 月min="
                f"{r['month_min'] * 100:.2f}%" if r["month_min"] else
                f"[{seg_name}][{mode}] {r['n']}单 月样本不足")
            log("     星期 " + " ".join(f"{k}:{r['by_wd'][k]['n']}/"
                                      f"{r['by_wd'][k]['win']}" for k in range(1, 8)))
    (WORK / "h3_weekday_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    pl.DataFrame(all_events).write_parquet(WORK / "h3_weekday_events.parquet")
    print("\n" + "=" * 72)
    for seg_name in ("val2", "test"):
        for mode in ("global", "mixed"):
            r = report["segments"][seg_name][mode]
            print(f"[{seg_name}][{mode}] n={r['n']} win={r['win'] * 100:.2f}% "
                  f"EV={r['ev']:+.3f} 周末={r['weekend_share'] * 100:.1f}% "
                  f"最差月={r['month_min'] * 100:.2f}%")
    print(f"产物：{WORK}/h3_weekday_report.json, h3_weekday_events.parquet")
    print("=" * 72)


if __name__ == "__main__":
    main()
