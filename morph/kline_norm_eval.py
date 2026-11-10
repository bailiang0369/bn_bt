#!/usr/bin/env python3
"""离线回放：因果滚动分位归一化能否在「总单量不变」下改善稳定性。

不训练、不下载大数据：直接读 Kaggle 挂载的 kline_fuse_hist_scores.parquet
（2022-01~2026-10 全历史 OOS 预测），把 |score| 换成「相对过去 W 天」的经验
分位，再按 @日均15单 全局取 top-K，比较单量分布与三条判据。

动机：全局 top-K 会向「分数整体偏高」的日子集中（实测有单天数仅 39%），
于是「滚动 100 单」退化成同一天的一批高度相关单，破线率被结构性放大。
归一化不减少总单量，只把名额分散到更多天，不触碰覆盖率约束。

env: SCORES_NAME / TPD / ROLL_N / W_LIST
"""
from __future__ import annotations

import glob
import os

import numpy as np
import polars as pl

TPD = int(os.environ.get("TPD", "15"))
ROLL_N = int(os.environ.get("ROLL_N", "100"))
SCORES_NAME = os.environ.get("SCORES_NAME", "kline_fuse_hist_scores.parquet")
W_LIST = [int(x) for x in os.environ.get("W_LIST", "0,7,30,90,365").split(",") if x.strip()]
DAY_MS = 86_400_000


def find_scores() -> str:
    hits = [p for p in glob.glob("/kaggle/input/**/*", recursive=True)
            if p.endswith(SCORES_NAME)]
    if not hits:
        raise SystemExit(f"找不到 {SCORES_NAME}；挂载目录：{glob.glob('/kaggle/input/*')[:5]}")
    return hits[0]


def rolling_u(abs_s: np.ndarray, day: np.ndarray, w: int) -> np.ndarray:
    """因果：第 d 天只用之前最多 w 天的 |s| 经验分布，当天的分数不参与自身分位。"""
    u = np.zeros(len(abs_s), dtype=np.float32)
    hist: list[np.ndarray] = []
    for d in np.unique(day):
        m = day == d
        ai = abs_s[m]
        if hist:
            past = np.sort(np.concatenate(hist[-w:] if w else hist))
            u[m] = np.searchsorted(past, ai, side="right") / len(past)
        hist.append(ai)
    return u


def evaluate(ct: np.ndarray, y: np.ndarray, s: np.ndarray,
             idx: np.ndarray, tag: str, nd: int | None = None) -> None:
    idx = np.sort(idx)
    w = ((s[idx] > 0).astype(int) == y[idx]).astype(np.int8)
    t, d = ct[idx], ct[idx] // DAY_MS
    nd = nd if nd is not None else len(np.unique(ct // DAY_MS))
    _, c = np.unique(d, return_counts=True)
    ym = (t // 1000).astype("datetime64[s]").astype("datetime64[M]")
    mv = np.array([w[ym == m].mean() for m in np.unique(ym)])
    cs = np.concatenate([[0], np.cumsum(w)])
    r = (cs[ROLL_N:] - cs[:-ROLL_N]) / ROLL_N
    streak = cur = 0
    for v in w:
        cur = cur + 1 if v == 0 else 0
        streak = max(streak, cur)
    print(f"{tag:>12s} n={len(w):6,d} 日均={len(w) / nd:5.2f} 总体={w.mean() * 100:6.2f}%"
          f" 月min={mv.min() * 100:5.1f}% 月<50={int((mv < .5).sum()):2d}/{len(mv)}"
          f" 滚{ROLL_N}min={r.min() * 100:5.1f}% 破50={100 * (r < .5).mean():5.1f}%"
          f" 连亏={streak:3d} 有单天={len(c):4d}/{nd} 单日std={c.std():5.1f} max={c.max():4d}",
          flush=True)


def main() -> None:
    path = find_scores()
    print(f"[scores] {path}", flush=True)
    d = pl.read_parquet(path)
    ct = d["ct"].to_numpy()
    p = (d["p_h15"].to_numpy() + d["p_h30"].to_numpy()) / 2
    s, y = p - 0.5, d["t15"].to_numpy().astype(np.int8)
    day, a = ct // DAY_MS, np.abs(s)
    ud = np.unique(day)
    print(f"rows={len(s):,} days={len(ud)} base={y.mean():.4f} W_LIST={W_LIST}", flush=True)

    # 所有 W 共用同一评估窗口，k 与日均完全相同，横向可比（不是靠缩短窗口换指标）
    w0 = min(max(W_LIST), len(ud) - 1)
    valid = day >= ud[w0]
    cand = np.where(valid)[0]
    nd_eval = len(np.unique(day[valid]))
    k = min(TPD * nd_eval, len(cand))
    print(f"[eval window] 从 {np.datetime64(int(ud[w0]), 'D')} 起，{nd_eval} 天，k={k:,}",
          flush=True)
    for w in W_LIST:
        u = a if w == 0 else rolling_u(a, day, w)
        sel = cand[np.argpartition(-u[valid], k - 1)[:k]]
        evaluate(ct, y, s, sel, f"W={w}" if w else "static", nd_eval)
    # 参考：static 全期（与上面共同窗口的 static 区分开）
    k_all = min(TPD * len(ud), len(s))
    evaluate(ct, y, s, np.argpartition(-a, k_all - 1)[:k_all], "static全期")

    print("DONE", flush=True)


if __name__ == "__main__":
    main()
