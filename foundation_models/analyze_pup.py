#!/usr/bin/env python3
"""本地分析 Chronos P(up) parquet：校准曲线 + 阈值下单胜率。"""
from pathlib import Path

import numpy as np
import polars as pl

P = Path(__file__).resolve().parent.parent / "artifacts" / "chronos" / "chronos_pup.parquet"
d = pl.read_parquet(P).to_pandas()
cal, test = d[d.is_test == 0], d[d.is_test == 1]
ODDS = 0.8
BE = 1 / (1 + ODDS) * 100


def auc(y, s):
    y = np.asarray(y).astype(int); s = np.asarray(s)
    order = np.argsort(s); ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1)
    n1 = int(y.sum()); n0 = len(y) - n1
    return float((ranks[y == 1].sum() - n1*(n1+1)/2)/(n1*n0)) if n1 and n0 else float("nan")


print(f"rows={len(d)} cal={len(cal)} test={len(test)} BE={BE:.2f}%")
for h in (3, 10, 60):
    pu, y, med = f"pup_{h}", f"true_up_{h}", f"pup_med_{h}"
    if pu not in d.columns:
        continue
    cv, tv = cal[[pu, y]].dropna(), test[[pu, y, med]].dropna()
    if len(tv) < 100:
        continue
    a = auc(tv[y], tv[pu])
    win_med = float(((tv[med] == 1) == (tv[y] == 1)).mean()) * 100
    print(f"\n=== h={h}  n_test={len(tv)}  AUC(Pup)={a:.4f}  win@median={win_med:.2f}% ===")
    print("  校准  P(up)档 : n / 实际上涨率 / 名义中心")
    for lo_p, hi_p in ((0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)):
        m = (tv[pu] >= lo_p) & (tv[pu] < hi_p)
        if m.sum() >= 20:
            print(f"    [{lo_p:.1f},{min(hi_p,1):.1f}]  n={int(m.sum()):5d}  "
                  f"实际上涨={tv[y][m].mean()*100:5.2f}%  名义≈{(lo_p+min(hi_p,1))/2*100:.0f}%")
    print("  阈值下单 (cal 定分位 -> test):")
    for frac in (0.25, 0.10, 0.05):
        thr = float(np.quantile(cv[pu], 1 - frac))
        sel = tv[tv[pu] >= thr]
        hi = sel[sel[pu] >= 0.5]
        if len(hi) < 20:
            continue
        w = float(hi[y].mean())
        se = np.sqrt(w*(1-w)/len(hi))
        print(f"    top{int(frac*100):2d}% thr={thr:.3f} n_pred_up={len(hi):4d} "
              f"胜率={w*100:.2f}% CI95=[{max(0,w-1.96*se)*100:.1f},{min(1,w+1.96*se)*100:.1f}] "
              f"{'PASS' if w*100>BE else 'FAIL'}")
