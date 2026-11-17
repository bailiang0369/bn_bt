#!/usr/bin/env python3
"""诊断大样本 Kronos 探针：cal/test topK 稳定性、按年分解、阈值敏感性。"""
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

base = Path(__file__).resolve().parent.parent / "artifacts" / "kronos"
for bar in (1, 5):
    p = base / f"kronos_large_b{bar}.parquet"
    if not p.exists():
        continue
    d = pl.read_parquet(p).to_pandas()
    print(f"\n{'='*80}\nBAR={bar}m rows={len(d)}")
    cal, test = d[d.is_test == 0], d[d.is_test == 1]
    print(f"cal={len(cal)} test={len(test)}")
    s, y = "pred_ret_3", "true_up_3"
    cv, tv = cal[[s, y]].dropna(), test[[s, y, "sec"]].dropna()
    print(f"AUC(test)={roc_auc_score(tv[y],tv[s]):.4f}  win@0(test)={((tv[s]>0)==(tv[y]==1)).mean()*100:.2f}%")
    # cal 阈值 -> test 评估（与主脚本一致）
    for frac in (0.25, 0.10, 0.05):
        thr = float(np.quantile(np.abs(cv[s]), 1 - frac))
        sel = tv[np.abs(tv[s]) >= thr]
        if not len(sel):
            continue
        w = float(((sel[s] > 0) == (sel[y] == 1)).mean())
        se = np.sqrt(w * (1 - w) / len(sel))
        yrs = pl.Series(sel["sec"]).str.slice(0, 4).to_numpy()
        yearly = [f"{yy}:{float(((sel[s].to_numpy()[yrs==yy]>0)==(sel[y].to_numpy()[yrs==yy]==1)).mean())*100:.1f}"
                  for yy in sorted(set(yrs)) if (yrs == yy).sum() >= 8]
        print(f"  top{int(frac*100):2d}% n={len(sel):4d} win={w*100:5.2f}% "
              f"CI95=[{max(0,w-1.96*se)*100:.1f},{min(1,w+1.96*se)*100:.1f}] "
              f"BE=55.56% up={sel[y].mean()*100:.1f}% | 分年 {' '.join(yearly)}")
