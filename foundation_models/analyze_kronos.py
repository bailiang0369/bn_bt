#!/usr/bin/env python3
"""诊断 Kronos 探针的 AUC vs 胜率：分数校准 + top-K 选择性。"""
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

p = Path(__file__).resolve().parent.parent / "artifacts" / "kronos" / "kronos_probe_long.parquet"
d = pl.read_parquet(p).to_pandas()
print(f"rows={len(d)} cols={list(d.columns)}")
for h in (3, 10, 30, 60, 120):
    s = d[f"pred_ret_{h}"].to_numpy()
    y = d[f"true_up_{h}"].to_numpy().astype(int)
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    auc = roc_auc_score(y, s)
    win = float(((s > 0) == (y == 1)).mean())
    print(f"\nh={h}: n={len(s)} AUC={auc:.4f} win@0={win*100:.2f}% "
          f"base_up={y.mean()*100:.2f}% pred_up={(s>0).mean()*100:.2f}%")
    print(f"  pred_ret: mean={s.mean():+.6f} median={np.median(s):+.6f} "
          f"std={s.std():.6f} p5={np.percentile(s,5):+.6f} p95={np.percentile(s,95):+.6f}")
    # 按 |pred_ret| 取 top-K（=项目 topK 口径的近似：只看高置信度信号）
    order = np.argsort(-np.abs(s))
    for frac in (0.5, 0.25, 0.1, 0.05):
        idx = order[:max(10, int(len(s) * frac))]
        w = float(((s[idx] > 0) == (y[idx] == 1)).mean())
        print(f"  top{int(frac*100):2d}% n={len(idx):3d} win={w*100:.2f}% "
              f"up_rate={y[idx].mean()*100:.2f}%")
