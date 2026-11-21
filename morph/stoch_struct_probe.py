#!/usr/bin/env python3
"""Stoch 结构点形态特征：诊断 kernel（不训练模型）。

目的：先回答"这些特征有没有信号、是否只是现役 49 列的换皮"，再决定要不要花几小时对撞。
产出：
  1. 因果自检（struct_leak_check）
  2. 各特征在 test 月的单特征 AUC（含方向反转检测）
  3. 与现役 49 列的最大 |Pearson r|（>0.9 视为换皮，先验降低）
  4. 覆盖率（非零比例）和结构点平均间隔

env: HORIZON K_PERIOD START_MONTH END_MONTH SC_SELFCHECK
"""
from __future__ import annotations

import datetime as dt
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from morph import kline_long as kl  # noqa: E402
from morph.h10_signal import add_flow  # noqa: E402
from morph.stoch_struct import FEAT_NAMES, build_struct, log, struct_leak_check  # noqa: E402
from morph.two_legs import add_target  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

HORIZON = int(os.environ.get("HORIZON", "10"))
K_PERIOD = int(os.environ.get("K_PERIOD", "80"))
START_MONTH = os.environ.get("START_MONTH", "2026-01")
END_MONTH = os.environ.get("END_MONTH", "2026-10")


def main() -> None:
    if os.environ.get("SC_SELFCHECK", "1") == "1":
        if not struct_leak_check(k_period=K_PERIOD):
            raise SystemExit("[sc] 泄露自检失败，拒绝继续")

    raw = kl.load_raw()
    sc, sc_names = build_struct(raw, k_period=K_PERIOD)
    df, base_feats = kl.build(raw)
    df, base_feats = add_flow(df, base_feats)
    log(f"结构特征 {len(sc_names)} 列 | 现役 base {len(base_feats)} 列")

    # 按 close_time 对齐（struct 在原始行上算，标签表砍掉尾部 H 行）
    sc_pl = pl.DataFrame({c: sc[:, i] for i, c in enumerate(sc_names)}).with_columns(
        pl.Series("close_time", raw["close_time"].to_numpy()))
    df = add_target(df, HORIZON).filter(
        pl.col("target").is_not_null() & pl.col("close").is_not_null()).sort("close_time")
    df = df.join(sc_pl, on="close_time", how="left")
    df = df.with_columns(
        pl.col("close_time").cast(pl.Datetime("ms")).dt.strftime("%Y-%m").alias("ym"))

    ct = df["close_time"].cast(pl.Int64).to_numpy()
    ym = df["ym"].to_numpy()
    y = df["target"].to_numpy().astype(int)
    months = [m for m in sorted(set(ym.tolist())) if START_MONTH <= m <= END_MONTH]
    te_idx = np.where(np.isin(ym, months))[0]
    log(f"test 月 {months[0]}~{months[-1]} | test 行 {len(te_idx):,}")

    Xsc = df.select(sc_names).to_numpy().astype(np.float64)
    Xbase = df.select(base_feats).to_numpy().astype(np.float64)
    yte = y[te_idx]

    # ---- 1) 单特征 AUC（test 全月合并；同时报反向 1-auc）----
    print(f"\n{'特征':<16}{'非零%':>8}{'AUC':>9}{'|AUC-0.5|':>11}{'方向':>6}")
    aucs = {}
    for j, nm in enumerate(sc_names):
        x = Xsc[te_idx, j]
        nz = float((x != 0).mean())
        if np.unique(x).size < 10:
            print(f"{nm:<16}{nz*100:>7.1f}%{'--':>9}")
            continue
        a = float(roc_auc_score(yte, x))
        aucs[nm] = a
        direction = "正" if a >= 0.5 else "反"
        print(f"{nm:<16}{nz*100:>7.1f}%{a:>9.4f}{abs(a-0.5):>11.4f}{direction:>6}")

    # ---- 2) 与现役 base 的最大相关（换皮检查）----
    print(f"\n{'特征':<16}{'与base最大|r|':>14}{'最近base列':>18}")
    base_arr = Xbase[te_idx]
    for j, nm in enumerate(sc_names):
        x = Xsc[te_idx, j]
        if np.std(x) < 1e-9:
            continue
        r = np.corrcoef(x, base_arr.T)
        rx = np.nan_to_num(r[0, 1:], nan=0.0)
        k = int(np.argmax(np.abs(rx)))
        print(f"{nm:<16}{abs(rx[k]):>14.3f}{base_feats[k]:>18}")

    # ---- 3) 结构点平均间隔（稀疏性）----
    pt_type = df["sc_ptype"].to_numpy()
    n_changes = int((np.diff(np.nan_to_num(pt_type)) != 0).sum())
    log(f"结构点切换次数（全样本）≈{n_changes:,} | 平均间隔≈{len(pt_type)/max(n_changes,1):.0f} 根")
    log("DONE")


if __name__ == "__main__":
    main()
