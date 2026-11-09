#!/usr/bin/env python3
"""E*：ETHUSDT 用现役 h=3 基线「同逻辑」评估（vol8 185 列纯盘口）。

与 BTC 现役基线逐项对齐（差别只有标的）：
  - 数据：`ob_ETHUSDT.parquet`（ethbuild kernel 产出的 vol8 185 列纯盘口）
          + `vision-base` 的 ETHUSDT 1m K 线 close（只用于构造标签，不进特征）
  - 目标 `close[t+3] > close[t]`，严格时序 64/16/20
  - 5 seed 概率平均、2000 轮 + val early stopping（`morph.two_legs.fit`）
  - `time` / `row` 双切分同一次运行
  - 四档胜率（@日均15单 / @1% / @5% / @25%）+ AUC，逐行分数落盘供离线复核
  - 不动模型、不改特征、不加闸门

数据经 `kernel_sources` 挂载 ethbuild 的 output，不依赖 Kaggle Dataset 或 Release 上传。

env: SPLITS / SEEDS / ROUNDS / TARGET_TPD / KLINE_FROM / HORIZON / EXP / OUT / ETH_OB
"""
from __future__ import annotations

import glob
import os
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

try:
    from morph import h3_gate_eval as ge
    from morph import ob_kline as ok
    from morph.h3_gate import split_wide, train_predict
    from morph.two_legs import TARGET_TPD, TRAIN_FRAC, VAL_FRAC, add_target
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from morph import h3_gate_eval as ge
    from morph import ob_kline as ok
    from morph.h3_gate import split_wide, train_predict
    from morph.two_legs import TARGET_TPD, TRAIN_FRAC, VAL_FRAC, add_target

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

SPLITS = [s.strip().lower() for s in os.environ.get("SPLITS", "time,row").split(",") if s.strip()]
HORIZON = int(os.environ.get("HORIZON", "3"))
KLINE_FROM = int(os.environ.get("KLINE_FROM", "0")) or 2025
EXP = os.environ.get("EXP", "E?")
OUT = os.environ.get("OUT", "/kaggle/working/h3_eth_pred.parquet")
ETH_OB = os.environ.get("ETH_OB", "")
T = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T:7.1f}s] {msg}", flush=True)


def load_ob() -> pl.DataFrame:
    p = ETH_OB
    if not p:
        hits = sorted(glob.glob("/kaggle/input/**/ob_ETHUSDT.parquet", recursive=True))
        if not hits:
            raise SystemExit("找不到 ob_ETHUSDT.parquet（ethbuild kernel output 未挂载？）")
        p = hits[0]
    log(f"[data] {p}")
    return (pl.read_parquet(p).unique(subset=["close_time"], keep="last")
            .sort("close_time"))


def main() -> None:
    ob = load_ob()
    obcols = [c for c in ob.columns if c != "close_time"]
    n = ob.height
    tr1, va1 = int(n * TRAIN_FRAC), int(n * (TRAIN_FRAC + VAL_FRAC))
    t_tr_end = int(ob["close_time"][tr1 - 1])
    t_va_end = int(ob["close_time"][va1 - 1])
    log(f"ETH 订单簿 {n:,} 行 x {len(obcols)} 特征列 | T_end={ok._iso(t_tr_end)}"
        f" | val 末={ok._iso(t_va_end)} | splits={SPLITS}")

    kfeat, _ = ok.load_kline_features(KLINE_FROM, symbol="ETHUSDT", keep_raw=False)
    wide = (ob.join(kfeat.select("close_time", "close"), on="close_time", how="left")
              .sort("close_time"))
    wide = add_target(wide, HORIZON).filter(
        pl.col("target").is_not_null() & pl.col("close").is_not_null())
    log(f"可用 {wide.height:,} 行（close 对齐后）")

    frames = []
    for how in SPLITS:
        tr, va, te = split_wide(wide, how, t_tr_end, t_va_end)
        if min(tr.height, va.height, te.height) < 1000:
            log(f"[{how}] 样本不足，跳过")
            continue
        tag = f"{EXP}-{how}"
        log(f"[{tag}] train={tr.height:,} val={va.height:,} test={te.height:,}")
        p_va, p_te = train_predict(tr, va, te, obcols, tag)

        y_va = va["target"].to_numpy().astype(np.int8)
        y_te = te["target"].to_numpy().astype(np.int8)
        s_va, s_te = p_va - 0.5, p_te - 0.5
        d_va = ge.day_id(va["close_time"].to_numpy())
        d_te = ge.day_id(te["close_time"].to_numpy())
        auc_va = roc_auc_score(y_va, p_va)
        auc_te = roc_auc_score(y_te, p_te)
        log(f"[{tag}] AUC val={auc_va:.4f} test={auc_te:.4f}")
        ge.report(f"{tag} val", s_va, y_va, d_va, TARGET_TPD)
        m = ge.report(f"{tag} test", s_te, y_te, d_te, TARGET_TPD)
        log(f"[{tag}] SUMMARY split={how} cols={len(obcols)} AUC={auc_te:.4f}"
            f" @15={m['日均15单'] * 100:.2f}% @5%={m['5%'] * 100:.2f}%"
            f" @25%={m['25%'] * 100:.2f}%")

        for seg, df, p in (("val", va, p_va), ("test", te, p_te)):
            frames.append(pl.DataFrame({
                "split": how, "segment": seg,
                "close_time": df["close_time"], "target": df["target"],
                "score": p, "day": ge.day_id(df["close_time"].to_numpy()),
            }))

    if OUT and frames:
        pl.concat(frames).write_parquet(OUT)
        log(f"预测已落盘: {OUT}（{sum(f.height for f in frames):,} 行）")
    log("DONE")


if __name__ == "__main__":
    main()
