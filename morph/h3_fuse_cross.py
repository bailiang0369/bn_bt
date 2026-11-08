#!/usr/bin/env python3
"""跨域特征接入**两腿基线**：宽表里 W5/X2 的 +1pp 能否传递到融合口径？

为什么单独一个文件
------------------
`two_legs.py` 是基线（h=3 纯订单簿 65.49%），这里不改它一行：口径（切分点、w 网格、
z-score 用 train 段统计）全部照抄，只是把订单簿腿的特征从「184 列」换成「184 + 跨域 11 列」。

`SPLITS` 为什么存在
------------------
宽表用**行号**切分、两腿用 **close_time 阈值**切分，两者在同一份数据上只差 2 行，但分别在
两次 kernel 运行里跑出来的 `ob` 基线差了 1pp（65.90 vs 66.94）。这不可能是切分造成的，
只可能是跨运行抖动（LightGBM 线程数/histogram 累加顺序随分配核数变化）。而跨域声称的
增益正好是 1pp，因此必须**同进程内**把两种切分并排跑，才能判断符号是否稳定。

env
---
    VARIANTS=ob,ob+cross   订单簿腿特征集（k 列不并入订单簿腿，K 线另有独立的腿 A）
    SPLITS=time,row        切分口径：time = two_legs 阈值；row = h3_wide 行号（可多选并排）
    FUSE=1                 1 = 训 K 线腿并做 z 融合；0 = 只比订单簿腿（诊断用，省 5 次大训练）
    HORIZON=3              只服务 3 分钟
    SEEDS=42,43,44,45,46   两腿都用这组种子做概率平均
    TARGET_TPD=15          评估口径
    EXP=F1                 实验编号（进日志）

判据（同 `h3宽表实验记录.md`）
---------------------------
融合后 test 日均 15 单胜率 ≥ 同口径 `ob` 版 +0.5pp 且 val 同向，才算跨域在融合口径里有效。
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

try:
    from morph import ob_kline as ok
    from morph.h3_wide import _amplitude as amplitude
    from morph.h3_wide import add_cross
    from morph.two_legs import (KLINE_YEARS, TARGET_TPD, TRAIN_FRAC, VAL_FRAC,
                                add_target, day_id, fit, global_topk, matrix, report)
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from morph import ob_kline as ok
    from morph.h3_wide import _amplitude as amplitude
    from morph.h3_wide import add_cross
    from morph.two_legs import (KLINE_YEARS, TARGET_TPD, TRAIN_FRAC, VAL_FRAC,
                                add_target, day_id, fit, global_topk, matrix, report)

VARIANTS = [v.strip().lower() for v in
            os.environ.get("VARIANTS", "ob,ob+cross").split(",") if v.strip()]
SPLITS = [s.strip().lower() for s in
          os.environ.get("SPLITS", "time").split(",") if s.strip()]
FUSE = int(os.environ.get("FUSE", "1"))
HORIZON = int(os.environ.get("HORIZON", "3"))
SEEDS = [int(s) for s in os.environ.get("SEEDS", "42,43,44,45,46").split(",") if s.strip()]
EXP = os.environ.get("EXP", "F?")
W_GRID = (0.0, 0.5, 1.0, 2.0, 3.0, 5.0)
T = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T:7.1f}s] {msg}", flush=True)


def kalign(df: pl.DataFrame, kfeat: pl.DataFrame, kcols: list[str]) -> np.ndarray:
    """把 `df` 的 close_time 对齐到 K 线特征矩阵（同 two_legs.kpred，但只对齐一次）。

    多个 seed 复用同一份矩阵：对 350 万行 K 线表做 5 次 join 纯属自找超时。
    """
    joined = (df.select("close_time")
                .join(kfeat.select("close_time", *kcols), on="close_time", how="left")
                .sort("close_time"))
    return matrix(joined, kcols)


def train_leg(Xtr, ytr, cols, Xva, yva, tag: str) -> list:
    """一条腿的多 seed 模型列表（供概率平均）。"""
    models = []
    for seed in SEEDS:
        m = fit(Xtr, ytr, cols, Xva, yva, seed=seed)
        log(f"[{tag}] seed={seed} best_iter={m.best_iteration}")
        models.append(m)
    return models


def split_wide(wide: pl.DataFrame, how: str, t_tr_end: int, t_va_end: int):
    """两种切分口径：time = two_legs 的 close_time 阈值；row = h3_wide 的行号。"""
    if how == "time":
        tr = wide.filter(pl.col("close_time") <= t_tr_end)
        va = wide.filter((pl.col("close_time") > t_tr_end) & (pl.col("close_time") <= t_va_end))
        te = wide.filter(pl.col("close_time") > t_va_end)
    elif how == "row":
        n = wide.height
        k1, k2 = int(n * TRAIN_FRAC), int(n * (TRAIN_FRAC + VAL_FRAC))
        tr, va, te = wide[:k1], wide[k1:k2], wide[k2:]
    else:
        raise SystemExit(f"SPLITS 只能是 time / row，收到 {how}")
    return tr, va, te


def main() -> None:
    ob = ok.load_ob()
    obcols = [c for c in ob.columns if c != "close_time"]
    n = ob.height
    tr1, va1 = int(n * TRAIN_FRAC), int(n * (TRAIN_FRAC + VAL_FRAC))
    # 切分点与 two_legs 完全一致（同一张订单簿表的同一行号）
    t_tr_end, t_va_end = int(ob["close_time"][tr1 - 1]), int(ob["close_time"][va1 - 1])
    log(f"订单簿 {n:,} 行 | T_end={ok._iso(t_tr_end)} | val 末={ok._iso(t_va_end)}"
        f" | 订单簿特征 {len(obcols)} 列 | seeds={SEEDS} | splits={SPLITS} | fuse={FUSE}")

    kfeat, kcols = ok.load_kline_features(KLINE_YEARS, keep_raw=True)
    log(f"K 线腿数据 {kfeat.height:,} 行 | 特征 {len(kcols)} 列")

    # ---- 跨域特征：只在订单簿窗口（+35min 预热）上算振幅，350 万行滚动纯属浪费
    lo, hi = int(ob["close_time"].min()), int(ob["close_time"].max())
    kw = kfeat.filter((pl.col("close_time") >= lo - 35 * 60_000) & (pl.col("close_time") <= hi))
    amp = amplitude(kw).select("close_time", "amp_mean_30m_bps")
    log(f"振幅: {amp.height:,} 行")

    raw_cols = [c for c in ("um_buy_vol", "um_sell_vol", "spot_buy_vol", "spot_sell_vol")
                if c in kfeat.columns]
    if not raw_cols:
        raise SystemExit("K 线里没有主动买卖量列（keep_raw 失效？）")
    wide = (ob.join(kfeat.select("close_time", "close", *raw_cols),
                    on="close_time", how="left")
              .join(amp, on="close_time", how="left").sort("close_time"))
    wide, cross_cols = add_cross(wide)
    log(f"跨域特征 {len(cross_cols)} 列: {', '.join(cross_cols)}")

    wide = add_target(wide, HORIZON).filter(
        pl.col("target").is_not_null() & pl.col("close").is_not_null())
    log(f"可用 {wide.height:,} 行")

    # ---- 腿 A：K 线（2020→T_end 长历史），与 variant/split 无关，只训一次
    mk: list = []
    if FUSE:
        kv = add_target(kfeat, HORIZON).sort("close_time")
        k_tr = kv.filter((pl.col("close_time") <= t_tr_end) & pl.col("target").is_not_null())
        k_va = kv.filter((pl.col("close_time") > t_tr_end) & (pl.col("close_time") <= t_va_end)
                         & pl.col("target").is_not_null())
        log(f"腿A K线: train={k_tr.height:,} val={k_va.height:,}")
        mk = train_leg(matrix(k_tr, kcols), k_tr["target"].to_numpy().astype(int), kcols,
                       matrix(k_va, kcols), k_va["target"].to_numpy().astype(int), "腿A")

    for how in SPLITS:
        ob_tr, ob_va, ob_te = split_wide(wide, how, t_tr_end, t_va_end)
        yb_tr = ob_tr["target"].to_numpy().astype(int)
        yb_va = ob_va["target"].to_numpy().astype(int)
        yb_te = ob_te["target"].to_numpy().astype(int)
        d_va = day_id(ob_va["close_time"].to_numpy())
        d_te = day_id(ob_te["close_time"].to_numpy())
        log(f"[{how}] 订单簿腿样本: train={ob_tr.height:,} val={ob_va.height:,}"
            f" test={ob_te.height:,}")

        zA_va = zA_te = None
        if FUSE:
            Xa_tr, Xa_va, Xa_te = (kalign(ob_tr, kfeat, kcols), kalign(ob_va, kfeat, kcols),
                                   kalign(ob_te, kfeat, kcols))
            pA_tr = np.mean([np.nan_to_num(m.predict(Xa_tr), nan=0.5) for m in mk], axis=0)
            pA_va = np.mean([np.nan_to_num(m.predict(Xa_va), nan=0.5) for m in mk], axis=0)
            pA_te = np.mean([np.nan_to_num(m.predict(Xa_te), nan=0.5) for m in mk], axis=0)
            log(f"[{how}] 腿A AUC val={roc_auc_score(yb_va, pA_va):.4f}"
                f" test={roc_auc_score(yb_te, pA_te):.4f}")
            mu, sd = float(pA_tr.mean()), float(pA_tr.std() + 1e-9)
            zA_va, zA_te = (pA_va - mu) / sd, (pA_te - mu) / sd

        for i, v in enumerate(VARIANTS):
            parts = v.split("+")
            ocols = list(obcols) + (cross_cols if "cross" in parts else [])
            tag = f"{EXP}-{how}-{i + 1}" if len(VARIANTS) > 1 else f"{EXP}-{how}"
            log(f"[{tag}] variant={v}: 订单簿腿 {len(ocols)} 列"
                f"（基础 {len(obcols)} / 跨域 {len(cross_cols) if 'cross' in parts else 0}）")

            mb = train_leg(matrix(ob_tr, ocols), yb_tr, ocols,
                           matrix(ob_va, ocols), yb_va, tag)
            pB_tr = np.mean([m.predict(matrix(ob_tr, ocols)) for m in mb], axis=0)
            pB_va = np.mean([m.predict(matrix(ob_va, ocols)) for m in mb], axis=0)
            pB_te = np.mean([m.predict(matrix(ob_te, ocols)) for m in mb], axis=0)
            log(f"[{tag}] AUC val={roc_auc_score(yb_va, pB_va):.4f}"
                f" test={roc_auc_score(yb_te, pB_te):.4f}")

            w_va = report(f"{tag} val 纯订单簿({v})", pB_va - 0.5, yb_va, d_va)
            w_te = report(f"{tag} test 纯订单簿({v})", pB_te - 0.5, yb_te, d_te)
            if not FUSE:
                log(f"[{tag}] SUMMARY split={how} variant={v} cols={len(ocols)}"
                    f" seeds={len(SEEDS)} val={w_va * 100:.2f}% test={w_te * 100:.2f}%")
                continue

            # 融合：z 统计量取 train 段，w 只在 val 上选（同 two_legs）
            mu, sd = float(pB_tr.mean()), float(pB_tr.std() + 1e-9)
            zB_va, zB_te = (pB_va - mu) / sd, (pB_te - mu) / sd
            best_w, best_acc = None, -1.0
            for w in W_GRID:
                acc = global_topk(zB_va + w * zA_va, yb_va, d_va, TARGET_TPD)
                log(f"[{tag}] val w={w:<4} 日均{TARGET_TPD}单 胜率={acc * 100:.2f}%")
                if acc > best_acc:
                    best_w, best_acc = w, acc
            f_va = report(f"{tag} val 融合({v}, w={best_w})",
                          zB_va + best_w * zA_va, yb_va, d_va)
            f_te = report(f"{tag} test 融合({v}, w={best_w})",
                          zB_te + best_w * zA_te, yb_te, d_te)
            log(f"[{tag}] SUMMARY split={how} variant={v} cols={len(ocols)}"
                f" seeds={len(SEEDS)} ob val={w_va * 100:.2f}% test={w_te * 100:.2f}% |"
                f" fuse w={best_w} val={f_va * 100:.2f}% test={f_te * 100:.2f}%")
    log("DONE")


if __name__ == "__main__":
    main()
