#!/usr/bin/env python3
"""G*：h=3 二阶段下单过滤（只改下单口径，不动模型 / 不改特征）。

与 §17 旧实验的区别（这是本文件存在的原因）
------------------------------------------
1. 基线换成现役 **纯订单簿 5 seed LGBM**（`FEATS=ob`、2000 轮、64/16/20）；
   §17 用的是 `z_ob+0.2*z_kl` 融合打分 + 60 轮旧链路。
2. 阈值 / 窗口 / 分位数**只在 val 定**，test 只评估；
   §17 的 0.62 / 45 天是在三个重叠 test 窗口上扫出来的。
3. 主判据用「固定单量 + 配对 bootstrap」（`h3_gate_eval.py`），并配随机放行
   placebo、缩量口径、覆盖率与每日单量分布；§17 只有 20 次随机对照均值。
4. 波动率状态量严格因果：`|r3|` 先 `shift(H+1)` 再做 7 天滚动中位数，
   分位阈值取 val 分位（不是全 test 分位）。
5. `time` / `row` 两种切分**同一次运行**复现（§4.4）。

env
---
    SPLITS=time,row        切分口径
    SEEDS=42,43,44,45,46   5 seed 概率平均
    ROUNDS=2000            two_legs 读取
    KLINE_FROM=2025        只需 close 列（构造标签 + 波动状态量）
    OB_MONTHS              smoke 时用 --set OB_MONTHS=2026-09 走月度分片
    BOOT=1000 / PLACEBO=200 / MIN_COV=0.34
    EXP=G1                 实验编号
    OUT=/kaggle/working/h3_gate_pred.parquet   落盘 val+test 逐行分数（供离线复核）

产物 OUT 列：split, segment, close_time, target, score(概率), s_t, day
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
    from morph import h3_gate_eval as ge
    from morph import ob_kline as ok
    from morph.two_legs import (KLINE_YEARS, TARGET_TPD, TRAIN_FRAC, VAL_FRAC,
                                add_target, fit, matrix)
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from morph import h3_gate_eval as ge
    from morph import ob_kline as ok
    from morph.two_legs import (KLINE_YEARS, TARGET_TPD, TRAIN_FRAC, VAL_FRAC,
                                add_target, fit, matrix)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

SPLITS = [s.strip().lower() for s in os.environ.get("SPLITS", "time,row").split(",") if s.strip()]
SEEDS = [int(s) for s in os.environ.get("SEEDS", "42,43,44,45,46").split(",") if s.strip()]
HORIZON = 3
KLINE_FROM = int(os.environ.get("KLINE_FROM", "0")) or 2025
EXP = os.environ.get("EXP", "G?")
OUT = os.environ.get("OUT", "/kaggle/working/h3_gate_pred.parquet")
BOOT = int(os.environ.get("BOOT", "1000"))
PLACEBO = int(os.environ.get("PLACEBO", "200"))
MIN_COV = float(os.environ.get("MIN_COV", "0.34"))
RM_WINDOWS = (15, 30, 45)
RM_THRESHOLDS = (0.58, 0.60, 0.62, 0.64)
VOL_QUANTILES = (0.20, 0.30, 0.50)
BARS_7D = 7 * 1440
T = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T:7.1f}s] {msg}", flush=True)


def split_wide(wide: pl.DataFrame, how: str, t_tr_end: int, t_va_end: int):
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


def train_predict(tr, va, te, ocols, tag: str):
    """现役基线：5 seed 概率平均；train 只用于建模，评估/闸门只用 val+test。"""
    Xtr, ytr = matrix(tr, ocols), tr["target"].to_numpy().astype(int)
    Xva, yva = matrix(va, ocols), va["target"].to_numpy().astype(int)
    ms = []
    for seed in SEEDS:
        m = fit(Xtr, ytr, ocols, Xva, yva, seed=seed)
        log(f"[{tag}] seed={seed} best_iter={m.best_iteration}")
        ms.append(m)

    def pred(df):
        return np.mean([m.predict(matrix(df, ocols)) for m in ms], axis=0)

    return pred(va), pred(te)


def build_candidates(s_va, y_va, d_va, s_te, y_te, d_te, vol_va, vol_te, tpd: int):
    """闸门候选；全部严格因果，阈值来源只有 val（滚动准确率的分位/阈值是预先给定的网格）。

    滚动监控的历史用 val+test 的连续 out-of-sample 序列：val 前 W 天冷启动放行，
    test 能用到 val 尾部的已实现结果，与实盘「每天看过去 45 天」一致。
    """
    all_s = np.concatenate([s_va, s_te])
    all_y = np.concatenate([y_va, y_te])
    all_d = np.concatenate([d_va, d_te])
    out = []
    for w in RM_WINDOWS:
        uniq, trail = ge.rolling_trailing(all_s, all_y, all_d, w, tpd)
        for thr in RM_THRESHOLDS:
            out.append(dict(name=f"rm W={w} thr={thr:.2f}", group="rm",
                            va=ge.trailing_gate(d_va, uniq, trail, thr),
                            te=ge.trailing_gate(d_te, uniq, trail, thr)))
    if np.isfinite(vol_va).any():
        for q in VOL_QUANTILES:
            thr = float(np.nanquantile(vol_va, q))
            for side in ("low", "high"):
                def gate(v):
                    ok = np.isfinite(v)
                    inside = (v <= thr) if side == "low" else (v >= thr)
                    return (~ok) | inside   # 冷启动 NaN 默认放行
                out.append(dict(name=f"vol {side} q={q:.2f} thr={thr:.2f}bp", group="vol",
                                va=gate(vol_va), te=gate(vol_te)))
    ones_v = np.ones(len(s_va), dtype=bool)
    ones_t = np.ones(len(s_te), dtype=bool)
    out.append(dict(name="none 全放行", group="none", va=ones_v, te=ones_t))
    return out


def val_utility(s_va, y_va, d_va, allowed, tpd: int):
    """val 选优只看 @25%（SE 最小，§4.2 次判据），并要求覆盖率下限与固定单量可行。"""
    cov = ge.coverage(allowed, d_va)
    k_full = tpd * cov["total_days"]
    if cov["day_cov"] < MIN_COV or int(allowed.sum()) < k_full:
        return (-np.inf, -np.inf), cov
    m0 = ge.four_tiers(s_va, y_va, d_va, tpd)
    m1 = ge.four_tiers(ge.apply_gate(s_va, allowed), y_va, d_va, tpd)
    return (m1["25%"] - m0["25%"], m1["5%"] - m0["5%"]), cov


def main() -> None:
    ob = ok.load_ob()
    obcols = [c for c in ob.columns if c != "close_time"]
    n = ob.height
    tr1, va1 = int(n * TRAIN_FRAC), int(n * (TRAIN_FRAC + VAL_FRAC))
    t_tr_end, t_va_end = int(ob["close_time"][tr1 - 1]), int(ob["close_time"][va1 - 1])
    log(f"订单簿 {n:,} 行 | T_end={ok._iso(t_tr_end)} | val 末={ok._iso(t_va_end)}"
        f" | 订单簿特征 {len(obcols)} 列 | seeds={SEEDS} | splits={SPLITS}")

    kfeat, _ = ok.load_kline_features(KLINE_FROM, keep_raw=False)
    wide = (ob.join(kfeat.select("close_time", "close"), on="close_time", how="left")
              .sort("close_time"))
    # 波动状态量：|r3| 用到 close[t+3]，先 shift(H+1) 再滚动，窗口末端仍严格在过去
    close = wide["close"].to_numpy().astype(np.float64)
    r3 = np.full(len(close), np.nan)
    r3[:-HORIZON] = (close[HORIZON:] / close[:-HORIZON] - 1.0) * 1e4
    s_t = (pl.Series(np.abs(r3)).shift(HORIZON + 1)
           .rolling_median(BARS_7D, min_samples=2880).to_numpy())
    wide = wide.with_columns(pl.Series("s_t", s_t))
    wide = add_target(wide, HORIZON).filter(
        pl.col("target").is_not_null() & pl.col("close").is_not_null())
    log(f"可用 {wide.height:,} 行 | s_t 有效 {np.isfinite(wide['s_t'].to_numpy()).sum():,} 行"
        f" | K 线起点 {KLINE_FROM} / 默认 {KLINE_YEARS}")

    frames = []
    for how in SPLITS:
        tr, va, te = split_wide(wide, how, t_tr_end, t_va_end)
        if min(tr.height, va.height, te.height) < 1000:
            log(f"[{how}] 样本不足（{tr.height}/{va.height}/{te.height}），跳过")
            continue
        tag = f"{EXP}-{how}"
        log(f"[{tag}] train={tr.height:,} val={va.height:,} test={te.height:,}")
        p_va, p_te = train_predict(tr, va, te, obcols, tag)
        log(f"[{tag}] AUC val={roc_auc_score(va['target'].to_numpy(), p_va):.4f}"
            f" test={roc_auc_score(te['target'].to_numpy(), p_te):.4f}")

        s_va, s_te = p_va - 0.5, p_te - 0.5
        y_va = va["target"].to_numpy().astype(np.int8)
        y_te = te["target"].to_numpy().astype(np.int8)
        d_va = ge.day_id(va["close_time"].to_numpy())
        d_te = ge.day_id(te["close_time"].to_numpy())
        vol_va = va["s_t"].to_numpy().astype(np.float64)
        vol_te = te["s_t"].to_numpy().astype(np.float64)
        ge.report(f"{tag} val 基线", s_va, y_va, d_va, TARGET_TPD)
        ge.report(f"{tag} test 基线", s_te, y_te, d_te, TARGET_TPD)

        for seg, df, p in (("val", va, p_va), ("test", te, p_te)):
            frames.append(pl.DataFrame({
                "split": how, "segment": seg,
                "close_time": df["close_time"], "target": df["target"],
                "score": p, "s_t": df["s_t"],
                "day": ge.day_id(df["close_time"].to_numpy()),
            }))

        cands = build_candidates(s_va, y_va, d_va, s_te, y_te, d_te,
                                 vol_va, vol_te, TARGET_TPD)
        scored = []
        for c in cands:
            util, cov = val_utility(s_va, y_va, d_va, c["va"], TARGET_TPD)
            scored.append((util, cov, c))
            log(f"[{tag}] 候选 {c['name']:<28} val @25%差={util[0] * 100:+.2f}pp "
                f"@5%差={util[1] * 100:+.2f}pp 覆盖率={cov['day_cov']:.1%}")
        scored.sort(key=lambda x: x[0], reverse=True)
        best_util, _, best = scored[0]
        log(f"[{tag}] val 选定 = {best['name']}（@25% {best_util[0] * 100:+.2f}pp）")

        plans = ge.make_plans(d_te, BOOT)
        # (a) 预注册复核：§17 的 45 天 / 0.62，阈值不再调，直接在 test 评估
        pre = next((c for c in cands if c["name"] == "rm W=45 thr=0.62"), None)
        if pre is not None:
            ge.evaluate_gate(s_te, y_te, d_te, pre["te"], plans,
                             f"{tag} 预注册(45d/0.62)", TARGET_TPD, PLACEBO)
        # (b) val 选出的闸门：test 只评估一次
        r = ge.evaluate_gate(s_te, y_te, d_te, best["te"], plans,
                             f"{tag} val选定({best['name']})", TARGET_TPD, PLACEBO)
        ci = r["paired"]["25%"]["lo"] * r["paired"]["25%"]["hi"] > 0
        log(f"[{tag}] SUMMARY split={how} best={best['name']} "
            f"覆盖率={r['cov']['day_cov']:.2f} @25%差={r['paired']['25%']['diff'] * 100:+.2f}pp "
            f"CI95[{r['paired']['25%']['lo'] * 100:+.2f},{r['paired']['25%']['hi'] * 100:+.2f}] "
            f"缩量={r['shrink'] * 100:.2f}% 同K全局={r['same_k'] * 100:.2f}% "
            f"结论={'CI不跨0' if ci else '不显著'}")

    if OUT and frames:
        pl.concat(frames).write_parquet(OUT)
        log(f"预测已落盘: {OUT}（{sum(f.height for f in frames):,} 行，val+test 逐行分数）")
    log("DONE")


if __name__ == "__main__":
    main()
