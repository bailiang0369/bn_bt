#!/usr/bin/env python3
"""两条腿分别训练，再在**预测层**融合（不做 K 线 × 订单簿的特征拼接）。

为什么分开（`融合实验记录.md` 的结论）
--------------------------------------
- K 线有 2020→今天（≈350 万行），订单簿只有 2025-07→2026-10（≈69 万行）。拼成一张表
  只能按订单簿那段短窗口训练，K 线 5 年多的历史会被裁掉——等于自废一条腿。
- 两侧 `close` 同源（相关 1.000000），窗口内特征高度共线，拼接不涨 AUC。
- 3min 订单簿腿强、15min K 线腿强，谁强取决于 horizon，只能在**融合层**按 val 定权重。

流程
----
    腿 A（K 线）：vision-base 的 1m K 线 2020→今天 → morph.kline_features（63 列）
                 训练集 = close_time ≤ T_end 的全部 K 线行；val = 订单簿 val 窗口那批
    腿 B（订单簿）：ob-dataset-vol8 的 ob_*（184 列）
                 训练集 = 订单簿时序 64%；val = 16%
    评估：两腿在**同一批 close_time**（订单簿 val / test 窗口）上预测
    融合：z-score（统计量取自订单簿 train 段）→ z_B + w·z_A，w 在 val 上按
          「日均 15 单」口径选，再原样套 test；候选含 w=0（纯订单簿）

env
---
    SYMBOL=BTCUSDT  KLINE_SRC=um|spot  HORIZONS=3,15  TARGET_TPD=15
    ROUNDS=2000     KLINE_YEARS=2020（起始年）
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

try:
    from morph import ob_kline as ok
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from morph import ob_kline as ok

HORIZONS = [int(x) for x in os.environ.get("HORIZONS", "3,15").split(",") if x.strip()]
TARGET_TPD = int(os.environ.get("TARGET_TPD", "15"))
ROUNDS = int(os.environ.get("ROUNDS", "2000"))
SEED = int(os.environ.get("SEED", "42"))
KLINE_YEARS = int(os.environ.get("KLINE_YEARS", "2020"))
TRAIN_FRAC, VAL_FRAC = 0.64, 0.16
T = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T:7.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------- 评估口径
def day_id(ct: np.ndarray) -> np.ndarray:
    return (ct // 86_400_000).astype(np.int64)


def global_topk(score: np.ndarray, y: np.ndarray, days: np.ndarray, kpd: int) -> float:
    """全段按 |score| 取前 kpd×天数 单：有信号的天多下、没信号的天不下。

    这才是"长期日均 kpd 单"的正确口径；强行每天凑满会系统性低估。
    """
    k = min(kpd * len(np.unique(days)), len(y))
    idx = np.argsort(-np.abs(score))[:k]
    return float(((score[idx] > 0).astype(int) == y[idx]).mean())


def acc_at_cov(score: np.ndarray, y: np.ndarray, cov: float) -> float:
    k = max(1, int(len(y) * cov))
    idx = np.argsort(-np.abs(score))[:k]
    return float(((score[idx] > 0).astype(int) == y[idx]).mean())


def report(tag: str, score: np.ndarray, y: np.ndarray, days: np.ndarray) -> float:
    k = min(TARGET_TPD * len(np.unique(days)), len(y))
    idx = np.argsort(-np.abs(score))[:k]
    win = float(((score[idx] > 0).astype(int) == y[idx]).mean())
    log(f"[{tag}] 日均{TARGET_TPD}单: 胜率={win * 100:.2f}% (n={k:,})"
        f" | @1%={acc_at_cov(score, y, .01) * 100:.2f}%"
        f" @5%={acc_at_cov(score, y, .05) * 100:.2f}%"
        f" @25%={acc_at_cov(score, y, .25) * 100:.2f}%")
    return win


# ---------------------------------------------------------------- 训练
def fit(Xtr: np.ndarray, ytr: np.ndarray, cols: list[str],
        Xva: np.ndarray | None = None, yva: np.ndarray | None = None) -> lgb.Booster:
    params = dict(objective="binary", metric="auc", boosting_type="gbdt",
                  learning_rate=0.05, num_leaves=63, feature_fraction=0.9,
                  bagging_fraction=0.9, bagging_freq=1, min_child_samples=100,
                  verbose=-1, seed=SEED, num_threads=os.cpu_count() or 4)
    ds_tr = lgb.Dataset(Xtr, ytr, feature_name=cols)
    valid, callbacks = [ds_tr], [lgb.log_evaluation(200)]
    if Xva is not None and len(yva):
        valid.append(lgb.Dataset(Xva, yva, reference=ds_tr))
        callbacks.append(lgb.early_stopping(50, verbose=False))
    return lgb.train(params, ds_tr, num_boost_round=ROUNDS,
                     valid_sets=valid, callbacks=callbacks)


def add_target(df: pl.DataFrame, h: int) -> pl.DataFrame:
    """t+h 收盘方向；连接键 `close_time - h*60_000`（写成 + 会变成过去方向）。"""
    fut = df.select((pl.col("close_time") - h * 60_000).alias("close_time"),
                    pl.col("close").alias("_cf"))
    df = df.join(fut, on="close_time", how="left")
    return df.with_columns((pl.col("_cf") > pl.col("close")).cast(pl.Int8).alias("target"))


def matrix(df: pl.DataFrame, cols: list[str]) -> np.ndarray:
    """null → NaN 交给 LightGBM；不能 drop_nulls（ob_*_vwap_* 空桶本就是 null）。"""
    return df.select([pl.col(c).cast(pl.Float32) for c in cols]).to_numpy()


# ---------------------------------------------------------------- 单 horizon
def run_horizon(kfeat: pl.DataFrame, kcols: list[str], ob: pl.DataFrame, obcols: list[str],
                t_tr_end: int, t_va_end: int, h: int) -> None:
    log(f"===== h={h} =====")

    # ---- 腿 B：订单簿，只吃订单簿自己那 64% 时序段
    obv = add_target(ob, h).sort("close_time")
    obv = obv.filter(pl.col("target").is_not_null() & pl.col("close").is_not_null())
    ob_tr = obv.filter(pl.col("close_time") <= t_tr_end)
    ob_va = obv.filter((pl.col("close_time") > t_tr_end) & (pl.col("close_time") <= t_va_end))
    ob_te = obv.filter(pl.col("close_time") > t_va_end)
    if min(ob_tr.height, ob_va.height, ob_te.height) < 5_000:
        log(f"[h={h}] 订单簿样本不足，跳过"); return

    Xb_tr, yb_tr = matrix(ob_tr, obcols), ob_tr["target"].to_numpy().astype(int)
    Xb_va, yb_va = matrix(ob_va, obcols), ob_va["target"].to_numpy().astype(int)
    mB = fit(Xb_tr, yb_tr, obcols, Xb_va, yb_va)
    pB_tr, pB_va, pB_te = mB.predict(Xb_tr), mB.predict(Xb_va), mB.predict(matrix(ob_te, obcols))
    log(f"[h={h}] 腿B 订单簿: train={ob_tr.height:,} val={ob_va.height:,}"
        f" test={ob_te.height:,} | best_iter={mB.best_iteration}")

    # ---- 腿 A：K 线，训练段是 2020→T_end（长历史），预测只看评估窗口
    kv = add_target(kfeat, h).sort("close_time")
    k_tr = kv.filter((pl.col("close_time") <= t_tr_end) & pl.col("target").is_not_null())
    k_va = kv.filter((pl.col("close_time") > t_tr_end) & (pl.col("close_time") <= t_va_end)
                     & pl.col("target").is_not_null())
    Xk_tr, yk_tr = matrix(k_tr, kcols), k_tr["target"].to_numpy().astype(int)
    mK = fit(Xk_tr, yk_tr, kcols, matrix(k_va, kcols),
             k_va["target"].to_numpy().astype(int))
    log(f"[h={h}] 腿A K线: train={k_tr.height:,}（{ok._iso(int(k_tr['close_time'].min()))}"
        f" ~ {ok._iso(t_tr_end)}）| val={k_va.height:,} | best_iter={mK.best_iteration}")

    # ---- 对齐到订单簿评估窗口（同一批 close_time 才能比较/融合）
    def kpred(target: pl.DataFrame) -> np.ndarray:
        joined = (target.select("close_time")
                        .join(kfeat, on="close_time", how="left").sort("close_time"))
        return np.nan_to_num(mK.predict(matrix(joined, kcols)), nan=0.5)

    pA_tr, pA_va, pA_te = kpred(ob_tr), kpred(ob_va), kpred(ob_te)

    y_va, y_te = yb_va, ob_te["target"].to_numpy().astype(int)
    d_va = day_id(ob_va["close_time"].to_numpy())
    d_te = day_id(ob_te["close_time"].to_numpy())
    log(f"[h={h}] AUC val: K线={roc_auc_score(y_va, pA_va):.4f}"
        f" 订单簿={roc_auc_score(y_va, pB_va):.4f}"
        f" | test: K线={roc_auc_score(y_te, pA_te):.4f}"
        f" 订单簿={roc_auc_score(y_te, pB_te):.4f}")

    # ---- 融合：z-score 统计量取订单簿 train 段（两腿同一批行，才可比）
    zA = lambda p: (p - float(pA_tr.mean())) / float(pA_tr.std() + 1e-9)   # noqa: E731
    zB = lambda p: (p - float(pB_tr.mean())) / float(pB_tr.std() + 1e-9)   # noqa: E731
    zA_va, zB_va = zA(pA_va), zB(pB_va)

    report(f"h{h} val 纯K线", pA_va - 0.5, y_va, d_va)
    report(f"h{h} val 纯订单簿", pB_va - 0.5, y_va, d_va)
    best_w, best_acc = None, -1.0
    for w in (0.0, 0.5, 1.0, 2.0, 3.0, 5.0):
        acc = global_topk((zB_va + w * zA_va), y_va, d_va, TARGET_TPD)
        log(f"[h{h}] val w={w:<4} 日均{TARGET_TPD}单 胜率={acc * 100:.2f}%")
        if acc > best_acc:
            best_w, best_acc = w, acc
    log(f"[h{h}] val 选 w={best_w}（胜率 {best_acc * 100:.2f}%）")

    report(f"h{h} test 纯K线", pA_te - 0.5, y_te, d_te)
    report(f"h{h} test 纯订单簿", pB_te - 0.5, y_te, d_te)
    report(f"h{h} test 融合(w={best_w})", zB(pB_te) + best_w * zA(pA_te), y_te, d_te)

    for tag, cols, imp in (("K线腿", kcols, np.array(mK.feature_importance("gain"))),
                           ("订单簿腿", obcols, np.array(mB.feature_importance("gain")))):
        log(f"[h{h}] {tag} Top8: " + ", ".join(cols[i] for i in np.argsort(imp)[::-1][:8]))


def main() -> None:
    ob = ok.load_ob()
    obcols = [c for c in ob.columns if c != "close_time"]   # 先定特征列，避免混进 close
    n = ob.height
    tr1, va1 = int(n * TRAIN_FRAC), int(n * (TRAIN_FRAC + VAL_FRAC))
    t_tr_end, t_va_end = int(ob["close_time"][tr1 - 1]), int(ob["close_time"][va1 - 1])
    log(f"订单簿 {n:,} 行 | T_end={ok._iso(t_tr_end)} | val 末={ok._iso(t_va_end)}"
        f" | 特征 {len(obcols)} 列")

    kfeat, kcols = ok.load_kline_features(KLINE_YEARS)
    log(f"K 线腿数据 {kfeat.height:,} 行 | 特征 {len(kcols)} 列")

    # 订单簿是纯特征表（185 列里没有 close）：标签要用的收盘价按 close_time 从 K 线取。
    # 这是**取标签**，不是把 K 线特征并进订单簿腿（obcols 在上面已经定好，不含 close）。
    ob = ob.join(kfeat.select("close_time", "close"), on="close_time", how="left")
    miss = int(ob["close"].null_count())
    if miss:
        log(f"注意：{miss:,} 行订单簿在 K 线里没有收盘价（标签不可用，会被丢掉）")

    for h in HORIZONS:
        run_horizon(kfeat, kcols, ob, obcols, t_tr_end, t_va_end, h)
    log("DONE")


if __name__ == "__main__":
    main()
