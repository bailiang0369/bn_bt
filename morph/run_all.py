#!/usr/bin/env python3
"""Kaggle 一键：Release 取数 -> 算特征 -> 训练 -> 评估（盯「日均 ≥15 单、胜率 ≥65%」）。

一条命令跑完，改特征只要改 `morph/kline_features.py` 或 `morph/ob_kline.py` 再推 kernel，
中间不落任何"数据集"产物、不需要先做一个 Dataset 再挂载。

    数据：bn_hisdata Release（vision-base 的 1m K 线 + ob-dataset-vol8 的订单簿特征）
    特征：K 线 63 列（morph.kline_features）+ 订单簿 184 列，主键 close_time 直接 join
    标签：t+h 分钟收盘方向（h 默认 3）
    切分：严格时序 64% / 16% / 20%，绝不打乱
    评估：阈值**只在验证集**按目标日均单数定，原样套测试集（不在测试集挑运营点）

env
---
    SYMBOL=BTCUSDT        KLINE_SRC=um|spot      WITH_BASIS=0|1
    HORIZON=3             预测 horizon（分钟）
    TARGET_TPD=15         验证集上瞄准的日均单数（决定阈值）
    ROUNDS=2000           LightGBM 最大轮数（early stopping 50）
    OB_MONTHS=2026-09     只跑单月（迭代期省时间）
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
    from morph.ob_kline import build
except ModuleNotFoundError:              # `python morph/run_all.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from morph.ob_kline import build

HORIZON = int(os.environ.get("HORIZON", "3"))
TARGET_TPD = float(os.environ.get("TARGET_TPD", "15"))
ROUNDS = int(os.environ.get("ROUNDS", "2000"))
SEED = int(os.environ.get("SEED", "42"))
TRAIN_FRAC, VAL_FRAC = 0.64, 0.16
T = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T:7.1f}s] {msg}", flush=True)


def make_target(df: pl.DataFrame, h: int) -> pl.DataFrame:
    """t+h 收盘价 vs t 收盘价 → target；同时留下未来收盘价用于算每单收益。

    连接键必须用 `close_time - h*60_000`：当前行 T 才能取到 T+h 的 close。
    写成 `+` 会变成"过去方向"，特征里全是收益率，会被完美拟合出虚假 AUC。
    """
    fut = df.select((pl.col("close_time") - h * 60_000).alias("close_time"),
                    pl.col("close").alias("_cf"))
    df = df.join(fut, on="close_time", how="left")
    df = df.with_columns((pl.col("_cf") > pl.col("close")).cast(pl.Int8).alias("target"))
    log(f"标签: t+{h}min 方向 | 正类占比 {df['target'].mean():.3f}")
    return df


def feature_matrix(df: pl.DataFrame, cols: list[str]) -> np.ndarray:
    """特征转 float32 矩阵；null 变 NaN 交给 LightGBM 原生处理。

    不能 `drop_nulls()`：订单簿的 `ob_*_vwap_*` 在某个深度桶没有挂单时就是 null
    （空桶），那不代表这一分钟缺数据，一 drop 会干掉大半行。
    """
    return df.select([pl.col(c).cast(pl.Float32) for c in cols]).to_numpy()


def report(name: str, prob: np.ndarray, y: np.ndarray, ret_bps: np.ndarray,
           days: np.ndarray, thr: float) -> None:
    """按给定置信度阈值评估：覆盖率、单/天、胜率、每单收益、日分布、方向拆分。

    "单/天"用**全部交易日**做分母（含零成交日），日分布也把零成交日算进去
    —— 只统计有单的日子会把流量波动抹平，"日均 15 单"就变成了平均值假象。
    """
    conf = np.abs(prob - 0.5)
    sel = conf >= thr
    if sel.sum() == 0:
        log(f"[{name}] 阈值 {thr:.4f} 下无成交")
        return
    pred = (prob >= 0.5).astype(int)
    acc = float((pred[sel] == y[sel]).mean())
    _, inv = np.unique(days, return_inverse=True)
    rd = np.bincount(inv[sel], minlength=int(inv.max()) + 1)   # 含零成交日
    log(f"[{name}] 阈值={thr:.4f} 覆盖={sel.mean() * 100:.2f}%"
        f" 单/天={sel.sum() / len(rd):.1f}"
        f" 胜率={acc * 100:.2f}% 每单={ret_bps[sel].mean():+.2f}bps")
    log(f"[{name}] 日成交分布: mean={rd.mean():.1f} std={rd.std():.1f}"
        f" min={rd.min()} max={rd.max()} 天={len(rd)} 零单日={int((rd == 0).sum())}")
    for side, mask in (("跌", pred == 0), ("涨", pred == 1)):
        m = sel & mask
        if m.sum():
            log(f"[{name}] {side}侧 {m.sum():,} 单 胜率={float((pred[m] == y[m]).mean()) * 100:.2f}%"
                f" 每单={ret_bps[m].mean():+.2f}bps")


def run_one(raw: pl.DataFrame, feat_cols: list[str], h: int) -> None:
    df = make_target(raw, h).sort("close_time")

    # 丢掉标签未知的尾部、以及 K 线缺失（close 为 null）的行；盘口 null 保留
    df = df.filter(pl.col("target").is_not_null() & pl.col("close").is_not_null())
    both_missing = (pl.col("ob_f_missing") == 1) & (pl.col("ob_s_missing") == 1)
    n0 = df.height
    df = df.filter(~both_missing)
    log(f"[h={h}] 可用行 {df.height:,}（丢两边都缺盘口 {n0 - df.height:,}）| 特征 {len(feat_cols)} 列")

    X = feature_matrix(df, feat_cols)
    y = df["target"].to_numpy().astype(int)
    close = df["close"].to_numpy()
    cf = df["_cf"].to_numpy()
    days = (df["close_time"].to_numpy() // 86_400_000)

    n = len(y)
    tr1, va1 = int(n * TRAIN_FRAC), int(n * (TRAIN_FRAC + VAL_FRAC))
    log(f"[h={h}] 时序切分 训练[0:{tr1}] 验证[{tr1}:{va1}] 测试[{va1}:{n}]"
        f" | 正类占比 {y[:tr1].mean():.3f}/{y[tr1:va1].mean():.3f}/{y[va1:].mean():.3f}")

    params = dict(objective="binary", metric="auc", boosting_type="gbdt",
                  learning_rate=0.05, num_leaves=63, feature_fraction=0.9,
                  bagging_fraction=0.9, bagging_freq=1, min_child_samples=100,
                  verbose=-1, seed=SEED, num_threads=os.cpu_count() or 4)
    ds_tr = lgb.Dataset(X[:tr1], y[:tr1], feature_name=feat_cols)
    ds_va = lgb.Dataset(X[tr1:va1], y[tr1:va1], reference=ds_tr)
    model = lgb.train(params, ds_tr, num_boost_round=ROUNDS,
                      valid_sets=[ds_tr, ds_va],
                      callbacks=[lgb.early_stopping(50, verbose=False),
                                 lgb.log_evaluation(100)])
    log(f"[h={h}] best_iteration={model.best_iteration}")

    p_va = model.predict(X[tr1:va1], num_iteration=model.best_iteration)
    p_te = model.predict(X[va1:], num_iteration=model.best_iteration)
    log(f"[h={h}] AUC 验证={roc_auc_score(y[tr1:va1], p_va):.4f}"
        f" 测试={roc_auc_score(y[va1:], p_te):.4f}"
        f" | 全量ACC 测试={float(((p_te >= 0.5) == y[va1:]).mean()) * 100:.2f}%")

    # 阈值只在验证集上定：取最自信的 k = TARGET_TPD × 验证天数 个
    val_days = len(np.unique(days[tr1:va1]))
    k = int(round(TARGET_TPD * val_days))
    conf_va = np.abs(p_va - 0.5)
    thr = float(np.sort(conf_va)[-k]) if 0 < k <= len(conf_va) else 0.0
    log(f"[h={h}] 阈值（验证集瞄准 {TARGET_TPD:g} 单/天 × {val_days} 天 = {k} 单）= {thr:.4f}")

    def ret(p: np.ndarray, sl: slice) -> np.ndarray:
        """每单收益（bps）：按预测方向持有的 t→t+h 收益。"""
        sign = np.where(p >= 0.5, 1.0, -1.0)
        return sign * (cf[sl] / close[sl] - 1.0) * 1e4

    report(f"h{h} 验证", p_va, y[tr1:va1], ret(p_va, slice(tr1, va1)), days[tr1:va1], thr)
    report(f"h{h} 测试", p_te, y[va1:], ret(p_te, slice(va1, None)), days[va1:], thr)

    imp = np.array(model.feature_importance(importance_type="gain"))
    order = np.argsort(imp)[::-1]
    log(f"[h={h}] Top15 特征(gain): " + ", ".join(feat_cols[i] for i in order[:15]))
    ob_f = np.array([c.startswith("ob_f_") for c in feat_cols])
    ob_s = np.array([c.startswith("ob_s_") for c in feat_cols])
    cross = np.array([c.startswith("ob_") for c in feat_cols]) & ~ob_f & ~ob_s
    for name, mask in (("K线", ~(ob_f | ob_s | cross)), ("盘口 ob_f_*", ob_f),
                       ("盘口 ob_s_*", ob_s), ("跨市场 ob_*", cross)):
        if mask.any():
            log(f"[h={h}] 特征组 {name}: {int(mask.sum())} 列, "
                f"gain 占比 {imp[mask].sum() / imp.sum() * 100:.1f}%")

    if Path("/kaggle/working").exists():
        model.save_model(f"/kaggle/working/model_h{h}.txt")


def main() -> None:
    raw, feat_cols, info = build()
    log(f"数据口径: {info}")
    # HORIZONS 可一次扫多个（如 3,5,10,15）：特征只算一次，标签/训练按 h 重跑，
    # 方便直接比较"哪个 horizon 更接近 65% 胜率"，而不是一次只改一个值重推。
    hs = [int(x) for x in os.environ.get("HORIZONS", "").split(",") if x.strip()]
    for h in hs or [HORIZON]:
        run_one(raw, feat_cols, h)
    log("DONE")


if __name__ == "__main__":
    main()
