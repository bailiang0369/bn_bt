#!/usr/bin/env python3
"""h=10 训练口径诊断：placebo（验泄漏）+ 标签重叠抽样（stride）A/B。

两个问题
--------
1. **train AUC 0.62 / valid 0.53 是否过拟合？** 不是。1.79M 行 × 49 特征、`num_leaves=63`、
   上限 2000 轮，真在记忆能到 0.85+；实际 best_iter 只有 29~197，说明模型只学到了很弱的信号。
   但 train AUC 的虚高有一半来自**标签重叠**：h=10 时相邻 10 行共用同一段价格路径，
   1.79M 行里独立样本只有约 18 万，拟合「这一段行情」就能同时拿到 10 个样本的正确率。
2. **有没有泄漏？** 标准检查是 placebo：把训练标签打乱重训，train/valid/test AUC 都应掉到
   0.50~0.52。若 train AUC 仍明显 >0.52，说明特征里有未来信息，那 63.59% 是假的。

本文件在同一窗口上跑三种配置，其余完全一致：
  base     全量训练（对照，应与 h10sigC 同量级）
  stride   训练集按 stride=HORIZON 抽样（去掉 90% 冗余样本）
  placebo  base + 打乱训练标签（验泄漏）

env
---
  WINDOWS=2026-08,2026-09,2026-10   诊断窗口（3 个够看方向，不必 11 个）
  STRIDES=1,10                      stride 配置
  SEEDS=42,43                       诊断用 2 seed（正式数字仍用 5 seed）
  HORIZON=10 / WIN_DAYS=1460 / ES_FRAC=0.05 / THR_FRAC=0.10 / TPD=8 / SYMBOL
"""
from __future__ import annotations

import datetime as dt
import os
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from morph import h3_gate_eval as ge  # noqa: E402
from morph import kline_long as kl  # noqa: E402
from morph.two_legs import add_target, fit, matrix  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

HORIZON = int(os.environ.get("HORIZON", "10"))
SEEDS = [int(s) for s in os.environ.get("SEEDS", "42,43").split(",") if s.strip()]
WIN_DAYS = int(os.environ.get("WIN_DAYS", "1460"))
ES_FRAC = float(os.environ.get("ES_FRAC", "0.05"))
THR_FRAC = float(os.environ.get("THR_FRAC", "0.10"))
TPD = float(os.environ.get("TPD", "8"))
SYMBOL = os.environ.get("SYMBOL", "BTCUSDT")
KLINE_FROM = int(os.environ.get("KLINE_FROM", "2020"))
WINDOWS = [m.strip() for m in os.environ.get(
    "WINDOWS", "2026-08,2026-09,2026-10").split(",") if m.strip()]
STRIDES = [int(x) for x in os.environ.get("STRIDES", "1,10").split(",") if x.strip()]
PAYOUT = 0.808
TRAIN_AUC_N = 200_000
T = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T:7.1f}s] {msg}", flush=True)


def main() -> None:
    raw = kl.load_raw()
    df, feats = kl.build(raw)
    df = add_target(df, HORIZON).filter(
        pl.col("target").is_not_null() & pl.col("close").is_not_null()).sort("close_time")
    df = df.with_columns(
        pl.col("close_time").cast(pl.Datetime("ms")).dt.strftime("%Y-%m").alias("ym"))
    log(f"{SYMBOL} 特征 {len(feats)} 列 | 可用 {df.height:,} 行 | 窗口={WINDOWS} "
        f"strides={STRIDES} seeds={len(SEEDS)}")

    print(f"\n{'窗口':9}{'配置':16}{'训练行数':>11}{'iter':>6}{'trainAUC':>10}"
          f"{'validAUC':>10}{'testAUC':>9}{'单/天':>7}{'胜率':>8}")
    for m in WINDOWS:
        m_start = int(dt.datetime.strptime(m + "-01", "%Y-%m-%d")
                      .replace(tzinfo=dt.timezone.utc).timestamp() * 1000)
        lo = m_start - WIN_DAYS * 86_400_000
        tr_all = df.filter((pl.col("close_time") < m_start) & (pl.col("close_time") >= lo))
        te = df.filter(pl.col("ym") == m)
        if tr_all.height < 100_000 or te.height < 1000:
            log(f"[{m}] 样本不足，跳过")
            continue
        c1 = int(tr_all.height * (1 - ES_FRAC - THR_FRAC))
        c2 = int(tr_all.height * (1 - THR_FRAC))
        tr0, es, th = tr_all[:c1], tr_all[c1:c2], tr_all[c2:]
        Xes, yes = matrix(es, feats), es["target"].to_numpy().astype(int)
        Xth = matrix(th, feats)
        Xte = matrix(te, feats)
        y = te["target"].to_numpy().astype(np.int8)
        days = ge.day_id(te["close_time"].to_numpy())
        nd = len(np.unique(days))

        runs = ([(f"base", 1, False)] +
                [(f"stride={s}", s, False) for s in STRIDES if s > 1] +
                [("placebo", 1, True)])
        for name, stride, shuffle in runs:
            tr = tr0.gather_every(stride) if stride > 1 else tr0
            Xtr, ytr = matrix(tr, feats), tr["target"].to_numpy().astype(int)
            if shuffle:
                rng = np.random.default_rng(7)
                ytr = rng.permutation(ytr)
            pv = np.zeros(es.height)
            pth = np.zeros(th.height)
            pt = np.zeros(te.height)
            its, tr_aucs = [], []
            for i, seed in enumerate(SEEDS):
                mdl = fit(Xtr, ytr, feats, Xes, yes, seed=seed)
                its.append(int(mdl.best_iteration))
                pv += mdl.predict(Xes, num_iteration=mdl.best_iteration) / len(SEEDS)
                pth += mdl.predict(Xth, num_iteration=mdl.best_iteration) / len(SEEDS)
                pt += mdl.predict(Xte, num_iteration=mdl.best_iteration) / len(SEEDS)
                if i == 0:
                    # train AUC 只在首个 seed 上、抽 20 万行评估（够用且便宜）
                    idx = np.arange(Xtr.shape[0])
                    if idx.size > TRAIN_AUC_N:
                        idx = np.random.default_rng(0).choice(idx, TRAIN_AUC_N, replace=False)
                    tr_aucs.append(float(roc_auc_score(ytr[idx], mdl.predict(Xtr[idx]))))
            auc_es = float(roc_auc_score(yes, pv))
            auc_te = float(roc_auc_score(y, pt))
            # 阈值只在 thr 段定（与 h10_signal 一致），不能用 es 段（早停段偏乐观）
            thr = float(np.quantile(np.abs(pth - .5), 1 - TPD / 1440))
            s = pt - .5
            hit = np.abs(s) >= thr
            nb = int(hit.sum())
            win = float(((s[hit] > 0).astype(int) == y[hit]).mean()) if nb >= 50 else float("nan")
            print(f"{m:9}{name:16}{Xtr.shape[0]:>11,}{int(np.mean(its)):>6}"
                  f"{tr_aucs[0]:>10.4f}{auc_es:>10.4f}{auc_te:>9.4f}"
                  f"{nb / nd:>7.1f}{win * 100:>7.2f}%", flush=True)
            del Xtr
    log("DONE — 判读：placebo 的 trainAUC 若 >0.53 则有泄漏；"
        "stride 版 trainAUC 应下降而 testAUC/胜率不变，则以后所有多窗实验都改用它（快 10 倍）")


if __name__ == "__main__":
    main()
