#!/usr/bin/env python3
"""AUC 提升探针：特征 / 超参两阶段对比（**新文件，不改现役代码**）。

背景（2026-11-16）
------------------
`续接说明.md` 与各 `kline_boost*` 的记录显示：3m 重采样、标签净化、形态聚类、
VP 量价分布、CNN/TCN、回归目标、跨品种 ETH、订单簿宽表、两腿融合、滚动归一化、
h1/2/3 分数叠加……全部试过且全部无增量，AUC 锁在 0.5269~0.530。

翻遍代码后，只剩两处**从未验证**：

1. **LightGBM 超参从未搜索**。全仓库硬编码
   `num_leaves=63 / min_child_samples=100 / feature_fraction=0.9 / 无 lambda_l2`。
   在 1.05M 行 × 49 列、AUC≈0.527 的极低信噪比下，`min_child_samples=100`
   意味着一个叶子只需 100 个样本就能分裂——这是典型的过拟合配置。
   `kline_boost4` 说「特征/集成都到顶了」，但从未说超参到顶。

2. **`h10_signal.add_struct` / `add_liq` 两组特征写好了、默认关闭、没有实验记录**。
   `add_struct` 的注释明说「本仓库 49 列里完全没有」：
   前高前低距离、整数关口距离、日内开盘距离、影线、多周期共振。
   `add_liq` 是「推动 1bp 需要多少量」的冲击/深度类比值特征。

做法
----
同一套 walk-forward 切分（与 mem_replay 完全一致），两阶段串行：
  Stage 1  固定 base 特征，比 PARAM_SETS 里各组超参的 OOS AUC
  Stage 2  用 Stage 1 最优超参，比 base / +struct / +liq / +both 四组特征

主指标是 OOS AUC（拼接全部 test 月）；辅以 @日均 PRIMARY_TPD 单的胜率，
避免「AUC 涨了但主操作点没动」。

注意：AUC 只涨 0.002~0.005 在 n=44 万时约 1σ，本实验用逐月 AUC 的
std 和「10 个月里几个月改善」判断稳健性，不只看汇总值。

env
---
  HORIZON=10 WIN_DAYS=90 START_MONTH=2026-01 END_MONTH=2026-10
  SEEDS=42,43,44 ROUNDS=2000 ES_FRAC=0.05 THR_FRAC=0.10
  PRIMARY_TPD=8 BARS_PER_DAY=1440
  STAGE=all|params|feats   （默认 all）
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
from morph import kline_long as kl  # noqa: E402
from morph.h10_signal import add_flow, add_liq, add_struct  # noqa: E402
from morph.two_legs import add_target, matrix  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

import lightgbm as lgb  # noqa: E402

HORIZON = int(os.environ.get("HORIZON", "10"))
WIN_DAYS = int(os.environ.get("WIN_DAYS", "90"))
SEEDS = [int(s) for s in os.environ.get("SEEDS", "42,43,44").split(",") if s.strip()]
ROUNDS = int(os.environ.get("ROUNDS", "2000"))
ES_FRAC = float(os.environ.get("ES_FRAC", "0.05"))
THR_FRAC = float(os.environ.get("THR_FRAC", "0.10"))
START_MONTH = os.environ.get("START_MONTH", "2026-01")
END_MONTH = os.environ.get("END_MONTH", "2026-10")
PRIMARY_TPD = float(os.environ.get("PRIMARY_TPD", "8"))
BARS_PER_DAY = int(os.environ.get("BARS_PER_DAY", "1440"))
STAGE = os.environ.get("STAGE", "all").strip().lower()
T = time.time()

# 现役参数即 base；其余为强正则化候选（低信噪比下应显著抬 min_child_samples）
PARAM_SETS: dict[str, dict] = {
    "base":    dict(learning_rate=.05, num_leaves=63, min_child_samples=100,
                    feature_fraction=.9, lambda_l2=0),
    "reg1":    dict(learning_rate=.05, num_leaves=31, min_child_samples=1000,
                    feature_fraction=.7, lambda_l2=10),
    "reg2":    dict(learning_rate=.05, num_leaves=15, min_child_samples=3000,
                    feature_fraction=.5, lambda_l2=100),
    "shallow": dict(learning_rate=.03, num_leaves=15, min_child_samples=500,
                    feature_fraction=.6, lambda_l2=20, max_depth=4),
    "slow":    dict(learning_rate=.01, num_leaves=31, min_child_samples=2000,
                    feature_fraction=.6, lambda_l2=50),
}


def log(msg: str) -> None:
    print(f"[{time.time() - T:7.1f}s] {msg}", flush=True)


def fit(Xtr, ytr, cols, Xes, yes, seed, extra: dict):
    params = dict(objective="binary", metric="auc", boosting_type="gbdt",
                  bagging_fraction=.9, bagging_freq=1, verbose=-1,
                  seed=seed, num_threads=os.cpu_count() or 4)
    params.update(extra)
    ds = lgb.Dataset(Xtr, ytr, feature_name=cols)
    va = lgb.Dataset(Xes, yes, reference=ds)
    return lgb.train(params, ds, num_boost_round=ROUNDS, valid_sets=[ds, va],
                     callbacks=[lgb.early_stopping(50, verbose=False)])


def main() -> None:
    raw = kl.load_raw()
    df, feats = kl.build(raw)
    df, feats = add_flow(df, feats)
    df, feats = add_struct(df, feats)       # 先全建，后面按列名挑子集
    df, feats = add_liq(df, feats)
    f_base = [c for c in feats if not c.startswith(("st_", "lq_"))]
    f_st = f_base + [c for c in feats if c.startswith("st_")]
    f_lq = f_base + [c for c in feats if c.startswith("lq_")]
    fset = {"base": f_base, "struct": f_st, "liq": f_lq, "both": feats}

    df = add_target(df, HORIZON).filter(
        pl.col("target").is_not_null() & pl.col("close").is_not_null()).sort("close_time")
    df = df.with_columns(
        pl.col("close_time").cast(pl.Datetime("ms")).dt.strftime("%Y-%m").alias("ym"))
    X_all = matrix(df, feats)
    y_all = df["target"].to_numpy().astype(int)
    ct = df["close_time"].cast(pl.Int64).to_numpy()
    ym = df["ym"].to_numpy()
    log(f"特征 base={len(f_base)} +struct={len(f_st) - len(f_base)} "
        f"+liq={len(f_lq) - len(f_base)} | 可用 {df.height:,} 行 | H={HORIZON} "
        f"| 窗 {WIN_DAYS}d | seeds={len(SEEDS)}")

    # ---- 切分只算一次，所有配置复用（保证完全配对）----
    months = [m for m in sorted(set(ym.tolist())) if START_MONTH <= m <= END_MONTH]
    splits = []
    for m in months:
        m_start = int(dt.datetime.strptime(m + "-01", "%Y-%m-%d")
                      .replace(tzinfo=dt.timezone.utc).timestamp() * 1000)
        lo = m_start - WIN_DAYS * 86_400_000
        i_tr = np.where((ct < m_start) & (ct >= lo))[0]
        i_te = np.where(ym == m)[0]
        if len(i_tr) < 100_000 or len(i_te) < 1_000:
            log(f"[{m}] 样本不足（{len(i_tr):,}/{len(i_te):,}），跳过")
            continue
        h = len(i_tr)
        c1 = int(h * (1 - ES_FRAC - THR_FRAC))
        c2 = int(h * (1 - THR_FRAC))
        splits.append((m, i_tr[:c1], i_tr[c1:c2], i_te))
    log(f"walk-forward {len(splits)} 个窗口：{splits[0][0]}~{splits[-1][0]}")

    def walk(cols: list[str], params: dict, tag: str) -> dict:
        ci = [feats.index(c) for c in cols]
        X = X_all[:, ci]
        ps, ys, ndays = [], [], 0
        mon_auc, its = [], []
        for m, i_tr, i_es, i_te in splits:
            p = np.zeros(len(i_te))
            it = []
            for seed in SEEDS:
                mdl = fit(X[i_tr], y_all[i_tr], cols, X[i_es], y_all[i_es], seed, params)
                p += mdl.predict(X[i_te], num_iteration=mdl.best_iteration) / len(SEEDS)
                it.append(mdl.best_iteration or ROUNDS)
            y = y_all[i_te]
            ps.append(p)
            ys.append(y)
            ndays += len(np.unique((ct[i_te] // 86_400_000)))
            mon_auc.append(float(roc_auc_score(y, p)))
            its.append(int(np.mean(it)))
        p = np.concatenate(ps)
        y = np.concatenate(ys)
        s = p - .5
        k = min(int(PRIMARY_TPD * ndays), len(y))
        idx = np.argsort(-np.abs(s))[:k]
        win = float(((s[idx] > 0).astype(int) == y[idx]).mean())
        a = np.array(mon_auc)
        auc = float(roc_auc_score(y, p))
        log(f"  {tag:<18} AUC={auc:.4f} | 月std={a.std() * 100:.2f}pp "
            f"| @日均{PRIMARY_TPD:g}单={win * 100:.2f}% (n={k:,}) | iter={np.mean(its):.0f}")
        return dict(tag=tag, auc=auc, mon=a, win=win, ndays=ndays)

    results: list[dict] = []
    best_p_name = "base"
    if STAGE in ("all", "params"):
        log("===== Stage 1：超参对比（特征 base）=====")
        st1 = [walk(f_base, p, f"p:{name}") for name, p in PARAM_SETS.items()]
        results += st1
        best = max(st1, key=lambda r: r["auc"])
        best_p_name = best["tag"].split(":", 1)[1]
        log(f"Stage 1 最优超参：{best_p_name}（AUC {best['auc']:.4f}）")

    if STAGE in ("all", "feats"):
        log(f"===== Stage 2：特征集对比（超参 {best_p_name}）=====")
        results += [walk(fset[name], PARAM_SETS[best_p_name], f"f:{name}")
                    for name in ("base", "struct", "liq", "both")]

    # ---- 汇总（两阶段各自以本阶段的 base 为基准，避免跨阶段混淆）----
    for prefix, title in (("p", "Stage 1 超参（基准 p:base）"), ("f", "Stage 2 特征（基准 f:base）")):
        rows = [r for r in results if r["tag"].startswith(prefix + ":")]
        if not rows:
            continue
        b = next((r for r in rows if r["tag"] == f"{prefix}:base"), None)
        print(f"\n--- {title} ---")
        print(f"{'配置':<20}{'AUC':>9}{'ΔAUC':>9}{'月std':>9}{'改善月':>8}{'@日均单':>10}")
        for r in rows:
            d = (r["auc"] - b["auc"]) if b else float("nan")
            better = int((np.array(r["mon"]) > np.array(b["mon"])).sum()) if b else 0
            print(f"{r['tag']:<20}{r['auc']:>9.4f}{d:>+9.4f}"
                  f"{r['mon'].std() * 100:>8.2f}pp{better:>5}/{len(r['mon'])}"
                  f"{r['win'] * 100:>9.2f}%")
    log("DONE")


if __name__ == "__main__":
    main()
