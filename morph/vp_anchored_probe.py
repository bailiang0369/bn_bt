#!/usr/bin/env python3
"""锚定当前价的 VP 特征：在 h10 **现役 base 口径**下的最小验证。

与旧 vp_probe 的区别
-------------------
- 特征换成 `vp_anchored.build_avp`：长回溯(2d/7d)、绝对价格分箱、以当前 close 定位桶，
  解决"当前价落在固定窗口边缘、看不到继续运行方向压力位"的口径问题。
- 用**现役 base 配置**（LGB_REG=base，num_leaves=63/min_child=100），不是 reg2；
  与 h10 63.32% 同一口径，可直接对撞。
- walk-forward + 因果阈值与 h10_signal 完全一致（train/es/thr 段切分，阈值只在 thr 段定）。

四个变体（同一数据/超参/seeds/splits，完全配对）：
  base   49 列现役特征
  av     10 列锚定 VP（判断它是否自带信息）
  both   49+10
  avadd  49 + 4 列局部不平衡/稀疏带（最可能非共线的两族）
先验低、一次性验证：要求 AUC 提升 >0.002 且配对 CI 不跨 0、6 组以上同向才采纳。

env: HORIZON WIN_DAYS SEEDS ROUNDS START_MONTH END_MONTH PRIMARY_TPD TPD_LIST
     AV_WINS AV_TICK AV_ANCHOR AV_CAP AV_SELFCHECK
"""
from __future__ import annotations

import datetime as dt
import os
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from morph import h3_gate_eval as ge  # noqa: E402
from morph import kline_long as kl  # noqa: E402
from morph.h10_signal import add_flow  # noqa: E402
from morph.two_legs import add_target, matrix  # noqa: E402
from morph.vp_anchored import avp_leak_check, build_avp, log  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

HORIZON = int(os.environ.get("HORIZON", "10"))
WIN_DAYS = int(os.environ.get("WIN_DAYS", "1460"))
SEEDS = [int(s) for s in os.environ.get("SEEDS", "42,43,44,45,46").split(",") if s.strip()]
ROUNDS = int(os.environ.get("ROUNDS", "2000"))
ES_FRAC = float(os.environ.get("ES_FRAC", "0.05"))
THR_FRAC = float(os.environ.get("THR_FRAC", "0.10"))
START_MONTH = os.environ.get("START_MONTH", "2026-01")
END_MONTH = os.environ.get("END_MONTH", "2026-10")
PAY = float(os.environ.get("PAY", "0.80"))
BARS_PER_DAY = int(os.environ.get("BARS_PER_DAY", "1440"))
TPD_LIST = [float(x) for x in os.environ.get("TPD_LIST", "2,5,8,15,30").split(",") if x.strip()]
PRIMARY_TPD = float(os.environ.get("PRIMARY_TPD", "8"))
AV_WINS = tuple(int(x) for x in os.environ.get("AV_WINS", "2880,10080").split(","))
AV_TICK = float(os.environ.get("AV_TICK", "5.0"))
AV_ANCHOR = int(os.environ.get("AV_ANCHOR", "120"))
AV_CAP = float(os.environ.get("AV_CAP", "300.0"))

# 现役 base 超参（与 h10sig 一致）
PARAMS = dict(objective="binary", metric="auc", boosting_type="gbdt",
              learning_rate=.05, num_leaves=63, feature_fraction=.9,
              bagging_fraction=.9, bagging_freq=1, min_child_samples=100,
              verbose=-1, num_threads=os.cpu_count() or 4)


def fit(Xtr, ytr, Xes, yes, seed):
    p = dict(PARAMS, seed=seed)
    ds = lgb.Dataset(Xtr, ytr)
    va = lgb.Dataset(Xes, yes, reference=ds)
    return lgb.train(p, ds, num_boost_round=ROUNDS, valid_sets=[ds, va],
                     callbacks=[lgb.early_stopping(50, verbose=False)])


def main() -> None:
    if os.environ.get("AV_SELFCHECK", "1") == "1":
        if not avp_leak_check(windows=(240, 960), anchor=AV_ANCHOR):
            raise SystemExit("[avp] 泄露自检失败，拒绝继续")

    raw = kl.load_raw()
    av, avcols = build_avp(raw, windows=AV_WINS, tick_bp=AV_TICK,
                           anchor=AV_ANCHOR, cap_bp=AV_CAP)
    log(f"锚定 VP {len(avcols)} 列：{avcols}")

    df, feats = kl.build(raw)
    df, feats = add_flow(df, feats)
    f_base = list(feats)

    # 按 close_time 对齐（AV 在原始行上算，标签表会砍掉尾部 H 行）
    av_pl = pl.DataFrame({c: av[:, i] for i, c in enumerate(avcols)}).with_columns(
        pl.Series("close_time", raw["close_time"].to_numpy()))
    df = add_target(df, HORIZON).filter(
        pl.col("target").is_not_null() & pl.col("close").is_not_null()).sort("close_time")
    df = df.join(av_pl, on="close_time", how="left")
    df = df.with_columns(
        pl.col("close_time").cast(pl.Datetime("ms")).dt.strftime("%Y-%m").alias("ym"))
    miss = int(df[avcols[0]].null_count())
    log(f"特征 base={len(f_base)} av={len(avcols)} | 可用 {df.height:,} 行 | "
        f"AV 未对齐 {miss:,} 行 | H={HORIZON} | 窗 {WIN_DAYS}d")

    f_imb = [c for c in avcols if c.startswith(("av_imb50_", "av_gap_"))]
    fsets = {
        "base": f_base,
        "av": avcols,
        "both": f_base + avcols,
        "avadd": f_base + f_imb,
    }
    feats_all = f_base + avcols
    X_all = matrix(df, feats_all)
    y_all = df["target"].to_numpy().astype(int)
    ct = df["close_time"].cast(pl.Int64).to_numpy()
    ym = df["ym"].to_numpy()

    months = [m for m in sorted(set(ym.tolist())) if START_MONTH <= m <= END_MONTH]
    splits = []
    for m in months:
        m_start = int(dt.datetime.strptime(m + "-01", "%Y-%m-%d")
                      .replace(tzinfo=dt.timezone.utc).timestamp() * 1000)
        lo = m_start - WIN_DAYS * 86_400_000
        i_tr = np.where((ct < m_start) & (ct >= lo))[0]
        i_te = np.where(ym == m)[0]
        if len(i_tr) < 100_000 or len(i_te) < 1_000:
            log(f"[{m}] 样本不足，跳过")
            continue
        h = len(i_tr)
        c1 = int(h * (1 - ES_FRAC - THR_FRAC))
        c2 = int(h * (1 - THR_FRAC))
        splits.append((m, i_tr[:c1], i_tr[c1:c2], i_tr[c2:], i_te))
    if not splits:
        raise SystemExit("无有效窗口")
    log(f"walk-forward {len(splits)} 窗口：{splits[0][0]}~{splits[-1][0]}")

    rows = []
    for name, cols in fsets.items():
        ci = [feats_all.index(c) for c in cols]
        X = X_all[:, ci]
        per = {t: dict(s=[], y=[], nd=[], mon=[]) for t in TPD_LIST}
        ps, ys = [], []
        for m, i_tr, i_es, i_th, i_te in splits:
            p_th = np.zeros(len(i_th))
            p_te = np.zeros(len(i_te))
            for seed in SEEDS:
                mdl = fit(X[i_tr], y_all[i_tr], X[i_es], y_all[i_es], seed)
                p_th += mdl.predict(X[i_th], num_iteration=mdl.best_iteration) / len(SEEDS)
                p_te += mdl.predict(X[i_te], num_iteration=mdl.best_iteration) / len(SEEDS)
            ps.append(p_te)
            ys.append(y_all[i_te])
            nd = len(np.unique(ct[i_te] // 86_400_000))
            s = p_te - .5
            # 主操作点的逐月胜率（配对用）
            thr_p = float(np.quantile(np.abs(p_th - .5), 1 - PRIMARY_TPD / BARS_PER_DAY))
            hit_p = np.abs(s) >= thr_p
            if hit_p.sum() >= 30:
                per[PRIMARY_TPD]["mon"].append(
                    float(((s[hit_p] > 0).astype(int) == y_all[i_te][hit_p]).mean()))
            for t in TPD_LIST:
                thr = float(np.quantile(np.abs(p_th - .5), 1 - t / BARS_PER_DAY))
                hit = np.abs(s) >= thr
                if hit.sum() < 30:
                    continue
                per[t]["s"].append(s[hit])
                per[t]["y"].append(y_all[i_te][hit])
                per[t]["nd"].append(nd)
        auc = float(roc_auc_score(np.concatenate(ys), np.concatenate(ps)))
        rows.append((name, auc, per, len(cols)))
        log(f"  {name:<7} 列数={len(cols):<3} OOS AUC={auc:.4f}")

    base = next(r for r in rows if r[0] == "base")
    print(f"\n{'变体':<8}{'列数':>5}{'AUC':>9}{'ΔAUC':>9}"
          + "".join(f"{'tpd' + format(t, 'g') + ' 单/天 胜率 EV':>26}" for t in TPD_LIST))
    for name, auc, per, nc in rows:
        cells = ""
        for t in TPD_LIST:
            if not per[t]["y"]:
                cells += f"{'-':>26}"
                continue
            s = np.concatenate(per[t]["s"])
            y = np.concatenate(per[t]["y"])
            nd = float(np.sum(per[t]["nd"]))
            wr = float(((s > 0).astype(int) == y).mean())
            cells += f"{len(y) / nd:>10.1f}{wr * 100:>6.2f}%{PAY * wr - (1 - wr):>+7.3f}"
        print(f"{name:<8}{nc:>5d}{auc:>9.4f}{auc - base[1]:>+9.4f}{cells}")

    # 主操作点逐月成对差值（判断是否 6 组以上同向）
    base_mon = np.array(base[2][PRIMARY_TPD]["mon"])
    if len(base_mon):
        print(f"\n[主操作点 tpd={PRIMARY_TPD:g}] 逐月 base vs 各变体（胜率 pp）：")
        for name, auc, per, nc in rows:
            mon = np.array(per[PRIMARY_TPD]["mon"])
            if len(mon) != len(base_mon):
                continue
            d = (mon - base_mon) * 100
            print(f"  {name:<7} Δ逐月 = " + " ".join(f"{x:+.1f}" for x in d)
                  + f" | 同向 {(d > 0).sum()}/{len(d)} 均值 {d.mean():+.2f}pp")
    log("DONE")


if __name__ == "__main__":
    main()
