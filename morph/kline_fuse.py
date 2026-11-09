#!/usr/bin/env python3
"""L2：长周期 h=15 / h=30 / 融合，按实盘验收判据输出。

用户判据（2026-11-09）：
  1) 总体胜率 >= 65%（稳妥），63% 只是可上实盘的门槛
  2) **滚动 100 期胜率 >= 50%**（任何时点都不能破）
  3) **月度胜率 >= 50%**（任何一个月都不能破）
  4) 标准差要小（大容易连续亏）

口径：`每天固定 top15`（W2/L1 已证明长周期必须按日内排名，全局 top-K 会低估 6.8pp）。
配置：
  A h15   —— 用 t15 训练、t15 结算
  B fuse  —— (p15+p30)/2，t15 结算（两个 horizon 的错误模式不同）
  C h30   —— 用 t30 训练、t30 结算（若产品到期是 30 分钟）
三个配置每月都用 5 seed、2 年滚动窗、早停窗末 10%。

env: SEEDS / ROUNDS / WIN_DAYS / START_MONTH / END_MONTH / TOPN
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

try:
    from morph import h3_gate_eval as ge
    from morph import ob_kline as ok
    from morph.kline_long import build, load_raw
    from morph.two_legs import TARGET_TPD, add_target, fit, matrix
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from morph import h3_gate_eval as ge
    from morph import ob_kline as ok
    from morph.kline_long import build, load_raw
    from morph.two_legs import TARGET_TPD, add_target, fit, matrix

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

SEEDS = [int(s) for s in os.environ.get("SEEDS", "42,43,44,45,46").split(",") if s.strip()]
WIN_DAYS = int(os.environ.get("WIN_DAYS", "730"))
ES_FRAC = float(os.environ.get("ES_FRAC", "0.1"))
START_MONTH = os.environ.get("START_MONTH", "2026-01")
END_MONTH = os.environ.get("END_MONTH", "2026-10")
TOPN = int(os.environ.get("TOPN", "15"))
OUT = os.environ.get("OUT", "/kaggle/working/kline_fuse.csv")
T = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T:7.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------- 判据
def daily_topn_seq(score, y, ct, days, n=TOPN):
    """每天按 |score| 取前 n 单，按时间顺序返回 (是否获胜, 盈亏序列所属日, 时间)。"""
    win, dd, tt = [], [], []
    for d in np.unique(days):
        idx = np.where(days == d)[0]
        k = min(n, len(idx))
        pick = idx[np.argsort(-np.abs(score[idx]))[:k]]
        pick = pick[np.argsort(ct[pick])]
        win.append((score[pick] > 0).astype(int) == y[pick])
        dd.append(np.full(k, d))
        tt.append(ct[pick])
    return np.concatenate(win).astype(np.int8), np.concatenate(dd), np.concatenate(tt)


def roll_rate(seq, w):
    cs = np.concatenate([[0], np.cumsum(seq.astype(np.int64))])
    return (cs[w:] - cs[:-w]) / w


def max_streak(seq):
    best = cur = 0
    for v in seq:
        cur = cur + 1 if v == 0 else 0
        best = max(best, cur)
    return best


def monthly_rate(win, ct_ms):
    ym = (ct_ms // 1000).astype("datetime64[s]").astype("datetime64[M]")
    return {str(m): float(win[ym == m].mean()) for m in np.unique(ym)}


def daily_rates(win, days):
    return {int(d): float(win[days == d].mean()) for d in np.unique(days)}


def report(name, win, days, ct):
    tot = float(win.mean())
    n = len(win)
    se = float(np.sqrt(tot * (1 - tot) / n))
    r100 = roll_rate(win, 100) if n > 100 else np.array([np.nan])
    md = monthly_rate(win, ct)
    mv = np.array(list(md.values()))
    dr = daily_rates(win, days)
    dser = np.array([dr[d] for d in sorted(dr)])
    if len(dser) > 100:
        cs = np.concatenate([[0], np.cumsum(dser)])
        r100d = (cs[100:] - cs[:-100]) / 100
    else:
        r100d = np.array([np.nan])
    log(f"--- {name} | n={n:,} 总体={tot * 100:.2f}%(SE {se * 100:.2f}pp)"
        f" | 门槛63%:{'过' if tot >= .63 else '未过'} 稳妥65%:{'过' if tot >= .65 else '未过'}")
    log(f"    月度: min={mv.min() * 100:.2f}% p10={np.quantile(mv, .1) * 100:.2f}%"
        f" median={np.median(mv) * 100:.2f}% max={mv.max() * 100:.2f}%"
        f" std={mv.std() * 100:.2f}pp | 低于50%的月={int((mv < .5).sum())}/{len(mv)}"
        f" {'OK' if mv.min() >= .5 else '破线'}")
    log(f"    滚动100单胜率: min={r100.min() * 100:.2f}% p5={np.quantile(r100, .05) * 100:.2f}%"
        f" median={np.median(r100) * 100:.2f}% | 破50%窗口占={(r100 < .5).mean() * 100:.1f}%"
        f" {'OK' if r100.min() >= .5 else '破线'}")
    log(f"    月度明细: " + ", ".join(f"{k}:{v * 100:.1f}%" for k, v in md.items()))
    log(f"    滚动100天胜率: min={r100d.min() * 100:.2f}% p5={np.quantile(r100d, .05) * 100:.2f}%")
    log(f"    最大连续亏损: {max_streak(win)} 单")
    return dict(name=name, n=n, total=tot, se=se, month_min=float(mv.min()),
                month_std=float(mv.std()), roll100_min=float(r100.min()),
                roll100_p5=float(np.quantile(r100, .05)), streak=max_streak(win))


def main() -> None:
    raw = load_raw()
    df, feats = build(raw)
    df = add_target(df, 15).rename({"target": "t15"})
    df = add_target(df, 30).rename({"target": "t30"})
    df = df.filter(pl.col("t15").is_not_null() & pl.col("t30").is_not_null()
                   & pl.col("close").is_not_null()).sort("close_time")
    df = df.with_columns(
        pl.col("close_time").cast(pl.Datetime("ms")).dt.strftime("%Y-%m").alias("ym"))
    log(f"特征 {len(feats)} 列 | 可用 {df.height:,} 行 | {df['ym'].min()} ~ {df['ym'].max()}")

    months = [m for m in sorted(df["ym"].unique().to_list()) if START_MONTH <= m <= END_MONTH]
    log(f"窗口 {len(months)} 个（{months[0]}~{months[-1]}）| seeds={len(SEEDS)} | topn={TOPN}")

    store = {"A_h15": [], "B_fuse": [], "C_h30": []}
    for m in months:
        m_start = int(dt.datetime.strptime(m + "-01", "%Y-%m-%d")
                      .replace(tzinfo=dt.timezone.utc).timestamp() * 1000)
        tr_all = df.filter((pl.col("close_time") < m_start)
                           & (pl.col("close_time") >= m_start - WIN_DAYS * 86_400_000))
        te = df.filter(pl.col("ym") == m)
        if tr_all.height < 50_000 or te.height < 1000:
            log(f"[{m}] 样本不足，跳过")
            continue
        cut = int(tr_all.height * (1 - ES_FRAC))
        tr, es = tr_all[:cut], tr_all[cut:]
        Xtr, Xes, Xte = matrix(tr, feats), matrix(es, feats), matrix(te, feats)
        ps = {}
        for tag, col in (("h15", "t15"), ("h30", "t30")):
            ytr, yes = tr[col].to_numpy().astype(int), es[col].to_numpy().astype(int)
            arr = []
            for seed in SEEDS:
                arr.append(fit(Xtr, ytr, feats, Xes, yes, seed=seed).predict(Xte))
            ps[tag] = np.mean(arr, axis=0)
        days = ge.day_id(te["close_time"].to_numpy())
        ct = te["close_time"].to_numpy()
        store["A_h15"].append((ps["h15"], te["t15"].to_numpy(), days, ct))
        store["B_fuse"].append(((ps["h15"] + ps["h30"]) / 2, te["t15"].to_numpy(), days, ct))
        store["C_h30"].append((ps["h30"], te["t30"].to_numpy(), days, ct))
        log(f"[{m}] train={tr_all.height:,} test={te.height:,} | "
            f"AUC h15={roc_auc_score(te['t15'].to_numpy(), ps['h15']):.4f}"
            f" h30={roc_auc_score(te['t30'].to_numpy(), ps['h30']):.4f}")

    log("")
    rows = []
    for key in ("A_h15", "B_fuse", "C_h30"):
        p = np.concatenate([x[0] for x in store[key]])
        y = np.concatenate([x[1] for x in store[key]]).astype(np.int8)
        d = np.concatenate([x[2] for x in store[key]])
        c = np.concatenate([x[3] for x in store[key]])
        win, dd, tt = daily_topn_seq(p - .5, y, c, d, TOPN)
        rows.append(report(key, win, dd, tt))
    if OUT:
        pl.DataFrame(rows).write_csv(OUT)
        log(f"汇总已落盘: {OUT}")
    log("DONE")


if __name__ == "__main__":
    main()
