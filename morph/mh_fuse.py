#!/usr/bin/env python3
"""方向 #2：多 horizon 分数融合（不改动生产入口）。

假设：h10/h15/h30 月度波动不同步，分数层平均可平抑尾部风险。
做法：每月对每个 h 用同一窗口各训一个模型，把三者在 test 月同一 close_time 上的
(p-0.5) 平均成融合分数，用 PRIMARY_H 的标签和赔率、走 h10_signal 的因果阈值口径评估。
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
from morph import h3_gate_eval as ge
from morph import kline_long as kl
from morph import h10_signal as hs
from morph.two_legs import add_target, fit, matrix

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

FUSE_H = [int(x) for x in os.environ.get("FUSE_H", "10,15,30").split(",")]
PRIMARY_H = int(os.environ.get("PRIMARY_H", "10"))
T = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T:7.1f}s] {msg}", flush=True)


def main() -> None:
    raw = kl.load_raw()
    df, feats = kl.build(raw)
    df, feats = hs.add_flow(df, feats)
    df = df.with_columns(pl.col("close_time").cast(pl.Datetime("ms")).dt.strftime("%Y-%m").alias("ym"))
    months = [m for m in sorted(df["ym"].unique().to_list())
              if hs.START_MONTH <= m <= hs.END_MONTH]
    log(f"FUSE_H={FUSE_H} PRIMARY_H={PRIMARY_H} | {len(months)} 窗 | {len(feats)} 列")

    tagged = {}
    for h in set(FUSE_H + [PRIMARY_H]):
        tagged[h] = add_target(df, h).filter(
            pl.col("target").is_not_null() & pl.col("close").is_not_null()).sort("close_time")
        log(f"  h={h} 可用 {tagged[h].height:,} 行")

    per_tpd = {t: dict(s=[], y=[], d=[], h=[], mon=[]) for t in hs.TPD_LIST}
    rows = []
    for m in months:
        m_start = int(dt.datetime.strptime(m + "-01", "%Y-%m-%d")
                      .replace(tzinfo=dt.timezone.utc).timestamp() * 1000)
        lo = m_start - hs.WIN_DAYS * 86_400_000
        scores: dict[int, dict[int, float]] = {}
        th_scores: dict[int, np.ndarray] = {}
        base_ct = None
        base_y = None
        for h in FUSE_H:
            t = tagged[h]
            tr_all = t.filter((pl.col("close_time") < m_start) & (pl.col("close_time") >= lo))
            te = t.filter(pl.col("ym") == m)
            if tr_all.height < hs.MIN_TRAIN or te.height < hs.MIN_TEST:
                log(f"[{m}] h={h} 样本不足，跳过该 h")
                continue
            c1 = int(tr_all.height * (1 - hs.ES_FRAC - hs.THR_FRAC))
            c2 = int(tr_all.height * (1 - hs.THR_FRAC))
            tr, es, th = tr_all[:c1], tr_all[c1:c2], tr_all[c2:]
            if hs.STRIDE > 1:
                tr = tr.gather_every(hs.STRIDE)
            Xtr, ytr = matrix(tr, feats), tr["target"].to_numpy().astype(int)
            Xes, yes = matrix(es, feats), es["target"].to_numpy().astype(int)
            pth = np.zeros(th.height)
            pte = np.zeros(te.height)
            for seed in hs.SEEDS:
                mdl = fit(Xtr, ytr, feats, Xes, yes, seed=seed, params_override=hs.LGB_PARAMS)
                pth += mdl.predict(Xth := matrix(th, feats), num_iteration=mdl.best_iteration) / len(hs.SEEDS)
                pte += mdl.predict(matrix(te, feats), num_iteration=mdl.best_iteration) / len(hs.SEEDS)
            th_scores[h] = pth - 0.5
            ct_h = te["close_time"].to_numpy()
            scores[h] = dict(zip(ct_h.tolist(), (pte - 0.5).tolist()))
            if h == PRIMARY_H:
                base_ct = ct_h
                base_y = te["target"].to_numpy().astype(np.int8)

        if base_ct is None:
            log(f"[{m}] PRIMARY_H={PRIMARY_H} 无 test，跳过")
            continue
        # 融合：公共 close_time 上平均可用 h 的分数
        common = [ct for ct in base_ct.tolist() if all(ct in scores[h] for h in FUSE_H)]
        if len(common) < hs.MIN_TEST:
            log(f"[{m}] 公共 test 行 {len(common)} 不足，跳过")
            continue
        s_fuse = np.array([np.mean([scores[h][ct] for h in FUSE_H]) for ct in common])
        y = np.array([base_y[list(base_ct).index(ct)] for ct in common], dtype=np.int8)
        ct_arr = np.array(common)
        days = ge.day_id(ct_arr)
        hours = ((ct_arr // 3_600_000) % 24).astype(np.int8)
        nd = len(np.unique(days))
        pth_fuse = np.mean([th_scores[h] for h in FUSE_H], axis=0)
        auc = float(roc_auc_score(y, s_fuse))
        line = f"[{m}] n={len(y)}({nd}天) AUC={auc:.4f}"
        for tpd in hs.TPD_LIST:
            thr = float(np.quantile(np.abs(pth_fuse), 1 - tpd / hs.BARS_PER_DAY))
            hit = np.abs(s_fuse) >= thr
            nb = int(hit.sum())
            if nb < 30:
                line += f" | tpd{tpd:g}:单量太少"
                continue
            w = ((s_fuse[hit] > 0).astype(int) == y[hit]).astype(np.int8)
            win = float(w.mean())
            per_tpd[tpd]["s"].append(s_fuse[hit]); per_tpd[tpd]["y"].append(y[hit])
            per_tpd[tpd]["d"].append(days[hit]); per_tpd[tpd]["h"].append(hours[hit])
            per_tpd[tpd]["mon"].append(win)
            if tpd == hs.PRIMARY_TPD:
                line += f" | 阈值={thr:.4f} {nb / nd:.1f}单/天 胜率={win * 100:.2f}%"
        rows.append(dict(ym=m, nd=nd))
        log(line)

    if not rows:
        raise SystemExit("无窗口完成")
    print(f"\n{'名义tpd':>8}{'实际单/天':>10}{'n':>7}{'胜率':>9}{'月min':>8}{'破50%月':>8}")
    nd_all = sum(r["nd"] for r in rows)
    for tpd in hs.TPD_LIST:
        p = per_tpd[tpd]
        if not p["y"]:
            continue
        yv = np.concatenate(p["y"]); sv = np.concatenate(p["s"])
        w = ((sv > 0).astype(int) == yv).astype(np.int8)
        win = float(w.mean()); mon = np.array(p["mon"])
        evs = " ".join(f"EV@{pay:g}={pay * win - (1 - win):+.3f}" for pay in hs.PAYOUTS)
        print(f"{tpd:>8.1f}{len(w) / nd_all:>10.1f}{len(w):>7d}{win * 100:>8.2f}%"
              f"{mon.min() * 100:>7.2f}%{int((mon < .5).sum()):>5}/{len(mon)}  {evs}")
    log("DONE")


if __name__ == "__main__":
    main()
