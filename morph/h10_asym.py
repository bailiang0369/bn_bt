#!/usr/bin/env python3
"""h=10：`asym`（涨/跌半方差）特征正式复核 —— 与 §33 的 63.32% 完全同口径（2026-11-15）。

为什么单独跑这个
----------------
`morph/h10_regime.py` 一遍对照 8 个配置后，7 个是噪声或负（分裂专模 −3.3~−9.8pp 且 AUC
同步下降，路线关闭）。唯一正向的是 `asym`：Δ胜率 +0.59pp、最差月 55.05%→61.04%。
但那次是 **2 seed + 5 窗口 + stride=10** 的探索口径，量级在噪声内，不能直接进基线。

本文件按**对外正式口径**复核（`续接说明.md` §35.3 定的规矩）：
1. 10 窗口（2026-01 ~ 2026-10），与 §33 的 63.32% 逐月对齐；
2. **5 seed**（42~46）概率平均，与 §33 相同种子；
3. **全量训练 stride=1**（探索期用的 stride=10 只是加速，正式数字用全量）；
4. 因果阈值（thr 段分位，无前视），主操作点预登记 TPD=8；
5. 判据不是"均值变好看了"，而是：**base 能否复现 63.32%** + 逐月配对差 + 按天 block
   bootstrap 的 Δ 置信区间（10 个月 ≈ 300 天，重采样天而不是单——标签重叠 10 分钟，
   按单重采样会把 CI 缩窄 ~3 倍，是虚高）。

特征（两档唯一差别就是 asym 那 8 列）
-------------------------------------
base：`kline_long.build()` 49 列（含 h10_signal 的 flow 3 族）
asym：+ `as_up/as_dn`{15,60,240} + `as_skew60`（不对称度）+ `as_lr60`（对数涨跌波动比）

env
---
  SYMBOL / HORIZON=10 / SEEDS=42,43,44,45,46 / ROUNDS=2000 / WIN_DAYS=1460
  ES_FRAC=0.05 / THR_FRAC=0.10 / START_MONTH / END_MONTH / TPD_LIST / PRIMARY_TPD=8
  ASYM=1 / FLOW=1 / BOOT=2000 / KLINE_FROM=2020
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
from morph.h10_regime import add_asym  # noqa: E402
from morph.h10_signal import add_flow  # noqa: E402
from morph.two_legs import add_target, fit, matrix  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

HORIZON = int(os.environ.get("HORIZON", "10"))
SEEDS = [int(s) for s in os.environ.get("SEEDS", "42,43,44,45,46").split(",") if s.strip()]
WIN_DAYS = int(os.environ.get("WIN_DAYS", "1460"))
ES_FRAC = float(os.environ.get("ES_FRAC", "0.05"))
THR_FRAC = float(os.environ.get("THR_FRAC", "0.10"))
START_MONTH = os.environ.get("START_MONTH", "2026-01")
END_MONTH = os.environ.get("END_MONTH", "2026-10")
TPD_LIST = [float(x) for x in os.environ.get("TPD_LIST", "2,5,8,15,30").split(",") if x.strip()]
PRIMARY_TPD = float(os.environ.get("PRIMARY_TPD", "8"))
SYMBOL = os.environ.get("SYMBOL", "BTCUSDT")
KLINE_FROM = int(os.environ.get("KLINE_FROM", "2020"))
WITH_ASYM = os.environ.get("ASYM", "1") == "1"
WITH_FLOW = os.environ.get("FLOW", "1") == "1"
BOOT = int(os.environ.get("BOOT", "2000"))
PAYOUTS = [0.80, 0.828]
BARS_PER_DAY = 1440
T = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T:7.1f}s] {msg}", flush=True)


def day_agg(days_hit: np.ndarray, wins: np.ndarray, nd: int) -> tuple[np.ndarray, np.ndarray]:
    """按天聚合成 (n_d, w_d)，供 block bootstrap 重采样天。"""
    idx = np.searchsorted(np.unique(days_hit), days_hit)
    n = np.bincount(idx, minlength=nd).astype(np.float64)
    w = np.bincount(idx, weights=wins, minlength=nd)
    return n, w


def boot_delta(na: np.ndarray, wa: np.ndarray, nb: np.ndarray, wb: np.ndarray,
               rng: np.random.Generator) -> np.ndarray:
    """按天重采样：返回 Δ胜率(=a−b) 的 bootstrap 分布。"""
    k = len(na)
    out = np.empty(BOOT)
    for i in range(BOOT):
        j = rng.integers(0, k, k)
        out[i] = (wa[j].sum() / max(na[j].sum(), 1)) - (wb[j].sum() / max(nb[j].sum(), 1))
    return out


def main() -> None:
    raw = kl.load_raw()
    df, feats0 = kl.build(raw)
    if WITH_FLOW:
        df, feats0 = add_flow(df, feats0)
    if WITH_ASYM:
        df, feats_a = add_asym(df, feats0)
    else:
        feats_a = feats0
    df = add_target(df, HORIZON).filter(
        pl.col("target").is_not_null() & pl.col("close").is_not_null()).sort("close_time")
    df = df.with_columns(
        pl.col("close_time").cast(pl.Datetime("ms")).dt.strftime("%Y-%m").alias("ym"))
    log(f"{SYMBOL} 特征 base={len(feats0)} asym={len(feats_a)} | H={HORIZON} "
        f"seeds={SEEDS} 滚动窗={WIN_DAYS}天 | {df['ym'].min()}~{df['ym'].max()}")

    months = [m for m in sorted(df["ym"].unique().to_list()) if START_MONTH <= m <= END_MONTH]
    cfgs = [("base", feats0)]
    if WITH_ASYM:
        cfgs.append(("asym", feats_a))
    log(f"walk-forward {len(months)} 窗口（{months[0]}~{months[-1]}）| 全量训练 stride=1 | "
        f"es={ES_FRAC:.0%} thr={THR_FRAC:.0%} | 主操作点 tpd={PRIMARY_TPD:g}")

    # 每配置每 tpd 档：月度胜率 + 按天 (n,w) 累计
    acc: dict[str, dict[float, dict]] = {
        n: {t: dict(mon=[], n=np.zeros(0), w=np.zeros(0)) for t in TPD_LIST} for n, _ in cfgs}
    rows, mos = [], []
    rng = np.random.default_rng(20261115)
    for m in months:
        m_start = int(dt.datetime.strptime(m + "-01", "%Y-%m-%d")
                      .replace(tzinfo=dt.timezone.utc).timestamp() * 1000)
        lo = m_start - WIN_DAYS * 86_400_000
        tr_all = df.filter((pl.col("close_time") < m_start) & (pl.col("close_time") >= lo))
        te = df.filter(pl.col("ym") == m)
        if tr_all.height < 500_000 or te.height < 1000:
            log(f"[{m}] 样本不足（{tr_all.height:,}/{te.height:,}），跳过")
            continue
        c1 = int(tr_all.height * (1 - ES_FRAC - THR_FRAC))
        c2 = int(tr_all.height * (1 - THR_FRAC))
        tr, es, th = tr_all[:c1], tr_all[c1:c2], tr_all[c2:]
        y = te["target"].to_numpy().astype(np.int8)
        days = ge.day_id(te["close_time"].to_numpy())
        nd = len(np.unique(days))
        yes = es["target"].to_numpy().astype(int)
        log(f"[{m}] train={tr.height:,} es={es.height:,} thr={th.height:,} test={te.height:,}({nd}天)")

        for name, fs in cfgs:
            Xtr = matrix(tr, fs)
            Xes = matrix(es, fs)
            Xth = matrix(th, fs)
            Xte = matrix(te, fs)
            pth = np.zeros(th.height)
            pte = np.zeros(te.height)
            its = []
            for seed in SEEDS:
                mdl = fit(Xtr, tr["target"].to_numpy().astype(int), fs, Xes, yes, seed=seed)
                its.append(int(mdl.best_iteration))
                pth += mdl.predict(Xth, num_iteration=mdl.best_iteration) / len(SEEDS)
                pte += mdl.predict(Xte, num_iteration=mdl.best_iteration) / len(SEEDS)
            del Xtr, Xes, Xth, Xte
            s = pte - .5
            auc = float(roc_auc_score(y, pte))
            line = f"  {name:6}iter={int(np.mean(its)):>4} AUC={auc:.4f}"
            for tpd in TPD_LIST:
                thr = float(np.quantile(np.abs(pth - .5), 1 - tpd / BARS_PER_DAY))
                hit = np.abs(s) >= thr
                nb = int(hit.sum())
                if nb < 30:
                    line += f" | tpd{tpd:g}:单量太少"
                    continue
                w = ((s[hit] > 0).astype(int) == y[hit]).astype(np.int8)
                win = float(w.mean())
                a = acc[name][tpd]
                a["mon"].append(win)
                n_d, w_d = day_agg(days[hit], w, nd)
                a["n"] = np.concatenate([a["n"], n_d])
                a["w"] = np.concatenate([a["w"], w_d])
                if tpd == PRIMARY_TPD:
                    line += f" | 阈值={thr:.4f} {nb / nd:.1f}单/天 胜率={win * 100:.2f}%"
            rows.append(dict(ym=m, cfg=name, auc=auc, iter=int(np.mean(its))))
            log(line)
        mos.append(m)

    if not rows:
        raise SystemExit("无窗口完成")
    t = pl.DataFrame(rows)
    for name, _ in cfgs:
        sub = t.filter(pl.col("cfg") == name)
        log(f"===== {name}: 窗口 {sub.height} | AUC均值 {sub['auc'].mean():.4f}")

    print(f"\n{'配置':6}{'名义tpd':>8}{'实际单/天':>10}{'n':>7}{'胜率':>9}{'SE*':>7}"
          f"{'EV@0.80':>9}{'EV@0.828':>10}{'月min':>8}{'月std':>7}"
          f"{'破50%月':>8}{'滚动10天min':>12}")
    for name, _ in cfgs:
        for tpd in TPD_LIST:
            a = acc[name][tpd]
            if not a["mon"]:
                continue
            n_tot, w_tot = a["n"].sum(), a["w"].sum()
            win = w_tot / max(n_tot, 1)
            mon = np.array(a["mon"])
            # 重叠标签：有效样本 = 单量/HORIZON
            se = float(np.sqrt(win * (1 - win) / max(n_tot / HORIZON, 2)))
            ev = "".join(f"{p * win - (1 - win):>+9.3f}" if i == 0
                         else f"{p * win - (1 - win):>+10.3f}" for i, p in enumerate(PAYOUTS))
            # 滚动 10 天最差：按天聚合的日均胜率（单量足够时等价于滚动 ~80 单）
            day_win = np.where(a["n"] > 0, a["w"] / np.maximum(a["n"], 1), np.nan)
            dw = day_win[np.isfinite(day_win)]
            k = 10
            rmin = float(np.convolve(dw, np.ones(k) / k, "valid").min()) if len(dw) >= k else float("nan")
            nd_tot = int((a["n"] > 0).sum())
            print(f"{name:6}{tpd:>8.1f}{n_tot / nd_tot:>10.1f}{int(n_tot):>7d}"
                  f"{win * 100:>8.2f}%{se * 100:>6.2f}%{ev}"
                  f"{mon.min() * 100:>7.2f}%{mon.std() * 100:>6.2f}pp"
                  f"{int((mon < .5).sum()):>5}/{len(mon)}{rmin * 100:>10.1f}%")

    if len(cfgs) > 1:
        a, b = acc["asym"], acc["base"]
        print(f"\n[asym − base] 主操作点 tpd={PRIMARY_TPD:g}")
        ma, mb = np.array(a[PRIMARY_TPD]["mon"]), np.array(b[PRIMARY_TPD]["mon"])
        d = ma - mb
        tstat = float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))) if len(d) > 2 else float("nan")
        print("  月度配对差(pp)：" + "  ".join(
            f"{m[2:]}:{(x - y) * 100:+.1f}"
            for m, x, y in zip(mos, ma, mb)))
        print(f"  月度均值 {d.mean() * 100:+.2f}pp  月std {d.std(ddof=1) * 100:.2f}pp  "
              f"t={tstat:+.2f}（n={len(d)} 月）")
        for tpd in TPD_LIST:
            na, wa = a[tpd]["n"], a[tpd]["w"]
            nb, wb = b[tpd]["n"], b[tpd]["w"]
            if not len(na) or not len(nb):
                continue
            star = " ← 主操作点" if tpd == PRIMARY_TPD else ""
            da = wa.sum() / max(na.sum(), 1) - wb.sum() / max(nb.sum(), 1)
            bs = boot_delta(na, wa, nb, wb, rng)
            lo_, hi_ = np.quantile(bs, [.025, .975])
            print(f"  tpd{tpd:>5.1f} 全期Δ={da * 100:>+5.2f}pp  "
                  f"按天bootstrap 95%CI[{lo_ * 100:+.2f},{hi_ * 100:+.2f}]  "
                  f"P(Δ>0)={float((bs > 0).mean()):.3f}{star}")
    log("DONE")


if __name__ == "__main__":
    main()
