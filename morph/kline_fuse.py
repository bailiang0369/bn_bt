#!/usr/bin/env python3
"""L2：长周期 h=15 / h=30 / 融合，按实盘验收判据输出。

用户判据（2026-11-09；滚动口径 2026-11-10 明确为「成交单」）：
  1) 总体胜率 >= 65%（稳妥），63% 只是可上实盘的门槛
  2) **滚动 ROLL_N 单胜率 >= 50%**（按成交单序列滚，不是 100 天/100 期；任何时点不能破）
  3) **月度胜率 >= 50%**（任何一个月都不能破）
  4) 标准差要小（大容易连续亏）

路线说明（2026-11-10 更正，曾误判后撤回）：`续接说明.md` §3「换长周期是降级」
只针对**订单簿挂单快照**在 h>=15 失效，不是长周期本身做不到。项目结论
（`融合实验记录.md` §6）是：长周期应抛开 orderbook，主力改用 K 线形态
（faiss 簇 / `morph_ts` CNN-TCN，§8 在 h15 上 1% 覆盖 0.6337）。
本文件用的是 55 列手工 K 线衍生特征 + LightGBM，主判据 @日均15单 56.70%，
**只说明这套特征集不够，不能否定长周期路线**。
h=3 订单簿是并行路线，现役自动信号见 `morph/h3_signal.py`；两者不可互相否定。

主判据口径（2026-11-10 修正）：`@日均15单` = 全 OOS 按 |score| 取前
TARGET_TPD × 天数 名。「日均」是总单数/天数的统计结果，每天实际几单不固定，
必须同时报单量分布（见 h3_gate_eval.daily_orders：日均是平均假象）。

严禁把「每天按日内排名取 topN」当达标依据：该口径要等当天所有分钟分数出来
才知道第 N 名的阈值，实盘无法执行，属于前视。2026-11-10 实测它把 56~57%
的真实水平虚高到 63~65%。只可用 REPORT_LOOKAHEAD=1 显式开启作对照。

配置：
  A h15   —— 用 t15 训练、t15 结算
  B fuse  —— (p15+p30)/2，t15 结算
  C h30   —— 用 t30 训练、t30 结算
三个配置每月都用 5 seed、2 年滚动窗、早停窗末 10%。

env: SEEDS / ROUNDS / WIN_DAYS / START_MONTH / END_MONTH / TOPN / ROLL_N / REPORT_LOOKAHEAD
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
# 滚动判据窗口：按「成交单」计，不是按天/按期数（用户 2026-11-10 明确）。
ROLL_N = int(os.environ.get("ROLL_N", "100"))
OUT = os.environ.get("OUT", "/kaggle/working/kline_fuse.csv")
T = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T:7.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------- 判据
def seq_from_idx(score, y, ct, days, idx):
    """把选中的下标按真实时间排序后打包；滚动窗口必须依赖时间顺序。"""
    idx = np.sort(np.asarray(idx))
    win = ((score[idx] > 0).astype(int) == y[idx]).astype(np.int8)
    return win, days[idx], ct[idx], score[idx], y[idx]


def global_topk_seq(score, y, ct, days, tpd=TARGET_TPD):
    """主判据 `@日均tpd单`：全 OOS 按 |score| 取前 tpd×天数 名。

    这是评估口径（precision@固定单量），不是逐日下单规则；每天几单不固定，
    「日均」只是平均，必须连同单量分布一起看。
    """
    k = min(int(tpd) * len(np.unique(days)), len(score))
    idx = np.argpartition(-np.abs(score), k - 1)[:k]
    return seq_from_idx(score, y, ct, days, idx)


def daily_topn_seq_lookahead(score, y, ct, days, n=TOPN):
    """【前视，不可用于达标判定】每天等当天全部分数出来后取 |score| 前 n 名。

    实盘无法执行：当天没结束就不知道第 n 名的阈值在哪。保留仅为复现历史
    数字、量化前视红利；须 REPORT_LOOKAHEAD=1 才会输出。
    """
    picks = []
    for d in np.unique(days):
        m = np.where(days == d)[0]
        picks.append(m[np.argsort(-np.abs(score[m]))[:min(n, len(m))]])
    return seq_from_idx(score, y, ct, days, np.concatenate(picks))


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


# ---------------------------------------------------------------- 失效段诊断
def _dstr(day: int) -> str:
    return str(np.datetime64(int(day), "D"))


def day_bg(ct, close):
    """逐日行情背景：log 收益 / 日内 RV / 日内振幅（判定失效是否与行情有关）。"""
    d = (ct // 86_400_000).astype(np.int64)
    ret, rv, rng = {}, {}, {}
    for u in np.unique(d):
        c = close[d == u]
        k = int(u)
        if len(c) > 2:
            lr = np.diff(np.log(c))
            ret[k], rv[k], rng[k] = float(lr.sum()), float(lr.std()), float(np.log(c.max() / c.min()))
        else:
            ret[k] = rv[k] = rng[k] = 0.0
    return ret, rv, rng


def diagnose(name, win, dd, tt, ss, ret, rv, rng, w=ROLL_N, k=8):
    """滚动 w 单最差窗口（去重叠）：起止日、窗口内行情、下单小时分布、置信度。"""
    r = roll_rate(win, w)
    picked: list[int] = []
    for o in np.argsort(r):
        o = int(o)
        if all(abs(o - q) >= w for q in picked):
            picked.append(o)
        if len(picked) >= k:
            break
    log(f"  [{name}] 滚动{w}单 最差 {len(picked)} 段：")
    out = []
    for o in sorted(picked):
        sw, sd, st = win[o:o + w], dd[o:o + w], tt[o:o + w]
        d0, d1 = int(sd.min()), int(sd.max())
        days = list(range(d0, d1 + 1))
        rd = sum(ret.get(x, 0.0) for x in days) * 100
        rvd = float(np.mean([rv.get(x, 0.0) for x in days])) * 100
        rgd = float(np.mean([rng.get(x, 0.0) for x in days])) * 100
        hr = (st // 3_600_000 % 24).astype(int)
        cnt = np.bincount(hr, minlength=24)
        top = ",".join(f"{h:02d}h:{cnt[h]}" for h in np.argsort(-cnt)[:4])
        log(f"    {_dstr(d0)}~{_dstr(d1)}（{len(np.unique(sd))}天）胜率={sw.mean() * 100:.1f}%"
            f" | BTC={rd:+.2f}% 日均RV={rvd:.2f}% 振幅={rgd:.2f}%"
            f" | 均|s|={np.abs(ss[o:o + w]).mean():.3f} | 小时={top}")
        out.append(dict(cfg=name, d0=_dstr(d0), d1=_dstr(d1), n=int(w),
                        rate=float(sw.mean()), btc=rd / 100, rv=rvd / 100,
                        rng=rgd / 100, abs_s=float(np.abs(ss[o:o + w]).mean())))
    # 小时维度整体胜率（看是否某时段系统性失效）
    hr_all = (tt // 3_600_000 % 24).astype(int)
    hrs = [(h, float(win[hr_all == h].mean()), int((hr_all == h).sum()))
           for h in range(24) if (hr_all == h).sum() >= 30]
    worst = sorted(hrs, key=lambda x: x[1])[:4]
    log(f"  [{name}] 小时胜率最差: " + ", ".join(f"{h:02d}h={r * 100:.1f}%(n={n})" for h, r, n in worst))
    return out


def report(name, win, days, ct, total_days=None):
    """total_days 必须是全 OOS 的自然日数；主判据下大量日子没单，
    若拿「有单天数」当分母会把日均算成几十单。"""
    tot = float(win.mean())
    n = len(win)
    se = float(np.sqrt(tot * (1 - tot) / n))
    r100 = roll_rate(win, ROLL_N) if n > ROLL_N else np.array([np.nan])
    md = monthly_rate(win, ct)
    mv = np.array(list(md.values()))
    log(f"--- {name} | n={n:,} 总体={tot * 100:.2f}%(SE {se * 100:.2f}pp)"
        f" | 门槛63%:{'过' if tot >= .63 else '未过'} 稳妥65%:{'过' if tot >= .65 else '未过'}")
    nd = int(total_days) if total_days else len(np.unique(days))
    _, cnt = np.unique(days, return_counts=True)
    log(f"    单量: 日均={n / nd:.2f}（{n}单/{nd}自然日）std={cnt.std():.2f} min={cnt.min()}"
        f" p10={np.percentile(cnt, 10):.0f} p50={np.median(cnt):.0f}"
        f" p90={np.percentile(cnt, 90):.0f} max={cnt.max()}"
        f" | 有单天数={len(cnt)}/{nd} 无单天数={nd - len(cnt)}（日均是平均假象）")
    log(f"    月度: min={mv.min() * 100:.2f}% p10={np.quantile(mv, .1) * 100:.2f}%"
        f" median={np.median(mv) * 100:.2f}% max={mv.max() * 100:.2f}%"
        f" std={mv.std() * 100:.2f}pp | 低于50%的月={int((mv < .5).sum())}/{len(mv)}"
        f" {'OK' if mv.min() >= .5 else '破线'}")
    log(f"    滚动{ROLL_N}单胜率: min={r100.min() * 100:.2f}% p5={np.quantile(r100, .05) * 100:.2f}%"
        f" median={np.median(r100) * 100:.2f}% | 破50%窗口占={(r100 < .5).mean() * 100:.1f}%"
        f" {'OK' if r100.min() >= .5 else '破线'}")
    log(f"    月度明细: " + ", ".join(f"{k}:{v * 100:.1f}%" for k, v in md.items()))
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
    log(f"窗口 {len(months)} 个（{months[0]}~{months[-1]}）| seeds={len(SEEDS)}"
        f" | 主判据=@日均{TARGET_TPD}单（全期top-K，非日内排名）")

    store = {"A_h15": [], "B_fuse": [], "C_h30": []}
    px_ct, px_c = [], []
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
        px_ct.append(ct)
        px_c.append(te["close"].to_numpy())
        store["A_h15"].append((ps["h15"], te["t15"].to_numpy(), days, ct))
        store["B_fuse"].append(((ps["h15"] + ps["h30"]) / 2, te["t15"].to_numpy(), days, ct))
        store["C_h30"].append((ps["h30"], te["t30"].to_numpy(), days, ct))
        log(f"[{m}] train={tr_all.height:,} test={te.height:,} | "
            f"AUC h15={roc_auc_score(te['t15'].to_numpy(), ps['h15']):.4f}"
            f" h30={roc_auc_score(te['t30'].to_numpy(), ps['h30']):.4f}")

    log("")
    ret, rv, rng = day_bg(np.concatenate(px_ct), np.concatenate(px_c))
    log(f"行情背景就绪：{len(ret)} 天 | 日均|收益|={np.mean([abs(v) for v in ret.values()]) * 100:.2f}%")
    # 主判据只有一个：全期 top-K（@日均15单）。日内排名口径默认不输出。
    lookahead = os.environ.get("REPORT_LOOKAHEAD", "0") == "1"
    total_days = len(np.unique(np.concatenate([x[2] for x in store["A_h15"]])))
    rows, diag, trades = [], [], []
    for key in ("A_h15", "B_fuse", "C_h30"):
        p = np.concatenate([x[0] for x in store[key]])
        y = np.concatenate([x[1] for x in store[key]]).astype(np.int8)
        d = np.concatenate([x[2] for x in store[key]])
        c = np.concatenate([x[3] for x in store[key]])
        s = p - .5
        win, dd, tt, ss, yy = global_topk_seq(s, y, c, d, TARGET_TPD)
        rows.append(report(key, win, dd, tt, total_days))
        diag += diagnose(key, win, dd, tt, ss, ret, rv, rng)
        trades.append(pl.DataFrame(dict(cfg=key, ct=tt, day=dd, score=ss, y=yy, win=win)))
        if lookahead:
            lw, ld, lt = daily_topn_seq_lookahead(s, y, c, d, TOPN)[:3]
            report(f"{key}_LOOKAHEAD[前视·禁止用于达标]", lw, ld, lt, total_days)
    if OUT:
        pl.DataFrame(rows).write_csv(OUT)
        base = str(Path(OUT).with_suffix(""))
        pl.concat(trades).write_csv(base + "_trades.csv")
        pl.DataFrame(diag).write_csv(base + "_diag.csv")
        # 全量 OOS 预测：以后换任何选单规则都能离线回放，不必重训
        pl.DataFrame(dict(
            ct=np.concatenate([x[3] for x in store["A_h15"]]),
            p_h15=np.concatenate([x[0] for x in store["A_h15"]]),
            p_h30=np.concatenate([x[0] for x in store["C_h30"]]),
            t15=np.concatenate([x[1] for x in store["A_h15"]]),
            t30=np.concatenate([x[1] for x in store["C_h30"]]),
        )).write_parquet(base + "_scores.parquet")
        log(f"汇总已落盘: {OUT} / {base}_trades.csv / {base}_diag.csv / {base}_scores.parquet")
    log("DONE")


if __name__ == "__main__":
    main()
