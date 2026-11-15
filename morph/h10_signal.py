#!/usr/bin/env python3
"""h=10 事件合约：**因果阈值**实盘口径（无前视），目标胜率 ≥60%。

为什么需要这个文件（与 kline_long.py 的区别）
--------------------------------------------
`kline_long.py` 的判据是 `@日均15单 = 全期 top-K`：要按 |score| 取全期前 K 名，
意味着必须知道**整个 OOS 期的分数分布**才能定阈值——这是前视，实盘不可执行。
`h3_signal.py` 已经把这个教训写死了（阈值只在校准段定），但那条纪律只落在 h=3 上。

本文件把同一纪律搬到 h=10：
1. 逐月 walk-forward，滚动窗 = WIN_DAYS 天（默认 1460）。
2. 窗内三段切：train / es / thr。**模型只在 train 上训练，早停用 es，
   阈值分位只在 thr 段算**（thr 是模型没训过的、且全部早于目标月 → 严格因果）。
3. 阈值 = thr 段 |p-0.5| 的 (1 - tpd/1440) 分位，之后整月固定不变。
4. 判据：逐月胜率分布 + 月度 min + 滚动 100 单最差 + 全期 EV。

为什么要收单量才可能到 60%
--------------------------
2026-11-15 实测（`h3bar`，单一切分但同口径）：h=10 K 线腿 @14.7 单/天 60.02%、
@9.9 单/天 60.56%、@4.7 单/天 61.71%。订单簿腿在 h=10 的曲线是**平**的（无弹性），
K 线腿有弹性——所以「挑高置信度」对 K 线腿是真有效的手段，对订单簿腿不是。
主操作点预登记为 PRIMARY_TPD=8（8 单/天，10 分钟持仓 ≈ 最多 3 个并发仓，可执行；
滚动 100 单判据约 12 天可算）。TPD_LIST 其余档只作敏感性，**不据此事后挑最优点**。

特征
----
`kline_long.build()` 的长周期特征（多尺度动量/均线结构/RV 期限结构/量价/位置/ER/季节性/
期现基差）+ 本文件追加的**合约 vs 现货主动成交失衡差**（外部 notebook 唯一没错的那条：
flow_divergence）。全部只用 past，无前视。

env
---
  SYMBOL / HORIZON=10 / SEEDS=42,... / ROUNDS / WIN_DAYS=1460 / ES_FRAC / THR_FRAC
  START_MONTH / END_MONTH / TPD_LIST / PRIMARY_TPD=8 / KLINE_FROM / PAYOUTS
  STRIDE=1（=HORIZON 抽样加速，正式数字保持 1）/ PLACEBO=0（=1 打乱标签查泄漏）
  PLACEBO_SEEDS=2 / BARS_PER_DAY=1440（阈值分位分母，与 1m bar 的机会数对齐）

换 horizon 用法（2026-11-15 起，h=30 摸底）
------------------------------------------
  HORIZON=30 WIN_DAYS=2400 STRIDE=30 TPD_LIST=1,2,4,8 PRIMARY_TPD=4
  PLACEBO=1 PAYOUTS=0.853,0.85
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
SEEDS = [int(s) for s in os.environ.get("SEEDS", "42,43,44,45,46").split(",") if s.strip()]
ROUNDS = int(os.environ.get("ROUNDS", "2000"))
WIN_DAYS = int(os.environ.get("WIN_DAYS", "1460"))
ES_FRAC = float(os.environ.get("ES_FRAC", "0.05"))
THR_FRAC = float(os.environ.get("THR_FRAC", "0.10"))
START_MONTH = os.environ.get("START_MONTH", "2026-01")
END_MONTH = os.environ.get("END_MONTH", "2026-10")
TPD_LIST = [float(x) for x in os.environ.get("TPD_LIST", "2,5,8,15,30").split(",") if x.strip()]
PRIMARY_TPD = float(os.environ.get("PRIMARY_TPD", "8"))
PAYOUTS = [float(x) for x in os.environ.get("PAYOUTS", "0.80,0.828").split(",") if x.strip()]
# 阈值分位分母：test 是 1m 行，所以 1440 = 每天 1440 个候选位置。
# 保持 1440 才与 h=10 的 63.32% 同口径（实盘合并同一段连续触发是另一回事，见 §36 之后的口径备忘）。
BARS_PER_DAY = int(os.environ.get("BARS_PER_DAY", "1440"))
# 标签重叠 HORIZON 根：stride=HORIZON 抽样已证等价（§35），只用于摸底/月度重训；
# 对外正式数字保持 STRIDE=1（全量）。
STRIDE = int(os.environ.get("STRIDE", "1"))
# 泄漏检查：训练+早停标签一起打乱，test AUC 必须回落到 0.50（§35.3 定的换 horizon 准入检查）
PLACEBO = os.environ.get("PLACEBO", "0") == "1"
PLACEBO_SEEDS = int(os.environ.get("PLACEBO_SEEDS", "2"))
# 结构位特征（前高前低/整数关口/日内开盘/影线/多周期共振），默认关闭保持基线不变
WITH_STRUCT = os.environ.get("STRUCT", "0") == "1"
# 时段筛选：白名单只用「训练窗 thr 段」的胜率定（严格因果），test 只在该时段下单
HOURFILT = os.environ.get("HOURFILT", "0") == "1"
HOUR_BLOCK = int(os.environ.get("HOUR_BLOCK", "3"))     # 每块几小时（24 除以它要是整数）
HOUR_MIN_WIN = float(os.environ.get("HOUR_MIN_WIN", "0.55"))  # 块胜率门槛
T = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T:7.1f}s] {msg}", flush=True)


def add_flow(df: pl.DataFrame, feats: list[str]) -> tuple[pl.DataFrame, list[str]]:
    """合约 vs 现货主动成交失衡差（外部 notebook 的 flow_divergence，因果实现）。"""
    c = df.columns
    need = ("um_buy_vol", "um_sell_vol", "spot_buy_vol", "spot_sell_vol")
    if not all(x in c for x in need):
        log(f"  跳过 flow 特征（缺列：{[x for x in need if x not in c]}）")
        return df, feats
    um = ((pl.col("um_buy_vol") - pl.col("um_sell_vol"))
          / (pl.col("um_buy_vol") + pl.col("um_sell_vol") + 1e-9))
    sp = ((pl.col("spot_buy_vol") - pl.col("spot_sell_vol"))
          / (pl.col("spot_buy_vol") + pl.col("spot_sell_vol") + 1e-9))
    e = [um.alias("lg_flow_um"), sp.alias("lg_flow_spot"),
         (um - sp).alias("lg_flow_div")]
    for n in (15, 60):
        e += [um.rolling_mean(n, min_samples=max(2, n // 2)).alias(f"lg_flow_um{n}"),
              (um - sp).rolling_mean(n, min_samples=max(2, n // 2)).alias(f"lg_flow_div{n}"),
              (um - sp).rolling_std(n, min_samples=max(2, n // 2)).alias(f"lg_flow_divstd{n}")]
    out = df.with_columns(e)
    new = [x for x in out.columns if x.startswith("lg_flow")]
    return out, feats + new


def add_struct(df: pl.DataFrame, feats: list[str]) -> tuple[pl.DataFrame, list[str]]:
    """结构位特征（直播类 30 分钟策略的核心信息，本仓库 49 列里完全没有）。

    现有 `lg_pos{n}` 是「区间内相对位置」，没有「距某个具体价位多少 bp」的概念；
    而 30 分钟级别的价格行为高度依赖前高前低 / 整数关口 / 日内开盘价。
    全部只用 past：rolling 之后 shift(1)，日开盘价是当日已知的第一根。
    """
    c = pl.col("close").cast(pl.Float64)
    e: list[pl.Expr] = []
    for n, tag in ((240, "4h"), (1440, "1d"), (10080, "7d")):
        hi = c.rolling_max(n, min_samples=max(2, n // 2)).shift(1)
        lo = c.rolling_min(n, min_samples=max(2, n // 2)).shift(1)
        e += [((hi - c) / c * 1e4).alias(f"st_dnhi_{tag}"),
              ((c - lo) / c * 1e4).alias(f"st_uplo_{tag}")]
    if {"open", "high", "low"} <= set(df.columns):
        o = pl.col("open").cast(pl.Float64)
        h = pl.col("high").cast(pl.Float64)
        lo_ = pl.col("low").cast(pl.Float64)
        body_hi = pl.max_horizontal(o, c)
        body_lo = pl.min_horizontal(o, c)
        e += [((h - body_hi) / c * 1e4).alias("st_upshadow"),
              ((body_lo - lo_) / c * 1e4).alias("st_dnshadow")]
    # 整数关口距离（BTC 按 1000、ETH 按 100）：距最近关口的带符号 bp
    step = 1000.0 if kl.SYMBOL.upper().startswith("BTC") else 100.0
    e.append(((((c / step) % 1.0) - 0.5) * step / c * 1e4).alias("st_round"))
    # 日内（UTC 0 点）开盘价距离
    day_open = c.first().over(pl.col("close_time") // 86_400_000)
    e.append(((c / day_open - 1) * 1e4).alias("st_dayopen"))
    # 多周期共振：5/15/60/240/1440 分钟动量同向的个数（-5..5）
    if "lg_ret1440" in df.columns:
        res = None
        for k in (5, 15, 60, 240, 1440):
            s = pl.col(f"lg_ret{k}").sign()
            res = s if res is None else res + s
        e.append(res.alias("st_reson5"))
    out = df.with_columns(e)
    new = [x for x in out.columns if x.startswith("st_")]
    return out, feats + new


def roll_worst(w: np.ndarray, n: int = 100) -> tuple[float, float]:
    """滚动 n 单最差胜率 / 破 50% 窗口占比（单序评估，最贴近实盘体感）。"""
    if len(w) < n:
        return float("nan"), float("nan")
    cs = np.concatenate([[0], np.cumsum(w)])
    r = (cs[n:] - cs[:-n]) / n
    return float(r.min()), float((r < .5).mean())


def main() -> None:
    raw = kl.load_raw()
    df, feats = kl.build(raw)
    df, feats = add_flow(df, feats)
    if WITH_STRUCT:
        df, feats = add_struct(df, feats)
    df = add_target(df, HORIZON).filter(
        pl.col("target").is_not_null() & pl.col("close").is_not_null()).sort("close_time")
    df = df.with_columns(
        pl.col("close_time").cast(pl.Datetime("ms")).dt.strftime("%Y-%m").alias("ym"))
    log(f"特征 {len(feats)} 列 | 可用 {df.height:,} 行 | {df['ym'].min()} ~ {df['ym'].max()}")

    months = [m for m in sorted(df["ym"].unique().to_list()) if START_MONTH <= m <= END_MONTH]
    log(f"walk-forward {len(months)} 窗口（{months[0]}~{months[-1]}）| H={HORIZON} | "
        f"滚动窗 {WIN_DAYS} 天 | es={ES_FRAC:.0%} thr={THR_FRAC:.0%} | seeds={len(SEEDS)}")
    log(f"因果阈值：分位只在「目标月之前的 thr 段」算，test 月只用固定阈值 → 无前视")

    per_tpd: dict[float, dict[str, list]] = {
        t: dict(s=[], y=[], d=[], h=[], mon=[]) for t in TPD_LIST}
    hf_s: list[np.ndarray] = []
    hf_y: list[np.ndarray] = []
    rows = []
    for m in months:
        m_start = int(dt.datetime.strptime(m + "-01", "%Y-%m-%d")
                      .replace(tzinfo=dt.timezone.utc).timestamp() * 1000)
        lo = m_start - WIN_DAYS * 86_400_000
        tr_all = df.filter((pl.col("close_time") < m_start) & (pl.col("close_time") >= lo))
        te = df.filter(pl.col("ym") == m)
        if tr_all.height < 100_000 or te.height < 1000:
            log(f"[{m}] 样本不足（{tr_all.height:,}/{te.height:,}），跳过")
            continue
        h_all = tr_all.height
        c1 = int(h_all * (1 - ES_FRAC - THR_FRAC))
        c2 = int(h_all * (1 - THR_FRAC))
        tr, es, th = tr_all[:c1], tr_all[c1:c2], tr_all[c2:]
        if STRIDE > 1:
            # 只对 train/es 抽样（标签重叠的冗余样本）；**thr 段必须全量**：
            # 阈值是「每 1m 行 1-tpd/1440 分位」，thr 一旦抽样，分位对应的粒度就变了，
            # 套到 1m 全量的 test 上命中率会被放大约 STRIDE 倍（h=30 实测名义 tpd=1
            # 变成 5.1 单/天，tpd=4 变成 7.1）。2026-11-15 修。
            tr, es = tr.gather_every(STRIDE), es.gather_every(STRIDE)
        Xtr, ytr = matrix(tr, feats), tr["target"].to_numpy().astype(int)
        Xes, yes = matrix(es, feats), es["target"].to_numpy().astype(int)
        Xth = matrix(th, feats)
        Xte = matrix(te, feats)
        pth = np.zeros(th.height)
        pte = np.zeros(te.height)
        its = []
        for seed in SEEDS:
            mdl = fit(Xtr, ytr, feats, Xes, yes, seed=seed)
            its.append(int(mdl.best_iteration))
            pth += mdl.predict(Xth, num_iteration=mdl.best_iteration) / len(SEEDS)
            pte += mdl.predict(Xte, num_iteration=mdl.best_iteration) / len(SEEDS)
        y = te["target"].to_numpy().astype(np.int8)
        ct_ms = te["close_time"].to_numpy()
        days = ge.day_id(ct_ms)
        hours = ((ct_ms // 3_600_000) % 24).astype(np.int8)
        nd = len(np.unique(days))
        pl_auc, pl_win = float("nan"), float("nan")
        if PLACEBO:
            # 泄漏检查：训练与早停标签一起打乱，test AUC 必须回落到 0.50
            rng = np.random.default_rng(20261115)
            ysh, yesh = ytr.copy(), yes.copy()
            rng.shuffle(ysh)
            rng.shuffle(yesh)
            npl = min(len(SEEDS), PLACEBO_SEEDS)
            pth_pl = np.zeros(th.height)
            pte_pl = np.zeros(te.height)
            for seed in SEEDS[:npl]:
                mdl = fit(Xtr, ysh, feats, Xes, yesh, seed=seed)
                pth_pl += mdl.predict(Xth, num_iteration=mdl.best_iteration) / npl
                pte_pl += mdl.predict(Xte, num_iteration=mdl.best_iteration) / npl
            pl_auc = float(roc_auc_score(y, pte_pl))
            thr_pl = float(np.quantile(np.abs(pth_pl - .5), 1 - PRIMARY_TPD / BARS_PER_DAY))
            hit_pl = np.abs(pte_pl - .5) >= thr_pl
            if int(hit_pl.sum()) >= 30:
                pl_win = float(((pte_pl[hit_pl] - .5 > 0).astype(int) == y[hit_pl]).mean())
            del pth_pl, pte_pl, hit_pl, ysh, yesh
        del Xtr, Xes, Xth, Xte
        s = pte - .5
        auc = float(roc_auc_score(y, pte))
        line = (f"[{m}] train={tr.height:,} es={es.height:,} thr={th.height:,} test={te.height:,}"
                f"({nd}天) iter={int(np.mean(its))} AUC={auc:.4f}")
        if PLACEBO:
            line += f" | placebo AUC={pl_auc:.4f} 胜率={pl_win * 100:.2f}%"
        # 【严格因果的时段筛选】白名单只用 thr 段（全部早于目标月）的块胜率定，
        # 绝不看 test。这是「直播类 15 单/天 62%」最可能的来源，必须单独验证。
        ok_blk: list[int] = []
        if HOURFILT:
            yth = th["target"].to_numpy().astype(np.int8)
            hth = ((th["close_time"].to_numpy() // 3_600_000) % 24).astype(np.int8)
            bth = (hth // HOUR_BLOCK).astype(np.int8)
            thr_p = float(np.quantile(np.abs(pth - .5), 1 - PRIMARY_TPD / BARS_PER_DAY))
            m_p = np.abs(pth - .5) >= thr_p
            parts = []
            for bi in range(24 // HOUR_BLOCK):
                mk = m_p & (bth == bi)
                if mk.sum() >= 30:
                    w_b = float(((pth[mk] - .5 > 0).astype(int) == yth[mk]).mean())
                    parts.append(f"{bi * HOUR_BLOCK:02d}-{(bi + 1) * HOUR_BLOCK - 1:02d}h"
                                 f":{w_b * 100:.0f}%({int(mk.sum())})")
                    if w_b >= HOUR_MIN_WIN:
                        ok_blk.append(bi)
            line += f"\n    波段(训练窗): " + " ".join(parts) + f" → 白名单 {ok_blk}"
        for tpd in TPD_LIST:
            thr = float(np.quantile(np.abs(pth - .5), 1 - tpd / BARS_PER_DAY))
            hit = np.abs(s) >= thr
            nb = int(hit.sum())
            if nb < 30:
                line += f" | tpd{tpd:g}:单量太少"
                continue
            w = ((s[hit] > 0).astype(int) == y[hit]).astype(np.int8)
            win = float(w.mean())
            per_tpd[tpd]["s"].append(s[hit])
            per_tpd[tpd]["y"].append(y[hit])
            per_tpd[tpd]["d"].append(days[hit])
            per_tpd[tpd]["h"].append(hours[hit])
            per_tpd[tpd]["mon"].append(win)
            if tpd == PRIMARY_TPD:
                line += f" | 阈值={thr:.4f} {nb / nd:.1f}单/天 胜率={win * 100:.2f}%"
                if ok_blk:
                    b_te = (hours // HOUR_BLOCK).astype(np.int8)
                    hit_f = hit & np.isin(b_te, ok_blk)
                    if int(hit_f.sum()) >= 30:
                        w_f = float(((s[hit_f] > 0).astype(int) == y[hit_f]).mean())
                        hf_s.append(s[hit_f])
                        hf_y.append(y[hit_f])
                        line += (f" | 时段筛选 {int(hit_f.sum()) / nd:.1f}单/天"
                                 f" 胜率={w_f * 100:.2f}%")
        rows.append(dict(ym=m, test=te.height, nd=nd, auc=auc, iter=int(np.mean(its)),
                         pl_auc=pl_auc))
        log(line)

    if not rows:
        raise SystemExit("无窗口完成")

    # 赔率档随 PAYOUTS 变化（h=30 是 0.853/0.85，不是 h=10 的 0.80/0.828）
    ev_hdr = "".join(f"{'EV@' + format(p, 'g'):>9}" if i == 0
                     else f"{'EV@' + format(p, 'g'):>10}" for i, p in enumerate(PAYOUTS))
    print(f"\n{'名义tpd':>8}{'实际单/天':>10}{'n':>7}{'胜率':>9}{'SE*':>7}{ev_hdr}"
          f"{'月min':>8}{'月std':>7}{'破50%月':>8}{'滚动100min':>11}{'破50%窗':>9}")
    for tpd in TPD_LIST:
        p = per_tpd[tpd]
        if not p["y"]:
            continue
        yv = np.concatenate(p["y"])
        sv = np.concatenate(p["s"])
        w = ((sv > 0).astype(int) == yv).astype(np.int8)
        win = float(w.mean())
        # 分母必须是**全部测试天数**，不是"有单的天数"：
        # 低频档（h=30 tpd=1，266 单只落在 43 天）用后者会把 1.8 单/天算成 6.2。
        nd = sum(r["nd"] for r in rows)
        mon = np.array(p["mon"])
        se = float(np.sqrt(win * (1 - win) / max(len(w) / HORIZON, 2)))
        rmin, rbad = roll_worst(w)
        evs = "".join(f"{pay * win - (1 - win):>+9.3f}" if i == 0
                      else f"{pay * win - (1 - win):>+10.3f}" for i, pay in enumerate(PAYOUTS))
        star = " ← 主操作点" if tpd == PRIMARY_TPD else ""
        print(f"{tpd:>8.1f}{len(w) / nd:>10.1f}{len(w):>7d}{win * 100:>8.2f}%{se * 100:>6.2f}%"
              f"{evs}{mon.min() * 100:>7.2f}%{mon.std() * 100:>6.2f}pp"
              f"{int((mon < .5).sum()):>5}/{len(mon)}{rmin * 100:>10.1f}%{rbad * 100:>8.1f}%{star}")

    p = per_tpd[PRIMARY_TPD]
    if p["mon"]:
        print(f"\n[主操作点 tpd={PRIMARY_TPD:g}] 逐月胜率：")
        print("  " + "  ".join(f"{r['ym'][2:]}:{m * 100:.1f}%"
                              for r, m in zip(rows, p["mon"])))
    if p["h"]:
        # 时段诊断（零训练成本）：若胜率集中在少数 UTC 小时，说明「全天候摊薄」是差距来源之一
        hv = np.concatenate(p["h"])
        yv2 = np.concatenate(p["y"])
        sv2 = np.concatenate(p["s"])
        wv = ((sv2 > 0).astype(int) == yv2).astype(np.int8)
        print(f"\n[主操作点 tpd={PRIMARY_TPD:g}] 按 UTC 小时：单量/胜率")
        cells = []
        for hh in range(24):
            mk = hv == hh
            n_h = int(mk.sum())
            if n_h >= 20:
                cells.append(f"{hh:02d}h:{n_h:>4} {wv[mk].mean() * 100:>5.1f}%")
            else:
                cells.append(f"{hh:02d}h:{n_h:>4}   —  ")
        for i in range(0, 24, 6):
            print("  " + " | ".join(cells[i:i + 6]))
    if hf_y:
        sv3 = np.concatenate(hf_s)
        yv3 = np.concatenate(hf_y)
        w3 = ((sv3 > 0).astype(int) == yv3)
        win3 = float(w3.mean())
        nd_all = sum(r["nd"] for r in rows)
        ev3 = "".join(f"{p * win3 - (1 - win3):>+9.3f}" for p in PAYOUTS)
        print(f"\n[时段筛选 tpd={PRIMARY_TPD:g}] n={len(w3)} 单/天={len(w3) / nd_all:.1f} "
              f"胜率={win3 * 100:.2f}% EV={ev3}  ← 白名单由训练窗定，无前视")
    log("DONE")


if __name__ == "__main__":
    main()
