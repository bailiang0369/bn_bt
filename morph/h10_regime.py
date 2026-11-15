#!/usr/bin/env python3
"""h=10：regime 分裂专模「重做版」——一遍同时对照 4 条路线（2026-11-15）。

背景
----
用户提出「先聚类 / 划分市场状态，再按状态训专模、预测时路由」，并指出旧实验
（`融合实验记录.md` §30 的 Regime 分裂专模）可能做得不够好——旧版是 **h=3 标签 +
盘口腿 15 个月**，硬规则（autocorr>0.05 或 bbexp>1.15）二分流，结果 +0.53pp（CI 跨 0）
与 −1.18/−1.03pp（负）。

本次重做的改进（旧版全都没有）
------------------------------
1. **腿换成 h=10 K 线 + 6.7 年历史**：旧版核心短板是盘口只有 15 个月，分流后各 30 万行。
   这里滚动 4 年窗口 × 1m K 线，stride=HORIZON 抽样（§35 已证等价且快 10 倍）。
2. **regime 定义贴近用户原话「缓涨急跌 / 暴涨阴跌」**：
   - `rv2` / `rv3`：60 分钟已实现波动分 2/3 档（活跃 vs 平静）
   - `er2`：60 分钟趋势效率比 |净变动|/路径长度分 2 档（趋势 vs 震荡）
   - `dv4`：方向（`lg_ret120` 中位分档）× 波动（`lg_rv60` 两档）2×2 —— 直接对应
     「缓涨 / 阴跌 / 暴涨 / 急跌」四象限
   全部阈值取**训练窗口分位数**，对 test 严格因果（旧版是固定常数规则）。
3. **四个对照组，把「regime 信息没用」和「硬切分砍样本砍坏了」分开**：
   - `base`：全部特征，不碰 regime
   - `cat_*`：regime 标签 one-hot 进**同一个**模型（不砍样本、无路由误差）
   - `asym`：涨/跌半方差不对称（「缓涨急跌」的直接量化）当特征
   - `split_*`：每个 expert 单独训练 + 按 regime 路由
4. **每个 expert 各自早停、各自定阈值**：分数分布随 regime 差别很大（活跃档 |p−0.5|
   天然更宽），共用阈值会把订单全挤到某一档，等于用一半样本换一半单量。
5. 主判据沿用 `h10_signal` 的因果阈值（thr 段分位，无前视），tpd=8；同时报 AUC（覆盖行）。

env
---
  SYMBOL / HORIZON=10 / SEEDS=42,43 / ROUNDS / WIN_DAYS=1460 / ES_FRAC / THR_FRAC
  WINDOWS=2026-06,... / STRIDE=10 / TPD=8 / KLINE_FROM=2020 / ASYM=1 / FLOW=1
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
from morph.h10_signal import add_flow  # noqa: E402
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
STRIDE = int(os.environ.get("STRIDE", "10"))
SYMBOL = os.environ.get("SYMBOL", "BTCUSDT")
KLINE_FROM = int(os.environ.get("KLINE_FROM", "2020"))
WITH_ASYM = os.environ.get("ASYM", "1") == "1"
WITH_FLOW = os.environ.get("FLOW", "1") == "1"
WINDOWS = [m.strip() for m in os.environ.get(
    "WINDOWS", "2026-06,2026-07,2026-08,2026-09,2026-10").split(",") if m.strip()]
MIN_EXPERT_ROWS = 20_000
MIN_EXPERT_ES = 2_000
BARS_PER_DAY = 1440
T = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T:7.1f}s] {msg}", flush=True)


def add_asym(df: pl.DataFrame, feats: list[str]) -> tuple[pl.DataFrame, list[str]]:
    """涨/跌半方差不对称（「缓涨急跌」的直接量化）。只用 past。"""
    c = pl.col("close").cast(pl.Float64)
    r1 = (c / c.shift(1)).log() * 1e4
    up2 = pl.when(r1 > 0).then(r1).otherwise(0.0).pow(2)
    dn2 = pl.when(r1 < 0).then(-r1).otherwise(0.0).pow(2)
    e = []
    for n in (15, 60, 240):
        e += [up2.rolling_mean(n, min_samples=max(2, n // 2)).sqrt().alias(f"as_up{n}"),
              dn2.rolling_mean(n, min_samples=max(2, n // 2)).sqrt().alias(f"as_dn{n}")]
    out = df.with_columns(e)
    # 上面新建的列不能在同一次 with_columns 里被引用，所以拆成第二步
    out = out.with_columns([
        ((pl.col("as_up60") - pl.col("as_dn60"))
         / (pl.col("as_up60") + pl.col("as_dn60") + 1e-9)).alias("as_skew60"),
        ((pl.col("as_up60") + 1e-9) / (pl.col("as_dn60") + 1e-9)).log().alias("as_lr60"),
    ])
    return out, feats + [x for x in out.columns if x.startswith("as_")]


def qmarks(v: np.ndarray, ks: list[float]) -> np.ndarray:
    return np.quantile(v[np.isfinite(v)], ks)


def bucket(v: np.ndarray, cuts: np.ndarray) -> np.ndarray:
    """按 cuts 分桶（0..len(cuts)）；nan 归到中位桶。"""
    v = np.asarray(v, dtype=np.float64)
    out = np.searchsorted(cuts, v, side="right")
    out[~np.isfinite(v)] = len(cuts) // 2
    return out


def regime_masks(spec: list[tuple[str, int]], frames: tuple[pl.DataFrame, ...],
                 ) -> tuple[list[np.ndarray], int]:
    """把 (列, 档数) 列表合成 1 维 regime id；阈值只取训练窗口（frames[0]）的分位。"""
    cuts = [qmarks(frames[0][c].to_numpy(), [k / n for k in range(1, n)])
            for c, n in spec]
    nreg = 1
    for _, n in spec:
        nreg *= n
    out = []
    for f in frames:
        b = np.zeros(f.height, dtype=np.int64)
        m = 1
        for (c, n), cu in zip(spec, cuts):
            b += bucket(f[c].to_numpy(), cu) * m
            m *= n
        out.append(b)
    return out, nreg


def run_cfg(name: str, feats: list[str], tr, es, th, te, Xte, y, days, nd, seed_list,
            spec: list[tuple[str, int]] | None = None, cat: bool = False) -> dict:
    """单配置：spec=None → 单模型；cat=True → regime one-hot 进同一模型；
    cat=False → 每个 regime 单独训练 + 路由（各自早停、各自定阈值）。"""
    Xtr_all = matrix(tr, feats)
    ytr_all = tr["target"].to_numpy().astype(int)
    Xes_all, yes = matrix(es, feats), es["target"].to_numpy().astype(int)
    Xth_all = matrix(th, feats)

    if spec:
        (b_tr, b_es, b_th, b_te), nreg = regime_masks(
            spec, (tr, es, th, te))
    else:
        nreg = 1
        b_tr, b_es = np.zeros(tr.height, np.int64), np.zeros(es.height, np.int64)
        b_th, b_te = np.zeros(th.height, np.int64), np.zeros(te.height, np.int64)

    if cat:
        # regime 作为 one-hot 特征进同一个模型（不砍样本）
        def oh(X, b):
            m = np.zeros((X.shape[0], nreg), dtype=np.float32)
            m[np.arange(X.shape[0]), b] = 1.0
            return np.hstack([X, m])
        Xtr_all, Xes_all = oh(Xtr_all, b_tr), oh(Xes_all, b_es)
        Xth_all, Xte = oh(Xth_all, b_th), oh(Xte, b_te)
        cols = feats + [f"_rg{i}" for i in range(nreg)]
    else:
        cols = feats

    single = cat or nreg == 1
    ok_tr = np.ones(tr.height, bool) if single else None
    ok_es = np.ones(es.height, bool) if single else None

    pth_s = np.zeros(th.height)
    pte_s = np.zeros(te.height)
    cnt_th = np.zeros(th.height)
    cnt_te = np.zeros(te.height)
    hit_idx: list[np.ndarray] = []
    used, nrows, its = 0, 0, []
    for k in ([0] if single else range(nreg)):
        mtr = ok_tr if ok_tr is not None else (b_tr == k)
        mes = ok_es if ok_es is not None else (b_es == k)
        mth = np.ones(th.height, bool) if single else (b_th == k)
        mte = np.ones(te.height, bool) if single else (b_te == k)
        if mtr.sum() < MIN_EXPERT_ROWS or mes.sum() < MIN_EXPERT_ES or mte.sum() < 200:
            continue
        Xa, ya = Xtr_all[mtr], ytr_all[mtr]
        Xb, yb = Xes_all[mes], yes[mes]
        pth_g = np.zeros(int(mth.sum()))
        pte_g = np.zeros(int(mte.sum()))
        for seed in seed_list:
            mdl = fit(Xa, ya, cols, Xb, yb, seed=seed)
            its.append(int(mdl.best_iteration))
            pth_g += mdl.predict(Xth_all[mth], num_iteration=mdl.best_iteration) / len(seed_list)
            pte_g += mdl.predict(Xte[mte], num_iteration=mdl.best_iteration) / len(seed_list)
        pth_s[mth] += pth_g
        pte_s[mte] += pte_g
        cnt_th[mth] += 1
        cnt_te[mte] += 1
        # 阈值按本 expert 自己的分数分布定（跨 regime 分布不同，共用阈值会偏单量）
        thr_k = float(np.quantile(np.abs(pth_g - .5), 1 - TPD / BARS_PER_DAY))
        sg = pte_g - .5
        sub = np.flatnonzero(mte)[np.abs(sg) >= thr_k]
        hit_idx.append(sub)
        used += 1
        nrows += int(mtr.sum())
    if used == 0:
        return dict(name=name, auc=float("nan"), win=float("nan"), tpd=0.0,
                    cov=0.0, rows=0, it=0)
    pte = np.where(cnt_te > 0, pte_s / np.maximum(cnt_te, 1), 0.5)
    cov = cnt_te > 0
    hit = np.zeros(te.height, bool)
    if hit_idx:
        hit[np.concatenate(hit_idx)] = True
    nb = int(hit.sum())
    win = float(((pte[hit] - .5 > 0).astype(int) == y[hit]).mean()) if nb >= 50 else float("nan")
    yv, pv = (y[cov], pte[cov]) if int(cov.sum()) < te.height else (y, pte)
    auc = float(roc_auc_score(yv, pv)) if len(np.unique(yv)) > 1 else float("nan")
    return dict(name=name, auc=auc, win=win, tpd=nb / nd, cov=float(cov.mean()),
                rows=nrows, it=int(np.mean(its)))


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
    log(f"{SYMBOL} 特征 base {len(feats0)} / +asym {len(feats_a)} | 窗口={WINDOWS} "
        f"seeds={len(SEEDS)} stride={STRIDE} tpd={TPD:g}")

    cfgs = [
        ("base", feats0, None, False),
        ("asym", feats_a, None, False),
        ("cat_rv2", feats0, [("lg_rv60", 2)], True),
        ("cat_dv4", feats0, [("lg_ret120", 2), ("lg_rv60", 2)], True),
        ("split_rv2", feats0, [("lg_rv60", 2)], False),
        ("split_rv3", feats0, [("lg_rv60", 3)], False),
        ("split_er2", feats0, [("lg_er60", 2)], False),
        ("split_dv4", feats0, [("lg_ret120", 2), ("lg_rv60", 2)], False),
    ]

    rows = []
    for m in WINDOWS:
        m_start = int(dt.datetime.strptime(m + "-01", "%Y-%m-%d")
                      .replace(tzinfo=dt.timezone.utc).timestamp() * 1000)
        lo = m_start - WIN_DAYS * 86_400_000
        tr_all = df.filter((pl.col("close_time") < m_start) & (pl.col("close_time") >= lo))
        te = df.filter(pl.col("ym") == m)
        if tr_all.height < 100_000 or te.height < 1000:
            log(f"[{m}] 样本不足，跳过")
            continue
        tr_all = tr_all.gather_every(STRIDE)
        c1 = int(tr_all.height * (1 - ES_FRAC - THR_FRAC))
        c2 = int(tr_all.height * (1 - THR_FRAC))
        tr, es, th = tr_all[:c1], tr_all[c1:c2], tr_all[c2:]
        y = te["target"].to_numpy().astype(np.int8)
        days = ge.day_id(te["close_time"].to_numpy())
        nd = len(np.unique(days))
        Xte0 = matrix(te, feats0)
        Xte_a = matrix(te, feats_a)
        log(f"[{m}] train={tr.height:,}(stride{STRIDE}) es={es.height:,} thr={th.height:,} "
            f"test={te.height:,}({nd}天)")

        print(f"{m:9}{'配置':<12}{'训练行':>9}{'iter':>6}{'AUC':>9}"
              f"{'覆盖':>7}{'单/天':>7}{'胜率':>8}", flush=True)
        for name, fs, spec, c in cfgs:
            Xt = Xte_a if fs is feats_a else Xte0
            r = run_cfg(name, fs, tr, es, th, te, Xt, y, days, nd, SEEDS, spec, c)
            r["ym"] = m
            rows.append(r)
            print(f"{m:9}{name:<12}{r['rows']:>9,}{r['it']:>6}{r['auc']:>9.4f}"
                  f"{r['cov'] * 100:>6.0f}%{r['tpd']:>7.1f}{r['win'] * 100:>7.2f}%",
                  flush=True)

    if not rows:
        raise SystemExit("无窗口完成")
    t = pl.DataFrame(rows)
    base = t.filter(pl.col("name") == "base")
    print(f"\n[汇总] 相对 base 的同窗口配对差（tpd={TPD:g}，n={base.height} 个月）")
    print(f"{'配置':<12}{'AUC':>9}{'ΔAUC':>9}{'胜率':>9}{'Δ胜率':>9}"
          f"{'月min胜率':>10}{'单/天':>7}{'覆盖':>7}")
    for name in t["name"].unique(maintain_order=True).to_list():
        sub = t.filter(pl.col("name") == name)
        j = sub.join(base, on="ym", suffix="_b")
        da = float((j["auc"] - j["auc_b"]).mean())
        dw = float((j["win"] - j["win_b"]).mean()) * 100
        print(f"{name:<12}{sub['auc'].mean():>9.4f}{da:>+9.4f}"
              f"{sub['win'].mean() * 100:>8.2f}%{dw:>+8.2f}pp"
              f"{sub['win'].min() * 100:>9.2f}%{sub['tpd'].mean():>7.1f}"
              f"{sub['cov'].mean() * 100:>6.0f}%")
    log("DONE")


if __name__ == "__main__":
    main()
