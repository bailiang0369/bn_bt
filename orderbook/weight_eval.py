#!/usr/bin/env python3
"""固定单量/固定口径不变，只改「每单下注多少」：按概率加权能否提升夏普？

⚠️ 两个必须钉死的前提（2026-11-13，用户指出前视后重写）
--------------------------------------------------------
1. **禁止日内排名**。当日 top15 需要当天全部 bar 的打分排序，实盘取不到。
   `续接说明.md` §11 明确：「每天固定 top15」是日内排名前视，实盘不可执行，
   实测把 56~57% 虚高到 63~65%，**禁用**；66.94% 是「全局 top15×天数」更前视。
   实盘口径只有两种：逐 bar 固定阈值（现役 `morph/h3_signal.py`）、
   或用户说的「用之前 24 小时」定阈值。
2. **现役基线不是每天 15 单**。`|p-0.5| >= 0.05806`（val 前 38 天 98.96 分位）
   在 test 上是日均 40.6 单、62.85%、每注 EV@0.8=+0.131。单量随分布漂移是真实表现
   （§11：不要试图把单量控成每天 15 单）。「15 单/天」只能当对照口径。

本脚本并列四种选单口径，只有前三种可用于实盘：
  A  现役阈值        |p−0.5| >= THR，逐 bar 独立        —— 主口径（GO/STOP 只看它）
  B  过去24h分位     THR_d 由**前一日** |p−0.5| 分布定，当天触发满 15 单即停
  C  固定节拍15单    每天 15 个固定时刻无条件下        —— 零选择下限
  D  当日top15       当天 |p−0.5| 前 15                  —— 前视，仅用于量化虚高幅度

仓位分配（本脚本真正要回答的）
------------------------------
同样这批单里，q=0.8 的样本是否该比 q=0.6 下注更多？只有 q 校准后才有正解。
  eq    每单 1.0
  lin   |q − 0.5|
  kelly clip(q·b − (1−q), 0)，b = PAYOUT = 0.8（盈亏平衡 55.56%）
权重统一按**当日均值 = 1** 归一：当日总投入与等权完全相同，只隔离「分配」这一个变量
（若按日总和=1 归一，会把「当天单量多少」的暴露一并改掉，答非所问）。
q = 所选方向的胜率估计：raw 用 p（方向为跌时取 1−p），iso 用 val 段 isotonic 校准后的 p。

预先写死的判据（不因结果调整）
------------------------------
  GO   主口径 A 下某加权方案的日收益夏普 ≥ 等权 × 1.2，
       且最大回撤不比等权更深，且有效注数 1/Σŵ² ≥ 5；B 口径方向需一致，否则 WARN。
  STOP 否则 → 保持等权。

数据（2026-11-13：Kaggle 数据集 byh850515/btcusdt-orderbook-features 已 403 不可挂载）：
  /kaggle/input 挂载 → GitHub Release merged-features/BTCUSDT.parquet → 同 Release 月度分片
"""
from __future__ import annotations

import datetime as dt
import glob
import os
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import lightgbm as lgb  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from orderbook.znorm_eval import BASE, H, complete_days, day_index, log  # noqa: E402

GH_REPO = "bailiang0369/bn_hisdata"
GH_API = "https://api.github.com"
GH_TAG = os.environ.get("GH_TAG", "merged-features")
NAME = "BTCUSDT.parquet"
CACHE = Path("/tmp/gh_merged")

PAYOUT = 0.8
TPD = int(os.environ.get("TPD", "15"))
SEEDS = [int(s) for s in os.environ.get("SEEDS", "42,43,44,45,46").split(",") if s.strip()]
ROUNDS = int(os.environ.get("ROUNDS", "2000"))
MIN_NEFF = float(os.environ.get("MIN_NEFF", "5.0"))
SHARPE_GAIN = float(os.environ.get("SHARPE_GAIN", "1.2"))
BOOT = int(os.environ.get("BOOT", "2000"))
DAY_MS = 86_400_000
BAR_MS = 60_000
BARS_PER_DAY = 1440


# ---------------------------------------------------------------- 取数
def _assets(tag: str) -> dict[str, dict]:
    """匿名配额 60/h，被限流时返回空 dict（调用方改用直链）。"""
    h = {"Accept": "application/vnd.github+json"}
    tok = os.environ.get("GH_TOKEN")
    if tok:
        h["Authorization"] = f"Bearer {tok}"
    for i in range(3):
        try:
            r = requests.get(f"{GH_API}/repos/{GH_REPO}/releases/tags/{tag}",
                             headers=h, timeout=60)
            if r.status_code == 200:
                out = {a["name"]: a for a in r.json()["assets"]}
                log(f"[gh] {tag}: {len(out)} 个资产")
                return out
            log(f"[gh] {tag} HTTP {r.status_code}（第{i+1}次）")
        except Exception as e:
            log(f"[gh] {tag} 第{i+1}次失败: {e}")
        time.sleep(3 * 2 ** i)
    log("[gh] 列表接口不可用（多半是限流），改走确定性直链")
    return {}


def _stream(url: str, name: str, size: int | None = None, h: dict | None = None) -> Path:
    """Range 续传下载到 /tmp 缓存；size 未知时用 Content-Length 判定完成。"""
    CACHE.mkdir(parents=True, exist_ok=True)
    dst = CACHE / name
    if size and dst.exists() and dst.stat().st_size == size:
        return dst
    part = dst.with_name(dst.name + ".part")
    for att in range(1, 9):
        have = part.stat().st_size if part.exists() else 0
        if size and have >= size:
            break
        hh = dict(h or {})
        if have:
            hh["Range"] = f"bytes={have}-"
        try:
            with requests.get(url, headers=hh, stream=True, timeout=900,
                              allow_redirects=True) as r:
                if r.status_code not in (200, 206):
                    raise RuntimeError(f"HTTP {r.status_code}")
                cl = int(r.headers.get("Content-Length") or 0)
                total = size or (have + cl if r.status_code == 206 else cl)
                mode = "ab" if (have and r.status_code == 206) else "wb"
                with open(part, mode) as f:
                    for chunk in r.iter_content(1 << 22):
                        f.write(chunk)
            if total and part.exists() and part.stat().st_size == total:
                part.replace(dst)
                return dst
        except Exception as e:
            got = part.stat().st_size if part.exists() else 0
            log(f"[dl] {name} 第{att}次中断: {e}（已 {got / 1e6:.0f}MB）")
            time.sleep(min(30, 3 * att))
    if not part.exists() or (size and part.stat().st_size != size):
        raise SystemExit(f"下载 {name} 未完成")
    part.replace(dst)
    return dst


def _gh_get(tag: str, name: str, meta: dict | None) -> Path:
    if meta:
        h = {"Accept": "application/octet-stream"}
        tok = os.environ.get("GH_TOKEN")
        if tok:
            h["Authorization"] = f"Bearer {tok}"
        url = f"{GH_API}/repos/{GH_REPO}/releases/assets/{meta['id']}"
        log(f"[dl] {name} {int(meta['size']) / 1e6:.0f}MB（asset API）")
        try:
            return _stream(url, name, int(meta["size"]), h)
        except SystemExit:
            log("[dl] asset API 失败，退直链")
    log(f"[dl] {name}（Release 直链）")
    return _stream(f"https://github.com/{GH_REPO}/releases/download/{tag}/{name}", name)


def load_btcusdt() -> tuple[pl.DataFrame, str]:
    hits = sorted(glob.glob("/kaggle/input/**/BTCUSDT.parquet", recursive=True))
    if hits:
        log(f"[data] 挂载 {hits[0]}")
        return pl.read_parquet(hits[0]), "kaggle-dataset"

    log("[data] /kaggle/input 无 BTCUSDT.parquet（数据集已不可用）→ GitHub merged-features")
    a = _assets(GH_TAG)
    if NAME in a or not a:
        p = _gh_get(GH_TAG, NAME, a.get(NAME))
        return pl.read_parquet(p), f"gh:{GH_TAG}/{NAME}"
    months = sorted(n for n in a
                    if n.startswith("merged_BTCUSDT_") and n.endswith(".parquet"))
    if not months:
        raise SystemExit(f"{GH_TAG} 内既无 {NAME} 也无月度分片；资产={sorted(a)[:20]}")
    log(f"[data] 月度分片 {len(months)} 个，拼接")
    df = pl.concat([pl.read_parquet(_gh_get(GH_TAG, n, a[n])) for n in months],
                   how="vertical_relaxed").unique(subset=["close_time"], keep="last")
    return df.sort("close_time"), f"gh:{GH_TAG}/monthly x{len(months)}"


# ---------------------------------------------------------------- 绩效与权重
def perf(a: np.ndarray) -> dict:
    a = np.asarray(a, dtype=np.float64)
    if a.size == 0:
        return dict(n=0, total=0.0, mean=0.0, sd=0.0, sharpe=0.0, mdd=0.0, worst=0.0)
    eq = np.cumsum(a)
    peak = np.maximum.accumulate(np.concatenate([[0.0], eq]))[1:]
    sd = float(a.std(ddof=1)) if a.size > 1 else 0.0
    return dict(n=int(a.size), total=float(a.sum()), mean=float(a.mean()), sd=sd,
                sharpe=float(a.mean() / sd * np.sqrt(365)) if sd > 0 else 0.0,
                mdd=float((eq - peak).min()), worst=float(a.min()))


def w_eq(q: np.ndarray) -> np.ndarray:
    return np.ones_like(q)


def w_lin(q: np.ndarray) -> np.ndarray:
    return np.abs(q - 0.5)


def w_kelly(q: np.ndarray) -> np.ndarray:
    return np.clip(q * PAYOUT - (1.0 - q), 0.0, None)


SCHEMES = {"eq": w_eq, "lin": w_lin, "kelly": w_kelly}


def unit_mean(c: np.ndarray) -> tuple[np.ndarray, float]:
    """当日均值归一到 1（总投入与等权相同），同时返回有效注数。"""
    m = float(c.mean())
    if m <= 1e-12:
        w = np.ones_like(c)
    else:
        w = c / m
    n_eff = float(w.sum() ** 2 / np.sum(w ** 2))
    return w, n_eff


def boot_p(base: np.ndarray, alt: np.ndarray, n: int, seed: int = 0) -> float:
    if n <= 0 or base.size < 5:
        return float("nan")
    d = np.asarray(alt) - np.asarray(base)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, (n, d.size))
    return float((d[idx].mean(1) <= 0).mean())


# ---------------------------------------------------------------- 四种选单口径
def thr_from(p: np.ndarray, mask: np.ndarray, tpd: int = TPD) -> float:
    """只吃校准段 |p-0.5| 的 (1-TPD/1440) 分位，与 h3_signal.calibrate 同口径。"""
    return float(np.quantile(np.abs(p[mask] - 0.5), 1 - tpd / BARS_PER_DAY))


def bets_threshold(p: np.ndarray, day: np.ndarray, thr: float) -> dict[int, np.ndarray]:
    """A：逐 bar 独立判断，无日内排名（现役实盘口径）。"""
    hit = np.abs(p - 0.5) >= thr
    return {int(d): np.where((day == d) & hit)[0] for d in np.unique(day)}


def bets_roll24h(p: np.ndarray, day: np.ndarray, k: int) -> dict[int, np.ndarray]:
    """B：阈值只用**当天之前 24 小时**的 |p-0.5| 分布；当天按时间顺序触发，满 k 单即停。"""
    out, hist = {}, []
    for d in np.unique(day):
        idx = np.where(day == d)[0]            # ct 已排序 → idx 天然按时间
        if hist:
            past = hist[-1]
            thr = float(np.quantile(past, 1.0 - k / past.size))
            out[int(d)] = idx[np.abs(p[idx] - 0.5) >= thr][:k]
        else:
            out[int(d)] = idx[:0]              # 冷启动不下单
        hist.append(np.abs(p[idx] - 0.5))
    return out


def _fixed_slots(mod: np.ndarray, k: int) -> np.ndarray:
    """当天 00:00 起每 1440/k 分钟一个时刻，取该时刻之后第一根可用 bar。"""
    step = BARS_PER_DAY // k
    sel, used = [], -1
    for t in range(0, BARS_PER_DAY, step):
        j = int(np.searchsorted(mod, t))
        while j < mod.size and j <= used:
            j += 1
        if j >= mod.size:
            break
        used = j
        sel.append(j)
    return np.asarray(sel, dtype=int)


def bets_fixed15(day: np.ndarray, ct: np.ndarray, k: int) -> dict[int, np.ndarray]:
    """C：零选择，固定节拍。"""
    mod_all = (ct % DAY_MS) // BAR_MS
    return {int(d): np.where(day == d)[0][_fixed_slots(mod_all[day == d], k)]
            for d in np.unique(day)}


def bets_lookahead(p: np.ndarray, day: np.ndarray, k: int) -> dict[int, np.ndarray]:
    """D：当天 |p-0.5| 前 k —— 前视，仅作对照。"""
    out = {}
    for d in np.unique(day):
        idx = np.where(day == d)[0]
        out[int(d)] = idx[np.argsort(-np.abs(p[idx] - 0.5))[:k]]
    return out


# ---------------------------------------------------------------- 模拟
def run_protocol(bets: dict[int, np.ndarray], keep: set[int], y: np.ndarray,
                 p: np.ndarray, iso_p: np.ndarray, ct: np.ndarray) -> dict:
    pnl = {s: [] for s in ("eq", "lin", "kelly", "lin_iso", "kelly_iso")}
    neff = {s: [] for s in pnl}
    win_seq, nbet, days_used = [], 0, []
    for d in sorted(bets):
        if d not in keep:
            continue
        sel = bets[d]
        if sel.size == 0:
            continue
        days_used.append(int(d))
        dirs = p[sel] > 0.5
        win = dirs == (y[sel] == 1)
        q_raw = np.where(dirs, p[sel], 1.0 - p[sel])
        q_iso = np.where(dirs, iso_p[sel], 1.0 - iso_p[sel])
        for name in pnl:
            q = q_iso if name.endswith("_iso") else q_raw
            c = SCHEMES[name.split("_")[0]](q)
            w, ne = unit_mean(c)
            pnl[name].append(float(np.sum(w * np.where(win, PAYOUT, -1.0))))
            neff[name].append(ne)
        win_seq.append(win.astype(np.int8))
        nbet += int(sel.size)
    seq = np.concatenate(win_seq) if win_seq else np.zeros(0, dtype=np.int8)
    roll = None
    if seq.size >= 100:
        cs = np.concatenate([[0], np.cumsum(seq)])
        roll = (cs[100:] - cs[:-100]) / 100.0
    streak = cur = 0
    for v in seq:
        cur = cur + 1 if v == 0 else 0
        streak = max(streak, cur)
    return dict(pnl={s: np.asarray(v) for s, v in pnl.items()},
                neff={s: float(np.mean(v)) for s, v in neff.items()},
                wins=int(seq.sum()) if seq.size else 0, nbet=nbet, days_used=days_used,
                roll_min=float(roll.min()) if roll is not None and roll.size else float("nan"),
                roll_bad=float((roll < 0.5).mean()) if roll is not None and roll.size else float("nan"),
                streak=int(streak))


def main() -> int:
    raw, src = load_btcusdt()
    df = raw.sort("close_time").with_columns(pl.col("close").shift(-H).alias("_fc"))
    df = (df.with_columns((pl.col("_fc") > pl.col("close")).cast(pl.Int8).alias("target"))
          .drop("_fc").filter(pl.col("target").is_not_null()).drop_nulls())
    if src != "kaggle-dataset":
        log("[warn] 走 GitHub 回退：列集可能不含 stoch/supertrend，"
            "绝对胜率与文档 62.85%/66.94% 不可混比")
    L = df.height
    ct = df["close_time"].to_numpy()
    y = df["target"].to_numpy().astype(np.int8)
    cols = [c for c in df.columns if c not in ("target", "close_time")]
    X = df.select([pl.col(c).cast(pl.Float32) for c in cols]).to_numpy()
    del df
    log(f"[data] rows={L} feats={len(cols)} 正类={y.mean():.4f} "
        f"{dt.datetime.utcfromtimestamp(ct[0] / 1000)} ~ "
        f"{dt.datetime.utcfromtimestamp(ct[-1] / 1000)}")

    k1, k2 = int(L * 0.64), int(L * 0.80)
    m_val = (ct >= ct[k1]) & (ct < ct[k2])
    m_te = ct >= ct[k2]
    log(f"[split] train[0:{k1-H}] val[{k1}:{k2-H}] test[{k2}:{L}]  "
        f"正类 {y[:k1].mean():.3f}/{y[m_val].mean():.3f}/{y[m_te].mean():.3f}")

    ps, iters = [], []
    for sd in SEEDS:
        q = dict(BASE, seed=sd, bagging_seed=sd + 1, feature_fraction_seed=sd + 2,
                 num_threads=4)
        ds_tr = lgb.Dataset(X[:k1 - H], y[:k1 - H], feature_name=cols)
        ds_va = lgb.Dataset(X[k1:k2 - H], y[k1:k2 - H], reference=ds_tr)
        m = lgb.train(q, ds_tr, num_boost_round=ROUNDS, valid_sets=[ds_va],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        ps.append(m.predict(X, num_iteration=m.best_iteration))
        iters.append(m.best_iteration)
        log(f"[model] seed={sd} best_iter={m.best_iteration} "
            f"test AUC={roc_auc_score(y[m_te], ps[-1][m_te]):.4f}")
    p_all = np.vstack(ps).mean(0)
    p_std = np.vstack(ps).std(0)
    log(f"[model] {len(SEEDS)} seed 平均 test AUC={roc_auc_score(y[m_te], p_all[m_te]):.4f} "
        f"iters={iters}")

    # 阈值只吃 val 前半（与 h3_signal 的三段式一致），test 完全不参与
    d_val = day_index(ct[m_val])
    ud = np.unique(d_val)
    cal = np.isin(d_val, ud[:len(ud) // 2])
    thr = thr_from(p_all[m_val], cal)
    log(f"[thr] 校准段 {len(ud)//2} 天 → THR={thr:.5f}（名义 {TPD} 单/天，之后固定）")

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(p_all[m_val], y[m_val])
    iso_p = iso.predict(p_all)

    # ---------------- 校准曲线 ----------------
    pv, yv = p_all[m_te], y[m_te]
    order = np.argsort(pv)
    log("")
    log("=" * 100)
    log("校准曲线（test 段，按 p 十等分；差>0 = 模型低估上涨）")
    log(f"{'档':>3}{'n':>8}{'平均p':>9}{'实际上涨':>10}{'差(pp)':>9}{'校准后p':>10}")
    for k in range(10):
        idx = order[k * len(order) // 10:(k + 1) * len(order) // 10]
        if idx.size == 0:
            continue
        log(f"{k:>3}{idx.size:>8,}{pv[idx].mean():>9.4f}{yv[idx].mean()*100:>9.2f}%"
            f"{(yv[idx].mean()-pv[idx].mean())*100:>+9.2f}"
            f"{iso_p[m_te][idx].mean():>10.4f}")
    # 只关心触发单的 q 是否校准（这才是下注用的概率）
    hit = np.abs(pv - 0.5) >= thr
    q_te = np.where(pv > 0.5, pv, 1 - pv)
    win_te = (pv > 0.5) == (yv == 1)
    if hit.sum() > 20:
        log(f"[触发单校准] n={int(hit.sum())} 平均q={q_te[hit].mean():.4f} "
            f"实际胜率={win_te[hit].mean()*100:.2f}% "
            f"差={(win_te[hit].mean()-q_te[hit].mean())*100:+.2f}pp "
            f"（盈亏平衡 {1/(1+PAYOUT)*100:.2f}%）")
    log("=" * 100)

    # ---------------- test 段四种口径 ----------------
    ct_te, p_te, y_te, iso_te = ct[m_te], p_all[m_te], y[m_te], iso_p[m_te]
    day = day_index(ct_te)
    keep = set(np.unique(day[complete_days(day)]).tolist())
    log(f"[test] 共 {len(np.unique(day))} 天，完整天 {len(keep)} 天（四种口径同一天集）")

    protos = {
        "A 现役阈值(因果)": bets_threshold(p_te, day, thr),
        "B 过去24h分位≤15(因果)": bets_roll24h(p_te, day, TPD),
        "C 固定节拍15单(因果)": bets_fixed15(day, ct_te, TPD),
        "D 当日top15(前视)": bets_lookahead(p_te, day, TPD),
    }

    results = {}
    for tag, bets in protos.items():
        r = run_protocol(bets, keep, y_te, p_te, iso_te, ct_te)
        results[tag] = r
        nd = sum(1 for d in bets if d in keep)
        cnt = np.array([bets[d].size for d in sorted(bets) if d in keep])
        acc = r["wins"] / max(r["nbet"], 1) * 100
        log("")
        log(f"【{tag}】{nd} 天  n={r['nbet']}  日均={r['nbet']/max(nd,1):.2f} "
            f"(std {cnt.std():.1f} p90 {np.percentile(cnt,90):.0f} max {cnt.max()})  "
            f"胜率={acc:.2f}%  滚动100单min={r['roll_min']*100:.1f}% "
            f"破50%={r['roll_bad']*100:.1f}%  连亏={r['streak']}")
        log(f"{'权重':>10}{'天数':>6}{'累计':>10}{'日均':>10}{'日std':>9}"
            f"{'夏普':>8}{'最大回撤':>11}{'最差日':>9}{'有效注数':>10}")
        for s, a in r["pnl"].items():
            m = perf(a)
            r.setdefault("stat", {})[s] = dict(m, neff=r["neff"][s])
            log(f"{s:>10}{m['n']:>6}{m['total']:>+10.1f}{m['mean']:>+10.3f}{m['sd']:>9.3f}"
                f"{m['sharpe']:>8.2f}{m['mdd']:>+11.1f}{m['worst']:>+9.2f}{r['neff'][s]:>10.2f}")

    # ---------------- 前视溢价 ----------------
    log("")
    log("=" * 100)
    log("【选单口径对照】同一模型、同一 test 段，只换选单规则")
    log(f"{'口径':<24}{'单量':>7}{'日均':>7}{'胜率':>9}{'每注EV@0.8':>12}")
    for tag, r in results.items():
        nd = sum(1 for d in protos[tag] if d in keep)
        acc = r["wins"] / max(r["nbet"], 1)
        log(f"{tag:<24}{r['nbet']:>7}{r['nbet']/max(nd,1):>7.1f}{acc*100:>8.2f}%"
            f"{(1+PAYOUT)*acc-1:>+12.3f}")
    preview = (results["D 当日top15(前视)"]["wins"] / max(results["D 当日top15(前视)"]["nbet"], 1)
               - results["A 现役阈值(因果)"]["wins"] / max(results["A 现役阈值(因果)"]["nbet"], 1))
    log(f"  → D − A = {preview*100:+.2f}pp（日内排名前视的虚高幅度）；"
        "文档口径：全局top15×天数 66.94%、每日top15 59.24%、现役阈值 62.85%")
    log("=" * 100)

    # ---------------- 月度（主口径 A） ----------------
    prim = "A 现役阈值(因果)"
    r, st = results[prim], results[prim]["stat"]
    best = max(("lin", "kelly", "lin_iso", "kelly_iso"), key=lambda s: st[s]["sharpe"])
    mon = [dt.datetime.utcfromtimestamp(d * 86400).strftime("%Y-%m") for d in r["days_used"]]
    log("")
    log(f"【月度·{prim}】等权 vs 最优加权({best})")
    for mm in sorted(set(mon)):
        ii = np.array([i for i, x in enumerate(mon) if x == mm])
        log(f"  {mm}: 天数={ii.size:>3}  等权={r['pnl']['eq'][ii].sum():+9.1f}  "
            f"{best}={r['pnl'][best][ii].sum():+9.1f}")

    # ---------------- 判据 ----------------
    base, bst = st["eq"], st[best]
    bp = base["sharpe"]
    go = bool(bp > 0 and bst["sharpe"] >= bp * SHARPE_GAIN
              and bst["mdd"] >= base["mdd"] and bst["neff"] >= MIN_NEFF)
    pb = boot_p(r["pnl"]["eq"], r["pnl"][best], BOOT)
    st_b = results["B 过去24h分位≤15(因果)"]["stat"]
    best_b = max(("lin", "kelly", "lin_iso", "kelly_iso"), key=lambda s: st_b[s]["sharpe"])
    agree = bool(st_b[best_b]["sharpe"] > st_b["eq"]["sharpe"])

    log("")
    log("=" * 100)
    log(f"[主口径] {prim}（逐 bar 独立判断，无日内排名；日均 {r['nbet']/max(len(dsel),1):.1f} 单）")
    log(f"[判据·预先写死] 加权夏普 >= 等权×{SHARPE_GAIN:.2f}、回撤不更深、有效注数 >= {MIN_NEFF:.0f}")
    log(f"[实测] 最优={best} 夏普 {bst['sharpe']:.2f} vs 等权 {bp:.2f}"
        f"（{bst['sharpe']/bp if bp else 0:.2f}x）| 回撤 {bst['mdd']:+.1f} vs {base['mdd']:+.1f}"
        f" | 有效注数 {bst['neff']:.2f} | bootstrap p={pb:.3f}")
    log(f"[B 口径·15单/天] 最优={best_b} 夏普 {st_b[best_b]['sharpe']:.2f} vs 等权 "
        f"{st_b['eq']['sharpe']:.2f} → {'方向一致' if agree else '方向相反'}")
    log(f"[DECISION] {'GO：固定口径内按 q 加权能提升夏普' if go else 'STOP：加权没有稳定优势，保持等权'}")
    if go and not agree:
        log("[WARN] 压到 15 单/天后加权优势消失 → GO 存疑，按 STOP 处理")
    log("边界：回测内仓位比较，未计实盘滑点/最小下注额；q 的校准只在 test 段验证。")
    log("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
