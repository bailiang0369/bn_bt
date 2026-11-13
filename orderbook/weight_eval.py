#!/usr/bin/env python3
"""固定 15 单/天不变，只改「每单下注多少」：按模型概率加权能否提升夏普？

⚠️ 前视修正（2026-11-13，用户指出）
------------------------------------
「当天按 |z| 取前 15 单」在实盘中**做不到**：下单时当天的 bar 还没走完，
不可能知道当天哪 15 根分数最高。此前文档里所有「按天 top15」的数字
（含 `morph/kline_norm_eval.py` 的全局 top-K 取 u）都属于**前视口径**，
只能当上界参照，不能当作实盘胜率。

本脚本因此并列三种选单口径，且只有后两种可用于实盘：
  P0  lookahead   当天 top15 by |z|              —— 前视上界（仅对照）
  P1  fixed15     每天固定 15 个时刻（每 96 分钟）—— 零选择，单量恰 15
  P2  roll24h     过去 24h 的 |z| 分位定阈值，当天按时间顺序触发、满 15 单即停
                  —— 严格因果，最贴合「用之前 24 小时信息」
P2 是主口径（GO/STOP 只看它）；P1 是「必须凑满 15 单」时的兜底；
P0 与 P2 的胜率差 = 前视溢价，直接量化旧口径虚高了多少。

仓位分配问题（本脚本真正要回答的）
----------------------------------
同样这批单里，q=0.75 的样本是否该比 q=0.57 下注更多？只有在 q 校准的前提下才有正解。
  eq    等额 1/n
  lin   |q − 0.5|
  kelly clip(q·b − (1−q), 0)，b = PAYOUT = 0.8（80% payout，保本胜率 55.6%）
q = 所选方向的实际胜率估计（p_up 校准后，p<0.5 时取 1−p）；校准用 val 段 isotonic，严格因果。

预先写死的判据（不因结果调整）
------------------------------
  GO   当且仅当主口径(P2)下某加权方案的「日收益夏普」≥ 等权 × 1.2，
       且最大回撤不比等权更深，且有效注数 1/Σw² ≥ 5（防权重塌到 1~2 单）。
  STOP 否则 → 保持等权。

数据（2026-11-13 起 Kaggle 数据集 byh850515/btcusdt-orderbook-features 已 403）：
  /kaggle/input 挂载 → GitHub Release merged-features/BTCUSDT.parquet → 同 Release 月度分片拼接
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

from orderbook.znorm_eval import (BASE, H, complete_days, day_index,  # noqa: E402
                                  log, rolling_z)

GH_REPO = "bailiang0369/bn_hisdata"
GH_API = "https://api.github.com"
GH_TAG = os.environ.get("GH_TAG", "merged-features")
NAME = "BTCUSDT.parquet"
CACHE = Path("/tmp/gh_merged")

PAYOUT = 0.8
TPD = int(os.environ.get("TPD", "15"))
SEEDS = [int(s) for s in os.environ.get("SEEDS", "42,43,44").split(",") if s.strip()]
ROUNDS = int(os.environ.get("ROUNDS", "2000"))
MIN_NEFF = float(os.environ.get("MIN_NEFF", "5.0"))
SHARPE_GAIN = float(os.environ.get("SHARPE_GAIN", "1.2"))
BOOT = int(os.environ.get("BOOT", "2000"))
DAY_MS = 86_400_000
BAR_MS = 60_000


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
    """日收益序列 → 累计 / 夏普(×√365) / 最大回撤 / 最差日。"""
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
    return np.full(q.size, 1.0 / q.size)


def w_lin(q: np.ndarray) -> np.ndarray:
    return np.abs(q - 0.5)


def w_kelly(q: np.ndarray) -> np.ndarray:
    return np.clip(q * PAYOUT - (1.0 - q), 0.0, None)


SCHEMES = {"eq": w_eq, "lin": w_lin, "kelly": w_kelly}


def norm(c: np.ndarray) -> np.ndarray:
    s = c.sum()
    return c / s if s > 1e-12 else np.full(c.size, 1.0 / c.size)


def boot_p(base: np.ndarray, alt: np.ndarray, n: int, seed: int = 0) -> float:
    """单侧配对 bootstrap：H0 = 加权日收益不优于等权，返回 p 值。"""
    if n <= 0 or base.size < 5:
        return float("nan")
    d = np.asarray(alt) - np.asarray(base)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, (n, d.size))
    return float((d[idx].mean(1) <= 0).mean())


# ---------------------------------------------------------------- 三种选单口径
def bets_lookahead(day: np.ndarray, z: np.ndarray, k: int) -> dict[int, np.ndarray]:
    out = {}
    for d in np.unique(day):
        idx = np.where(day == d)[0]
        out[int(d)] = idx[np.argsort(-np.abs(z[idx]))[:k]]
    return out


def _fixed_slots(mod: np.ndarray, k: int) -> np.ndarray:
    """当天 00:00 起每 1440/k 分钟一个时刻，取该时刻之后第一根可用 bar。"""
    step = 1440 // k
    sel, used = [], -1
    for t in range(0, 1440, step):
        j = int(np.searchsorted(mod, t))
        while j < mod.size and j <= used:
            j += 1
        if j >= mod.size:
            break
        used = j
        sel.append(j)
    return np.asarray(sel, dtype=int)


def bets_fixed15(day: np.ndarray, z: np.ndarray, ct: np.ndarray,
                 k: int) -> dict[int, np.ndarray]:
    """零选择：固定节拍。单量恰 k（当天数据缺失时可能少几单）。"""
    mod_all = (ct % DAY_MS) // BAR_MS
    out = {}
    for d in np.unique(day):
        idx = np.where(day == d)[0]
        out[int(d)] = idx[_fixed_slots(mod_all[idx], k)]
    return out


def bets_roll24h(day: np.ndarray, z: np.ndarray, k: int,
                 win_days: int = 1) -> dict[int, np.ndarray]:
    """严格因果：只用**当天之前** win_days 天的 |z| 定分位阈值，当天按时间顺序触发。"""
    out, hist = {}, []
    for d in np.unique(day):
        idx = np.where(day == d)[0]           # ct 已排序 → idx 天然按时间
        if hist:
            past = np.concatenate(hist[-win_days:])
            thr = float(np.quantile(past, 1.0 - k / past.size))
            cand = idx[np.abs(z[idx]) >= thr]
            out[int(d)] = cand[:k]
        else:                                  # 冷启动：没有历史就不下单
            out[int(d)] = idx[:0]
        hist.append(np.abs(z[idx]))
    return out


# ---------------------------------------------------------------- 模拟
def simulate_sel(sel_list: list[np.ndarray], y: np.ndarray, p_up: np.ndarray,
                 iso_p: np.ndarray, dirs_list: list[np.ndarray]) -> dict:
    pnl = {s: [] for s in ("eq", "lin", "kelly", "lin_iso", "kelly_iso")}
    neff = {s: [] for s in pnl}
    wins = nbet = 0
    for sel, dirs in zip(sel_list, dirs_list):
        if sel.size == 0:
            continue
        win = dirs == (y[sel] == 1)
        q_raw = np.where(dirs, p_up[sel], 1.0 - p_up[sel])
        q_iso = np.where(dirs, iso_p[sel], 1.0 - iso_p[sel])
        for name in pnl:
            q = q_iso if name.endswith("_iso") else q_raw
            f = w_eq if name == "eq" else (w_lin if name.startswith("lin") else w_kelly)
            w = norm(f(q))
            pnl[name].append(float(np.sum(w * np.where(win, PAYOUT, -1.0))))
            neff[name].append(float(1.0 / np.sum(w ** 2)))
        wins += int(win.sum())
        nbet += int(sel.size)
    return dict(pnl={s: np.asarray(v) for s, v in pnl.items()},
                neff={s: float(np.mean(v)) for s, v in neff.items()},
                wins=wins, nbet=nbet)


def main() -> int:
    raw, src = load_btcusdt()
    df = raw.sort("close_time").with_columns(pl.col("close").shift(-H).alias("_fc"))
    df = (df.with_columns((pl.col("_fc") > pl.col("close")).cast(pl.Int8).alias("target"))
          .drop("_fc").filter(pl.col("target").is_not_null()).drop_nulls())
    if src != "kaggle-dataset":
        log("[warn] 走 GitHub 回退：列集可能不含 stoch/supertrend，"
            "胜率绝对值与文档 66.94% 不可直接比（后者本身还是前视口径）")
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
    P = np.vstack(ps)
    p_up = P.mean(0)
    p_std = P.std(0)
    log(f"[model] {len(SEEDS)} seed 平均 test AUC={roc_auc_score(y[m_te], p_up[m_te]):.4f} "
        f"iters={iters}")

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(p_up[m_val], y[m_val])
    iso_p = iso.predict(p_up)

    # ---------------- 校准曲线 ----------------
    pv, yv = p_up[m_te], y[m_te]
    order = np.argsort(pv)
    log("")
    log("=" * 100)
    log("校准曲线（test 段，按 p_up 十等分；差>0 = 模型低估上涨）")
    log(f"{'档':>3}{'n':>8}{'平均p':>9}{'实际上涨':>10}{'差(pp)':>9}"
        f"{'校准后p':>10}{'seed std':>10}")
    for k in range(10):
        idx = order[k * len(order) // 10:(k + 1) * len(order) // 10]
        if idx.size == 0:
            continue
        log(f"{k:>3}{idx.size:>8,}{pv[idx].mean():>9.4f}{yv[idx].mean()*100:>9.2f}%"
            f"{(yv[idx].mean()-pv[idx].mean())*100:>+9.2f}"
            f"{iso_p[m_te][idx].mean():>10.4f}{p_std[m_te][idx].mean():>10.4f}")
    log("=" * 100)

    # ---------------- test 段：三种选单口径 ----------------
    ct_te, z = ct[m_te], (p_up - p_up[m_val].mean()) / (p_up[m_val].std() + 1e-9)
    day = day_index(ct_te)
    keep = complete_days(day)
    days_ok = set(np.unique(day[keep]).tolist())
    log(f"[test] 共 {len(np.unique(day))} 天，完整天 {len(days_ok)} 天")

    protos = {
        "P0 当天top15(前视)": bets_lookahead(day, z, TPD),
        "P1 固定节拍15单": bets_fixed15(day, z, ct_te, TPD),
        "P2 过去24h阈值(因果)": bets_roll24h(day, z, TPD, win_days=1),
    }

    results, stats = {}, {}
    for tag, bets in protos.items():
        sel_list, dirs_list = [], []
        for d in sorted(bets):
            if d not in days_ok:
                continue
            sel = bets[d]
            sel_list.append(sel)
            dirs_list.append(z[sel] > 0)
        r = simulate_sel(sel_list, y[m_te], p_up[m_te], iso_p[m_te], dirs_list)
        results[tag] = r
        st = {}
        log("")
        log(f"【{tag}】{len(sel_list)} 天，日均 {r['nbet']/max(len(sel_list),1):.2f} 单，"
            f"总胜率 {r['wins']/max(r['nbet'],1)*100:.2f}%")
        log(f"{'权重':>10}{'天数':>6}{'累计':>10}{'日均':>10}{'日std':>9}"
            f"{'夏普':>8}{'最大回撤':>11}{'最差日':>9}{'有效注数':>10}")
        for s, a in r["pnl"].items():
            m = perf(a)
            st[s] = dict(m, neff=r["neff"][s])
            log(f"{s:>10}{m['n']:>6}{m['total']:>+10.3f}{m['mean']:>+10.4f}{m['sd']:>9.4f}"
                f"{m['sharpe']:>8.2f}{m['mdd']:>+11.3f}{m['worst']:>+9.3f}{r['neff'][s]:>10.2f}")
        stats[tag] = st

    # ---------------- 前视溢价 ----------------
    acc = {t: results[t]["wins"] / max(results[t]["nbet"], 1) * 100 for t in protos}
    log("")
    log("=" * 100)
    log("【前视溢价】同一模型、同一 test 段，只换选单口径的胜率")
    for t in protos:
        log(f"  {t:<22} 单量={results[t]['nbet']:>5d}  胜率={acc[t]:6.2f}%")
    log(f"  → P0 − P2 = {acc['P0 当天top15(前视)']-acc['P2 过去24h阈值(因果)']:+.2f}pp "
        f"是旧口径虚高部分；实盘只能看 P1/P2。")
    log("=" * 100)

    # ---------------- 稳健性：过去 7 天阈值 ----------------
    b7 = bets_roll24h(day, z, TPD, win_days=7)
    sel7, dir7 = [], []
    for d in sorted(b7):
        if d in days_ok:
            sel7.append(b7[d])
            dir7.append(z[b7[d]] > 0)
    r7 = simulate_sel(sel7, y[m_te], p_up[m_te], iso_p[m_te], dir7)
    st7 = {s: dict(perf(a), neff=r7["neff"][s]) for s, a in r7["pnl"].items()}
    log("")
    log(f"【P2b 过去7天阈值(因果)】{len(sel7)} 天，日均 {r7['nbet']/max(len(sel7),1):.2f} 单，"
        f"胜率 {r7['wins']/max(r7['nbet'],1)*100:.2f}%")
    for s in ("eq", "lin", "kelly", "lin_iso", "kelly_iso"):
        log(f"  {s:<10} 累计={st7[s]['total']:+8.3f} 夏普={st7[s]['sharpe']:6.2f} "
            f"回撤={st7[s]['mdd']:+8.3f} 有效注数={st7[s]['neff']:5.2f}")

    # ---------------- seed 分歧度诊断 ----------------
    log("")
    log("【诊断】seed 分歧度（跨 seed p 的标准差）五分位 vs 上涨比例")
    o2 = np.argsort(p_std[m_te])
    for k in range(5):
        idx = o2[k * len(o2) // 5:(k + 1) * len(o2) // 5]
        log(f"  std 第{k*20}-{(k+1)*20}%: 平均std={p_std[m_te][idx].mean():.4f} "
            f"上涨比例={yv[idx].mean()*100:.2f}%")

    # ---------------- 月度稳定性（主口径） ----------------
    prim = "P2 过去24h阈值(因果)"
    r, st = results[prim], stats[prim]
    best = max(("lin", "kelly", "lin_iso", "kelly_iso"), key=lambda s: st[s]["sharpe"])
    dsel = [d for d in sorted(protos[prim]) if d in days_ok]
    mon = [dt.datetime.utcfromtimestamp(d * 86400).strftime("%Y-%m") for d in dsel]
    log("")
    log(f"【月度·{prim}】等权 vs 最优加权({best})")
    for mm in sorted(set(mon)):
        ii = np.array([i for i, x in enumerate(mon) if x == mm])
        log(f"  {mm}: 天数={ii.size:>3}  等权={r['pnl']['eq'][ii].sum():+8.3f}  "
            f"{best}={r['pnl'][best][ii].sum():+8.3f}")

    # ---------------- 判据 ----------------
    base, bst = st["eq"], st[best]
    bp = base["sharpe"]
    neff = bst["neff"]
    go = bool(bp > 0 and bst["sharpe"] >= bp * SHARPE_GAIN
              and bst["mdd"] >= base["mdd"] and neff >= MIN_NEFF)
    pb = boot_p(r["pnl"]["eq"], r["pnl"][best], BOOT)
    bp1 = stats["P1 固定节拍15单"]
    best1 = max(("lin", "kelly", "lin_iso", "kelly_iso"), key=lambda s: bp1[s]["sharpe"])
    robust = bool(bp1[best1]["sharpe"] > bp1["eq"]["sharpe"])

    log("")
    log("=" * 100)
    log(f"[主口径] {prim}（严格因果，实盘可执行）")
    log(f"[判据·预先写死] 加权夏普 >= 等权×{SHARPE_GAIN:.2f}、回撤不更深、"
        f"有效注数 >= {MIN_NEFF:.0f}")
    log(f"[实测] 最优={best} 夏普 {bst['sharpe']:.2f} vs 等权 {bp:.2f}"
        f"（{bst['sharpe']/bp if bp else 0:.2f}x）| 回撤 {bst['mdd']:+.3f} vs {base['mdd']:+.3f}"
        f" | 有效注数 {neff:.2f} | 配对bootstrap p={pb:.3f}")
    log(f"[稳健性] P1 固定节拍下最优={best1} "
        f"{bp1[best1]['sharpe']:.2f} vs 等权 {bp1['eq']['sharpe']:.2f} → "
        f"{'方向一致' if robust else '方向相反'}")
    log(f"[DECISION] {'GO：固定单量内按 q 加权能提升夏普' if go else 'STOP：加权没有稳定优势，保持等权'}")
    if go and not robust:
        log("[WARN] 换选单口径后加权优势消失 → GO 存疑，按 STOP 处理")
    log("边界：回测内仓位比较，未计实盘滑点/最小下注额/单场上限；q 的校准只在 test 段验证。")
    log("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
