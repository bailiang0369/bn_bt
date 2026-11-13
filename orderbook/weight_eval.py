#!/usr/bin/env python3
"""固定 15 单/天不变，只改「每单下注多少」：按模型概率加权能否提升夏普？

动机
----
现役流程是「模型打分 z → 每天取 |z| 前 15 单 → 方向 = sign(z) → **每单等额**」。
业务已明确禁止「少下单换胜率」，所以覆盖率、单量、模型都不动；
唯一剩下的自由度是**仓位分配**：同样 15 单里，p=0.75 的样本是否该比 p=0.57 下注更多？

这个问题只有在 q（所选方向的实际胜率）**校准**的前提下才有正解，所以先画校准曲线，
再模拟权重。二元期权口径：赢 +0.8×stake、输 −1×stake（80% payout，保本胜率 55.6%）。

口径（与 `orderbook/train.py` 完全一致，保证 66.94% 那个数字可比）
----
- 数据：`BTCUSDT.parquet`（盘口 185 列 + close）。**2026-11-13 起 Kaggle 数据集
  `byh850515/btcusdt-orderbook-features` 已不可用（403）**，故取数三级回退：
  /kaggle/input 挂载 → GitHub Release `merged-features/BTCUSDT.parquet` → 同 Release 月度分片拼接
- target：t+H 的 close 与 t 的 close 比，涨=1（等价于 train.py 的 join 口径）
- 切分：时间序 64% / 16% / 20%，无 shuffle；训练段尾部 H 根因标签越界被剔除
- 选单：static_val z = (p − p_val.mean) / p_val.std，按天取 |z| 前 15；sign(z) 定方向

三种权重（q = 下注方向的实际胜率，p<0.5 时 q = 1−p）：
  eq    等额 1/15
  lin   |q − 0.5|
  kelly clip(q·b − (1−q), 0)，b = PAYOUT（归一化后常数项无影响）

预先写死的判据（不因结果调整）：
  GO   当且仅当某加权方案的「日收益夏普」≥ 等权 × 1.2，
       且最大回撤不比等权更深，且有效注数 1/Σw² ≥ 5（防权重塌到 1~2 单）。
  STOP 否则 → 保持等权，不要再动仓位。

边界：回测内仓位比较，未计实盘滑点/最小下注额/单场上限；校准只在 test 段验证，
      实盘需滚动复核。另附 roll_1d 选单口径做稳健性对照（若两口径结论打架则 GO 存疑）。
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


# ---------------------------------------------------------------- 工具
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


# ---------------------------------------------------------------- 选单 + 模拟
def simulate(ct, z, y, p_up, p_std, iso_p, k=TPD):
    """按天取 |z| 前 k 单，返回 {scheme: 日PNL数组} + 诊断。"""
    day = day_index(ct)
    keep = complete_days(day)
    pnl = {s: [] for s in SCHEMES}
    neff = {s: [] for s in SCHEMES}
    days, wins, nbet, pstd_sel = [], [], [], []
    for d in np.unique(day[keep]):
        idx = np.where((day == d) & keep)[0]
        sel = idx[np.argsort(-np.abs(z[idx]))[:k]]
        if sel.size == 0:
            continue
        dirs = z[sel] > 0                       # 生产口径：sign(z) 定方向
        win = dirs == (y[sel] == 1)
        q_raw = np.where(dirs, p_up[sel], 1.0 - p_up[sel])
        q_iso = np.where(dirs, iso_p[sel], 1.0 - iso_p[sel])
        qs = {"eq": q_raw, "lin": q_raw, "kelly": q_raw,
              "lin_iso": q_iso, "kelly_iso": q_iso}
        for name, q in qs.items():
            w = norm(w_eq(q) if name == "eq" else
                     w_lin(q) if name.startswith("lin") else w_kelly(q))
            pay = np.where(win, PAYOUT, -1.0)
            pnl.setdefault(name, []).append(float(np.sum(w * pay)))
            neff.setdefault(name, []).append(float(1.0 / np.sum(w ** 2)))
        days.append(int(d))
        wins.append(int(win.sum()))
        nbet.append(int(sel.size))
        pstd_sel.append(float(p_std[sel].mean()))
    return dict(pnl={s: np.asarray(v) for s, v in pnl.items()},
                neff={s: float(np.mean(v)) for s, v in neff.items()},
                days=days, wins=np.asarray(wins), nbet=np.asarray(nbet),
                pstd=np.asarray(pstd_sel))


def report(res, tag: str) -> dict:
    log("")
    log(f"【{tag}】每天 top{TPD} 的仓位模拟（每单 stake 归一，日投入恒为 1 单位）")
    log(f"{'权重':>10}{'天数':>6}{'累计':>10}{'日均':>10}{'日std':>9}"
        f"{'夏普':>8}{'最大回撤':>11}{'最差日':>9}{'有效注数':>10}")
    st = {}
    for s, a in res["pnl"].items():
        st[s] = perf(a)
        st[s]["neff"] = res["neff"][s]
        m = st[s]
        log(f"{s:>10}{m['n']:>6}{m['total']:>+10.3f}{m['mean']:>+10.4f}{m['sd']:>9.4f}"
            f"{m['sharpe']:>8.2f}{m['mdd']:>+11.3f}{m['worst']:>+9.3f}{m['neff']:>10.2f}")
    return st


def main() -> int:
    raw, src = load_btcusdt()
    # target：t+H 的 close 与 t 的 close 比（与 train.py 的 join 口径等价，前提是分钟网格连续）
    df = raw.sort("close_time").with_columns(pl.col("close").shift(-H).alias("_fc"))
    df = (df.with_columns((pl.col("_fc") > pl.col("close")).cast(pl.Int8).alias("target"))
          .drop("_fc").filter(pl.col("target").is_not_null()).drop_nulls())
    if src != "kaggle-dataset":
        log("[warn] 走 GitHub 回退：列集可能不含 stoch/supertrend 两个派生列，"
            "胜率绝对值要与文档 66.94% 分开看（末列特征数会打印）")
    L = df.height
    ct = df["close_time"].to_numpy()
    y = df["target"].to_numpy().astype(np.int8)
    cols = [c for c in df.columns if c not in ("target", "close_time")]
    X = df.select([pl.col(c).cast(pl.Float32) for c in cols]).to_numpy()
    del df
    log(f"[data] rows={L} feats={len(cols)} 正类={y.mean():.4f} "
        f"{dt.datetime.utcfromtimestamp(ct[0]/1000)} ~ "
        f"{dt.datetime.utcfromtimestamp(ct[-1]/1000)}")

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

    # 概率→概率的保序校准（只用 val 拟合，test 应用）
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(p_up[m_val], y[m_val])
    iso_p = iso.predict(p_up)

    # ---------------- 校准曲线（针对 p_up vs 实际上涨比例） ----------------
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

    # ---------------- 两种标准化口径各自模拟 ----------------
    z_static = (p_up - p_up[m_val].mean()) / (p_up[m_val].std() + 1e-9)
    z_roll = rolling_z(p_up, 1440)
    bases = {"static_val(生产口径)": z_static, "roll_1d(稳健性)": z_roll}

    all_res = {}
    for tag, z in bases.items():
        r = simulate(ct[m_te], z[m_te], y[m_te], p_up[m_te], p_std[m_te], iso_p[m_te])
        all_res[tag] = (r, report(r, tag))
        log(f"[{tag}] 覆盖 {len(r['days'])} 天，日均 {r['nbet'].mean():.1f} 单，"
            f"top{TPD} 胜率={r['wins'].sum()/r['nbet'].sum()*100:.2f}%")

    # 参考档：只取 5 单（会降覆盖率，仅作反例）
    z0 = z_static[m_te]
    r5 = simulate(ct[m_te], z0, y[m_te], p_up[m_te], p_std[m_te], iso_p[m_te], k=5)
    st5 = perf(r5["pnl"]["eq"])
    log(f"\n[参考·不可选] 每天 5 单等权：胜率={r5['wins'].sum()/r5['nbet'].sum()*100:.2f}% "
        f"累计={st5['total']:+.2f} 夏普={st5['sharpe']:.2f}（覆盖率下降，业务禁止）")

    # ---------------- seed 分歧度诊断 ----------------
    log("")
    log("【诊断】seed 分歧度（跨 seed p 的标准差）五分位 vs 上涨比例")
    o2 = np.argsort(p_std[m_te])
    for k in range(5):
        idx = o2[k * len(o2) // 5:(k + 1) * len(o2) // 5]
        log(f"  std 第{k*20}-{(k+1)*20}%: 平均std={p_std[m_te][idx].mean():.4f} "
            f"上涨比例={yv[idx].mean()*100:.2f}%")

    # ---------------- 月度稳定性（primary 口径） ----------------
    r, st = all_res["static_val(生产口径)"]
    best = max(("lin", "kelly", "lin_iso", "kelly_iso"),
               key=lambda s: st[s]["sharpe"])
    log("")
    log(f"【月度】等权 vs 最优加权({best}) 的月累计收益")
    mon = [dt.datetime.utcfromtimestamp(d * 86400).strftime("%Y-%m") for d in r["days"]]
    for m in sorted(set(mon)):
        ii = np.array([i for i, x in enumerate(mon) if x == m])
        log(f"  {m}: 天数={ii.size:>3}  等权={r['pnl']['eq'][ii].sum():+8.3f}  "
            f"{best}={r['pnl'][best][ii].sum():+8.3f}")

    # ---------------- 判据 ----------------
    base = st["eq"]
    bp = base["sharpe"]
    bst = st[best]
    neff = bst["neff"]
    go = bool(bp > 0 and bst["sharpe"] >= bp * SHARPE_GAIN
              and bst["mdd"] >= base["mdd"] and neff >= MIN_NEFF)
    pb = boot_p(r["pnl"]["eq"], r["pnl"][best], BOOT)
    # 稳健性：换 roll_1d 选单后，最优方案是否仍优于等权
    _, st_r = all_res["roll_1d(稳健性)"]
    rb = max(("lin", "kelly", "lin_iso", "kelly_iso"), key=lambda s: st_r[s]["sharpe"])
    robust = bool(st_r[rb]["sharpe"] > st_r["eq"]["sharpe"])

    log("")
    log("=" * 100)
    log(f"[判据·预先写死] 加权夏普 >= 等权×{SHARPE_GAIN:.2f}、回撤不更深、"
        f"有效注数 >= {MIN_NEFF:.0f}")
    log(f"[实测] 最优={best} 夏普 {bst['sharpe']:.2f} vs 等权 {bp:.2f}"
        f"（{bst['sharpe']/bp if bp else 0:.2f}x）| 回撤 {bst['mdd']:+.3f} vs {base['mdd']:+.3f}"
        f" | 有效注数 {neff:.2f} | 配对bootstrap p={pb:.3f}")
    log(f"[稳健性] roll_1d 口径下最优={rb} "
        f"{st_r[rb]['sharpe']:.2f} vs 等权 {st_r['eq']['sharpe']:.2f} → "
        f"{'方向一致' if robust else '方向相反，加权优势不成立'}")
    log(f"[DECISION] {'GO：固定单量内按 q 加权能显著提升夏普' if go else 'STOP：加权没有稳定优势，保持等权'}")
    if go and not robust:
        log("[WARN] 换选单口径后加权优势消失 → GO 存疑，建议按 STOP 处理")
    log("边界：回测内仓位比较，未计实盘滑点/最小下注额/单场上限；")
    log("      p 的校准只在 test 段验证，实盘需用滚动校准复核。")
    log("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
