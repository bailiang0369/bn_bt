"""第十二轮：VP（量价分布，Volume Profile）指标。

口径说明（遵守硬约束）：
  - 总成交量 volume **不直接入模**，只出现在 VP 分布里（这是唯一允许的场景）；
  - CVD 仍走合约 taker_buy_volume 合成，本文件不碰现货；
  - 所有特征只用当前 bar 及之前的数据（VP 在 anchor 上算，向后 ffill，无未来 bar）；
  - 任何窗口 / 阈值都不在 test 上选。

对每个窗口 W 产出 5 个特征（共 3 窗口 = 15 列）：
  vp_poc_{W}   当前 close 相对 POC（成交量最大价位）的距离，bp，正=价在 POC 上方
  vp_pos_{W}   当前 close 在价值区间（70% 量能）内的位置，0=下沿 1=上沿
  vp_wide_{W}  价值区间宽度，bp（用成交量定义的价格波动，与 ATR 不同源）
  vp_imb_{W}   当前价上下方成交量比 (up-dn)/(up+dn)，正=上方套牢盘多
  vp_vwap_{W}  close 相对滚动 VWAP 的偏离，bp（逐 bar 精确，不下采样）

实现要点：
  - 价格分箱用 log 价格固定步长（默认 2bp），保证跨年份箱边界对齐；
  - 直方图每 anchor 根 bar 算一次（默认 10 根），POC/价值区间/量比是慢变量；
  - 但距离类特征用**当前** close 减去 anchor 处算出的价位，避免整列滞后 10 分钟。

实验设计（同一 test、同一评估口径）：
  base63    63 特征 × h∈HS 的等权 z 平均（复现第十一轮 LGB 腿）
  +vp       78 特征同流程（新增 15 列）
  vp_only   只用 15 列 VP，判断 VP 是否自带信息
  同时打印每条 VP 特征在 val / test 上的单独 AUC（不训练，纯诊断）。
"""
import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lightgbm as lgb  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from morph.kline_boost import (OB_END, PARAMS, TEST_START, VAL_START,  # noqa: E402
                               build_features, load_klines, log, zscore)
from morph.kline_multi import HS, evaluate  # noqa: E402

VP_WINS = tuple(int(x) for x in os.environ.get("VP_WINS", "240,960,1440").split(","))
VP_TICK = float(os.environ.get("VP_TICK", "2.0"))     # 分箱步长（bp）
VP_ANCHOR = int(os.environ.get("VP_ANCHOR", "10"))    # 每多少根 bar 重算一次直方图


def build_vp(df, windows=VP_WINS, tick_bp=VP_TICK, anchor=VP_ANCHOR):
    """返回 (float32 矩阵 (n, 5*len(windows)), 列名)。只用 volume（VP 内）与 OHLC。"""
    if "volume" not in df.columns:
        raise SystemExit("[vp] 数据缺 volume，无法算 VP")
    c = df["close"].to_numpy().astype(np.float64)
    h = df["high"].to_numpy().astype(np.float64)
    l = df["low"].to_numpy().astype(np.float64)
    v = df["volume"].fill_null(0).to_numpy().astype(np.float64)
    n = len(c)
    tp = (h + l + c) / 3.0
    step = tick_bp * 1e-4
    bins = np.floor(np.log(np.maximum(tp, 1e-9)) / step).astype(np.int64)
    pos = np.arange(n)

    a_idx = np.arange(anchor - 1, n, anchor)          # 直方图锚点 9, 19, 29, ...
    # 每根 bar 只能用它**之前或当前**的锚点：t=9..18 -> 锚点 9，t=19 -> 锚点 19。
    # 注意不能写成 pos // anchor（t=10..18 会被映射到 t=19 的锚点 = 未来数据，
    # 实测该错误让 AUC 从 0.53 虚增到 0.77）。
    k_raw = (pos + 1) // anchor - 1
    ready = k_raw >= 0
    k_of = np.clip(k_raw, 0, len(a_idx) - 1)

    # 逐 bar 精确 VWAP：用 scale 避免 cumsum 精度损失
    vs = v / 1e3
    cum_v = np.concatenate(([0.0], np.cumsum(vs)))
    cum_pv = np.concatenate(([0.0], np.cumsum(vs * tp / 1e3)))

    feats, names = [], []
    for W in windows:
        i0 = np.maximum(pos - W + 1, 0)
        sv = cum_v[pos + 1] - cum_v[i0]
        spv = cum_pv[pos + 1] - cum_pv[i0]
        vwap = np.where(sv > 0, spv * 1e3 / np.maximum(sv, 1e-12), c)
        dev = (c - vwap) / c * 1e4
        dev[pos < W - 1] = 0.0

        poc_l = np.zeros(len(a_idx))
        lo_l = np.zeros(len(a_idx))
        hi_l = np.zeros(len(a_idx))
        imb_l = np.zeros(len(a_idx))
        for j, a in enumerate(a_idx):
            s = a - W + 1
            if s < 0:
                continue
            bb = bins[s:a + 1]
            vv = v[s:a + 1]
            tot = vv.sum()
            if tot <= 0:
                continue
            m = bb.min()
            hist = np.bincount(bb - m, weights=vv)
            nb = len(hist)
            poc_b = int(hist.argmax())
            poc_l[j] = np.exp((m + poc_b + 0.5) * step)
            # 价值区间：按量降序累到 70%
            order = np.argsort(-hist)
            k = int(np.searchsorted(np.cumsum(hist[order]), 0.7 * tot)) + 1
            sel = order[:min(k, nb)]
            lo_b, hi_b = m + int(sel.min()), m + int(sel.max())
            lo_l[j] = np.exp(lo_b * step)
            hi_l[j] = np.exp((hi_b + 1) * step)
            # 当前价（锚点那根）上下方的成交量
            cb = int(np.clip(bins[a] - m, 0, nb - 1))
            ch = np.cumsum(hist)
            dn = ch[cb - 1] if cb > 0 else 0.0
            up = tot - ch[cb]
            imb_l[j] = (up - dn) / (up + dn + 1e-12)

        poc = poc_l[k_of]
        va_lo = lo_l[k_of]
        va_hi = hi_l[k_of]
        ok = (poc > 0) & ready
        poc_d = np.where(ok, (c - poc) / c * 1e4, 0.0)
        pos_va = np.where(ok, (c - va_lo) / np.maximum(va_hi - va_lo, 1e-9), 0.0)
        pos_va = np.clip(pos_va, -0.5, 1.5)
        wide = np.where(ok, (va_hi - va_lo) / c * 1e4, 0.0)
        imb = np.where(ok, imb_l[k_of], 0.0)

        for nm, arr in ((f"vp_poc_{W}", poc_d), (f"vp_pos_{W}", pos_va),
                        (f"vp_wide_{W}", wide), (f"vp_imb_{W}", imb),
                        (f"vp_vwap_{W}", dev)):
            feats.append(np.nan_to_num(arr, nan=0.0).astype(np.float32))
            names.append(nm)
        log(f"  [vp W={W}] 锚点={len(a_idx)} 箱步={tick_bp}bp 完成")

    return np.vstack(feats).T, names


def vp_leak_check(n=20000, cut=17000, windows=(240, 960), anchor=10, seed=0):
    """未来数据自检：随机化 cut 之后的全部行情，cut 之前的 VP 输出必须逐位不变。

    这条检查抓到过 `pos // anchor` 的锚点错位（t=10..18 用到 t=19 的直方图），
    当时 AUC 从 0.5277 虚增到 0.7658。任何新增 VP/窗口类特征都应先过这一关。
    """
    # cut 不能落在 anchor 整倍数上，否则最后一个锚点仍在 cut 之前，错位的锚点抓不到
    cut = (cut // anchor) * anchor + anchor // 2
    rng = np.random.default_rng(seed)
    c = 60000 * np.exp(np.cumsum(rng.normal(0, 3e-4, n)))
    o = c * (1 + rng.normal(0, 1e-4, n))
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 2e-4, n)))
    l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 2e-4, n)))
    v = rng.lognormal(3, 1, n)
    df = pl.DataFrame({"close": c, "open": o, "high": h, "low": l, "volume": v})
    x1, _ = build_vp(df, windows=windows, anchor=anchor)
    bad_df = pl.DataFrame({
        "close": c[cut:] * 1.3, "open": o[cut:] * 0.7, "high": h[cut:] * 2.0,
        "low": l[cut:] * 0.5, "volume": v[cut:] * 5.0,
    })
    x2, _ = build_vp(pl.concat([df.head(cut), bad_df]), windows=windows, anchor=anchor)
    bad = int((x1[:cut] != x2[:cut]).sum())
    log(f"[vp] 泄露自检：cut 之前不一致元素 {bad} -> {'通过' if bad == 0 else '失败!'}")
    return bad == 0


def fit_z(Xtr, Xpr, y, tr, va, nv, seeds, rounds, names):
    """按 horizon 训练，返回长度为 len(Xpr) 的 z 分数（val 段做参考）。"""
    acc = np.zeros(len(Xpr))
    for sd in seeds:
        p = dict(PARAMS, seed=sd, bagging_seed=sd, feature_fraction_seed=sd,
                 data_random_seed=sd)
        m = lgb.train(p, lgb.Dataset(Xtr, y[tr], feature_name=names),
                      num_boost_round=rounds)
        acc += m.predict(Xpr)
    ph = acc / len(seeds)
    return zscore(ph, ph[:nv]), m


def main():
    if os.environ.get("VP_SELFCHECK", "1") == "1":
        if not vp_leak_check():
            raise SystemExit("[vp] 泄露自检失败，拒绝继续（先修特征）")

    seeds = [int(x) for x in os.environ.get("SEEDS", "42,43").split(",")]
    rounds = int(os.environ.get("ROUNDS", "60"))
    nmax_h = max(HS)

    raw = load_klines().filter(pl.col("close_time") <= OB_END)
    ft, cols = build_features(raw)
    nmax = raw.height - nmax_h
    ct = ft.head(nmax)["close_time"].to_numpy()
    close = raw["close"].to_numpy().astype(np.float64)
    log(f"[data] rows={nmax} base_cols={len(cols)}")

    vp, vpcols = build_vp(raw)
    log(f"[vp] {len(vpcols)} 列 {vpcols}")

    tr = np.where(ct < VAL_START)[0]
    va = np.where((ct >= VAL_START) & (ct < TEST_START))[0]
    te = np.where(ct >= TEST_START)[0]
    va = va[va + nmax_h < int(te[0])]
    tr = tr[tr + nmax_h < int(va[0])]
    pr = np.concatenate([va, te])
    nv = len(va)
    log(f"[split] train={len(tr)} val={len(va)} test={len(te)} "
        f"天数={len(np.unique(ct[te] // 86_400_000))}")

    y3 = (close[3:3 + nmax] > close[:nmax]).astype(np.int8)

    # ---- 诊断：单特征 AUC（不训练）；val 与 test 都报，只作诊断不选参 ----
    log("\n[diag] 单特征 AUC（val / test）")
    for i, nm in enumerate(vpcols):
        x = vp[:nmax, i]
        av = float(roc_auc_score(y3[va], x[va]))
        at = float(roc_auc_score(y3[te], x[te]))
        log(f"  {nm:<14} val={av:.4f} test={at:.4f} "
            f"{'（反向）' if at < 0.5 else ''}")

    X = ft.head(nmax).select(cols).to_numpy().astype(np.float32)
    Xa = np.hstack([X, vp[:nmax]])
    del X
    Xtr_a, Xpr_a = Xa[tr], Xa[pr]
    Xtr_v, Xpr_v = Xtr_a[:, len(cols):], Xpr_a[:, len(cols):]
    del Xa
    log(f"[X] full={Xtr_a.shape} vp_only={Xtr_v.shape}")

    rows = {}
    tag_a, tag_b, tag_c = "base63", "base63+vp", "vp_only"

    # ---- A: 63 特征（复现第十一轮 LGB 腿）----
    za = {}
    for h in HS:
        y = (close[h:h + nmax] > close[:nmax]).astype(np.int8)
        z, m = fit_z(Xtr_a[:, :len(cols)], Xpr_a[:, :len(cols)], y, tr, va, nv,
                     seeds, rounds, cols)
        za[h] = z
        log(f"[A h={h}] done")
    full = np.zeros(nmax)
    for h in HS:
        full[pr] = za[h]
        if h == 3:
            evaluate("h3_base63", full, ct, y3, va, rows)
    full[pr] = np.mean([za[h] for h in HS], axis=0)
    evaluate(tag_a, full, ct, y3, va, rows)

    # ---- B: 63 + 15 VP ----
    allcols = cols + vpcols
    zb = {}
    for h in HS:
        y = (close[h:h + nmax] > close[:nmax]).astype(np.int8)
        z, m = fit_z(Xtr_a, Xpr_a, y, tr, va, nv, seeds, rounds, allcols)
        zb[h] = z
        if h == 3:
            imp = sorted(zip(allcols, m.feature_importance("gain")), key=lambda x: -x[1])
            rank = {nm: i + 1 for i, (nm, _) in enumerate(imp)}
            log("  [gain] top10 " + ", ".join(f"{n}:{v:.0f}" for n, v in imp[:10]))
            log("  [gain] VP 排名 " + ", ".join(f"{n}:{rank[n]}" for n in vpcols))
        log(f"[B h={h}] done")
    full = np.zeros(nmax)
    for h in HS:
        full[pr] = zb[h]
        if h == 3:
            evaluate("h3_base63+vp", full, ct, y3, va, rows)
    full[pr] = np.mean([zb[h] for h in HS], axis=0)
    evaluate(tag_b, full, ct, y3, va, rows)

    # ---- C: 只用 VP（1 seed，只作诊断）----
    zc = {}
    for h in HS:
        y = (close[h:h + nmax] > close[:nmax]).astype(np.int8)
        z, _ = fit_z(Xtr_v, Xpr_v, y, tr, va, nv, seeds[:1], rounds, vpcols)
        zc[h] = z
        log(f"[C h={h}] done")
    full = np.zeros(nmax)
    full[pr] = np.mean([zc[h] for h in HS], axis=0)
    evaluate(tag_c, full, ct, y3, va, rows)

    log(f"\n{'变体':<18}{'static':>9}{'AUC':>8}{'按天':>8}{'窗口4':>8}{'4段sd':>8}")
    for k, r in rows.items():
        log(f"{k:<18}{r['static']:>9.4f}{r['auc']:>8.4f}{r['daily']:>8.4f}"
            f"{r['w4']:>8.4f}{r['w4sd']:>8.4f}  明细={r['wins']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
