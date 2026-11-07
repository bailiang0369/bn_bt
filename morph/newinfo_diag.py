"""诊断 kline_newinfo 的异常高 AUC（0.734）：定位泄漏特征或时间对齐错误。

检查项
  1. 时间轴：funding / metrics 的 t 范围、条数、as-of 滞后中位数（应为 0~8h / 0~24h）
  2. 单特征 AUC：任一特征单独就能到 0.6+ 说明它含未来信息
  3. 前向/后向相关对比：corr(f_t, ret[t,t+3]) vs corr(f_t, ret[t-3,t])，
     前向远大于后向 = 前视泄漏

输出全部打印，不做训练，秒级完成。
"""
import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.metrics import roc_auc_score  # noqa: E402

from morph.kline_boost import OB_END, build_features, load_klines, log, make_target  # noqa: E402
from morph.kline_newinfo import NEW, add_new, build, load_kind  # noqa: E402

H = 3


def main():
    raw = load_klines().filter(pl.col("close_time") <= OB_END)
    ft, base_cols = build_features(make_target(raw, H))
    ct = ft["close_time"].to_numpy()
    close = ft["close"].to_numpy().astype(np.float64)
    y = ft["target"].to_numpy().astype(np.int8)
    log(f"[data] rows={len(ct)} {int(ct[0])}~{int(ct[-1])}")

    fd = load_kind("fundingRate")
    md = load_kind("metrics")
    for name, d, tcol in (("funding", fd, "calc_time"), ("metrics", md, "create_time")):
        if d is None:
            log(f"[{name}] 无数据")
            continue
        log(f"[{name}] raw rows={d.height} cols={d.columns[:6]} dtype({tcol})={d.schema.get(tcol)}")
    f = build("fundingRate", fd)
    m = build("metrics", md)
    for name, d in (("funding", f), ("metrics", m)):
        if d is None:
            continue
        t = d["t"].to_numpy()
        log(f"[{name}] 解析后 n={len(t)} t={int(t.min())}~{int(t.max())} "
            f"= {pl.Series(t).cast(pl.Datetime('ms')).min()} ~ {pl.Series(t).cast(pl.Datetime('ms')).max()} "
            f"空值={int(np.isnan(t.astype(float)).sum())}")
        idx = np.searchsorted(t, ct, side="right") - 1
        lag = (ct - t[np.clip(idx, 0, len(t) - 1)]) / 60000.0
        lag = lag[idx >= 0]
        log(f"[{name}] as-of 滞后(分钟) 中位数={np.median(lag):.1f} p95={np.percentile(lag, 95):.1f} "
            f"负值占比={(lag < 0).mean():.4f}")

    feats = add_new(ct, close, f, m)
    # 前向/后向收益
    r_fwd = np.full(len(close), np.nan)
    r_bwd = np.full(len(close), np.nan)
    r_fwd[:-H] = close[H:] / close[:-H] - 1.0
    r_bwd[H:] = close[H:] / close[:-H] - 1.0

    # 测试窗（最后 96 天）与"时间本身"的 AUC：若时间 AUC 也很高，说明是漂移不是泄漏
    te = ct >= (int(ct[-1]) - 96 * 86_400_000)
    t_auc = roc_auc_score(y[te], ct[te].astype(np.float64))
    log(f"[基准] 时间特征在测试窗的 AUC={t_auc:.4f}（若≈0.68 说明是趋势漂移，不是泄漏）")
    log(f"[基准] 测试窗基准正例率={y[te].mean():.4f} 全样本={y.mean():.4f}")

    out = [f"{'特征':<16}{'AUC全':>8}{'AUC测试':>9}{'corr前向':>10}{'corr后向':>10}"
           f"{'非空':>8}{'测试窗非空':>10}{'中位数':>12}"]
    for c in NEW:
        v = feats[c].astype(np.float64)
        ok = np.isfinite(v)
        f_ok = ok & te
        auc = roc_auc_score(y[ok], v[ok]) if ok.sum() > 1000 and len(np.unique(v[ok])) > 2 else np.nan
        auc_t = (roc_auc_score(y[f_ok], v[f_ok])
                 if f_ok.sum() > 1000 and len(np.unique(v[f_ok])) > 2 else np.nan)
        a = ok & np.isfinite(r_fwd)
        b = ok & np.isfinite(r_bwd)
        cf = np.corrcoef(v[a], r_fwd[a])[0, 1] if a.sum() > 1000 else np.nan
        cb = np.corrcoef(v[b], r_bwd[b])[0, 1] if b.sum() > 1000 else np.nan
        out.append(f"{c:<16}{auc:>8.4f}{auc_t:>9.4f}{cf:>10.4f}{cb:>10.4f}"
                   f"{ok.mean():>8.2%}{ok[te].mean():>10.2%}{float(np.nanmedian(v)):>12.4g}")
    txt = "\n".join(out)
    log("\n" + txt)
    try:
        with open("/kaggle/working/newdiag.txt", "w", encoding="utf-8") as fh:
            fh.write(txt + "\n")
    except Exception as e:
        log(f"[save] 失败 {e}")

    # 对齐抽查：取测试窗内 5 个时刻，打印特征值与前后 3 分钟收益
    idx = np.where(te & np.isfinite(feats["n_oi_px_div"]))[0]
    log("\n[对齐抽查] t, n_doi_7d, n_oi_px_div, ret前3m, ret后3m")
    for i in idx[:: max(1, len(idx) // 5)][:5]:
        log(f"  {int(ct[i])} {feats['n_doi_7d'][i]:.5f} {feats['n_oi_px_div'][i]:+.5f} "
            f"{r_bwd[i]*1e4:+.1f}bp {r_fwd[i]*1e4:+.1f}bp")

    # ---- 滞后敏感性：区分"真信号"与"5 分钟桶内前视" ----
    # metrics 的 create_time 若是窗口**起点**，则 t 处使用它会看到 t~t+5m 的成交 → 泄漏。
    # 真信号会随滞后缓慢衰减；前视会在 lag>=5min 时断崖归零。
    if m is not None:
        t = m["t"].to_numpy()
        tak = (m["sum_taker_long_short_vol_ratio"].to_numpy()
               if "sum_taker_long_short_vol_ratio" in m.columns else None)
        log(f"\n[metrics 原始行] {pl.Series(t[:3]).cast(pl.Datetime('ms')).to_list()} "
            f"步长(分钟)={np.unique(np.diff(t[:5]) / 60000.0)}")
        if tak is not None:
            sp = pl.DataFrame({"tak": tak}).with_columns(
                ((pl.col("tak") - pl.col("tak").rolling_mean(30))
                 / (pl.col("tak").rolling_std(30) + 1e-12)).alias("z"))
            log(f"{'lag(min)':>9}{'AUC_raw':>9}{'AUC_z30':>9}{'corr前3m':>10}{'corr后3m':>10}")
            for lag in (0, 5, 10, 15, 30, 60):
                ct_l = ct - lag * 60_000
                v1 = asof_t(t, sp["tak"].to_numpy(), ct_l)
                v2 = asof_t(t, sp["z"].to_numpy(), ct_l)
                ok1 = np.isfinite(v1)
                t_ok = np.isfinite(v2) & te
                a1 = roc_auc_score(y[ok1], v1[ok1]) if ok1.sum() > 1000 else np.nan
                a2 = roc_auc_score(y[t_ok], v2[t_ok]) if t_ok.sum() > 1000 else np.nan
                a = np.isfinite(v1) & np.isfinite(r_fwd)
                b = np.isfinite(v1) & np.isfinite(r_bwd)
                cf = np.corrcoef(v1[a], r_fwd[a])[0, 1] if a.sum() > 1000 else np.nan
                cb = np.corrcoef(v1[b], r_bwd[b])[0, 1] if b.sum() > 1000 else np.nan
                log(f"{lag:>9}{a1:>9.4f}{a2:>9.4f}{cf:>10.4f}{cb:>10.4f}")
    log("[done]")
    return 0


def asof_t(ts, vals, query):
    ts = np.asarray(ts, dtype=np.int64)
    idx = np.searchsorted(ts, query, side="right") - 1
    return np.where(idx >= 0, np.asarray(vals, dtype=np.float64)[np.clip(idx, 0, len(vals) - 1)],
                    np.nan)


if __name__ == "__main__":
    raise SystemExit(main())
