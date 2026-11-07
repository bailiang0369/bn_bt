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

    log(f"\n{'特征':<16}{'AUC':>8}{'corr前向':>10}{'corr后向':>10}{'非空':>8}{'取值样例':>12}")
    for c in NEW:
        v = feats[c].astype(np.float64)
        ok = np.isfinite(v)
        auc = roc_auc_score(y[ok], v[ok]) if ok.sum() > 1000 and len(np.unique(v[ok])) > 2 else np.nan
        a = np.isfinite(v) & np.isfinite(r_fwd)
        b = np.isfinite(v) & np.isfinite(r_bwd)
        cf = np.corrcoef(v[a], r_fwd[a])[0, 1] if a.sum() > 1000 else np.nan
        cb = np.corrcoef(v[b], r_bwd[b])[0, 1] if b.sum() > 1000 else np.nan
        log(f"{c:<16}{auc:>8.4f}{cf:>10.4f}{cb:>10.4f}{ok.mean():>8.2%}{float(np.nanmedian(v)):>12.4g}")
    log("[done]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
