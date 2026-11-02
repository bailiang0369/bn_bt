"""3min 攻坚 3：时段稳定性验证（多 test 段）。

Run P 诊断：test 上 18-24 UTC 的按天 top15 准确率 0.656，其他时段 0.55~0.58；
但 val 全样本只有 0.53，无法确认。本脚本在多个 test 段（时间轴前移 0/60/120 天）上
重复"分时段取 top-k"，看 18-24 UTC 的优势是否稳定存在。

评估口径：
  all      —— 当天全部样本取 |score| top-k（= 日均 15 单基线）
  s18_24   —— 只在 18-24 UTC 内取 top-k（单量不变，但要求该时段有足够样本）
  s00_06 / s06_12 / s12_18 同理

用法：
    python -m orderbook.h3_session
"""
import os
import sys
import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import polars as pl
from orderbook.fusion_test import (
    load_orderbook, train_lgb_avg, day_index, complete_days)

REG = dict(num_leaves=31, min_child_samples=500, feature_fraction=0.6, lambda_l2=5.0)
SEGS = {"s00_06": (0, 6), "s06_12": (6, 12), "s12_18": (12, 18), "s18_24": (18, 24)}


def hour_of(ct):
    return ((ct % 86_400_000) // 3_600_000).astype(int)


def daily_topk_in(score, y, day_id, mask, k, seg_mask=None):
    s = score[mask] if seg_mask is None else np.where(seg_mask[mask], score[mask], np.nan)
    yy = y[mask]
    dd = day_id[mask]
    accs = []
    for u in np.unique(dd):
        m = dd == u
        sv = s[m]
        valid = ~np.isnan(sv)
        if valid.sum() == 0:
            continue
        kk = min(k, int(valid.sum()))
        idx = np.argsort(-np.abs(sv[valid]))[:kk]
        accs.append(float(((sv[valid][idx] > 0).astype(int) == yy[m][valid][idx]).mean()))
    return np.array(accs)


def run_offset(ob0, h, seeds, rounds, k, off):
    if off > 0:
        ob0 = ob0.head(ob0.height - off * 1440)
    ob = ob0.with_columns((pl.col("close").shift(-h) > pl.col("close"))
                          .cast(pl.Int8).alias("target")).filter(pl.col("target").is_not_null())
    L = ob.height
    k1, k2 = int(L * 0.64), int(L * 0.80)
    tr, va, te = slice(0, k1 - h), slice(k1, k2 - h), slice(k2, L)
    ct = ob["close_time"].to_numpy()
    y = ob["target"].to_numpy().astype(np.int8)
    cols = [c for c in ob.columns if c not in ("target", "close_time")]
    X = ob.select(cols).to_numpy().astype(np.float32)
    p, iters = train_lgb_avg(X, y, tr, va, cols, seeds, rounds, None, REG)
    m, s = float(p[tr].mean()), float(p[tr].std() + 1e-9)
    z = (p - m) / s
    print(f"[off={off}] rows={L} iters={iters} "
          f"test[{k2}:{L}] days={len(np.unique(day_index(ct[te])))}")

    d_te = day_index(ct[te])
    m_te = complete_days(d_te)
    hr = hour_of(ct[te])
    out = {}
    a_all = daily_topk_in(z[te], y[te], d_te, m_te, k)
    out["all"] = (len(a_all), float(a_all.mean()))
    for name, (lo, hi) in SEGS.items():
        sm = (hr >= lo) & (hr < hi)
        acc = daily_topk_in(z[te], y[te], d_te, m_te, k, sm)
        out[name] = (len(acc), float(acc.mean()) if len(acc) else float("nan"))
    print(f"  {'全天':<8}{out['all'][0]:>4}天 {out['all'][1]:.4f}   " + "  ".join(
        f"{n}:{v[1]:.4f}({v[0]}天)" for n, v in out.items() if n != "all"))
    return out


def main():
    ap = argparse.ArgumentParser(description="3min 时段稳定性（多段验证）")
    ap.add_argument("--seeds", default=os.environ.get("KAGGLE_SEEDS", "42,43,44,45,46"))
    ap.add_argument("--rounds", type=int, default=1000)
    ap.add_argument("--daily-k", type=int, default=15)
    ap.add_argument("--h", type=int, default=3)
    ap.add_argument("--offsets", default=os.environ.get("KAGGLE_TEST_OFFSETS", "0,60,120"))
    a = ap.parse_args()

    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    offsets = [int(s) for s in a.offsets.split(",") if s.strip()]
    ob0 = load_orderbook()

    res = {}
    for off in offsets:
        res[off] = run_offset(ob0, a.h, seeds, a.rounds, a.daily_k, off)

    print("\n" + "=" * 78)
    print("时段稳定性汇总（按天 top%d 的日均准确率）" % a.daily_k)
    print(f"{'前移天':>7}{'全天':>10}" + "".join(f"{n:>10}" for n in SEGS))
    for off, r in res.items():
        print(f"{off:>7}{r['all'][1]:>10.4f}" + "".join(f"{r[n][1]:>10.4f}" for n in SEGS))
    print("\n[说明] 只有多个 test 段都稳定领先的时段才算真优势；单段高就是噪声。")


if __name__ == "__main__":
    main()
