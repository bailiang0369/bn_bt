#!/usr/bin/env python3
"""上 Kaggle 前的本地合成数据自检（不联网、不训练）。

覆盖 `续接说明.md` §6 的坑：IO、类型、表达式、因果性、评估口径先用小样本验完。
    python -m morph.h3_gate_selftest
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import polars as pl

from morph import h3_gate as hg
from morph import h3_gate_eval as ge


def main() -> None:
    rng = np.random.default_rng(0)
    n_days, bars = 40, 300
    n = n_days * bars
    days = np.repeat(np.arange(n_days, dtype=np.int64), bars)
    ct = days * 86_400_000 + np.tile(np.arange(bars, dtype=np.int64) * 60_000, n_days)
    score = rng.normal(0, 0.05, n)
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-score * 20))).astype(np.int8)
    vol = rng.lognormal(1.0, 0.3, n)
    vol[:bars * 3] = np.nan  # 冷启动段

    # 1) topk_win / apply_gate 的基本语义
    a = np.array([0.3, -0.2, 0.1])
    yy = np.array([1, 0, 1], dtype=np.int8)
    assert ge.topk_win(a, yy, 2) == 1.0
    g = ge.apply_gate(a, np.array([True, False, True]))
    assert g[1] == 0.0 and np.allclose(g[[0, 2]], a[[0, 2]])
    assert ge.topk_win(g, yy, 2) == 1.0

    # 2) 滚动监控严格因果：改第 i 天的标签不能改变 trail[i]
    s = score.copy()
    uniq, trail = ge.rolling_trailing(s, y, days, window_days=15, tpd=5)
    y2 = y.copy()
    y2[days == 20] = 1 - y2[days == 20]
    _, trail2 = ge.rolling_trailing(s, y2, days, window_days=15, tpd=5)
    assert np.isclose(trail[20], trail2[20], equal_nan=True), "trail[i] 偷看了当天结果"
    assert not np.isclose(trail[21], trail2[21], equal_nan=True), "历史应能影响后续"
    keep = ge.trailing_gate(days, uniq, trail, 0.60)
    assert keep.dtype == bool and len(keep) == n

    # 3) 候选闸门 / val 选优 / 双切分
    k1, k2 = int(n * hg.TRAIN_FRAC), int(n * (hg.TRAIN_FRAC + hg.VAL_FRAC))
    idx = np.arange(n)
    va_m, te_m = (idx >= k1) & (idx < k2), idx >= k2
    d_va, d_te = days[va_m], days[te_m]
    wide = pl.DataFrame({"close_time": ct, "close": 100.0 + np.cumsum(score),
                         "s_t": vol, "target": y})
    tr, va, te = hg.split_wide(wide, "row", ct[k1 - 1], ct[k2 - 1])
    assert (tr.height, va.height, te.height) == (k1, k2 - k1, n - k2)
    cands = hg.build_candidates(s[va_m], y[va_m], d_va, s[te_m], y[te_m], d_te,
                                vol[va_m], vol[te_m], 5)
    assert len(cands) == len(hg.RM_WINDOWS) * len(hg.RM_THRESHOLDS) + len(hg.VOL_QUANTILES) * 2 + 1
    for c in cands:
        assert c["va"].dtype == bool and c["te"].dtype == bool
        assert len(c["va"]) == int(va_m.sum()) and len(c["te"]) == int(te_m.sum())
        util, cov = hg.val_utility(s[va_m], y[va_m], d_va, c["va"], 5)
        assert 0.0 <= cov["day_cov"] <= 1.0
        assert np.isfinite(util[1]) or util[0] == -np.inf

    # 4) 评估口径（固定单量配对 CI + 缩量 + placebo + 曲线 + 每日单量）
    plans = ge.make_plans(d_te, n_boot=30, seed=1)
    allowed = ge.gate_by_days(d_te, np.unique(d_te)[::2])
    r = ge.evaluate_gate(s[te_m], y[te_m], d_te, allowed, plans, "selftest",
                         tpd=5, n_placebo=5)
    assert set(r["paired"]) == {"日均15单", "1%", "5%", "25%"}
    assert 0.0 <= r["cov"]["day_cov"] <= 1.0
    ge.mde_scan(s[te_m], y[te_m], d_te, plans, coverages=(0.7, 0.5),
                ref_score=s[te_m] + rng.normal(0, 0.01, int(te_m.sum())),
                tpd=5, n_rep=1)

    # 5) 落盘 IO（与 h3_gate.py 的 frames 同列）
    with tempfile.TemporaryDirectory() as td:
        fp = Path(td) / "gate_pred.parquet"
        pl.DataFrame({"split": "row", "segment": "test", "close_time": ct[te_m],
                      "target": y[te_m], "score": s[te_m] + 0.5,
                      "s_t": vol[te_m], "day": d_te}).write_parquet(fp)
        back = pl.read_parquet(fp)
        assert back.height == int(te_m.sum())
        assert back.columns == ["split", "segment", "close_time", "target", "score", "s_t", "day"]
        assert (back["score"].to_numpy() - 0.5).shape == s[te_m].shape

    print("SELFTEST DONE")


if __name__ == "__main__":
    main()
