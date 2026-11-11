#!/usr/bin/env python3
"""扫描发现 × 现役信号的**正交性交叉检验**（不重训模型，零训练成本）。

背景
----
`scan.py` 在 2020-2026 全量 1m 上找到一组跨 h=3/15/60 同向的反转条件：
    cvd_x_dir 桶0/9、cvd_slope_q 桶0/4、pos_q 桶0/4、body_q 桶0/4
其中 `pos_q`（过去 1 天价格位置 = 1440 分钟 Stoch %K）在 h=60 达 ±4.3pp。

本脚本回答一个问题，**只回答这一个**：
    在现役 `h3_signal.py` 已触发的单里，上面这些条件还能不能进一步区分胜负？
    如果不能（各桶胜率与全体同水平），说明现役订单簿模型已经把这些信息吃掉了，
    这些条件对我们是零增量 —— 那就不必再花钱做增量 A/B。

方法
----
1. 读 `h3_gate_pred.parquet`（现役信号数据源，split=time）。
2. 读原始 1m bar，复用 `scan.build_dims` 算出同样口径的条件标签（因果、trailing 分位）。
3. 按 `close_time` 精确对齐（要求 100% 命中，否则报错而不是静默丢弃）。
4. 现役阈值 0.05806 触发单，按每个条件桶分组算胜率：
       vs 全体 + z + 3 段同号
5. **正交性分层**：按 |p-0.5| 三分位分层，层内再看条件桶是否仍有分化。
   这是关键 —— 若分化只存在于"高置信度层"，说明条件桶只是 |p-0.5| 的代理，不是新信息。

判据
----
零增量：各桶胜率极差 < 1.5pp，或分化在 |p-0.5| 分层后消失。
有增量：至少一个桶在**每一层内**都与全体差 ≥ 2pp 且 3 段同号 → 才进入增量 A/B。

用法
----
    python -m morph.statfeat.xcheck \
        --pred /kaggle/input/btcusdt-h3-gate/h3_gate_pred.parquet
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from morph.h3_signal import calibrate, emit          # noqa: E402
from morph.statfeat.scan import build_dims, log     # noqa: E402
from morph.statfeat.validate import load_data        # noqa: E402

N_SEG = 3
THR_DEFAULT = 0.05806
MIN_N = 60          # 桶内最少成交单
WATCH = ("pos_q", "cvd_slope_q", "cvd_x_dir", "body_q")


def bucket_table(lab: np.ndarray, y: np.ndarray, base: float,
                 seg: np.ndarray) -> list[dict]:
    out = []
    for b in np.unique(lab):
        if b < 0:
            continue
        m = lab == b
        n = int(m.sum())
        if n < MIN_N:
            continue
        p = float(y[m].mean())
        se = float(np.sqrt(base * (1 - base) / n))
        z = (p - base) / (se + 1e-12)
        sp = []
        for s in range(N_SEG):
            mm = m & (seg == s)
            if mm.sum() >= 20:
                sp.append(float(y[mm].mean()))
        same = bool(len(sp) == N_SEG and (all(v > base for v in sp) or all(v < base for v in sp)))
        out.append(dict(bucket=int(b), n=n, p=p, gap_pp=(p - base) * 100, z=z,
                        sign_consistent=same))
    return out


def main() -> None:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    ap = argparse.ArgumentParser(description="扫描条件 × 现役信号的正交性交叉检验")
    ap.add_argument("--pred", default=os.environ.get(
        "PRED", "/kaggle/input/btcusdt-h3-gate/h3_gate_pred.parquet"))
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--rows", type=int, default=200000)
    ap.add_argument("--payout", type=float, default=0.8)
    a = ap.parse_args(list(sys.argv[1:]) + os.environ.get("XCHECK_ARGS", "").split())

    df = load_data(a.synthetic, a.rows)
    log(f"[raw] rows={df.height:,}  span {df['close_time'].min()} ~ {df['close_time'].max()}")
    dims = build_dims(df)
    ct_raw = df["close_time"].to_numpy().astype(np.int64)

    pred = pl.read_parquet(a.pred).filter(pl.col("split") == "time").sort("close_time")
    ct = pred["close_time"].to_numpy().astype(np.int64)
    log(f"[pred] rows={pred.height:,}  列={pred.columns}")

    # 严格对齐：必须全部命中，否则说明时间基准不一致，宁可报错也不要静默丢样本
    pos = np.searchsorted(ct_raw, ct)
    pos = np.clip(pos, 0, len(ct_raw) - 1)
    hit = ct_raw[pos] == ct
    log(f"[join] 命中 {hit.sum():,}/{len(ct):,} = {hit.mean() * 100:.3f}%")
    if hit.mean() < 0.999:
        bad = ct[~hit][:3]
        raise SystemExit(f"对齐失败，示例 close_time={bad}")
    dim = {k: v[pos] for k, v in dims.items()}

    va = pred.filter(pl.col("segment") == "val")
    te = pred.filter(pl.col("segment") == "test")
    is_te = (pred["segment"] == "test").to_numpy()

    # 阈值：与 h3_signal.py 一致（val 前半 98.96 分位）
    ud = np.unique(va["day"].to_numpy()) if "day" in va.columns else \
        np.unique(va["close_time"].to_numpy() // 86_400_000)
    day_all = (pred["day"].to_numpy() if "day" in pred.columns
               else pred["close_time"].to_numpy() // 86_400_000)
    cal = np.isin(day_all[~is_te], ud[: len(ud) // 2])
    thr = calibrate(va["score"].to_numpy(), cal)
    log(f"[thr] {thr:.5f}（现役）| 盈亏平衡 {1 / (1 + a.payout) * 100:.2f}%")

    score = pred["score"].to_numpy()
    target = pred["target"].to_numpy().astype(np.int8)
    dirs = emit(score, thr)
    idx = np.where(is_te & (dirs != 0))[0]
    if len(idx) < 200:
        raise SystemExit(f"test 触发单不足：{len(idx)}")
    win = ((dirs[idx] > 0).astype(int) == target[idx]).astype(np.int8)
    base = float(win.mean())
    seg = np.minimum(np.arange(len(idx)) * N_SEG // len(idx), N_SEG - 1)
    log(f"[test] 触发单 n={len(idx):,}  总体胜率={base * 100:.2f}%  "
        f"EV@{a.payout}={((1 + a.payout) * base - 1):+.3f}")
    log("")

    # ---------- 1. 触发单上的条件桶分化
    log("=== 1. 现役触发单按条件桶分组（基准 = 全体 %.2f%%）===" % (base * 100))
    log(f"{'维度':<14}{'桶':>4}{'n':>7}{'胜率':>9}{'差pp':>8}{'z':>7}  3段同号")
    any_gain = False
    for k in WATCH:
        if k not in dim:
            continue
        rows = bucket_table(dim[k][idx], win.astype(np.float64), base, seg)
        for r in sorted(rows, key=lambda x: -abs(x["gap_pp"])):
            flag = " *" if abs(r["gap_pp"]) >= 2.0 and r["sign_consistent"] else ""
            any_gain = any_gain or bool(flag)
            log(f"{k:<14}{r['bucket']:>4}{r['n']:>7}{r['p'] * 100:>8.2f}%"
                f"{r['gap_pp']:>+8.2f}{r['z']:>+7.2f}  {'是' if r['sign_consistent'] else '否'}{flag}")

    # ---------- 2. 正交性分层：控制 |p-0.5| 后再看
    log("")
    log("=== 2. 正交性分层：按 |p-0.5| 三分位分层，层内再按条件桶分组 ===")
    log("（若分化在此消失 → 条件桶只是 |p-0.5| 的代理，不是新信息）")
    conf = np.abs(score[idx] - 0.5)
    qs = np.quantile(conf, [1 / 3, 2 / 3])
    layer = np.digitize(conf, qs)
    for k in WATCH:
        if k not in dim:
            continue
        log(f"  --- {k} ---")
        for L in range(3):
            m = layer == L
            if m.sum() < 100:
                continue
            b_l = float(win[m].mean())
            rows = bucket_table(dim[k][idx][m], win[m].astype(np.float64), b_l,
                                np.minimum(np.arange(m.sum()) * N_SEG // m.sum(), N_SEG - 1))
            best = sorted(rows, key=lambda x: -abs(x["gap_pp"]))[:3]
            s = "  ".join(f"桶{r['bucket']}(n={r['n']},差{r['gap_pp']:+.1f}pp"
                          f"{',同号' if r['sign_consistent'] else ''})" for r in best)
            log(f"    |p-.5|层{L} (n={int(m.sum())}, 层内基准={b_l * 100:.2f}%): {s}")

    # ---------- 3. 结论
    log("")
    if any_gain:
        log("【结论】条件桶在触发单上**有**分化（见 * 标记）。"
            "仍需第 2 节确认分层后是否保留；保留才进增量 A/B。")
    else:
        log("【结论】条件桶在现役触发单上**无** ≥2pp 且 3 段同号的分化 → "
            "现役订单簿模型已吸收这些信息，对我们是零增量，不做增量 A/B。")


if __name__ == "__main__":
    main()
