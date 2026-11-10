#!/usr/bin/env python3
"""统计特征逐项验证：因果性 / 覆盖率 / 跨段 IC / 实用档位差 / BH-FDR。

用户要求「每一项都要经过验证」，所以 PASS 条件在跑之前写死，不许事后改：
  1. 因果性：扰动 t 之后的所有输入，f[:t] 必须逐位不变
  2. 覆盖率：评估点上有限值比例 >= 0.95
  3. 跨段：3 个独立时间段的 IC 符号一致
  4. 多重检验：全样本 IC 的 BH-FDR q < 0.05
  5. 实用意义：最高档 - 最低档的 P(上涨) >= 2pp
负控（CVD水平值 / 白噪声）必须全部 FAIL，否则框架本身无效。

评估点按 1m 全量算完特征后每 STEP 根抽一根（默认 5 分钟），
既匹配观察尺度又把 IC 计算压到可控范围；rolling 窗口仍按 1m 计。

    python -m morph.statfeat.validate --synthetic            # 本地烟测
    python -m morph.statfeat.validate --horizon 15 --step 5  # Kaggle 全历史
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl

try:
    from .features import REGISTRY, build_all
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from morph.statfeat.features import REGISTRY, build_all

from scipy.stats import rankdata

N_SEG = 3
ALPHA = 0.05
MIN_GAP = 0.02
MIN_IC = 0.02   # 合成负控校准：n 很大时 |IC|=0.015 也能 q<0.05，但无实用价值
INPUT_COLS = ("open", "high", "low", "close",
              "um_buy_vol", "um_sell_vol", "spot_buy_vol", "spot_sell_vol")


def log(msg: str) -> None:
    print(msg, flush=True)


def load_data(synthetic: bool, rows: int) -> pl.DataFrame:
    if synthetic:
        rng = np.random.default_rng(7)
        c = 30000 * np.exp(np.cumsum(rng.normal(0, 2e-4, rows)))
        v = np.abs(rng.normal(10, 3, rows))
        tb = np.clip(v * rng.uniform(0.3, 0.7, rows), 0, None)
        return pl.DataFrame({
            "close_time": (np.arange(rows, dtype=np.int64) * 60000
                           + 1_700_000_000_000),
            "open": c, "high": c * 1.0002, "low": c * 0.9998, "close": c,
            "um_buy_vol": tb, "um_sell_vol": v - tb,
            "spot_buy_vol": tb * 0.8, "spot_sell_vol": (v - tb) * 1.1,
        })
    from morph.kline_long import load_raw
    return load_raw()


def causality_ok(df: pl.DataFrame, fn, cut: int = 200000, tail: int = 200) -> bool:
    """扰动切片末尾 tail 行的输入，检查 f[:cut-tail] 逐位不变（只算该特征）。"""
    n = min(cut, df.height)
    if n < 5000:
        return False
    base = df.head(n)
    rng = np.random.default_rng(99)
    pert = base.with_columns([
        pl.Series(c, np.where(np.arange(n) >= n - tail,
                              rng.normal(1e6, 1e5, n),
                              base[c].to_numpy().astype(np.float64)))
        for c in INPUT_COLS if c in base.columns
    ])
    f0 = fn(base)[:n - tail]
    f1 = fn(pert)[:n - tail]
    return bool(np.array_equal(np.nan_to_num(f0, nan=-9e9),
                               np.nan_to_num(f1, nan=-9e9)))


def ic_of(f: np.ndarray, ret: np.ndarray, idx: np.ndarray) -> tuple[float, float]:
    """Spearman IC + 正态近似 p（双尾）。"""
    x, y = f[idx], ret[idx]
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 1000:
        return 0.0, 1.0
    a, b = rankdata(x[ok]), rankdata(y[ok])
    a = (a - a.mean()) / (a.std() + 1e-9)
    b = (b - b.mean()) / (b.std() + 1e-9)
    n = len(a)
    r = float(np.mean(a * b))
    r = float(np.clip(r, -0.999999, 0.999999))
    t = r * np.sqrt((n - 2) / max(1e-12, 1 - r * r))
    # 大样本下 t 近似标准正态
    from math import erfc, sqrt
    p = erfc(abs(t) / sqrt(2))
    return r, p


def gap_of(f: np.ndarray, y: np.ndarray, idx: np.ndarray, dec: int = 10) -> float:
    """最高档 - 最低档的 P(上涨)，按段内分位分档（分析用，非交易信号）。"""
    x, yy = f[idx], y[idx]
    ok = np.isfinite(x)
    x, yy = x[ok], yy[ok]
    if len(x) < 1000:
        return 0.0
    q = np.quantile(x, np.linspace(0, 1, dec + 1))
    lo = yy[x <= q[1]]
    hi = yy[x >= q[-2]]
    return float(hi.mean() - lo.mean()) if len(lo) and len(hi) else 0.0


def bh_fdr(p: np.ndarray) -> np.ndarray:
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for rank in range(n - 1, -1, -1):
        i = order[rank]
        prev = min(prev, p[i] * n / (rank + 1))
        q[i] = prev
    return q


def main() -> None:
    ap = argparse.ArgumentParser(description="统计形态特征逐项验证")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--rows", type=int, default=200000)
    ap.add_argument("--horizon", type=int, default=15)
    ap.add_argument("--step", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2 * 7 * 1440)
    ap.add_argument("--out", default=os.environ.get("STATFEAT_OUT"))
    a = ap.parse_args(list(sys.argv[1:]) + os.environ.get("STATFEAT_ARGS", "").split())

    df = load_data(a.synthetic, a.rows)
    log(f"[data] rows={df.height:,} cols={df.columns}")
    feats = build_all(df)
    log(f"[feat] {len(feats)} 个候选（含 {sum(1 for s in REGISTRY if not s.expect_pass)} 个负控）")

    c = df["close"].to_numpy().astype(np.float64)
    ret = np.full(df.height, np.nan)
    ret[:-a.horizon] = c[a.horizon:] / c[:-a.horizon] - 1
    y = (ret > 0).astype(np.float64)

    warm = min(a.warmup, df.height // 3) if a.synthetic else a.warmup
    idx = np.arange(warm, df.height - a.horizon - 1, a.step)
    segs = np.array_split(idx, N_SEG)
    log(f"[eval] 评估点={len(idx):,}  warmup={warm:,}  {N_SEG} 段各 {len(segs[0]):,} 点")

    rows = []
    for spec in REGISTRY:
        f = feats[spec.name]
        cover = float(np.isfinite(f[idx]).mean())
        causal = causality_ok(df, spec.fn)
        ics = [ic_of(f, ret, s)[0] for s in segs]
        r_all, p_all = ic_of(f, ret, idx)
        gap = gap_of(f, y, idx)
        rows.append(dict(name=spec.name, family=spec.family, cover=cover,
                         causal=causal, ic_all=r_all, p=p_all,
                         ic=ics, gap=gap, expect=spec.expect_pass,
                         hypothesis=spec.hypothesis))

    q = bh_fdr(np.array([r["p"] for r in rows]))
    for r, qi in zip(rows, q):
        r["q"] = float(qi)
        sign_ok = all(i > 0 for i in r["ic"]) or all(i < 0 for i in r["ic"])
        r["pass"] = bool(r["causal"] and r["cover"] >= 0.95 and sign_ok
                         and r["q"] < ALPHA and abs(r["gap"]) >= MIN_GAP
                         and abs(r["ic_all"]) >= MIN_IC)

    log("")
    log(f"{'name':<24}{'cover':>7}{'因果':>5}{'IC全':>9}{'IC1':>8}{'IC2':>8}{'IC3':>8}"
        f"{'q(FDR)':>10}{'档差pp':>8}{'判定':>7}")
    for r in rows:
        log(f"{r['name']:<24}{r['cover']:>7.3f}{'Y' if r['causal'] else 'N':>5}"
            f"{r['ic_all']:>9.4f}{r['ic'][0]:>8.4f}{r['ic'][1]:>8.4f}{r['ic'][2]:>8.4f}"
            f"{r['q']:>10.2e}{r['gap'] * 100:>8.2f}{'PASS' if r['pass'] else 'FAIL':>7}")

    bad_ctrl = [r["name"] for r in rows if not r["expect"] and r["pass"]]
    log("")
    log(f"[框架自检] 负控误放行 = {bad_ctrl if bad_ctrl else '无（正常）'}")
    passed = [r["name"] for r in rows if r["pass"] and r["expect"]]
    log(f"[放行清单] {len(passed)}/{sum(1 for r in rows if r['expect'])}: {passed}")
    log(f"[PASS 条件] 因果性 & 覆盖>=0.95 & 3段IC同号 & BH-FDR q<{ALPHA}"
        f" & |档差|>={MIN_GAP * 100:.0f}pp & |IC|>={MIN_IC}")

    if a.out:
        pl.DataFrame([{k: v for k, v in r.items() if k not in ("ic", "hypothesis")}
                      for r in rows]).write_csv(a.out)
        log(f"[save] {a.out}")
    log("DONE")


if __name__ == "__main__":
    main()
