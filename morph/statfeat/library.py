#!/usr/bin/env python3
"""统计**知识库**扫描：多目标 × 多 horizon 的条件画像（不设交易准入门）。

与 scan.py / validate.py 的口径差别（重要）
------------------------------------------
- `validate.py`：7 关准入门，只回答「能不能进 h=3/15 事件合约模型」。
- `scan.py`：只回答「什么条件下未来上涨概率偏离基准」，目标是找交易假设。
- **本模块**：不设门槛、不做淘汰，只把「每个条件桶在每个目标上的完整分布」
  归档成可检索的长表。目标变量不只「方向」，还包括波动、成交量、CVD、回撤。
  **结论对二元合约无用也可以归档** —— 用户 2026-11-11 明确：挖掘不一定服务于
  当前产品，未来总有用处（周线/现货/风控/做市都可能用得上）。

设计
----
条件维度：复用 `scan.build_dims` 的 16 个（时间 7 个 + 状态 9 个），全部因果。
目标变量（每个 horizon 各算一份）：
    ret      未来 h 分钟收益（均值/中位/标准差 = 方向与幅度）
    absret   |ret|（波动幅度，二元合约的「出大振幅」能力）
    rng      未来 h 内 (max-min)/close（真实波幅）
    ddown    未来 h 内自起点最大回撤（做多风险）
    dvol     未来 h 成交量 / 过去 1 天同时长成交量（量能变化）
    dcvd     未来 h 净主动买量变化 / 过去 1 天平量（资金流变化）
每个 (维度, 桶, 目标, horizon) 输出：n / mean / median / std / p10 / p90 /
t 值 / 年化后的口径提示。**不做 FDR 淘汰**：知识库要全量留档，
显著性只作为一列（t）供检索时自己判断。

抽样：step = h（非重叠），避免标签重叠把 n 夸大几个数量级。

用法
----
    python -m morph.statfeat.library --synthetic
    python -m morph.statfeat.library --out /kaggle/working/library_btc.csv
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from morph.statfeat.scan import build_dims, log   # noqa: E402
from morph.statfeat.validate import load_data      # noqa: E402

HORIZONS = (3, 15, 60, 240, 1440)
TARGETS = ("ret", "absret", "rng", "ddown", "dvol", "dcvd")
MIN_N = 100
DAY = 1440


def _roll_max(x: np.ndarray, w: int) -> np.ndarray:
    return pl.Series("x", x).rolling_max(w, min_samples=w).to_numpy()


def _roll_min(x: np.ndarray, w: int) -> np.ndarray:
    return pl.Series("x", x).rolling_min(w, min_samples=w).to_numpy()


def _roll_sum(x: np.ndarray, w: int) -> np.ndarray:
    v = np.nan_to_num(x, nan=0.0)
    c = np.cumsum(np.insert(v, 0, 0.0))
    out = np.full(len(v), np.nan)
    out[w - 1:] = c[w:] - c[:-w]
    return out


def build_targets(df: pl.DataFrame, h: int) -> dict[str, np.ndarray]:
    """所有目标都以 t 为起点、窗口 [t, t+h]，只在 t 处可算完（评估时按 t+h 可用性筛）。"""
    n = df.height
    c = df["close"].to_numpy().astype(np.float64)
    vol = (df["um_buy_vol"].fill_null(0).to_numpy() + df["um_sell_vol"].fill_null(0).to_numpy()
           ).astype(np.float64)
    net = (df["um_buy_vol"].fill_null(0).to_numpy()
           - df["um_sell_vol"].fill_null(0).to_numpy()).astype(np.float64)

    ret = np.full(n, np.nan)
    ret[:-h] = c[h:] / c[:-h] - 1

    fmax = np.full(n, np.nan)
    fmin = np.full(n, np.nan)
    fmax[:-h] = _roll_max(c, h + 1)[h:]
    fmin[:-h] = _roll_min(c, h + 1)[h:]

    rng = np.full(n, np.nan)
    ok = np.isfinite(fmax) & np.isfinite(fmin)
    rng[ok] = (fmax[ok] - fmin[ok]) / c[ok]
    ddown = np.full(n, np.nan)
    ddown[ok] = fmin[ok] / c[ok] - 1

    fs = _roll_sum(vol, h)
    ps = _roll_sum(vol, DAY)
    dvol = np.full(n, np.nan)
    m = np.isfinite(fs) & np.isfinite(ps) & (ps > 0)
    dvol[m] = fs[m] / (ps[m] * h / DAY + 1e-12)

    # 归一化基准必须用「过去 1 天的平均绝对净流」而不是「净和」：
    # 后者在净买入抵消到 0 时会把 dcvd 炸到 1e5 量级（同 vol_ratio_spot_um 的坑）。
    fcv = _roll_sum(net, h)
    pcv = _roll_sum(np.abs(net), DAY) / DAY
    dcvd = np.full(n, np.nan)
    m2 = np.isfinite(fcv) & np.isfinite(pcv) & (pcv > 0)
    dcvd[m2] = fcv[m2] / (pcv[m2] * h + 1e-9)

    return dict(ret=ret, absret=np.abs(ret), rng=rng, ddown=ddown, dvol=dvol, dcvd=dcvd)


def describe(v: np.ndarray) -> dict:
    v = v[np.isfinite(v)]
    if len(v) < MIN_N:
        return {}
    sd = float(np.std(v))
    mean = float(np.mean(v))
    return dict(n=len(v), mean=mean, median=float(np.median(v)), std=sd,
                p10=float(np.quantile(v, 0.10)), p90=float(np.quantile(v, 0.90)),
                t=float(mean / (sd / np.sqrt(len(v)) + 1e-12)))


def main() -> None:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    ap = argparse.ArgumentParser(description="统计知识库：多目标 × 多 horizon 条件画像")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--rows", type=int, default=200000)
    ap.add_argument("--symbol", default=os.environ.get("SYMBOL", "BTCUSDT"))
    ap.add_argument("--warmup", type=int, default=2 * DAY)
    ap.add_argument("--out", default=os.environ.get("LIB_OUT"))
    ap.add_argument("--top", type=int, default=12)
    a = ap.parse_args(list(sys.argv[1:]) + os.environ.get("LIB_ARGS", "").split())

    df = load_data(a.synthetic, a.rows)
    n = df.height
    log(f"[data] symbol={a.symbol} rows={n:,}")
    dims = build_dims(df)
    log(f"[dims] {len(dims)} 个")
    warm = min(a.warmup, n // 3) if a.synthetic else a.warmup

    rows: list[dict] = []
    for h in HORIZONS:
        tg = build_targets(df, h)
        keep = np.zeros(n, dtype=bool)
        keep[warm:n - h:h] = True
        idx = np.where(keep)[0]
        if len(idx) < 500:
            log(f"\n--- h={h}m 独立样本仅 {len(idx)}，跳过 ---")
            continue
        log("")
        log(f"=== h={h}m | 独立样本 {len(idx):,} ===")
        log(f"  {'目标':<8}{'全局均值':>12}{'全局中位':>12}{'std':>10}")
        for tname in TARGETS:
            d = describe(tg[tname][idx])
            if d:
                log(f"  {tname:<8}{d['mean']:>12.5f}{d['median']:>12.5f}{d['std']:>10.5f}")

        # 预先算好「每个维度每个桶在抽样点上的位置」，供 6 个目标复用，
        # 否则 6 目标 × 16 维 × ~10 桶 会做 4800 次 O(n) 全表比较。
        pre: dict[str, list[tuple[int, np.ndarray]]] = {}
        for dname, lab in dims.items():
            lb = lab[idx]
            pre[dname] = [(int(b), np.where(lb == b)[0]) for b in np.unique(lb) if b >= 0]

        for tname in TARGETS:
            yv = tg[tname][idx]
            ok = np.isfinite(yv)
            if ok.sum() < 500:
                continue
            for dname, items in pre.items():
                for b, pos in items:
                    d = describe(yv[pos[ok[pos]]])
                    if not d:
                        continue
                    rows.append(dict(symbol=a.symbol, horizon=h, target=tname,
                                     dim=dname, bucket=b, **d))
        log(f"  累计归档 {len(rows):,} 行")

    if not rows:
        raise SystemExit("无可用观测")
    out = pl.DataFrame(rows)
    if a.out:
        out.write_csv(a.out)
        log("")
        log(f"[out] {a.out}  {out.height:,} 行 × {len(out.columns)} 列")

    # ---------- 摘要：每个目标 × horizon 最强/最弱的 3 个桶（按 |t|，仅供参考）
    log("")
    log("=== 摘要：各目标 × horizon 绝对值最大的条件桶（按 |t|，供检索用，非判定）===")
    for h in HORIZONS:
        for tname in TARGETS:
            sub = out.filter((pl.col("horizon") == h) & (pl.col("target") == tname)
                             & pl.col("t").is_finite())
            if sub.height == 0:
                continue
            top = sub.with_columns(pl.col("t").abs().alias("_a")).sort("_a", descending=True).head(3)
            s = "  ".join(f"{r['dim']}#{r['bucket']}(μ={r['mean']:+.5f},t={r['t']:+.1f})"
                          for r in top.iter_rows(named=True))
            log(f"  h={h:<5}{tname:<8} {s}")

    log("")
    log("【说明】本表是**知识库**：不做淘汰、不判可否交易。t 值只作检索排序，"
        "任何一条要变成策略仍需多段配对验证。")


if __name__ == "__main__":
    main()
