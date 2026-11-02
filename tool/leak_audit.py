"""数据泄露审计（每次改特征/切分/评估口径都要跑）。

审计项
  A 切分：train/val/test 时间严格递增、区间互不重叠
  B purge：样本标签窗口 [i, i+horizon] 不得跨入下一段（训练标签碰 val、val 标签碰 test）
  C 未来信息：把 t 之后的所有数据随机化，t 时刻的模型输入必须逐位不变
  D 标签：target[i] == int(close[i+horizon] > close[i])（抽样）
  E 真实数据（有则跑）：K 线 60s 间隔、K 线/orderbook close_time 对齐率、close 相关性
  F 口径：报告各覆盖率折算的"日均单量"（1440 bar/天）

用法：
    python tool/leak_audit.py           # 合成数据（无需真实数据）
    python tool/leak_audit.py --real    # 追加真实数据检查（需 data/ 下有 parquet）
"""
import os
import sys
import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import polars as pl
from morph import morph_ts as mt

COVS = [0.005, 0.01, 0.02, 0.05, 0.10, 0.25]
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    return ok


# ---------------- A/B: 切分与 purge ----------------
def audit_split(ct, horizon=15, window=120, test_days=92, val_days=30):
    tr, va, te = mt.split_with_purge(ct, horizon, window, test_days, val_days)
    check("A1 切分非空", len(tr) > 0 and len(va) > 0 and len(te) > 0,
          f"train={len(tr)} val={len(va)} test={len(te)}")
    check("A2 切分严格递增", tr.max() < va.min() and va.max() < te.min(),
          f"train_max={ct[tr.max()]} val_min={ct[va.min()]} val_max={ct[va.max()]} test_min={ct[te.min()]}")
    check("A3 区间无交集", len(np.intersect1d(tr, va)) == 0 and len(np.intersect1d(va, te)) == 0)
    check("B1 训练标签不碰 val", int(tr.max()) + horizon < int(va.min()),
          f"tr_last+{horizon}={int(tr.max()) + horizon} < val_first={int(va.min())}")
    check("B2 val 标签不碰 test", int(va.max()) + horizon < int(te.min()),
          f"va_last+{horizon}={int(va.max()) + horizon} < test_first={int(te.min())}")
    check("B3 窗口不回看训练段之外的时间（train 首样本）",
          int(tr.min()) >= window - 1, f"tr_first={int(tr.min())}")
    return tr, va, te


# ---------------- C: 未来扰动 ----------------
def audit_future_perturbation(rows=4000, t=3000, window=120, seed=0, horizon=15):
    df = mt.synthetic_klines(rows, seed)
    ch, names, mask = mt.build_channels(df)
    x0 = mt.window_input(ch, mask, t, window)

    rng = np.random.default_rng(7)
    df2 = df.clone()
    for col in ["open", "high", "low", "close", "volume", "taker_buy_volume"]:
        arr = df2[col].to_numpy().copy()
        arr[t + 1:] = arr[t + 1:] * rng.uniform(0.3, 3.0, rows - t - 1)
        df2 = df2.with_columns(pl.Series(col, arr))
    ch2, _, _ = mt.build_channels(df2)
    x1 = mt.window_input(ch2, mask, t, window)

    same = np.array_equal(x0, x1)
    check("C1 扰动 t 之后的数据，t 处输入逐位不变", same,
          f"channels={names} max|diff|={float(np.max(np.abs(x0 - x1))):.3e}")
    check("C2 输入窗口长度=window", x0.shape[1] == window, f"shape={x0.shape}")
    check("C3 归一化通道落在 [0,1]", bool(
        np.all(x0[mask] >= -1e-6) and np.all(x0[mask] <= 1 + 1e-6)))

    # 反向验证：扰动 t 之前的数据，t 处输入必须变化（否则说明特征没用上过去数据）
    df3 = df.clone()
    arr = df3["close"].to_numpy().copy()
    arr[t - 20:t] = arr[t - 20:t] * 1.01
    df3 = df3.with_columns(pl.Series("close", arr))
    ch3, _, _ = mt.build_channels(df3)
    x3 = mt.window_input(ch3, mask, t, window)
    check("C4 敏感度对照（改过去数据 → 输入应变化）", not np.array_equal(x0, x3),
          f"max|diff|={float(np.max(np.abs(x0 - x3))):.3e}")


# ---------------- D: 标签 ----------------
def audit_target(rows=20000, horizon=15, seed=0):
    df = mt.synthetic_klines(rows, seed)
    d = mt.add_target(df, horizon)
    close = df["close"].to_numpy()
    tgt = d["target"].to_numpy()
    idx = np.arange(0, rows - horizon, 37)
    bad = [int(i) for i in idx if int(tgt[i]) != int(close[i + horizon] > close[i])]
    check("D1 target = 未来 horizon 根的方向", len(bad) == 0,
          f"抽样={len(idx)} 不一致={len(bad)}")
    check("D2 末尾 horizon 行无标签", bool(d["target"].tail(horizon).is_null().all()))


# ---------------- E: 真实数据 ----------------
def _find(names):
    hits = []
    for n in names:
        for p in (ROOT / "data" / n, ROOT / "data" / "raw" / n):
            if p.exists():
                hits.append(str(p))
    return hits


def audit_real(horizon=15):
    kl = _find(["btcusdt_1m.parquet", "btc_1m.parquet"])
    if not kl:
        print("[SKIP] E: 本地无 1m K 线 parquet")
        return
    df = pl.read_parquet(kl[0]).sort("close_time")
    ct = df["close_time"].to_numpy()
    d = np.diff(ct)
    check("E1 K线 close_time 严格递增", bool(np.all(d > 0)), f"rows={df.height}")
    check("E2 K线 1m 间隔（中位数=60000ms）", int(np.median(d)) == 60000,
          f"median={int(np.median(d))} p99={int(np.percentile(d, 99))} max={int(d.max())}")
    dup = int(len(ct) - len(np.unique(ct)))
    check("E3 K线 close_time 无重复", dup == 0, f"dups={dup}")

    obp = _find(["BTCUSDT.parquet"])
    if not obp:
        print("[SKIP] E4-E6: 本地无 orderbook parquet")
        return
    ob = pl.scan_parquet(obp[0]).select(["close_time", "close"]).sort("close_time").collect()
    ob_ct = ob["close_time"].to_numpy()
    kl_ct = ct + 1  # Binance K线 close=open+59999，orderbook close=open+60000
    pos = np.clip(np.searchsorted(kl_ct, ob_ct), 1, len(kl_ct) - 1)
    left, right = kl_ct[pos - 1], kl_ct[pos]
    nearest = np.where(np.abs(ob_ct - left) <= np.abs(ob_ct - right), pos - 1, pos)
    diff = ob_ct - kl_ct[nearest]
    match = np.abs(diff) <= 2000
    check("E4 对齐匹配率（±2s）", float(match.mean()) > 0.90,
          f"match={match.mean():.4f} |diff|max={int(np.abs(diff).max())}ms")
    ob_close = ob["close"].to_numpy()[match]
    kl_close = df["close"].to_numpy()[nearest[match]]
    cc = float(np.corrcoef(ob_close.astype(np.float64), kl_close.astype(np.float64))[0, 1])
    check("E5 orderbook close 与 K线 close 一致", cc > 0.9999, f"corr={cc:.6f}")
    # orderbook 自身 bar 是否 1m、无重复
    od = np.diff(ob_ct)
    check("E6 orderbook bar 间隔中位数=60000ms", int(np.median(od)) == 60000,
          f"median={int(np.median(od))} p99={int(np.percentile(od, 99))} max={int(od.max())}")
    # 用真实 K 线跑一遍切分/purge
    dd = mt.add_target(df, horizon).filter(pl.col("target").is_not_null())
    audit_split(dd["close_time"].to_numpy(), horizon=horizon)


# ---------------- F: 口径 ----------------
def report_units():
    print("\n[口径] 1% 覆盖率 ≈ 14.4 单/天（1440 bar/天），与『日均 15 单』目标对应：")
    for cov in COVS:
        print(f"  cov={cov:>6.3%} → {cov * 1440:>6.2f} 单/天")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true", help="追加真实数据检查")
    ap.add_argument("--horizon", type=int, default=15)
    a = ap.parse_args()

    print("=" * 78)
    print(f"数据泄露审计  horizon={a.horizon}")
    print("=" * 78)
    # 合成时间线：208 天
    rows = 300_000
    ct = 1_577_836_800_000 + np.arange(rows, dtype=np.int64) * 60_000
    audit_split(ct, a.horizon)
    audit_future_perturbation(horizon=a.horizon)
    audit_target(horizon=a.horizon)
    if a.real:
        audit_real(a.horizon)
    report_units()

    ng = [n for n, ok, _ in RESULTS if not ok]
    print("\n" + "=" * 78)
    print(f"结论：{len(RESULTS) - len(ng)}/{len(RESULTS)} 通过"
          + (f"，失败项：{ng}" if ng else "，无失败项"))
    print("=" * 78)
    return 1 if ng else 0


if __name__ == "__main__":
    sys.exit(main())
