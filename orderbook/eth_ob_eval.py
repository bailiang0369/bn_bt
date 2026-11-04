"""ETH 3min 单腿评估 —— 与 BTC 的 0.6465 同口径，用于跨标的比较。

口径完全对齐 `orderbook/fusion_kl.py`：
  - h=3（3 分钟方向），固定 60 轮 × 5 seeds（Run R 结论：早停不稳）
  - 切分按行比例 k1=64% / k2=80%，末尾 purge h 行
  - 特征 = 除 target/close_time 之外的全部数值列
  - z-score 用 val 段统计量
  - 主口径 `global_topk_eval`（K = 15 × test 天数），按天 top15 仅作对照

与 BTC 的差别只在数据源：ETH 的 207 列是 build_dataset 产出的
「vision 19 列 + ob_f/ob_s 各 61 列 + 跨市场特征」，没有独立的 faiss/CNN K 线腿，
所以这里只能比「单腿」：ETH(207 列) vs BTC 盘口腿(150 列)，不能直接比融合值。

用法：
    python -m orderbook.eth_ob_eval
    python -m orderbook.eth_ob_eval --symbol ETHUSDT --save-pred /kaggle/working/eth_pred.parquet
"""
import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from orderbook.fusion_test import (  # noqa: E402
    add_target, train_lgb_avg, day_index, complete_days,
    daily_topk_eval, global_topk_eval)

OB_REG = dict(num_leaves=31, min_child_samples=500, feature_fraction=0.6, lambda_l2=5.0)

# 参照值（来自 融合实验记录.md §16，off=0，同一口径 global topK K=15×天数）
BTC_REF = {
    "盘口腿": 0.6410,
    "融合 w=0.2": 0.6465,
    "LR stacker": 0.6472,
}


def load_symbol(symbol: str) -> pl.DataFrame:
    hits = glob.glob(f"/kaggle/input/**/{symbol}.parquet", recursive=True)
    if not hits:
        local = [str(p) for p in (ROOT / "data").rglob(f"{symbol}.parquet")]
        hits = local
    if not hits:
        raise FileNotFoundError(f"找不到 {symbol}.parquet（/kaggle/input 或 data/ 下）")
    print(f"[data] {hits[0]}")
    df = pl.read_parquet(hits[0]).sort("close_time")
    print(f"[data] rows={df.height} cols={df.width}")
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description="ETH 3min 单腿评估（对齐 BTC 口径）")
    ap.add_argument("--h", type=int, default=int(os.environ.get("KAGGLE_HORIZON", "3")))
    ap.add_argument("--seeds", default=os.environ.get("KAGGLE_OSEEDS", "42,43,44,45,46"))
    ap.add_argument("--rounds", type=int, default=60)
    ap.add_argument("--daily-k", type=int, default=15)
    ap.add_argument("--symbol", default=os.environ.get("SYMBOL", "ETHUSDT"))
    ap.add_argument("--save-pred", default=os.environ.get("KAGGLE_FUSION_PRED"))
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    h = a.h

    df = load_symbol(a.symbol)
    if "close" not in df.columns:
        raise SystemExit("[data] 缺少 close 列，无法造 target")

    df = add_target(df, h).filter(pl.col("target").is_not_null())
    # 全 null 列会让 LightGBM 白跑一列，先剔掉；同时报告剔了哪些
    cand = [c for c in df.columns if c not in ("target", "close_time")]
    allnull = [c for c in cand if df[c].null_count() == df.height]
    cols = [c for c in cand if c not in allnull]
    if allnull:
        print(f"[warn] 剔除全 null 列 {len(allnull)} 个: {allnull[:10]}")

    L = len(df)
    ct = df["close_time"].to_numpy()
    y = df["target"].to_numpy().astype(np.int8)
    X = df.select(cols).to_numpy().astype(np.float32)
    print(f"[ob] rows={L} feats={len(cols)} 时间 {ct[0]} ~ {ct[-1]}")

    k1, k2 = int(L * 0.64), int(L * 0.80)
    tr, va = slice(0, k1 - h), slice(k1, k2 - h)
    val_start, test_start = int(ct[k1]), int(ct[k2])
    print(f"[split] train[0:{k1 - h}] val[{k1}:{k2 - h}] test[{k2}:{L}]")
    print(f"[split] val_start={val_start} test_start={test_start} purge={h}")

    p, iters = train_lgb_avg(X, y, tr, va, cols, seeds, a.rounds, None, OB_REG)
    print(f"[lgb] iters={iters} seeds={seeds} rounds={a.rounds}")

    dyn = day_index(ct)
    m_val = (ct >= val_start) & (ct < test_start)
    m_te = ct >= test_start
    dte = dyn[m_te]
    mt = complete_days(dte)
    n_days = len(np.unique(dte))

    z = (p - p[m_val].mean()) / (p[m_val].std() + 1e-9)

    def gacc(mask):
        idx = np.where(mask)[0]
        return global_topk_eval(z[idx], y[idx], dyn[idx], a.daily_k)

    def dacc(mask, cm):
        idx = np.where(mask)[0]
        return daily_topk_eval(z[idx], y[idx], dyn[idx], cm, a.daily_k)

    auc = float(roc_auc_score(y[m_te], z[m_te]))
    gk = gacc(m_te)
    dk = float(dacc(m_te, mt).mean())
    gv = gacc(m_val)
    base = float(y[m_te].mean())

    print(f"\n[test] n={int(m_te.sum())} 天数={n_days} 基准={base:.4f} 全局K={a.daily_k * n_days}")
    print(f"[test] AUC={auc:.4f}  全局topK={gk:.4f}  按天topK={dk:.4f}")
    print(f"[val ] 全局topK={gv:.4f}")

    # 集中度：被选中样本落在多少天
    idx = np.argsort(-np.abs(z[m_te]))[:a.daily_k * n_days]
    u, c = np.unique(dte[idx], return_counts=True)
    print(f"[集中度] 覆盖 {len(u)}/{n_days} 天，单/天 {c.min()}~{c.max()}（均 {c.mean():.1f}）")

    accs = daily_topk_eval(z[m_te], y[m_te], dte, mt, a.daily_k)
    print(f"[按天top{a.daily_k}] 天数={len(accs)} 日均={accs.mean():.4f} 中位={np.median(accs):.4f} "
          f"p10={np.percentile(accs, 10):.4f} 最差={accs.min():.4f} ≥65%天={float((accs >= 0.65).mean()):.3f}")

    print(f"\n===== ETH vs BTC（同口径·单腿对比）=====")
    print(f"{'模型':<22}{'全局topK':>10}{'差值':>10}")
    print(f"{'BTC 盘口腿(150列)':<22}{BTC_REF['盘口腿']:>10.4f}{'—':>10}")
    print(f"{f'ETH 单腿({len(cols)}列)':<22}{gk:>10.4f}{gk - BTC_REF['盘口腿']:>+10.4f}")
    print(f"{'BTC 融合(w=0.2)':<22}{BTC_REF['融合 w=0.2']:>10.4f}{'—':>10}")
    print("注：BTC 融合值含独立的 faiss K 线腿，ETH 没有该腿，只能与「盘口腿」对比。")

    if a.save_pred:
        p_out = Path(a.save_pred)
        p_out.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame({
            "close_time": ct[m_te],
            "target": y[m_te],
            "score": z[m_te],
        }).write_parquet(p_out)
        print(f"[save] {p_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
