"""ETH 对齐 BTC 特征集后重跑 3min 评估（同数据处理 + 同训练口径）。

对齐方式完全复用 `orderbook/extend_features.py` 对 BTC 做过的那两件事：
  1) 删掉订单簿逐档分箱列 `_(ask|bid)_b[0-4]`（ETH 新建时带了 60 个，BTC 早先已删）
  2) 补 stoch(60,2,2) + supertrend(10,3.0)（BTC 有，ETH 的 run_eth 流程漏了）

对齐后列集合应与 BTC 完全一致，此时再比 0.6410 才有意义。

本脚本自包含：自己 clone 仓库拿 tool/pl_exp.py，所以不依赖 git push。
`orderbook/fusion_test.py` 顶层 import faiss（direct 模式不想多装），故这里把它那 6 个
原语逐字复制过来，口径不变。
"""
import glob
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/bailiang0369/bn_bt"
ROOT = "/tmp/bn_bt"

if not os.path.isdir(ROOT):
    subprocess.run(["git", "clone", "--depth=1", REPO, ROOT], check=True)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tool"))

try:
    import lightgbm as lgb  # noqa: E402
except ImportError:  # Kaggle 一般自带，缺了就装一下
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "lightgbm"], check=True)
    import lightgbm as lgb  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from pl_exp import PolarsExp  # noqa: E402

OB_REG = dict(num_leaves=31, min_child_samples=500, feature_fraction=0.6, lambda_l2=5.0)
BTC_REF = {"盘口腿": 0.6410, "融合 w=0.2": 0.6465}


def log(m: str) -> None:
    print(m, flush=True)


# ---- 以下 6 个函数逐字复制自 orderbook/fusion_test.py（+ morph/morph.py 的 add_target）----
def add_target(df, horizon=3):
    return df.with_columns(
        (pl.col("close").shift(-horizon) > pl.col("close")).cast(pl.Int8).alias("target"))


def day_index(ct_ms):
    return (np.asarray(ct_ms) // 86_400_000).astype(np.int64)


def complete_days(day_id, min_bars=1200):
    u, c = np.unique(day_id, return_counts=True)
    return np.isin(day_id, u[c >= min_bars])


def global_topk_eval(score, y, day_id, k_per_day):
    n_days = len(np.unique(day_id))
    k = min(int(k_per_day) * n_days, len(y))
    idx = np.argsort(-np.abs(score))[:k]
    return float(((score[idx] > 0).astype(int) == y[idx]).mean())


def daily_topk_eval(score, y, day_id, mask, k):
    s, yy, d = score[mask], y[mask], day_id[mask]
    accs = []
    for u in np.unique(d):
        m = d == u
        kk = min(k, int(m.sum()))
        idx = np.argsort(-np.abs(s[m]))[:kk]
        accs.append(float(((s[m][idx] > 0).astype(int) == yy[m][idx]).mean()))
    return np.array(accs, dtype=np.float64)


def train_lgb_avg(X, y, tr, va, cols, seeds, rounds, weight=None, extra=None):
    probs = np.zeros(len(y), dtype=np.float64)
    iters = []
    for sd in seeds:
        params = dict(objective="binary", metric="auc", learning_rate=0.05, num_leaves=63,
                      feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1,
                      min_child_samples=100, verbose=-1,
                      seed=sd, bagging_seed=sd, feature_fraction_seed=sd, data_random_seed=sd)
        if extra:
            params.update(extra)
        w_tr = None if weight is None else weight[tr]
        w_va = None if weight is None else weight[va]
        ds = lgb.Dataset(X[tr], y[tr], feature_name=cols, weight=w_tr)
        vs = lgb.Dataset(X[va], y[va], reference=ds, weight=w_va)
        m = lgb.train(params, ds, num_boost_round=rounds, valid_sets=[ds, vs],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        probs += m.predict(X, num_iteration=m.best_iteration)
        iters.append(int(m.best_iteration))
    return (probs / len(seeds)).astype(np.float32), iters


def find(sym: str) -> str:
    exact = f"{sym}.parquet"
    hits = [h for h in glob.glob(f"/kaggle/input/**/{exact}", recursive=True)
            if os.path.basename(h) == exact]
    if not hits:
        hits = [str(p) for p in (Path(ROOT) / "data").rglob(exact)]
    if not hits:
        raise FileNotFoundError(sym)
    log(f"[data] {hits[0]}")
    return hits[0]


def align_like_btc(df: pl.DataFrame, ref_cols: list[str]) -> pl.DataFrame:
    n0, c0 = df.height, df.width
    bin_re = re.compile(r"_(ask|bid)_b[0-4]")
    drop = [c for c in df.columns if bin_re.search(c)]
    if drop:
        df = df.drop(drop)
    log(f"[align] 删分箱列 {len(drop)} 个 -> {df.width} 列")

    pe = PolarsExp()
    if "stoch_k_60_2" not in df.columns:
        k, d = pe.stoch("high", "low", "close", 60, 2, 2, alias_suffix=True)
        df = df.with_columns([k, d])
        log("[align] 补 stoch(60,2,2)")
    if "supertrend" not in df.columns:
        df = pe.supertrend(df, "high", "low", "close", 10, 3.0)
        log("[align] 补 supertrend(10,3.0)")

    only_eth = sorted(set(df.columns) - set(ref_cols))
    only_btc = sorted(set(ref_cols) - set(df.columns))
    log(f"[align] {n0}x{c0} -> {df.height}x{df.width}")
    log(f"[align] 仅 ETH 有={only_eth}  仅 BTC 有={only_btc}")
    if not only_eth and not only_btc:
        log("[align] ✅ 列集合与 BTC 完全一致")
    return df


def evaluate(df: pl.DataFrame, h: int, seeds: list[int], rounds: int, daily_k: int,
             tag: str, save: str | None) -> dict:
    cols = [c for c in df.columns if c not in ("target", "close_time")]
    L = len(df)
    ct = df["close_time"].to_numpy()
    y = df["target"].to_numpy().astype(np.int8)
    X = df.select(cols).to_numpy().astype(np.float32)

    k1, k2 = int(L * 0.64), int(L * 0.80)
    tr, va = slice(0, k1 - h), slice(k1, k2 - h)
    val_start, test_start = int(ct[k1]), int(ct[k2])
    log(f"\n[{tag}] rows={L} feats={len(cols)} val_start={val_start} test_start={test_start}")

    p, iters = train_lgb_avg(X, y, tr, va, cols, seeds, rounds, None, OB_REG)
    log(f"[{tag}] iters={iters}")

    dyn = day_index(ct)
    m_val = (ct >= val_start) & (ct < test_start)
    m_te = ct >= test_start
    dte = dyn[m_te]
    mt = complete_days(dte)
    n_days = len(np.unique(dte))
    z = (p - p[m_val].mean()) / (p[m_val].std() + 1e-9)

    def gacc(mask):
        idx = np.where(mask)[0]
        return global_topk_eval(z[idx], y[idx], dyn[idx], daily_k)

    auc = float(roc_auc_score(y[m_te], z[m_te]))
    accs = daily_topk_eval(z[m_te], y[m_te], dte, mt, daily_k)
    gk = gacc(m_te)
    dk = float(accs.mean()) if len(accs) else float("nan")
    log(f"[{tag}] test n={int(m_te.sum())} 天数={n_days} 完整天={int(len(accs))} 基准={float(y[m_te].mean()):.4f}")
    log(f"[{tag}] AUC={auc:.4f}  全局topK={gk:.4f}  按天topK={dk:.4f}  val全局={gacc(m_val):.4f}")

    if len(accs):
        log(f"[{tag}] 按天明细 日均={accs.mean():.4f} 中位={np.median(accs):.4f} "
            f"p10={np.percentile(accs, 10):.4f} 最差={accs.min():.4f} ≥65%天={float((accs >= 0.65).mean()):.3f}")
    else:
        log(f"[{tag}] [warn] 无完整交易日（complete_days min_bars=1200 全部落选），按天口径不可用")

    if save:
        pl.DataFrame({"close_time": ct[m_te], "target": y[m_te], "score": z[m_te]}).write_parquet(save)
        log(f"[save] {save}")
    return {"auc": auc, "g": gk, "d": dk, "f": len(cols)}


def main() -> int:
    h = int(os.environ.get("KAGGLE_HORIZON", "3"))
    seeds = [int(s) for s in os.environ.get("KAGGLE_OSEEDS", "42,43,44,45,46").split(",") if s.strip()]
    rounds = int(os.environ.get("ROUNDS", "60"))
    daily_k = int(os.environ.get("DAILY_K", "15"))
    save = os.environ.get("ALIGNED_OUT", "/kaggle/working/eth_aligned/ETHUSDT.parquet")
    pred_out = os.environ.get("KAGGLE_FUSION_PRED", "/kaggle/working/eth_pred_aligned.parquet")

    btc_cols = list(pl.read_parquet_schema(find("BTCUSDT")).keys())
    log(f"[ref] BTC cols={len(btc_cols)}")

    raw = pl.read_parquet(find("ETHUSDT")).sort("close_time")
    log(f"[eth] raw rows={raw.height} cols={raw.width}")

    aligned = align_like_btc(raw, btc_cols)
    aligned = add_target(aligned, h).filter(pl.col("target").is_not_null())

    if save:
        os.makedirs(os.path.dirname(save), exist_ok=True)
        aligned.write_parquet(save)
        log(f"[save] 对齐后数据 -> {save} ({aligned.height}x{aligned.width})")

    r = evaluate(aligned, h, seeds, rounds, daily_k, "ETH对齐", pred_out)

    eth_name = f"ETH 对齐后({r['f']}特征)"
    log("\n===== 公平对比（同特征集 / 同口径）=====")
    log(f"{'模型':<24}{'AUC':>8}{'全局topK':>10}{'差值':>9}")
    log(f"{'BTC 盘口腿(150特征)':<24}{0.5412:>8.4f}{BTC_REF['盘口腿']:>10.4f}{'—':>9}")
    log(f"{eth_name:<24}{r['auc']:>8.4f}{r['g']:>10.4f}{r['g'] - BTC_REF['盘口腿']:>+9.4f}")
    log(f"{'BTC 融合(w=0.2)':<24}{'—':>8}{BTC_REF['融合 w=0.2']:>10.4f}{'—':>9}")
    log("\n上一轮 ETH 未对齐(206特征) 全局topK=0.5806，可与本次对照看逐档特征是否有用。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
