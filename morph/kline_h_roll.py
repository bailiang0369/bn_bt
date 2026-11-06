"""K 线腿多 horizon 在**滚动 z 口径**下的复核（此前 §26 的 h 扫描用的是 static val z）。

背景：§26 显示 K 线腿 3min 只有 0.6070（全局 topK），但 10min/15min 到 0.6509/0.6272；
而 §28 证明 static z 会虚高 3~4pp（盘口腿 0.6410 → roll_1d 0.6167）。
所以"长周期更好"这个结论**在诚实口径下是否成立，还没验过**。

本脚本：同一套 63 特征、同一行网格，对 h ∈ 3/5/10/15/30 各训一个 LGB，
在 off=0/60/120 三个时间窗上同时报 static / roll_1d / roll_3d / 按天 / AUC。
目的只有一个：判断「15min 事件合约」是否比现在的 3min（盘口腿 roll_1d 0.6167）更值得做。

口径与 morph/borrow_ab.py 完全一致（score_eval 直接复用），不引入新口径。
环境变量：HS / OFFSETS / SEEDS / ROUNDS / TRAIN_STRIDE
"""
import os
import sys
import time

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lightgbm as lgb  # noqa: E402

from morph.borrow_ab import score_eval  # noqa: E402
from morph.kline_boost import (OB_END, PARAMS, TEST_START, VAL_START,  # noqa: E402
                               build_features, load_klines, log)

DAY = 86_400_000


def fit_scores(X, y, tr, cols, seeds, rounds):
    acc = np.zeros(X.shape[0], dtype=np.float64)
    for sd in seeds:
        q = dict(PARAMS, seed=sd, bagging_seed=sd, feature_fraction_seed=sd, data_random_seed=sd)
        acc += lgb.train(q, lgb.Dataset(X[tr], y[tr], feature_name=cols),
                         num_boost_round=rounds).predict(X)
    return (acc / len(seeds)).astype(np.float32)


def main():
    hs = [int(x) for x in os.environ.get("HS", "3,5,10,15,30").split(",")]
    seeds = [int(x) for x in os.environ.get("SEEDS", "42,43").split(",")]
    rounds = int(os.environ.get("ROUNDS", "60"))
    stride = int(os.environ.get("TRAIN_STRIDE", "2"))
    offs = [int(x) for x in os.environ.get("OFFSETS", "0,60,120").split(",")]
    t0 = time.time()

    raw = load_klines().filter(pl.col("close_time") <= OB_END)
    ft, cols = build_features(raw)
    nmax = raw.height - max(hs)
    ct = ft.head(nmax)["close_time"].to_numpy()
    close = raw["close"].to_numpy().astype(np.float64)
    X = ft.head(nmax).select(cols).to_numpy().astype(np.float32)
    del ft, raw
    test_len = OB_END - TEST_START
    log(f"[data] rows={X.shape[0]} feats={len(cols)} h={hs}")

    rows = {}
    for off in offs:
        test_end = OB_END - off * DAY
        test_start = test_end - test_len
        val_start = VAL_START - off * DAY
        va_m = (ct >= val_start) & (ct < test_start)
        te_m = (ct >= test_start) & (ct < test_end)
        va_idx = np.where(va_m)[0]
        if len(va_idx) == 0 or te_m.sum() == 0:
            log(f"[off={off}] 空区间，跳过")
            continue
        for h in hs:
            y = (close[h:h + nmax] > close[:nmax]).astype(np.int8)
            tr = np.where(ct < val_start)[0]
            tr = tr[tr + h < int(va_idx[0])][::stride]
            s = fit_scores(X, y, tr, cols, seeds, rounds)
            r = score_eval(s, ct, y, va_m, te_m)
            r.update(h=h, off=off, train=len(tr))
            rows[(off, h)] = r
            log(f"[h={h:<2} off={off:<3}] static={r['static']:.4f} roll1d={r['roll1d']:.4f} "
                f"roll3d={r['roll3d']:.4f} AUC={r['auc']:.4f} 按天={r['daily']:.4f} "
                f"(基准{r['base']:.4f} n={r['n']} {r['days']}天)")

    log(f"\n{'h':>3}{'off':>5}{'static':>9}{'roll1d':>9}{'roll3d':>9}{'AUC':>8}{'按天':>8}")
    for (off, h), r in rows.items():
        log(f"{h:>3}{off:>5}{r['static']:>9.4f}{r['roll1d']:>9.4f}"
            f"{r['roll3d']:>9.4f}{r['auc']:>8.4f}{r['daily']:>8.4f}")

    log(f"\n{'h':>3}{'roll1d均值':>11}{'roll1d最差':>11}{'roll3d均值':>11}"
        f"{'static均值':>11}{'AUC均值':>9}{'按天均值':>10}")
    for h in hs:
        rs = [rows[(o, h)] for o in offs if (o, h) in rows]
        if not rs:
            continue
        log(f"{h:>3}{np.mean([r['roll1d'] for r in rs]):>11.4f}"
            f"{np.min([r['roll1d'] for r in rs]):>11.4f}"
            f"{np.mean([r['roll3d'] for r in rs]):>11.4f}"
            f"{np.mean([r['static'] for r in rs]):>11.4f}"
            f"{np.mean([r['auc'] for r in rs]):>9.4f}"
            f"{np.mean([r['daily'] for r in rs]):>10.4f}")
    log(f"\n[done] {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
