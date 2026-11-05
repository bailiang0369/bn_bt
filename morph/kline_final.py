"""第十一轮：63 特征多 horizon LGB + 无总成交量 CNN 融合。

⚠ 第十二~十四轮之后，**定稿已迁到 `morph/kline_final2.py`**（USE_VP=0、CNN_CHANS=7、
  CNN_WINDOW=60、CNN_EPOCHS=5、SEEDS=42..46，launcher 项目 `boost14`）：
  seed 2→5 不改 AUC（0.5285），但按天 topK 0.5937→0.5979、4 窗口 sd 0.0260→0.0234。
  本文件保留为第 11 轮结论的可复现入口（2 seed 版本）。

来自前十轮的确定性结论：
  - topK 的噪声地板：seed sd 0.47pt、窗口 sd 3.0pt，1pt 以内不可分；
  - AUC 是所有改动里唯一能区分"有没有新信息"的指标，前八轮全部锁死 0.5269；
  - 多 horizon（h=1,2,3,5,10）等权集成把 AUC 推到 0.5277，4/4 窗口同时提升；
  - CNN 的 vol 通道（log1p 总成交量）是负贡献，去掉后 0.5382 -> 0.5514；
  - CNN（AUC 0.5237）与 LGB（0.5269）相关系数仅 0.198，属弱但互补的腿。

本轮把三者组合，扫 CNN 权重（0.1~0.3，不调参只报曲线）：
  base   = 63 特征 × 5 horizon × 2 seed 的 z 平均
  +cnn   = base + w * z(CNN7)
CNN7 通道：px / body / upper / lower / ret1 / st / cvd(合约)，无总成交量。
"""
import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lightgbm as lgb  # noqa: E402

from morph.kline_boost import (OB_END, PARAMS, TEST_START, VAL_START,  # noqa: E402
                               build_features, load_klines, log, zscore)
from morph.kline_dl2 import cnn_train  # noqa: E402
from morph.kline_multi import HS, evaluate  # noqa: E402
from morph.kline_novol import build_chans7  # noqa: E402


def main():
    seeds = [int(x) for x in os.environ.get("SEEDS", "42,43").split(",")]
    epochs = int(os.environ.get("CNN_EPOCHS", "5"))
    stride = int(os.environ.get("CNN_STRIDE", "8"))
    rounds = int(os.environ.get("ROUNDS", "60"))

    raw = load_klines().filter(pl.col("close_time") <= OB_END)
    ft, cols = build_features(raw)
    nmax = raw.height - max(HS)
    X = ft.head(nmax).select(cols).to_numpy().astype(np.float32)
    ct = ft.head(nmax)["close_time"].to_numpy()
    close = raw["close"].to_numpy().astype(np.float64)
    log(f"[data] rows={nmax} cols={len(cols)}")

    tr = np.where(ct < VAL_START)[0]
    va = np.where((ct >= VAL_START) & (ct < TEST_START))[0]
    tr = tr[tr + max(HS) < int(va[0])]

    zs = {}
    for h in HS:
        y = (close[h:h + nmax] > close[:nmax]).astype(np.int8)
        acc = np.zeros(nmax)
        for sd in seeds:
            p = dict(PARAMS, seed=sd, bagging_seed=sd, feature_fraction_seed=sd,
                     data_random_seed=sd)
            acc += lgb.train(p, lgb.Dataset(X[tr], y[tr], feature_name=cols),
                             num_boost_round=rounds).predict(X)
        ph = acc / len(seeds)
        zs[h] = zscore(ph, ph[va])
        log(f"[lgb h={h}] done")
    del X

    y3 = (close[3:3 + nmax] > close[:nmax]).astype(np.int8)
    base = np.mean([zs[h] for h in HS], axis=0)

    rows = {}
    evaluate("lgb_multi", base, ct, y3, va, rows)

    log("[cnn7] 无总成交量 7 通道")
    chans, names, norm = build_chans7(ft.head(nmax))
    pc, _, _ = cnn_train(chans, norm, y3, ct, 60, seeds, epochs, stride)
    m = ~np.isnan(pc)
    pcf = np.where(m, pc, np.nanmedian(pc[m]))
    zc = zscore(pcf, pcf[va])
    evaluate("cnn7", pcf, ct, y3, va, rows)

    for w in (0.10, 0.15, 0.20, 0.25):
        evaluate(f"multi+{w}cnn", base + w * zc, ct, y3, va, rows)

    log(f"\n{'变体':<16}{'static':>9}{'AUC':>8}{'按天':>8}{'窗口4':>8}{'4段sd':>8}")
    for k, r in rows.items():
        log(f"{k:<16}{r['static']:>9.4f}{r['auc']:>8.4f}{r['daily']:>8.4f}"
            f"{r['w4']:>8.4f}{r['w4sd']:>8.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
