"""第十轮：按"总成交量是垃圾指标"的口径重建输入。

口径纠正：
  1. CNN 之前有 vol 通道 = log1p(总成交量)，删掉。通道改为
     px / body / upper / lower / ret1 / st / cvd 共 7 条；
  2. cvd 是**合约**口径：2 * taker_buy_volume - volume，数据源为
     futures/um/monthly/klines 的 taker_buy_volume，不是现货；
  3. LGB 侧剔除 build_features 里 19 个量能衍生列：
       log_vol, vr_*          —— 纯总成交量
       tbr_*, tbrs_*          —— 以总成交量为分母的主动买占比
       net_*, vpin_*          —— 以总成交量为分母的净主动买
     余下 44 列全是价格/形态/时间/波动，不含任何总量信息。

对照：同一份数据、同一 test 段、同一口径下，63 特征的多 horizon 集成
（第九轮实测 static 0.6174）vs 44 特征的多 horizon 集成。
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

BAN = (["log_vol"] + [f"vr_{w}" for w in (5, 15, 30, 60, 120)]
       + [f"tbr_{w}" for w in (1, 5, 15, 30)]
       + [f"tbrs_{w}" for w in (5, 15, 30)]
       + [f"net_{w}" for w in (1, 3, 5, 15)]
       + [f"vpin_{w}" for w in (15, 30)])


def build_chans7(df):
    """7 通道：价格路径 + K 线形状 + 合约 CVD（无总成交量）。"""
    c = df["close"].to_numpy().astype(np.float64)
    o = df["open"].to_numpy().astype(np.float64)
    h = df["high"].to_numpy().astype(np.float64)
    l = df["low"].to_numpy().astype(np.float64)
    chans = [c, (c - o) / c * 1e4, (h - np.maximum(c, o)) / c * 1e4,
             (np.minimum(c, o) - l) / c * 1e4]
    names = ["px", "body", "upper", "lower"]
    r1 = np.zeros_like(c)
    r1[1:] = (c[1:] / c[:-1] - 1.0) * 1e4
    chans.append(r1)
    names.append("ret1")
    lo = pl.Series(l).rolling_min(29).to_numpy()
    hi = pl.Series(h).rolling_max(29).to_numpy()
    chans.append(np.nan_to_num((c - lo) / np.maximum(hi - lo, 1e-9), nan=0.5))
    names.append("st")
    if "taker_buy_volume" in df.columns and "volume" in df.columns:
        v = df["volume"].fill_null(0).to_numpy().astype(np.float64)
        tb = df["taker_buy_volume"].fill_null(0).to_numpy().astype(np.float64)
        chans.append(np.cumsum(2.0 * tb - v) / 1e6)   # 合约 CVD
        names.append("cvd")
    norm = np.array([True] * (len(chans) - 1) + [False])
    x = np.vstack([np.nan_to_num(v, nan=0.0).astype(np.float32).reshape(1, -1)
                   for v in chans])
    log(f"  [chans7] {names} shape={x.shape}")
    return x, names, norm


def main():
    seeds = [int(x) for x in os.environ.get("SEEDS", "42,43").split(",")]
    epochs = int(os.environ.get("CNN_EPOCHS", "5"))
    stride = int(os.environ.get("CNN_STRIDE", "8"))
    rounds = int(os.environ.get("ROUNDS", "60"))

    raw = load_klines().filter(pl.col("close_time") <= OB_END)
    ft, cols = build_features(raw)
    nmax = raw.height - max(HS)
    keep = [c for c in cols if c not in BAN]
    log(f"[cols] 全量={len(cols)} 剔除={len(cols) - len(keep)} 剩余={len(keep)}")
    log(f"[cols] 剔除项={[c for c in cols if c in BAN]}")

    X = ft.head(nmax).select(keep).to_numpy().astype(np.float32)
    ct = ft.head(nmax)["close_time"].to_numpy()
    close = raw["close"].to_numpy().astype(np.float64)
    log(f"[data] rows={nmax}")

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
            acc += lgb.train(p, lgb.Dataset(X[tr], y[tr], feature_name=keep),
                             num_boost_round=rounds).predict(X)
        ph = acc / len(seeds)
        zs[h] = zscore(ph, ph[va])
        log(f"[h={h}] done")

    y3 = (close[3:3 + nmax] > close[:nmax]).astype(np.int8)
    rows = {}
    evaluate("novol_h3", zs[3], ct, y3, va, rows)
    evaluate("novol_mean", np.mean([zs[h] for h in HS], axis=0), ct, y3, va, rows)

    log("[cnn7] 7 通道（无总成交量）")
    chans, names, norm = build_chans7(ft.head(nmax))
    pc, _, _ = cnn_train(chans, norm, y3, ct, 60, seeds, epochs, stride)
    m = ~np.isnan(pc)
    pcf = np.where(m, pc, np.nanmedian(pc[m]))
    evaluate("cnn7", pcf, ct, y3, va, rows)
    zc = zscore(pcf, pcf[va])
    evaluate("novol_mean+cnn7", np.mean([zs[h] for h in HS], axis=0) + 0.15 * zc,
             ct, y3, va, rows)

    log(f"\n{'变体':<18}{'static':>9}{'AUC':>8}{'按天':>8}{'窗口4':>8}{'4段sd':>8}")
    for k, r in rows.items():
        log(f"{k:<18}{r['static']:>9.4f}{r['auc']:>8.4f}{r['daily']:>8.4f}"
            f"{r['w4']:>8.4f}{r['w4sd']:>8.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
