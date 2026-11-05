"""第十三轮：定稿二 —— seed 加到 5 + CNN 窗口 120/10 epoch/CVD 斜率加速度。

第十二轮给出 VP 是否入模的结论（读 boost12 日志）；本文件沿用它，同时做两件
与 VP 正交的事：

  LGB 腿 : 每个 horizon 的 seed 从 2 加到 5（round11 为 42,43）
  CNN 腿 : 通道 7 -> 9（+ cvd 斜率、cvd 加速度，仍然**不含总成交量**）
           窗口 60 -> 120，训练 5 -> 10 epoch
           训练抽样 stride 8 -> 16（相邻 1m 高度冗余，只是省 CPU）

融合权重照旧报 0.10/0.15/0.20/0.25 四条曲线，不在 test 上挑最优。
四个口径（static / 按天 / AUC / 4 窗口）全部输出。

环境变量：SEEDS、ROUNDS、USE_VP(0/1)、VP_WINS、CNN_SEEDS、CNN_EPOCHS、
          CNN_WINDOW、CNN_STRIDE
"""
import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from morph.kline_boost import (OB_END, TEST_START, VAL_START,  # noqa: E402
                               build_features, load_klines, log, zscore)
from morph.kline_dl2 import cnn_train  # noqa: E402
from morph.kline_multi import HS, evaluate  # noqa: E402
from morph.kline_vp import build_vp, fit_z  # noqa: E402


def build_chans9(df):
    """9 通道：7 条定稿通道 + CVD 斜率、CVD 加速度（无总成交量）。"""
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
    if "taker_buy_volume" not in df.columns or "volume" not in df.columns:
        raise SystemExit("[chans9] 数据缺 volume/taker_buy_volume，无法生成 CVD 通道")
    v = df["volume"].fill_null(0).to_numpy().astype(np.float64)
    tb = df["taker_buy_volume"].fill_null(0).to_numpy().astype(np.float64)
    cvd = np.cumsum(2.0 * tb - v) / 1e6          # 合约口径 CVD

    def diff5(x):
        y = np.zeros_like(x)
        y[5:] = x[5:] - x[:-5]
        return y

    sl = diff5(cvd)
    ac = diff5(sl)
    chans += [cvd, sl, ac]
    names += ["cvd", "cvd_sl", "cvd_acc"]
    norm = np.array([True] * (len(chans) - 3) + [False, True, True])
    x = np.vstack([np.nan_to_num(a, nan=0.0).astype(np.float32).reshape(1, -1)
                   for a in chans])
    log(f"  [chans9] {names} shape={x.shape}")
    return x, names, norm


def main():
    seeds = [int(x) for x in os.environ.get("SEEDS", "42,43,44,45,46").split(",")]
    cnn_seeds = [int(x) for x in os.environ.get("CNN_SEEDS", "42,43").split(",")]
    epochs = int(os.environ.get("CNN_EPOCHS", "10"))
    window = int(os.environ.get("CNN_WINDOW", "120"))
    stride = int(os.environ.get("CNN_STRIDE", "16"))
    rounds = int(os.environ.get("ROUNDS", "60"))
    use_vp = os.environ.get("USE_VP", "1") == "1"
    nmax_h = max(HS)

    raw = load_klines().filter(pl.col("close_time") <= OB_END)
    ft, cols = build_features(raw)
    nmax = raw.height - nmax_h
    use = list(cols)
    if use_vp:
        vp, vpcols = build_vp(raw)
        use += vpcols
    ct = ft.head(nmax)["close_time"].to_numpy()
    close = raw["close"].to_numpy().astype(np.float64)
    log(f"[data] rows={nmax} cols={len(use)} vp={use_vp}")

    tr = np.where(ct < VAL_START)[0]
    va = np.where((ct >= VAL_START) & (ct < TEST_START))[0]
    te = np.where(ct >= TEST_START)[0]
    va = va[va + nmax_h < int(te[0])]
    tr = tr[tr + nmax_h < int(va[0])]
    pr = np.concatenate([va, te])
    nv = len(va)
    log(f"[split] train={len(tr)} val={len(va)} test={len(te)}")

    X = ft.head(nmax).select(cols).to_numpy().astype(np.float32)
    if use_vp:
        X = np.hstack([X, vp[:nmax]])
    Xtr, Xpr = X[tr], X[pr]
    del X
    y3 = (close[3:3 + nmax] > close[:nmax]).astype(np.int8)

    rows = {}
    zs = {}
    for h in HS:
        y = (close[h:h + nmax] > close[:nmax]).astype(np.int8)
        zs[h], _ = fit_z(Xtr, Xpr, y, tr, va, nv, seeds, rounds, use)
        log(f"[lgb h={h}] done ({len(seeds)} seeds)")
    del Xtr
    base_z = np.mean([zs[h] for h in HS], axis=0)
    full = np.zeros(nmax)
    full[pr] = base_z
    evaluate("lgb5s", full, ct, y3, va, rows)

    log(f"[cnn9] 窗口={window} epoch={epochs} stride={stride} 9 通道（无总成交量）")
    chans, names, norm = build_chans9(ft.head(nmax))
    pc, cva, cte = cnn_train(chans, norm, y3, ct, window, cnn_seeds, epochs, stride)
    m = ~np.isnan(pc)
    pcf = np.where(m, pc, np.nanmedian(pc[m]))
    fullc = np.zeros(nmax)
    fullc[pr] = zscore(pcf[pr], pcf[cva])
    evaluate("cnn9", fullc, ct, y3, va, rows)

    for w in (0.10, 0.15, 0.20, 0.25):
        full[pr] = base_z + w * fullc[pr]
        evaluate(f"lgb5s+{w}cnn9", full, ct, y3, va, rows)

    log(f"\n{'变体':<16}{'static':>9}{'AUC':>8}{'按天':>8}{'窗口4':>8}{'4段sd':>8}")
    for k, r in rows.items():
        log(f"{k:<16}{r['static']:>9.4f}{r['auc']:>8.4f}{r['daily']:>8.4f}"
            f"{r['w4']:>8.4f}{r['w4sd']:>8.4f}  明细={r['wins']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
