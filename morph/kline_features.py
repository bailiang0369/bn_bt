"""1m K 线特征工程（供 3min 方向模型用）。

设计原则：
  1. 全部用 rolling/shift（右对齐），**任何特征只用当前 bar 及之前的数据**；
  2. 不含 target，不 drop 行，缺口由 LightGBM 原生处理（NaN）；
  3. 覆盖 5 类：动量 / 波动 / 量能（需要 volume+taker_buy_volume）/ 形态指标 / 时间。

为什么值得做：K 线有 2020-2026 共 3,549,600 行（orderbook 只有 15 个月 658,712 行），
此前 faiss/morph 只用了 close + stoch 两个通道，剩下 4.4 倍的数据量完全没利用。

用法：
    from morph.kline_features import build_features
    df2, cols = build_features(df)      # df 需含 open/high/low/close/close_time
"""
import numpy as np
import polars as pl

# 需要 volume / taker_buy_volume 才能算的量能特征
VOL_FEATS = ["log_vol", "vr_5", "vr_15", "vr_30", "vr_60", "vr_120",
             "tbr_1", "tbr_5", "tbr_15", "tbr_30", "tbrs_5", "tbrs_15", "tbrs_30",
             "net_1", "net_3", "net_5", "net_15", "vpin_15", "vpin_30"]


def build_features(df):
    """返回 (带特征的新 df, 特征列名列表)。不修改 df。"""
    c, h, l, o = pl.col("close"), pl.col("high"), pl.col("low"), pl.col("open")
    r1 = c / c.shift(1) - 1.0
    e, cols = [], []

    def add(expr, name):
        e.append(expr.cast(pl.Float64).alias(name))
        cols.append(name)

    # ---- 1) 动量：多尺度收益 ----
    for n in (1, 2, 3, 5, 10, 15, 30, 60, 120):
        add(c / c.shift(n) - 1.0, f"k_ret_{n}")

    # ---- 2) 波动 ----
    for w in (5, 15, 30, 60, 120):
        add(r1.rolling_std(w), f"k_vol_{w}")
        add(r1.abs().rolling_mean(w), f"k_absret_{w}")
    tr = pl.max_horizontal(h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs())
    for w in (14, 30, 60):
        add(tr.rolling_mean(w) / c, f"k_atr_{w}")
    # 区间位置（close 在近 w 根高低点中的位置）
    for w in (15, 30, 60, 120):
        lo, hi = l.rolling_min(w), h.rolling_max(w)
        add((c - lo) / (hi - lo + 1e-9), f"k_pos_{w}")

    # ---- 3) 量能 ----
    if "volume" in df.columns:
        v = pl.col("volume").cast(pl.Float64)
        add(v.log1p(), "log_vol")
        for w in (5, 15, 30, 60, 120):
            add(v / (v.rolling_mean(w) + 1e-9), f"vr_{w}")
        if "taker_buy_volume" in df.columns:
            tb = pl.col("taker_buy_volume").cast(pl.Float64)
            ratio = tb / (v + 1e-9)
            for w in (1, 5, 15, 30):
                add(ratio.rolling_mean(w), f"tbr_{w}")
            for w in (5, 15, 30):
                add(ratio.rolling_std(w), f"tbrs_{w}")
            net = 2.0 * tb - v
            for w in (1, 3, 5, 15):
                add(net.rolling_sum(w) / (v.rolling_sum(w) + 1e-9), f"net_{w}")
            # VPIN 代理：|净主动买| / 总量
            for w in (15, 30):
                add(net.rolling_sum(w).abs() / (v.rolling_sum(w) + 1e-9), f"vpin_{w}")

    # ---- 4) 形态 / 技术指标 ----
    rng = (h - l)
    add((c - o) / (rng + 1e-9), "k_body")
    add((h - pl.max_horizontal(c, o)) / (rng + 1e-9), "k_upper")
    add((pl.min_horizontal(c, o) - l) / (rng + 1e-9), "k_lower")
    add(o / c.shift(1) - 1.0, "k_gap")
    up = (c - c.shift(1)).clip(lower_bound=0.0)
    dn = (c.shift(1) - c).clip(lower_bound=0.0)
    for w in (14, 30, 60):
        add(up.rolling_mean(w) / (up.rolling_mean(w) + dn.rolling_mean(w) + 1e-9), f"k_rsi_{w}")
    for w in (20, 60):
        ma, sd = c.rolling_mean(w), c.rolling_std(w)
        add((c - ma) / (sd + 1e-9), f"k_boll_{w}")
    for w in (29, 69):
        lo, hi = l.rolling_min(w), h.rolling_max(w)
        add((c - lo) / (hi - lo + 1e-9), f"k_stoch_{w}")
    streak = pl.when(r1 > 0).then(1.0).otherwise(-1.0)
    add(streak.rolling_sum(3), "k_streak3")
    add(streak.rolling_sum(10), "k_streak10")

    # ---- 5) 时间（周期编码）----
    ct = pl.col("close_time")
    hour = (ct % 86_400_000) / 3_600_000.0
    dow = (ct // 86_400_000) % 7
    add((hour * 2 * np.pi / 24).sin(), "hour_sin")
    add((hour * 2 * np.pi / 24).cos(), "hour_cos")
    add((dow * 2 * np.pi / 7).sin(), "dow_sin")
    add((dow * 2 * np.pi / 7).cos(), "dow_cos")
    add(((hour >= 13.5) & (hour < 20.0)).cast(pl.Int8), "us_session")

    out = df.with_columns(e)
    return out, cols
