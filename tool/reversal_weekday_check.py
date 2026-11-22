"""本地因果拆解：h3/h10 信号是反转还是顺势，按周末/工作日分别统计。

只用项目内已有产物，不访问网络、不推 Kaggle。
"""
import polars as pl
from sklearn.metrics import roc_auc_score

BASE = r"e:/codebuddy_wrokspace/bn_bt/.codebuddy/artifacts"
KL_PATH = r"e:/codebuddy_wrokspace/bn_hisdata/data/vision/BTCUSDT_1m_klines_2026.parquet"
H3_PATH = f"{BASE}/h3gate/h3_gate_pred.parquet"
H10_PATH = f"{BASE}/livesim/live_sim/events.parquet"
THR = 0.05806
PAYOUT = 0.8


def sgn(col):
    return pl.when(col > 0).then(1).when(col < 0).then(-1).otherwise(0)


def ev(win):
    return PAYOUT * win - (1 - win)


def ci(win, n, overlap):
    neff = max(n / overlap, 1)
    se = (win * (1 - win) / neff) ** 0.5
    return f"[{win - 1.96 * se:.4f}, {win + 1.96 * se:.4f}]"


kl = (
    pl.read_parquet(KL_PATH)
    .select(["close_time", "close"])
    .sort("close_time")
    .with_columns(
        (pl.from_epoch("close_time", time_unit="ms").dt.weekday() >= 6)
        .alias("weekend"),
        (pl.col("close") / pl.col("close").shift(15) - 1).alias("ret15"),
        (pl.col("close") / pl.col("close").shift(60) - 1).alias("ret60"),
        (pl.col("close").shift(-3) > pl.col("close")).cast(pl.Int8).alias("tg3"),
        (pl.col("close").shift(-10) > pl.col("close")).cast(pl.Int8).alias("f10"),
    )
)

print("=" * 78)
print("A. h3（time split，逐 bar 分数；标签 close[t+3] > close[t]）")
print("=" * 78)
h3 = (
    pl.read_parquet(H3_PATH)
    .filter(pl.col("split") == "time")
    .join(kl, on="close_time", how="inner")
    .with_columns(
        sgn(pl.col("score") - 0.5).alias("dir"),
        (pl.col("score") - 0.5).abs().alias("mag"),
    )
    .with_columns(
        (pl.col("mag") >= THR).alias("trig"),
        (pl.col("dir") == (2 * pl.col("target") - 1)).alias("win"),
        (pl.col("dir") * sgn(pl.col("ret15")) < 0).alias("rev15"),
        (pl.col("dir") * sgn(pl.col("ret15")) > 0).alias("mom15"),
        (pl.col("dir") * sgn(pl.col("ret60")) < 0).alias("rev60"),
        (pl.col("dir") * sgn(pl.col("ret60")) > 0).alias("mom60"),
    )
)
assert (h3["target"] == h3["tg3"]).all(), "h3 标签与 BTC K 线不一致"

for seg in ["val", "test"]:
    d = h3.filter(pl.col("segment") == seg)
    print(f"\n-- segment={seg} --")
    for wk in [False, True]:
        s = d.filter(pl.col("weekend") == wk)
        t = s.filter(pl.col("trig"))
        auc = roc_auc_score(s["target"].to_numpy(), s["score"].to_numpy())
        wr = t["win"].mean() if len(t) else float("nan")
        print(
            f"{'周末' if wk else '工作日'}: bars={len(s)}, AUC={auc:.4f}, "
            f"触发={len(t)}, 触发率={len(t)/len(s):.4%}, "
            f"胜率={wr:.4f}{ci(wr, len(t), 3) if len(t) else ''}, EV={ev(wr):+.4f}"
        )
        for name, a, b in [("ret15", "rev15", "mom15"), ("ret60", "rev60", "mom60")]:
            parts = []
            for lab, flag in [("逆势", a), ("顺势", b)]:
                x = t.filter(pl.col(flag) & (sgn(pl.col(name)) != 0))
                w = x["win"].mean() if len(x) else float("nan")
                parts.append(f"{lab} n={len(x)} win={w:.4f} EV={ev(w):+.4f}")
            print(f"    {name}: " + " | ".join(parts))

print("\n-- 盲基线（不看模型，只按过去收益方向押注，h3）--")
for seg in ["val", "test"]:
    d = h3.filter((pl.col("segment") == seg) & (pl.col("mag") > 0.005))
    for wk in [False, True]:
        s = d.filter(pl.col("weekend") == wk)
        parts = []
        for name, mult in [("ret15", 1), ("ret15", -1), ("ret60", 1), ("ret60", -1)]:
            pred = sgn(pl.col(name)) * mult
            w = s.with_columns((pred == (2 * pl.col("target") - 1)).alias("ok"))["ok"].mean()
            parts.append(f"{'顺势' if mult > 0 else '逆势'}{name}={w:.4f}")
        print(f"{seg} {'周末' if wk else '工作日'} n={len(s)}: " + " ".join(parts))

print("\n" + "=" * 78)
print("B. h10（live_sim 逐 bar replay，2026-01~10，3 槽位接受单）")
print("=" * 78)
e = (
    pl.read_parquet(H10_PATH)
    .filter(pl.col("accepted"))
    .join(
        kl.select(
            ["close_time", "weekend", "ret15", "ret60"],
        ),
        on="close_time",
        how="inner",
    )
    .with_columns(
        (2 * pl.col("direction") - 1).alias("dir"),
        (pl.col("direction") == pl.col("target")).alias("win"),
    )
    .with_columns(
        (pl.col("dir") * sgn(pl.col("ret15")) < 0).alias("rev15"),
        (pl.col("dir") * sgn(pl.col("ret15")) > 0).alias("mom15"),
        (pl.col("dir") * sgn(pl.col("ret60")) < 0).alias("rev60"),
        (pl.col("dir") * sgn(pl.col("ret60")) > 0).alias("mom60"),
    )
)
for wk in [False, True]:
    s = e.filter(pl.col("weekend") == wk)
    w = s["win"].mean()
    print(
        f"{'周末' if wk else '工作日'}: n={len(s)}, 胜率={w:.4f}{ci(w, len(s), 10)}, "
        f"EV@0.8={ev(w):+.4f}"
    )
    for name, a, b in [("ret15", "rev15", "mom15"), ("ret60", "rev60", "mom60")]:
        parts = []
        for lab, flag in [("逆势", a), ("顺势", b)]:
            x = s.filter(pl.col(flag) & (sgn(pl.col(name)) != 0))
            v = x["win"].mean() if len(x) else float("nan")
            parts.append(f"{lab} n={len(x)} win={v:.4f} EV={ev(v):+.4f}")
        print(f"    {name}: " + " | ".join(parts))
print("\n-- 盲基线（h10，按过去收益方向押注）--")
for wk in [False, True]:
    s = kl.filter((pl.col("weekend") == wk) & (pl.col("ret15").abs() > 0.0005))
    parts = []
    for name, mult in [("ret15", 1), ("ret15", -1), ("ret60", 1), ("ret60", -1)]:
        pred = sgn(pl.col(name)) * mult
        w = s.with_columns((pred == (2 * pl.col("f10") - 1)).alias("ok"))["ok"].mean()
        parts.append(f"{'顺势' if mult > 0 else '逆势'}{name}={w:.4f}")
    print(f"{'周末' if wk else '工作日'} n={len(s)}: " + " ".join(parts))
