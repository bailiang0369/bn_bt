#!/usr/bin/env python3
"""h=10 事件合约：**因果阈值**实盘口径（无前视），目标胜率 ≥60%。

为什么需要这个文件（与 kline_long.py 的区别）
--------------------------------------------
`kline_long.py` 的判据是 `@日均15单 = 全期 top-K`：要按 |score| 取全期前 K 名，
意味着必须知道**整个 OOS 期的分数分布**才能定阈值——这是前视，实盘不可执行。
`h3_signal.py` 已经把这个教训写死了（阈值只在校准段定），但那条纪律只落在 h=3 上。

本文件把同一纪律搬到 h=10：
1. 逐月 walk-forward，滚动窗 = WIN_DAYS 天（默认 1460）。
2. 窗内三段切：train / es / thr。**模型只在 train 上训练，早停用 es，
   阈值分位只在 thr 段算**（thr 是模型没训过的、且全部早于目标月 → 严格因果）。
3. 阈值 = thr 段 |p-0.5| 的 (1 - tpd/1440) 分位，之后整月固定不变。
4. 判据：逐月胜率分布 + 月度 min + 滚动 100 单最差 + 全期 EV。

为什么要收单量才可能到 60%
--------------------------
2026-11-15 实测（`h3bar`，单一切分但同口径）：h=10 K 线腿 @14.7 单/天 60.02%、
@9.9 单/天 60.56%、@4.7 单/天 61.71%。订单簿腿在 h=10 的曲线是**平**的（无弹性），
K 线腿有弹性——所以「挑高置信度」对 K 线腿是真有效的手段，对订单簿腿不是。
主操作点预登记为 PRIMARY_TPD=8（8 单/天，10 分钟持仓 ≈ 最多 3 个并发仓，可执行；
滚动 100 单判据约 12 天可算）。TPD_LIST 其余档只作敏感性，**不据此事后挑最优点**。

特征
----
`kline_long.build()` 的长周期特征（多尺度动量/均线结构/RV 期限结构/量价/位置/ER/季节性/
期现基差）+ 本文件追加的**合约 vs 现货主动成交失衡差**（外部 notebook 唯一没错的那条：
flow_divergence）。全部只用 past，无前视。

env
---
  SYMBOL / HORIZON=10 / SEEDS=42,... / ROUNDS / WIN_DAYS=1460 / ES_FRAC / THR_FRAC
  START_MONTH / END_MONTH / TPD_LIST / PRIMARY_TPD=8 / KLINE_FROM / PAYOUTS
  STRIDE=1（=HORIZON 抽样加速，正式数字保持 1）/ PLACEBO=0（=1 打乱标签查泄漏）
  PLACEBO_SEEDS=2 / BARS_PER_DAY=1440（阈值分位分母，与 1m bar 的机会数对齐）
  LGB_REG=base|reg1|**reg2**|shallow（默认 base，见下方"超参档位"）

换 horizon 用法（2026-11-15 起，h=30 摸底）
------------------------------------------
  HORIZON=30 WIN_DAYS=2400 STRIDE=30 TPD_LIST=1,2,4,8 PRIMARY_TPD=4
  PLACEBO=1 PAYOUTS=0.853,0.85

  HORIZON=15 STRIDE=15（其余与 h=10 基线完全一致，tpd 相同 → 单/天相同，可与
  63.32%@19.9 单/天 直接对撞）PAYOUTS=0.838,0.853 PLACEBO=1
"""
from __future__ import annotations

import datetime as dt
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from morph import h3_gate_eval as ge  # noqa: E402
from morph import kline_long as kl  # noqa: E402
from morph.two_legs import add_target, fit, matrix  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass


def _git_rev() -> str:
    """打印当前代码版本，防止「本地改了没 push → kernel 跑旧代码」再次静默发生。"""
    try:
        root = Path(__file__).resolve().parent.parent
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=root, stderr=subprocess.DEVNULL,
                                       text=True).strip()
    except Exception:
        return "unknown"


CODE_REV = _git_rev()

HORIZON = int(os.environ.get("HORIZON", "10"))
WEEKDAY_ONLY = True  # 本文件固定只做 UTC 周一~周五；不动 morph/h10_signal.py
# bar 尺度（分钟）：1 = 原始 1m；>1 = 真 K 线重采样，并在新 bar 上重算全部特征
BAR = int(os.environ.get("BAR", "1"))
SEEDS = [int(s) for s in os.environ.get("SEEDS", "42,43,44,45,46").split(",") if s.strip()]
ROUNDS = int(os.environ.get("ROUNDS", "2000"))
WIN_DAYS = int(os.environ.get("WIN_DAYS", "1460"))
ES_FRAC = float(os.environ.get("ES_FRAC", "0.05"))
THR_FRAC = float(os.environ.get("THR_FRAC", "0.10"))
START_MONTH = os.environ.get("START_MONTH", "2026-01")
END_MONTH = os.environ.get("END_MONTH", "2026-10")
TPD_LIST = [float(x) for x in os.environ.get("TPD_LIST", "2,5,8,15,30").split(",") if x.strip()]
PRIMARY_TPD = float(os.environ.get("PRIMARY_TPD", "8"))
PAYOUTS = [float(x) for x in os.environ.get("PAYOUTS", "0.80,0.828").split(",") if x.strip()]
# 阈值分位分母：test 是 1m 行，所以 1440 = 每天 1440 个候选位置。
# 保持 1440 才与 h=10 的 63.32% 同口径（实盘合并同一段连续触发是另一回事，见 §36 之后的口径备忘）。
BARS_PER_DAY = int(os.environ.get("BARS_PER_DAY", str(1440 // BAR)))
# 1h 等粗粒度下，4 年窗口只有 ~35k 行，默认 100k 门槛会跳过所有月；
# 用 MIN_TRAIN 覆盖（默认 100_000 保持现役口径不变）
MIN_TRAIN = int(os.environ.get("MIN_TRAIN", "100000"))
MIN_TEST = int(os.environ.get("MIN_TEST", "1000"))
# 标签重叠 HORIZON 根：STRIDE=HORIZON 只在「整段聚合胜率」上近似等价
# （h10 实测 STRIDE=10 62.35% vs 全量 63.32%，差在 SE≈3.2pp 内），但早停 best_iter
# 和月稳定性并不等价（月 min 55.74%→50.99%，单月单 seed 可差 6.8pp）。
# 摸底/月度重训可用，正式数字保持 STRIDE=1（全量）。
STRIDE = int(os.environ.get("STRIDE", "1"))
# 泄漏检查：训练+早停标签一起打乱，test AUC 必须回落到 0.50（§35.3 定的换 horizon 准入检查）
PLACEBO = os.environ.get("PLACEBO", "0") == "1"
PLACEBO_SEEDS = int(os.environ.get("PLACEBO_SEEDS", "2"))
# 结构位特征（前高前低/整数关口/日内开盘/影线/多周期共振），默认关闭保持基线不变
WITH_STRUCT = os.environ.get("STRUCT", "0") == "1"
# 流动性与价格冲击（taker 占比/冲击/深度/单笔规模），默认关闭
WITH_LIQ = os.environ.get("LIQ", "0") == "1"
# 时段筛选：白名单只用「训练窗 thr 段」的胜率定（严格因果），test 只在该时段下单
HOURFILT = os.environ.get("HOURFILT", "0") == "1"
HOUR_BLOCK = int(os.environ.get("HOUR_BLOCK", "3"))     # 每块几小时（24 除以它要是整数）
HOUR_MIN_WIN = float(os.environ.get("HOUR_MIN_WIN", "0.55"))  # 块胜率门槛
# 2026-11-18 审计修复：原逻辑用 30 单门槛 + 裸胜率 55% 在 8 个块里挑，
# 多重比较下必然选出噪声（样本外 60.13%→40.82%）。改为单侧二项检验 +
# Bonferroni 校正，且每块至少 HOUR_MIN_N 单。
HOUR_MIN_N = int(os.environ.get("HOUR_MIN_N", "100"))
HOUR_ALPHA = float(os.environ.get("HOUR_ALPHA", "0.05"))
# ---------------------------------------------------------------- 目标函数（2026-11-18，方向#1）
# binary = 现役全样本 logloss；rank = LambdaRank 按 UTC 日分组优化尾部排序。
# rank 输出是相对分数不是概率：用 thr 段中位数中心化后，走与 binary 相同的
# |s-中心| 分位阈值 + sign 方向口径，保证两种目标函数在同一评估纪律下对撞。
OBJECTIVE = os.environ.get("OBJECTIVE", "binary").strip().lower()
if OBJECTIVE not in ("binary", "rank"):
    raise SystemExit(f"OBJECTIVE={OBJECTIVE} 只能是 binary|rank")
RANK_K = int(os.environ.get("RANK_K", "30"))  # lambdarank 截断：每天 top30 名参与 NDCG
# 方向 #4：非对称阈值。对称口径用 |p-.5| 一个分位；ASYM=1 时多/空各按 thr 段
# 自己那侧的 |p-.5| 分位定阈值，使两侧各触发 tpd/2，适配分数分布的位置偏移。
ASYM = os.environ.get("ASYM", "0") == "1"
# 方向 #6：thr 段直接用日收益目标选阈值（候选 = 各 tpd 分位 + 网格），全程因果。
# 判据：thr 段单量/天 × EV(用 thr 段胜率估计)，并要求 thr 段成交数 ≥ THR_OPT_MIN_N。
THR_OPT = os.environ.get("THR_OPT", "0") == "1"
THR_OPT_MIN_N = int(os.environ.get("THR_OPT_MIN_N", "200"))
# ---------------------------------------------------------------- 超参档位（2026-11-16）
# 本仓库长期硬编码 num_leaves=63 / min_child_samples=100，在 H5/H10/H15 的
# 「1.05M 行 × 49 列、AUC≈0.53」极低信噪比下明显过拟合。reg2 强正则化在
# mem_replay 的实盘口径下 9/9 同向改善（H10 AUC 0.5324、2026 胜率 +3.7~5.9pp、
# 2025 全年 EV 由负转正），因此接线到本生产入口；单一真源在此，mem_replay 也从这里导入。
# 默认 base → 现役数字（tpd=8 63.32% @7.4 单/天）保持不变。
# 旧日志里的 @19.9 单/天是「仅有单天数」作分母的旧 bug，2026-11-15 已修为全部测试天数。
LGB_PRESETS = {
    "base": {},
    "reg1": dict(num_leaves=31, min_child_samples=1000, feature_fraction=.7, lambda_l2=10),
    "reg2": dict(num_leaves=15, min_child_samples=3000, feature_fraction=.5, lambda_l2=100),
    "shallow": dict(learning_rate=.03, num_leaves=15, min_child_samples=500,
                    feature_fraction=.6, lambda_l2=20, max_depth=4),
}
LGB_REG = os.environ.get("LGB_REG", "base").strip().lower()
if LGB_REG not in LGB_PRESETS:
    raise SystemExit(f"LGB_REG={LGB_REG} 不在 {sorted(LGB_PRESETS)}（拼错会静默跑成 base）")
LGB_PARAMS = LGB_PRESETS[LGB_REG]
T = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T:7.1f}s] {msg}", flush=True)


def add_flow(df: pl.DataFrame, feats: list[str]) -> tuple[pl.DataFrame, list[str]]:
    """合约 vs 现货主动成交失衡差（外部 notebook 的 flow_divergence，因果实现）。"""
    c = df.columns
    need = ("um_buy_vol", "um_sell_vol", "spot_buy_vol", "spot_sell_vol")
    if not all(x in c for x in need):
        log(f"  跳过 flow 特征（缺列：{[x for x in need if x not in c]}）")
        return df, feats
    um = ((pl.col("um_buy_vol") - pl.col("um_sell_vol"))
          / (pl.col("um_buy_vol") + pl.col("um_sell_vol") + 1e-9))
    sp = ((pl.col("spot_buy_vol") - pl.col("spot_sell_vol"))
          / (pl.col("spot_buy_vol") + pl.col("spot_sell_vol") + 1e-9))
    e = [um.alias("lg_flow_um"), sp.alias("lg_flow_spot"),
         (um - sp).alias("lg_flow_div")]
    for n in (15, 60):
        e += [um.rolling_mean(n, min_samples=max(2, n // 2)).alias(f"lg_flow_um{n}"),
              (um - sp).rolling_mean(n, min_samples=max(2, n // 2)).alias(f"lg_flow_div{n}"),
              (um - sp).rolling_std(n, min_samples=max(2, n // 2)).alias(f"lg_flow_divstd{n}")]
    out = df.with_columns(e)
    new = [x for x in out.columns if x.startswith("lg_flow")]
    return out, feats + new


def add_struct(df: pl.DataFrame, feats: list[str]) -> tuple[pl.DataFrame, list[str]]:
    """结构位特征（直播类 30 分钟策略的核心信息，本仓库 49 列里完全没有）。

    现有 `lg_pos{n}` 是「区间内相对位置」，没有「距某个具体价位多少 bp」的概念；
    而 30 分钟级别的价格行为高度依赖前高前低 / 整数关口 / 日内开盘价。
    全部只用 past：rolling 之后 shift(1)，日开盘价是当日已知的第一根。
    """
    c = pl.col("close").cast(pl.Float64)
    e: list[pl.Expr] = []
    for n, tag in ((240, "4h"), (1440, "1d"), (10080, "7d")):
        hi = c.rolling_max(n, min_samples=max(2, n // 2)).shift(1)
        lo = c.rolling_min(n, min_samples=max(2, n // 2)).shift(1)
        e += [((hi - c) / c * 1e4).alias(f"st_dnhi_{tag}"),
              ((c - lo) / c * 1e4).alias(f"st_uplo_{tag}")]
    if {"open", "high", "low"} <= set(df.columns):
        o = pl.col("open").cast(pl.Float64)
        h = pl.col("high").cast(pl.Float64)
        lo_ = pl.col("low").cast(pl.Float64)
        body_hi = pl.max_horizontal(o, c)
        body_lo = pl.min_horizontal(o, c)
        e += [((h - body_hi) / c * 1e4).alias("st_upshadow"),
              ((body_lo - lo_) / c * 1e4).alias("st_dnshadow")]
    # 整数关口距离（BTC 按 1000、ETH 按 100）：距最近关口的带符号 bp
    step = 1000.0 if kl.SYMBOL.upper().startswith("BTC") else 100.0
    e.append(((((c / step) % 1.0) - 0.5) * step / c * 1e4).alias("st_round"))
    # 日内（UTC 0 点）开盘价距离
    day_open = c.first().over(pl.col("close_time") // 86_400_000)
    e.append(((c / day_open - 1) * 1e4).alias("st_dayopen"))
    # 多周期共振：5/15/60/240/1440 分钟动量同向的个数（-5..5）
    if "lg_ret1440" in df.columns:
        res = None
        for k in (5, 15, 60, 240, 1440):
            s = pl.col(f"lg_ret{k}").sign()
            res = s if res is None else res + s
        e.append(res.alias("st_reson5"))
    out = df.with_columns(e)
    new = [x for x in out.columns if x.startswith("st_")]
    return out, feats + new


def to_bar(df: pl.DataFrame, bar: int) -> pl.DataFrame:
    """1m → bar 分钟**真 K 线**（OHLCV 按桶聚合，不是「取每桶最后一个 1m 特征快照」）。

    区别很关键：`h3_bartf.py`（E2）是先算 1m 特征再 `last()` 取快照，等于只换了标签的
    采样粒度、特征还停在 1 分钟信噪比上，所以它只能证明「采样无关」。这里是真的重采样。
    """
    if bar == 1:
        return df
    agg = [pl.col("open").first(), pl.col("high").max(), pl.col("low").min(),
           pl.col("close").last(), pl.col("um_buy_vol").sum(),
           pl.col("um_sell_vol").sum(), pl.col("um_count").sum()]
    if "spot_close" in df.columns:
        agg += [pl.col("spot_close").last(), pl.col("spot_buy_vol").sum(),
                pl.col("spot_sell_vol").sum()]
    out = (df.with_columns(pl.from_epoch("close_time", time_unit="ms").alias("ts"))
             .sort("ts")
             .group_by_dynamic("ts", every=f"{bar}m")
             .agg(agg + [pl.col("close_time").last()])
             .drop("ts")
             .sort("close_time"))
    log(f"  1m {df.height:,} 行 → {bar}m {out.height:,} 行（真 K 线聚合）")
    return out


def build_bar(df: pl.DataFrame, bar: int) -> tuple[pl.DataFrame, list[str]]:
    """在 bar 尺度上重算全部特征（**分钟语义**：窗口仍按分钟给，内部换算成根数）。

    窗口以「一个 horizon」为中心铺开：`HORIZON=60` 时 `b_ret60` 正好是 1 个 horizon，
    `b_ret1440` 是 24 个。比直接复用 `kl.build()`（窗口按根解释，5m 上会放大 5 倍）可控得多。
    """
    c = pl.col("close").cast(pl.Float64)
    r1 = (c / c.shift(1)).log() * 1e4

    def k_of(m: int) -> int:
        return max(1, m // bar)

    def ms(k: int, want: int = 2) -> int:      # min_samples 不得超过窗口长度（粗 bar 下 k=1）
        return min(max(1, want, k // 2), k)

    e: list[pl.Expr] = []
    for m in (5, 10, 15, 30, 60, 120, 240, 480, 1440):
        e.append(((c / c.shift(k_of(m))).log() * 1e4).alias(f"b_ret{m}"))
    for m in (30, 60, 240, 1440):
        k = k_of(m)
        ma = c.rolling_mean(k, min_samples=ms(k))
        e += [((c / ma - 1) * 1e4).alias(f"b_madev{m}"),
              ((ma / ma.shift(k) - 1) * 1e4).alias(f"b_maslope{m}"),
              r1.rolling_std(k, min_samples=ms(k, 3)).alias(f"b_rv{m}")]
    for m in (240, 1440):
        k = k_of(m)
        lo = c.rolling_min(k, min_samples=ms(k))
        hi = c.rolling_max(k, min_samples=ms(k))
        e.append(((c - lo) / (hi - lo + 1e-9)).alias(f"b_pos{m}"))
    abs_r1 = r1.abs()
    for m in (60, 240):
        k = k_of(m)
        e.append((r1.rolling_sum(k).abs()
                  / (abs_r1.rolling_sum(k) + 1e-9)).alias(f"b_er{m}"))
    vol = pl.col("um_buy_vol") + pl.col("um_sell_vol")
    km = k_of(1440)
    vm = vol.rolling_mean(km, min_samples=ms(km))
    vs = vol.rolling_std(km, min_samples=ms(km))
    e += [((vol - vm) / (vs + 1e-9)).alias("b_volz"),
          (vol / (vm + 1e-9)).alias("b_volratio")]
    tk = pl.col("um_buy_vol") / (vol + 1e-9)
    for m in (15, 60, 240):
        k = k_of(m)
        e.append(tk.rolling_mean(k, min_samples=ms(k)).alias(f"b_taker{m}"))
    if "spot_close" in df.columns:
        basis = (c / pl.col("spot_close").cast(pl.Float64) - 1) * 1e4
        kb = k_of(60)
        e += [basis.alias("b_basis"),
              (basis - basis.shift(kb)).alias("b_basis_chg"),
              basis.rolling_mean(kb, min_samples=ms(kb)).alias("b_basis_ma")]
    ct = pl.col("close_time")
    hour = ((ct // 3_600_000) % 24).cast(pl.Float64)
    dow = ((ct // 86_400_000) % 7).cast(pl.Float64)
    e += [(hour * (2 * np.pi / 24)).sin().alias("b_hour_sin"),
          (hour * (2 * np.pi / 24)).cos().alias("b_hour_cos"),
          (dow * (2 * np.pi / 7)).sin().alias("b_dow_sin"),
          (dow * (2 * np.pi / 7)).cos().alias("b_dow_cos")]
    out = df.with_columns(e)
    e2: list[pl.Expr] = []
    for m in (60, 240, 1440):
        e2.append((pl.col(f"b_ret{m}") / (pl.col(f"b_rv{m}") + 1e-9)).alias(f"b_zret{m}"))
    for a_, b_ in ((30, 60), (60, 240), (240, 1440)):
        e2.append((pl.col(f"b_rv{a_}") / (pl.col(f"b_rv{b_}") + 1e-9)).alias(f"b_rvr{a_}_{b_}"))
    out = out.with_columns(e2)
    return out, [x for x in out.columns if x.startswith("b_")]


def add_liq(df: pl.DataFrame, feats: list[str]) -> tuple[pl.DataFrame, list[str]]:
    """流动性与价格冲击特征（**不做成交量绝对值/量比**）。

    用户的判断（2026-11-15 采纳）：成交量单独是垃圾——它只反映「现在是不是活跃时段」
    （日内 U 型、牛熊、周内变化），与方向无关；量比同理。真正的信息在
    「**推动 1bp 需要多少量**」：挂单薄的时候，很小的成交量也能推动大波动。

    所以这里只用比值/强度：
      lq_taker{n}   主动买占比（谁在吃单，有方向性）
      lq_takerd{n}  快均 taker − 240 均 taker（主动性**突变**比水平更有意义）
      lq_impact{n}  log(Σ|r1| / Σ量) —— 单位成交量的价格冲击，低量推大波动 → 值大
      lq_depth{n}   Σ量 / 价格区间 —— 每 bp 波动消耗多少量，薄盘 → 值小
      lq_avgtrd     量 / 笔数 —— 平均单笔规模（大单 vs 散单的代理）
    """
    c = pl.col("close").cast(pl.Float64)
    r1 = (c / c.shift(1)).log() * 1e4
    b = pl.col("um_buy_vol").cast(pl.Float64)
    s = pl.col("um_sell_vol").cast(pl.Float64)
    v = b + s
    cnt = pl.col("um_count").cast(pl.Float64)
    tk = b / (v + 1e-9)
    e: list[pl.Expr] = []
    for n in (15, 60, 240):
        ms = max(2, n // 2)
        e.append(tk.rolling_mean(n, min_samples=ms).alias(f"lq_taker{n}"))
        e.append((tk.rolling_mean(n, min_samples=ms)
                  - tk.rolling_mean(240, min_samples=120)).alias(f"lq_takerd{n}"))
    for n in (15, 60, 240):
        ms = max(2, n // 2)
        imp = (r1.abs().rolling_sum(n, min_samples=ms)
               / (v.rolling_sum(n, min_samples=ms) / 1e3 + 1e-9))
        e.append(imp.log1p().alias(f"lq_impact{n}"))
    for n in (60, 240):
        ms = max(2, n // 2)
        rng = ((c.rolling_max(n, min_samples=ms) - c.rolling_min(n, min_samples=ms))
               / c * 1e4)
        e.append((v.rolling_sum(n, min_samples=ms) / (rng + 1e-9) / 1e3).alias(f"lq_depth{n}"))
    e.append((v / (cnt + 1e-9)).alias("lq_avgtrd"))
    out = df.with_columns(e)
    return out, feats + [x for x in out.columns if x.startswith("lq_")]


def roll_worst(w: np.ndarray, n: int = 100) -> tuple[float, float]:
    """滚动 n 单最差胜率 / 破 50% 窗口占比（单序评估，最贴近实盘体感）。"""
    if len(w) < n:
        return float("nan"), float("nan")
    cs = np.concatenate([[0], np.cumsum(w)])
    r = (cs[n:] - cs[:-n]) / n
    return float(r.min()), float((r < .5).mean())


def main() -> None:
    raw = kl.load_raw()
    if BAR > 1:
        raw = to_bar(raw, BAR)
        df, feats = build_bar(raw, BAR)
        log(f"bar={BAR}m 原生特征 {len(feats)} 列 | HORIZON={HORIZON} 分钟 "
            f"= {HORIZON / BAR:g} 步 | 每天 {BARS_PER_DAY} 个候选位置")
    else:
        df, feats = kl.build(raw)
    df, feats = add_flow(df, feats)
    if WITH_STRUCT:
        df, feats = add_struct(df, feats)
    if WITH_LIQ:
        df, feats = add_liq(df, feats)
    df = add_target(df, HORIZON).filter(
        pl.col("target").is_not_null() & pl.col("close").is_not_null()).sort("close_time")
    if WEEKDAY_ONLY:
        # 特征与标签已在完整连续序列上算好；这里只筛行。
        df = (df.with_columns(pl.from_epoch(pl.col("close_time"), time_unit="ms")
                              .dt.weekday().alias("wd"))
                .filter(pl.col("wd") <= 5).drop("wd"))
        log("WEEKDAY_ONLY：仅 UTC 周一~周五，h10_signal.py 基线未修改")
    df = df.with_columns(
        pl.col("close_time").cast(pl.Datetime("ms")).dt.strftime("%Y-%m").alias("ym"))
    log(f"特征 {len(feats)} 列 | 可用 {df.height:,} 行 | {df['ym'].min()} ~ {df['ym'].max()}")

    months = [m for m in sorted(df["ym"].unique().to_list()) if START_MONTH <= m <= END_MONTH]
    log(f"walk-forward {len(months)} 窗口（{months[0]}~{months[-1]}）| H={HORIZON} | "
        f"滚动窗 {WIN_DAYS} 天 | es={ES_FRAC:.0%} thr={THR_FRAC:.0%} | seeds={len(SEEDS)}")
    log(f"CODE_REV={CODE_REV} | MODEL={os.environ.get('MODEL','lgb')}"
        + (f" XGB_PRESET={os.environ.get('XGB_PRESET','default')}"
           f" XGB_PARAMS={os.environ.get('XGB_PARAMS','')[:120]}"
           if os.environ.get('MODEL') == 'xgb' else "")
        + f" | OBJECTIVE={OBJECTIVE}"
        + (f" RANK_K={RANK_K}" if OBJECTIVE == "rank" else "")
        + f" | ASYM={ASYM} | THR_OPT={THR_OPT} | LGB_REG={LGB_REG} "
        + f"{LGB_PARAMS or '（历史默认 63/100）'}")
    log(f"因果阈值：分位只在「目标月之前的 thr 段」算，test 月只用固定阈值 → 无前视")

    per_tpd: dict[float, dict[str, list]] = {
        t: dict(s=[], y=[], d=[], h=[], mon=[]) for t in TPD_LIST}
    hf_s: list[np.ndarray] = []
    hf_y: list[np.ndarray] = []
    rows = []
    for m in months:
        m_start = int(dt.datetime.strptime(m + "-01", "%Y-%m-%d")
                      .replace(tzinfo=dt.timezone.utc).timestamp() * 1000)
        lo = m_start - WIN_DAYS * 86_400_000
        tr_all = df.filter((pl.col("close_time") < m_start) & (pl.col("close_time") >= lo))
        te = df.filter(pl.col("ym") == m)
        if tr_all.height < MIN_TRAIN or te.height < MIN_TEST:
            log(f"[{m}] 样本不足（{tr_all.height:,}/{te.height:,}），跳过")
            continue
        h_all = tr_all.height
        c1 = int(h_all * (1 - ES_FRAC - THR_FRAC))
        c2 = int(h_all * (1 - THR_FRAC))
        tr, es, th = tr_all[:c1], tr_all[c1:c2], tr_all[c2:]
        if STRIDE > 1:
            # **只抽 train**：标签重叠 HORIZON 根，训练抽 stride 可去掉冗余、加速。
            # es 和 thr 必须全量（2026-11-18 审计修复）：
            #   - es 被抽样时从 105k 行降到 10k，early_stopping(50) 的 AUC 噪声使
            #     best_iter 从 235 掉到 1（只训 1 棵树），同月同种子胜率 59.28%→52.46%。
            #   - thr 被抽样时「每 1m 行 1-tpd/1440 分位」的粒度变了，套到全量 test
            #     会把 tpd 放大约 STRIDE 倍（h=30 实测 tpd=1→5.1 单/天）。2026-11-15 修。
            tr = tr.gather_every(STRIDE)
        Xtr, ytr = matrix(tr, feats), tr["target"].to_numpy().astype(int)
        Xes, yes = matrix(es, feats), es["target"].to_numpy().astype(int)
        Xth = matrix(th, feats)
        Xte = matrix(te, feats)
        pth = np.zeros(th.height)
        pte = np.zeros(te.height)
        its = []
        params_eff = dict(LGB_PARAMS)
        if OBJECTIVE == "rank":
            params_eff.update(objective="lambdarank", metric="ndcg",
                              lambdarank_truncation_level=RANK_K)
            gtr = np.bincount(np.unique(ge.day_id(tr["close_time"].to_numpy()),
                                        return_inverse=True)[1])
            gva = np.bincount(np.unique(ge.day_id(es["close_time"].to_numpy()),
                                        return_inverse=True)[1])
        else:
            gtr = gva = None
        for seed in SEEDS:
            mdl = fit(Xtr, ytr, feats, Xes, yes, seed=seed, params_override=params_eff,
                      gtr=gtr, gva=gva)
            its.append(int(mdl.best_iteration))
            pth += mdl.predict(Xth, num_iteration=mdl.best_iteration) / len(SEEDS)
            pte += mdl.predict(Xte, num_iteration=mdl.best_iteration) / len(SEEDS)
        if OBJECTIVE == "rank":
            # rank 输出是相对分数；平移到「伪概率」空间，使 thr 段预测涨占比 = 实际涨占比。
            # 之后所有 |p-.5| 阈值 / sign 方向口径与 binary 完全一样，公平对撞。
            yth_ = th["target"].to_numpy().astype(int)
            center = float(np.quantile(pth, 1 - yth_.mean()))
            pth = pth - center + 0.5
            pte = pte - center + 0.5
            log(f"  [rank] center={center:.4f} thr涨占比={yth_.mean():.3f} 平移后伪概率")
        y = te["target"].to_numpy().astype(np.int8)
        ct_ms = te["close_time"].to_numpy()
        days = ge.day_id(ct_ms)
        hours = ((ct_ms // 3_600_000) % 24).astype(np.int8)
        nd = len(np.unique(days))
        pl_auc, pl_win = float("nan"), float("nan")
        if PLACEBO:
            # 泄漏检查：训练与早停标签一起打乱，test AUC 必须回落到 0.50
            rng = np.random.default_rng(20261115)
            ysh, yesh = ytr.copy(), yes.copy()
            rng.shuffle(ysh)
            rng.shuffle(yesh)
            npl = min(len(SEEDS), PLACEBO_SEEDS)
            pth_pl = np.zeros(th.height)
            pte_pl = np.zeros(te.height)
            for seed in SEEDS[:npl]:
                mdl = fit(Xtr, ysh, feats, Xes, yesh, seed=seed, params_override=params_eff,
                          gtr=gtr, gva=gva)
                pth_pl += mdl.predict(Xth, num_iteration=mdl.best_iteration) / npl
                pte_pl += mdl.predict(Xte, num_iteration=mdl.best_iteration) / npl
            if OBJECTIVE == "rank":
                c_pl = float(np.quantile(pth_pl, 1 - ysh.mean()))
                pth_pl = pth_pl - c_pl + 0.5
                pte_pl = pte_pl - c_pl + 0.5
            pl_auc = float(roc_auc_score(y, pte_pl))
            if ASYM:
                q_pl = 1 - (PRIMARY_TPD / 2) / BARS_PER_DAY
                tu_pl = float(np.quantile(np.abs(pth_pl[pth_pl > .5] - .5), q_pl)) if (pth_pl > .5).sum() >= 30 else np.inf
                td_pl = float(np.quantile(np.abs(pth_pl[pth_pl < .5] - .5), q_pl)) if (pth_pl < .5).sum() >= 30 else np.inf
                s_pl = pte_pl - .5
                hit_pl = ((s_pl > 0) & (s_pl >= tu_pl)) | ((s_pl < 0) & (-s_pl >= td_pl))
            else:
                thr_pl = float(np.quantile(np.abs(pth_pl - .5), 1 - PRIMARY_TPD / BARS_PER_DAY))
                hit_pl = np.abs(pte_pl - .5) >= thr_pl
            if int(hit_pl.sum()) >= 30:
                pl_win = float(((pte_pl[hit_pl] - .5 > 0).astype(int) == y[hit_pl]).mean())
            del pth_pl, pte_pl, hit_pl, ysh, yesh
        del Xtr, Xes, Xth, Xte
        s = pte - .5
        auc = float(roc_auc_score(y, pte))
        line = (f"[{m}] train={tr.height:,} es={es.height:,} thr={th.height:,} test={te.height:,}"
                f"({nd}天) iter={int(np.mean(its))} AUC={auc:.4f}")
        if PLACEBO:
            line += f" | placebo AUC={pl_auc:.4f} 胜率={pl_win * 100:.2f}%"
        # 【严格因果的时段筛选】白名单只用 thr 段（全部早于目标月）的块胜率定，
        # 绝不看 test。这是「直播类 15 单/天 62%」最可能的来源，必须单独验证。
        ok_blk: list[int] = []
        if HOURFILT:
            yth = th["target"].to_numpy().astype(np.int8)
            hth = ((th["close_time"].to_numpy() // 3_600_000) % 24).astype(np.int8)
            bth = (hth // HOUR_BLOCK).astype(np.int8)
            thr_p = float(np.quantile(np.abs(pth - .5), 1 - PRIMARY_TPD / BARS_PER_DAY))
            m_p = np.abs(pth - .5) >= thr_p
            parts = []
            n_blocks = 24 // HOUR_BLOCK
            for bi in range(n_blocks):
                mk = m_p & (bth == bi)
                if mk.sum() >= HOUR_MIN_N:
                    wins_b = int(((pth[mk] - .5 > 0).astype(int) == yth[mk]).sum())
                    w_b = wins_b / mk.sum()
                    p_adj = min(1.0, binomtest(wins_b, mk.sum(), 0.5,
                                              alternative="greater").pvalue * n_blocks)
                    parts.append(f"{bi * HOUR_BLOCK:02d}-{(bi + 1) * HOUR_BLOCK - 1:02d}h"
                                 f":{w_b * 100:.0f}%(n={int(mk.sum())},p*={p_adj:.3f})")
                    if w_b >= HOUR_MIN_WIN and p_adj < HOUR_ALPHA:
                        ok_blk.append(bi)
            line += (f"\n    波段(训练窗, n>={HOUR_MIN_N}, 二项检验+Bonferroni α={HOUR_ALPHA}): "
                     + " ".join(parts) + f" → 白名单 {ok_blk}")
        for tpd in TPD_LIST:
            if ASYM:
                # 多空两侧分别在 thr 段定阈值，各自目标 tpd/2 单/天（严格因果）
                q = 1 - (tpd / 2) / BARS_PER_DAY
                thr_up = float(np.quantile(np.abs(pth[pth > .5] - .5), q)) if (pth > .5).sum() >= 30 else np.inf
                thr_dn = float(np.quantile(np.abs(pth[pth < .5] - .5), q)) if (pth < .5).sum() >= 30 else np.inf
                hit = ((s > 0) & (s >= thr_up)) | ((s < 0) & (-s >= thr_dn))
                thr = float("nan")
            else:
                thr = float(np.quantile(np.abs(pth - .5), 1 - tpd / BARS_PER_DAY))
                if THR_OPT:
                    # 围绕名义 tpd 扫 4 档候选，在 thr 段用日收益=(单/天)×EV 选最优。
                    # 全程只用 thr 段（早于 test），严格因果；test 月固定用选中的阈值。
                    st = np.abs(pth - .5)
                    yth_opt = th["target"].to_numpy().astype(np.int8)
                    ndays_th = len(np.unique(ge.day_id(th["close_time"].to_numpy())))
                    best_dr, best_thr = -np.inf, thr
                    for mult in (0.25, 0.5, 1.0, 2.0, 4.0):
                        tq = 1 - min(tpd * mult, BARS_PER_DAY * 0.5) / BARS_PER_DAY
                        cand = float(np.quantile(st, tq))
                        m_th = st >= cand
                        n_th = int(m_th.sum())
                        if n_th < THR_OPT_MIN_N:
                            continue
                        win_th = float(((pth[m_th] - .5 > 0).astype(int) == yth_opt[m_th]).mean())
                        ev_th = PAYOUTS[0] * win_th - (1 - win_th)
                        dr = (n_th / max(ndays_th, 1)) * ev_th
                        if dr > best_dr:
                            best_dr, best_thr = dr, cand
                    thr = best_thr
                hit = np.abs(s) >= thr
            nb = int(hit.sum())
            if nb < 30:
                line += f" | tpd{tpd:g}:单量太少"
                continue
            w = ((s[hit] > 0).astype(int) == y[hit]).astype(np.int8)
            win = float(w.mean())
            per_tpd[tpd]["s"].append(s[hit])
            per_tpd[tpd]["y"].append(y[hit])
            per_tpd[tpd]["d"].append(days[hit])
            per_tpd[tpd]["h"].append(hours[hit])
            per_tpd[tpd]["mon"].append(win)
            if tpd == PRIMARY_TPD:
                line += f" | 阈值={thr:.4f} {nb / nd:.1f}单/天 胜率={win * 100:.2f}%"
                if ok_blk:
                    b_te = (hours // HOUR_BLOCK).astype(np.int8)
                    hit_f = hit & np.isin(b_te, ok_blk)
                    if int(hit_f.sum()) >= 30:
                        w_f = float(((s[hit_f] > 0).astype(int) == y[hit_f]).mean())
                        hf_s.append(s[hit_f])
                        hf_y.append(y[hit_f])
                        line += (f" | 时段筛选 {int(hit_f.sum()) / nd:.1f}单/天"
                                 f" 胜率={w_f * 100:.2f}%")
        rows.append(dict(ym=m, test=te.height, nd=nd, auc=auc, iter=int(np.mean(its)),
                         pl_auc=pl_auc))
        log(line)

    if not rows:
        raise SystemExit("无窗口完成")

    # 赔率档随 PAYOUTS 变化（h=30 是 0.853/0.85，不是 h=10 的 0.80/0.828）
    ev_hdr = "".join(f"{'EV@' + format(p, 'g'):>9}" if i == 0
                     else f"{'EV@' + format(p, 'g'):>10}" for i, p in enumerate(PAYOUTS))
    print(f"\n{'名义tpd':>8}{'实际单/天':>10}{'n':>7}{'胜率':>9}{'SE*':>7}{ev_hdr}"
          f"{'月min':>8}{'月std':>7}{'破50%月':>8}{'滚动100min':>11}{'破50%窗':>9}")
    for tpd in TPD_LIST:
        p = per_tpd[tpd]
        if not p["y"]:
            continue
        yv = np.concatenate(p["y"])
        sv = np.concatenate(p["s"])
        w = ((sv > 0).astype(int) == yv).astype(np.int8)
        win = float(w.mean())
        # 分母必须是**全部测试天数**，不是"有单的天数"：
        # 低频档（h=30 tpd=1，266 单只落在 43 天）用后者会把 1.8 单/天算成 6.2。
        nd = sum(r["nd"] for r in rows)
        mon = np.array(p["mon"])
        se = float(np.sqrt(win * (1 - win) / max(len(w) / HORIZON, 2)))
        rmin, rbad = roll_worst(w)
        evs = "".join(f"{pay * win - (1 - win):>+9.3f}" if i == 0
                      else f"{pay * win - (1 - win):>+10.3f}" for i, pay in enumerate(PAYOUTS))
        star = " ← 主操作点" if tpd == PRIMARY_TPD else ""
        print(f"{tpd:>8.1f}{len(w) / nd:>10.1f}{len(w):>7d}{win * 100:>8.2f}%{se * 100:>6.2f}%"
              f"{evs}{mon.min() * 100:>7.2f}%{mon.std() * 100:>6.2f}pp"
              f"{int((mon < .5).sum()):>5}/{len(mon)}{rmin * 100:>10.1f}%{rbad * 100:>8.1f}%{star}")

    p = per_tpd[PRIMARY_TPD]
    if p["mon"]:
        print(f"\n[主操作点 tpd={PRIMARY_TPD:g}] 逐月胜率：")
        print("  " + "  ".join(f"{r['ym'][2:]}:{m * 100:.1f}%"
                              for r, m in zip(rows, p["mon"])))
    if p["h"]:
        # 时段诊断（零训练成本）：若胜率集中在少数 UTC 小时，说明「全天候摊薄」是差距来源之一
        hv = np.concatenate(p["h"])
        yv2 = np.concatenate(p["y"])
        sv2 = np.concatenate(p["s"])
        wv = ((sv2 > 0).astype(int) == yv2).astype(np.int8)
        print(f"\n[主操作点 tpd={PRIMARY_TPD:g}] 按 UTC 小时：单量/胜率")
        cells = []
        for hh in range(24):
            mk = hv == hh
            n_h = int(mk.sum())
            if n_h >= 20:
                cells.append(f"{hh:02d}h:{n_h:>4} {wv[mk].mean() * 100:>5.1f}%")
            else:
                cells.append(f"{hh:02d}h:{n_h:>4}   —  ")
        for i in range(0, 24, 6):
            print("  " + " | ".join(cells[i:i + 6]))
    if hf_y:
        sv3 = np.concatenate(hf_s)
        yv3 = np.concatenate(hf_y)
        w3 = ((sv3 > 0).astype(int) == yv3)
        win3 = float(w3.mean())
        nd_all = sum(r["nd"] for r in rows)
        ev3 = "".join(f"{p * win3 - (1 - win3):>+9.3f}" for p in PAYOUTS)
        print(f"\n[时段筛选 tpd={PRIMARY_TPD:g}] n={len(w3)} 单/天={len(w3) / nd_all:.1f} "
              f"胜率={win3 * 100:.2f}% EV={ev3}  ← 白名单由训练窗定，无前视")
    log("DONE")


if __name__ == "__main__":
    main()
