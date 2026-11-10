"""统计形态特征（全部因果：只用 t 及之前的数据）。

动机（用户 2026-11-11 观察）：
1. CVD 是累积量，日线级别长期向下且斜率稳定 → **水平值无信息**，
   有信息的是「斜率被打破」以及现货/合约斜率「不再同步」。
2. 5 分钟级别，暴跌后若 CVD 出现反弹（价格新低、CVD 不再新低），
   后续多根 K 上涨概率回升 → 价格与累积主动量的**背离**。
3. 周线 stoch 长期偏在 0 轴下方 → 固定阈值 50 无意义，应使用**滚动经验分位**。

因此本模块不放「CVD 水平值」「stoch 绝对值」这类量，只放相对自身历史形态的偏离。
每个特征必须过 validate.py 的因果性 / 跨段 IC / FDR / 去相关四关才放行。

约定：fn(df) -> np.ndarray[float32]，与 df 等长，前若干根为 NaN（冷启动）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import polars as pl

DAY = 1440          # 1 分钟 bar 的一天
WEEK = 7 * DAY


@dataclass(frozen=True)
class Spec:
    name: str
    family: str
    fn: Callable[[pl.DataFrame], np.ndarray]
    hypothesis: str
    expect_pass: bool = True      # False = 负控，用来证明验证框架真能淘汰伪特征


REGISTRY: list[Spec] = []


def register(name: str, family: str, hypothesis: str, expect_pass: bool = True):
    def deco(fn):
        REGISTRY.append(Spec(name, family, fn, hypothesis, expect_pass))
        return fn
    return deco


# ---------------------------------------------------------------- 因果工具
def _col(df: pl.DataFrame, name: str) -> np.ndarray:
    return df[name].fill_null(0).to_numpy().astype(np.float64)


def _cvd(df: pl.DataFrame, side: str) -> np.ndarray:
    """前缀累积净主动买；水平值只作中间量，不单独注册为特征。"""
    b, s = (("um_buy_vol", "um_sell_vol") if side == "um"
            else ("spot_buy_vol", "spot_sell_vol"))
    return np.cumsum(_col(df, b) - _col(df, s))


def _slope(x: np.ndarray, w: int) -> np.ndarray:
    """(x[t]-x[t-w])/w：w 窗口净变化率，因果（含 t，不含未来）。"""
    out = np.full(len(x), np.nan, dtype=np.float64)
    out[w:] = (x[w:] - x[:-w]) / w
    return out


def _roll_z(x: np.ndarray, w: int) -> np.ndarray:
    """trailing z-score（窗口以 t 结尾），因果。"""
    s = pl.Series("x", x)
    m = s.rolling_mean(w, min_samples=max(2, w // 2)).to_numpy()
    sd = s.rolling_std(w, min_samples=max(2, w // 2)).to_numpy()
    return (x - m) / (sd + 1e-9)


def _roll_rank(x: np.ndarray, w: int) -> np.ndarray:
    """trailing 经验分位（分位网格近似），因果。

    用 5 个 rolling_quantile 网格把当前值映射到 [0,1]；比 z-score 更能处理
    「stoch 长期偏在 0 轴下方」这类分布偏斜。
    """
    s = pl.Series("x", x)
    grid = (0.1, 0.3, 0.5, 0.7, 0.9)
    r = np.zeros(len(x), dtype=np.float64)
    for q in grid:
        qv = s.rolling_quantile(q, window_size=w, min_samples=max(2, w // 2)).to_numpy()
        r += (x > qv).astype(np.float64)
    return (r + 0.5) / (len(grid) + 1)


def _roll_min(x: np.ndarray, w: int) -> np.ndarray:
    return pl.Series("x", x).rolling_min(w, min_samples=1).to_numpy()


def _roll_max(x: np.ndarray, w: int) -> np.ndarray:
    return pl.Series("x", x).rolling_max(w, min_samples=1).to_numpy()


def _boxcar(x: np.ndarray, w: int) -> np.ndarray:
    """O(N) trailing 均值；NaN 填 0（冷启动段之后才有评估点，不影响）。"""
    v = np.nan_to_num(np.asarray(x, dtype=np.float64), nan=0.0)
    c = np.cumsum(np.insert(v, 0, 0.0))
    out = np.full(len(v), np.nan)
    out[w - 1:] = (c[w:] - c[:-w]) / w
    return out


def _roll_corr(a: np.ndarray, b: np.ndarray, w: int) -> np.ndarray:
    """O(N) trailing Pearson 相关（部分 polars 版本没有 rolling_corr）。"""
    aa = np.nan_to_num(np.asarray(a, dtype=np.float64), nan=0.0)
    bb = np.nan_to_num(np.asarray(b, dtype=np.float64), nan=0.0)
    ma, mb, mab = _boxcar(aa, w), _boxcar(bb, w), _boxcar(aa * bb, w)
    va, vb = _boxcar(aa * aa, w) - ma ** 2, _boxcar(bb * bb, w) - mb ** 2
    return (mab - ma * mb) / np.sqrt(np.maximum(va * vb, 1e-18))


def _f(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


# ---------------------------------------------------------------- A: CVD 斜率突破
@register("cvdu_slope_z7d", "A_斜率突破",
          "合约CVD 1日斜率相对过去7日的z：斜率被打破向上时未来偏涨")
def f_cvdu_slope_z(df):
    return _f(_roll_z(_slope(_cvd(df, "um"), DAY), WEEK))


@register("cvds_slope_z7d", "A_斜率突破",
          "现货CVD 1日斜率相对过去7日的z")
def f_cvds_slope_z(df):
    return _f(_roll_z(_slope(_cvd(df, "spot"), DAY), WEEK))


@register("cvdu_break_60_1440", "A_斜率突破",
          "合约CVD 短斜率(60m) 减 长斜率(1d) 的z：短端率先转向")
def f_cvdu_break(df):
    c = _cvd(df, "um")
    return _f(_roll_z(_slope(c, 60) - _slope(c, DAY), WEEK))


@register("cvd_slope_div", "A_斜率突破",
          "合约与现货CVD斜率z之差：长期同步被打破时有信息")
def f_cvd_slope_div(df):
    zu = _roll_z(_slope(_cvd(df, "um"), 60), WEEK)
    zs = _roll_z(_slope(_cvd(df, "spot"), 60), WEEK)
    return _f(zu - zs)


@register("cvd_slope_corr7d", "A_斜率突破",
          "双侧CVD 60m斜率的7日滚动相关：同步性下降（分歧）")
def f_cvd_slope_corr(df):
    a = _slope(_cvd(df, "um"), 60)
    b = _slope(_cvd(df, "spot"), 60)
    return _f(_roll_corr(a, b, WEEK))


# ---------------------------------------------------------------- B: 暴跌后 CVD 反弹
@register("ret5_z", "B_暴跌反弹",
          "5分钟收益的滚动z：用相对分位定义『暴跌』，不用固定阈值")
def f_ret5_z(df):
    c = df["close"].to_numpy().astype(np.float64)
    r = np.full(len(c), np.nan)
    r[5:] = c[5:] / c[:-5] - 1
    return _f(_roll_z(r, WEEK))


@register("cvdu_rebound5", "B_暴跌反弹",
          "合约CVD 相对过去5根最低点的反弹幅度（标准化）")
def f_cvdu_rebound(df):
    c = _cvd(df, "um")
    lo, hi = _roll_min(c, 5), _roll_max(c, 5)
    return _f((c - lo) / (hi - lo + 1e-9))


@register("cvds_rebound5", "B_暴跌反弹",
          "现货CVD 相对过去5根最低点的反弹幅度")
def f_cvds_rebound(df):
    c = _cvd(df, "spot")
    lo, hi = _roll_min(c, 5), _roll_max(c, 5)
    return _f((c - lo) / (hi - lo + 1e-9))


@register("div_rebound", "B_暴跌反弹",
          "背离强度 = 价格下跌程度 × CVD反弹幅度（图2 红箭头）")
def f_div_rebound(df):
    cpx = df["close"].to_numpy().astype(np.float64)
    r = np.full(len(cpx), np.nan)
    r[5:] = cpx[5:] / cpx[:-5] - 1
    down = np.clip(-_roll_z(r, WEEK), 0, None)
    cv = _cvd(df, "um")
    lo, hi = _roll_min(cv, 5), _roll_max(cv, 5)
    return _f(down * (cv - lo) / (hi - lo + 1e-9))


@register("price_cvd_newlow_div", "B_暴跌反弹",
          "价格创5根新低但合约CVD未创新低（背离布尔）")
def f_newlow_div(df):
    cpx = df["close"].to_numpy().astype(np.float64)
    cv = _cvd(df, "um")
    plow = cpx <= _roll_min(cpx, 5)
    cnew = cv <= _roll_min(cv, 5)
    return _f((plow & ~cnew).astype(np.float64))


# ---------------------------------------------------------------- C: 分布偏斜 / 滚动分位
def _stoch(df) -> np.ndarray:
    hi = pl.Series("h", df["high"].to_numpy()).rolling_max(29, min_samples=15).to_numpy()
    lo = pl.Series("l", df["low"].to_numpy()).rolling_min(29, min_samples=15).to_numpy()
    c = df["close"].to_numpy().astype(np.float64)
    return (c - lo) / (hi - lo + 1e-9) * 100


@register("stoch_rank7d", "C_滚动分位",
          "stoch 相对过去7日的经验分位：修正『长期偏在0轴下方』的分布偏斜")
def f_stoch_rank(df):
    return _f(_roll_rank(_stoch(df), WEEK))


@register("stoch_z7d", "C_滚动分位",
          "stoch 的7日滚动z（与经验分位对照，检验偏斜是否重要）")
def f_stoch_z(df):
    return _f(_roll_z(_stoch(df), WEEK))


# ---------------------------------------------------------------- 负控（应当被淘汰）
@register("CTRL_cvd_level", "Z_负控",
          "CVD水平值：与时间共线，预期无真实信息；用于验证框架能否淘汰伪特征",
          expect_pass=False)
def f_ctrl_level(df):
    return _f(_cvd(df, "um") / 1e6)


@register("CTRL_noise", "Z_负控",
          "固定种子白噪声：用于校准多重检验假阳性率", expect_pass=False)
def f_ctrl_noise(df):
    return _f(np.random.default_rng(20261111).standard_normal(df.height))


def build_all(df: pl.DataFrame) -> dict[str, np.ndarray]:
    return {s.name: s.fn(df) for s in REGISTRY}
