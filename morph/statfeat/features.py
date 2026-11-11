"""统计形态特征（全部因果：只用 t 及之前的数据）。

动机（用户 2026-11-11 观察）：
1. CVD 是累积量，日线级别长期向下且斜率稳定 → **水平值无信息**，
   有信息的是「斜率被打破」以及现货/合约斜率「不再同步」。
2. 5 分钟级别，暴跌后若 CVD 出现反弹（价格新低、CVD 不再新低），
   后续多根 K 上涨概率回升 → 价格与累积主动量的**背离**。
3. 周线 stoch 长期偏在 0 轴下方 → 固定阈值 50 无意义，应使用**滚动经验分位**。

因此本模块不放「CVD 水平值」「stoch 绝对值」这类量，只放相对自身历史形态的偏离。
每个特征必须过 validate.py 的 7 关才放行。

**2026-11-11 全历史验证结论**（3,603,427 行 / 716,651 评估点 / 3 段 / h=3 与 h=15）：
- 放行 3/12：`cvdu_break_60_1440`、`cvdu_rebound5`、`cvds_rebound5`
  （IC 均为负，与已有 lg_* 最大 |r| 分别 0.424 / 0.447 / 0.372）
- 淘汰（换皮）：`ret5_z`(r=0.89 vs lg_ret5)、`stoch_rank7d`(0.98)、`stoch_z7d`(1.00)
- 淘汰（效应太弱）：`div_rebound`(方向符合观察但档差仅 1.4~2.1pp)、
  `price_cvd_newlow_div`、三个 slope_z/corr、两个负控
- 「滚动经验分位优于 z-score」不成立：rank 与 z 几乎等价且都只是原始 stoch 的换皮
- 放行 ≠ 能提升模型尾部：单变量 IC 显著不代表 LightGBM 的 top1% 会改善，
  下一步必须做增量价值验证（同段 A/B，多段配对）。

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


# ---------------------------------------------------------------- 扩展工具
def _moments(x: np.ndarray, w: int):
    """trailing 方差/偏度/超额峰度（O(N) 矩估计）。"""
    v = np.nan_to_num(np.asarray(x, dtype=np.float64), nan=0.0)
    m1, m2 = _boxcar(v, w), _boxcar(v * v, w)
    m3, m4 = _boxcar(v ** 3, w), _boxcar(v ** 4, w)
    var = np.maximum(m2 - m1 ** 2, 1e-18)
    skew = (m3 - 3 * m1 * m2 + 2 * m1 ** 3) / var ** 1.5
    kurt = (m4 - 4 * m1 * m3 + 6 * m1 ** 2 * m2 - 3 * m1 ** 4) / var ** 2 - 3
    return var, skew, kurt


def _run_len(mask: np.ndarray) -> np.ndarray:
    """当前连续满足条件的长度（因果，O(N)）。"""
    out = np.zeros(len(mask), dtype=np.float64)
    cur = 0
    for i, v in enumerate(mask):
        cur = cur + 1 if v else 0
        out[i] = cur
    return out


def _ret1(df: pl.DataFrame) -> np.ndarray:
    c = df["close"].to_numpy().astype(np.float64)
    r = np.zeros(len(c))
    r[1:] = c[1:] / c[:-1] - 1
    return r


def _linfit_resid(x: np.ndarray, w: int):
    """trailing 线性回归的 R² 与残差 z；x 对时间做回归，O(N)。

    R² 高 = 斜率稳定（用户图1 那些直线段）；残差突变 = 斜率被打破。
    """
    v = np.nan_to_num(np.asarray(x, dtype=np.float64), nan=0.0)
    t = np.arange(w, dtype=np.float64)
    t -= t.mean()
    denom = float(t @ t)
    sx, sxx = _boxcar(v, w), _boxcar(v * v, w)
    stx = np.full(len(v), np.nan)
    c = np.cumsum(np.insert(v, 0, 0.0))
    # trailing Σ(t_i - tbar)*v_i，用前缀和递推两项即可
    p1 = np.cumsum(np.insert(v * np.arange(len(v), dtype=np.float64), 0, 0.0))
    k = np.arange(len(v), dtype=np.float64)
    for_ = np.full(len(v), np.nan)
    for_[w - 1:] = (p1[w:] - p1[:-w]) - (w - 1) / 2 * (c[w:] - c[:-w])
    b = for_ / denom
    var = np.maximum(sxx - sx ** 2, 1e-18)
    r2 = np.clip((b ** 2) * denom / (w * var), 0, 1)
    resid = v - (sx + b * t[-1])          # 末端拟合值 vs 实际
    return r2, resid / (np.sqrt(var) + 1e-9)


# ---------------------------------------------------------------- D: 分布矩 / 偏斜
@register("ret_skew7d", "D_分布矩", "1m收益的7日滚动偏度：收益分布不对称")
def f_ret_skew(df):
    return _f(_moments(_ret1(df), WEEK)[1])


@register("ret_kurt7d", "D_分布矩", "1m收益的7日超额峰度：尾部厚度")
def f_ret_kurt(df):
    return _f(_moments(_ret1(df), WEEK)[2])


@register("down_frac7d", "D_分布矩", "过去7天下跌bar占比（用户『0轴下方比例大』的直接统计）")
def f_down_frac(df):
    r = _ret1(df)
    return _f(_boxcar((r < 0).astype(np.float64), WEEK))


@register("stoch_below50_frac7d", "D_分布矩", "stoch低于50的7日占比：分布偏斜的持续程度")
def f_stoch_below(df):
    s = _stoch(df)
    return _f(_boxcar((s < 50).astype(np.float64), WEEK))


@register("absret_iqr_ratio7d", "D_分布矩", "绝对收益的滚动IQR/std：尾部相对厚度")
def f_iqr_ratio(df):
    r = np.abs(_ret1(df))
    s = pl.Series("r", r)
    q75 = s.rolling_quantile(0.75, window_size=WEEK, min_samples=DAY).to_numpy()
    q25 = s.rolling_quantile(0.25, window_size=WEEK, min_samples=DAY).to_numpy()
    sd = s.rolling_std(WEEK, min_samples=DAY).to_numpy()
    return _f((q75 - q25) / (sd + 1e-12))


# ---------------------------------------------------------------- E: 游程 / 状态持续性
@register("up_run_len", "E_游程", "当前连续上涨bar长度")
def f_up_run(df):
    return _f(_run_len(_ret1(df) > 0))


@register("down_run_len", "E_游程", "当前连续下跌bar长度")
def f_down_run(df):
    return _f(_run_len(_ret1(df) < 0))


@register("cvd_down_run", "E_游程", "合约CVD连续下降的时长（用户：长期向下且斜率稳定）")
def f_cvd_down_run(df):
    s = _slope(_cvd(df, "um"), 60)
    return _f(_run_len(s < 0))


@register("stoch_below_run", "E_游程", "stoch持续低于50的时长")
def f_stoch_below_run(df):
    return _f(_run_len(_stoch(df) < 50))


@register("ret_var_ratio_60_15", "F_长记忆", "方差比VR(60,15)：>1趋势、<1均值回复")
def f_var_ratio(df):
    r = _ret1(df)
    v = np.maximum(_moments(r, 60)[0], 1e-18)
    v15 = np.maximum(_moments(r, 15)[0], 1e-18)
    return _f(v / (4 * v15 + 1e-18))


@register("ac1_ret7d", "F_长记忆", "1m收益一阶自相关的7日滚动估计")
def f_ac1(df):
    r = _ret1(df)
    a, b = r[1:], r[:-1]
    prod = np.concatenate([[0.0], a * b])
    m = _boxcar(r, WEEK)
    # trailing cov(r_t, r_{t-1}) / var
    cov = _boxcar(prod, WEEK)
    var = np.maximum(_moments(r, WEEK)[0], 1e-18)
    return _f(cov / var)


@register("rv_regime_z7d", "F_长记忆", "已实现波动的7日滚动z：波动状态")
def f_rv_z(df):
    r = _ret1(df)
    rv = np.sqrt(np.maximum(_moments(r, 60)[0], 1e-18))
    return _f(_roll_z(rv, WEEK))


# ---------------------------------------------------------------- G: CVD 形态回归
@register("cvdu_fit_r2_1d", "G_CVD形态", "合约CVD对时间滚动回归R²：斜率有多稳定")
def f_cvdu_r2(df):
    return _f(_linfit_resid(_cvd(df, "um"), DAY)[0])


@register("cvdu_fit_resid_z", "G_CVD形态", "合约CVD偏离拟合直线的残差z：斜率被打破")
def f_cvdu_resid(df):
    return _f(_linfit_resid(_cvd(df, "um"), DAY)[1])


@register("cvd_accel_1d", "G_CVD形态", "合约CVD斜率的二阶变化（60m斜率的变化率）")
def f_cvd_accel(df):
    s = _slope(_cvd(df, "um"), 60)
    return _f(_roll_z(_slope(s, 60), WEEK))


@register("cvd_price_beta1d", "G_CVD形态", "CVD与价格的1日滚动beta：量价耦合强度")
def f_cvd_beta(df):
    cv, r = _cvd(df, "um"), _ret1(df)
    cov = _boxcar(cv * r, DAY) - _boxcar(cv, DAY) * _boxcar(r, DAY)
    return _f(cov / (np.maximum(_moments(r, DAY)[0], 1e-18)))


# ---------------------------------------------------------------- H: 跨市场 / 极值
@register("cvd_slope_spread1d", "H_跨市场", "现货与合约CVD 1日斜率之差的z：两侧资金分歧")
def f_slope_spread(df):
    a = _slope(_cvd(df, "spot"), DAY)
    b = _slope(_cvd(df, "um"), DAY)
    return _f(_roll_z(a - b, WEEK))


@register("vol_ratio_spot_um", "H_跨市场", "现货/合约成交量比的滚动z")
def f_vol_ratio(df):
    sv = _col(df, "spot_buy_vol") + _col(df, "spot_sell_vol")
    uv = _col(df, "um_buy_vol") + _col(df, "um_sell_vol")
    return _f(_roll_z(sv / (uv + 1e-9), WEEK))


@register("basis_z7d", "H_跨市场", "perp-spot基差的7日滚动z（若含spot_close）")
def f_basis(df):
    if "spot_close" not in df.columns:
        return _f(np.full(df.height, np.nan))
    sp = df["spot_close"].fill_null(0).to_numpy().astype(np.float64)
    c = df["close"].to_numpy().astype(np.float64)
    return _f(_roll_z((c / (sp + 1e-9) - 1) * 1e4, WEEK))


@register("maxdd_1d", "I_极值", "过去1日价格最大回撤")
def f_maxdd(df):
    c = df["close"].to_numpy().astype(np.float64)
    peak = pl.Series("c", c).rolling_max(DAY, min_samples=DAY // 2).to_numpy()
    return _f(c / (peak + 1e-9) - 1)


@register("tail_asym7d", "I_极值", "正/负极值幅度比：上下尾不对称")
def f_tail_asym(df):
    r = _ret1(df)
    hi = pl.Series("r", r).rolling_max(WEEK, min_samples=DAY).to_numpy()
    lo = pl.Series("r", r).rolling_min(WEEK, min_samples=DAY).to_numpy()
    return _f((hi + lo) / (hi - lo + 1e-12))


def build_all(df: pl.DataFrame) -> dict[str, np.ndarray]:
    return {s.name: s.fn(df) for s in REGISTRY}
