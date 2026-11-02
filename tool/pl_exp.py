"""Polars 技术指标与特征表达式工具集 (PolarsExp)。

设计约定
--------
- 优先用 polars 表达式 (``pl.Expr``) 实现，便于在 ``LazyFrame``/``DataFrame`` 上
  ``.with_columns(...)`` 组合；只有**必须做状态机迭代**的指标（如 SuperTrend）才退化为
  逐行计算（DataFrame 级方法），并明确标注。
- 所有"指标"方法默认返回 ``pl.Expr``（可命名），方便拼进流水线；返回多个列的方法返回
  元组。
- 时间窗口类滚动一律用 ``rolling_*``（按行号滑动），不做全局标准化。
- 量纲：指标本身保留原始口径（如 RSI 0~100、价格相关以 bps 表示），是否缩放交由调用方；
  本模块另提供一批**无量纲/稳健**辅助方法（见 ``Dimensionless / Robust helpers`` 节），
  用于下游特征工程（中位数优先、滚动标准化、长短期结合）。

注意：本文件只实现"指标计算"，不含任何前视（所有窗口只看 t 及以前）。
"""

import polars as pl
from polars import col
from typing import Optional, Tuple, Union
import numpy as np


class PolarsExp:

    # ===================== 私有辅助方法 =====================
    @staticmethod
    def _true_range(high_col: str = "high", low_col: str = "low", close_col: str = "close") -> pl.Expr:
        """真实波幅 (True Range)。"""
        tr1 = (col(high_col) - col(low_col)).abs()
        prev_close = col(close_col).shift(1).fill_null(col(close_col))
        tr2 = (col(high_col) - prev_close).abs()
        tr3 = (col(low_col) - prev_close).abs()
        return pl.max_horizontal(tr1, tr2, tr3)

    @staticmethod
    def _typical_price(high_col: str = "high", low_col: str = "low", close_col: str = "close") -> pl.Expr:
        """典型价格 (Typical Price) = (H+L+C)/3。"""
        return (col(high_col) + col(low_col) + col(close_col)) / 3

    @staticmethod
    def _delta_gain_loss(price_col: str = "close") -> Tuple[pl.Expr, pl.Expr, pl.Expr]:
        """价格一阶差分、上涨量、下跌量。"""
        delta = col(price_col).diff(1)
        gain = pl.when(delta > 0).then(delta).otherwise(0.0)
        loss = pl.when(delta < 0).then(-delta).otherwise(0.0)
        return delta, gain, loss

    @staticmethod
    def _rolling_high_low(high_col: str = "high", low_col: str = "low", period: int = 14) -> Tuple[pl.Expr, pl.Expr]:
        """滚动最高价 / 最低价（rolling_max / rolling_min）。"""
        highest_high = col(high_col).rolling_max(window_size=period)
        lowest_low = col(low_col).rolling_min(window_size=period)
        return highest_high, lowest_low

    @staticmethod
    def _rolling_mean_std(price_col: str = "close", period: int = 20) -> Tuple[pl.Expr, pl.Expr]:
        """滚动均值 / 标准差。"""
        mean_val = col(price_col).rolling_mean(window_size=period)
        std_val = col(price_col).rolling_std(window_size=period)
        return mean_val, std_val

    # ===================== 基础均线 =====================
    @staticmethod
    def ema(price_col: str = "close", period: int = 20, alias_prefix: str = "", alias_suffix: bool = False) -> pl.Expr:
        """指数移动平均 (EMA)。alpha = 2/(period+1)（标准 EMA）。"""
        _ema = col(price_col).ewm_mean(span=period, adjust=False)
        suffix = f"_{period}" if alias_suffix else ""
        return _ema.alias(f"{alias_prefix}ema{suffix}")

    @staticmethod
    def sma(price_col: str = "close", period: int = 5, alias_prefix: str = "", alias_suffix: bool = False) -> pl.Expr:
        """简单移动平均 (SMA)。"""
        _sma = col(price_col).rolling_mean(window_size=period)
        suffix = f"_{period}" if alias_suffix else ""
        return _sma.alias(f"{alias_prefix}sma{suffix}")

    # ===================== MACD =====================
    @staticmethod
    def macd(price_col: str = "close",
             fast_period: int = 12,
             slow_period: int = 26,
             signal_period: int = 9,
             alias_prefix: str = "",
             alias_suffix: bool = False) -> Tuple[pl.Expr, pl.Expr, pl.Expr]:
        """MACD (DIF, DEA, HIST)。柱状采用 TA-Lib 标准 HIST = DIF - DEA。

        注意：MACD 绝对值与价格水平绑定，单点值跨价位不可比，下游如需使用应做相对化。
        """
        fast_ema = col(price_col).ewm_mean(span=fast_period, adjust=False)
        slow_ema = col(price_col).ewm_mean(span=slow_period, adjust=False)

        dif = fast_ema - slow_ema
        dea = dif.ewm_mean(span=signal_period, adjust=False)
        bar = dif - dea  # TA-Lib 标准柱状

        suffix = f"_{fast_period}_{slow_period}_{signal_period}" if alias_suffix else ""
        return (
            dif.alias(f"{alias_prefix}macd{suffix}"),
            dea.alias(f"{alias_prefix}macd_sig{suffix}"),
            bar.alias(f"{alias_prefix}macd_hist{suffix}"),
        )

    # ===================== RSI =====================
    @staticmethod
    def rsi(price_col: str = "close", period: int = 14, alias_prefix: str = "", alias_suffix: bool = False) -> pl.Expr:
        """RSI (0~100)，采用 Wilder 平滑 (alpha = 1/period)。

        前 ``period`` 根因种子（SMA 初值）与纯 ewm 有微小差异，下游跨价位的"单点 RSI"
        意义有限，建议配合 ``rsi_pair`` / ``rsi_scaled`` 使用。
        """
        delta, gain, loss = PolarsExp._delta_gain_loss(price_col)
        avg_gain = gain.ewm_mean(alpha=1.0 / period, adjust=False)
        avg_loss = loss.ewm_mean(alpha=1.0 / period, adjust=False)
        rs = avg_gain / pl.when(avg_loss != 0).then(avg_loss).otherwise(1e-6)
        rsi_expr = 100 - (100 / (1 + rs))
        suffix = f"_{period}" if alias_suffix else ""
        return rsi_expr.alias(f"{alias_prefix}rsi{suffix}")

    # ===================== Stochastic RSI =====================
    @staticmethod
    def stochrsi(price_col: str = "close", rsi_period: int = 14, stoch_period: int = 14, d_period: int = 3,
                 alias_prefix: str = "", alias_suffix: bool = False) -> Tuple[pl.Expr, pl.Expr]:
        """Stochastic RSI (%K, %D)，内部 RSI 用 Wilder 平滑。

        %K = (RSI - min(RSI)) / (max(RSI) - min(RSI)) * 100；区间为零或缺失时填 50。
        """
        delta, gain, loss = PolarsExp._delta_gain_loss(price_col)
        avg_gain = gain.ewm_mean(alpha=1.0 / rsi_period, adjust=False)
        avg_loss = loss.ewm_mean(alpha=1.0 / rsi_period, adjust=False)
        rs = avg_gain / pl.when(avg_loss != 0).then(avg_loss).otherwise(1e-6)
        rsi_expr = 100 - (100 / (1 + rs))

        lowest_rsi = rsi_expr.rolling_min(window_size=stoch_period, min_samples=stoch_period)
        highest_rsi = rsi_expr.rolling_max(window_size=stoch_period, min_samples=stoch_period)
        range_rsi = highest_rsi - lowest_rsi
        k_denom = pl.when(range_rsi.is_null() | (range_rsi == 0)).then(1e-6).otherwise(range_rsi)

        stochrsi_k_expr = ((rsi_expr - lowest_rsi) / k_denom * 100).fill_null(50.0).fill_nan(50.0)
        stochrsi_d_expr = stochrsi_k_expr.rolling_mean(window_size=d_period, min_samples=1)

        suffix = f"_{rsi_period}_{stoch_period}_{d_period}" if alias_suffix else ""
        return (
            stochrsi_k_expr.alias(f"{alias_prefix}stochrsi_k{suffix}"),
            stochrsi_d_expr.alias(f"{alias_prefix}stochrsi_d{suffix}"),
        )

    # ===================== 布林带 =====================
    @staticmethod
    def bbands(price_col: str = "close", period: int = 20, num_std: float = 2.0,
               alias_prefix: str = "", alias_suffix: bool = False) -> Tuple[pl.Expr, pl.Expr, pl.Expr, pl.Expr, pl.Expr]:
        """布林带 (upper, middle, lower, %B, BBW)。

        BBW 以 middle 的百分比表示（无量纲）；%B 落在 [0,100]。
        """
        middle_band = col(price_col).rolling_mean(window_size=period)
        rolling_std = col(price_col).rolling_std(window_size=period)

        upper_band = middle_band + (rolling_std * num_std)
        lower_band = middle_band - (rolling_std * num_std)

        denom = pl.when((upper_band - lower_band) != 0).then(upper_band - lower_band).otherwise(None)
        bbw_expr = (upper_band - lower_band) / middle_band * 100
        bbp_expr = (col(price_col) - lower_band) / denom * 100

        suffix = f"_{period}_{num_std}" if alias_suffix else ""
        period_suffix = f"_{period}" if alias_suffix else ""
        return (
            upper_band.alias(f"{alias_prefix}bb_upper{suffix}"),
            middle_band.alias(f"{alias_prefix}bb_middle{period_suffix}"),
            lower_band.alias(f"{alias_prefix}bb_lower{suffix}"),
            bbw_expr.alias(f"{alias_prefix}bbw{period_suffix}"),
            bbp_expr.alias(f"{alias_prefix}bbp{period_suffix}"),
        )

    # ===================== ATR =====================
    @staticmethod
    def atr(high_col: str = "high", low_col: str = "low", close_col: str = "close",
            period: int = 14, alias_prefix: str = "", alias_suffix: bool = False) -> pl.Expr:
        """平均真实波幅 (ATR)，Wilder 平滑 (alpha = 1/period)。

        ATR 是绝对价格幅度（带量纲），下游跨价位比较时应除以价格做相对化（如 ATR%）。
        """
        true_range = PolarsExp._true_range(high_col, low_col, close_col)
        atr_expr = true_range.ewm_mean(alpha=1.0 / period, adjust=False)
        suffix = f"_{period}" if alias_suffix else ""
        return atr_expr.alias(f"{alias_prefix}atr{suffix}")

    # ===================== 随机指标 (STOCH) =====================
    @staticmethod
    def stoch(high_col: str = "high", low_col: str = "low", close_col: str = "close",
              k_period: int = 14, d_period: int = 3, k_smooth: int = 1,
              alias_prefix: str = "", alias_suffix: bool = False) -> Tuple[pl.Expr, pl.Expr]:
        """随机指标 %K / %D（慢速随机：(fastK, slowK, slowD)）。

        %K = 100 * (C - LL) / (HH - LL)，在 ``k_period`` 窗口内；
        ``k_smooth`` > 1 时对 %K 再做 ``rolling_mean`` 平滑（慢速 %K，默认 1 = 不平滑）；
        %D = 平滑后 %K 的 ``rolling_mean``（窗口 ``d_period``）。
        区间 (H-L) 为 0 或缺失时 %K 置空，避免除零。仅输出 %K / %D 两路。
        """
        highest_high, lowest_low = PolarsExp._rolling_high_low(high_col, low_col, k_period)
        rng = highest_high - lowest_low
        denom = pl.when(rng != 0).then(rng).otherwise(None)
        raw_k = (col(close_col) - lowest_low) / denom * 100
        k_expr = raw_k.rolling_mean(window_size=k_smooth) if k_smooth > 1 else raw_k
        d_expr = k_expr.rolling_mean(window_size=d_period)

        suffix = f"_{k_period}_{d_period}" if alias_suffix else ""
        return (
            k_expr.alias(f"{alias_prefix}stoch_k{suffix}"),
            d_expr.alias(f"{alias_prefix}stoch_d{suffix}"),
        )

    # ===================== KDJ =====================
    @staticmethod
    def kdj(high_col: str = "high", low_col: str = "low", close_col: str = "close",
            n: int = 9, smooth: int = 3, alias_prefix: str = "", alias_suffix: bool = False) -> Tuple[pl.Expr, pl.Expr, pl.Expr]:
        """KDJ (K, D, J)。RSV 窗口 n；K/D 用经典平滑 alpha = 1/smooth（默认 3 → 2/3 递推）。

        K_t = (1-1/smooth) K_{t-1} + (1/smooth) RSV_t，D 同理作用于 K。这是 EMA 递推近似，
        与从 50 初始值的精确递推仅有初值差异。
        """
        highest_high, lowest_low = PolarsExp._rolling_high_low(high_col, low_col, n)
        rng = highest_high - lowest_low
        rsv_denom = pl.when(rng != 0).then(rng).otherwise(1e-6)
        rsv = ((col(close_col) - lowest_low) / rsv_denom * 100).fill_null(50.0)

        a = 1.0 / smooth
        k_expr = rsv.ewm_mean(alpha=a, adjust=False).fill_null(50.0)
        d_expr = k_expr.ewm_mean(alpha=a, adjust=False).fill_null(50.0)
        j_expr = 3 * k_expr - 2 * d_expr

        suffix = f"_{n}_{smooth}" if alias_suffix else ""
        return (
            k_expr.alias(f"{alias_prefix}k{suffix}"),
            d_expr.alias(f"{alias_prefix}d{suffix}"),
            j_expr.alias(f"{alias_prefix}j{suffix}"),
        )

    # ===================== SuperTrend（状态机，DataFrame 级） =====================
    @staticmethod
    def supertrend(df: pl.DataFrame, high_col: str = "high", low_col: str = "low", close_col: str = "close",
                   period: int = 10, multiplier: float = 3.0, prefix: str = "", suffix: str = "") -> pl.DataFrame:
        """SuperTrend（正确状态机实现）。

        SuperTrend 的上/下轨依赖前一根的最终轨（无法用纯向量化表达式表示），故在 DataFrame
        级用 numpy 逐行迭代。返回原 df 追加两列：``{prefix}supertrend{suffix}`` 与
        ``{prefix}supertrend_dir{suffix}``（dir: 1 多头 / -1 空头）。
        """
        high = df[high_col].to_numpy().astype(float)
        low = df[low_col].to_numpy().astype(float)
        close = df[close_col].to_numpy().astype(float)
        n = len(close)
        if n == 0:
            return df

        tp = (high + low + close) / 3.0
        tr = np.empty(n, dtype=float)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))

        atr = np.empty(n, dtype=float)
        atr[0] = tr[0]
        for i in range(1, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

        basic_ub = tp + multiplier * atr
        basic_lb = tp - multiplier * atr

        final_ub = np.empty(n, dtype=float)
        final_lb = np.empty(n, dtype=float)
        st = np.empty(n, dtype=float)
        direction = np.empty(n, dtype=float)

        for i in range(n):
            if i == 0:
                final_ub[i] = basic_ub[i]
                final_lb[i] = basic_lb[i]
                st[i] = basic_ub[i]  # 初始任选一侧
                direction[i] = 1.0 if close[i] >= st[i] else -1.0
                continue
            # 上轨：仅当价格跌破前上轨时才下移，否则维持
            if basic_ub[i] < final_ub[i - 1] or close[i - 1] > final_ub[i - 1]:
                final_ub[i] = basic_ub[i]
            else:
                final_ub[i] = final_ub[i - 1]
            # 下轨
            if basic_lb[i] > final_lb[i - 1] or close[i - 1] < final_lb[i - 1]:
                final_lb[i] = basic_lb[i]
            else:
                final_lb[i] = final_lb[i - 1]
            # 确定 ST 与方向
            if st[i - 1] == final_ub[i - 1]:
                if close[i] <= final_ub[i]:
                    st[i] = final_ub[i]
                    direction[i] = -1.0
                else:
                    st[i] = final_lb[i]
                    direction[i] = 1.0
            else:  # 前一根为下轨
                if close[i] >= final_lb[i]:
                    st[i] = final_lb[i]
                    direction[i] = 1.0
                else:
                    st[i] = final_ub[i]
                    direction[i] = -1.0

        return df.with_columns([
            pl.Series(f"{prefix}supertrend{suffix}", st),
            pl.Series(f"{prefix}supertrend_dir{suffix}", direction),
        ])

    # ===================== Williams %R =====================
    @staticmethod
    def williams_r(high_col: str = "high", low_col: str = "low", close_col: str = "close",
                  period: int = 14, alias_prefix: str = "", alias_suffix: bool = False) -> pl.Expr:
        """Williams %R = -100 * (HH - C) / (HH - LL)。区间为 0 或缺失时置空。"""
        highest_high, lowest_low = PolarsExp._rolling_high_low(high_col, low_col, period)
        rng = highest_high - lowest_low
        denom = pl.when(rng != 0).then(rng).otherwise(None)
        wr_expr = (highest_high - col(close_col)) / denom * -100
        suffix = f"_{period}" if alias_suffix else ""
        return wr_expr.alias(f"{alias_prefix}wr{suffix}")

    # ===================== CCI =====================
    @staticmethod
    def cci(high_col: str = "high", low_col: str = "low", close_col: str = "close",
            period: int = 20, alias_prefix: str = "", alias_suffix: bool = False) -> pl.Expr:
        """商品通道指数 (CCI)。mean deviation 为 0 或缺失时置空。"""
        typical_price = PolarsExp._typical_price(high_col, low_col, close_col)
        sma_tp = typical_price.rolling_mean(window_size=period)
        mean_dev = (typical_price - sma_tp).abs().rolling_mean(window_size=period)
        denom = pl.when(mean_dev != 0).then(0.015 * mean_dev).otherwise(None)
        cci_expr = (typical_price - sma_tp) / denom
        suffix = f"_{period}" if alias_suffix else ""
        return cci_expr.alias(f"{alias_prefix}cci{suffix}")

    # ===================== 连续计数 =====================
    @staticmethod
    def consecutive_periods(boolean_col: str, alias_prefix: str = "", alias_suffix: bool = False) -> pl.Expr:
        """布尔列连续相同值的期数，正号=为 True 的连续段，负号=为 False 的连续段。"""
        run_id_expr = pl.col(boolean_col).rle_id()
        consecutive_count_expr = pl.int_range(1, pl.len() + 1).over(run_id_expr)
        signed = (
            pl.when(pl.col(boolean_col))
            .then(consecutive_count_expr)
            .otherwise(-consecutive_count_expr)
        )
        suffix = f"_{boolean_col}_consecutive" if alias_suffix else "_consecutive"
        return signed.alias(f"{alias_prefix}consecutive{suffix}")

    # ===================== 金叉/死叉信号 =====================
    @staticmethod
    def golden_cross(fast_col: str, slow_col: str, alias_prefix: str = "", alias_suffix: bool = False) -> pl.Expr:
        """金叉/死叉信号：1=金叉, -1=死叉, 0=无（仅在交叉当根标记）。"""
        fast = pl.col(fast_col)
        slow = pl.col(slow_col)
        fast_prev = fast.shift(1)
        slow_prev = slow.shift(1)
        cross_expr = (
            pl.when((fast_prev <= slow_prev) & (fast > slow)).then(pl.lit(1))
            .when((fast_prev >= slow_prev) & (fast < slow)).then(pl.lit(-1))
            .otherwise(pl.lit(0))
        )
        suffix = f"_{fast_col}_{slow_col}" if alias_suffix else ""
        return cross_expr.alias(f"{alias_prefix}cross_signal{suffix}")

    # ===================== 滚动分位数分箱 =====================
    @staticmethod
    def quantile_binning(target_col_name: str, window: int = 300, quantile_num: int = 20,
                         alias_prefix: str = "", alias_suffix: bool = False) -> pl.Expr:
        """对列做滚动分位数（等频）分箱，返回 0..quantile_num-1 的整数桶号。

        用 ``rolling_rank`` 计算每个点在滚动窗口内的百分位排名（1-based 平均秩，自动处理并列），
        再映射为等频分箱：``bucket = floor(pct * quantile_num)``，裁剪到 [0, quantile_num-1]。

        窗口计数用 ``is_not_null`` 的 ``rolling_sum`` 取得（warmup 期为展开长度，保证等频语义一致）；
        单点窗口（cnt==1）归入 0 号桶。
        """
        allowed = [10, 20, 50, 100]
        if quantile_num not in allowed:
            raise ValueError(f"quantile_num must be one of {allowed}, got {quantile_num}")

        cnt = (
            pl.col(target_col_name).is_not_null().cast(pl.Int64)
            .rolling_sum(window_size=window, min_samples=1)
        )
        rank = pl.col(target_col_name).rolling_rank(window_size=window, min_samples=1)
        pct = pl.when(cnt > 1).then((rank - 1) / (cnt - 1)).otherwise(pl.lit(0.0))
        bin_expr = ((pct * quantile_num).floor()).clip(0, quantile_num - 1).cast(pl.Int64)

        suffix = f"_{window}_{quantile_num}" if alias_suffix else ""
        return bin_expr.alias(f"{alias_prefix}{target_col_name}_bin{suffix}")

    # ===================== 动量因子 =====================
    @staticmethod
    def roc(price_col: str = "close", period: int = 10, alias_prefix: str = "", alias_suffix: bool = False) -> pl.Expr:
        """变化率 ROC (%) = (C - C_shift) / C_shift * 100。"""
        base = col(price_col).shift(period)
        roc_expr = (col(price_col) - base) / pl.when(base != 0).then(base).otherwise(1e-6) * 100
        suffix = f"_{period}" if alias_suffix else ""
        return roc_expr.alias(f"{alias_prefix}roc{suffix}")

    @staticmethod
    def momentum(price_col: str = "close", period: int = 10, alias_prefix: str = "", alias_suffix: bool = False) -> pl.Expr:
        """动量 = C - C_shift(period)（绝对价格差，带量纲；跨价位不可比）。"""
        mom_expr = col(price_col) - col(price_col).shift(period)
        suffix = f"_{period}" if alias_suffix else ""
        return mom_expr.alias(f"{alias_prefix}mom{suffix}")

    @staticmethod
    def mfi(high_col: str = "high", low_col: str = "low", close_col: str = "close",
            volume_col: str = "volume", period: int = 14, alias_prefix: str = "", alias_suffix: bool = False) -> pl.Expr:
        """资金流量指数 (MFI, 0~100)。"""
        typical_price = PolarsExp._typical_price(high_col, low_col, close_col)
        money_flow = typical_price * col(volume_col)
        positive_flow = pl.when(typical_price > typical_price.shift(1)).then(money_flow).otherwise(0.0)
        negative_flow = pl.when(typical_price < typical_price.shift(1)).then(money_flow).otherwise(0.0)
        positive_mf = positive_flow.rolling_sum(window_size=period)
        negative_mf = negative_flow.rolling_sum(window_size=period)
        ratio = positive_mf / pl.when(negative_mf != 0).then(negative_mf).otherwise(1e-6)
        mfi_expr = 100 - (100 / (1 + ratio))
        suffix = f"_{period}" if alias_suffix else ""
        return mfi_expr.alias(f"{alias_prefix}mfi{suffix}")

    # ===================== 波动率因子 =====================
    @staticmethod
    def historical_volatility(price_col: str = "close", period: int = 20,
                              periods_per_year: Optional[float] = None,
                              alias_prefix: str = "", alias_suffix: bool = False) -> pl.Expr:
        """历史波动率（基于对数收益滚动 std）。

        ``periods_per_year`` 为 None 时不年化（单位：每根 bar 的对数收益 std）；
        传入如 525600（分钟级、加密 24/7）则年化并 ×100（百分比）。
        """
        log_return = (col(price_col) / col(price_col).shift(1)).log()
        hv = log_return.rolling_std(window_size=period)
        if periods_per_year is not None:
            hv = hv * np.sqrt(periods_per_year) * 100
        suffix = f"_{period}" if alias_suffix else ""
        return hv.alias(f"{alias_prefix}hv{suffix}")

    @staticmethod
    def parkinson_volatility(high_col: str = "high", low_col: str = "low", period: int = 20,
                             periods_per_year: Optional[float] = None,
                             alias_prefix: str = "", alias_suffix: bool = False) -> pl.Expr:
        """Parkinson 波动率（基于高低价极差）。同样用 ``periods_per_year`` 控制年化。"""
        hl_ratio = (col(high_col) / col(low_col)).log()
        pv = (hl_ratio ** 2).rolling_mean(window_size=period).sqrt() * np.sqrt(1.0 / (4.0 * np.log(2.0)))
        if periods_per_year is not None:
            pv = pv * np.sqrt(periods_per_year) * 100
        suffix = f"_{period}" if alias_suffix else ""
        return pv.alias(f"{alias_prefix}pv{suffix}")

    # ===================== 成交量因子 =====================
    @staticmethod
    def obv(close_col: str = "close", volume_col: str = "volume", alias_prefix: str = "", alias_suffix: bool = False) -> pl.Expr:
        """能量潮 (OBV) 累计值（绝对量，带量纲）。"""
        obv_change = (
            pl.when(col(close_col) > col(close_col).shift(1)).then(col(volume_col))
            .when(col(close_col) < col(close_col).shift(1)).then(-col(volume_col))
            .otherwise(0)
        )
        return obv_change.cum_sum().alias(f"{alias_prefix}obv")

    @staticmethod
    def volume_roc(volume_col: str = "volume", period: int = 10, alias_prefix: str = "", alias_suffix: bool = False) -> pl.Expr:
        """成交量变化率 (%)。"""
        base = col(volume_col).shift(period)
        vol_roc = (col(volume_col) - base) / pl.when(base != 0).then(base).otherwise(1e-6) * 100
        suffix = f"_{period}" if alias_suffix else ""
        return vol_roc.alias(f"{alias_prefix}vol_roc{suffix}")

    @staticmethod
    def volume_ma(volume_col: str = "volume", period: int = 20, alias_prefix: str = "", alias_suffix: bool = False) -> pl.Expr:
        """成交量移动平均。"""
        suffix = f"_{period}" if alias_suffix else ""
        return col(volume_col).rolling_mean(window_size=period).alias(f"{alias_prefix}vol_ma{suffix}")

    @staticmethod
    def vwap(high_col: str = "high", low_col: str = "low", close_col: str = "close",
             volume_col: str = "volume", alias_prefix: str = "", alias_suffix: bool = False) -> pl.Expr:
        """累计 VWAP（整段不重置；日内 VWAP 需自行按 session 分组后调用）。"""
        typical_price = PolarsExp._typical_price(high_col, low_col, close_col)
        tp_volume = typical_price * col(volume_col)
        vwap_expr = tp_volume.cum_sum() / col(volume_col).cum_sum()
        return vwap_expr.alias(f"{alias_prefix}vwap")

    @staticmethod
    def volume_ratio(volume_col: str = "volume", period: int = 20, alias_prefix: str = "", alias_suffix: bool = False) -> pl.Expr:
        """成交量比率 = V / MA(V) * 100。"""
        vol_ma = col(volume_col).rolling_mean(window_size=period)
        vr_expr = col(volume_col) / pl.when(vol_ma != 0).then(vol_ma).otherwise(1e-6) * 100
        suffix = f"_{period}" if alias_suffix else ""
        return vr_expr.alias(f"{alias_prefix}vr{suffix}")

    # ===================== 价格位置与统计 =====================
    @staticmethod
    def price_position(high_col: str = "high", low_col: str = "low", close_col: str = "close",
                       period: int = 20, alias_prefix: str = "", alias_suffix: bool = False) -> pl.Expr:
        """价格在滚动 [min low, max high] 区间内的相对位置 (0~100)。区间为 0/缺失时置空。"""
        highest, lowest = PolarsExp._rolling_high_low(high_col, low_col, period)
        rng = highest - lowest
        denom = pl.when(rng != 0).then(rng).otherwise(None)
        pp_expr = (col(close_col) - lowest) / denom * 100
        suffix = f"_{period}" if alias_suffix else ""
        return pp_expr.alias(f"{alias_prefix}pp{suffix}")

    @staticmethod
    def skewness(price_col: str = "close", period: int = 20, alias_prefix: str = "", alias_suffix: bool = False) -> pl.Expr:
        """滚动偏度（总体估计）。精确样本偏度建议用 scipy 在滑窗上计算。"""
        mean_val, std_val = PolarsExp._rolling_mean_std(price_col, period)
        skew_expr = ((col(price_col) - mean_val) / pl.when(std_val != 0).then(std_val).otherwise(1e-6)).pow(3).rolling_mean(window_size=period)
        suffix = f"_{period}" if alias_suffix else ""
        return skew_expr.alias(f"{alias_prefix}skew{suffix}")

    @staticmethod
    def kurtosis(price_col: str = "close", period: int = 20, alias_prefix: str = "", alias_suffix: bool = False) -> pl.Expr:
        """滚动峰度（总体估计，已减 3 即超额峰度）。"""
        mean_val, std_val = PolarsExp._rolling_mean_std(price_col, period)
        kurt_expr = ((col(price_col) - mean_val) / pl.when(std_val != 0).then(std_val).otherwise(1e-6)).pow(4).rolling_mean(window_size=period) - 3
        suffix = f"_{period}" if alias_suffix else ""
        return kurt_expr.alias(f"{alias_prefix}kurt{suffix}")

    # ===================== DMI / ADX =====================
    @staticmethod
    def dmi(high_col: str = "high", low_col: str = "low", close_col: str = "close",
            period: int = 14, alias_prefix: str = "", alias_suffix: bool = False) -> Tuple[pl.Expr, pl.Expr, pl.Expr, pl.Expr]:
        """DMI 趋向指标 (ADX, +DI, -DI, DX)，DM/TR 均用 Wilder 平滑 (alpha = 1/period)。"""
        high_prev = col(high_col).shift(1).fill_null(col(high_col))
        low_prev = col(low_col).shift(1).fill_null(col(low_col))

        up_move = col(high_col) - high_prev
        down_move = low_prev - col(low_col)
        plus_dm = pl.when((up_move > down_move) & (up_move > 0)).then(up_move).otherwise(0.0)
        minus_dm = pl.when((down_move > up_move) & (down_move > 0)).then(down_move).otherwise(0.0)

        true_range = PolarsExp._true_range(high_col, low_col, close_col)
        smoothed_plus_dm = plus_dm.ewm_mean(alpha=1.0 / period, adjust=False)
        smoothed_minus_dm = minus_dm.ewm_mean(alpha=1.0 / period, adjust=False)
        smoothed_tr = true_range.ewm_mean(alpha=1.0 / period, adjust=False)

        di_denom = pl.when(smoothed_tr != 0).then(smoothed_tr).otherwise(1e-6)
        plus_di = 100 * smoothed_plus_dm / di_denom
        minus_di = 100 * smoothed_minus_dm / di_denom
        dx_denom = pl.when((plus_di + minus_di) != 0).then(plus_di + minus_di).otherwise(1e-6)
        dx = 100 * (plus_di - minus_di).abs() / dx_denom
        adx = dx.ewm_mean(alpha=1.0 / period, adjust=False).fill_null(0.0).fill_nan(0.0)

        suffix = f"_{period}" if alias_suffix else ""
        return (
            adx.alias(f"{alias_prefix}adx{suffix}"),
            plus_di.alias(f"{alias_prefix}plus_di{suffix}"),
            minus_di.alias(f"{alias_prefix}minus_di{suffix}"),
            dx.alias(f"{alias_prefix}dx{suffix}"),
        )

    # ===================== 均线交叉 =====================
    @staticmethod
    def ema_cross(price_col: str = "close", fast_period: int = 12, slow_period: int = 26,
                  alias_prefix: str = "", alias_suffix: bool = False) -> pl.Expr:
        """EMA 交叉信号：1=金叉, -1=死叉, 0=无（仅交叉当根标记）。"""
        fast_ema = col(price_col).ewm_mean(span=fast_period, adjust=False)
        slow_ema = col(price_col).ewm_mean(span=slow_period, adjust=False)
        fast_prev = fast_ema.shift(1)
        slow_prev = slow_ema.shift(1)
        cross_expr = (
            pl.when((fast_prev <= slow_prev) & (fast_ema > slow_ema)).then(pl.lit(1))
            .when((fast_prev >= slow_prev) & (fast_ema < slow_ema)).then(pl.lit(-1))
            .otherwise(pl.lit(0))
        )
        suffix = f"_{fast_period}_{slow_period}" if alias_suffix else ""
        return cross_expr.alias(f"{alias_prefix}ema_cross{suffix}")

    @staticmethod
    def price_to_ema(price_col: str = "close", period: int = 20, alias_prefix: str = "", alias_suffix: bool = False) -> pl.Expr:
        """价格相对 EMA 的偏离 (%) = (C - EMA)/EMA * 100。"""
        ema_val = col(price_col).ewm_mean(span=period, adjust=False)
        ratio_expr = (col(price_col) - ema_val) / pl.when(ema_val != 0).then(ema_val).otherwise(1e-6) * 100
        suffix = f"_{period}" if alias_suffix else ""
        return ratio_expr.alias(f"{alias_prefix}price_ema_ratio{suffix}")

    # ===================== Dimensionless / Robust helpers =====================
    # 下面这批方法面向"下游特征工程"，遵循：无量纲、滚动（非全局）、中位数优先。
    @staticmethod
    def log_ret_bps(price_col: str = "close", period: int = 1, alias: Optional[str] = None) -> pl.Expr:
        """对数收益（bps）= ln(C / C_shift(period)) * 1e4。无量纲。"""
        return (col(price_col) / col(price_col).shift(period)).log().mul(1e4) \
            .alias(alias or f"{price_col}_lr{period}_bps")

    @staticmethod
    def rolling_median(target_col: str, window: int, min_samples: int = 1, alias: Optional[str] = None) -> pl.Expr:
        """滚动中位数。"""
        return col(target_col).rolling_median(window_size=window, min_samples=min_samples) \
            .alias(alias or f"{target_col}_med{window}")

    @staticmethod
    def rolling_mad(target_col: str, window: int, min_samples: int = 1, alias: Optional[str] = None) -> pl.Expr:
        """滚动 MAD = median(|x - rolling_median(x)|)。

        先取逐点滚动中位数，再对绝对偏差做滚动中位数（polars 原生 ``rolling_median``，
        全程向量化，无 ``map_batches``）。MAD 对异常值鲁棒。
        """
        med = col(target_col).rolling_median(window_size=window, min_samples=min_samples)
        mad = (col(target_col) - med).abs().rolling_median(window_size=window, min_samples=min_samples)
        return mad.alias(alias or f"{target_col}_mad{window}")

    @staticmethod
    def robust_zscore(target_col: str, window: int, min_samples: int = 1, alias: Optional[str] = None) -> pl.Expr:
        """稳健 z-score = (x - rolling_median(x)) / (1.4826 · rolling_MAD)。

        中位数 + MAD 对异常值鲁棒；MAD=0 时置空（避免除零 / 无意义）。
        与 ``rolling_mad`` 同口径，polars 原生向量化实现。
        """
        med = col(target_col).rolling_median(window_size=window, min_samples=min_samples)
        mad = (col(target_col) - med).abs().rolling_median(window_size=window, min_samples=min_samples)
        scale = 1.4826 * mad
        z = (col(target_col) - med) / pl.when(scale != 0).then(scale).otherwise(None)
        return z.alias(alias or f"{target_col}_rz{window}")

    @staticmethod
    def rsi_scaled(price_col: str = "close", period: int = 14, center: float = 50.0, scale: float = 50.0,
                   alias: Optional[str] = None) -> pl.Expr:
        """居中缩放后的 RSI：映射到 (RSI-center)/scale，理论区间 [-1,1]。"""
        rsi = PolarsExp.rsi(price_col, period)
        return ((rsi - center) / scale).alias(alias or f"rsi{period}_sc")

    @staticmethod
    def rsi_pair(short_period: int = 14, long_period: int = 55, center: float = 50.0, scale: float = 50.0,
                 alias_prefix: str = "") -> Tuple[pl.Expr, pl.Expr, pl.Expr]:
        """长短周期 RSI 配对（用户要求：长周期 >=50 与短周期结合）。

        返回 (rsi_short_scaled, rsi_long_scaled, rsi_div=short-long)。
        ``rsi_div`` 捕捉动量在短/长窗口上的背离。
        """
        s = PolarsExp.rsi_scaled("close", short_period, center, scale)
        l = PolarsExp.rsi_scaled("close", long_period, center, scale)
        div = (s - l).alias(f"{alias_prefix}rsi_div_{short_period}_{long_period}")
        return (
            s.alias(f"{alias_prefix}rsi{short_period}_sc"),
            l.alias(f"{alias_prefix}rsi{long_period}_sc"),
            div,
        )
