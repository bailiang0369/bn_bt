# 订单簿 × vision 合并特征规格说明（feature_spec）

> 本文件是 `features/` 模块的特征口径与列清单。产物体积小（vision ~1.7MB/月、
> 合并表 ~40MB/月），下游可直接按 Release 拉取，不重跑管线。
>
> **边界**：本仓库只产出原始底座 + 订单簿衍生特征（`ob_*`），合并成
> `BTCUSDT.parquet`（`merged-features` Release）。OHLC / 主动买卖量的衍生特征工程
> 依赖下游预测目标、千变万化，**不在本仓库做**，下沉到下游项目；aggTrades
> 微观结构特征（`um_agg_*` 等）经评估增益小、成本高，亦不纳入（见 DATA_ACCESS §3.1）。

---

## 0. 先说清楚：现货 vs 合约「不合并」

订单簿特征对 **futures 和 spot 是分别独立计算的**，各自产出完整的一套 L1+L2 特征，
**绝不做加权合并**。原因是两侧的档位粒度差一个量级（futures tick 0.1、spot tick 0.01，
同价位带内 spot 档数是 futures 的 ~10 倍），任何"按权重合并"都会把两个不同的盘口结构
糊在一起，丢失"谁先动、谁更深"的信息。

两侧通过**列前缀**区分：

| 前缀 | 含义 |
|---|---|
| `ob_f_*` | futures（U 本位合约）盘口特征 |
| `ob_s_*` | spot 盘口特征 |
| `ob_*`（无后缀） | 跨市场特征（见 §3） |

跨市场只保留少数几个**比率/差值**（基差、深度比、价差比、领先滞后），见 §3。

---

## 1. 数据流与主键

```
data-merged 月度 band+mid ──> ob_features.py ──> L1 截面 (ob_*)
                                                    │ 全窗口拼接 -> reindex -> 滚动
vision-base 月度宽表 ────────────────────────> build_dataset.py ──> merged_BTCUSDT_{m}.parquet
                                                    │ 振幅中位数 + 振幅倍数不平衡
                                                    └ 校验门禁
```

- **主键 `close_time`**：Int64 毫秒 UTC，与 vision-base 完全同键。
  订单簿原始 `bucket`（分钟起点）在 `ob_features.py` 内转成 `close_time = bucket + 60000`。
- **对齐语义**：`bucket=t` 是"第 t 分钟内最后一条消息后的盘口" = 该分钟收盘时刻的状态，
  所以 `close_time` 与 vision 的 K 线收盘时刻是同一瞬间。用它预测 **t+1 及以后** 无前视。
- 全窗口 2025-07 ~ 2026-10，分钟粒度。

---

## 2. L1 截面特征（每个交易所一份，`ob_f_`/`ob_s_`）

所有距离统一为**相对 mid 的 bps**（`d = |px−mid|/mid × 1e4`），所有量是 **BTC 枚数**。
深度按固定分箱 `b0..b4` = `[0,2) [2,5) [5,10) [10,20) [20,30)` bps。

### 2.1 基础

| 列 | 公式 | 含义 |
|---|---|---|
| `ob_mid` | `(best_bid+best_ask)/2` | 中间价 |
| `ob_spread_bps` | `(ask−bid)/mid ×1e4` | 相对价差 |

### 2.2 分箱（`{s}`=bid/ask，`{b}`=b0..b4）

| 列 | 含义 |
|---|---|
| `ob_q_{s}_{b}` | 该箱累计挂单量（BTC） |
| `ob_n_{s}_{b}` | 该箱档位数 |
| `ob_vwap_{s}_{b}_bps` | 该箱量加权均价相对 mid 的 bps |

### 2.3 侧总量（`{s}`=bid/ask）

| 列 | 含义 |
|---|---|
| `ob_q_{s}` | 带内（±30bps）总挂单量 |
| `ob_n_{s}` | 带内总档位数 |
| `ob_qmax_{s}` | 带内单档最大挂单量 |
| `ob_qmax_dist_bps_{s}` | 该最大单档距 mid 的 bps（"大墙"位置） |

### 2.4 不平衡（`Qb`/`Qa` 为 bid/ask 深度）

| 列 | 公式 | 含义 |
|---|---|---|
| `ob_obi_total` | `(Qb−Qa)/(Qb+Qa)` | 全带不平衡 |
| `ob_obi_b0` | 0-2bps 不平衡 | 最近端 |
| `ob_obi_near` | 0-5bps 不平衡 | 近端 |
| `ob_obi_far` | 10-30bps 不平衡 | 远端 |
| `ob_obi_tilt` | `obi_far − obi_near` | 盘口形态倾斜（正=远端更偏买） |

### 2.5 价格

| 列 | 公式 | 含义 |
|---|---|---|
| `ob_wmid_bps` | `((VWAP_a·Qb+VWAP_b·Qa)/(Qa+Qb) − mid)/mid ×1e4` | 全带加权中价偏离 |

### 2.6 冲击 / 形态

| 列 | 公式 | 含义 |
|---|---|---|
| `ob_impact_bid_bps_per_btc` | `10 / Q_bid(0-10bps)` | 吃掉 10bps 需要多少量，反比冲击 |
| `ob_impact_ask_bps_per_btc` | `10 / Q_ask(0-10bps)` | 同上（卖侧） |
| `ob_impact_imb` | `(imp_bid−imp_ask)/(imp_bid+imp_ask)` | 冲击不对称 |
| `ob_depth_conc_{s}` | `Q_{s}(0-5bps)/Q_{s}` | 近端深度占比 |
| `ob_levels_ratio` | `n_bid/n_ask` | 档数比 |
| `ob_qmean_{s}` | `Q_{s}/n_{s}` | 平均单档量 |
| `ob_hhi_{s}` | `Σq²/(Σq)²` | 挂单集中度（越大越集中） |
| `ob_qmax_ratio_{s}` | `qmax/qmean` | 最大单档相对均值 |
| `ob_qcv_{s}` | `std(q)/mean(q)` | 挂单量变异系数 |

---

## 3. L2 时序特征（滚动在**全窗口拼接**上做，跨月不截断）

| 列 | 公式 | 含义 |
|---|---|---|
| `ob_missing` | — | 该分钟订单簿缺失 = 1（见 §5） |
| `ob_ret_1m_bps` | `mid.pct_change() ×1e4` | 中间价 1 分钟收益 |
| `ob_dobi_1m` / `_5m` | `obi_total − shift(n)` | Δ不平衡（OFI 近似） |
| `ob_ddepth_bid_1m` / `_ask_1m` / `_bid_5m` | `Q_{s} − shift(n)` | Δ深度（挂单补充/消耗） |
| `ob_dwmid_1m` | Δ wmid_bps | 全带加权中价位移 |
| `ob_dspread_1m` | Δ spread | 价差变化 |
| `ob_rv_5m_bps` / `_15m_bps` | 1m 收益的滚动 std | 订单簿实现波动 |
| `ob_obi_mean_5m/15m`、`ob_obi_std_5m/15m` | 滚动均值/标准差 | 不平衡水平与波动 |
| `ob_spread_mean_5m/15m`、`ob_spread_std_5m/15m` | 同上 | 价差水平与波动 |
| `ob_impact_mean_5m/15m` | `(imp_bid+imp_ask)/2` 滚动均值 | 冲击水平 |
| `ob_obi_z_5m/15m`、`ob_spread_z_15m` | `(x−mean)/std` | 标准化 |

> 窗口只取 **{5m, 15m}**，对齐预测周期 3/5/10/15 分钟。不用 60m 这类长窗——
> 订单簿挂单随时可撤/新增，只对短期有效，长周期趋势该用 K 线/资金面特征。
> 窗口用 `rolling(index_column=ts, period=...)`，按**日历时间**滑动；`shift(n)` 在
> reindex 之后恰好等于 n 分钟前。

---

## 4. 跨市场特征（futures × spot）

| 列 | 公式 | 含义 |
|---|---|---|
| `ob_basis_bps` | `(ob_f_mid − ob_s_mid)/ob_s_mid ×1e4` | 同刻基差（比 kline `basis` 更干净） |
| `ob_depth_ratio_f_s` | `(f_bid+f_ask)/(s_bid+s_ask)` | 合约/现货深度比 |
| `ob_spread_ratio_f_s` | `f_spread/s_spread` | 价差比 |
| `ob_lead_f_s_bps` | `ob_f_ret_1m − ob_s_ret_1m` | 领先滞后（谁先动） |

---

## 5. 振幅倍数不平衡（vision 振幅 × 订单簿不平衡）

振幅定义为**近 5 分钟这一整段的区间**（high−low），再对其做 30m 滚动均值，得到
"近期典型波动幅度"；再看该幅度的不同倍数处的买卖不平衡，且**确保在快照带
（±0.3% = 30bps）以内**。

| 列 | 公式 | 含义 |
|---|---|---|
| `amp_5m_bps` | `(近5分钟 max(high)−min(low)) / close ×1e4` | 5 分钟区间振幅 |
| `amp_mean_30m_bps` | `amp_5m_bps` 的 30m 滚动均值 | 振幅的近期水平（用于缩放） |
| `ob_amp_obi_015/035/080` | 距离 `0.15/0.35/0.8 × amp_mean_30m` 内的 `(Qb−Qa)/(Qb+Qa)` | 振幅倍数的涨跌不平衡（futures 盘口，近端） |
| `ob_amp_clip_015/035/080` | `k×amp_mean_30m > 30bps` ? 1 : 0 | 该倍数超出快照带、已被截断到 30bps |

实现要点：
- 深度在距离 `x` 处的累计量，用 L1 固定分箱做**线性插值**（箱内假设均匀）。
- `thr = min(k×amp_mean_30m, 30)`；`amp_mean_30m` 为空或两侧深度都为 0 时不平衡置 null。
- 倍数取小数（0.15/0.35/0.8）：3/5 分钟涨跌只看近端订单簿。`0.8×` 实测 ~10bps，
  基本都在 30bps 带内，`ob_amp_clip_*` 仅在振幅极端时才会标 1。

---

## 6. 攻防特征（主动成交 × 当前挂单）

"攻"= 近 1 分钟主动成交（flow，`um_buy_vol/um_sell_vol`、`spot_buy_vol/spot_sell_vol`），
"守"= 当前挂单快照（stock，`ob_*_q_bid/q_ask`）。两者同 `close_time` 对齐、单位都是 BTC。
逻辑（**弱参考**，假设主动买卖力量持续）：攻方 ≫ 守方 → 可能突破、出大振幅；攻方 ≪ 守方 →
突破无望。现货/合约结论常不一致，**两侧都算、下游综合看**。

| 列 | 公式 | 含义 |
|---|---|---|
| `atk_f_imb` / `atk_s_imb` | `(主动买−主动卖)/(主动买+主动卖)` | 攻方方向不平衡 |
| `atk_f_def_gap` / `atk_s_def_gap` | `atk_imb − obi` | 攻守背离（>0：挂单偏卖但成交偏买） |
| `atk_f_absorb_buy` / `atk_s_absorb_buy` | `主动买量 / 卖盘总挂单` | 买方攻击相对卖方防守（能否破墙） |
| `atk_f_absorb_sell` / `atk_s_absorb_sell` | `主动卖量 / 买盘总挂单` | 卖方攻击相对买方防守 |

> `atk_f_*` 用合约（um 成交 vs futures 盘口），`atk_s_*` 用现货（spot 成交 vs spot 盘口）。
> 时序是当前快照 + 刚结束这一分钟的成交，不用历史快照。

---

## 7. 缺失值与缺分钟策略

- 缺分钟（源端断流 / 单边有量）**在特征层不补**：`build_dataset` reindex 到 vision 的
  完整分钟网格，缺失分钟的 `ob_*` 全 null，并打 `ob_missing=1`。
- 滚动窗口按时间滑过缺口（缺口分钟不参与 agg），所以滚动特征在缺口后仍有效。
- 单边分钟（只有 bid 或只有 ask）同样产出 null + `ob_missing=1`。
- **统一去 null 放在建模前**（下游职责）：训练时 drop 掉含 `ob_missing=1` 或关键特征
  为 null 的行，或按列策略填充。特征层不做 imputation，避免静默造数。

---

## 8. 前视 / 因果约定

- 所有特征只用 `t` 及之前的信息，无前视。
- `ob_*` 在 `close_time=t` 的语义是"第 t 根 bar 收盘瞬间的盘口"。用它预测
  **t+1 及以后**的目标是因果的；**不要**用它解释同一根 bar 的收益率（同一时刻，内生）。
- 分箱 vwap/深度是 `±0.3%` 截断口径，不是全盘；跨市场比较只认 bps 分箱 + 相对量。

---

## 9. 消费注意事项

- 列类型：`close_time` 为 i64，`ob_mid`（合并后 `ob_f_mid`/`ob_s_mid`）为 f64，
  其余均为 f32。所有特征已做非有限值清洗（inf/NaN → null）。
- 月份文件按 `close_time` 月归属切分，边界与 vision 一致；逐月拼接无缺口。
- `metrics` 源列（`sum_open_interest` 起 6 列）是 5 分钟粒度前向填充，勿当分钟独立信号。
- 两端 `ob_f_mid` / `ob_s_mid` 是同一中间价的两次记录，数值近乎相等（差 = ob_basis_bps）。
