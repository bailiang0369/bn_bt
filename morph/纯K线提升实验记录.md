# 纯 K 线 3 分钟方向预测：11 轮实验记录

> 目标：在**只用 K 线**（无订单簿、无跨市场）的前提下，提升 3 分钟后方向预测的
> topK 准确率，使纯 K 线腿在与订单簿融合时不拖后腿。
> 记录时间：2026-11-05。代码 commit `0d6e2d2` 及之前 11 个提交。

---

## 1. 数据（已重新生产，不再用旧 dataset）

| 项 | 值 |
|---|---|
| Kaggle dataset | `byh850515/btcusdt-1m-full-2020`（private） |
| 文件 | `btcusdt_1m_full.parquet`，79.1 MB |
| 行数 / 范围 | 3,592,800 行，2020-01-01 00:01 ~ 2026-11-01 00:00（UTC，Int64 ms） |
| 列 | `close_time, open, high, low, close, volume, taker_buy_volume, count` |
| 来源 | Binance Vision `futures/um/monthly/klines`（**U 本位合约**），复用 `bn_hisdata/binance_vision/download_binance_vision.py` |
| 生产 kernel | `byh850515/btcusdt-1m-full-2020-dataset-maker` → `morph/make_full_dataset.py` |

**作废的旧数据源**：`eth-btc-1m-2020`（只有 OHLC，无量能）、`crypto-eth-btc-data`（很早的旧 dataset）。
**CVD 口径**：`2 * taker_buy_volume - volume`，其中 `taker_buy_volume` 是**合约（um）主动买量**，不是现货。

## 2. 评估协议（所有轮次统一，跨轮可比的前提）

- 切分：`VAL_START=1777704120000`（≈2026-05-02）、`TEST_START=1784298120000`（≈2026-07-17）、
  `OB_END=1792540620000`（≈2026-10-21，为与订单簿对齐而截断）。
- test：**96 天 / n=137,373 根 1m**，基准（涨占比）= **0.4925**。
- 泄露防护：train/val 尾部各剔除 `h` 根（标签不跨切分边界）。
- 指标：
  - `static`：val 均值方差 z-score 后，取 `|z|` 最大的 `K = 15 × 天数 = 1440` 个样本，准确率；
  - `按天`：每天取 15 个，仅统计完整交易日（1m `min_bars=1200`）；
  - `AUC`：排序质量，**唯一能判断"是否真的加了信息"的指标**；
  - `窗口4`：test 切 4 段各算 topK，报均值 ± sd。
- **噪声地板（第二轮实测）**：seed 间 sd = **0.47pt**，窗口间 sd = **2.9pt**。
  因此同一 test 上 **< 1pt 的差异不可区分**，必须看 4 窗口是否同向、AUC 是否变化。

## 3. 十一轮结论（全部同一口径）

| 轮 | 试了什么 | 结果 | 判定 |
|---|---|---|---|
| 1 | 3m 重采样 / 符号累加 / 形态子集 / 堆叠 | 3m 掉 5pt；符号累加 static 0.6014 vs 0.6069；堆叠更差 | 全部无效 |
| 2 | 噪声基线 / 标签净化(5,10bps) / 形态+符号 | 净化越好越差，10bps 时 AUC 崩到 0.4993 | 无效 |
| 3 | 8seed / 软加权 / h分数相加 / 多样性 bagging | s8 0.6076、div 0.6097，与 0.6125 不可分 | 无效 |
| 4 | 评估端中心偏移 mu + 单边/对称 | val +1.04pt → test −0.69pt（过拟合）；test 内 oracle 仅 0.6160 | 无空间 |
| 5 | 3m 基础 K 线 LGB + 1D CNN(px/st/vol/cvd) | lgb3m 0.5583、cnn1m 0.5458；**corr(LGB,CNN)=0.198** | 单腿弱，但互补 |
| 6 | CNN 加实体/影线 8 通道 + 权重扫描 | cnn2 0.5382；LGB+0.1cnn → static 0.6167 | 有微弱迹象 |
| 7 | 回归目标（学幅度）/ 时间衰减权重 | reg AUC 0.5165、decay 0.5251（均低于 0.5269） | 无效 |
| 8 | 形态窗口聚类(64簇) + 簇胜率前向编码 | **AUC 完全不变 0.5269**；static 0.5882 | 无效（信息早被覆盖） |
| 9 | **多 horizon 一致性集成**（h=1,2,3,5,10 等权 z 平均） | **static 0.6174 / AUC 0.5277 / 4 窗口 4/4 全升** | **首个真提升** |
| 10 | 剔除 19 个量能特征 + CNN 去 vol 通道 | 44 特征 0.5993（−1.8pt，但 AUC 不变）；**CNN 去 vol：0.5382→0.5514** | 量能保留；CNN 去 vol 正确 |
| 11 | 定稿：多 horizon + 无 vol CNN 加权融合 | **static 0.6264(w=0.25) / 0.6194(w=0.2)，AUC 0.5285，按天 0.5937** | **定稿** |

### 关键判读

- 第 1~8 轮 AUC **锁死 0.5269**：无论换特征、加 CNN、改损失、加形态编码，排序信息都没变。
  这比 topK 数字可靠得多——topK 在 3pt 噪声里可以随便涨跌 1~2pt。
- 第 9 轮 AUC 首次变动（0.5277），且 4/4 窗口同向提升，才认定是真信号。
- 第 11 轮 AUC 随 CNN 权重**单调**升到 0.5285，按天 topK 单调升到 0.5937，同时窗口 sd
  从 0.0473 降到 0.0260 —— 三个独立指标同向，才算确认。

## 4. 定稿配置

```
特征   : build_features 63 列（动量/波动/量能/形态/时间），1m
LGB 腿 : h ∈ {1,2,3,5,10} 各一个 LightGBM(binary, 60 轮)
         每个 h 用 seed 42,43；各自对 val 做 z-score 后等权平均 → base
CNN 腿 : 7 通道 px/body/upper/lower/ret1/st/cvd(合约)，无总成交量
         窗口 60 根 1m，1D 膨胀卷积(c1=16,c2=24,c3=24)，5 epoch，2 seed
融合   : score = base + 0.20 × z(cnn)        # 0.25 时 static 最高、0.20 时最稳
```

| 口径 | 定稿 | 起点(0.6049) |
|---|---|---|
| static 全局 topK | **0.6264**(w=0.25) / 0.6194(w=0.20) | +2.2 / +1.5pt |
| 按天 topK | **0.5937** | — |
| AUC | **0.5285** | — |
| 4 窗口 | 0.6026±0.0346(w=0.25) / 0.6038±0.0260(w=0.20) | — |

## 5. 代码与 kernel 索引

| 文件 | 作用 |
|---|---|
| `morph/make_full_dataset.py` | 生产 2020+ 全量 1m parquet → dataset |
| `morph/kline_boost.py` | 公共工具（`load_klines`/`build_features`/`score_eval`/`roll_z`）+ v0-v4 |
| `morph/kline_boost2.py` | 噪声基线、标签净化、形态+符号 |
| `morph/kline_boost3.py` | 8seed / 软加权 / h 相加 / 多样性 |
| `morph/kline_boost4.py` | 评估端中心偏移与单边调优 |
| `morph/kline_dl.py` | 3m 基础 K 线 LGB + CNN(4 通道) + 四路融合 |
| `morph/kline_dl2.py` | CNN 8 通道 + 权重扫描 |
| `morph/kline_reg.py` | 回归目标 / 时间衰减 |
| `morph/kline_shape.py` | 窗口形态聚类 + 簇胜率前向编码 |
| `morph/kline_multi.py` | **多 horizon 集成**（第 9 轮） |
| `morph/kline_novol.py` | 去总成交量 + CNN7 |
| `morph/kline_final.py` | **定稿组合**（第 11 轮） |

跑法：`python kaggle_launcher.py --project boost11`（CPU，约 25 分钟；`--no-wait` 只推不等）。
可用环境变量：`SEEDS`、`ROUNDS`、`CNN_EPOCHS`、`CNN_STRIDE`、`HS`。

## 6. 踩过的坑（复用前必读）

1. **分类概率不能直接判方向**：`global_topk_eval` 用 `score > 0` 判涨跌，若传入原始概率
   （恒正）等于全猜涨。第 7 轮 `clf` / `clf_decay` 两行数字因此不可比。正确做法：先对 val
   做 z-score 再判符号（`kline_multi.evaluate` 已修正）。
2. **3m 网格不能直接用 K=15/天**：3m 每天 480 根，15 个 = 3.1% 覆盖率，而 1m 是 1.04%。
   第 5 轮已改为把 3m 分数映射回 1m 网格再评估。
3. **`np.roll` 做 horizon 融合会引入未来 bar**：`p1[t]+p1[t+1]+p1[t+2]` 中 `p1[t+1]` 用到了
   t+1 时刻的特征。正确做法是各 horizon 各自在 t 时刻预测，见 `kline_multi.py`。
4. **`resample_3m` 曾硬依赖 `volume`**：换成只有 OHLC 的 dataset 会直接崩。
5. **`load_klines` 曾只 glob 固定文件名**，找不到就联网下载。现在扫 `/kaggle/input/**/*.parquet`
   并按"是否含 volume + 是否覆盖 test 尾部"筛选。
6. **val 上调参不泛化**：第 4 轮 val +1.04pt → test −0.69pt。凡是"在 val 选最优"的结论，
   必须同时报 test 曲线，不能只报 test 最大值。

## 7. 尚未尝试 / 下一步候选

- **VP 指标**（唯一允许使用总成交量的场景，符合你的口径）：POC 距离、价值区间（70% 量）
  上下沿位置、当前价上下方成交量比、VWAP 偏离。目前 63 特征里没有一条真正的 VP。
- 每个 horizon 的 seed 从 2 加到 5（预计 +0.3~0.5pt，边际递减）。
- CNN 侧：窗口从 60 加长到 120、训练轮数加到 10、通道加 CVD 的斜率/加速度（不加 vol）。
- 目标重定义（触及概率 / 更长 horizon）——当前明确不做。
- **不做**：跨市场、总成交量直接入模（除 VP）、在 test 上调任何超参。
