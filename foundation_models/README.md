# 时序基础模型 / RL 框架调研：能否用于本项目

> 调研日期 2026-11-17。目标：判断 Kronos / KiT / FinCast / Tycheon / TimesFM / Chronos / Moirai / Lag-Llama / FinRL
> 哪些能接入当前项目，是否可能带来新发现。所有模型信息来自官方 GitHub / arXiv / HF（联网核对），非猜测。

## 0. 先把"我们的项目"说清楚（判断一切的基准）

| 维度 | 现状（`实验总结.md` / `特征规格`） |
|---|---|
| 任务 | **事件合约二元涨跌预测**：目标 `close[t+h] > close[t]`，h=3/5/10/15/30；赔率 0.8 → 盈亏平衡胜率 55.56% |
| 现役主力 | h=10 K 线 LGBM 5 seed，63.32% @7.4 单/天，AUC 0.5366；h=3 盘口 62.85% @40.6 单/天 |
| 数据 | BTCUSDT 1m K 线 2020-2026（361 万根，vision-base）；订单簿 185 列 2025-07~2026-10（68.7 万分钟）；ETH 部分 |
| 特征 | 49 列 OHLCV 衍生（K 线）/ 185 列盘口；已榨干，struct/liq/VP/形态/亚分钟全部证伪 |
| 算力 | Kaggle GPU 仅神经网络/时序项目可开；torch 不进 requirements（镜像自带 CPU 版） |
| 判据 | 配对 bootstrap CI 不跨 0；6 组以上周期×年份同向；MDE 先行；AUC 不是优化目标，尾部胜率/EV 才是 |
| 硬约束 | 无前视（信号 t 确认、目标 t+h）；test 只评估不在上面调参；月级 walk-forward |

**关键矛盾**：这些模型几乎都做**数值/路径预测**（point/ probabilistic forecasting），而我们要的是**二元方向 + 尾部胜率**。
"预测得准"（MSE / RankIC 好）不等于"押方向胜率高"，更不等于"能过 55.56% 的线"。历史上 CNN/LGB AUC 0.53 附近、
时序模型（TCN）已被证伪，这是评估这些模型时必须先过的心理关。

## 1. 九个模型逐项评估

### A. 金融 K 线专用基础模型（与我们的数据形态最匹配）

#### Kronos — `shiyu-coder/Kronos`，arXiv 2508.02739，AAAI 2026，MIT ✅ 最值得试

- 定位：**首个开源金融 K 线基础模型**，decoder-only；把 OHLCV 经 BSQ 量化成层次离散 token；45+ 交易所数据预训练。
- 规格：`mini` 4.1M / `small` 24.7M (ctx 512) / `base` 102.3M (ctx 512) / `large` 499M（**未开源**）。
- 输出：未来 OHLCV **路径**（可 `sample_count` 采样多条），非方向标签。
- 论文口径：零样本价格预测 RankIC +93%（相对基线）、波动率 MAE −9%；live demo 是 BTC/USDT 未来 24h。
- 有完整微调脚本（tokenizer + predictor + 回测），Qlib 日频示例；预测器 `predict_batch` 支持 GPU。
- **适配性**：输入就是我们的 1m OHLC（volume 可选），上下文 512 根 ≈ 8.5 小时，足够覆盖 h=3~30。
  用法有两条：① 零样本直接出未来 K 线 → 取 close 变化方向做二元目标；② 用它的 hidden state / 预测路径统计量作为 LGBM 的额外特征（需极小心无前视）。
- **风险**：论文指标是 RankIC（横截面排序），我们是单时序二分类；路径预测在 h=3 这种 3 分钟尺度上大概率近随机游走；
  预训练含大量股票/日频数据，加密 1m 的分布是否覆盖未知。
- **判定：优先级最高**。建议先零样本小样本 smoke（几百个 anchor），看 h=3/10 原始方向胜率，再决定是否微调。

#### KiT — `Luciferbobo/KiT`，arXiv 2609.34507（2026-09），AGPL-3.0 ⚠️ 可试但license/成熟度存疑

- 定位：K 线 **diffusion transformer**，flow matching 生成未来 OHLCV 轨迹；输入为 5 维 log-ratio `(gap, body, up, dn, vol)`；
  在数十亿 bar、多市场多尺度预训练。
- 规模：唯一开源档 KiT-L ~284M；窗口表已定义 1m: history 1200 / horizon 120、5m: 960/96、30m: 640/40（**horizon 比我们的 h 长得多**）。
- 论文：3 市场 × 7 分辨率，收益 RankIC 0.057、波动 RankIC 0.66；代码权重已放出（HF `Lucifer744/KiT_model`），但**训练代码 TBD**，
  回测示例是 A 股 Top-N 组合（T+1、印花税），不是我们的场景。
- **判定：第二梯队**。需 GPU 6GB+；AGPL-3.0 对闭源使用有传染性（我们项目公开 MIT/研究用无碍，但要知道）。
  horizon 太长是其与二元短 h 的结构性错配——只能拿它的中间表示当特征，不能直接当 predictor。

#### FinCast — `vincent05r/FinCast-fts`，CIKM 2025，arXiv 2508.19609，Apache-2.0 ⚠️ 通用金融 TS，不一定 K 线

- decoder-only transformer，20B+ 金融时间点预训练，跨领域多分辨率；输出是数值预测。
- 有推理代码与 HF 权重（`Vincent05R/FinCast`），**有微调代码**。
- **判定：备选**。先确认它是否吃 OHLCV 多变量、上下文长度是否支持 1m 短周期；优先级低于 Kronos/KiT。

### B. 通用时序基础模型（非金融专用，作为对照组）

#### TimesFM (Google, v2.5, 200M, ctx 16K, Apache-2.0)

- 零样本 point forecasting，400B 时间点预训练；context 16K 很长（≈11 天 1m），可直接喂全天上下文。
- **判定：值得做对照组**。它不代表金融专用，但"通用 SOTA 能否打过我们的 LGBM"本身就是有价值的负/正结论。
  注意其输出是连续值，需自行二值化；batch composition 有坑（官方说明）。

#### Chronos / Chronos-2 (Amazon, T5-based, Apache-2.0)

- 把时序 token 化后送 T5，概率预测（采样路径）；`chronos-t5-mini` 轻量；新版 Chronos-2 2025-10 发布。
- **判定：对照组**。与 Tycheon 天然兼容（Tycheon 官方示例就是 Chronos-2）。概率输出可自然算方向概率。

#### Moirai (Salesforce, Uni2TS, 2024, CC-BY-NC 4.0 for 2.0)

- 支持多变量/多频率的统一预测 transformer，multi-patch 掩码。
- **判定：低优先级**。非金融、许可证 NC（商用受限）；除非做 benchmark 对照，否则不投入。

#### Lag-Llama (2023, 首个开源 TS FM)

- 单变量概率预测，基于 lag 特征 + Llama；年代较早，规模小。
- **判定：最低优先级**。仅作为 "TS FM 基线" 文献对照，无实际接入价值。

### C. 非预测模型（辅助层 / 框架）

#### Tycheon — `anilatambharii/tycheon`，Apache-2.0，Alpha 0.6.0 ✅ 方法论高度契合，但只是工具层

- **不是预测模型**，是套在 forecaster 之上的"可信度 / 风险"层：conformal 校准区间、VaR/ES/回撤概率、
  **walk-forward 防泄漏评估**、**强制与 random-walk 基线对比**、Diebold-Mariano 检验；有 REST API / MCP server。
- 明确说明"合成数据上 Kronos 等都无法与随机游走区分"——与我们 SCMP 回测的观察一致。
- **判定：立即有用**。它的评估纪律（holdout 校准、随机基线对照、DM 检验）正是我们项目判据的现成实现，
  可借鉴接入评估脚本；但不要指望它产生 alpha。建议作为评估工具而非信号源引入。

#### FinRL (AI4Finance, 多版本持续更新)

- 深度强化学习交易框架（股票/加密），最新 FinRL-X（PAKDD 2026 workshop）走模块化 infra。
- **判定：暂不接入**。RL 交易需要连续动作空间/仓位管理，与我们的**事件合约二元单次下注**目标不同构；
  且项目历史对"模型复杂度"已证伪（24 组 AUC 极差 0.002）。除非未来做组合/仓位优化，否则不动。

## 2. 优先级排序与理由

| 优先级 | 模型 | 动作 |
|---|---|---|
| 1 | **Kronos** | 零样本 smoke（h=3/10 方向胜率）→ 若有信号再谈微调；最匹配数据形态、MIT、有微调链路 |
| 2 | **Tycheon** | 只借鉴评估层：walk-forward + random-walk 基线 + 校准区间；不产生信号 |
| 3 | **TimesFM / Chronos** | 通用 TS 对照组，回答"专用 vs 通用"是否有差异 |
| 4 | KiT / FinCast | 有 GPU 预算、且 Kronos 路径走通后再试；KiT 的 horizon 与 license 需注意 |
| 5 | Moirai / Lag-Llama | 文献对照，不投入工程 |
| — | FinRL | 与二元事件合约目标不同构，暂不考虑 |

## 3. 统一的验证协议（做任何模型都必须先过）

1. **零样本优先**：先不微调，用预训练权重在 2020-2026 BTCUSDT 1m 上按 anchor 采样（如每隔 60 根一个点，≥1 万点）。
2. **目标对齐**：模型输出（路径/概率）→ 二值化 `sign(Σ predicted log-return)` 或 `P(close[t+h]>close[t])`，与 two_legs 的 target 严格同定义。
3. **无前视**：模型输入只到 t 的 close；预测 t+1..t+h；与项目 `add_target` 同口径。
4. **判据**：val/test 严格按项目时间切分；主看 test 胜率 vs 55.56% 与按天 EV；配对 bootstrap；至少 6 组周期×年份同向；
   同时报 random-walk 基线（Tycheon 强调的纪律）。
5. **MDE 先行**：h=10、tpd=8 的 SE≈3.2%，动作前先算能检出的最小提升，低于 MDE 不开 kernel。
6. **算力**：推理可走 GPU（`kaggle_launcher.py` PROJECTS 里 `gpu=True`）；torch 不入 requirements，用 Kaggle 镜像自带。

## 4. 预期与红线

- **先验期望偏低**：历史证伪链很长（K 线特征、CNN/TCN、形态、亚分钟、SMC 三种规则全负；AUC 长期卡 0.53-0.54）。
  基础模型若只是"把 LGBM 换成 transformer"，大概率复制同样的天花板。**真正可能有新意的是**：
  （a）预训练带来的跨市场/跨周期先验能否在**低波动时段**（现役模型无方向信息的段）提供额外信息；
  （b）概率/路径输出能否改善**尾部校准**（当前 topK 胜率与阈值失准是已知痛点，见 `实验总结.md` §8.2）。
- **红线**：不得在 test 上选模型/阈值；不得用全期 topK（前视口径，禁验收）；不得把"RankIC 高"直接宣称"能交易"。

## 5. 魔搭 ModelScope 执行注意事项

魔搭在国内，`git clone github.com` 和 `huggingface.co` 直连经常卡住，必须用镜像：

```bash
# 1. 克隆仓库：用 ghproxy 镜像（直连会卡在 "正克隆到 'bn_bt'..."）
cd /mnt/workspace
git clone https://ghfast.top/https://github.com/bailiang0369/bn_bt.git
# 若仍失败，改用 zip 下载：
# wget https://ghfast.top/https://github.com/bailiang0369/bn_bt/archive/refs/heads/main.zip -O bn_bt.zip
# unzip bn_bt.zip && mv bn_bt-main bn_bt

# 2. 一键运行（已自动设 HF_ENDPOINT=hf-mirror.com、数据下载带 ghproxy 镜像、不碰预装 torch）
cd bn_bt
python -m foundation_models.run_modelscope --model timesfm --anchors 3000 --bar 1
```

- 魔搭 Notebook 入口：modelscope.cn → 我的 Notebook → 实例"运行中" → 点"查看 Notebook"进 JupyterLab。
- 免费 A10 实例：空闲 1h 自动关、单次 8h 强制停；同时只能跑 1 个实例，启动新实例一直卡住通常是旧实例还占着。
- `run_modelscope.py` 做的处理：设 `HF_ENDPOINT=https://hf-mirror.com`、只装轻量依赖
  （`pip install timesfm==2.0.2` 不带 `[torch]` extra，保留魔搭预装 torch/cuda）、
  数据从 GitHub Release 下载失败自动切 ghfast.top 镜像、结果写 `/mnt/workspace/bn_bt/artifacts/`。

## 6. 参考链接

- Kronos: https://github.com/shiyu-coder/Kronos | arXiv 2508.02739 | HF `NeoQuasar/Kronos-small`
- KiT: https://github.com/Luciferbobo/KiT | arXiv 2609.34507 | HF `Lucifer744/KiT_model`
- FinCast: https://github.com/vincent05r/FinCast-fts | arXiv 2508.19609 | HF `Vincent05R/FinCast`
- Tycheon: https://github.com/anilatambharii/tycheon | https://ambharii.com/tycheon | Zenodo 10.5281/zenodo.24041298
- TimesFM: https://github.com/google-research/timesfm
- Chronos: https://github.com/amazon-science/chronos-forecasting
- Moirai / Uni2TS: https://github.com/SaleforceAI-Lab/Uni2TS
- Lag-Llama: https://github.com/time-series-foundation-models/lag-llama
- FinRL: https://github.com/AI4Finance-Foundation/FinRL
