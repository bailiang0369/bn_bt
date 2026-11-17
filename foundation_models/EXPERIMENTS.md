# 基础模型探索：全部试验记录（2026-11-17 ~ 11-18）

> 一句话结论：**基础模型（Kronos / TimesFM / Chronos）零样本预测方向，跨架构、跨粒度、跨读法全部判负（AUC 0.49~0.52）；但同样任务用 LGBM + 52 列特征在 1h K 线上做到 56.98% @tpd4、10/10 月正 EV——"用对方法"和"用基础模型"是两回事。**
>
> 所有结果已推送 GitHub、产物在 `artifacts/{kronos,timesfm,chronos}/`，Kaggle kernel 全部私有。

---

## 0. 背景与问题

用户提出：研究 Kronos / KiT / FinCast / Tycheon / TimesFM / Chronos / Moirai / Lag-Llama / FinRL 九种模型，看哪些能用于项目（事件合约二元涨跌预测，赔率 0.8 → 平衡线 55.56%）。

九种模型的调研结论见 `foundation_models/README.md`；本文件只记录**实测**。

统一口径（所有探针）：
- 数据：vision-base BTCUSDT futures(um) close，2020-2026（361 万根 1m）
- 目标：`close[t+h] > close[t]`（与 `two_legs.add_target` 完全相同）
- 赔率 0.8 → 盈亏平衡胜率 55.56%（部分长周期用 0.85 → 54.05%）
- cal 段定阈值、test 段固定评估（杜绝事后 topK 前视）
- GPU/CPU 均用 Kaggle kernel（`kk` 工具编排）

---

## 1. Kronos（金融 K 线专用，自回归，MIT）

脚本：`foundation_models/kronos_probe.py`、`kronos_large.py`

| 测试 | 配置 | 结果 | 判定 |
|---|---|---|---|
| 零样本 smoke | GPU，300 anchors，h=3/10 | h3 51.33% / h10 47.67%，AUC 0.504/0.470，连 always-up 都打不过 | FAIL |
| 长 horizon | 300 anchors，h=3/10/30/60/120 | 胜率 49~52%，AUC 0.48~0.53，**拉长 40 倍无改善** | FAIL |
| 大样本 1m | 3000 anchors，cal/test 分离 | AUC 0.5181，top10% 58.62% CI[45.9,71.3]，分年 67.9/43.8/57.1（符号翻转） | FAIL |
| 大样本 5m | 3000 anchors，BAR=5 | AUC 0.5039，top10% 46.15% CI[32.6,59.7] | FAIL |

**教训**：第一次 top10% 70%（30 样本）是小样本假象；大样本 + 严格 cal/test 后回落到 58% 且分年不稳定。"预测太短"的假设被 h=120 测试直接否定。

---

## 2. TimesFM 2.5（Google 通用，patch 非自回归，Apache-2.0）

脚本：`foundation_models/timesfm_probe.py`

| 测试 | 配置 | 结果 | 判定 |
|---|---|---|---|
| CPU 小样本 | 1000 anchors，h=3/10 | AUC 0.479/0.485；h10 top10% 69.23%（**n=13**，CI[44,94]） | FAIL |
| GPU 大样本 | 3000 anchors，h=3/10 | AUC 0.512/0.523；top10% 47.89%（n=71）/ 54.55%（n=88），CI 跨线 | FAIL |

**关键**：CPU 小样本那个 69.23% 是 13 个样本的噪声——GPU 放大到 88 样本后回落到 54.55%。TimesFM 2.5 实际 API 是 `TimesFM_2p5_200M_torch.from_pretrained` + `ForecastConfig(max_horizon=...)`，非文档里的 `TimesFm`。

**输入**：只有 log(close) 一条序列（单变量模型），没有量/盘口/任何特征。

---

## 3. Chronos-Bolt mini（Amazon 通用，量化 T5 概率，Apache-2.0）

脚本：`foundation_models/chronos_probe.py`、`chronos_pup.py`

| 测试 | 配置 | 结果 | 判定 |
|---|---|---|---|
| CPU 1000 | h=3/10 | AUC 0.527/0.527；h3 top10% 70.59%（n=17） | FAIL（小样本） |
| CPU 10000 | h=3/10 | AUC 0.488/0.499；top10% 48.91%（n=229）/ 52.82%（n=195） | FAIL |
| CPU 10000 h60 | h=60 | AUC 0.490；top10% 53.49%（n=215）CI[46.8,60.2] | FAIL |
| P(up) 分位数法 | 9 分位数算方向概率，校准曲线 | AUC 0.49；名义 90% 档实际只有 45~52% 上涨；阈值下单全 FAIL | FAIL |
| 1h bar | BAR=60，预测下一根，20000 anchors | AUC 0.498；top10% 53.52%（n=327）CI[48.1,58.9] | FAIL |

**P(up) 是最有方法论价值的一次尝试**：用概率模型输出的方式（数 9 个分位数里有几个在现价上方）而非只取中位数。结果校准曲线是平的甚至反的——模型给的"置信度"不对应真实概率。这证明问题不在"怎么读分布"，而在分布本身不携带 BTC 短周期方向信息。

**速度**：Chronos-Bolt CPU 10000 anchors 只要 90 秒预测（比 Kronos 自回归快几十倍），因此成了主力测试模型。

---

## 4. LGBM + 1h K 线预测下一根（唯一正面结果）

脚本：`morph/h10_signal.py`（已有 `BAR` 机制，本次加了 `MIN_TRAIN/MIN_TEST` 覆盖和 `ms()` 窗口钳制）
Kernel project：`kaggle_launcher.py` 的 `bar60sig`（slug `byh850515/btcusdt-60m-bar-next-signal`）

配置：`BAR=60 HORIZON=60`（1h K 线预测 1 小时后），52 列 b_* 原生特征 + flow，walk-forward 10 窗口（2026-01~10），5 seeds，因果阈值，无前视。

| 名义 tpd | 实际单/天 | n | 胜率 | EV@0.85 | 月 min | 破 50% 月 |
|---|---|---|---|---|---|---|
| 2 | 1.9 | 574 | 57.67% | +0.067 | 51.85% | 0/9 |
| **4（主操作点）** | 4.1 | 1232 | **56.98%** | **+0.054** | **52.81%** | **0/10** |
| 8 | 8.0 | 2426 | 55.19% | +0.021 | 52.00% | 0/10 |

- 逐月胜率 52.8%~61.4%，月 std 2.80pp，**10 个月全部正**
- Placebo（打乱标签）AUC 0.46~0.51、胜率 39~51% → **无泄漏**
- UTC 小时分布：18-23h 与 00h 胜率 64~68%，01h/06-08h/12h/16h 只有 39~47%（事后观察，未验证增量）

**定位**：正 EV 但**不如现役 h=10 主力**（63.32% @7.4 单/天，EV +0.14）。EV 只有主力的 1/3，且依赖 payout 0.85 的假设——若实际 0.8（平衡线 55.56%），56.98% 只剩 +1.4pp。

---

## 5. 方法论教训（最重要，以后别重蹈）

1. **小样本假象反复出现**：70%（n=13~30）→ 大样本全部回落到 50% 附近。Kronos、TimesFM、Chronos 三次重演。任何 n<100 的 topK 胜率都不能采信。
2. **AUC ≠ 胜率**：AUC 是排序能力，胜率是 `sign(score)` 的正确率。分数分布有偏时（如 Chronos top10% 档只有 36% 押涨），AUC 0.53 也能胜率 50%。
3. **topK 口径必须区分**：全局分位（前视，禁验收）vs 每日 topK vs cal 阈值 + test 固定。今晚所有最终结果都用后者。
4. **概率模型的分布要校准才有用**：P(up)=0.9 若实际只有 0.5 上涨，这个"置信度"毫无意义。Chronos 的校准曲线证明通用模型的分位数不对应真实方向概率。
5. **"预测太短"不是基础模型失败的原因**：Kronos 拉到 h=120、Chronos 拉到 h=60、换 5m/1h bar，全部 AUC ≈ 0.5。
6. **基础模型 vs 专用 LGBM 的差别是任务对齐**：基础模型零样本优化的是 MSE/RankIC（路径准不准），LGBM 监督训练直接优化二元方向。同一个 1h 任务，Chronos AUC 0.498 FAIL，LGBM 56.98% 正 EV。
7. **换 bar 粒度要 clamp min_samples**：`build_bar` 在 BAR=60 时 `k_of(30)=1`，`rolling_std(1, min_samples=3)` 会报错；已加 `ms()` 钳制。

---

## 6. Kaggle / 魔搭 工程踩坑（以后直接复用）

### Kaggle（kk 工具）
- **推之前先查所有 kernel 状态**（用户明确要求）：`kk status --slug X`；有 running/queued 的先判断能否停，别反复重推占 GPU 配额（吃过 "Maximum batch GPU session count of 2 reached" 和长时间 queued 的亏）。
- 用**新 slug** 推变体，不要覆盖正在跑的 kernel（重推会 cancel 旧的）。
- 用户反馈 wait timeout 设太长浪费时间 → 改短轮询（30~60s/次，360~900s 上限）。
- `kk status` 是准的；running 时 `logs` 可能空。

### 魔搭 ModelScope（用户自己跑）
- 入口：modelscope.cn → 我的 Notebook → 实例"运行中" → "查看 Notebook" 进 JupyterLab。
- 免费 A10：空闲 1h 自动关、单次 8h 强制停、同时只能 1 个实例（启动新实例一直转通常是旧的占着）。
- 国内网络：`git clone` 卡住 → 用 `https://ghfast.top/https://github.com/...`；HF 权重 → `HF_ENDPOINT=https://hf-mirror.com`；pip → `pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/`。
- 魔搭**不预装 polars**，TimesFM 脚本要 `pip install polars pandas requests`；torch 用预装版，`pip install timesfm==2.0.2`（不带 `[torch]` extra）+ `SKIP_TF_INSTALL=1` 避免覆盖 CUDA。
- 一键脚本：`python -m foundation_models.run_modelscope --model timesfm|kronos`。

### API 版本坑
- TimesFM 2.5：`TimesFM_2p5_200M_torch`（`timesfm==2.0.2`），3.x 换了 `TimesFM3Evaluator`；`ForecastConfig` 必须设 `max_horizon`，log 价要 `infer_is_positive=False`。
- Chronos-Bolt：`BaseChronosPipeline.from_pretrained(..., device_map, torch_dtype)`；CPU 用 `torch.float32`（bfloat16 慢/不支持）；`predict(inputs, prediction_length=...)` 第一个参数是位置参数，不是 `context=`；输出 `[n, 9 分位数, pred_len]`，中位数是第 5 个。

---

## 7. 产物清单（全部已 commit/push）

```
foundation_models/
  README.md              九种模型调研 + 魔搭执行说明
  EXPERIMENTS.md         本文件
  kronos_probe.py / kronos_large.py / analyze_kronos.py / analyze_large.py
  timesfm_probe.py
  chronos_probe.py / chronos_pup.py / analyze_pup.py
  run_modelscope.py
artifacts/
  kronos/   kronos_probe.parquet, kronos_probe_long.parquet,
            kronos_large_b1/b5.parquet, kronos_pup.parquet, chronos_h1_b60.parquet, *.log
  timesfm/  timesfm_probe_b1.parquet, *.log
  chronos/  chronos_probe_cpu_b1/cpu_10k/cpu_h60.parquet, *.log
morph/h10_signal.py      加 MIN_TRAIN/MIN_TEST 覆盖 + build_bar 的 ms() 钳制
kaggle_launcher.py       加 bar60sig project
```

---

## 8. 最终判定与后续

- **基础模型这条线关闭**：4 模型/3 架构/多粒度/多读法，AUC 一致 0.49~0.52，无稳定信号。
- **KiT / FinCast 不再测**：KiT 更慢（284M diffusion，horizon 固定 120）且 license AGPL；FinCast 无清晰推理 API，先验更低。
- **1h LGBM（bar60sig）是唯一新发现**：正 EV 但弱于现役 h=10。若要跟进，方向是（a）用 payout 0.8 重算确认 EV；（b）严格验证 18-23h 时段过滤的增量（train/val 切分，不事后挑）；（c）只做多（长周期空头端被 `kline_thr_wf2` 证明基本无效）。
- **现役主线不变**：h=10 K 线因果阈值 63.32% 仍是最强；精力应放回信息源（链上/跨所/资金费率时序）和目标函数对齐尾部。
