"""本地 Kaggle 编排器（不进 kernel，仅本地用）。

按 --project 在临时目录生成 kernel（拷贝 kaggle_bootstrap.py + 写 entry.txt + 写 kernel-metadata.json），
然后 kaggle kernels push → 轮询 status → 拉 logs（可选解析关键指标）。

约定：
- 默认 enable_gpu=false，只用 CPU（faiss-cpu / lightgbm CPU），绝不自动开 GPU；
- GPU 例外（固定策略）：神经网络/时序模型（CNN/TCN/LSTM/Transformer、morph_ts、
  nn_tcn、kline_dl* 等）在 PROJECTS 对应项写 gpu=True 即可开 GPU；其余项目一律 CPU，
  不装 faiss-gpu。详见 `Kaggle工作流设计.md` §9.6；
- 运行由你显式触发，不会像某些 IDE 插件偷偷占 Kaggle GPU 配额；
- kernel-metadata 模板化在此，项目文件夹里不再散落 json。
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
BOOTSTRAP = os.path.join(REPO_ROOT, "kaggle_bootstrap.py")
SECRETS = os.path.join(REPO_ROOT, ".codebuddy", "settings.json")   # 已被 .gitignore


def _secrets(names):
    """从本地 settings.json 取指定密钥；缺失就跳过（对应 env 由项目自己校验）。"""
    if not names:
        return {}
    try:
        with open(SECRETS, encoding="utf-8") as f:
            d = json.load(f)
    except Exception:
        print(f"[warn] 读不到 {SECRETS}，缺 {','.join(names)}")
        return {}
    return {k: d[k] for k in names if k in d}

# 各项目的 kernel 配置（slug / title / 入口 / 数据集 / 是否需联网）
PROJECTS = {
    "morph": dict(
        slug="byh850515/btc-vs-eth-faiss-cpu-2020",
        title="BTC vs ETH 价格形态聚类存在性证明 (faiss-cpu, 2020+)",
        entry="morph.kaggle_entry",
        enable_internet=True,
        dataset_sources=["byh850515/eth-btc-1m-2020"],
    ),
    # 外部参考（termbin f8mx）可借鉴项 A/B：方向步进 / 自相关状态 / 分裂专模 / 极浅树
    # 判据：roll3d 三窗均值 +≥0.5pp 且最差窗不退化（单窗 <1.5pp 视为噪声）
    "borrow": dict(
        slug="byh850515/btcusdt-borrowed-idea-ab",
        title="BTCUSDT Borrowed Idea AB",
        entry="morph.borrow_ab",
        enable_internet=True,
        dataset_sources=["byh850515/btcusdt-1m-full-2020"],
        env={"OFFSETS": "0,60,120", "SEEDS": "42,43", "ROUNDS": "60",
             "TRAIN_STRIDE": "2",
             "VARIANTS": "base,dir,auto,diraut,regime,shallow",
             "OUT": "/kaggle/working/borrow_pred.parquet"},
    ),
    # K 线腿引入非价格新信息（资金费率/持仓量/大户多空比/taker 多空量比）
    "newinfo": dict(
        slug="byh850515/btcusdt-kline-newinfo",
        title="BTCUSDT Kline NewInfo",
        entry="morph.kline_newinfo",
        enable_internet=True,
        dataset_sources=["byh850515/btcusdt-1m-full-2020"],
        env={"OFFSETS": "0,60,120", "SEEDS": "42,43", "ROUNDS": "60",
             "TRAIN_STRIDE": "2"},
    ),
    # Stoch「指标趋势线突破」（用户实盘画法）：斜率是否决定延续性、到极值要多久
    # 纯统计，CPU；数据（ETHUSDT 1m）在 kernel 内从 GitHub Release 拉，不占本地带宽
    "kdtrend": dict(
        slug="byh850515/ethusdt-stoch-trendline",
        title="ETHUSDT Stoch Trendline Breakout",
        entry="morph.kd_trendline",
        enable_internet=True,
        dataset_sources=[],
        env={"SYMBOL": "ETHUSDT", "YEARS": "2024,2025,2026", "MINS": "1,5"},
    ),
    # 波动率分桶：低波动期是否更好预测、是否该只在低波动档下单
    "volbkt": dict(
        slug="byh850515/btcusdt-vol-bucket",
        title="BTCUSDT Vol Bucket",
        entry="orderbook.vol_bucket",
        enable_internet=True,
        dataset_sources=["byh850515/btcusdt-orderbook-features"],
        env={"OFFS": "0,30", "SEEDS": "42,43,44,45,46", "ROUNDS": "60", "NB": "5"},
    ),
    # 指标稳定性：只换种子组，看 static / roll_1d / roll_3d 各自抖多少
    "mstab": dict(
        slug="byh850515/btcusdt-metric-stability",
        title="BTCUSDT Metric Stability",
        entry="orderbook.metric_stability",
        enable_internet=True,
        dataset_sources=["byh850515/btcusdt-orderbook-features"],
        env={"OFF": "0", "ROUNDS": "60",
             "SEED_GROUPS": "42,43,44,45,46;142,143,144,145,146;242,243,244,245,246"},
    ),
    # 复跑 2026-11-03 那条产出 0.6410 的旧链路（fusion_kl.py），落 pred 供离线核对
    "fuseold": dict(
        slug="byh850515/btcusdt-fuse-old-repro",
        title="BTCUSDT Fuse Old Repro",
        entry="orderbook.fusion_kl",
        enable_internet=True,
        dataset_sources=["byh850515/btcusdt-orderbook-features",
                         "byh850515/btcusdt-1m-full-2020"],
        env={"KAGGLE_OFFSETS": "0", "KAGGLE_OSEEDS": "42,43,44,45,46",
             "KAGGLE_KSEEDS": "42,43,44", "KAGGLE_HORIZON": "3",
             "KAGGLE_FUSION_PRED": "/kaggle/working/old_pred.parquet"},
    ),
    # 口径对照：同一模型在 static_val / roll_1d / roll_3d 下的数字 + 覆盖天/单天峰值自证
    "caliber": dict(
        slug="byh850515/btcusdt-caliber-check",
        title="BTCUSDT Caliber Check",
        entry="orderbook.caliber_check",
        enable_internet=True,
        dataset_sources=["byh850515/btcusdt-orderbook-features"],
        env={"OFFS": "0", "SEEDS": "42,43,44,45,46", "ROUNDS": "60", "WF_WIN": "365",
             "ES": "1", "WF": "0"},
    ),
    # 三项已验证有效的改动合成：wf365 重训 + no_mom + 融合，扫 w×K 给最优操作点
    "best": dict(
        slug="byh850515/btcusdt-best-combo",
        title="BTCUSDT Best Combo",
        entry="orderbook.best_combo",
        enable_internet=True,
        dataset_sources=["byh850515/btcusdt-orderbook-features",
                         "byh850515/btcusdt-1m-full-2020"],
        env={"OFFS": "0,60,120", "OSEEDS": "42,43,44", "KSEEDS": "42,43",
             "ROUNDS": "60", "WF_WIN": "365"},
    ),
    # taker 主动成交多空量比（5 分钟 metrics）作为 K 线腿新信息：lag0 vs lag5 对照前视幅度
    "taker": dict(
        slug="byh850515/btcusdt-kline-taker",
        title="BTCUSDT Kline Taker",
        entry="morph.kline_taker",
        enable_internet=True,
        dataset_sources=["byh850515/btcusdt-1m-full-2020"],
        env={"OFFSETS": "0,60,120", "SEEDS": "42,43", "ROUNDS": "60",
             "TRAIN_STRIDE": "1", "VARIANTS": "base,recent,lag5,lag0,takonly"},
    ),
    # 诊断：newinfo 异常高 AUC 的泄漏源
    "newdiag": dict(
        slug="byh850515/btcusdt-newinfo-diag",
        title="BTCUSDT NewInfo Diag",
        entry="morph.newinfo_diag",
        enable_internet=True,
        dataset_sources=["byh850515/btcusdt-1m-full-2020"],
        env={},
    ),
    # 盘口腿 walk-forward 滚动重训：验证"模型太旧/漂移"是不是 0.62 卡住的主因
    "wfrefit": dict(
        slug="byh850515/btcusdt-wf-refit",
        title="BTCUSDT WF Refit",
        entry="orderbook.wf_refit",
        enable_internet=True,
        dataset_sources=["byh850515/btcusdt-orderbook-features"],
        env={"OFFS": "0,60,120", "SEEDS": "42,43,44", "BASE_SEEDS": "42,43,44,45,46",
             "ROUNDS": "60", "WINS": "180,365"},
    ),
    # 两腿融合 + 频次-准确率曲线（65% 目标的落点）
    "fusek": dict(
        slug="byh850515/btcusdt-fuse-kbest",
        title="BTCUSDT Fuse KBest",
        entry="orderbook.fuse_kbest",
        enable_internet=True,
        dataset_sources=["byh850515/btcusdt-orderbook-features",
                         "byh850515/btcusdt-1m-full-2020"],
        env={"OFFS": "0,60,120", "OSEEDS": "42,43,44,45,46", "KSEEDS": "42,43",
             "ROUNDS": "60", "KL_SUBSET": "no_mom",
             "OUT": "/kaggle/working/fuse_pred.parquet"},
    ),
    # K 线腿特征组消融：定位独立信息在哪一类特征（第 1 步：单腿提升）
    "ablate": dict(
        slug="byh850515/btcusdt-kline-ablate",
        title="BTCUSDT Kline Ablate",
        entry="morph.kline_ablate",
        enable_internet=True,
        dataset_sources=["byh850515/btcusdt-1m-full-2020"],
        env={"OFFSETS": "0,60,120", "SEEDS": "42,43", "ROUNDS": "60",
             "TRAIN_STRIDE": "2",
             "VARIANTS": "all,no_mom,flow_shape,flow_only,shape_only,vol_shape",
             "OUT": "/kaggle/working/ablate_pred.parquet"},
    ),
    # K 线腿多 horizon 的滚动 z 复核：§26 的 h 扫描是 static 口径，需确认 10/15min 优势是否真实
    "khz": dict(
        slug="byh850515/btcusdt-kline-h-roll",
        title="BTCUSDT Kline H Roll",
        entry="morph.kline_h_roll",
        enable_internet=True,
        dataset_sources=["byh850515/btcusdt-1m-full-2020"],
        env={"HS": "3,5,10,15,30", "OFFSETS": "0,60,120", "SEEDS": "42,43",
             "ROUNDS": "60", "TRAIN_STRIDE": "2"},
    ),
    # 同上实验的最终裁决：把 regime 分裂专模搬到**盘口腿（生产腿）**，口径与 znorm_eval 一致
    "borrowob": dict(
        slug="byh850515/btcusdt-borrow-regime-ob",
        title="BTCUSDT Borrow Regime OB",
        entry="orderbook.borrow_ob",
        enable_internet=True,
        dataset_sources=["byh850515/btcusdt-orderbook-features"],
        env={"OFFS": "0,60,120", "SEEDS": "42,43,44,45,46", "ROUNDS": "60",
             "VARIANTS": "base,auto,regime,rauto"},
    ),
    "orderbook": dict(
        slug="byh850515/btc-orderbook-lgbm",
        title="BTCUSDT OrderBook Features - LightGBM t+3",
        entry="orderbook.train",
        enable_internet=True,
        dataset_sources=["byh850515/btcusdt-orderbook-features"],
    ),
    "fusion": dict(
        slug="byh850515/btcusdt-orderbook-x-faiss-morph-fusion-a-b",
        title="BTCUSDT OrderBook x faiss Morph Fusion A/B",
        entry="orderbook.fusion_test",
        enable_internet=True,
        dataset_sources=["byh850515/btcusdt-orderbook-features", "byh850515/eth-btc-1m-2020"],
        env={"KAGGLE_HORIZON": "3",
             "KAGGLE_FUSION_PRED": "/kaggle/working/fusion_pred_3m.parquet",
             "KAGGLE_FUSION_DERIVED": "1",
             "KAGGLE_FUSION_DECAY": "120",
             "KAGGLE_FUSION_WFIXED": "0"},
    ),
    "fusion15": dict(
        slug="byh850515/btcusdt-orderbook-x-faiss-morph-fusion-a-b",
        title="BTCUSDT OrderBook x faiss Morph Fusion A/B",
        entry="orderbook.fusion_test",
        enable_internet=True,
        dataset_sources=["byh850515/btcusdt-orderbook-features", "byh850515/eth-btc-1m-2020"],
        env={"KAGGLE_HORIZON": "15"},
    ),
    # 3min orderbook 变体矩阵：一次跑多组特征/权重/过滤/超参，快速定位有效方向
    "matrix3": dict(
        slug="byh850515/btcusdt-fusion-matrix-3min",
        title="BTCUSDT Fusion Matrix 3min",
        entry="orderbook.fusion_matrix",
        enable_internet=True,
        dataset_sources=["byh850515/btcusdt-orderbook-features"],
        env={"KAGGLE_HORIZON": "3",
             "KAGGLE_HORIZONS": "1",
             "KAGGLE_VARIANTS": "base,derived,reg,dr",
             "KAGGLE_TEST_OFFSETS": "0",
             "KAGGLE_SEEDS": "42,43,44,45,46,47,48,49,50,51",
             "KAGGLE_FUSION_PRED": "/kaggle/working/matrix_pred.parquet"},
    ),
    # 纯 K 线 3min 模型：5.5 年数据（2020+）+ 60 余个 K 线特征
    "kline3": dict(
        slug="byh850515/btcusdt-kline-feats-3min",
        title="BTCUSDT Kline Feats 3min",
        entry="morph.kline_lgb",
        enable_internet=True,
        dataset_sources=[],
        env={"KAGGLE_HORIZON": "3",
             "KAGGLE_TEST_OFFSETS": "0"},
    ),
    # 3min 攻坚 4：固定轮数 vs 早停（修集成不稳定）
    "h3fixed": dict(
        slug="byh850515/btcusdt-h3-fixed",
        title="BTCUSDT H3 Fixed",
        entry="orderbook.h3_fixed",
        enable_internet=True,
        dataset_sources=["byh850515/btcusdt-orderbook-features"],
    ),
    # 3min 攻坚 3：时段稳定性验证（多 test 段）
    "h3session": dict(
        slug="byh850515/btcusdt-h3-session",
        title="BTCUSDT H3 Session",
        entry="orderbook.h3_session",
        enable_internet=True,
        dataset_sources=["byh850515/btcusdt-orderbook-features"],
    ),
    # 3min 攻坚 2：时段分层 + 3 分钟累积订单流特征
    "h3time": dict(
        slug="byh850515/btcusdt-h3-time",
        title="BTCUSDT H3 Time",
        entry="orderbook.h3_time",
        enable_internet=True,
        dataset_sources=["byh850515/btcusdt-orderbook-features"],
    ),
    # 3min 专项攻坚（事件合约最小周期=3min）：一致性打分 + 连续加权
    "h3push": dict(
        slug="byh850515/btcusdt-h3-push",
        title="BTCUSDT H3 Push",
        entry="orderbook.h3_push",
        enable_internet=True,
        dataset_sources=["byh850515/btcusdt-orderbook-features"],
        env={"KAGGLE_FUSION_PRED": "/kaggle/working/h3_push_pred.parquet"},
    ),
    # 多 horizon 分数混合：h=1 强信号补 h=3（合并成单一打分）
    "blend3": dict(
        slug="byh850515/btcusdt-blend-h1-h2-h3",
        title="BTCUSDT Blend H1 H2 H3",
        entry="orderbook.blend_h",
        enable_internet=True,
        dataset_sources=["byh850515/btcusdt-orderbook-features"],
        env={"KAGGLE_HORIZONS": "1,2,3",
             "KAGGLE_FUSION_PRED": "/kaggle/working/blend_pred.parquet"},
    ),
    # 两腿融合：K 线 5.5 年（63 量价特征）× 盘口 15 个月（150 微观特征）——两条腿用**不同的训练集**
    "fusionkl": dict(
        slug="byh850515/btcusdt-fusion-kline-x-orderbook",
        title="BTCUSDT Fusion Kline x OrderBook",
        entry="orderbook.fusion_kl",
        enable_internet=True,
        dataset_sources=["byh850515/btcusdt-orderbook-features"],
        env={"KAGGLE_HORIZON": "3",
             "KAGGLE_KSEEDS": "42,43,44",
             "KAGGLE_OSEEDS": "42,43,44,45,46",
             "KAGGLE_OFFSETS": "0,60,120",
             "KAGGLE_W_FIXED": "0.2",
             "KAGGLE_FUSION_PRED": "/kaggle/working/fusion_kl_pred.parquet"},
    ),
    # 15min 形态时序模型（1D CNN，CPU）：faiss 腿的替代/升级，产出 pred parquet 供组合器使用
    "morph15": dict(
        slug="byh850515/btcusdt-15min-morph-ts-1d-cnn-cpu",
        title="BTCUSDT 15min Morph TS 1D CNN CPU",
        entry="morph.morph_ts",
        enable_internet=True,
        dataset_sources=["byh850515/eth-btc-1m-2020"],
        env={"KAGGLE_HORIZON": "15",
             "KAGGLE_MORPH_TS_OUT": "/kaggle/working/morph_ts_15m.parquet"},
    ),
    # v2 单变量对照：v1 只用了 px/st 两通道（误挂 OHLC-only 数据）且明显欠拟合
    # (train loss 0.6911→0.6902、早停第 3 轮)。改挂 btcusdt-1m-full-2020
    # (含 volume/taker_buy_volume) 后 build_channels 自动补 vol+cvd 两通道。
    # --max-ct 把数据末尾钉到 v1 的 2026-09-30，train/val/test 切分严格一致。
    "morph15v2a": dict(
        slug="byh850515/btcusdt-15min-morph-ts-v2a-channels",
        title="BTCUSDT 15min Morph TS v2a Channels",
        entry="morph.morph_ts",
        enable_internet=True,
        dataset_sources=["byh850515/btcusdt-1m-full-2020"],
        env={"KAGGLE_HORIZON": "15",
             "KAGGLE_MORPH_TS_OUT": "/kaggle/working/morph_ts_v2a.parquet",
             # 与 v1 完全相同的超参，唯一差异 = 多两个通道
             "MORPH_TS_ARGS": "--max-ct 1790811900000 --seeds 42,43 --epochs 6 "
                              "--patience 2 --lr 0.002 --drop 0.2 --train-stride 8 "
                              "--test-days 92 --val-days 30 --window 120"},
    ),
    "morph15v2b": dict(
        slug="byh850515/btcusdt-15min-morph-ts-v2b-fit",
        title="BTCUSDT 15min Morph TS v2b Fit",
        entry="morph.morph_ts",
        enable_internet=True,
        dataset_sources=["byh850515/btcusdt-1m-full-2020"],
        env={"KAGGLE_HORIZON": "15",
             "KAGGLE_MORPH_TS_OUT": "/kaggle/working/morph_ts_v2b.parquet",
             # 加通道 + 修欠拟合：降 lr、放宽 dropout、样本加倍、允许多轮
             "MORPH_TS_ARGS": "--max-ct 1790811900000 --seeds 42,43 --epochs 10 "
                              "--patience 3 --lr 0.0007 --drop 0.05 --train-stride 4 "
                              "--test-days 92 --val-days 30 --window 120"},
    ),
    # v3：把「训练/选模目标」直接对准高置信尾部（§14 根因：目标错配）。
    # 回到 v1 的 2 通道 OHLC 数据与完全相同的超参，各只改一个变量，严格可比。
    "morph15v3a": dict(
        slug="byh850515/btcusdt-15min-morph-ts-v3a-tail-select",
        title="BTCUSDT 15min Morph TS v3a Tail Select",
        entry="morph.morph_ts",
        enable_internet=True,
        dataset_sources=["byh850515/eth-btc-1m-2020"],
        env={"KAGGLE_HORIZON": "15",
             "KAGGLE_MORPH_TS_OUT": "/kaggle/working/morph_ts_v3a.parquet",
             "MORPH_TS_ARGS": "--max-ct 1790811900000 --seeds 42,43 --epochs 6 "
                              "--patience 2 --lr 0.002 --drop 0.2 --train-stride 8 "
                              "--test-days 92 --val-days 30 --window 120 "
                              "--select-metric tail"},
    ),
    "morph15v3b": dict(
        slug="byh850515/btcusdt-15min-morph-ts-v3b-topq-loss",
        title="BTCUSDT 15min Morph TS v3b TopQ Loss",
        entry="morph.morph_ts",
        enable_internet=True,
        dataset_sources=["byh850515/eth-btc-1m-2020"],
        env={"KAGGLE_HORIZON": "15",
             "KAGGLE_MORPH_TS_OUT": "/kaggle/working/morph_ts_v3b.parquet",
             "MORPH_TS_ARGS": "--max-ct 1790811900000 --seeds 42,43 --epochs 6 "
                              "--patience 2 --lr 0.002 --drop 0.2 --train-stride 8 "
                              "--test-days 92 --val-days 30 --window 120 "
                              "--loss-mode topq --topq 0.10"},
    ),
    # 基线可信度检验：v1 配置完全不变，只把 test 段整体前移 92 / 184 天。
    # 若 0.6337 复现不了，说明它是 test 段运气，前面所有「优化变差」都不是改动之过。
    "morph15off1": dict(
        slug="byh850515/btcusdt-15min-morph-ts-offset-1",
        title="BTCUSDT 15min Morph TS Offset 1",
        entry="morph.morph_ts",
        enable_internet=True,
        dataset_sources=["byh850515/eth-btc-1m-2020"],
        env={"KAGGLE_HORIZON": "15",
             "KAGGLE_MORPH_TS_OUT": "/kaggle/working/morph_ts_off1.parquet",
             "MORPH_TS_ARGS": "--max-ct 1782863100000 --seeds 42,43 --epochs 6 "
                              "--patience 2 --lr 0.002 --drop 0.2 --train-stride 8 "
                              "--test-days 92 --val-days 30 --window 120"},
    ),
    "morph15off2": dict(
        slug="byh850515/btcusdt-15min-morph-ts-offset-2",
        title="BTCUSDT 15min Morph TS Offset 2",
        entry="morph.morph_ts",
        enable_internet=True,
        dataset_sources=["byh850515/eth-btc-1m-2020"],
        env={"KAGGLE_HORIZON": "15",
             "KAGGLE_MORPH_TS_OUT": "/kaggle/working/morph_ts_off2.parquet",
             "MORPH_TS_ARGS": "--max-ct 1774914300000 --seeds 42,43 --epochs 6 "
                              "--patience 2 --lr 0.002 --drop 0.2 --train-stride 8 "
                              "--test-days 92 --val-days 30 --window 120"},
    ),
    # 数据生产：2020+ 全量 um 1m K 线（含 volume/taker），产物注册为 Kaggle Dataset
    "klineds": dict(
        slug="byh850515/btcusdt-1m-full-2020-dataset-maker",
        title="BTCUSDT 1m Full 2020 Dataset Maker",
        entry="morph.make_full_dataset",
        enable_internet=True,
        dataset_sources=[],
    ),
    # 纯 K 线第十三轮：定稿二 —— 5 seed + CNN 窗口120/10epoch/CVD斜率加速度
    # 第十二轮已证 VP 无效（AUC 0.5277->0.5280，2/4 窗口反向），故 USE_VP=0
    "boost13": dict(
        slug="byh850515/btcusdt-kline-final2",
        title="BTCUSDT Kline Final2",
        entry="morph.kline_final2",
        enable_internet=True,
        dataset_sources=["byh850515/btcusdt-1m-full-2020"],
        env={"SEEDS": "42,43,44,45,46", "ROUNDS": "60", "USE_VP": "0",
             "VP_WINS": "240,960,1440", "VP_TICK": "2.0", "VP_ANCHOR": "10",
             "CNN_CHANS": "9", "CNN_SEEDS": "42,43", "CNN_EPOCHS": "10",
             "CNN_WINDOW": "120", "CNN_STRIDE": "16"},
    ),
    # 纯 K 线第十六轮：多数票目标（修正 off-by-one 标签）+ 连涨连跌形态条件概率表
    "boost16": dict(
        slug="byh850515/btcusdt-kline-mvote2",
        title="BTCUSDT Kline MVote2",
        entry="morph.kline_mvote2",
        enable_internet=True,
        dataset_sources=["byh850515/btcusdt-1m-full-2020"],
        env={"SEEDS": "42,43,44,45,46", "MV_SEEDS": "42,43,44", "ROUNDS": "60",
             "MVS": "5:3,5:4,10:5,10:6,10:7"},
    ),
    # 纯 K 线第十五轮：三条"新视角"复核 —— 3m 公平头对头 + 多数票目标（连开）
    "boost15": dict(
        slug="byh850515/btcusdt-kline-probe-3m-mvote",
        title="BTCUSDT Kline Probe 3m MVote",
        entry="morph.kline_probe",
        enable_internet=True,
        dataset_sources=["byh850515/btcusdt-1m-full-2020"],
        env={"SEEDS": "42,43,44,45,46", "MV_SEEDS": "42,43,44", "ROUNDS": "60",
             "MVS": "5:3,10:6,10:5"},
    ),
    # 纯 K 线第十四轮：只把 seed 加到 5，CNN 保持第十一轮的 7 通道/60 窗口/5 epoch，
    # 隔离"seed 增益"（第十三轮 CNN9 与 CNN7 打平，混在一起看不清）
    "boost14": dict(
        slug="byh850515/btcusdt-kline-final2-cnn7",
        title="BTCUSDT Kline Final2 CNN7",
        entry="morph.kline_final2",
        enable_internet=True,
        dataset_sources=["byh850515/btcusdt-1m-full-2020"],
        env={"SEEDS": "42,43,44,45,46", "ROUNDS": "60", "USE_VP": "0",
             "CNN_CHANS": "7", "CNN_SEEDS": "42,43", "CNN_EPOCHS": "5",
             "CNN_WINDOW": "60", "CNN_STRIDE": "8"},
    ),
    # 纯 K 线第十二轮：VP（量价分布）指标 —— POC 距离/价值区间位置/上下方量比/VWAP 偏离
    "boost12": dict(
        slug="byh850515/btcusdt-kline-vp",
        title="BTCUSDT Kline VP",
        entry="morph.kline_vp",
        enable_internet=True,
        dataset_sources=["byh850515/btcusdt-1m-full-2020"],
        env={"SEEDS": "42,43", "ROUNDS": "60",
             "VP_WINS": "240,960,1440", "VP_TICK": "2.0", "VP_ANCHOR": "10"},
    ),
    # 纯 K 线第十一轮（定稿）：63特征多horizon + 无vol CNN 融合
    "boost11": dict(
        slug="byh850515/btcusdt-kline-final",
        title="BTCUSDT Kline Final",
        entry="morph.kline_final",
        enable_internet=True,
        dataset_sources=["byh850515/btcusdt-1m-full-2020"],
        env={"SEEDS": "42,43", "CNN_EPOCHS": "5", "CNN_STRIDE": "8", "ROUNDS": "60"},
    ),
    # 纯 K 线第十轮：剔除总成交量特征 + CNN 7 通道（合约 CVD）
    "boost10": dict(
        slug="byh850515/btcusdt-kline-novol",
        title="BTCUSDT Kline NoVol",
        entry="morph.kline_novol",
        enable_internet=True,
        dataset_sources=["byh850515/btcusdt-1m-full-2020"],
        env={"SEEDS": "42,43", "CNN_EPOCHS": "5", "CNN_STRIDE": "8", "ROUNDS": "60"},
    ),
    # 纯 K 线第九轮：多 horizon 一致性加权（h=1/2/3/5/10 互相验证）
    "boost9": dict(
        slug="byh850515/btcusdt-kline-multi-h",
        title="BTCUSDT Kline Multi H",
        entry="morph.kline_multi",
        enable_internet=True,
        dataset_sources=["byh850515/btcusdt-1m-full-2020"],
        env={"SEEDS": "42,43", "ROUNDS": "60"},
    ),
    # 纯 K 线第八轮：窗口形状聚类 + 簇胜率前向编码 作为 LGB 特征
    "boost8": dict(
        slug="byh850515/btcusdt-kline-shape-cluster",
        title="BTCUSDT Kline Shape Cluster",
        entry="morph.kline_shape",
        enable_internet=True,
        dataset_sources=["byh850515/btcusdt-1m-full-2020"],
        env={"SEEDS": "42,43,44", "ROUNDS": "60"},
    ),
    # 纯 K 线第七轮：回归目标（学幅度）+ 时间衰减权重 + 分类×回归组合
    "boost7": dict(
        slug="byh850515/btcusdt-kline-reg",
        title="BTCUSDT Kline Reg",
        entry="morph.kline_reg",
        enable_internet=True,
        dataset_sources=["byh850515/btcusdt-1m-full-2020"],
        env={"SEEDS": "42,43", "ROUNDS": "60"},
    ),
    # 纯 K 线第六轮：CNN 8 通道（实体/影线/收益）+ 权重扫描融合
    "boost6": dict(
        slug="byh850515/btcusdt-kline-dl2-shape",
        title="BTCUSDT Kline DL2 Shape",
        entry="morph.kline_dl2",
        enable_internet=True,
        dataset_sources=["byh850515/btcusdt-1m-full-2020"],
        env={"SEEDS": "42,43,44", "CNN_SEEDS": "42,43", "CNN_EPOCHS": "5",
             "CNN_STRIDE": "8", "ROUNDS": "60"},
    ),
    # 纯 K 线第五轮：3m 基础 K 线 + 1D CNN 形态模型 + 四路融合
    "boost5": dict(
        slug="byh850515/btcusdt-kline-dl-3m-cnn",
        title="BTCUSDT Kline DL 3m CNN",
        entry="morph.kline_dl",
        enable_internet=True,
        dataset_sources=["byh850515/btcusdt-1m-full-2020"],
        env={"SEEDS": "42,43,44,45,46", "CNN_SEEDS": "42,43", "CNN_EPOCHS": "4",
             "CNN_STRIDE": "8", "CNN_STRIDE3": "4", "ROUNDS": "60"},
    ),
    # 纯 K 线第四轮：评估端中心偏移调优（mu / 单边 vs 对称），val 选参 test 评估
    "boost4": dict(
        slug="byh850515/btcusdt-kline-boost4-threshold",
        title="BTCUSDT Kline Boost4 Threshold",
        entry="morph.kline_boost4",
        enable_internet=True,
        dataset_sources=["byh850515/btcusdt-1m-full-2020"],
        env={"SEEDS": "42,43,44,45,46", "ROUNDS": "60"},
    ),
    # 纯 K 线第三轮：8seed / 软加权 / h1-2-3 融合 / 多样性 bagging
    "boost3": dict(
        slug="byh850515/btcusdt-kline-boost3-ens",
        title="BTCUSDT Kline Boost3 Ens",
        entry="morph.kline_boost3",
        enable_internet=True,
        dataset_sources=["byh850515/btcusdt-1m-full-2020"],
        env={"SEEDS": "42,43,44,45,46,47,48,49", "ROUNDS": "60", "ONLY": ""},
    ),
    # 纯 K 线第二轮：噪声基线（seed/窗口 sd）+ 标签净化 + 形态×符号
    "boost2": dict(
        slug="byh850515/btcusdt-kline-boost2-noise-label",
        title="BTCUSDT Kline Boost2 Noise Label",
        entry="morph.kline_boost2",
        enable_internet=True,
        dataset_sources=["byh850515/btcusdt-1m-full-2020"],
        env={"SEEDS": "42,43,44,45,46", "ROUNDS": "60", "ONLY": ""},
    ),
    # 纯 K 线提升对照 v0-v4（1m/3m 重采样/符号累加/形态子集/堆叠），数据直接取 dataset
    "boost": dict(
        slug="byh850515/btcusdt-kline-boost-v0-v4",
        title="BTCUSDT Kline Boost v0-v4",
        entry="morph.kline_boost",
        enable_internet=True,
        dataset_sources=["byh850515/btcusdt-1m-full-2020"],
        env={"SEEDS": "42,43", "ROUNDS": "60", "ONLY": ""},
    ),
    # 一条命令跑完：kernel 内直接拉 bn_hisdata 的 Release（K 线 + 订单簿特征）->
    # 算特征 -> 训练 -> 评估。迭代特征就改代码重推，不落"数据集"、不挂 Dataset。
    "twolegs": dict(
        slug="byh850515/btcusdt-two-legs-fusion",
        title="BTCUSDT Two Legs Fusion",
        entry="morph.two_legs",
        enable_internet=True,
        dataset_sources=[],
        # 一次扫 4 个 horizon：特征/数据只取一次，每条腿按 h 重训（≈11 分钟）
        env={"SYMBOL": "BTCUSDT", "KLINE_SRC": "um", "WITH_BASIS": "0",
             "HORIZONS": "3,5,10,15", "TARGET_TPD": "15", "ROUNDS": "2000"},
    ),
    # h=3 专用宽表尝试（K线 + 订单簿 + 跨域），**独立于 twolegs 基线**，便于回退；
    # 每次跑完把 EXP 行抄进 `h3宽表实验记录.md`。
    "h3wide": dict(
        slug="byh850515/btcusdt-h3-wide",
        title="BTCUSDT H3 Wide",
        entry="morph.h3_wide",
        enable_internet=True,
        dataset_sources=[],
        # FEATS 逗号分隔：一次内核里跑多组特征集（数据/跨域只算一次）
        # 多 seed（≥5）概率平均：单 seed 的 ±0.5pp 差异淹没在种子方差里（记录 §待办 1）
        env={"SYMBOL": "BTCUSDT", "FEATS": "ob,ob+cross,k+ob,k+ob+cross", "HORIZON": "3",
             "TARGET_TPD": "15", "ROUNDS": "2000", "EXP": "W",
             "SEEDS": "42,43,44,45,46"},
    ),
    # 跨域接入两腿基线：口径照抄 twolegs，只把订单簿腿换成 ob / ob+cross 做对照
    "h3fuse": dict(
        slug="byh850515/btcusdt-h3-fuse-cross",
        title="BTCUSDT H3 Fuse Cross",
        entry="morph.h3_fuse_cross",
        enable_internet=True,
        dataset_sources=[],
        env={"SYMBOL": "BTCUSDT", "KLINE_SRC": "um", "WITH_BASIS": "0",
             "VARIANTS": "ob,ob+cross", "HORIZON": "3", "TARGET_TPD": "15",
             "ROUNDS": "2000", "SEEDS": "42,43,44,45,46", "EXP": "F"},
    ),
    # 诊断：同进程内并排跑两种切分口径（time/row）× ob / ob+cross，FUSE=0 只比订单簿腿
    "h3ab": dict(
        slug="byh850515/btcusdt-h3-fuse-cross",
        title="BTCUSDT H3 Fuse Cross",
        entry="morph.h3_fuse_cross",
        enable_internet=True,
        dataset_sources=[],
        env={"SYMBOL": "BTCUSDT", "KLINE_SRC": "um", "WITH_BASIS": "0",
             "VARIANTS": "ob,ob+cross", "SPLITS": "time,row", "FUSE": "0",
             "HORIZON": "3", "TARGET_TPD": "15", "ROUNDS": "2000",
             "KLINE_FROM": "2025", "SEEDS": "42,43,44,45,46", "EXP": "A",
             "OUT": "/kaggle/working/h3_ab_pred.parquet"},
    ),
    # 亚分钟盘口动态（S*）：从原始 orderbook 事件流提取"分钟内过程"，补 185 列丢掉的信息
    # key 从 .codebuddy/settings.json 注入，不进仓库
    "obsub": dict(
        slug="byh850515/btcusdt-ob-subminute",
        title="BTCUSDT OB Subminute",
        entry="morph.ob_subminute",
        enable_internet=True,
        dataset_sources=[],
        secrets=["CHD_API_KEY"],
        env={"SYMBOL": "BTCUSDT", "MONTH": "2026-09", "TARGET_TPD": "15",
             "ROUNDS": "2000", "SEEDS": "42,43,44,45,46", "EXP": "S1",
             "KLINE_FROM": "2025", "MAX_HOURS": "0",
             "OUT": "/kaggle/working/ob_subminute.parquet"},
    ),
    # 复核：换一组种子重跑 ob / ob+cross，确认跨域增益不是种子偶然（X* 编号）
    "h3wide2": dict(
        slug="byh850515/btcusdt-h3-wide",
        title="BTCUSDT H3 Wide",
        entry="morph.h3_wide",
        enable_internet=True,
        dataset_sources=[],
        env={"SYMBOL": "BTCUSDT", "FEATS": "ob,ob+cross", "HORIZON": "3",
             "TARGET_TPD": "15", "ROUNDS": "2000", "EXP": "X",
             "SEEDS": "142,143,144,145,146"},
    ),
    # G*：二阶段下单过滤（不动模型/特征，只改下单口径）；val 定阈值、test 配对 bootstrap
    # smoke：--set OB_MONTHS=2026-09 --set KLINE_FROM=2026 --set SEEDS=42 --set ROUNDS=50
    #        --set BOOT=50 --set PLACEBO=20 --set SPLITS=time --set EXP=G0
    "h3gate": dict(
        slug="byh850515/btcusdt-h3-gate",
        title="BTCUSDT H3 Gate",
        entry="morph.h3_gate",
        enable_internet=True,
        dataset_sources=[],
        env={"SYMBOL": "BTCUSDT", "KLINE_SRC": "um", "WITH_BASIS": "0",
             "KLINE_FROM": "2025", "SPLITS": "time,row",
             "SEEDS": "42,43,44,45,46", "ROUNDS": "2000", "TARGET_TPD": "15",
             "BOOT": "1000", "PLACEBO": "200", "MIN_COV": "0.34", "EXP": "G1",
             "OUT": "/kaggle/working/h3_gate_pred.parquet"},
    ),
    # ETH 数据集 schema 探针（只读挂载数据，不训练）
    "ethprobe": dict(
        slug="byh850515/ethusdt-schema-probe",
        title="ETHUSDT Schema Probe",
        entry="morph.eth_probe",
        enable_internet=True,
        dataset_sources=["byh850515/ethusdt-merged-features"],
        env={},
    ),
    # ETH vol8 纯盘口数据生产（185 列，与 ob_BTCUSDT.parquet 同构）；7h 长任务
    # smoke：--set MODE=smoke --set MAX_DAYS=2（只验证 band + L1(vol8) 管线）
    "ethbuild": dict(
        slug="byh850515/ethusdt-vol8-ob-build",
        title="ETHUSDT Vol8 OB Build",
        entry="morph.eth_build",
        enable_internet=True,
        dataset_sources=[],
        secrets=["CHD_API_KEY"],
        env={"MODE": "build", "OB_SYMBOL": "ETHUSDT",
             "START": "2025-07-01", "END": "2026-10-20", "MAX_DAYS": "0"},
    ),
    # E*：ETH 用现役 h=3 基线同逻辑评估；挂载 ethbuild 的 output（等 build 完成后再推）
    "h3eth": dict(
        slug="byh850515/ethusdt-h3-baseline",
        title="ETHUSDT H3 Baseline",
        entry="morph.h3_eth",
        enable_internet=True,
        dataset_sources=[],
        kernel_sources=["byh850515/ethusdt-vol8-ob-build"],
        env={"SYMBOL": "ETHUSDT", "KLINE_SRC": "um", "WITH_BASIS": "0",
             "KLINE_FROM": "2025", "SPLITS": "time,row",
             "SEEDS": "42,43,44,45,46", "ROUNDS": "2000", "TARGET_TPD": "15",
             "EXP": "E1", "OUT": "/kaggle/working/h3_eth_pred.parquet"},
    ),
    # W1：walk-forward 逐月重训（实盘口径）；回答固定划分那 6.7pp 差距里
    # 有多少是「模型陈旧」、多少是「时段差异」
    "wfeval": dict(
        slug="byh850515/btcusdt-h3-walkforward",
        title="BTCUSDT H3 Walk Forward",
        entry="morph.wf_eval",
        enable_internet=True,
        dataset_sources=[],
        env={"SEEDS": "42,43,44,45,46", "ROUNDS": "2000", "KLINE_FROM": "2025",
             "START_MONTH": "2026-01", "END_MONTH": "2026-10", "ES_FRAC": "0.2",
             "OUT": "/kaggle/working/wf_monthly.csv"},
    ),
    # 同一逻辑的 5 seed 版（原 slug 推 v2 时报 409，换 slug 建新 kernel）
    "wfeval5": dict(
        slug="byh850515/btcusdt-h3-walkforward-5seed",
        title="BTCUSDT H3 Walk Forward 5seed",
        entry="morph.wf_eval",
        enable_internet=True,
        dataset_sources=[],
        env={"SEEDS": "42,43,44,45,46", "ROUNDS": "2000", "KLINE_FROM": "2025",
             "START_MONTH": "2026-01", "END_MONTH": "2026-10", "ES_FRAC": "0.2",
             "OUT": "/kaggle/working/wf_monthly.csv"},
    ),
    # W1 复跑：只加 ROLL_OUT=1，把逐日滚动胜率序列打进日志（不落 parquet、不下载产物）。
    # SEEDS=42 与 wf_eval.log 的原始 W1 保持一致，用于画覆盖 2026-01~10 的完整滚动曲线。
    "wfevalroll": dict(
        slug="byh850515/btcusdt-h3-walkforward-rolling",
        title="BTCUSDT H3 Walk Forward Rolling",
        entry="morph.wf_eval",
        enable_internet=True,
        dataset_sources=[],
        env={"SEEDS": "42", "ROUNDS": "2000", "KLINE_FROM": "2025",
             "START_MONTH": "2026-01", "END_MONTH": "2026-10", "ES_FRAC": "0.2",
             "ROLL_OUT": "1", "OUT": "/kaggle/working/wf_monthly_roll.csv"},
    ),
    # W2：长 horizon 的实盘口径（h=15/30/60，盘口 184 列原样，只改标签）。
    # 目的：在换特征源之前，先确认「盘口在长 horizon 是否彻底没戏」
    "wf15": dict(
        slug="byh850515/btcusdt-wf-h15", title="BTCUSDT WF H15", entry="morph.wf_eval",
        enable_internet=True, dataset_sources=[],
        env={"HORIZON": "15", "SEEDS": "42", "ROUNDS": "2000", "KLINE_FROM": "2025",
             "START_MONTH": "2026-01", "END_MONTH": "2026-10", "ES_FRAC": "0.2"},
    ),
    "wf30": dict(
        slug="byh850515/btcusdt-wf-h30", title="BTCUSDT WF H30", entry="morph.wf_eval",
        enable_internet=True, dataset_sources=[],
        env={"HORIZON": "30", "SEEDS": "42", "ROUNDS": "2000", "KLINE_FROM": "2025",
             "START_MONTH": "2026-01", "END_MONTH": "2026-10", "ES_FRAC": "0.2"},
    ),
    "wf60": dict(
        slug="byh850515/btcusdt-wf-h60", title="BTCUSDT WF H60", entry="morph.wf_eval",
        enable_internet=True, dataset_sources=[],
        env={"HORIZON": "60", "SEEDS": "42", "ROUNDS": "2000", "KLINE_FROM": "2025",
             "START_MONTH": "2026-01", "END_MONTH": "2026-10", "ES_FRAC": "0.2"},
    ),
    # L1：长周期（15/30/60）K 线特征 + walk-forward。长周期不用盘口（W2 已判死）。
    # 特征：多尺度动量/均线/波动结构/量价/位置/效率比/季节性/基差，约 55 列
    "klinelong": dict(
        slug="byh850515/btcusdt-kline-long-h15",
        title="BTCUSDT Kline Long H15",
        entry="morph.kline_long",
        enable_internet=True,
        dataset_sources=[],
        env={"HORIZON": "15", "SEEDS": "42", "ROUNDS": "2000", "KLINE_FROM": "2020",
             "WIN_DAYS": "730", "START_MONTH": "2026-01", "END_MONTH": "2026-10",
             "OUT": "/kaggle/working/kline_long_h15.csv"},
    ),
    # L1b：同一套长周期特征，5 seed（平滑概率）+ h=30（趋势更持久？）
    "klinelong5": dict(
        slug="byh850515/btcusdt-kline-long-h15-5seed",
        title="BTCUSDT Kline Long H15 5seed",
        entry="morph.kline_long",
        enable_internet=True,
        dataset_sources=[],
        env={"HORIZON": "15", "SEEDS": "42,43,44,45,46", "ROUNDS": "2000",
             "KLINE_FROM": "2020", "WIN_DAYS": "730",
             "START_MONTH": "2026-01", "END_MONTH": "2026-10",
             "OUT": "/kaggle/working/kline_long_h15_5s.csv"},
    ),
    "klinelong30": dict(
        slug="byh850515/btcusdt-kline-long-h30",
        title="BTCUSDT Kline Long H30",
        entry="morph.kline_long",
        enable_internet=True,
        dataset_sources=[],
        env={"HORIZON": "30", "SEEDS": "42", "ROUNDS": "2000",
             "KLINE_FROM": "2020", "WIN_DAYS": "730",
             "START_MONTH": "2026-01", "END_MONTH": "2026-10",
             "OUT": "/kaggle/working/kline_long_h30.csv"},
    ),
    "klinelong30s": dict(
        slug="byh850515/btcusdt-kline-long-h30-5seed",
        title="BTCUSDT Kline Long H30 5seed",
        entry="morph.kline_long",
        enable_internet=True,
        dataset_sources=[],
        env={"HORIZON": "30", "SEEDS": "42,43,44,45,46", "ROUNDS": "2000",
             "KLINE_FROM": "2020", "WIN_DAYS": "730",
             "START_MONTH": "2026-01", "END_MONTH": "2026-10",
             "OUT": "/kaggle/working/kline_long_h30_5s.csv"},
    ),
    # 赔率驱动的两个补测（2026-11-12）：小所 10m 0.828（平衡 54.71%）、ETH 30m 0.853（平衡 53.97%）。
    # 这两档此前只有 3 窗均值 / 完全未测，需用同一套 10 个月 walk-forward + 5 seed 口径确认。
    "klinelong10s": dict(
        slug="byh850515/btcusdt-kline-long-h10-5seed",
        title="BTCUSDT Kline Long H10 5seed",
        entry="morph.kline_long",
        enable_internet=True,
        dataset_sources=[],
        env={"HORIZON": "10", "SEEDS": "42,43,44,45,46", "ROUNDS": "2000",
             "KLINE_FROM": "2020", "WIN_DAYS": "730",
             "START_MONTH": "2026-01", "END_MONTH": "2026-10",
             "OUT": "/kaggle/working/kline_long_h10_5s.csv"},
    ),
    "klinelong_eth30s": dict(
        slug="byh850515/ethusdt-kline-long-h30-5seed",
        title="ETHUSDT Kline Long H30 5seed",
        entry="morph.kline_long",
        enable_internet=True,
        dataset_sources=[],
        env={"SYMBOL": "ETHUSDT", "HORIZON": "30", "SEEDS": "42,43,44,45,46",
             "ROUNDS": "2000", "KLINE_FROM": "2020", "WIN_DAYS": "730",
             "START_MONTH": "2026-01", "END_MONTH": "2026-10",
             "OUT": "/kaggle/working/kline_long_eth_h30_5s.csv"},
    ),
    # L2：h15 / h30 / 融合三配置，按实盘验收判据（总体≥63/65、滚动100期≥50、月度≥50）
    "klinefuse": dict(
        slug="byh850515/btcusdt-kline-fuse-h15-h30",
        title="BTCUSDT Kline Fuse H15 H30",
        entry="morph.kline_fuse",
        enable_internet=True,
        dataset_sources=[],
        env={"SEEDS": "42,43,44,45,46", "ROUNDS": "2000", "KLINE_FROM": "2020",
             "WIN_DAYS": "730", "START_MONTH": "2026-01", "END_MONTH": "2026-10",
             "TOPN": "15", "OUT": "/kaggle/working/kline_fuse.csv"},
    ),
    # 与 klinefuse 完全同代码，备用 slug。2026-11-10 实测：同一 kernel 推新版本
    # 必须等上一版本完全收尾（含 nbconvert），否则新版本会一直 QUEUED 僵尸化；
    # 这不是账号配额或单 kernel 次数限制，换 slug / 等收尾后重推即可启动。
    "klinefuse2": dict(
        slug="byh850515/btcusdt-kline-fuse-diag",
        title="BTCUSDT Kline Fuse Diag",
        entry="morph.kline_fuse",
        enable_internet=True,
        dataset_sources=[],
        env={"SEEDS": "42,43,44,45,46", "ROUNDS": "2000", "KLINE_FROM": "2020",
             "WIN_DAYS": "730", "START_MONTH": "2026-01", "END_MONTH": "2026-10",
             "TOPN": "15", "OUT": "/kaggle/working/kline_fuse.csv"},
    ),
    # 离线回放（不训练）：挂载 klinefuse2 的全历史 OOS 分数，测试因果滚动分位归一化
    # 能否在总单量不变下把单量分散到更多天、改善滚动100单与连亏。
    "klinenorm": dict(
        slug="byh850515/btcusdt-kline-norm-eval",
        title="BTCUSDT Kline Norm Eval",
        entry="morph.kline_norm_eval",
        enable_internet=True,
        dataset_sources=[],
        kernel_sources=["byh850515/btcusdt-kline-fuse-diag"],
        env={"SCORES_NAME": "kline_fuse_hist_scores.parquet",
             "TPD": "15", "ROLL_N": "100", "W_LIST": "0,7,30,90,365"},
    ),
    # 统计形态特征逐项验证（用户 2026-11-11）：因果性 / 跨段IC / FDR / 档差 / 负控。
    # 数据源 vision-base 2020+ 全量 1m（含现货+合约四列主动买卖量），不下载大产物，
    # 只输出小 csv + 日志。h=15 对应形态腿，h=3 对应产品周期。
    "statfeat15": dict(
        slug="byh850515/btcusdt-statfeat-validate-h15",
        title="BTCUSDT Statfeat Validate H15",
        entry="morph.statfeat.validate",
        enable_internet=True,
        dataset_sources=[],
        env={"KLINE_FROM": "2020",
             "STATFEAT_ARGS": "--horizon 15 --step 5 --out /kaggle/working/statfeat_h15.csv"},
    ),
    "statfeat3": dict(
        slug="byh850515/btcusdt-statfeat-validate-h3",
        title="BTCUSDT Statfeat Validate H3",
        entry="morph.statfeat.validate",
        enable_internet=True,
        dataset_sources=[],
        env={"KLINE_FROM": "2020",
             "STATFEAT_ARGS": "--horizon 3 --step 5 --out /kaggle/working/statfeat_h3.csv"},
    ),
    # 统计特征**画像归档**（不淘汰）：8 个 horizon(3m~1w) × 3 段 × 双标的，
    # 长 horizon 用非重叠抽样避免标签重叠伪相关，输出结构化档案 csv。
    "statprof_btc": dict(
        slug="byh850515/btcusdt-statfeat-profile",
        title="BTCUSDT Statfeat Profile",
        entry="morph.statfeat.profile",
        enable_internet=True,
        dataset_sources=[],
        env={"SYMBOL": "BTCUSDT", "KLINE_FROM": "2020",
             "STATPROF_ARGS": "--step 5 --out /kaggle/working/profile_btc.csv"},
    ),
    "statprof_eth": dict(
        slug="byh850515/ethusdt-statfeat-profile",
        title="ETHUSDT Statfeat Profile",
        entry="morph.statfeat.profile",
        enable_internet=True,
        dataset_sources=[],
        env={"SYMBOL": "ETHUSDT", "KLINE_FROM": "2020",
             "STATPROF_ARGS": "--step 5 --out /kaggle/working/profile_eth.csv"},
    ),
    # **无假设**条件分桶扫描（hypothesis generator）：16 个因果条件维度 × h=3/15/60，
    # 与 features.py 的区别是「不写公式、只枚举条件」，用来发现我们没想到的规律。
    # 口径：非重叠抽样(step=h) + BH-FDR + 3 段同号 + 强制报假阳性预期。
    # 只输出小 csv + 日志，不落大数据产物。显著数 ≤ 假阳性预期即判该 horizon 无规律。
    "statscan_btc": dict(
        slug="byh850515/btcusdt-statfeat-scan",
        title="BTCUSDT Statfeat Scan",
        entry="morph.statfeat.scan",
        enable_internet=True,
        dataset_sources=[],
        env={"SYMBOL": "BTCUSDT", "KLINE_FROM": "2020",
             "SCAN_OUT": "/kaggle/working/scan_btc.csv"},
    ),
    "statscan_eth": dict(
        slug="byh850515/ethusdt-statfeat-scan",
        title="ETHUSDT Statfeat Scan",
        entry="morph.statfeat.scan",
        enable_internet=True,
        dataset_sources=[],
        env={"SYMBOL": "ETHUSDT", "KLINE_FROM": "2020",
             "SCAN_OUT": "/kaggle/working/scan_eth.csv"},
    ),
    # 统计**知识库**（不设交易准入门）：6 个目标（收益/绝对收益/波幅/回撤/量能/资金流）
    # × 5 个 horizon（3m~1d）× 16 个条件维度，全量归档成可检索长表。
    # 与 scan 的区别：scan 只问「涨跌概率」，本项把「什么时候波动大/量大/回撤深」也归档，
    # 结论对二元合约无用也保留（用户 2026-11-11 明确口径）。
    "statlib_btc": dict(
        slug="byh850515/btcusdt-statfeat-library",
        title="BTCUSDT Statfeat Library",
        entry="morph.statfeat.library",
        enable_internet=True,
        dataset_sources=[],
        env={"SYMBOL": "BTCUSDT", "KLINE_FROM": "2020",
             "LIB_OUT": "/kaggle/working/library_btc.csv"},
    ),
    "statlib_eth": dict(
        slug="byh850515/ethusdt-statfeat-library",
        title="ETHUSDT Statfeat Library",
        entry="morph.statfeat.library",
        enable_internet=True,
        dataset_sources=[],
        env={"SYMBOL": "ETHUSDT", "KLINE_FROM": "2020",
             "LIB_OUT": "/kaggle/working/library_eth.csv"},
    ),
    # 扫描发现 × 现役信号的**正交性交叉检验**（零训练成本，不重训模型）：
    # 把 scan.py 找到的条件桶套到 h3_signal 已触发的单上，看是否还有分化；
    # 并按 |p-0.5| 三分位分层，判断条件桶是真的新信息还是置信度的代理。
    # 需要挂 h3gate kernel 的 output（h3_gate_pred.parquet）。
    "statxcheck": dict(
        slug="byh850515/btcusdt-statfeat-xcheck",
        title="BTCUSDT Statfeat Xcheck",
        entry="morph.statfeat.xcheck",
        enable_internet=True,
        dataset_sources=[],
        kernel_sources=["byh850515/btcusdt-h3-gate"],
        env={"KLINE_FROM": "2025",
             "PRED": "/kaggle/input/btcusdt-h3-gate/h3_gate_pred.parquet"},
    ),
    # 仓位分配：固定每天 15 单不变，只比「每单下注多少」（等权 / |q−0.5| / Kelly）。
    # 先出十等分校准曲线，再按预注册判据（夏普≥1.2×、回撤不更深、有效注数≥5）判 GO/STOP。
    # 注意：数据集 byh850515/btcusdt-orderbook-features 已 403 不可挂载（2026-11-13 实测），
    # 故不挂 dataset，改由脚本内从 bn_hisdata 的 merged-features Release 拉 BTCUSDT.parquet。
    "weight": dict(
        slug="byh850515/btcusdt-weight-eval",
        title="BTCUSDT Weight Eval",
        entry="orderbook.weight_eval",
        enable_internet=True,
        dataset_sources=[],
        env={"SEEDS": "42,43,44,45,46", "ROUNDS": "2000", "TPD": "15",
             "MIN_NEFF": "5", "SHARPE_GAIN": "1.2", "BOOT": "2000"},
    ),
    # D0：ETH 为什么比 BTC 差（机制诊断）。同窗口同特征同口径，1 seed；
    # 只把统计打到 stdout + 一个小 json，**不落大数据产物**
    "ethdiag": dict(
        slug="byh850515/eth-vs-btc-h3-diag",
        title="ETH vs BTC H3 Diagnostic",
        entry="morph.eth_diag",
        enable_internet=True,
        dataset_sources=[],
        kernel_sources=["byh850515/ethusdt-vol8-ob-build"],
        env={"SEED": "42", "ROUNDS": "400", "TARGET_TPD": "15",
             "OUT": "/kaggle/working/eth_diag.json"},
    ),
}


def stage(project, overrides=None):
    cfg = PROJECTS[project]
    d = tempfile.mkdtemp(prefix=f"kaggle_{project}_")
    # Kaggle script kernel 只上传 code_file 一个文件，旁路文件(entry.txt)不会被传；
    # 因此把入口直接注入 bootstrap 副本首行（环境变量），确保入口随 code_file 一起上传。
    boot = open(BOOTSTRAP, encoding="utf-8").read()
    lines = ["import os as _os", f'_os.environ["KAGGLE_ENTRY"] = {cfg["entry"]!r}']
    env = dict(cfg.get("env", {}))
    env.update(overrides or {})      # --set K=V：临时覆盖，不用改配置就能跑 smoke
    # 密钥只从本地 .codebuddy/settings.json 读（该目录被 .gitignore 忽略），绝不写进仓库
    for k, v in _secrets(cfg.get("secrets", [])).items():
        env[k] = v
    for k, v in env.items():
        lines.append(f"_os.environ[{k!r}] = {v!r}")
    inject = "\n".join(lines) + "\n"
    with open(os.path.join(d, "kaggle_bootstrap.py"), "w", encoding="utf-8") as f:
        f.write(inject + boot)
    meta = {
        "id": cfg["slug"],
        "title": cfg["title"],
        "code_file": "kaggle_bootstrap.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        # GPU 策略（Kaggle工作流设计.md §9.6）：默认 CPU；仅神经网络/时序模型项目
        # 在 PROJECTS 里写 gpu=True 才开，其余（LightGBM/faiss/数据处理）一律 CPU。
        "enable_gpu": bool(cfg.get("gpu", False)),
        "enable_internet": cfg["enable_internet"],
        "dataset_sources": cfg["dataset_sources"],
        "competition_sources": [],
        "kernel_sources": cfg.get("kernel_sources", []),
    }
    with open(os.path.join(d, "kernel-metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return d


def _poll(slug):
    print(f"[poll] {slug}")
    while True:
        out = subprocess.run(["kaggle", "kernels", "status", slug],
                             capture_output=True, text=True).stdout
        line = out.strip().splitlines()[-1] if out.strip() else ""
        print("   ", line)
        low = line.lower()
        if "complete" in low or "error" in low:
            return "error" not in low
        time.sleep(30)


def _decode_logs(raw):
    """kaggle kernels logs 返回 JSON 数组 [{stream_name,time,data},...]，
    也可能带前导逗号或逐行对象，统一解码为纯文本。"""
    import json as _json
    raw = (raw or "").strip()
    if not raw:
        return ""
    for candidate in (raw, raw.lstrip(",")):
        try:
            arr = _json.loads(candidate)
            if isinstance(arr, list):
                return "".join(i.get("data", "") for i in arr)
        except Exception:
            pass
    parts = []
    for line in raw.splitlines():
        line = line.strip().lstrip(",").rstrip(",")
        if not line:
            continue
        try:
            obj = _json.loads(line)
            if isinstance(obj, dict) and "data" in obj:
                parts.append(obj["data"])
        except Exception:
            pass
    return "".join(parts) if parts else raw


def _logs(slug, parse=False):
    # Windows 控制台默认 GBK，kaggle CLI 输出是 UTF-8：不解码会 UnicodeDecodeError
    out = subprocess.run(["kaggle", "kernels", "logs", slug],
                         capture_output=True, encoding="utf-8", errors="replace").stdout
    text = _decode_logs(out or "")
    print(text)
    if parse:
        print("=== 关键指标 ===")
        for ln in text.splitlines():
            s = ln.strip()
            if not s:
                continue
            if any(k in s for k in ("comb@", "基准", "准确率", "AUC", "ACC",
                                    "选择性", "覆盖率", "best_iteration", "差值", "结论",
                                    "检验", "假阳性", "触发单", "结论", "桶=")):
                print("  ", s)


def run(project, push, wait, logs, parse, overrides=None):
    cfg = PROJECTS[project]
    d = stage(project, overrides)
    print(f"[stage] {d}")
    if push:
        subprocess.run(["kaggle", "kernels", "push", "-p", d], check=True)
    if wait:
        ok = _poll(cfg["slug"])
        if not ok:
            print("[!] kernel 运行出错，详见 logs")
    if logs:
        _logs(cfg["slug"], parse=parse)


def main():
    ap = argparse.ArgumentParser(description="Kaggle 项目编排器")
    ap.add_argument("--project", required=True, choices=list(PROJECTS))
    ap.add_argument("--no-push", action="store_true", help="只生成 kernel 目录，不 push")
    ap.add_argument("--no-wait", action="store_true", help="push 后不轮询")
    ap.add_argument("--no-logs", action="store_true", help="不拉 logs")
    ap.add_argument("--parse", action="store_true", help="从 logs 中解析关键指标")
    ap.add_argument("--set", action="append", default=[], metavar="K=V",
                    help="临时覆盖 env（可多次），用于 smoke 而不用改配置")
    a = ap.parse_args()
    overrides = {}
    for kv in a.set:
        if "=" not in kv:
            ap.error(f"--set 需要 K=V，收到 {kv!r}")
        k, v = kv.split("=", 1)
        overrides[k.strip()] = v.strip()
    run(a.project,
        push=not a.no_push,
        wait=not a.no_wait,
        logs=not a.no_logs,
        parse=a.parse,
        overrides=overrides)


if __name__ == "__main__":
    main()
