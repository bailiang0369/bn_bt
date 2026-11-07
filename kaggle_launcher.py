"""本地 Kaggle 编排器（不进 kernel，仅本地用）。

按 --project 在临时目录生成 kernel（拷贝 kaggle_bootstrap.py + 写 entry.txt + 写 kernel-metadata.json），
然后 kaggle kernels push → 轮询 status → 拉 logs（可选解析关键指标）。

约定：
- 默认 enable_gpu=false，只用 CPU（faiss-cpu / lightgbm CPU），绝不自动开 GPU；
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
}


def stage(project):
    cfg = PROJECTS[project]
    d = tempfile.mkdtemp(prefix=f"kaggle_{project}_")
    # Kaggle script kernel 只上传 code_file 一个文件，旁路文件(entry.txt)不会被传；
    # 因此把入口直接注入 bootstrap 副本首行（环境变量），确保入口随 code_file 一起上传。
    boot = open(BOOTSTRAP, encoding="utf-8").read()
    lines = ["import os as _os", f'_os.environ["KAGGLE_ENTRY"] = {cfg["entry"]!r}']
    for k, v in cfg.get("env", {}).items():
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
        "enable_gpu": False,
        "enable_internet": cfg["enable_internet"],
        "dataset_sources": cfg["dataset_sources"],
        "competition_sources": [],
        "kernel_sources": [],
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
    raw = raw.strip()
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
    out = subprocess.run(["kaggle", "kernels", "logs", slug],
                         capture_output=True, text=True).stdout
    text = _decode_logs(out)
    print(text)
    if parse:
        print("=== 关键指标 ===")
        for ln in text.splitlines():
            s = ln.strip()
            if not s:
                continue
            if any(k in s for k in ("comb@", "基准", "准确率", "AUC", "ACC",
                                    "选择性", "覆盖率", "best_iteration", "差值", "结论")):
                print("  ", s)


def run(project, push, wait, logs, parse):
    cfg = PROJECTS[project]
    d = stage(project)
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
    a = ap.parse_args()
    run(a.project,
        push=not a.no_push,
        wait=not a.no_wait,
        logs=not a.no_logs,
        parse=a.parse)


if __name__ == "__main__":
    main()
