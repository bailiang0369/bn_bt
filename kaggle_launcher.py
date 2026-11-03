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
