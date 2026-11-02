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
        enable_internet=False,
        dataset_sources=["byh850515/btcusdt-orderbook-features"],
    ),
}


def stage(project):
    cfg = PROJECTS[project]
    d = tempfile.mkdtemp(prefix=f"kaggle_{project}_")
    shutil.copy(BOOTSTRAP, os.path.join(d, "kaggle_bootstrap.py"))
    with open(os.path.join(d, "entry.txt"), "w", encoding="utf-8") as f:
        f.write(cfg["entry"])
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


def _logs(slug, parse=False):
    out = subprocess.run(["kaggle", "kernels", "logs", slug],
                         capture_output=True, text=True).stdout
    print(out)
    if parse:
        print("=== 关键指标 ===")
        for ln in out.splitlines():
            if any(k in ln for k in ("ACC=", "AUC=", "选择性", "覆盖率", "best_iteration")):
                print("  ", ln.strip())


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
