"""Kaggle kernel 的 code_file（极小且稳定）。

运行时：从 GitHub clone 整个项目 → 按 requirements.txt 装依赖 → 按 entry.txt（由 launcher 写入）
导入对应入口并调 main()。这样改代码只改 GitHub 仓库，kernel 永远重拉最新，不再把整个项目
inline 进单个 py 反复推。
"""
import os
import sys
import subprocess
import importlib

REPO = "https://github.com/bailiang0369/bn_bt"
# 克隆到 /tmp 而不是 /kaggle/working：/kaggle/working 下的所有东西都会进 kernel output，
# 之前仓库整个被当成产物，`kaggle kernels output` 会重下几十 MB 项目代码。
ROOT = "/tmp/bn_bt"


def _entry():
    # launcher 推 kernel 时在同目录写入 entry.txt 指定入口模块；
    # 本地直接跑则回退到默认（形态聚类）或环境变量 KAGGLE_ENTRY。
    here = os.path.dirname(os.path.abspath(__file__))
    p = os.path.join(here, "entry.txt")
    if os.path.exists(p):
        return open(p, encoding="utf-8").read().strip()
    return os.environ.get("KAGGLE_ENTRY", "morph.kaggle_entry")


def main():
    if not os.path.exists(ROOT):
        # 公开库直连 clone；若以后设为私有，用 Kaggle Secret 注入 GITHUB_TOKEN 拼进 URL：
        # repo = f"https://{os.environ['GITHUB_TOKEN']}:x-oauth-basic@github.com/bailiang0369/bn_bt"
        subprocess.run(["git", "clone", "--depth=1", REPO, ROOT], check=True)
    else:
        # 复用已有克隆时拉最新
        subprocess.run(["git", "-C", ROOT, "pull", "--depth=1"], check=True)

    sys.path.insert(0, ROOT)
    req = os.path.join(ROOT, "requirements.txt")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", req], check=True)

    importlib.import_module(_entry()).main()


if __name__ == "__main__":
    main()
