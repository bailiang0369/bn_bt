#!/usr/bin/env python3
"""Kaggle 入口：跑 bn_hisdata 的 ETH vol8 数据生产脚本（7h 长任务）。

`kaggle_bootstrap.py` 只 clone `bn_bt`，而 `run_eth.py` 属于另一个仓库；这里补一次
clone 后以 `__main__` 方式执行它（脚本自带 ensure_repo / 依赖安装 / 凭据读取）。

env（由 PROJECTS["ethbuild"] 注入）：
    MODE=build|smoke   OB_SYMBOL=ETHUSDT   START/END   MAX_DAYS
    CHD_API_KEY        由 launcher 的 secrets 从本地 .codebuddy/settings.json 注入
"""
from __future__ import annotations

import runpy
import subprocess
from pathlib import Path

REPO = "https://github.com/bailiang0369/bn_hisdata"
ROOT = Path("/tmp/bn_hisdata")
ENTRY = ROOT / "features/kaggle/run_eth.py"


def main() -> None:
    if not (ROOT / ".git").exists():
        subprocess.run(["git", "clone", "--depth=1", REPO, str(ROOT)], check=True)
    else:
        subprocess.run(["git", "-C", str(ROOT), "pull", "--depth=1"], check=True)
    if not ENTRY.exists():
        raise SystemExit(f"缺少入口 {ENTRY}")
    runpy.run_path(str(ENTRY), run_name="__main__")


if __name__ == "__main__":
    main()
