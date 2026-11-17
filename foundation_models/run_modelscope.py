#!/usr/bin/env python3
"""魔搭 ModelScope Notebook 一键运行器（TimesFM / Kronos 探针）。

在魔搭 Notebook（JupyterLab）里：
  1. 新建一个终端，cd /mnt/workspace
  2. git clone https://github.com/bailiang0369/bn_bt.git
  3. cd bn_bt
  4. python -m foundation_models.run_modelscope --model timesfm   （或 --model kronos）

本脚本做的事：
  - 设置 HF_ENDPOINT=https://hf-mirror.com（国内下载 TimesFM/Kronos 权重必需）
  - 只装轻量依赖（不碰魔搭预装的 torch/cuda，避免版本冲突）
  - 数据从 GitHub Release 下载，失败时自动试 ghproxy 镜像
  - 调用对应探针脚本，结果落到 /mnt/workspace/bn_bt/artifacts/

魔搭环境：A10 24G，/mnt/workspace 持久化；空闲 1h 自动关、单次 8h 强制停。
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

os.environ["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
ROOT = Path(__file__).resolve().parent.parent


def log(msg: str) -> None:
    print(msg, flush=True)


def pip(*pkgs: str) -> None:
    # 魔搭在国内，强制用阿里云 PyPI 镜像，避免卡在 pypi.org
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "--index-url", "https://mirrors.aliyun.com/pypi/simple/",
                    *pkgs], check=True)


def ensure_data(symbol: str, years: list[str]) -> None:
    """下载 vision-base 年度 K 线到 /tmp（魔搭持久化盘留代码/结果，数据临时即可）。"""
    import requests
    cache = ROOT / "data" / "vision"
    cache.mkdir(parents=True, exist_ok=True)
    urls = [
        f"https://github.com/bailiang0369/bn_hisdata/releases/download/vision-base",
        f"https://ghfast.top/https://github.com/bailiang0369/bn_hisdata/releases/download/vision-base",
    ]
    for y in years:
        dst = cache / f"{symbol}_1m_klines_{y}.parquet"
        if dst.exists() and dst.stat().st_size > 1e6:
            log(f"[data] {dst.name} 已存在")
            continue
        for base in urls:
            url = f"{base}/{symbol}_1m_klines_{y}.parquet"
            try:
                with requests.get(url, stream=True, timeout=120) as r:
                    r.raise_for_status()
                    part = dst.with_suffix(".part")
                    with open(part, "wb") as f:
                        for chunk in r.iter_content(1 << 20):
                            if chunk:
                                f.write(chunk)
                part.replace(dst)
                log(f"[data] {dst.name} {dst.stat().st_size / 1e6:.1f}MB")
                break
            except Exception as e:  # noqa: BLE001
                log(f"[data] {y} via {base[:40]}: {type(e).__name__}")
        else:
            raise SystemExit(f"数据下载失败: {y}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("timesfm", "kronos"), default="timesfm")
    ap.add_argument("--anchors", type=int, default=3000)
    ap.add_argument("--horizons", default="3,10")
    ap.add_argument("--bar", type=int, default=1)
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()

    log(f"HF_ENDPOINT={os.environ['HF_ENDPOINT']}")
    # 轻量依赖；torch 用魔搭预装版，绝不 pip install torch / timesfm[torch]
    pip("numpy>=1.26.4", "huggingface_hub>=0.23.0", "safetensors>=0.5.3",
        "scipy", "scikit-learn", "einshape", "jaxtyping",
        "polars", "pandas", "requests")
    if args.model == "timesfm":
        pip("timesfm==2.0.2")            # 不带 [torch] extra，保留预装 torch/cuda
    else:
        pip("einops==0.8.1", "pandas", "tqdm")

    if not args.skip_download:
        ensure_data("BTCUSDT", [str(y) for y in range(2020, 2027)])
    else:
        (ROOT / "data" / "vision").mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env.update({"BAR": str(args.bar), "N_ANCHORS": str(args.anchors),
                "HORIZONS": args.horizons, "SKIP_TF_INSTALL": "1",
                "DATA_DIR": str(ROOT / "data" / "vision"),
                "OUT": str(ROOT / "artifacts" / f"{args.model}_ms_b{args.bar}.parquet")})
    entry = ("foundation_models.timesfm_probe" if args.model == "timesfm"
             else "foundation_models.kronos_probe")
    log(f"运行 {entry} anchors={args.anchors} bar={args.bar}m horizons={args.horizons}")
    subprocess.run([sys.executable, "-m", entry], cwd=str(ROOT), env=env, check=True)
    log(f"完成，结果: {env['OUT']}")


if __name__ == "__main__":
    main()
