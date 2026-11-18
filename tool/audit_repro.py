#!/usr/bin/env python3
"""本地复核包装器：用本地 parquet 跑 morph.h10_signal / morph.kline_long，不改生产代码。

用法：
    python tool/audit_repro.py '{"HORIZON":"10", ...}'
环境变量 AUDIT_DATA 指定本地 vision-base parquet 目录。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CACHE = Path(os.environ.get("AUDIT_DATA", r"e:/codebuddy_wrokspace/bn_hisdata/data/vision"))
from morph import ob_kline as ok  # noqa: E402

_files = {p.name: p for p in CACHE.glob("*_1m_klines_*.parquet")}
if not _files:
    raise SystemExit(f"本地没有 K 线 parquet: {CACHE}")
ok._assets = lambda tag: {n: {"size": p.stat().st_size, "url": "", "id": 0}
                          for n, p in _files.items()}
ok.fetch = lambda tag, name: _files[name]

if len(sys.argv) > 1:
    os.environ.update(json.loads(sys.argv[1]))

entry = os.environ.get("AUDIT_ENTRY", "morph.h10_signal")
mod = __import__(entry, fromlist=["main"])
mod.main()
