#!/usr/bin/env python3
"""ETH 数据集 schema 探针（不训练）：确认列名 / 类型 / 行数 / 时间范围，供 h3_eth.py 定特征集。

本地自检：ETH_PARQUET 指向任意 parquet（含合成的），输出格式与 Kaggle 上一致。
    python -m morph.eth_probe
"""
from __future__ import annotations

import glob
import os
from collections import Counter

import polars as pl


def main() -> None:
    p = os.environ.get("ETH_PARQUET", "")
    if not p:
        hits = sorted(glob.glob("/kaggle/input/**/ETHUSDT.parquet", recursive=True))
        if not hits:
            raise SystemExit("找不到 /kaggle/input/**/ETHUSDT.parquet")
        p = hits[0]
    print(f"[probe] path={p}", flush=True)

    lf = pl.scan_parquet(p)
    schema = lf.collect_schema()
    names = sorted(schema.names())
    print(f"[probe] ncols={len(names)}", flush=True)
    print("[probe] columns=" + ",".join(names), flush=True)
    print("[probe] prefix=" + ",".join(f"{k}:{v}" for k, v in
                                       sorted(Counter(c.split("_")[0] for c in names).items())),
          flush=True)
    print("[probe] dtypes=" + ",".join(f"{c}:{schema[c]}" for c in names[:12]), flush=True)

    keep = [c for c in ("close_time", "close", "target", "high", "low", "open")
            if c in schema]
    d = lf.select(keep).collect()
    print(f"[probe] nrows={d.height}", flush=True)
    if "close_time" in d.columns:
        ct = d["close_time"]
        print(f"[probe] close_time {ct.min()} ~ {ct.max()} "
              f"(null={ct.null_count()})", flush=True)
    for c in keep:
        if c != "close_time":
            print(f"[probe] {c}: null={d[c].null_count()}", flush=True)
    if "close" in d.columns:
        print(f"[probe] close min/max={d['close'].min()}/{d['close'].max()}", flush=True)
    print("PROBE DONE", flush=True)


if __name__ == "__main__":
    main()
