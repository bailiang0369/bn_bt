"""Kaggle kernel: 下载 ETH/BTC 2020+ 1m K 线 → 存 parquet → 自动注册为 dataset。

全程在 Kaggle 容器内执行，文件从 /kaggle/working 直接上传为 dataset，
不经过本机磁盘。
"""
import sys
import io
import datetime as _dt
import subprocess

print(">>> 安装依赖: requests polars")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "requests", "polars"], check=False)

import requests
import polars as pl
import zipfile
import json


def download_klines(symbol, start_date="2020-01-01", end_date="2026-09-30"):
    BASE = "https://data.binance.vision/data/futures/um/monthly/klines"
    start = _dt.date.fromisoformat(start_date)
    end = _dt.date.fromisoformat(end_date)
    y, m = start.year, start.month
    cols = ["open_time", "open", "high", "low", "close", "volume", "close_time",
            "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]
    frames = []
    while (y, m) <= (end.year, end.month):
        period = f"{y:04d}-{m:02d}"
        url = f"{BASE}/{symbol}/1m/{symbol}-1m-{period}.zip"
        try:
            r = requests.get(url, timeout=60)
            if r.status_code != 200:
                print(f"  [dl] {symbol} {period} 跳过 (HTTP {r.status_code})")
            else:
                z = zipfile.ZipFile(io.BytesIO(r.content))
                csv_name = [n for n in z.namelist() if n.endswith(".csv")][0]
                raw = io.BytesIO(z.read(csv_name))
                first = raw.readline().decode("utf-8", "replace")
                raw.seek(0)
                if first.split(",")[0].strip() == "open_time":
                    df = pl.read_csv(raw, has_header=True)
                else:
                    df = pl.read_csv(raw, has_header=False, new_columns=cols)
                df = df.select("close_time", "open", "high", "low", "close", "volume", "taker_buy_volume")
                frames.append(df)
                print(f"  [dl] {symbol} {period} ok ({df.height} 行)")
        except Exception as e:
            print(f"  [dl] {symbol} {period} 出错: {e}")
        m += 1
        if m > 12:
            m, y = 1, y + 1
    total = sum(f.height for f in frames)
    print(f"[dl] {symbol} 合计 {total} 行")
    return pl.concat(frames, how="vertical_relaxed").sort("close_time")


def main():
    for sym in ["ETHUSDT", "BTCUSDT"]:
        print(f"\n>>> 下载 {sym} 2020+ 1m K 线")
        df = download_klines(sym)
        key = sym.lower()
        out = f"/kaggle/working/{key}_1m.parquet"
        df.write_parquet(out)
        print(f">>> 已写出 {out} ({df.height} 行)")

    meta = {
        "title": "ETH BTC 1m 2020+ Klines (Binance Futures UM)",
        "id": "byh850515/eth-btc-1m-2020",
        "licenses": [{"name": "CC0-1.0"}],
    }
    with open("/kaggle/working/dataset-metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(">>> 已写出 dataset-metadata.json")

    # 在 Kaggle 容器内直接创建 dataset（文件从 /kaggle/working 上传，不经本机）
    print(">>> 创建 dataset: kaggle datasets create -p /kaggle/working")
    r = subprocess.run(["kaggle", "datasets", "create", "-p", "/kaggle/working"],
                       capture_output=True, text=True)
    print(r.stdout)
    print(r.stderr)
    if r.returncode != 0 and "already exists" in (r.stderr or ""):
        print(">>> dataset 已存在，改用 version 更新")
        r2 = subprocess.run(["kaggle", "datasets", "version", "-p", "/kaggle/working",
                             "-m", "refresh ETH/BTC 1m parquet"],
                            capture_output=True, text=True)
        print(r2.stdout)
        print(r2.stderr)
    print(">>> DONE: dataset 创建/更新完成")


if __name__ == "__main__":
    main()
