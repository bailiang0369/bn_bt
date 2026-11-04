"""Kaggle kernel: 生成 2020+ 全量 BTCUSDT um 1m K 线并注册为 Kaggle Dataset。

数据源：Binance Vision 官方月度 zip（um/BTCUSDT/1m），下载与清洗复用 bn_hisdata
的 `binance_vision/download_binance_vision.py`（buy_vol/sell_vol 拆分、close_time
+1ms 规整、断点续传、校验和）。

输出（进入 dataset 的只有这两样，中间态全在 /tmp）：
  /kaggle/working/ds/btcusdt_1m_full.parquet
      close_time/open/high/low/close/volume/taker_buy_volume/count
  /kaggle/working/ds/dataset-metadata.json
      id = byh850515/btcusdt-1m-full-2020

产物供 morph/kline_boost.py 直接挂载使用。
"""
import glob
import json
import os
import subprocess
import sys

WORK = "/kaggle/working"
DS_DIR = os.path.join(WORK, "ds")
TMP = "/tmp"
HIS = os.path.join(TMP, "bn_hisdata")
DATA = os.path.join(TMP, "vision")
OUT = os.path.join(DS_DIR, "btcusdt_1m_full.parquet")
DS_ID = "byh850515/btcusdt-1m-full-2020"
START, END = "2020-01-01", "2026-10-01"
SYMBOL = "BTCUSDT"


def log(m):
    print(m, flush=True)


def run(args):
    log("+ " + " ".join(args))
    rc = subprocess.run(args, check=False).returncode
    if rc != 0:
        log(f"  !! rc={rc}")
    return rc


def main():
    if not os.path.isdir(HIS):
        subprocess.run(["git", "clone", "--depth=1",
                        "https://github.com/bailiang0369/bn_hisdata.git", HIS], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "polars", "requests"],
                   check=True)

    run([sys.executable, os.path.join(HIS, "binance_vision", "download_binance_vision.py"),
         "--market", "um", "--symbol", SYMBOL, "--category", "klines",
         "--interval", "1m", "--granularity", "monthly",
         "--start-date", START, "--end-date", END,
         "--dest-root", DATA, "-j", "8"])

    import polars as pl

    files = sorted(glob.glob(os.path.join(DATA, "**", "*.parquet"), recursive=True))
    files = [f for f in files if "/klines/" in f.replace("\\", "/")]
    log(f"[files] {len(files)} 个月度 parquet")
    if not files:
        raise SystemExit("没有下载到任何 klines 文件")

    lf = pl.scan_parquet(files).select(
        pl.col("close_time").cast(pl.Int64),
        pl.col("open").cast(pl.Float32),
        pl.col("high").cast(pl.Float32),
        pl.col("low").cast(pl.Float32),
        pl.col("close").cast(pl.Float32),
        (pl.col("buy_vol") + pl.col("sell_vol")).cast(pl.Float32).alias("volume"),
        pl.col("buy_vol").cast(pl.Float32).alias("taker_buy_volume"),
        pl.col("count").cast(pl.Int64),
    ).sort("close_time")

    os.makedirs(DS_DIR, exist_ok=True)
    lf.sink_parquet(OUT, compression="zstd")

    d = pl.read_parquet(OUT, columns=["close_time", "close", "volume", "taker_buy_volume"])
    log(f"[out] rows={d.height} {int(d['close_time'][0])}~{int(d['close_time'][-1])}")
    log(f"[out] close NaN={d['close'].null_count()} volume<0={(d['volume'] < 0).sum()}")
    per_month = (d.with_columns(pl.col("close_time").cast(pl.Datetime("ms"))
                                .dt.strftime("%Y-%m").alias("ym"))
                 .group_by("ym").agg(pl.len().alias("n")).sort("ym"))
    log(f"[out] 月份数={per_month.height} 行数范围="
        f"{per_month['n'].min()}~{per_month['n'].max()}")
    short = per_month.filter(pl.col("n") < 40_000)
    if short.height:
        log(f"[out] 行数偏少的月份: {short.to_dicts()}")
    log(f"[out] size={os.path.getsize(OUT) / 1e6:.1f} MB")

    meta = {
        "title": "BTCUSDT 1m 2020+ Full Klines (volume + taker buy)",
        "id": DS_ID,
        "licenses": [{"name": "CC0-1.0"}],
    }
    with open(os.path.join(DS_DIR, "dataset-metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    r = subprocess.run(["kaggle", "datasets", "create", "-p", DS_DIR],
                       capture_output=True, text=True)
    log(r.stdout)
    log(r.stderr)
    if r.returncode != 0:
        log(">>> create 失败，改用 version 更新")
        r = subprocess.run(["kaggle", "datasets", "version", "-p", DS_DIR,
                            "-m", "refresh 2020+ full klines"], capture_output=True, text=True)
        log(r.stdout)
        log(r.stderr)
    log("DONE")


if __name__ == "__main__":
    main()
