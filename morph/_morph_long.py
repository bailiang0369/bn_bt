"""把 _morph.py 的数据源换成 bn_hisdata 下载器产出的 2020+ 月度 K 线。

下载器输出：bn_hisdata/binance_vision/data/um/monthly/klines/BTCUSDT/1m/*.parquet
（列：open_time/open/high/low/close/close_time/count/buy_vol/sell_vol）。
本脚本只挑 OHLC + close_time，复用 _morph 的全部聚类/报告逻辑。
"""
import polars as pl
from pathlib import Path
import _morph as M


def load_long():
    root = Path(__file__).resolve().parent.parent  # workspace 根
    base = root / "bn_hisdata/binance_vision/data/um/monthly/klines/BTCUSDT/1m"
    files = sorted(base.glob("*.parquet"))
    if not files:
        raise SystemExit(f"[load_long] 在 {base} 没找到 K 线 parquet，先跑下载器")
    print(f"[load_long] {len(files)} 个月文件: {files[0].name} .. {files[-1].name}")
    df = pl.concat([pl.read_parquet(f) for f in files], how="vertical_relaxed")
    return df.select("close_time", "open", "high", "low", "close").sort("close_time")


M.load_df = load_long

if __name__ == "__main__":
    M.main(15)
