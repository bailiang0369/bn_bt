"""
Download Binance public market data from https://data.binance.vision (S3).

Supports
  markets   : spot, futures/um  (--market both downloads both)
  symbols   : BTCUSDT / ETHUSDT / all / comma-separated list
              (e.g. BTCUSDT,ETHUSDT,SOLUSDT)
  categories: klines (needs --interval), metrics (um only), aggTrades
              (--category all = klines + metrics; comma lists supported)
  granularity: daily  -> one zip per day   (good for testing / a few days)
               monthly -> one zip per month (much faster for bulk backfills)
  workers   : parallel download & processing via --workers / -j (default 4)

Disk layout (under <module>/data by default):
  data/{market}/{granularity}/{category}/{symbol}/{interval?}/{symbol}-{type}-{period}.{parquet|csv}
  The source .zip is downloaded to the same folder, processed into this cleaned
  parquet (default) or CSV (--format csv), and then deleted (the zip is only a
  transient intermediate). Resume is keyed on the cleaned output file, so
  re-running is a safe no-op once it exists.

NOTE: historical Binance CSVs are inconsistent -- some periods have a header
row, others don't. process() detects this from the first token and always
emits a clean, typed output, then the source .zip is deleted. Format facts
learned from REAL files (BTCUSDT, 2026-10, both markets):
  * aggTrades has TWO real layouts:
      - um  : 7 columns WITH header (agg_trade_id..is_buyer_maker, NO
              is_best_match), transact_time in MILLIseconds;
      - spot: 8 columns HEADER-LESS (the 8th is is_best_match), transact_time
              in MICROseconds (~1.79e15), booleans spelled "True"/"False".
    process() resolves column names from the header when present, else from
    the canonical list truncated to the actual field count, filters the
    schema overrides / casts to the columns that exist, and normalizes
    transact_time to ms. Hard-coding 8 columns breaks on um files; skipping
    the unit normalization silently corrupts every spot aggTrades timestamp.
  * metrics' create_time is a datetime string
    ("2025-06-01 00:00:00"), not an ms epoch -- it is converted so every
    timestamp in the output is Int64 ms.
klines output keeps open_time,
close_time + OHLC + count + buy_vol/sell_vol and drops volume /
quote_volume / taker_buy_quote_volume / ignore, where the taker-buy volume
buy_vol = taker_buy_volume and the taker-sell volume sell_vol = volume -
taker_buy_volume. close_time (+1ms) aligns to orderbook snapshots
(== next candle open_time).

Correctness notes (from the 2026-10 review of the parallel rewrite):
  * schema_overrides pins every column's dtype at parse time (fast path), but
    klines' volume / taker_buy_volume are deliberately parsed as Float64 even
    when the output dtype is Float32: sell_vol = volume - taker_buy_volume
    must be computed at full precision BEFORE the cast. f32 subtraction of two
    large, close numbers has an absolute error up to ~1e-4, which is a large
    RELATIVE error on quiet minutes where sell_vol is small -- exactly the
    imbalance features this data is collected for.
  * aggTrades' output dtypes are additionally pinned with explicit casts after
    the scan, so they never silently depend on polars' schema_overrides
    behavior across versions.
  * No silent failures: worker exceptions are caught, logged and counted as
    failed, and the process exits non-zero when anything failed, so an
    automated run can never look "green" while files are missing.
  * Downloads and outputs land via ".part" temp files + atomic rename, and
    .part files are removed on EVERY failure path (including checksum
    mismatch), so an interrupted run can never leave a truncated file that a
    later run would mistake for complete work.

Float columns are float32 by default (--dtype float64 to opt out). At a ~100k
BTC price float32's ULP is ~0.008, i.e. a relative error of ~4e-8 -- far below
anything that matters for feature computation. Timestamps stay Int64 (ms epoch
~1.7e12 overflows int32). NOTE: do NOT apply float32 to the orderbook module's
px: above 131072 the ULP (0.0156) exceeds the spot tick (0.01) and adjacent
price levels would collapse into one value.

Scope decisions (Binance Vision layout):
  - spot -> klines only (spot has no metrics / fundingRate).
  - futures/um -> klines + metrics; metrics is daily-only (monthly auto-skipped).
  - fundingRate is a separate, monthly-only category, intentionally NOT
    downloaded: the spot/futures basis is computed from klines closes.

Every .zip has a sibling .CHECKSUM (sha256); verified by default. Already
downloaded files are skipped, so re-running is a safe resume.

Examples
  # test: a few days of BTC, U-margined futures, 1m klines + metrics
  python binance_vision/download_binance_vision.py --market um --symbol BTCUSDT \
      --category all --interval 1m --granularity daily \
      --start-date 2026-10-10 --end-date 2026-10-12

  # one calendar month of klines (e.g. 2025-06): ONE zip per market/symbol.
  # NOTE: monthly ignores the day-of-month and downloads whole months.
  python binance_vision/download_binance_vision.py --market both --symbol all \
      --category klines --interval 1m --granularity monthly \
      --start-date 2025-06-01 --end-date 2025-06-30

  # ... plus metrics for the same month. Binance publishes metrics per DAY
  # only, so --granularity monthly auto-skips it (see the run summary);
  # a full month of metrics needs one daily run (30 files per symbol here).
  python binance_vision/download_binance_vision.py --market um --symbol all \
      --category metrics --granularity daily \
      --start-date 2025-06-01 --end-date 2025-06-30

  # aggTrades (microstructure features; NOT part of --category all)
  python binance_vision/download_binance_vision.py --market um --symbol BTCUSDT \
      --category aggTrades --interval 1m --granularity daily \
      --start-date 2025-06-01 --end-date 2025-06-30
"""
import argparse
import concurrent.futures
import hashlib
import sys
import threading
import time
import zipfile
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
BASE = "https://data.binance.vision/data"

VALID_INTERVALS = {
    "1s", "1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h",
    "6h", "8h", "12h", "1d", "3d", "1w", "1mo",
}
MARKETS = {"spot": "spot", "um": "futures/um"}
VALID_CATEGORIES = {"klines", "aggTrades", "metrics"}

# "all" downloads klines + metrics; aggTrades is intentionally excluded for now
# (kept selectable via --category aggTrades once its usefulness is decided).
ALL_CATS = ["klines", "metrics"]

# Canonical column order per category. Used (a) when a file has NO header row
# -- some historical Binance CSVs omit the header -- and (b) to detect headers
# by checking the first token against EXPECTED_FIRST.
KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
              "quote_volume", "count", "taker_buy_volume",
              "taker_buy_quote_volume", "ignore"]
# klines output: keep open_time + close_time + OHLC + count + taker buy/sell
# volume (buy_vol / sell_vol); drop volume, quote_volume, taker_buy_quote_volume
# and ignore. 主动买 buy_vol = taker_buy_volume；主动卖 sell_vol = volume -
# taker_buy_volume。close_time (+1ms) is the candle's end instant and the natural
# join key to orderbook snapshots (close_time + 1 == next candle's open_time).
# open_time is kept too since it is that same next-candle join key viewed from
# the preceding bar.
KLINE_KEEP = ["open_time", "open", "high", "low", "close", "close_time",
              "count", "buy_vol", "sell_vol"]
METRICS_COLS = ["create_time", "symbol", "sum_open_interest",
                "sum_open_interest_value", "count_toptrader_long_short_ratio",
                "sum_toptrader_long_short_ratio", "count_long_short_ratio",
                "sum_taker_long_short_vol_ratio"]
# aggTrades 规范列名（8 列，最后一列 is_best_match）。真实文件有两种格式：
#   um   = 7 列、有表头、毫秒时间戳（2026-10 实测，没有 is_best_match）
#   spot = 8 列、无表头、微秒时间戳（2026-10 实测）
# 因此列名/override/cast 一律按文件实际列数过滤，is_best_match 视为可选列。
AGGTRADES_COLS = ["agg_trade_id", "price", "quantity", "first_trade_id",
                  "last_trade_id", "transact_time", "is_buyer_maker",
                  "is_best_match"]
CANONICAL = {"klines": KLINE_COLS, "metrics": METRICS_COLS,
             "aggTrades": AGGTRADES_COLS}
# First column name when a header IS present (to detect header-less files).
EXPECTED_FIRST = {"klines": "open_time", "metrics": "create_time",
                  "aggTrades": "agg_trade_id"}


def create_session(max_workers: int = 4) -> requests.Session:
    """Create a requests Session with HTTP connection pooling and automatic retries."""
    s = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        raise_on_status=False,
    )
    pool_size = max(10, max_workers * 2)
    adapter = HTTPAdapter(max_retries=retries, pool_connections=pool_size, pool_maxsize=pool_size)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    s.headers.update({"User-Agent": "Mozilla/5.0 (bn_hisdata/binance_vision)"})
    return s


def period_strings(granularity: str, start: date, end: date):
    if granularity == "daily":
        d = start
        while d <= end:
            yield d.strftime("%Y-%m-%d")
            d += timedelta(days=1)
    else:  # monthly
        y, m = start.year, start.month
        while (y, m) <= (end.year, end.month):
            yield f"{y:04d}-{m:02d}"
            m += 1
            if m > 12:
                m, y = 1, y + 1


def url_for(market_path: str, granularity: str, category: str,
            symbol: str, interval: str, period: str) -> str:
    if category == "klines":
        return (f"{BASE}/{market_path}/{granularity}/klines/{symbol}/{interval}/"
                f"{symbol}-{interval}-{period}.zip")
    return (f"{BASE}/{market_path}/{granularity}/{category}/{symbol}/"
            f"{symbol}-{category}-{period}.zip")


def dest_path(data_root: Path, market_key: str, granularity: str, category: str,
              symbol: str, interval: str, period: str) -> Path:
    if category == "klines":
        return (data_root / market_key / granularity / category / symbol /
                interval / f"{symbol}-{interval}-{period}.zip")
    return (data_root / market_key / granularity / category / symbol /
            f"{symbol}-{category}-{period}.zip")


def fetch_expected_sha256(session: requests.Session, url_checksum: str, log) -> str | None:
    """GET the sibling .CHECKSUM file; return the expected sha256 hex.

    None means "cannot verify" (missing/empty/unreachable) and the caller
    skips verification. Verification is a safety net, never a hard gate:
    only a real mismatch makes the caller discard the download.
    """
    try:
        r = session.get(url_checksum, timeout=30)
        if r.status_code != 200:
            log(f"no checksum file (HTTP {r.status_code}), skipped")
            return None
        text = r.text.strip()
        if not text:
            log("empty checksum file, skipped")
            return None
        return text.split()[0].lower()
    except Exception as e:  # 校验是安全网：取不到就跳过，绝不误伤正常下载
        log(f"checksum check skipped: {e}")
        return None


def download_one(session: requests.Session, url: str, dest: Path, verify: bool, log) -> str:
    if dest.exists():
        log("skip (exists)")
        return "skip"
    dest.parent.mkdir(parents=True, exist_ok=True)
    # 先写 .part 再原子改名：中断/崩溃留下的残件永远不会被当成完整 zip
    # 复用（旧实现直接写 dest，Ctrl-C 会留下截断文件，下次运行 process 必败，
    # 且要再跑一次脚本才能真正下载到）。
    tmp = dest.parent / (dest.name + ".part")
    last = None
    for attempt in range(1, 4):
        try:
            h = hashlib.sha256()
            size = 0
            with session.get(url, timeout=(30, 120), stream=True) as r:
                if r.status_code == 200:
                    # 流式落盘：不再把整个 zip 读进内存（月度 aggTrades 数百 MB），
                    # 同时边写边算 sha256，校验无需回读全文。
                    with tmp.open("wb") as fh:
                        for chunk in r.iter_content(chunk_size=1 << 20):
                            if chunk:
                                fh.write(chunk)
                                h.update(chunk)
                                size += len(chunk)
                    expected = (fetch_expected_sha256(session, url + ".CHECKSUM", log)
                                if verify else None)
                    if expected is not None and expected != h.hexdigest():
                        # 校验和真的不匹配 -> 丢弃本次响应并重试，绝不写出损坏文件
                        log(f"CHECKSUM MISMATCH: expected {expected}, "
                            f"got {h.hexdigest()}")
                        last = "checksum mismatch"
                    else:
                        if expected is not None:
                            log("checksum OK")
                        tmp.replace(dest)  # 原子：dest 上只会出现完整文件
                        log(f"saved {size / 1e6:.1f} MB")
                        return "ok"
                elif r.status_code == 404:
                    log(f"MISSING (HTTP 404): {url}")
                    return "missing"
                else:
                    last = r.status_code
        except Exception as e:  # transient network error -> retry
            last = e
        # .part 必须在每一条失败路径上都清掉（网络异常 AND 校验和不匹配）：
        # 下次 attempt 的 open("wb") 虽然会截断它，但清掉才能让
        # “失败后无残件”成为不变量，也避免 3 次全败时留下垃圾。
        tmp.unlink(missing_ok=True)
        if attempt < 3:
            time.sleep(2 ** attempt)  # 指数退避，避免连续猛打源站
            log(f"attempt {attempt} failed ({last}); retrying")
    log(f"FAILED after retries: {url}")
    return "failed"


def get_schema_overrides(category: str, fdtype: pl.DataType) -> dict[str, pl.DataType]:
    """Return explicit schema overrides for Polars scan_csv to optimize CSV parsing.

    NOTE (klines): `volume` and `taker_buy_volume` are intentionally Float64
    even when the output dtype is Float32. They feed
    sell_vol = volume - taker_buy_volume, and that subtraction must happen at
    full precision BEFORE the cast (see module docstring). The other price
    columns are never subtracted, so they take fdtype directly.
    """
    if category == "klines":
        return {
            "open_time": pl.Int64,
            "open": fdtype,
            "high": fdtype,
            "low": fdtype,
            "close": fdtype,
            "volume": pl.Float64,          # 全精度解析：sell_vol 减法用
            "close_time": pl.Int64,
            "quote_volume": fdtype,
            "count": pl.Int64,
            "taker_buy_volume": pl.Float64,  # 全精度解析：sell_vol 减法用
            "taker_buy_quote_volume": fdtype,
            "ignore": pl.String,
        }
    elif category == "aggTrades":
        return {
            "agg_trade_id": pl.Int64,
            "price": fdtype,
            "quantity": fdtype,
            "first_trade_id": pl.Int64,
            "last_trade_id": pl.Int64,
            "transact_time": pl.Int64,
            "is_buyer_maker": pl.Boolean,
            "is_best_match": pl.Boolean,
        }
    elif category == "metrics":
        # create_time 故意不 override：它是日期时间字符串，需要按扫描出的
        # schema 判断走 strptime 还是直接 cast（见 process）。
        return {
            "sum_open_interest": fdtype,
            "sum_open_interest_value": fdtype,
            "count_toptrader_long_short_ratio": fdtype,
            "sum_toptrader_long_short_ratio": fdtype,
            "count_long_short_ratio": fdtype,
            "sum_taker_long_short_vol_ratio": fdtype,
        }
    return {}


def process(zip_path: Path, category: str, fmt: str, fdtype, log) -> None:
    """Read the zip's CSV, normalize headers, trim klines, write a typed file.

    Binance historical files are inconsistent: some periods ship a header row,
    others don't. We detect this from the first token and always emit a clean,
    typed output next to the .zip (klines trimmed per KLINE_KEEP).

    Streaming end to end: the zip entry is expanded to a ".src.part" temp
    file, then parsed with scan_csv and sunk to parquet/CSV in one pass, so
    the frame is never materialized. Why the temp file instead of handing the
    zip entry to polars directly: polars buffers a Python file object whole to
    make it seekable -- measured on a 6M-row / 400 MB CSV, that alone cost
    ~400 MB of RSS (924 MB peak via file object vs 533 MB via path). Monthly
    aggTrades are GBs uncompressed, so this is the difference between working
    and OOM-ing on an 8 GB box. The trade is transient disk usage of roughly
    the CSV size; the temp file is always removed in a finally block.
    """
    out = zip_path.with_suffix(f".{fmt}")
    if out.exists():
        return
    # 输出先写 .part 再原子改名：sink 中途失败不会留下"看起来已完成"的输出
    # 文件（resume 以输出文件存在为准，残件会被误判为完成）。
    # 注意两个临时名不能撞车：--format csv 时 out 的后缀恰好是 .csv，
    # 若源临时文件也叫 .csv.part，sink 会覆盖自己正在读的文件（Windows 下
    # 直接 os error 1006）。源用 .src.part，输出用 .<fmt>.part。
    tmp_out = out.parent / (out.name + ".part")
    tmp_csv = out.parent / (zip_path.stem + ".src.part")
    try:
        with zipfile.ZipFile(zip_path) as z:
            csv_in = [n for n in z.namelist() if n.endswith(".csv")]
            if not csv_in:
                log("no csv inside zip")
                return
            with z.open(csv_in[0]) as src, tmp_csv.open("wb") as dst:
                while True:
                    chunk = src.read(1 << 20)
                    if not chunk:
                        break
                    dst.write(chunk)
        with tmp_csv.open("rb") as fh:
            first = fh.readline().decode("utf-8", errors="replace")
            if not first.strip():
                log("empty csv inside zip")
                return
            # Detect header by the first token (a header's first col name is text;
            # a data row's first field is a timestamp/number).
            exp = EXPECTED_FIRST.get(category)
            is_header = bool(exp) and first.split(",")[0].strip() == exp
            if is_header:
                names = [c.strip() for c in first.split(",")]
            else:
                # 无表头：按实际字段数取规范列名。aggTrades 有 7 列（um）和
                # 8 列（spot）两种真实格式，写死 8 个名字会在 7 列文件上炸。
                n_fields = len(first.split(","))
                canon = CANONICAL[category]
                names = (canon[:n_fields] if n_fields <= len(canon)
                         else canon + [f"column_{i}"
                                       for i in range(len(canon), n_fields)])
            # schema_overrides 只保留实际存在的列：um 7 列文件没有
            # is_best_match；spot 无表头文件靠 new_columns 取名。
            schema_ovr = {k: v for k, v in
                          get_schema_overrides(category, fdtype).items()
                          if k in names}
        # lazy scan：sink 之前不物化。按路径扫描（mmap 流式），不要直接传
        # zip 条目/file object——见 docstring 的实测说明。
        if is_header:
            lf = pl.scan_csv(tmp_csv, has_header=True,
                             schema_overrides=schema_ovr)
        else:
            lf = pl.scan_csv(tmp_csv, has_header=False,
                             new_columns=names,
                             schema_overrides=schema_ovr)
        if category == "klines":
            # 主动买成交量 buy_vol = taker_buy_volume；主动卖成交量 sell_vol = 总量 - 主动买。
            # volume / quote_volume / taker_buy_quote_volume / ignore 一律丢弃。
            # 时间戳与笔数保持 Int64（毫秒戳 ~1.7e12，int32 会溢出）；价格与成交量用
            # fdtype（默认 Float32：BTC 价位下 ULP≈0.008、相对误差 ~4e-8，对特征计算无影响）。
            # 减法在 cast 之前按全精度算（volume/taker_buy_volume 已按 Float64 解析，
            # 见 get_schema_overrides），不会先丢精度再相减。
            lf = lf.with_columns([
                pl.col("taker_buy_volume").cast(fdtype).alias("buy_vol"),
                (pl.col("volume") - pl.col("taker_buy_volume")).cast(fdtype).alias("sell_vol"),
                pl.when(pl.col("close_time") > 1e14)
                .then(pl.col("close_time") // 1000 + 1)
                .otherwise(pl.col("close_time") + 1)
                .alias("close_time"),
                pl.when(pl.col("open_time") > 1e14)
                .then(pl.col("open_time") // 1000)
                .otherwise(pl.col("open_time"))
                .alias("open_time"),
            ]).select(KLINE_KEEP)
        elif category == "aggTrades":
            # schema_overrides 已在解析期定型，这里再显式 cast 一遍：
            # 输出 dtype 不依赖 polars 各版本的 override 行为（防静默漂移）。
            # 按实际存在的列 cast：um 真实文件 7 列、没有 is_best_match，
            # 写死 8 列会直接报错（2026-10-20 实测）。
            agg_cast_map = {
                "agg_trade_id": pl.Int64,
                "first_trade_id": pl.Int64,
                "last_trade_id": pl.Int64,
                "transact_time": pl.Int64,
                "price": fdtype,
                "quantity": fdtype,
                "is_buyer_maker": pl.Boolean,  # "True"/"False" 大写也正常解析
                "is_best_match": pl.Boolean,
            }
            lf = lf.with_columns([
                pl.col(c).cast(d) for c, d in agg_cast_map.items() if c in names
            ])
            # transact_time 归一化到毫秒：spot 真实文件是微秒 epoch（~1.79e15，
            # 无表头 8 列），um 是毫秒。不归一化则跨市场单位不一致，下游
            # compute_aggtrade_features 按 ms 解释会全部错位。
            if "transact_time" in names:
                lf = lf.with_columns(
                    pl.when(pl.col("transact_time") > 1e14)
                    .then(pl.col("transact_time") // 1000)
                    .otherwise(pl.col("transact_time"))
                    .alias("transact_time")
                )
        elif category == "metrics":
            # create_time 在 metrics CSV 里是日期时间字符串
            # （"2025-06-01 00:00:00"，2023-2025 真实文件实测一致），不是毫秒
            # epoch——早期实现直接 cast(Int64) 在这里必炸。统一转成 ms epoch
            # 与 klines 的 open_time/close_time 对齐，便于 join。按扫描出的
            # schema 判断：字符串走 strptime，万一遇到 epoch 整数则直接 cast。
            if lf.collect_schema()["create_time"].is_numeric():
                create_time = pl.col("create_time").cast(pl.Int64)
            else:
                create_time = (pl.col("create_time")
                               .str.to_datetime(format="%Y-%m-%d %H:%M:%S")
                               .dt.epoch("ms").cast(pl.Int64))
            lf = lf.with_columns([create_time])
        if fmt == "parquet":
            # parquet：带类型、zstd 压缩体积更小、列式读取，特征计算更省事
            lf.sink_parquet(str(tmp_out), compression="zstd")
        else:
            lf.sink_csv(str(tmp_out))  # --format csv 时才走这里
        tmp_out.replace(out)
        log(f"cleaned -> {out.name}")
    finally:
        # 临时 CSV 与 sink 残件都必须清掉：前者可达 GB 级，后者会让下次运行
        # 的 resume 检查看到"半完成"的输出。
        tmp_csv.unlink(missing_ok=True)
        tmp_out.unlink(missing_ok=True)


def parse_list_arg(val: str) -> list[str]:
    return [item.strip() for item in val.split(",") if item.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--market", choices=["spot", "um", "both"], default="both")
    ap.add_argument("--symbol", default="BTCUSDT",
                    help="BTCUSDT / ETHUSDT / all / comma-separated list "
                         "(e.g. BTCUSDT,ETHUSDT,SOLUSDT)")
    ap.add_argument("--category", default="all",
                    help="klines / aggTrades / metrics / all / comma-separated list "
                         "(e.g. klines,metrics)。all = klines + metrics，不含 aggTrades")
    ap.add_argument("--interval", default="1m", help="klines interval")
    ap.add_argument("--granularity", choices=["daily", "monthly"], default="daily",
                    help="daily=每天一个文件（测试/少量天数）；monthly=每月一个文件"
                         "（批量回补快得多）。monthly 按整月取文件，起止日期中的“日”"
                         "会被忽略；metrics 只有日度文件，monthly 下会自动跳过")
    ap.add_argument("--start-date", dest="start_date", required=True,
                    help="ISO 日期，如 2025-06-01；monthly 粒度下同月内任意日期等价")
    ap.add_argument("--end-date", dest="end_date", required=True,
                    help="ISO 日期，如 2025-06-30")
    ap.add_argument("--dest-root", default=str(DATA),
                    help="Local output root (default: <module>/data)")
    ap.add_argument("--no-checksum", action="store_true")
    ap.add_argument("--no-extract", action="store_true")
    ap.add_argument("--format", choices=["parquet", "csv"], default="parquet",
                    help="输出格式，默认 parquet（带类型、zstd 压缩更小、列式读取更快）。"
                         "断点续传按输出文件判定，切换格式会重新下载")
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float32",
                    help="浮点列精度，默认 float32：BTC 价位下 ULP≈0.008、相对误差 ~4e-8，"
                         "对特征计算无影响且体积减半；需要逐 tick 精确价位时用 float64")
    ap.add_argument("--workers", "-j", type=int, default=4,
                    help="并发下载和处理的工作线程数（默认: 4）")
    args = ap.parse_args()

    if args.interval not in VALID_INTERVALS:
        ap.error(f"--interval must be one of {sorted(VALID_INTERVALS)}")
    if args.workers < 1:
        ap.error("--workers must be >= 1")

    data_root = Path(args.dest_root)
    markets = ["spot", "um"] if args.market == "both" else [args.market]

    if args.symbol.lower() == "all":
        symbols = ["BTCUSDT", "ETHUSDT"]
    else:
        symbols = [s.upper() for s in parse_list_arg(args.symbol)]

    if args.category.lower() == "all":
        cats = ALL_CATS
    else:
        cats = parse_list_arg(args.category)
    for cat in cats:
        if cat not in VALID_CATEGORIES:
            ap.error(f"Invalid category '{cat}'. Must be one of {sorted(VALID_CATEGORIES)}")

    start = date.fromisoformat(args.start_date)
    end = date.fromisoformat(args.end_date)
    if start > end:
        ap.error("--start-date must be <= --end-date")

    fdtype = pl.Float32 if args.dtype == "float32" else pl.Float64

    if args.granularity == "monthly" and not (
            start.day == 1 and (end + timedelta(days=1)).day == 1):
        # monthly 按整月取文件：提醒起止日会被忽略，避免"只想下 6 月 15-20 日"
        # 结果下了整个 6 月，或跨月区间多下整月。
        print(f"NOTE: --granularity monthly 按整月下载，起止日期中的“日”会被忽略："
              f"{args.start_date} ~ {args.end_date} 覆盖的每个整月都会被下载。",
              flush=True)

    print_lock = threading.Lock()
    totals_lock = threading.Lock()

    def log(tag: str, msg: str) -> None:
        with print_lock:
            print(f"[{tag}] {msg}", flush=True)

    totals = {"ok": 0, "skip": 0, "missing": 0, "failed": 0}
    skipped = []  # 按设计跳过的组合，汇总时集中打印，避免淹没在逐文件日志里

    def record_total(status: str, count: int = 1) -> None:
        with totals_lock:
            totals[status] = totals.get(status, 0) + count

    def handle_task(task: dict, session: requests.Session) -> None:
        tag = task["tag"]
        # 最外层兜底：任何未预期异常都要计数为 failed——并发场景下
        # 裸 wait() 会静默吞掉 worker 崩溃，summary 看起来一切正常。
        try:
            url = task["url"]
            dp = task["dp"]
            cat = task["cat"]
            out = task["out"]

            if out.exists():
                log(tag, f"skip (cleaned .{args.format} exists)")
                record_total("skip")
                return

            status = download_one(session, url, dp, not args.no_checksum,
                                  lambda m: log(tag, m))
            if status not in ("ok", "skip"):
                record_total(status)
                return

            if args.no_extract:
                record_total(status)
                return

            try:
                process(dp, cat, args.format, fdtype, lambda m: log(tag, m))
            except Exception as e:
                # 坏 zip / 列数不符只影响这一个周期：强制重新下载一次再试，
                # 旧实现复用坏的残留 zip 只能记 failed，要再跑一次整个脚本
                # 才可能恢复。
                log(tag, f"process FAILED: {e}; forcing re-download and retrying once")
                if dp.exists():
                    dp.unlink()
                if download_one(session, url, dp, not args.no_checksum,
                                lambda m: log(tag, m)) != "ok":
                    record_total("failed")
                    return
                try:
                    process(dp, cat, args.format, fdtype, lambda m: log(tag, m))
                except Exception as e2:
                    log(tag, f"process FAILED again: {e2}; giving up on this period")
                    if dp.exists():
                        dp.unlink()
                    record_total("failed")
                    return

            # Source zip is now redundant; drop it to save space
            # (matters a lot for monthly backfills).
            if dp.exists():
                dp.unlink()
                log(tag, "removed source zip")
            # status=="skip" 表示复用了上次中断残留的 zip；本次确实完成了处理，计为 ok
            record_total("ok")
        except Exception as e:  # worker 不允许静默死亡
            log(tag, f"UNEXPECTED FAILURE: {e!r}")
            record_total("failed")

    tasks = []
    for market_key in markets:
        market_path = MARKETS[market_key]
        for symbol in symbols:
            for cat in cats:
                if cat == "metrics" and market_key == "spot":
                    log(f"{market_key}/{cat}/{symbol}", "skip: spot has no metrics")
                    skipped.append(f"{market_key}/metrics/{symbol}: spot 无 metrics 数据")
                    continue
                if cat == "metrics" and args.granularity == "monthly":
                    log(f"{market_key}/{cat}/{symbol}",
                        "skip: metrics is daily-only")
                    skipped.append(f"{market_key}/metrics/{symbol}: "
                                   "metrics 只有日度文件，monthly 自动跳过")
                    continue
                for period in period_strings(args.granularity, start, end):
                    tag = f"{market_key}/{cat}/{symbol}/{period}"
                    url = url_for(market_path, args.granularity, cat, symbol,
                                  args.interval, period)
                    dp = dest_path(data_root, market_key, args.granularity, cat,
                                   symbol, args.interval, period)
                    out = dp.with_suffix(f".{args.format}")
                    # Resume key is the cleaned output file, not the zip (we delete
                    # the zip after processing, so a leftover zip means an
                    # interrupted run, not a finished one).
                    if out.exists():
                        log(tag, f"skip (cleaned .{args.format} exists)")
                        totals["skip"] += 1
                        continue
                    tasks.append({"tag": tag, "url": url, "dp": dp,
                                  "cat": cat, "out": out})

    if tasks:
        session = create_session(max_workers=args.workers)
        if args.workers > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = [executor.submit(handle_task, task, session)
                           for task in tasks]
                concurrent.futures.wait(futures)
                # wait() 不传播异常；handle_task 虽已兜底，这里再查一遍，
                # 双保险确保没有 worker 崩溃被静默吞掉。
                for fut in futures:
                    exc = fut.exception()
                    if exc is not None:
                        log("main", f"worker crashed: {exc!r}")
                        record_total("failed")
        else:
            for task in tasks:
                handle_task(task, session)

    print("\n==== summary ====")
    print(f"ok={totals['ok']} skip={totals['skip']} "
          f"missing={totals['missing']} failed={totals['failed']}")
    if skipped:
        print("---- skipped by design ----")
        for s in skipped:
            print(f"  {s}")
        print("  整月 metrics 需另跑一条日度命令，例如：")
        print("    python binance_vision/download_binance_vision.py "
              f"--market um --category metrics --granularity daily "
              f"--start-date {args.start_date} --end-date {args.end_date}")
    if totals["failed"]:
        # 非零退出码：自动化运行（如 Jules）不会把“有文件没下到”看成成功。
        # 重跑同一命令即可断点续传补齐。
        print(f"ERROR: {totals['failed']} period(s) failed; re-run the same "
              f"command to resume.")
        sys.exit(1)


if __name__ == "__main__":
    main()
