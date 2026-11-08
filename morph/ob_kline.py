#!/usr/bin/env python3
"""分别取「K 线」与「订单簿」两份数据（**不合并**），供 `morph/two_legs.py` 两条腿各训一版。

数据源（`bailiang0369/bn_hisdata` 的公开 Release，不需要 token）
----------------------------------------------------------------
| tag | 资产 | 内容 |
|---|---|---|
| `vision-base` | `{SYMBOL}_1m_klines_{YYYY}.parquet` | 15 列原始 1m K 线：um OHLC/主动买卖量/笔数 + spot 同结构 |
| `ob-dataset-vol8` | `ob_{SYMBOL}.parquet` | 185 列订单簿特征（纯盘口，不含 K 线），主键 `close_time` |

**为什么分开取**：K 线有 2020→今天（≈350 万行），订单簿只有 2025-07→2026-10
（≈69 万行）。拼成一张表只能按订单簿那段短窗口训练，K 线 5 年多的历史会被裁掉；
两者的正确用法是**各自训练、在预测层融合**（结论见 `融合实验记录.md`）。

口径
----
- K 线 OHLC 二选一：`KLINE_SRC=um`（合约，默认）/ `spot`（现货）；`WITH_BASIS=1`
  才额外算一列 `k_basis_bps`，仍只保留一侧 OHLC。
- 两侧主键都是 `close_time`（bn_hisdata 已把 `bucket+60000` 换算好，**不要**再 -60000）。
"""
from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import requests

try:                                     # kernel 里按包导入（morph.ob_kline）
    from morph.kline_features import build_features
except ModuleNotFoundError:              # 脚本模式：morph/ 目录里的 morph.py 会抢包名
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from kline_features import build_features

REPO = "bailiang0369/bn_hisdata"
GH_API = "https://api.github.com"

SYMBOL = os.environ.get("SYMBOL", "BTCUSDT")
KLINE_SRC = os.environ.get("KLINE_SRC", "um").lower()
WITH_BASIS = os.environ.get("WITH_BASIS", "0") not in ("0", "", "false", "False")
OB_TAG = os.environ.get("OB_TAG", "ob-dataset-vol8")
KLINE_TAG = os.environ.get("KLINE_TAG", "vision-base")
OB_MONTHS = [m.strip() for m in os.environ.get("OB_MONTHS", "").split(",") if m.strip()]
CACHE = Path(os.environ.get("CACHE_DIR") or "/tmp/bn_hisdata_cache")
WARMUP_MS = 130 * 60_000                 # 最长滚动窗口 120 分钟 + 余量
T = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T:7.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------- 下载
_ASSETS: dict[str, dict[str, dict]] = {}


def _assets(tag: str) -> dict[str, dict]:
    """{资产名: {size, url}}；公开仓库匿名即可（同样带重试：网关 5xx 很常见）。"""
    if tag in _ASSETS:
        return _ASSETS[tag]
    h = {"Accept": "application/vnd.github+json"}
    tok = os.environ.get("GH_TOKEN")
    if tok:
        h["Authorization"] = f"Bearer {tok}"
    last: Exception | None = None
    for attempt in range(1, RETRIES + 1):
        try:
            r = requests.get(f"{GH_API}/repos/{REPO}/releases/tags/{tag}",
                             headers=h, timeout=60)
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code} {r.text[:120]}")
            out = {a["name"]: {"size": a["size"], "url": a["browser_download_url"]}
                   for a in r.json()["assets"]}
            _ASSETS[tag] = out
            log(f"release {tag}: {len(out)} 个资产")
            return out
        except Exception as e:
            last = e
            log(f"取 release {tag} 第 {attempt}/{RETRIES} 次失败: {e}")
            if attempt < RETRIES:
                time.sleep(min(60, 5 * 2 ** (attempt - 1)))
    raise SystemExit(f"取 release {tag} 重试 {RETRIES} 次仍失败: {last}")


RETRIES = int(os.environ.get("DL_RETRIES", "12"))


def _part_bytes(part: Path) -> int:
    return part.stat().st_size if part.exists() else 0


def fetch(tag: str, name: str) -> Path:
    """Range 续传下载到 /tmp 缓存；已下满（大小一致）直接复用。

    GitHub Release 对 393MB 的订单簿包会偶发 **504**（网关读超时），而且一次失败就整个
    kernel 挂掉，白等十几分钟。所以这里做成「失败 → 续传重试」：504 时 `.part` 保留，
    下一次带 `Range` 从断点继续，退避 5/10/20/40/60s 最多 6 次。
    """
    a = _assets(tag).get(name)
    if a is None:
        raise SystemExit(f"{tag} 里没有资产 {name}")
    CACHE.mkdir(parents=True, exist_ok=True)
    dst, size = CACHE / name, a["size"]
    if dst.exists():
        if dst.stat().st_size == size:
            return dst
        dst.unlink()
    part = dst.with_name(dst.name + ".part")

    for attempt in range(1, RETRIES + 1):
        start = _part_bytes(part)
        if start > size:                      # 上次留下的垃圾，重下
            part.unlink()
            start = 0
        if start == size:                     # 断点已到末尾：直接落盘
            part.replace(dst)
            return dst
        headers = {"Range": f"bytes={start}-"} if start else {}
        log(f"下载 {name} {size / 1e6:.1f} MB"
            + (f"（续传 {start / 1e6:.1f} MB）" if start else "")
            + (f" 第 {attempt}/{RETRIES} 次" if attempt > 1 else ""))
        try:
            with requests.get(a["url"], stream=True, timeout=300, headers=headers) as r:
                if start and r.status_code == 206:
                    mode = "ab"
                elif r.status_code in (200, 206):
                    mode = "wb"
                else:
                    raise RuntimeError(f"HTTP {r.status_code}")
                with open(part, mode) as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
        except Exception as e:                # 504 / 连接重置 / 读到一半断开
            log(f"下载 {name} 第 {attempt}/{RETRIES} 次失败: {e}"
                f"（已落 {_part_bytes(part) / 1e6:.1f} MB）")
            if attempt == RETRIES:
                raise SystemExit(f"下载 {name} 重试 {RETRIES} 次仍失败: {e}")
            # 504 是 CDN 回源失败，前几次几乎必然失败；退避短一点，靠次数取胜
            time.sleep(min(30, 3 * 2 ** (attempt - 1)))
            continue
        if _part_bytes(part) == size:         # 校验大小后才算成功
            part.replace(dst)
            return dst
        log(f"下载 {name} 未下满（{_part_bytes(part) / 1e6:.1f} / {size / 1e6:.1f} MB），续传")

    raise SystemExit(f"下载 {name} 重试 {RETRIES} 次仍未下满")


# ---------------------------------------------------------------- 订单簿腿
def load_ob(symbol: str | None = None, months: list[str] | None = None) -> pl.DataFrame:
    """订单簿特征表（185 列，主键 close_time）。行范围天然是 2025-07~2026-10。

    `OB_MONTHS=auto` 时自动改用 Release 里的月份分片（每个 17~27MB）：整包 393MB 走
    GitHub Release CDN 时 504 概率高得多，分片单文件回源快、失败也只影响一个月。
    """
    symbol = symbol or SYMBOL
    months = OB_MONTHS if months is None else months
    if [m.lower() for m in months] == ["auto"]:
        prefix, suffix = f"ob_{symbol}_", ".parquet"
        months = sorted(n[len(prefix):-len(suffix)] for n in _assets(OB_TAG)
                        if n.startswith(prefix) and n.endswith(suffix))
        log(f"自动分片: {len(months)} 个（{months[0]} ~ {months[-1]}）" if months
            else "自动分片: Release 里没有月度分片，回退整包")
    if months:
        frames = [pl.read_parquet(fetch(OB_TAG, f"ob_{symbol}_{m}.parquet")) for m in months]
        df = pl.concat(frames, how="vertical_relaxed")
    else:
        df = pl.read_parquet(fetch(OB_TAG, f"ob_{symbol}.parquet"))
    df = df.unique(subset=["close_time"], keep="last").sort("close_time")
    log(f"订单簿: {df.height:,} 行 x {df.width} 列 | "
        f"{_iso(int(df['close_time'].min()))} ~ {_iso(int(df['close_time'].max()))}")
    return df


# ---------------------------------------------------------------- K 线腿
_SRC = {
    "um": ("open", "high", "low", "close", "um_buy_vol", "um_sell_vol"),
    "spot": ("spot_open", "spot_high", "spot_low", "spot_close",
             "spot_buy_vol", "spot_sell_vol"),
}


def load_kline_features(y0: int = 2020, symbol: str | None = None,
                        src: str | None = None, with_basis: bool | None = None,
                        keep_raw: bool = False) -> tuple[pl.DataFrame, list[str]]:
    """K 线 2020→今天 + 63 个 K 线特征（`morph.kline_features`），**不裁窗口**。

    这条腿的价值就在长历史：训练段是 2020→订单簿 train 末尾，而不是订单簿那 3 个月。
    `keep_raw=True` 时额外保留 `high/low` 与四列主动买卖量，供 h3 宽表算跨域特征
    （`atk_*` / `ob_amp_obi_*`）用；这些原始列**不算特征**，不放进返回的 cols。
    """
    symbol = symbol or SYMBOL
    src = (src or KLINE_SRC).lower()
    basis = WITH_BASIS if with_basis is None else with_basis
    if src not in _SRC:
        raise SystemExit(f"kline_src 只能是 um / spot，收到 {src}")

    assets = _assets(KLINE_TAG)
    y1 = datetime.now(timezone.utc).year
    names = [f"{symbol}_1m_klines_{y}.parquet" for y in range(y0, y1 + 1)]
    have = [n for n in names if n in assets]
    if not have:
        raise SystemExit(f"{KLINE_TAG} 里没有 {y0}~{y1} 的 K 线资产")
    raw = (pl.concat([pl.scan_parquet(fetch(KLINE_TAG, n)) for n in have])
             .unique(subset=["close_time"], keep="last").collect().sort("close_time"))

    o, h, l, c, buy, sell = _SRC[src]
    # keep_raw 要保留**两侧**四列主动买卖量：跨域特征 atk_f_* / atk_s_* 都要用，
    # 不能只留当前 src 那一侧。
    vols = [v for v in ("um_buy_vol", "um_sell_vol", "spot_buy_vol", "spot_sell_vol")
            if v in raw.columns]
    k = raw.select(
        pl.col("close_time"),
        pl.col(o).alias("open"), pl.col(h).alias("high"),
        pl.col(l).alias("low"), pl.col(c).alias("close"),
        # build_features 认 volume / taker_buy_volume；K 线表只有主动买/卖两列
        (pl.col(buy) + pl.col(sell)).alias("volume"),
        pl.col(buy).alias("taker_buy_volume"),
        *([pl.col(v) for v in vols] if keep_raw else []),
        *([] if not basis else
          [pl.col("close").alias("_um_close"), pl.col("spot_close").alias("_spot_close")]),
    )
    out, cols = build_features(k)
    if basis:
        out = out.with_columns(
            ((pl.col("_um_close") - pl.col("_spot_close"))
             / pl.col("_spot_close") * 1e4).alias("k_basis_bps")
        ).drop(["_um_close", "_spot_close"])
        cols = cols + ["k_basis_bps"]

    # 特征一律 f32：350 万行 × 63 列用 f64 是 1.8GB，训练矩阵本来也要 f32
    extra = [] if not keep_raw else [
        c for c in ("high", "low", "um_buy_vol", "um_sell_vol",
                    "spot_buy_vol", "spot_sell_vol") if c in out.columns]
    out = out.select([pl.col("close_time"), pl.col("close").cast(pl.Float64),
                      *[pl.col(c2).cast(pl.Float32) for c2 in cols],
                      *[pl.col(c2).cast(pl.Float64) for c2 in extra]])
    log(f"K 线: {out.height:,} 行 x {len(cols)} 特征 | "
        f"{_iso(int(out['close_time'].min()))} ~ {_iso(int(out['close_time'].max()))}")
    return out, cols


def _iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
