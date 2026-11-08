#!/usr/bin/env python3
"""Kaggle kernel: 从 GitHub Release 拉「K 线 + 订单簿特征」，合成一张训练宽表。

数据源（`bailiang0369/bn_hisdata` 的公开 Release，**不需要 token**）
------------------------------------------------------------------
| tag | 资产 | 内容 |
|---|---|---|
| `vision-base` | `{SYMBOL}_1m_klines_{YYYY}.parquet` | 15 列原始 1m K 线：um OHLC / 主动买卖量 / 笔数 + spot 同结构 |
| `ob-dataset-vol8` | `ob_{SYMBOL}.parquet`<br>`ob_{SYMBOL}_{YYYY-MM}.parquet` | 185 列订单簿特征（纯盘口，不含 K 线），主键 `close_time` |

产物：`/kaggle/working/{SYMBOL}_ob_kline.parquet`
      列 = `close_time` + `close` + `k_*`（K 线特征）+ `ob_*`（订单簿特征）
      另写同名 `.meta.json`（`feature_cols` / 行数 / 时间窗 / 口径），下游按它取特征，
      别再"除了 target/close_time 全是特征"——那会把 `close` 自己也当特征。

为什么放 Kaggle
--------------
订单簿单文件 375 MB、K 线两年 ~500 MB。本机只改代码，不在本地下载大文件：
`kaggle_launcher.py --project dataset --push` 即可（联网拉 Release，产物在 kernel output）。

口径（要紧的几条）
------------------
1. **行范围只取订单簿覆盖的时间段**（当前 `2025-07` ~ `2026-10`）。K 线会多读
   `WARMUP_MS` 的头部用于滚动特征预热（最长窗口 120 分钟），算完裁掉，不进产物。
2. **K 线 OHLC 二选一**：`KLINE_SRC=um`（合约，默认）/ `spot`（现货）。另一侧默认
   完全不用；`WITH_BASIS=1` 时才额外算一列 `k_basis_bps`（期现基差），仍只保留一侧 OHLC。
3. 主键 `close_time` 两侧同键（bn_hisdata 已把订单簿的 `bucket+60000` 换算好，**不要**
   再 -60000），左表是订单簿：订单簿缺分钟已被 reindex 成 `ob_*` 全 null 的行，
   K 线侧缺分钟则 `k_*`/`close` 为 null（LightGBM 原生处理，特征层不做 imputation）。

env
---
    SYMBOL=BTCUSDT            标的（订单簿目前只有 BTCUSDT）
    KLINE_SRC=um|spot         K 线口径，默认 um
    WITH_BASIS=0|1            1 = 多加 k_basis_bps（默认 0，不碰另一侧）
    OB_TAG / KLINE_TAG        Release tag，默认 ob-dataset-vol8 / vision-base
    OB_MONTHS=2026-09,2026-10 只拉这几个月订单簿（单月 26MB，迭代/冒烟用）
    START / END               手动裁时间窗（默认由订单簿数据的 min/max 决定）
    OUT=/kaggle/working/xxx.parquet
    UPLOAD_TAG=ob-kline       （可选）把产物传到该 GitHub Release，需 GH_TOKEN；
                              不设就只留在 kernel output（避免再把它下载回本地）
    ALLOW_LOCAL=1             允许非 Kaggle 环境运行（默认拒绝，防本地拉大文件）
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import requests

try:                                     # kernel 里按包导入（morph.build_dataset）
    from morph.kline_features import build_features
except ModuleNotFoundError:              # `python morph/build_dataset.py`：直接跑脚本
    # 脚本模式下 sys.path 里有 morph/ 目录，其中的 morph.py 是**普通模块**，
    # 优先级高于 morph/ 命名空间包，`import morph` 会拿到它并报 "not a package"。
    # 干脆把同目录的 kline_features.py 当顶层模块直接导入，绕开这个同名冲突。
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from kline_features import build_features

REPO = "bailiang0369/bn_hisdata"
GH_API = "https://api.github.com"
DL_BASE = f"https://github.com/{REPO}/releases/download"

SYMBOL = os.environ.get("SYMBOL", "BTCUSDT")
KLINE_SRC = os.environ.get("KLINE_SRC", "um").lower()
WITH_BASIS = os.environ.get("WITH_BASIS", "0") not in ("0", "", "false", "False")
OB_TAG = os.environ.get("OB_TAG", "ob-dataset-vol8")
KLINE_TAG = os.environ.get("KLINE_TAG", "vision-base")
OB_MONTHS = [m.strip() for m in os.environ.get("OB_MONTHS", "").split(",") if m.strip()]
START = os.environ.get("START") or None
END = os.environ.get("END") or None
OUT = Path(os.environ.get("OUT") or f"/kaggle/working/{SYMBOL}_ob_kline.parquet")
UPLOAD_TAG = os.environ.get("UPLOAD_TAG", "").strip()
CACHE = Path("/tmp/bn_hisdata_cache")    # 不进 /kaggle/working，避免被当成产物打包
WARMUP_MS = 130 * 60_000                 # 最长滚动窗口 120 分钟 + 余量
T = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T:7.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------- 下载
_ASSETS: dict[str, dict[str, dict]] = {}


def _assets(tag: str) -> dict[str, dict]:
    """{资产名: {size, url}}；公开仓库匿名即可，失败才提示建 token。"""
    if tag in _ASSETS:
        return _ASSETS[tag]
    h = {"Accept": "application/vnd.github+json"}
    tok = os.environ.get("GH_TOKEN")
    if tok:
        h["Authorization"] = f"Bearer {tok}"
    r = requests.get(f"{GH_API}/repos/{REPO}/releases/tags/{tag}", headers=h, timeout=60)
    if r.status_code != 200:
        raise SystemExit(f"取 release {tag} 失败: HTTP {r.status_code} {r.text[:200]}")
    out = {a["name"]: {"size": a["size"], "url": a["browser_download_url"]}
           for a in r.json()["assets"]}
    _ASSETS[tag] = out
    log(f"release {tag}: {len(out)} 个资产")
    return out


def fetch(tag: str, name: str) -> Path:
    """Range 续传下载到 /tmp 缓存；已下满直接复用（大小校验）。"""
    a = _assets(tag).get(name)
    if a is None:
        raise SystemExit(f"{tag} 里没有资产 {name}（先 `--list` 看看）")
    CACHE.mkdir(parents=True, exist_ok=True)
    dst, size = CACHE / name, a["size"]
    if dst.exists():
        if dst.stat().st_size == size:
            return dst
        dst.unlink()
    part = dst.with_name(dst.name + ".part")
    start = part.stat().st_size if part.exists() else 0
    if start > size:
        part.unlink()
        start = 0
    if start == size:
        part.replace(dst)
        return dst
    headers = {"Range": f"bytes={start}-"} if start else {}
    log(f"下载 {name} {size / 1e6:.1f} MB" + (f"（续传 {start / 1e6:.1f} MB）" if start else ""))
    with requests.get(a["url"], stream=True, timeout=300, headers=headers) as r:
        if start and r.status_code == 206:
            mode = "ab"
        elif r.status_code in (200, 206):
            mode = "wb"
        else:
            raise SystemExit(f"下载 {name} 失败: HTTP {r.status_code}")
        with open(part, mode) as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    part.replace(dst)
    if dst.stat().st_size != size:
        raise SystemExit(f"{name} 大小不符: {dst.stat().st_size} != {size}")
    return dst


# ---------------------------------------------------------------- 数据
def load_ob() -> pl.DataFrame:
    """订单簿特征表（行范围的定义者）。"""
    if OB_MONTHS:
        frames = [pl.read_parquet(fetch(OB_TAG, f"ob_{SYMBOL}_{m}.parquet"))
                  for m in OB_MONTHS]
        df = pl.concat(frames, how="vertical_relaxed")
    else:
        df = pl.read_parquet(fetch(OB_TAG, f"ob_{SYMBOL}.parquet"))
    df = df.unique(subset=["close_time"], keep="last").sort("close_time")
    log(f"订单簿: {df.height:,} 行 x {df.width} 列")
    return df


def _year_files(lo_ms: int, hi_ms: int) -> list[Path]:
    y0 = datetime.fromtimestamp(lo_ms / 1000, tz=timezone.utc).year
    y1 = datetime.fromtimestamp(hi_ms / 1000, tz=timezone.utc).year
    return [fetch(KLINE_TAG, f"{SYMBOL}_1m_klines_{y}.parquet") for y in range(y0, y1 + 1)]


def load_klines(lo_ms: int, hi_ms: int) -> pl.DataFrame:
    """只读 [lo-warmup, hi] 的 K 线；多读的点专供滚动特征预热。"""
    files = _year_files(lo_ms - WARMUP_MS, hi_ms)
    lf = pl.concat([pl.scan_parquet(f) for f in files])
    df = (lf.filter((pl.col("close_time") >= lo_ms - WARMUP_MS)
                    & (pl.col("close_time") <= hi_ms))
          .collect()
          .unique(subset=["close_time"], keep="last")
          .sort("close_time"))
    log(f"K 线: {df.height:,} 行（含 {WARMUP_MS // 60_000}min 预热）")
    return df


_SRC = {
    "um": ("open", "high", "low", "close", "um_buy_vol", "um_sell_vol"),
    "spot": ("spot_open", "spot_high", "spot_low", "spot_close",
             "spot_buy_vol", "spot_sell_vol"),
}


def kline_table(raw: pl.DataFrame, lo_ms: int, hi_ms: int) -> tuple[pl.DataFrame, list[str]]:
    """选定一侧 OHLC -> morph.kline_features.build_features -> 裁回订单簿窗口。"""
    if KLINE_SRC not in _SRC:
        raise SystemExit(f"KLINE_SRC 只能是 um / spot，收到 {KLINE_SRC}")
    o, h, l, c, buy, sell = _SRC[KLINE_SRC]

    k = raw.select(
        pl.col("close_time"),
        pl.col(o).alias("open"), pl.col(h).alias("high"),
        pl.col(l).alias("low"), pl.col(c).alias("close"),
        # build_features 认的是 volume / taker_buy_volume；K 线表只有主动买/卖两列
        (pl.col(buy) + pl.col(sell)).alias("volume"),
        pl.col(buy).alias("taker_buy_volume"),
        *([] if not WITH_BASIS else
          [pl.col("close").alias("_um_close"), pl.col("spot_close").alias("_spot_close")]),
    )
    out, cols = build_features(k)
    if WITH_BASIS:
        out = out.with_columns(
            ((pl.col("_um_close") - pl.col("_spot_close"))
             / pl.col("_spot_close") * 1e4).alias("k_basis_bps")
        ).drop(["_um_close", "_spot_close"])
        cols = cols + ["k_basis_bps"]

    # 裁回订单簿窗口：预热行只参与滚动，不进产物
    out = (out.filter((pl.col("close_time") >= lo_ms) & (pl.col("close_time") <= hi_ms))
              .select(["close_time", "close", *cols]))
    log(f"K 线特征: {out.height:,} 行 x {len(cols)} 列（src={KLINE_SRC}"
        f"{', +basis' if WITH_BASIS else ''}）")
    return out, cols


# ---------------------------------------------------------------- 上传（可选）
def upload_release(tag: str, fp: Path) -> None:
    """把产物覆盖式传到 Release（同名先删后传），避免再下载回本地。

    没有 GH_TOKEN 就跳过——kernel 里默认没有，需要时由 launcher `--env-from GH_TOKEN`
    注入（bootstrap 的 code_file 首行机制见 tools/push_kernel.py）。
    """
    tok = os.environ.get("GH_TOKEN")
    if not tok:
        log("跳过上传：未设置 GH_TOKEN（产物只在 kernel output）")
        return
    h = {"Authorization": f"Bearer {tok}", "Accept": "application/vnd.github+json"}
    api = f"{GH_API}/repos/{REPO}/releases"
    r = requests.get(f"{api}/tags/{tag}", headers=h, timeout=60)
    if r.status_code == 404:
        r = requests.post(api, headers=h, timeout=60,
                          json={"tag_name": tag, "name": tag, "prerelease": True})
    rid = r.json()["id"]
    for a in requests.get(f"{api}/{rid}/assets?per_page=100", headers=h,
                          timeout=60).json():
        if a["name"] == fp.name:
            requests.delete(f"{GH_API}/repos/{REPO}/releases/assets/{a['id']}",
                            headers=h, timeout=60)
    with open(fp, "rb") as fh:
        r = requests.post(
            f"https://uploads.github.com/repos/{REPO}/releases/{rid}/assets",
            params={"name": fp.name}, headers={**h,
                                               "Content-Type": "application/octet-stream"},
            data=fh, timeout=1800)
    log(f"上传 {fp.name} -> {tag}: {r.status_code}")


# ---------------------------------------------------------------- 主流程
def main() -> None:
    if not Path("/kaggle/working").exists() and os.environ.get("ALLOW_LOCAL") != "1":
        raise SystemExit(
            "拒绝在本机跑：会下载 375MB+ 的订单簿。请用 "
            "`python kaggle_launcher.py --project dataset --push`；\n"
            "确要本地冒烟：设 ALLOW_LOCAL=1 且 OB_MONTHS=2026-09（单月 26MB）。")

    ob = load_ob()
    lo = int(ob["close_time"].min()) if START is None else _ms(START)
    hi = int(ob["close_time"].max()) if END is None else _ms(END)
    ob = ob.filter((pl.col("close_time") >= lo) & (pl.col("close_time") <= hi))
    log(f"窗口: {_iso(lo)} ~ {_iso(hi)}（{ob.height:,} 行，订单簿定界）")

    raw = load_klines(lo, hi)
    kf, kcols = kline_table(raw, lo, hi)

    df = ob.join(kf, on="close_time", how="left").sort("close_time")
    ob_cols = [c for c in ob.columns if c != "close_time"]
    df = df.select(["close_time", "close", *kcols, *ob_cols])

    # 体检：K 线缺分钟 / 订单簿缺分钟 / 交集
    miss_k = int(df["close"].null_count())
    miss_ob = int(df[f"ob_f_missing"].sum()) if "ob_f_missing" in df.columns else -1
    log(f"合成: {df.height:,} 行 x {df.width} 列 | K线缺失 {miss_k:,}"
        f" | 订单簿缺分钟 {miss_ob:,}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(OUT, compression="zstd")
    meta = {
        "symbol": SYMBOL,
        "rows": df.height,
        "cols": df.width,
        "start": _iso(int(df["close_time"].min())),
        "end": _iso(int(df["close_time"].max())),
        "kline_src": KLINE_SRC,
        "with_basis": WITH_BASIS,
        "ob_tag": OB_TAG,
        "kline_tag": KLINE_TAG,
        "ob_months": OB_MONTHS or "all",
        "feature_cols": [*kcols, *ob_cols],
        "target_anchor": "close",
    }
    OUT.with_suffix(".meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"写出 {OUT} {OUT.stat().st_size / 1e6:.1f} MB；特征 {len(meta['feature_cols'])} 列")
    log(f"输出目录: {sorted(p.name for p in OUT.parent.glob('*.parquet'))}")
    if UPLOAD_TAG:
        upload_release(UPLOAD_TAG, OUT)
        upload_release(UPLOAD_TAG, OUT.with_suffix(".meta.json"))


def _ms(day: str) -> int:
    """'YYYY-MM-DD' -> 当天 00:00 UTC 的毫秒戳（close_time 口径）。"""
    d = datetime.fromisoformat(day).replace(tzinfo=timezone.utc)
    return int(d.timestamp() * 1000)


def _iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


if __name__ == "__main__":
    main()
