#!/usr/bin/env python3
"""亚分钟盘口动态（S*）：把订单簿原始事件流压成「每分钟一行」的过程特征。

为什么要它
----------
现役 185 列（`ob-dataset-vol8`）全部来自 `bn_hisdata/orderbook/build_minute_snapshots.py`：
每个分钟桶只取**该分钟最后一条消息后的盘口截面**（mid±0.3% 内的档位）。分钟内的撤单潮、
挂单消耗速度、价差抖动、事件密度被整体丢弃。h=3 上宽表 / 跨域 / K 线 / 两腿融合 / TCN
都已证伪，唯一没试过的就是这条「换数据」。

做法
----
1. 逐小时从 CryptoHFTData 的 R2 拉 `binance_futures` 原始 orderbook parquet
   （列：side/price/quantity/event_time/event_type），算完立即删，磁盘只压一小时。
2. 事件级增量：按 (side, price) 对 quantity 做 diff（snapshot 行作基准），
   delta>0 = 新挂/加量，delta<0 = 撤单/减量。
3. 按分钟聚合出 sb_* ：事件数、挂单量、撤单量、撤单比、流量不平衡、近似最优价/价差、
   最后一条事件距分钟末的毫秒、单笔最大量；再补分钟级派生（mid 位移、30m z-score）。
4. 与 185 列 join，训 `base` / `base+sb` 两版，报 AUC、各覆盖档、配对 bootstrap。

样本量提醒（重要）
------------------
一个月只有 ~43k 分钟，test 约 8.6k 行：@日均15单（约 450 单）的 SE ≈2.3pp，什么都判不出。
本脚本主看 **AUC**、**@25%**（2.1k 单，SE≈1.1pp）和每个 sb 列的**单变量 AUC**。
确认有信号后再把 MONTH 扩到 3~6 个月，那时才谈 0.5pp。

env
---
    MONTH=2026-09     处理哪个月（UTC）。须 ≥2026-08-19：之后是裸 parquet，之前是 .parquet.zst
    SYMBOL=BTCUSDT
    SEEDS=42,43,44,45,46
    EXP=S1
    OUT=/kaggle/working/ob_subminute.parquet   特征落盘（可选）
    MAX_HOURS=0       调试：只处理前 N 小时（0 = 整月）
    KLINE_FROM=2025   取 close 用的 K 线起始年
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import date as ddate
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

try:
    from morph import ob_kline as ok
    from morph.h3_fuse_cross import paired_boot
    from morph.two_legs import (TARGET_TPD, TRAIN_FRAC, VAL_FRAC, add_target,
                                fit, matrix, report)
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from morph import ob_kline as ok
    from morph.h3_fuse_cross import paired_boot
    from morph.two_legs import (TARGET_TPD, TRAIN_FRAC, VAL_FRAC, add_target,
                                fit, matrix, report)

SYMBOL = os.environ.get("SYMBOL", "BTCUSDT")
MONTH = os.environ.get("MONTH", "2026-09")
SEEDS = [int(s) for s in os.environ.get("SEEDS", "42,43,44,45,46").split(",") if s.strip()]
EXP = os.environ.get("EXP", "S?")
OUT = os.environ.get("OUT", "")
MAX_HOURS = int(os.environ.get("MAX_HOURS", "0"))
KLINE_FROM = int(os.environ.get("KLINE_FROM", "2025"))
HORIZON = int(os.environ.get("HORIZON", "3"))

RAW_DIR = Path("/tmp/ob_raw")
CHD_DIR = Path("/tmp/bn_hisdata")
CHD_REPO = "https://github.com/bailiang0369/bn_hisdata"
T = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T:7.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------- 下载
def ensure_chd() -> None:
    """复用 bn_hisdata 的 R2 客户端（s3lite 只依赖 requests，无额外安装）。"""
    if not CHD_DIR.exists():
        subprocess.run(["git", "clone", "--depth=1", CHD_REPO, str(CHD_DIR)], check=True)
    p = str(CHD_DIR / "orderbook")
    if p not in sys.path:
        sys.path.insert(0, p)


def make_client(api_key: str):
    ensure_chd()
    from download_orderbook_s3 import get_credentials
    from s3lite import S3LiteClient
    cr = get_credentials(api_key)
    cl = S3LiteClient(endpoint=cr["endpoint"],
                      access_key_id=cr["credentials"]["access_key_id"],
                      secret_access_key=cr["credentials"]["secret_access_key"],
                      session_token=cr["credentials"].get("session_token"),
                      region=cr.get("region") or "auto",
                      max_pool_connections=8, timeout=600, max_attempts=3)
    return cl, cr["bucket"]


def fetch_hour(cl, bucket: str, day: str, hour: int) -> Path | None:
    ensure_chd()
    from download_orderbook_s3 import _write_payload
    from s3lite import S3LiteError
    dst = RAW_DIR / day / f"{hour:02d}.parquet"
    dst.parent.mkdir(parents=True, exist_ok=True)
    key = f"binance_futures/{day}/{hour:02d}/{SYMBOL}_orderbook.parquet"
    try:
        resp = cl.get_object(Bucket=bucket, Key=key)
    except S3LiteError as e:
        code = e.response.get("Error", {}).get("Code", str(e)[:60])
        log(f"[miss] {day} {hour:02d}: {code}")
        return None
    try:
        _write_payload(resp["Body"], dst, decompress=True)
    finally:
        resp["Body"].close()
    return dst


def drop(p: Path) -> None:
    try:
        p.unlink()
    except OSError:
        pass
    try:
        p.parent.rmdir()
    except OSError:
        pass


# ---------------------------------------------------------------- 特征
def hour_feats(path: Path) -> pl.DataFrame:
    """一小时原始事件流 → 该小时每个分钟桶一行 sb_* 基础聚合。"""
    need = {"side", "price", "quantity", "event_time", "event_type"}
    schema = pl.read_parquet_schema(path)
    if not need.issubset(set(schema)):
        log(f"[skip] {path.name} 列不全: {sorted(schema)}")
        return pl.DataFrame()
    df = pl.read_parquet(path, columns=list(need))
    df = df.with_columns(
        pl.when(pl.col("event_type") == "snapshot").then(1).otherwise(0).alias("_snap"))
    # snapshot 必须排在同一毫秒的 update 之前，否则基准量会被反算
    df = df.sort(["event_time", "_snap"], descending=[False, True])
    df = df.with_columns(
        (pl.col("quantity") - pl.col("quantity").shift(1).over(["side", "price"]))
        .alias("_d"))
    u = df.filter((pl.col("event_type") == "update") & pl.col("_d").is_not_null())
    if u.height == 0:
        return pl.DataFrame()
    bid = pl.col("side") == "bid"
    u = u.with_columns(
        (pl.col("event_time") // 60_000).alias("_m"),
        pl.when(pl.col("_d") > 0).then(pl.col("_d")).otherwise(0.0).alias("_add"),
        pl.when(pl.col("_d") < 0).then(-pl.col("_d")).otherwise(0.0).alias("_cxl"),
    )
    return u.group_by("_m").agg(
        pl.len().alias("sb_n"),
        bid.sum().alias("sb_n_bid"),
        (~bid).sum().alias("sb_n_ask"),
        pl.when(bid).then(pl.col("_add")).otherwise(0.0).sum().alias("sb_add_bid"),
        pl.when(~bid).then(pl.col("_add")).otherwise(0.0).sum().alias("sb_add_ask"),
        pl.when(bid).then(pl.col("_cxl")).otherwise(0.0).sum().alias("sb_cxl_bid"),
        pl.when(~bid).then(pl.col("_cxl")).otherwise(0.0).sum().alias("sb_cxl_ask"),
        pl.when(bid).then(pl.col("price")).otherwise(pl.lit(None, dtype=pl.Float64))
        .max().alias("sb_pmax_bid"),
        pl.when(~bid).then(pl.col("price")).otherwise(pl.lit(None, dtype=pl.Float64))
        .min().alias("sb_pmin_ask"),
        pl.col("quantity").max().alias("sb_qmax"),
        pl.col("quantity").mean().alias("sb_qmean"),
        (pl.col("event_time").max() % 60_000).alias("sb_last_ms"),
    ).sort("_m")


def add_derived(feat: pl.DataFrame) -> pl.DataFrame:
    feat = feat.with_columns(
        ((pl.col("sb_pmax_bid") + pl.col("sb_pmin_ask")) / 2).alias("sb_mid"),
        (pl.col("sb_add_bid") - pl.col("sb_cxl_bid")).alias("_nb"),
        (pl.col("sb_add_ask") - pl.col("sb_cxl_ask")).alias("_na"),
    )
    feat = feat.with_columns(
        ((pl.col("_nb") - pl.col("_na")) / ((pl.col("_nb") + pl.col("_na")).abs() + 1.0))
        .alias("sb_flow"),
        ((pl.col("sb_cxl_bid") + pl.col("sb_cxl_ask"))
         / (pl.col("sb_add_bid") + pl.col("sb_add_ask")
            + pl.col("sb_cxl_bid") + pl.col("sb_cxl_ask") + 1.0)).alias("sb_cxl_ratio"),
        (pl.col("sb_pmin_ask") - pl.col("sb_pmax_bid")).alias("sb_spread_approx"),
        (60_000 - pl.col("sb_last_ms")).alias("sb_gap_ms"),
    )
    return feat.with_columns(
        (pl.col("sb_mid") - pl.col("sb_mid").shift(1)).alias("sb_mid_d1"),
        ((pl.col("sb_mid") - pl.col("sb_mid").rolling_mean(30))
         / (pl.col("sb_mid").rolling_std(30) + 1e-9)).alias("sb_mid_z"),
        ((pl.col("sb_flow") - pl.col("sb_flow").rolling_mean(30))
         / (pl.col("sb_flow").rolling_std(30) + 1e-9)).alias("sb_flow_z"),
        ((pl.col("sb_cxl_ratio") - pl.col("sb_cxl_ratio").rolling_mean(30))
         / (pl.col("sb_cxl_ratio").rolling_std(30) + 1e-9)).alias("sb_cxl_z"),
        ((pl.col("sb_n") - pl.col("sb_n").rolling_mean(30))
         / (pl.col("sb_n").rolling_std(30) + 1e-9)).alias("sb_n_z"),
    ).drop(["_nb", "_na"])


def main() -> None:
    api_key = os.environ.get("CHD_API_KEY")
    if not api_key:
        raise SystemExit("缺 CHD_API_KEY（本地 .codebuddy/settings.json 注入）")

    y0, m0 = (int(x) for x in MONTH.split("-"))
    d0 = ddate(y0, m0, 1)
    d1 = ddate(y0 + (m0 == 12), m0 % 12 + 1, 1)
    days = [d0 + timedelta(days=i) for i in range((d1 - d0).days)]
    ms0 = int(datetime(y0, m0, 1, tzinfo=timezone.utc).timestamp() * 1000)
    ms1 = int(datetime(d1.year, d1.month, 1, tzinfo=timezone.utc).timestamp() * 1000)
    log(f"{EXP} 月={MONTH} 天={len(days)} 种子={SEEDS} 上限={MAX_HOURS or '整月'}")

    cl, bucket = make_client(api_key)
    t_cli = time.time()
    log("R2 凭证就绪")

    frames: list[pl.DataFrame] = []
    done = miss = 0
    for day in days:
        for hour in range(24):
            if MAX_HOURS and done >= MAX_HOURS:
                break
            if time.time() - t_cli > 1500:          # R2 临时凭证约 1 小时，提前换
                log("刷新 R2 凭证")
                cl, bucket = make_client(api_key)
                t_cli = time.time()
            p = fetch_hour(cl, bucket, day.isoformat(), hour)
            if p is None:
                miss += 1
                continue
            try:
                f = hour_feats(p)
            finally:
                drop(p)
            if f.height:
                frames.append(f)
                done += 1
            if done and done % 48 == 0:
                log(f"{done} 小时（缺 {miss}）| 累计 "
                    f"{sum(x.height for x in frames):,} 分钟")
        if MAX_HOURS and done >= MAX_HOURS:
            break
    if not frames:
        raise SystemExit("没取到任何数据，检查 MONTH / 权限")

    feat = pl.concat(frames).sort("_m").unique(subset=["_m"], keep="last")
    log(f"聚合完成：{feat.height:,} 个分钟桶（{done} 小时，缺 {miss}）")
    log("样例: " + str(feat.head(3).to_dicts()[0]))

    # 补齐连续分钟：缺分钟的滚动窗口按日历时间滑过，不能压缩行序
    grid = pl.DataFrame({"_m": np.arange(ms0 // 60_000, ms1 // 60_000, dtype=np.int64)})
    feat = grid.join(feat, on="_m", how="left").with_columns(
        pl.col("sb_n").is_null().cast(pl.Int8).alias("sb_missing"))
    feat = add_derived(feat).with_columns(
        ((pl.col("_m") + 1) * 60_000).cast(pl.Int64).alias("close_time"))
    sb_cols = [c for c in feat.columns if c.startswith("sb_")]
    log(f"sb 特征 {len(sb_cols)} 列，缺失分钟 {int(feat['sb_missing'].sum()):,}")
    if OUT:
        feat.select("close_time", *sb_cols).write_parquet(OUT)
        log(f"特征落盘: {OUT}")

    # ------------------------------------------------------------ 训练对比
    ob = ok.load_ob().filter((pl.col("close_time") >= ms0) & (pl.col("close_time") < ms1))
    base = [c for c in ob.columns if c != "close_time"]
    kf, _ = ok.load_kline_features(KLINE_FROM)
    data = (feat.join(ob, on="close_time", how="left")
                .join(kf.select("close_time", "close"), on="close_time", how="left"))
    data = add_target(data, HORIZON).filter(
        pl.col("target").is_not_null() & pl.col("close").is_not_null()).sort("close_time")
    n = data.height
    tr1, va1 = int(n * TRAIN_FRAC), int(n * (TRAIN_FRAC + VAL_FRAC))
    y = data["target"].to_numpy().astype(int)
    days_arr = (data["close_time"].to_numpy() // 86_400_000)
    log(f"样本 {n:,}（train {tr1:,} / val {va1 - tr1:,} / test {n - va1:,}）")

    tr, va, te = data[:tr1], data[tr1:va1], data[va1:]
    ytr, yva, yte = y[:tr1], y[tr1:va1], y[va1:]
    dva, dte = days_arr[tr1:va1], days_arr[va1:]
    log(f"test 天数 {len(np.unique(dte))}，@日均{TARGET_TPD}单 ≈ "
        f"{TARGET_TPD * len(np.unique(dte))} 单（SE≈2pp，只能看 AUC）")

    # 单变量 AUC：sb 列单独有没有信息（这一步不受样本量拖累）
    singles = []
    for c in sb_cols:
        x = te[c].cast(pl.Float64).fill_null(float("nan")).to_numpy()
        okm = np.isfinite(x)
        if okm.sum() < len(yte) * 0.2 or np.unique(x[okm]).size < 10:
            continue
        a = roc_auc_score(yte[okm], x[okm])
        singles.append((c, max(a, 1 - a)))
    singles.sort(key=lambda t: -t[1])
    log("单变量 AUC Top10: " + ", ".join(f"{c}={a:.3f}" for c, a in singles[:10]))

    out = {}
    for name, cols in (("base", base), ("base+sb", base + sb_cols)):
        p_va = np.zeros(va1 - tr1)
        p_te = np.zeros(n - va1)
        imp = np.zeros(len(cols))
        for seed in SEEDS:
            m = fit(matrix(tr, cols), ytr, cols, matrix(va, cols), yva, seed=seed)
            p_va += m.predict(matrix(va, cols)) / len(SEEDS)
            p_te += m.predict(matrix(te, cols)) / len(SEEDS)
            imp += np.array(m.feature_importance("gain")) / len(SEEDS)
            log(f"[{name}] seed={seed} best_iter={m.best_iteration}")
        auc_va, auc_te = roc_auc_score(yva, p_va), roc_auc_score(yte, p_te)
        w_va = report(f"{EXP} {name} val", p_va - 0.5, yva, dva)
        w_te = report(f"{EXP} {name} test", p_te - 0.5, yte, dte)
        log(f"[{EXP} {name}] AUC val={auc_va:.4f} test={auc_te:.4f} | "
            f"SUMMARY {name} cols={len(cols)} val={w_va * 100:.2f}% test={w_te * 100:.2f}%")
        if len(sb_cols) and name == "base+sb":
            order = np.argsort(imp)[::-1][:10]
            log("gain Top10: " + ", ".join(f"{cols[i]}({imp[i] / imp.sum() * 100:.1f}%)"
                                           for i in order))
        out[name] = dict(variant=name, score=p_te - 0.5, y=yte, days=dte)

    paired_boot(out["base"], out["base+sb"], f"配对[{EXP}]")
    log("DONE")


if __name__ == "__main__":
    main()
