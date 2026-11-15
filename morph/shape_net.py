#!/usr/bin/env python3
"""价格形态端到端学习（h=10 同口径），**独立新文件，不改动任何现役代码**。

为什么要新开一个文件
--------------------
2026-11-15 判定：形态不该先给出「窗口几根 / 怎么归一化 / 用哪种距离」的定义再让算法去匹配，
而是让模型自己从原始序列里学表示。历史三次形态尝试都是「固定窗 + 刚体距离」：
  - `kline_shape.py` 第 8 轮：固定 W=60、每 6 根采样成 10 点、KMeans(64)
  - faiss 形态腿：固定 10/15/30/60/120 五档窗口 + 逐窗 min-max + 欧氏
  - `morph_ts`：固定 60/120 根窗 CNN（且早停第 3 轮，明显欠拟合）
它们都无法识别「同一套路、不同时间伸缩」的形态，因此按 `纯K线提升实验记录.md` §0/§7，
**不构成"形态有没有用"的证据**。本文件按端到端重做。

为什么不用 DTW
--------------
单窗 train 1.79M 行 → stride 抽样后 18 万行，shapelet 库 1000 条、平均长度 40 根，
FastDTW ≈100μs/次 → 约 5 小时/窗，10 窗 × 5 seed 不可行；样本砍到能跑的量又会空窗。
改用**多尺度卷积 + 时间轴池化**表达尺度不变性：同一形态被拉长 2 倍由长核匹配、
压缩一半由短核匹配，伸缩是学出来的，不是枚举对齐出来的。

设计
----
1. 通道（7 个，全部无量纲，不含任何手工指标）：
     ret1 单根对数收益(bps) / body 实体 / up 上影 / dn 下影 / rng 振幅 /
     lvol 对数成交量 / tbr 主动买占比
   **没有**动量、均线、布林、RSI、区间位置、波动率——这些交给卷积自己组合。
2. 尺不变：卷积核 4/8/16/32/64 根 × 时间轴 max+avg 双池化；再拼 MLP 出 logit。
3. 窗内逐通道 z-score（只留形状、丢掉量纲与波动水平），可 NORM=0 关闭。
4. 训练/评估纪律与 `h10_signal.py` 完全一致（逐月 walk-forward、train/es/thr 三段、
   阈值只在 thr 段定、段末 purge HORIZON 根防标签穿越、可开 placebo），
   所以胜率可以直接和现役 LightGBM 的 63.32%@19.9 单/天 对撞。

env
---
  HORIZON=10 WIN=240 CH=7 EPOCHS=6 BATCH=1024 MAX_TRAIN=400000 LR=3e-3
  SEEDS=42,43 WIN_DAYS=1460 ES_FRAC=0.05 THR_FRAC=0.10 STRIDE=10
  START_MONTH=2026-01 END_MONTH=2026-10 TPD_LIST=2,5,8,15,30 PRIMARY_TPD=8
  BARS_PER_DAY=1440 PAYOUTS=0.80,0.828 PLACEBO=0 NORM=1 CONTEXT=0

用法
----
  python -m morph.shape_net                    # 本地（CPU 也能跑，慢）
  python kaggle_launcher.py --project shapenet # Kaggle（GPU，见 launcher 里 gpu=True）
"""
from __future__ import annotations

import datetime as dt
import os
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from morph import kline_long as kl  # noqa: E402
from morph.two_legs import add_target  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

HORIZON = int(os.environ.get("HORIZON", "10"))
WIN = int(os.environ.get("WIN", "240"))          # 窗口根数（1m bar）
EPOCHS = int(os.environ.get("EPOCHS", "6"))
BATCH = int(os.environ.get("BATCH", "1024"))
MAX_TRAIN = int(os.environ.get("MAX_TRAIN", "400000"))
LR = float(os.environ.get("LR", "3e-3"))
SEEDS = [int(s) for s in os.environ.get("SEEDS", "42,43").split(",") if s.strip()]
WIN_DAYS = int(os.environ.get("WIN_DAYS", "1460"))
ES_FRAC = float(os.environ.get("ES_FRAC", "0.05"))
THR_FRAC = float(os.environ.get("THR_FRAC", "0.10"))
STRIDE = int(os.environ.get("STRIDE", "10"))
START_MONTH = os.environ.get("START_MONTH", "2026-01")
END_MONTH = os.environ.get("END_MONTH", "2026-10")
TPD_LIST = [float(x) for x in os.environ.get("TPD_LIST", "2,5,8,15,30").split(",") if x.strip()]
PRIMARY_TPD = float(os.environ.get("PRIMARY_TPD", "8"))
BARS_PER_DAY = int(os.environ.get("BARS_PER_DAY", "1440"))
PAYOUTS = [float(x) for x in os.environ.get("PAYOUTS", "0.80,0.828").split(",") if x.strip()]
PLACEBO = os.environ.get("PLACEBO", "0") == "1"
NORM = os.environ.get("NORM", "1") == "1"
CONTEXT = os.environ.get("CONTEXT", "0") == "1"
KERNELS = tuple(int(k) for k in os.environ.get("KERNELS", "4,8,16,32,64").split(",") if k.strip())
DEV = "cuda" if torch.cuda.is_available() else "cpu"
T = time.time()

BASE_CH = ["ret1", "body", "up", "dn", "rng", "lvol", "tbr"]
CTX_CH = ["hsin", "hcos", "lrv"]


def log(msg: str) -> None:
    print(f"[{time.time() - T:7.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------- 数据
def build_channels(df: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    """7 个无量纲通道，全部只用当根及之前的数据。"""
    c = df.columns
    vol_n = next((x for x in ("volume", "um_volume") if x in c), None)
    tb_n = next((x for x in ("taker_buy_volume", "um_taker_buy_vol") if x in c), None)

    o = pl.col("open").cast(pl.Float64)
    h = pl.col("high").cast(pl.Float64)
    lo = pl.col("low").cast(pl.Float64)
    cl = pl.col("close").cast(pl.Float64)
    e = [
        ((cl / cl.shift(1)).log() * 1e4).alias("ret1"),
        ((cl - o) / cl * 1e4).alias("body"),
        ((h - pl.max_horizontal(o, cl)) / cl * 1e4).alias("up"),
        ((pl.min_horizontal(o, cl) - lo) / cl * 1e4).alias("dn"),
        ((h - lo) / cl * 1e4).alias("rng"),
    ]
    if vol_n:
        v = pl.col(vol_n).cast(pl.Float64)
        e.append(v.log1p().alias("lvol"))
        if tb_n:
            e.append((pl.col(tb_n).cast(pl.Float64) / (v + 1e-9) - 0.5).alias("tbr"))
        else:
            e.append(pl.lit(0.0).alias("tbr"))
    else:
        e += [pl.lit(0.0).alias("lvol"), pl.lit(0.0).alias("tbr")]
    out = df.with_columns(e)
    ch = list(BASE_CH)
    if CONTEXT:
        # 时段与波动水平属于「上下文」，不是价格形状；只在前景对照里开
        hour = (pl.col("close_time") // 3_600_000 % 24).cast(pl.Float64)
        out = out.with_columns([
            (hour * 2 * np.pi / 24).sin().alias("hsin"),
            (hour * 2 * np.pi / 24).cos().alias("hcos"),
            (pl.col("ret1").rolling_std(1440, min_samples=240).log1p()).alias("lrv"),
        ])
        ch += CTX_CH
    return out, ch


def load_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = kl.load_raw()
    df, ch = build_channels(raw)
    df = add_target(df, HORIZON).filter(
        pl.col("target").is_not_null() & pl.col("close").is_not_null()).sort("close_time")
    arr = (df.select([pl.col(k).cast(pl.Float32) for k in ch]).to_numpy()
           .astype(np.float32, copy=False))
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if NORM:
        # 通道级粗限幅：极端插针（新闻/闪崩）会在窗内 z-score 里放大成噪声
        for j in range(arr.shape[1]):
            sd = float(np.std(arr[:, j])) or 1.0
            arr[:, j] = np.clip(arr[:, j], -8 * sd, 8 * sd)
    y = df["target"].to_numpy().astype(np.int8)
    ct = df["close_time"].cast(pl.Int64).to_numpy()
    log(f"通道 {len(ch)} 列 {ch} | 可用 {df.height:,} 行 | WIN={WIN} | dev={DEV}")
    return arr, y, ct


def make_batch(arr: np.ndarray, idx: np.ndarray) -> torch.Tensor:
    """(B, C, WIN) 张量；窗口结束于 idx（含当根），只用过去。"""
    off = np.arange(WIN) - (WIN - 1)
    x = arr[idx[:, None] + off[None, :]]                    # (B, WIN, C)
    x = np.transpose(x, (0, 2, 1)).astype(np.float32, copy=False)
    t = torch.from_numpy(x)
    if NORM:
        m = t.mean(dim=2, keepdim=True)
        s = t.std(dim=2, keepdim=True).clamp_min(1e-3)
        t = (t - m) / s
    return t


# ---------------------------------------------------------------- 模型
class MultiScale(nn.Module):
    """多尺度因果卷积 + 时间轴双池化 → 尺度不变表示。

    核长 4/8/16/32/64 覆盖约 16 倍尺度跨度：同一形态被拉长/压缩时，
    由不同核长的分支匹配，伸缩不需要显式对齐。
    """

    def __init__(self, cin: int, kernels: tuple[int, ...] = KERNELS, co: int = 32):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(cin, co, k, padding=k // 2), nn.BatchNorm1d(co), nn.ReLU(),
                nn.Conv1d(co, co, k, padding=k // 2), nn.BatchNorm1d(co), nn.ReLU())
            for k in kernels])
        self.head = nn.Sequential(
            nn.Linear(co * len(kernels) * 2, 64), nn.ReLU(),
            nn.Dropout(0.10), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = []
        for b in self.branches:
            y = b(x)
            f.append(y.amax(dim=2))
            f.append(y.mean(dim=2))
        return self.head(torch.cat(f, dim=1)).squeeze(1)


def predict(model: nn.Module, arr: np.ndarray, idx: np.ndarray, bs: int = 2048) -> np.ndarray:
    model.eval()
    out = np.empty(len(idx), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            xb = make_batch(arr, idx[i:i + bs]).to(DEV, non_blocking=True)
            out[i:i + bs] = torch.sigmoid(model(xb)).float().cpu().numpy()
    return out


def train_one(arr: np.ndarray, idx_tr: np.ndarray, y_tr: np.ndarray,
              idx_es: np.ndarray, y_es: np.ndarray, seed: int) -> tuple[nn.Module, float]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MultiScale(arr.shape[1]).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(seed)
    best_auc, best_state = -1.0, None
    for ep in range(EPOCHS):
        model.train()
        perm = rng.permutation(len(idx_tr))
        tot = 0.0
        for i in range(0, len(perm), BATCH):
            sel = perm[i:i + BATCH]
            xb = make_batch(arr, idx_tr[sel]).to(DEV, non_blocking=True)
            yb = torch.from_numpy(y_tr[sel].astype(np.float32)).to(DEV, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = lossf(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss) * len(sel)
        p_es = predict(model, arr, idx_es)
        auc = float(roc_auc_score(y_es, p_es))
        log(f"    ep{ep + 1}/{EPOCHS} loss={tot / max(1, len(idx_tr)):.4f} es_auc={auc:.4f}")
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_auc


# ---------------------------------------------------------------- 评估
def roll_worst(w: np.ndarray, n: int = 100) -> tuple[float, float]:
    if len(w) < n:
        return float("nan"), float("nan")
    cs = np.concatenate([[0], np.cumsum(w)])
    r = (cs[n:] - cs[:-n]) / n
    return float(r.min()), float((r < .5).mean())


def _run(arr: np.ndarray, y: np.ndarray, ct: np.ndarray) -> None:
    months = [m for m in sorted(set(
        pl.Series(ct).cast(pl.Datetime("ms")).dt.strftime("%Y-%m").to_list()))
        if START_MONTH <= m <= END_MONTH]
    log(f"walk-forward {len(months)} 窗（{months[0]}~{months[-1]}）| H={HORIZON} | "
        f"窗 {WIN_DAYS} 天 | es={ES_FRAC:.0%} thr={THR_FRAC:.0%} | seeds={len(SEEDS)} | "
        f"stride={STRIDE} | placebo={int(PLACEBO)}")

    per: dict[float, dict[str, list]] = {t: dict(s=[], y=[], mon=[]) for t in TPD_LIST}
    rows = []
    for m in months:
        m_start = int(dt.datetime.strptime(m + "-01", "%Y-%m-%d")
                      .replace(tzinfo=dt.timezone.utc).timestamp() * 1000)
        lo = m_start - WIN_DAYS * 86_400_000
        i_tr = np.where((ct < m_start) & (ct >= lo))[0]
        # test 段 = 自然月：下个月 1 号做上界
        nxt = (dt.datetime.strptime(m + "-01", "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)
               + dt.timedelta(days=32)).replace(day=1)
        i_te = np.where((ct >= m_start) & (ct < int(nxt.timestamp() * 1000)))[0]
        if len(i_tr) < 100_000 or len(i_te) < 1_000:
            log(f"[{m}] 样本不足（{len(i_tr):,}/{len(i_te):,}），跳过")
            continue
        h_all = len(i_tr)
        c1 = int(h_all * (1 - ES_FRAC - THR_FRAC))
        c2 = int(h_all * (1 - THR_FRAC))
        # 段末 purge HORIZON 根：这些行的标签落到下一段/未来，必须丢掉
        seg = [i_tr[:max(0, c1 - HORIZON)], i_tr[c1:c2 - HORIZON], i_tr[c2:]]
        if STRIDE > 1:
            seg[0] = seg[0][::STRIDE]
            seg[1] = seg[1][::STRIDE]
        i_tr_s, i_es, i_th = seg
        i_tr_s = i_tr_s[np.clip(i_tr_s, WIN - 1, None) == i_tr_s]
        i_es = i_es[np.clip(i_es, WIN - 1, None) == i_es]
        i_th = i_th[np.clip(i_th, WIN - 1, None) == i_th]
        i_te = i_te[np.clip(i_te, WIN - 1, None) == i_te]
        if len(i_tr_s) > MAX_TRAIN:
            i_tr_s = i_tr_s[:: max(1, len(i_tr_s) // MAX_TRAIN)][:MAX_TRAIN]

        y_tr = y[i_tr_s].astype(np.int8)
        if PLACEBO:
            y_tr = y_tr.copy()
            np.random.default_rng(20261115).shuffle(y_tr)
        y_es, y_th, y_te = y[i_es], y[i_th], y[i_te]

        p_th = np.zeros(len(i_th), dtype=np.float32)
        p_te = np.zeros(len(i_te), dtype=np.float32)
        aucs = []
        for seed in SEEDS:
            mdl, a = train_one(arr, i_tr_s, y_tr, i_es, y_es, seed)
            p_th += predict(mdl, arr, i_th) / len(SEEDS)
            p_te += predict(mdl, arr, i_te) / len(SEEDS)
            aucs.append(a)
            if DEV == "cuda":
                torch.cuda.empty_cache()

        auc = float(roc_auc_score(y_te, p_te))
        days = (ct[i_te] // 86_400_000).astype(np.int64)
        nd = len(np.unique(days))
        s = p_te - .5
        line = (f"[{m}] train={len(i_tr_s):,} es={len(i_es):,} thr={len(i_th):,} "
                f"test={len(i_te):,}({nd}天) esAUC={np.mean(aucs):.4f} testAUC={auc:.4f}")
        for tpd in TPD_LIST:
            thr = float(np.quantile(np.abs(p_th - .5), 1 - tpd / BARS_PER_DAY))
            hit = np.abs(s) >= thr
            nb = int(hit.sum())
            if nb < 30:
                line += f" | tpd{tpd:g}:单量太少"
                continue
            w = ((s[hit] > 0).astype(int) == y_te[hit]).astype(np.int8)
            per[tpd]["s"].append(s[hit])
            per[tpd]["y"].append(y_te[hit])
            per[tpd]["mon"].append(float(w.mean()))
            if tpd == PRIMARY_TPD:
                line += f" | 阈值={thr:.4f} {nb / nd:.1f}单/天 胜率={w.mean() * 100:.2f}%"
        rows.append(dict(ym=m, nd=nd, auc=auc))
        log(line)

    if not rows:
        raise SystemExit("无窗口完成")

    ev_hdr = "".join(f"{'EV@' + format(p, 'g'):>10}" for p in PAYOUTS)
    print(f"\n{'名义tpd':>8}{'实际单/天':>10}{'n':>7}{'胜率':>9}{'SE*':>7}{ev_hdr}"
          f"{'月min':>8}{'月std':>7}{'破50%月':>8}{'滚动100min':>11}{'破50%窗':>9}")
    nd_all = sum(r["nd"] for r in rows)
    for tpd in TPD_LIST:
        p = per[tpd]
        if not p["y"]:
            continue
        yv = np.concatenate(p["y"])
        sv = np.concatenate(p["s"])
        w = ((sv > 0).astype(int) == yv).astype(np.int8)
        win = float(w.mean())
        mon = np.array(p["mon"])
        se = float(np.sqrt(win * (1 - win) / max(len(w) / HORIZON, 2)))
        rmin, rbad = roll_worst(w)
        evs = "".join(f"{pay * win - (1 - win):>+10.3f}" for pay in PAYOUTS)
        star = " ← 主操作点" if tpd == PRIMARY_TPD else ""
        print(f"{tpd:>8.1f}{len(w) / nd_all:>10.1f}{len(w):>7d}{win * 100:>8.2f}%"
              f"{se * 100:>6.2f}%{evs}{mon.min() * 100:>7.2f}%{mon.std() * 100:>6.2f}pp"
              f"{int((mon < .5).sum()):>5}/{len(mon)}{rmin * 100:>10.1f}%"
              f"{rbad * 100:>8.1f}%{star}")
    log("DONE")


def synthetic(n: int = 240_000) -> None:
    """无数据烟测：只证明链路通 + 早停/阈值/汇总逻辑不炸，数值无意义。"""
    global EPOCHS, WIN, WIN_DAYS, MAX_TRAIN, SEEDS, BATCH, STRIDE, START_MONTH, END_MONTH
    EPOCHS, WIN, WIN_DAYS, MAX_TRAIN, SEEDS, BATCH, STRIDE = 1, 64, 120, 20_000, [42], 512, 5
    START_MONTH, END_MONTH = "2024-01", "2024-12"
    rng = np.random.default_rng(0)
    n = max(n, 200_000)
    r = rng.standard_normal(n).astype(np.float32)
    arr = np.stack([r, r * .6, np.abs(r) * .4, np.abs(r) * .4, np.abs(r),
                    rng.random(n, dtype=np.float32), rng.standard_normal(n).astype(np.float32) * .1], 1)
    fut = np.concatenate([r[1:], [0]])
    y = ((fut + 0.02 * r) > 0).astype(np.int8)      # 埋一点弱信号，保证 AUC 不是 0.5
    ct = (1704067200000 + np.arange(n) * 60_000).astype(np.int64)
    log(f"[synthetic] arr={arr.shape} 正例率={y.mean():.3f}")
    _run(arr, y, ct)


def main() -> None:
    arr, y, ct = load_arrays()
    _run(arr, y, ct)


if __name__ == "__main__":
    if "--synthetic" in sys.argv:
        synthetic()
    else:
        main()
