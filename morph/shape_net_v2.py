#!/usr/bin/env python3
"""端到端形态模型 v2：修复 v1 的三处已知缺陷，首跑验证「纯形态能否达到 LGBM 量级」。

为什么不改 v1 / 为什么新建（审查 2026-11-22）
------------------------------------------------
`shape_net.py`（v1）的架构是对的（多尺度卷积 + 时间轴双池化表达时间伸缩，绕开 faiss
固定窗/刚体距离和 FastDTW 的慢），但从未运行过，且带三个会重蹈 morph_ts 覆辙的缺陷：
  1. 偶数核 `Conv1d(k, padding=k//2)`：输出长度 WIN+1、双池化吃到 padding 的 0 边界，
     且 k//2 对偶数核不是严格因果语义。v2 改 `F.pad(x,(k-1,0)) + conv(padding=0)`：
     输出 i 只看 x[i-k+1..i]，长度恒为 WIN。
  2. 早停用 es_AUC，但验收是尾部 top-tpd 胜率 —— morph_ts v2/v3 四次改动全部
     「模型指标变好、尾部变差」(详录 §14/§15)。v2 早停改 es 段尾部胜率。
  3. STRIDE=10 / MAX_TRAIN=40万 在 GPU 上没吃满数据（欠拟合伪问题）。v2 用 GPU 扩样本
     + cosine LR 退火，而不是硬拉 epoch。

四段职责（严格因果，与 h10_signal 一致）
----------------------------------------
train 训练 / es 早停(尾部胜率) / thr 定阈值(|p-.5| 分位) / test 只评估。
段末各 purge HORIZON 根防标签穿越。窗口结束于 idx（含当根），只用过去。

预登记判据（首跑前写死，禁止事后挑）
-------------------------------------
主操作点 tpd=8：
  >= 63.32%  -> 纯形态可替代现役 LGBM，进入多段复核
  58~63.32%  -> 有信号但不及 LGBM，仅考虑小权重融合（0.1~0.25 起扫）
  <  58%     -> 判死，不扩 seed / 不扩窗（与 launcher shapenet 判据一致）
同时报工作日/周末拆分（用户核心目标），但拆分只作诊断，不用于选参。

用法（只在 Kaggle 跑，GPU）
---------------------------
  python kaggle_launcher.py --project shapenetv2
  （新 slug byh850515/btcusdt-shape-net-v2，不覆盖旧 kernel）
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
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from morph import kline_long as kl  # noqa: E402
from morph.two_legs import add_target  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

HORIZON = int(os.environ.get("HORIZON", "10"))
WIN = int(os.environ.get("WIN", "240"))
EPOCHS = int(os.environ.get("EPOCHS", "8"))
BATCH = int(os.environ.get("BATCH", "512"))
MAX_TRAIN = int(os.environ.get("MAX_TRAIN", "1500000"))
LR = float(os.environ.get("LR", "2e-3"))
WD = float(os.environ.get("WD", "1e-4"))
SEEDS = [int(s) for s in os.environ.get("SEEDS", "42,43,44").split(",") if s.strip()]
WIN_DAYS = int(os.environ.get("WIN_DAYS", "1460"))
ES_FRAC = float(os.environ.get("ES_FRAC", "0.05"))
THR_FRAC = float(os.environ.get("THR_FRAC", "0.10"))
STRIDE = int(os.environ.get("STRIDE", "2"))
START_MONTH = os.environ.get("START_MONTH", "2026-01")
END_MONTH = os.environ.get("END_MONTH", "2026-10")
TPD_LIST = [float(x) for x in os.environ.get("TPD_LIST", "2,5,8,15,30").split(",") if x.strip()]
PRIMARY_TPD = float(os.environ.get("PRIMARY_TPD", "8"))
BARS_PER_DAY = int(os.environ.get("BARS_PER_DAY", "1440"))
PAYOUTS = [float(x) for x in os.environ.get("PAYOUTS", "0.80,0.828").split(",") if x.strip()]
PLACEBO = os.environ.get("PLACEBO", "0") == "1"
NORM = os.environ.get("NORM", "1") == "1"
KERNELS = tuple(int(k) for k in os.environ.get("KERNELS", "4,8,16,32,64").split(",") if k.strip())
CO = int(os.environ.get("CO", "32"))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
T = time.time()
BASE_CH = ["ret1", "body", "up", "dn", "rng", "lvol", "tbr"]


def log(msg: str) -> None:
    print(f"[{time.time() - T:7.1f}s] {msg}", flush=True)


def build_channels(df: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    c = df.columns
    vol_n = next((x for x in ("volume", "um_volume") if x in c), None)
    tb_n = next((x for x in ("taker_buy_volume", "um_taker_buy_vol") if x in c), None)
    o, h, lo, cl = (pl.col("open").cast(pl.Float64), pl.col("high").cast(pl.Float64),
                    pl.col("low").cast(pl.Float64), pl.col("close").cast(pl.Float64))
    e = [((cl / cl.shift(1)).log() * 1e4).alias("ret1"),
         ((cl - o) / cl * 1e4).alias("body"),
         ((h - pl.max_horizontal(o, cl)) / cl * 1e4).alias("up"),
         ((pl.min_horizontal(o, cl) - lo) / cl * 1e4).alias("dn"),
         ((h - lo) / cl * 1e4).alias("rng")]
    if vol_n:
        v = pl.col(vol_n).cast(pl.Float64)
        e.append(v.log1p().alias("lvol"))
        e.append((pl.col(tb_n).cast(pl.Float64) / (v + 1e-9) - 0.5).alias("tbr")
                 if tb_n else pl.lit(0.0).alias("tbr"))
    else:
        e += [pl.lit(0.0).alias("lvol"), pl.lit(0.0).alias("tbr")]
    out = df.with_columns(e)
    return out, list(BASE_CH)


def load_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = kl.load_raw()
    df, ch = build_channels(raw)
    df = add_target(df, HORIZON).filter(
        pl.col("target").is_not_null() & pl.col("close").is_not_null()).sort("close_time")
    arr = (df.select([pl.col(k).cast(pl.Float32) for k in ch]).to_numpy()
           .astype(np.float32, copy=False))
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if NORM:
        for j in range(arr.shape[1]):
            sd = float(np.std(arr[:, j])) or 1.0
            arr[:, j] = np.clip(arr[:, j], -8 * sd, 8 * sd)
    y = df["target"].to_numpy().astype(np.int8)
    ct = df["close_time"].cast(pl.Int64).to_numpy()
    log(f"通道 {len(ch)} {ch} | 可用 {df.height:,} 行 | WIN={WIN} | dev={DEV} "
        f"| kernels={KERNELS} co={CO}")
    return arr, y, ct


def make_batch(arr: np.ndarray, idx: np.ndarray) -> torch.Tensor:
    off = np.arange(WIN) - (WIN - 1)
    x = np.transpose(arr[idx[:, None] + off[None, :]], (0, 2, 1)).astype(np.float32, copy=False)
    t = torch.from_numpy(x)
    if NORM:
        t = (t - t.mean(dim=2, keepdim=True)) / t.std(dim=2, keepdim=True).clamp_min(1e-3)
    return t


class MultiScaleCausal(nn.Module):
    """严格因果多尺度卷积 + 时间轴双池化。输出 i 只看 x[i-k+1..i]。"""

    def __init__(self, cin: int):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(cin, CO, k, padding=0), nn.BatchNorm1d(CO), nn.ReLU(),
                nn.Conv1d(CO, CO, k, padding=0), nn.BatchNorm1d(CO), nn.ReLU())
            for k in KERNELS])
        self.kernels = KERNELS
        self.head = nn.Sequential(
            nn.Linear(CO * len(KERNELS) * 2, 64), nn.ReLU(),
            nn.Dropout(0.10), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = []
        for k, branch in zip(self.kernels, self.convs):
            y = branch(F.pad(x, (k - 1, 0)))   # 左侧 pad k-1，右侧 0：严格因果，长度不变
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


def tail_win(p: np.ndarray, y: np.ndarray, tpd: float) -> tuple[float, int]:
    """尾部 top-tpd 命中后的方向胜率（与验收同口径）。"""
    s = p - 0.5
    thr = float(np.quantile(np.abs(s), 1 - min(tpd / BARS_PER_DAY, 0.5)))
    hit = np.abs(s) >= thr
    if hit.sum() < 30:
        return 0.5, int(hit.sum())
    return float(((s[hit] > 0).astype(int) == y[hit]).mean()), int(hit.sum())


def train_one(arr, idx_tr, y_tr, idx_es, y_es, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MultiScaleCausal(arr.shape[1]).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    steps = max(1, int(np.ceil(len(idx_tr) / BATCH)) * EPOCHS)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=LR * 0.05)
    lossf = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(seed)
    best_metric, best_state, best_auc = -1.0, None, float("nan")
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
            sched.step()
            tot += float(loss) * len(sel)
        p_es = predict(model, arr, idx_es)
        tw, nhit = tail_win(p_es, y_es, PRIMARY_TPD)      # 修复点 2：尾部胜率早停
        auc = float(roc_auc_score(y_es, p_es))
        log(f"    ep{ep + 1}/{EPOCHS} loss={tot / max(1, len(idx_tr)):.4f} "
            f"es_tailwin={tw * 100:.2f}%(n={nhit}) es_auc={auc:.4f}")
        if tw > best_metric:
            best_metric, best_auc = tw, auc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_metric, best_auc


def roll_worst(w: np.ndarray, n: int = 100):
    if len(w) < n:
        return float("nan"), float("nan")
    cs = np.concatenate([[0], np.cumsum(w)])
    r = (cs[n:] - cs[:-n]) / n
    return float(r.min()), float((r < .5).mean())


def _run(arr, y, ct):
    months = [m for m in sorted(set(
        pl.Series(ct).cast(pl.Datetime("ms")).dt.strftime("%Y-%m").to_list()))
        if START_MONTH <= m <= END_MONTH]
    log(f"walk-forward {len(months)} 窗（{months[0]}~{months[-1]}）| H={HORIZON} | "
        f"窗 {WIN_DAYS} 天 | es={ES_FRAC:.0%} thr={THR_FRAC:.0%} | seeds={len(SEEDS)} "
        f"| stride={STRIDE} maxtrain={MAX_TRAIN} | 早停=es尾部胜率 | placebo={int(PLACEBO)}")

    per = {t: dict(s=[], y=[], mon=[], wd=[], ct=[]) for t in TPD_LIST}
    rows = []
    for m in months:
        m_start = int(dt.datetime.strptime(m + "-01", "%Y-%m-%d")
                      .replace(tzinfo=dt.timezone.utc).timestamp() * 1000)
        lo = m_start - WIN_DAYS * 86_400_000
        i_tr = np.where((ct < m_start) & (ct >= lo))[0]
        nxt = (dt.datetime.strptime(m + "-01", "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)
               + dt.timedelta(days=32)).replace(day=1)
        i_te = np.where((ct >= m_start) & (ct < int(nxt.timestamp() * 1000)))[0]
        if len(i_tr) < 100_000 or len(i_te) < 1_000:
            log(f"[{m}] 样本不足（{len(i_tr):,}/{len(i_te):,}），跳过")
            continue
        h_all = len(i_tr)
        c1 = int(h_all * (1 - ES_FRAC - THR_FRAC))
        c2 = int(h_all * (1 - THR_FRAC))
        seg = [i_tr[:max(0, c1 - HORIZON)], i_tr[c1:c2 - HORIZON], i_tr[c2:]]
        if STRIDE > 1:
            seg[0] = seg[0][::STRIDE]
            seg[1] = seg[1][::STRIDE]
        i_tr_s, i_es, i_th = seg
        i_tr_s = i_tr_s[i_tr_s >= WIN - 1]
        i_es = i_es[i_es >= WIN - 1]
        i_th = i_th[i_th >= WIN - 1]
        i_te = i_te[i_te >= WIN - 1]
        if len(i_tr_s) > MAX_TRAIN:
            i_tr_s = i_tr_s[::max(1, len(i_tr_s) // MAX_TRAIN)][:MAX_TRAIN]
        y_tr = y[i_tr_s].astype(np.int8)
        if PLACEBO:
            y_tr = y_tr.copy()
            np.random.default_rng(20261115).shuffle(y_tr)
        y_es, y_th, y_te = y[i_es], y[i_th], y[i_te]

        p_th = np.zeros(len(i_th), dtype=np.float32)
        p_te = np.zeros(len(i_te), dtype=np.float32)
        tws = []
        for seed in SEEDS:
            mdl, tw, _ = train_one(arr, i_tr_s, y_tr, i_es, y_es, seed)
            p_th += predict(mdl, arr, i_th) / len(SEEDS)
            p_te += predict(mdl, arr, i_te) / len(SEEDS)
            tws.append(tw)
            if DEV == "cuda":
                torch.cuda.empty_cache()

        auc = float(roc_auc_score(y_te, p_te))
        days = (ct[i_te] // 86_400_000).astype(np.int64)
        nd = len(np.unique(days))
        s = p_te - .5
        wd = pl.from_epoch(pl.Series(ct[i_te]), time_unit="ms").dt.weekday().to_numpy()
        line = (f"[{m}] train={len(i_tr_s):,} es={len(i_es):,} thr={len(i_th):,} "
                f"test={len(i_te):,}({nd}天) es_tail={np.mean(tws)*100:.2f}% "
                f"testAUC={auc:.4f}")
        for tpd in TPD_LIST:
            thr = float(np.quantile(np.abs(p_th - .5), 1 - tpd / BARS_PER_DAY))
            hit = np.abs(s) >= thr
            if hit.sum() < 30:
                line += f" | tpd{tpd:g}:单量太少"
                continue
            hit_wd = wd[hit] <= 5
            per[tpd]["s"].append(s[hit])
            per[tpd]["y"].append(y_te[hit])
            per[tpd]["mon"].append(float(((s[hit] > 0).astype(int) == y_te[hit]).mean()))
            per[tpd]["wd"].append(hit_wd)
            per[tpd]["ct"].append(ct[i_te][hit])
            if tpd == PRIMARY_TPD:
                hw = float(((s[hit][hit_wd] > 0).astype(int) == y_te[hit][hit_wd]).mean()) \
                    if hit_wd.sum() >= 20 else float("nan")
                line += (f" | 阈值={thr:.4f} {hit.sum()/nd:.1f}单/天 胜率="
                         f"{((s[hit]>0).astype(int)==y_te[hit]).mean()*100:.2f}% 工作日={hw*100:.2f}%")
        rows.append(dict(ym=m, nd=nd, auc=auc))
        log(line)

    if not rows:
        raise SystemExit("无窗口完成")

    ev_hdr = "".join(f"{'EV@' + format(p, 'g'):>10}" for p in PAYOUTS)
    print(f"\n{'名义tpd':>8}{'实际单/天':>10}{'n':>7}{'胜率':>9}{'SE*':>7}{ev_hdr}"
          f"{'月min':>8}{'月std':>7}{'破50%月':>8}{'滚动100min':>11}{'工作日':>9}{'周末':>9}")
    nd_all = sum(r["nd"] for r in rows)
    main_win = float("nan")
    for tpd in TPD_LIST:
        p = per[tpd]
        if not p["y"]:
            continue
        yv = np.concatenate(p["y"])
        sv = np.concatenate(p["s"])
        w = ((sv > 0).astype(int) == yv).astype(np.int8)
        win = float(w.mean())
        if tpd == PRIMARY_TPD:
            main_win = win
        mon = np.array(p["mon"])
        se = float(np.sqrt(win * (1 - win) / max(len(w) / HORIZON, 2)))
        rmin, _ = roll_worst(w)
        evs = "".join(f"{pay * win - (1 - win):>+10.3f}" for pay in PAYOUTS)
        wd = np.concatenate(p["wd"])
        ww = float(w[wd].mean()) if wd.sum() >= 100 else float("nan")
        ew = float(w[~wd].mean()) if (~wd).sum() >= 100 else float("nan")
        star = " ← 主操作点" if tpd == PRIMARY_TPD else ""
        print(f"{tpd:>8.1f}{len(w)/nd_all:>10.1f}{len(w):>7d}{win*100:>8.2f}%{se*100:>6.2f}%"
              f"{evs}{mon.min()*100:>7.2f}%{mon.std()*100:>6.2f}pp"
              f"{int((mon<.5).sum()):>5}/{len(mon)}{rmin*100:>10.1f}%"
              f"{ww*100:>8.2f}%{ew*100:>8.2f}%{star}")
    verdict = ("可替代LGBM,进多段复核" if main_win >= 0.6332 else
               "有信号但不及LGBM,仅小权重融合" if main_win >= 0.58 else "判死")
    log(f"[预登记判据] 主操作点 tpd={PRIMARY_TPD:g} 胜率={main_win*100:.2f}% -> {verdict}")
    log("DONE")


def main() -> None:
    arr, y, ct = load_arrays()
    _run(arr, y, ct)


if __name__ == "__main__":
    main()
