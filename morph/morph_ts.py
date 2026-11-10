"""15min 形态时序模型（1D 膨胀卷积 CNN，CPU-only）。

动机（见 融合实验记录.md）：
  15min horizon 下单点订单簿快照的信息显著衰减，orderbook 腿近乎失效；
  faiss 形态聚类在 1% 覆盖率能到 ~60%，但它本质是"查簇 edge 表"——
  簇内差异全被抹平、信号平滑。本模块用可训练的 1D CNN 直接吃 1m K 线的
  多通道形态序列，输出 15min 方向概率，作为 faiss 腿的替代/升级。

输入通道（逐窗归一化，只用窗口内的过去数据，无未来信息）：
  px  : close 价格路径            窗内 min-max
  st  : stoch(29) / 100           保留绝对超买超卖含义，不再归一化
  vol : log1p(volume)             窗内 min-max（量能形态）
  cvd : cumsum(2*taker_buy - vol) 窗内 min-max（净主动买压力路径），缺列则跳过

切分（按时间，训练只用过去）：
  test  = 数据末尾 --test-days 天（默认 92，对齐 fusion 最近 ~3 个月的 test 段）
  val   = test 之前 --val-days 天（早停/选模型，不碰 test）
  train = 其余（2020 → val_start），按 --train-stride 抽样控制 CPU 时间

评估：AUC + 覆盖率-精度（top 1/2/5/10/25%），多 seed 概率平均。
输出：--out 指定 parquet（split/close_time/target/pred），供组合器与 orderbook 腿 join。

用法：
    python morph/morph_ts.py --horizon 15 --epochs 6 --seeds 42,43
    python morph/morph_ts.py --synthetic --rows 300000 --epochs 2   # 无数据烟测
"""
import os
import sys
import glob
import time
import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
# 直接 `python morph/morph_ts.py` 时 sys.path[0] 是 morph/，会把 morph.py 当成包，先剔除
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != str(Path(__file__).resolve().parent)]
sys.path.insert(0, str(ROOT))

import polars as pl
from sklearn.metrics import roc_auc_score
from morph.morph import stoch as _stoch

COVS = [0.01, 0.02, 0.05, 0.10, 0.25]
DAY_MS = 86_400_000


# ============================ 数据 ============================
def find_kline(path=None):
    if path:
        print(f"[kline] {path}")
        return pl.read_parquet(path).sort("close_time")
    # full 版含 volume / taker_buy_volume，build_channels 会自动加 vol+cvd 两通道；
    # 必须优先选它，否则会误用 OHLC-only 那份（v1 就是这么只有 px/st 两通道的）。
    names = ["btcusdt_1m_full.parquet", "btcusdt_1m.parquet",
             "btc_1m.parquet", "BTCUSDT_1m.parquet"]
    hits = []
    on_kaggle = bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE")) or os.path.isdir("/kaggle/input")
    for n in names:
        if on_kaggle:
            hits += glob.glob(f"/kaggle/input/**/{n}", recursive=True)
            hits += glob.glob(f"/kaggle/input/**/{n.lower()}", recursive=True)
        for p in (ROOT / "data" / n, ROOT / "data" / "klines" / n):
            if p.exists():
                hits.append(str(p))
    hits = sorted(set(hits))
    if not hits:
        raise FileNotFoundError("找不到 1m K 线 parquet，用 --kline-file 指定，或 --synthetic 烟测")
    full = [p for p in hits if p.endswith("btcusdt_1m_full.parquet")]
    pick = full[0] if full else hits[0]
    print(f"[kline] {pick}  candidates={len(hits)}")
    return pl.read_parquet(pick).sort("close_time")


def _boxcar(x, k):
    """长度不变的滑动均值（cumsum 实现，O(N)）。"""
    c = np.cumsum(np.insert(x.astype(np.float64), 0, 0.0))
    out = (c[k:] - c[:-k]) / k
    return np.concatenate([np.full(k - 1, out[0]), out])


def synthetic_klines(rows=300_000, seed=0):
    """随机游走 + 平滑漂移：窗口内的价格斜率对未来 15min 方向有真实信息，
    用于在无数据环境下验证"训练/评估/多 seed"链路是否跑通、AUC 能否 > 0.5。"""
    rng = np.random.default_rng(seed)
    drift = _boxcar(rng.standard_normal(rows), 60) / np.sqrt(60.0)
    ret = 0.0012 * drift + rng.standard_normal(rows) * 0.00025
    close = (30_000.0 * np.exp(np.cumsum(ret))).astype(np.float64)
    open_ = np.concatenate([[close[0]], close[:-1]])
    wig = np.abs(rng.standard_normal(rows)) * 0.0002
    high = np.maximum(open_, close) * (1.0 + wig)
    low = np.minimum(open_, close) * (1.0 - wig)
    volume = rng.lognormal(0.0, 0.6, rows)
    taker = volume * (0.5 + 0.05 * np.tanh(drift))
    start = 1_577_836_800_000  # 2020-01-01
    ct = start + np.arange(rows, dtype=np.int64) * 60_000
    print(f"[synthetic] rows={rows} seed={seed}")
    return pl.DataFrame({
        "close_time": ct, "open": open_, "high": high, "low": low, "close": close,
        "volume": volume, "taker_buy_volume": taker,
    })


def add_target(df, horizon):
    return df.with_columns(
        (pl.col("close").shift(-horizon) > pl.col("close")).cast(pl.Int8).alias("target"))

def build_channels(df):
    """返回 (chans[C,N] float32, names, norm_mask[C] bool)。"""
    close = df["close"].to_numpy().astype(np.float32)
    st = (_stoch(df, 29) / 100.0).astype(np.float32)
    chans, names, norm = [close, st], ["px", "st"], [True, False]
    if "volume" in df.columns:
        vol = np.log1p(df["volume"].fill_null(0).to_numpy().astype(np.float64))
        chans.append(vol.astype(np.float32))
        names.append("vol")
        norm.append(True)
        if "taker_buy_volume" in df.columns:
            v = df["volume"].fill_null(0).to_numpy().astype(np.float64)
            tb = df["taker_buy_volume"].fill_null(0).to_numpy().astype(np.float64)
            # 固定常数缩放（不用全局 mean/std）：审计里"随机化 t 之后的数据，
            # t 处输入须逐位不变"，任何全局统计都会破坏这条，也会被质疑成未来信息。
            cvd = np.cumsum(2.0 * tb - v) / 1e6
            chans.append(cvd.astype(np.float32))
            names.append("cvd")
            norm.append(True)
    chans = np.vstack([c.reshape(1, -1) for c in chans]).astype(np.float32)
    print(f"[channels] {names}  shape={chans.shape}  norm={norm}")
    return chans, names, np.array(norm, dtype=bool)


def window_input(chans, mask, i, window):
    """样本 i 的模型输入（numpy 版，与模型 forward 的 torch 归一化等价）。
    只读取 [i-W+1, i]，用于审计"未来数据扰动后 t 处输入不变"。"""
    x = chans[:, i - window + 1: i + 1].copy()
    if mask.any():
        sub = x[mask]
        lo = sub.min(1, keepdims=True)
        hi = sub.max(1, keepdims=True)
        x[mask] = (sub - lo) / np.maximum(hi - lo, 1e-9)
    return x


def split_by_time(ct, test_days, val_days):
    end = int(ct[-1])
    test_start = end - test_days * DAY_MS
    val_start = test_start - val_days * DAY_MS
    tr = np.where(ct < val_start)[0]
    va = np.where((ct >= val_start) & (ct < test_start))[0]
    te = np.where(ct >= test_start)[0]
    def _d(ms):
        return time.strftime("%Y-%m-%d", time.gmtime(ms / 1000))
    print(f"[split] train[0:{len(tr)}]<{_d(val_start)}  "
          f"val[{len(va)}] {_d(val_start)}~{_d(test_start)}  "
          f"test[{len(te)}] {_d(test_start)}~{_d(end)}")
    return tr, va, te


def split_with_purge(ct, horizon, window, test_days, val_days):
    """按时间切分 + purge：样本标签窗口 [i, i+horizon] 不得跨入下一段。
    否则训练集最后 horizon 个样本的标签落在 val 里（val 同理碰 test），
    属于典型的时间序列边界泄漏。"""
    tr, va, te = split_by_time(ct, test_days, val_days)
    w = window - 1
    tr, va, te = tr[tr >= w], va[va >= w], te[te >= w]
    n_tr, n_va = len(tr), len(va)
    if len(te):
        test0 = int(te[0])
        if len(va):
            tr = tr[tr + horizon < int(va[0])]
            va = va[va + horizon < test0]
        else:
            tr = tr[tr + horizon < test0]
    print(f"[purge] 丢弃跨边界样本: train -{n_tr - len(tr)}  val -{n_va - len(va)}"
          f"（horizon={horizon}，标签窗口不得跨段）")
    return tr, va, te


# ============================ 数据集 / 模型 ============================
def _make_dataset(chans, idx, y, window):
    import torch
    from torch.utils.data import Dataset

    class WindowDS(Dataset):
        def __init__(self):
            self.idx = idx
            self.W = window

        def __len__(self):
            return len(self.idx)

        def __getitem__(self, k):
            i = int(self.idx[k])
            x = chans[:, i - self.W + 1: i + 1].copy()
            return torch.from_numpy(np.ascontiguousarray(x)), torch.tensor(float(y[i]))

    return WindowDS()


def build_model(n_ch, norm_mask, c1=16, c2=24, c3=24, drop=0.2):
    """逐窗 min-max 放进 forward（torch 向量化，比 Dataset 里 numpy 快）。"""
    import torch
    import torch.nn as nn

    def blk(i, o, k, d):
        return nn.Sequential(
            nn.Conv1d(i, o, k, padding=d * (k - 1) // 2, dilation=d),
            nn.BatchNorm1d(o), nn.ReLU(inplace=True))

    class MorphNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = nn.Sequential(
                blk(n_ch, c1, 5, 1), blk(c1, c2, 5, 2), blk(c2, c3, 5, 4), blk(c3, c3, 3, 1),
                nn.AdaptiveAvgPool1d(8), nn.Flatten(), nn.Dropout(drop),
                nn.Linear(c3 * 8, 32), nn.ReLU(inplace=True), nn.Linear(32, 1))
            self.register_buffer("mask", torch.tensor(norm_mask, dtype=torch.bool).view(1, -1, 1))

        def forward(self, x):
            lo = x.amin(2, keepdim=True)
            hi = x.amax(2, keepdim=True)
            xn = (x - lo) / (hi - lo).clamp_min(1e-9)
            return self.body(torch.where(self.mask, xn, x)).squeeze(1)

    return MorphNet()


def _predict(model, dl):
    import torch
    model.eval()
    outs = []
    with torch.no_grad():
        for xb, _ in dl:
            outs.append(torch.sigmoid(model(xb)).numpy())
    return np.concatenate(outs) if outs else np.zeros(0, dtype=np.float32)


def train_one(chans, norm, y, tr, va, te, a, seed):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))

    ds_tr = _make_dataset(chans, tr, y, a.window)
    ds_va = _make_dataset(chans, va, y, a.window)
    ds_te = _make_dataset(chans, te, y, a.window)
    dl_tr = DataLoader(ds_tr, batch_size=a.batch, shuffle=True,
                       num_workers=a.workers, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=a.batch * 4, shuffle=False, num_workers=a.workers)
    dl_te = DataLoader(ds_te, batch_size=a.batch * 4, shuffle=False, num_workers=a.workers)

    model = build_model(chans.shape[0], norm, drop=a.drop)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, a.epochs))
    lossf = nn.BCEWithLogitsLoss()

    best_auc, best_state, bad = -1.0, None, 0
    for ep in range(a.epochs):
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        for xb, yb in dl_tr:
            opt.zero_grad(set_to_none=True)
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.detach().item()
            nb += 1
        sched.step()
        p_va = _predict(model, dl_va)
        auc = float(roc_auc_score(y[va], p_va))
        acc1, _ = acc_at_cov(p_va - 0.5, y[va], 0.01)
        print(f"  [seed {seed}] ep{ep + 1}/{a.epochs} loss={tot / max(nb, 1):.4f} "
              f"val_auc={auc:.4f} val@1%={acc1:.4f} ({time.time() - t0:.0f}s)")
        if auc > best_auc + 1e-5:
            best_auc, bad = auc, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= a.patience:
                print(f"  [seed {seed}] 早停（val_auc 连续 {bad} 轮无提升）")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    p_va = _predict(model, dl_va)
    p_te = _predict(model, dl_te)
    return p_va, p_te, best_auc


# ============================ 评估 ============================
def acc_at_cov(score, y, cov):
    n = len(y)
    k = max(1, int(n * cov))
    idx = np.argsort(-np.abs(score))[:k]
    pred = (score[idx] > 0).astype(int)
    return float((pred == y[idx]).mean()), k


def report(y_te, p_te, seeds, out_path):
    print("\n" + "=" * 78)
    print(f"[test] n={len(y_te)} 基准={y_te.mean():.4f} seeds={len(seeds)}")
    print(f"[AUC] {roc_auc_score(y_te, p_te):.4f}")
    print("=" * 78)
    print(f"{'cov':>6}{'n':>9}{'acc':>10}{'vs_base':>10}")
    base = float(y_te.mean())
    for cov in COVS:
        acc, n = acc_at_cov(p_te - 0.5, y_te, cov)
        print(f"{cov:>6.0%}{n:>9}{acc:>10.4f}{acc - base:>+10.4f}")
    if out_path:
        print(f"[save] {out_path}")


def main():
    ap = argparse.ArgumentParser(description="15min 形态时序模型（1D CNN，CPU）")
    ap.add_argument("--horizon", type=int, default=int(os.environ.get("KAGGLE_HORIZON", "15")))
    ap.add_argument("--window", type=int, default=120, help="输入 1m bar 数（默认 2 小时）")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--drop", type=float, default=0.2, help="head dropout，v1=0.2（欠拟合可降到 0.05）")
    ap.add_argument("--seeds", default="42,43")
    ap.add_argument("--train-stride", type=int, default=8, help="训练集抽样步长（CPU 时间控制）")
    ap.add_argument("--max-ct", type=int, default=0,
                    help="截断到该 close_time(ms)，用于与 v1 固定同一数据末尾、严格可比")
    ap.add_argument("--test-days", type=int, default=92)
    ap.add_argument("--val-days", type=int, default=30)
    ap.add_argument("--workers", type=int, default=0 if os.name == "nt" else 2)
    ap.add_argument("--kline-file", default=None)
    ap.add_argument("--synthetic", action="store_true", help="合成数据烟测（无需真实数据）")
    ap.add_argument("--rows", type=int, default=0, help="只取前 N 行（烟测）")
    ap.add_argument("--out", default=os.environ.get("KAGGLE_MORPH_TS_OUT"),
                    help="预测 parquet（split/close_time/target/pred）")
    # Kaggle 用 env 传超参（launcher 的 env 只能设环境变量，不能改 CLI）
    extra = os.environ.get("MORPH_TS_ARGS", "").split()
    a = ap.parse_args(list(sys.argv[1:]) + extra)

    try:
        import torch  # noqa: F401
    except ImportError:
        raise SystemExit("[!] 需要 PyTorch（Kaggle 镜像预装；本地 pip install torch）")

    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    df = synthetic_klines(a.rows or 300_000) if a.synthetic else find_kline(a.kline_file)
    if a.rows:
        df = df.head(a.rows)
    if a.max_ct:
        df = df.filter(pl.col("close_time") <= a.max_ct)
        print(f"[kline] 已截断到 close_time <= {a.max_ct}（与 v1 同数据末尾）")
    df = add_target(df, a.horizon).filter(pl.col("target").is_not_null())
    print(f"[kline] rows={df.height} horizon={a.horizon}min window={a.window}")

    chans, names, norm = build_channels(df)
    y = df["target"].to_numpy().astype(np.int8)
    ct = df["close_time"].to_numpy()

    tr, va, te = split_with_purge(ct, a.horizon, a.window, a.test_days, a.val_days)
    tr = tr[::a.train_stride]
    print(f"[samples] train={len(tr)}（stride={a.train_stride}） val={len(va)} test={len(te)}")

    p_va_sum = np.zeros(len(va), dtype=np.float64)
    p_te_sum = np.zeros(len(te), dtype=np.float64)
    aucs = []
    for sd in seeds:
        pv, pt, auc = train_one(chans, norm, y, tr, va, te, a, sd)
        p_va_sum += pv
        p_te_sum += pt
        aucs.append(auc)
    p_va = (p_va_sum / len(seeds)).astype(np.float32)
    p_te = (p_te_sum / len(seeds)).astype(np.float32)
    print(f"[seeds] 单 seed val_auc={[round(x, 4) for x in aucs]}  "
          f"集成 val_auc={roc_auc_score(y[va], p_va):.4f}")

    y_te = y[te]
    report(y_te, p_te, seeds, a.out)
    acc1, _ = acc_at_cov(p_te - 0.5, y_te, 0.01)
    acc5, _ = acc_at_cov(p_te - 0.5, y_te, 0.05)
    acc25, _ = acc_at_cov(p_te - 0.5, y_te, 0.25)
    print(f"\n[结论] morph_ts@{a.horizon}min: AUC={roc_auc_score(y_te, p_te):.4f} "
          f"1%={acc1:.4f} 5%={acc5:.4f} 25%={acc25:.4f} "
          f"（faiss 基线: 1%~0.60）")

    if a.out:
        # close_time +1ms：Binance K 线 close=open+59999，orderbook close=open+60000，
        # +1 后可直接与 orderbook 行 join（与 orderbook/fusion_test.py 的对齐口径一致）
        out = pl.concat([
            pl.DataFrame({"split": ["val"] * len(va), "close_time": ct[va] + 1,
                          "target": y[va].astype(np.int8), "pred": p_va}),
            pl.DataFrame({"split": ["test"] * len(te), "close_time": ct[te] + 1,
                          "target": y_te.astype(np.int8), "pred": p_te}),
        ])
        out.write_parquet(a.out)
        print(f"[save] {a.out} rows={out.height}")


if __name__ == "__main__":
    main()
