"""时序神经网络（TCN）预测 3 分钟方向 —— Kaggle GPU。

动机：GBDT 逐样本独立，只靠手工滚动特征间接利用时序；TCN 用因果空洞卷积直接建模
过去 W 根 bar 的演化。目标是在**同一口径**下超过 GBDT 基线
（roll_1d topK 0.6167 / roll_3d 0.6229）。

严格因果：Conv1d 用左侧 padding（padding=dilation），取最后时刻输出，不看未来。
评估口径与 `orderbook/znorm_eval.py` 一致：滚动 z-score + global_topk_eval(K=15×天数)。
"""
import argparse
import glob
import os
import sys

import numpy as np
import polars as pl

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score

H = 3
DAILY_K = 15


def log(m):
    print(m, flush=True)


# --- 复制自 orderbook/fusion_test.py ---
def day_index(ct_ms):
    return (np.asarray(ct_ms) // 86_400_000).astype(np.int64)


def complete_days(day_id, min_bars=1200):
    u, c = np.unique(day_id, return_counts=True)
    return np.isin(day_id, u[c >= min_bars])


def global_topk_eval(score, y, day_id, k_per_day):
    n_days = len(np.unique(day_id))
    k = min(int(k_per_day) * n_days, len(y))
    idx = np.argsort(-np.abs(score))[:k]
    return float(((score[idx] > 0).astype(int) == y[idx]).mean())


def daily_topk_eval(score, y, day_id, mask, k):
    s, yy, d = score[mask], y[mask], day_id[mask]
    accs = []
    for u in np.unique(d):
        m = d == u
        kk = min(k, int(m.sum()))
        idx = np.argsort(-np.abs(s[m]))[:kk]
        accs.append(float(((s[m][idx] > 0).astype(int) == yy[m][idx]).mean()))
    return np.array(accs, dtype=np.float64)


def roll_z(x, win):
    s = pl.Series("x", x)
    mp = max(win // 4, 30)
    mu = s.rolling_mean(win, min_samples=mp).shift(1)
    sd = s.rolling_std(win, min_samples=mp).shift(1)
    return ((s - mu) / (sd + 1e-9)).fill_nan(0.0).fill_null(0.0).to_numpy()


def find(sym="BTCUSDT"):
    exact = f"{sym}.parquet"
    hits = [h for h in glob.glob(f"/kaggle/input/**/{exact}", recursive=True)
            if os.path.basename(h) == exact]
    if not hits:
        raise FileNotFoundError(exact)
    log(f"[data] {hits[0]}")
    return hits[0]


class SeqDS(Dataset):
    """第 i 个样本取 X[t-W+1 : t+1]，标签 y[t]，只要求 t-W+1 >= 0。"""

    def __init__(self, X, y, idx, W):
        self.X, self.y, self.idx, self.W = X, y, idx, W

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        t = self.idx[i]
        return (torch.from_numpy(self.X[t - self.W + 1:t + 1]),
                torch.tensor(float(self.y[t])))


class CausalConv(nn.Module):
    """严格因果卷积：只在左侧 pad (k-1)*d，out[i] 只看 in[:i]。"""

    def __init__(self, cin, cout, k=3, d=1):
        super().__init__()
        self.pad = (k - 1) * d
        self.conv = nn.Conv1d(cin, cout, k, dilation=d)

    def forward(self, x):
        return self.conv(F.pad(x, (self.pad, 0)))


class TCN(nn.Module):
    def __init__(self, nf, ch=64, drop=0.2):
        super().__init__()
        layers, cin = [], nf
        for d in (1, 2, 4, 8):
            layers += [CausalConv(cin, ch, 3, d), nn.GELU(), nn.Dropout(drop)]
            cin = ch
        self.body = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Linear(ch, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, x):           # x: (B, W, F)
        h = self.body(x.transpose(1, 2))   # (B, C, W)
        return self.head(h[:, :, -1]).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wins", default=os.environ.get("WINS", "1,16"),
                    help="逗号分隔的序列窗口；W=1 等价 MLP，用作序列信息是否有用的对照")
    ap.add_argument("--epochs", type=int, default=int(os.environ.get("EPOCHS", "8")))
    ap.add_argument("--bs", type=int, default=int(os.environ.get("BS", "512")))
    ap.add_argument("--lr", type=float, default=float(os.environ.get("LR", "1e-3")))
    ap.add_argument("--ch", type=int, default=int(os.environ.get("CH", "64")))
    ap.add_argument("--offs", default=os.environ.get("OFFS", "0"))
    ap.add_argument("--seeds", default=os.environ.get("SEEDS", "42,43"))
    a = ap.parse_args()
    offs = [int(x) for x in a.offs.split(",")]
    seeds = [int(x) for x in a.seeds.split(",")]
    wins = [int(x) for x in a.wins.split(",")]

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"[dev] {dev} {torch.cuda.get_device_name(0) if dev == 'cuda' else ''}")

    df = pl.read_parquet(find()).sort("close_time")
    df = df.with_columns(pl.col("close").shift(-H).alias("_fc"))
    df = (df.with_columns((pl.col("_fc") > pl.col("close")).cast(pl.Int8).alias("target"))
          .drop("_fc").filter(pl.col("target").is_not_null()))
    L = df.height
    ct = df["close_time"].to_numpy()
    y = df["target"].to_numpy().astype(np.int8)
    cols = [c for c in df.columns if c not in ("target", "close_time")]
    X = df.select(cols).to_numpy().astype(np.float32)
    del df
    log(f"[data] rows={L} feats={len(cols)}")
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    np.clip(X, -1e6, 1e6, out=X)

    k1_0, k2_0 = int(L * 0.64), int(L * 0.80)
    dyn = day_index(ct)

    for off in offs:
        k1, k2 = k1_0 - off * 1440, k2_0 - off * 1440
        val_start, test_start = int(ct[k1]), int(ct[k2])
        m_val = (ct >= val_start) & (ct < test_start)
        m_te = ct >= test_start
        dte = dyn[m_te]
        mt = complete_days(dte)
        n_te = len(np.unique(dte))

        Wmax = max(wins)
        tr_idx = np.where(np.arange(L) < k1 - H)[0]
        tr_idx = tr_idx[tr_idx >= Wmax - 1]
        va_idx = np.where(m_val)[0]
        va_idx = va_idx[va_idx >= Wmax - 1]
        te_idx = np.where(m_te)[0]
        log(f"\n===== off={off} train={len(tr_idx)} val={len(va_idx)} "
            f"test={len(te_idx)} 天数={n_te} 基准={y[m_te].mean():.4f} =====")

        # 特征标准化：只用训练段统计量（实盘合法），各 W 共用同一套
        mu = X[tr_idx].mean(axis=0, keepdims=True)
        sd = X[tr_idx].std(axis=0, keepdims=True) + 1e-6
        Xn = ((X - mu) / sd).astype(np.float32)
        log("[norm] 训练段逐特征 z-score 完成")

        for W in wins:
            log(f"\n----- W={W} {'(等价 MLP，无序列)' if W == 1 else ''} -----")
            probs = np.zeros(L, dtype=np.float64)
            for sd_i in seeds:
                torch.manual_seed(sd_i)
                np.random.seed(sd_i)
                model = TCN(len(cols), a.ch).to(dev)
                opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
                lossf = nn.BCEWithLogitsLoss()
                dl = DataLoader(SeqDS(Xn, y, tr_idx, W), batch_size=a.bs, shuffle=True,
                                num_workers=2, pin_memory=(dev == "cuda"), drop_last=True)
                for ep in range(a.epochs):
                    model.train()
                    tot, nb = 0.0, 0
                    for xb, yb in dl:
                        xb, yb = xb.to(dev, non_blocking=True), yb.to(dev, non_blocking=True)
                        opt.zero_grad(set_to_none=True)
                        loss = lossf(model(xb), yb)
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        opt.step()
                        tot += float(loss.detach())
                        nb += 1
                    sched.step()
                    if (ep + 1) % 5 == 0 or ep == 0:
                        log(f"  [ep {ep+1}/{a.epochs}] loss={tot/max(nb,1):.5f}")

                model.eval()
                out = np.zeros(L, dtype=np.float32)
                with torch.no_grad():
                    dl2 = DataLoader(SeqDS(Xn, y, np.concatenate([va_idx, te_idx]), W),
                                     batch_size=1024, shuffle=False, num_workers=2,
                                     pin_memory=(dev == "cuda"))
                    buf = []
                    for xb, _ in dl2:
                        buf.append(torch.sigmoid(model(xb.to(dev))).cpu().numpy())
                    pr = np.concatenate(buf)
                    out[va_idx] = pr[:len(va_idx)]
                    out[te_idx] = pr[len(va_idx):]
                probs += out
            p = (probs / len(seeds)).astype(np.float32)

            for win in (1440, 4320):
                z = roll_z(p, win)
                auc = float(roc_auc_score(y[m_te], z[m_te]))
                gk = global_topk_eval(z[m_te], y[m_te], dte, DAILY_K)
                accs = daily_topk_eval(z[m_te], y[m_te], dte, mt, DAILY_K)
                idx = np.argsort(-np.abs(z[m_te]))[:DAILY_K * n_te]
                u, c = np.unique(dte[idx], return_counts=True)
                log(f"[W={W} off={off} roll_{win//1440}d] AUC={auc:.4f} topK={gk:.4f} "
                    f"按天={accs.mean():.4f} 覆盖={len(u)}/{n_te} 单天max={int(c.max())}")
        del Xn
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
