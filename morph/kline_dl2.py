"""第六轮：给 CNN 喂真正的 K 线形态通道，并按权重融合（不再等权）。

第五轮的发现：
  - CNN 与 LGB 分数相关系数仅 0.198 —— 信息几乎正交，融合有空间；
  - 但 CNN 单独只有 0.5458，等权平均把 LGB 从 0.6125 拉到 0.5625，
    说明必须小权重；
  - 第五轮 CNN 只吃 px/st/vol/cvd，没吃到"K 线形态"本身（实体、上下影线、
    单根收益）。faiss 形态聚类能到 60% 靠的就是这些。

本轮：
  1. CNN 输入扩到 8 通道：px / body / upper / lower / ret1 / st / vol / cvd，
     其中 body/upper/lower 就是每根 K 线的形状，逐窗 min-max 保留相对形态；
  2. 融合不再等权，扫 w ∈ [0, 0.5]，同时报 val 与 test 的整条曲线，
     看提升是否稳定（只看 test 最大值等于作弊）。
"""
import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from morph.kline_boost import (DAILY_K, OB_END, TEST_START, VAL_START,  # noqa: E402
                               build_features, complete_days, daily_topk_eval,
                               day_index, global_topk_eval, load_klines, log,
                               make_target, zscore)
from morph.kline_dl import evaluate, lgb_scores  # noqa: E402
from morph.morph_ts import _make_dataset, _predict, build_model  # noqa: E402


def build_chans2(df):
    """8 通道：价格路径 + K 线形状 + 量能/主动买 + 随机指标。"""
    c = df["close"].to_numpy().astype(np.float64)
    o = df["open"].to_numpy().astype(np.float64)
    h = df["high"].to_numpy().astype(np.float64)
    l = df["low"].to_numpy().astype(np.float64)
    rng = np.maximum(h - l, 1e-9)
    bps = 1e4
    chans = [c, (c - o) / c * bps, (h - np.maximum(c, o)) / c * bps,
             (np.minimum(c, o) - l) / c * bps]
    names = ["px", "body", "upper", "lower"]
    r1 = np.zeros_like(c)
    r1[1:] = (c[1:] / c[:-1] - 1.0) * bps
    chans.append(r1)
    names.append("ret1")
    if "volume" in df.columns:
        v = df["volume"].fill_null(0).to_numpy().astype(np.float64)
        chans.append(np.log1p(v))
        names.append("vol")
        if "taker_buy_volume" in df.columns:
            tb = df["taker_buy_volume"].fill_null(0).to_numpy().astype(np.float64)
            chans.append(np.cumsum(2.0 * tb - v) / 1e6)
            names.append("cvd")
    lo = pl.Series(l).rolling_min(29).to_numpy()
    hi = pl.Series(h).rolling_max(29).to_numpy()
    st = (c - lo) / np.maximum(hi - lo, 1e-9)
    chans.append(np.nan_to_num(st, nan=0.5))
    names.append("st")
    norm = np.array([True] * (len(chans) - 1) + [False])
    x = np.vstack([np.nan_to_num(v, nan=0.0).astype(np.float32).reshape(1, -1)
                   for v in chans])
    log(f"  [chans2] {names} shape={x.shape} 窗内归一化={int(norm.sum())}/{len(norm)}")
    return x, names, norm


def cnn_train(chans, norm, y, ct, window, seeds, epochs, stride, batch=1024, lr=2e-3):
    tr = np.where(ct < VAL_START)[0]
    va = np.where((ct >= VAL_START) & (ct < TEST_START))[0]
    te = np.where(ct >= TEST_START)[0]
    tr = tr[tr >= window - 1][::stride]
    va = va[va + 3 < int(te[0])]
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
    p_va, p_te = np.zeros(len(va)), np.zeros(len(te))
    for sd in seeds:
        torch.manual_seed(sd)
        np.random.seed(sd)
        dl_tr = DataLoader(_make_dataset(chans, tr, y, window), batch_size=batch,
                           shuffle=True, num_workers=2, drop_last=True)
        dl_va = DataLoader(_make_dataset(chans, va, y, window), batch_size=batch * 4,
                           shuffle=False, num_workers=2)
        dl_te = DataLoader(_make_dataset(chans, te, y, window), batch_size=batch * 4,
                           shuffle=False, num_workers=2)
        model = build_model(len(chans), norm)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=3e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
        lossf = nn.BCEWithLogitsLoss()
        best, best_state = -1.0, None
        for ep in range(epochs):
            model.train()
            tot, nb = 0.0, 0
            for xb, yb in dl_tr:
                opt.zero_grad(set_to_none=True)
                loss = lossf(model(xb), yb)
                loss.backward()
                opt.step()
                tot += loss.detach().item()
                nb += 1
            sched.step()
            pv = _predict(model, dl_va)
            auc = float(roc_auc_score(y[va], pv))
            log(f"  [cnn2 seed {sd}] ep{ep + 1}/{epochs} loss={tot / max(nb, 1):.4f} "
                f"val_auc={auc:.4f}")
            if auc > best:
                best, best_state = auc, {k: v.detach().clone()
                                         for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        p_va += _predict(model, dl_va)
        p_te += _predict(model, dl_te)
    full = np.full(len(y), np.nan)
    full[va] = p_va / len(seeds)
    full[te] = p_te / len(seeds)
    return full, va, te


def sweep(z1, zc, ct, y):
    """扫融合权重，同时看 val 与 test。"""
    va = np.where((ct >= VAL_START) & (ct < TEST_START))[0]
    te = np.where(ct >= TEST_START)[0]
    va = va[va + 3 < int(te[0])]
    dva, dte = day_index(ct[va]), day_index(ct[te])
    out = []
    for w in np.arange(0.0, 0.55, 0.05):
        z = z1 + w * zc
        za = zscore(z[va], z[va])
        zt = zscore(z[te], z[va])
        out.append((round(float(w), 2),
                    global_topk_eval(za, y[va], dva, DAILY_K),
                    global_topk_eval(zt, y[te], dte, DAILY_K),
                    float(daily_topk_eval(zt, y[te], dte,
                                          complete_days(dte, 1200), DAILY_K).mean()),
                    float(roc_auc_score(y[te], zt))))
    return out


def main():
    seeds = [int(x) for x in os.environ.get("SEEDS", "42,43,44").split(",")]
    cnn_seeds = [int(x) for x in os.environ.get("CNN_SEEDS", "42,43").split(",")]
    epochs = int(os.environ.get("CNN_EPOCHS", "5"))
    stride = int(os.environ.get("CNN_STRIDE", "8"))
    rounds = int(os.environ.get("ROUNDS", "60"))

    raw = load_klines().filter(pl.col("close_time") <= OB_END)
    log(f"[1m] rows={raw.height}")
    d1 = make_target(raw, 3)
    ft1, c1 = build_features(d1)
    ct = ft1["close_time"].to_numpy()
    y = ft1["target"].to_numpy().astype(np.int8)

    rows = {}
    log(f"[lgb1m] {len(seeds)} seed")
    p1, _, _ = lgb_scores(ft1, c1, 3, seeds, rounds)
    z1 = evaluate("lgb1m", p1, ct, y, rows)
    del ft1

    log("[cnn2 1m] 8 通道形态")
    chans, names, norm = build_chans2(d1)
    pc, va, te = cnn_train(chans, norm, y, ct, 60, cnn_seeds, epochs, stride)
    m = ~np.isnan(pc)
    pcf = np.where(m, pc, np.nanmedian(pc[m]))
    evaluate("cnn2", pcf, ct, y, rows)
    zc = zscore(pcf, pcf[va])

    log("[sweep] w: val_static / test_static / test_daily / test_auc")
    best_w, best_te = 0.0, 0.0
    for w, av, at, ad, aa in sweep(z1, zc, ct, y):
        log(f"  w={w:<4} val={av:.4f} test={at:.4f} 按天={ad:.4f} AUC={aa:.4f}")
        if at > best_te:
            best_w, best_te = w, at
    log(f"[sweep] test 最优 w={best_w} -> {best_te:.4f}（仅作参考，选 w 看 val 曲线一致性）")

    for w in (0.15, 0.25, 0.35):
        mix = z1 + w * zc
        evaluate(f"lgb1+{w}cnn", mix, ct, y, rows)

    log(f"\n{'变体':<16}{'static':>9}{'AUC':>8}{'按天':>8}{'窗口4':>8}{'4段sd':>8}")
    for k, r in rows.items():
        log(f"{k:<16}{r['static']:>9.4f}{r['auc']:>8.4f}{r['daily']:>8.4f}"
            f"{r['w4']:>8.4f}{r['w4sd']:>8.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
