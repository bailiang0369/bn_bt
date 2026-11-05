"""第五轮：3m 基础 K 线 + 1D CNN 形态模型 + 四路融合（全部在 1m 网格同口径评估）。

前四轮只用了"在 1m 上算 63 个统计特征 + LightGBM"这一种表达。本轮补两件事：

1. **3m 基础 K 线**：先把 1m 重采样成 3m bar，在 3m 上重新算全部 63 个特征
   （窗口单位随之变成 15min~6h），用下一根 3m 的方向当标签。之前 v1 直接把它
   放在 3m 网格上取 topK=15/天，那是 3.1% 覆盖率，而 1m 是 1.04% —— 口径不同，
   必然显低。本轮把 3m 分数映射回 1m 网格再评估，覆盖率一致。

2. **CNN 形态模型**：直接吃最近 N 根 bar 的多通道归一化序列（price/stoch/vol/cvd），
   用 1D 膨胀卷积学形态，而不是把形态压成统计量。1m（window=60）与 3m
   （window=40）各训一个。

四路分数在 val 上各自 z-score 后等权平均，不调权重（前几轮已证明 val 调参会过拟合）。
"""
import os
import sys
import time

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lightgbm as lgb  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from morph.kline_boost import (DAILY_K, OB_END, PARAMS, TEST_START, VAL_START,  # noqa: E402
                               align_to, build_features, complete_days,
                               daily_topk_eval, day_index, global_topk_eval, load_klines,
                               log, make_target, resample_3m, zscore)
from morph.morph_ts import _make_dataset, _predict, build_channels, build_model  # noqa: E402


def lgb_scores(ft, cols, h, seeds, rounds):
    ct = ft["close_time"].to_numpy()
    y = ft["target"].to_numpy().astype(np.int8)
    X = ft.select(cols).to_numpy().astype(np.float32)
    tr = np.where(ct < VAL_START)[0]
    va = np.where((ct >= VAL_START) & (ct < TEST_START))[0]
    tr = tr[tr + h < int(va[0])]
    acc = np.zeros(len(y))
    for sd in seeds:
        p = dict(PARAMS, seed=sd, bagging_seed=sd, feature_fraction_seed=sd,
                 data_random_seed=sd)
        m = lgb.train(p, lgb.Dataset(X[tr], y[tr], feature_name=cols),
                      num_boost_round=rounds)
        acc += m.predict(X)
    del X
    return (acc / len(seeds)).astype(np.float32), ct, y


def cnn_scores(df, ct, y, window, seeds, epochs, stride, batch=1024, lr=2e-3):
    chans, names, norm = build_channels(df)
    tr = np.where(ct < VAL_START)[0]
    va = np.where((ct >= VAL_START) & (ct < TEST_START))[0]
    te = np.where(ct >= TEST_START)[0]
    tr = tr[tr >= window - 1]
    va = va[va + 3 < int(te[0])]
    tr_s = tr[::stride]
    log(f"  [cnn] chans={names} train={len(tr_s)}(stride={stride}) "
        f"val={len(va)} test={len(te)}")

    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
    p_va = np.zeros(len(va))
    p_te = np.zeros(len(te))

    for sd in seeds:
        torch.manual_seed(sd)
        np.random.seed(sd)
        dl_tr = DataLoader(_make_dataset(chans, tr_s, y, window), batch_size=batch,
                           shuffle=True, num_workers=2, drop_last=True)
        dl_va = DataLoader(_make_dataset(chans, va, y, window), batch_size=batch * 4,
                           shuffle=False, num_workers=2)
        dl_te = DataLoader(_make_dataset(chans, te, y, window), batch_size=batch * 4,
                           shuffle=False, num_workers=2)
        model = build_model(len(chans), norm)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
        lossf = nn.BCEWithLogitsLoss()
        best_auc, best_state = -1.0, None
        for ep in range(epochs):
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
            pv = _predict(model, dl_va)
            auc = float(roc_auc_score(y[va], pv))
            log(f"  [cnn seed {sd}] ep{ep + 1}/{epochs} loss={tot / max(nb, 1):.4f} "
                f"val_auc={auc:.4f} ({time.time() - t0:.0f}s)")
            if auc > best_auc:
                best_auc = auc
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        p_va += _predict(model, dl_va)
        p_te += _predict(model, dl_te)
    p_va /= len(seeds)
    p_te /= len(seeds)
    full = np.full(len(y), np.nan)
    full[va] = p_va
    full[te] = p_te
    return full, va, te


def window_eval(z, y, ct, te, n_win=4):
    edges = np.linspace(int(ct[te[0]]), int(ct[te[-1]]) + 1, n_win + 1).astype(np.int64)
    out = []
    for i in range(n_win):
        m = (ct[te] >= edges[i]) & (ct[te] < edges[i + 1])
        idx = te[m]
        if len(idx):
            out.append(global_topk_eval(z[idx], y[idx], day_index(ct[idx]), DAILY_K))
    return np.array(out)


def evaluate(tag, z, ct, y, rows):
    te = np.where(ct >= TEST_START)[0]
    va = np.where((ct >= VAL_START) & (ct < TEST_START))[0]
    va = va[va + 3 < int(te[0])]
    zz = zscore(z, z[va])
    dd = day_index(ct[te])
    r = {"static": global_topk_eval(zz[te], y[te], dd, DAILY_K),
         "auc": float(roc_auc_score(y[te], zz[te])),
         "daily": float(daily_topk_eval(zz[te], y[te], dd,
                                        complete_days(dd, 1200), DAILY_K).mean())}
    w = window_eval(zz, y, ct, te)
    r["w4"], r["w4sd"] = float(w.mean()), float(w.std(ddof=1))
    rows[tag] = r
    log(f"[{tag}] static={r['static']:.4f} AUC={r['auc']:.4f} "
        f"按天={r['daily']:.4f} 窗口4={r['w4']:.4f}±{r['w4sd']:.4f}")
    return zz


def main():
    lgb_seeds1 = [int(x) for x in os.environ.get("SEEDS", "42,43,44,45,46").split(",")]
    lgb_seeds3 = lgb_seeds1[:3]
    cnn_seeds = [int(x) for x in os.environ.get("CNN_SEEDS", "42,43").split(",")]
    epochs = int(os.environ.get("CNN_EPOCHS", "4"))
    stride1 = int(os.environ.get("CNN_STRIDE", "8"))
    stride3 = int(os.environ.get("CNN_STRIDE3", "4"))
    rounds = int(os.environ.get("ROUNDS", "60"))

    raw = load_klines().filter(pl.col("close_time") <= OB_END)
    log(f"[1m] rows={raw.height}")
    raw3 = resample_3m(raw)
    log(f"[3m] rows={raw3.height}")

    d1 = make_target(raw, 3)
    ft1, c1 = build_features(d1)
    d3 = make_target(raw3, 1)
    ft3, c3 = build_features(d3)

    rows, store = {}, {}

    log("[lgb 1m] 63 特征 × %d seed" % len(lgb_seeds1))
    p1, ct, y = lgb_scores(ft1, c1, 3, lgb_seeds1, rounds)
    store["lgb1m"] = evaluate("lgb1m", p1, ct, y, rows)

    log("[lgb 3m] 63 特征 × %d seed（映射回 1m 网格）" % len(lgb_seeds3))
    p3, ct3, _ = lgb_scores(ft3, c3, 1, lgb_seeds3, rounds)
    p3_on1m = align_to(ct3, p3, ct)
    store["lgb3m"] = evaluate("lgb3m", p3_on1m, ct, y, rows)

    log("[cnn 1m] window=60")
    pc1, _, _ = cnn_scores(d1, ct, y, 60, cnn_seeds, epochs, stride1)
    m = ~np.isnan(pc1)
    pc1f = np.where(m, pc1, np.nanmedian(pc1[m]))
    store["cnn1m"] = evaluate("cnn1m", pc1f, ct, y, rows)

    log("[cnn 3m] window=40（映射回 1m 网格）")
    ct3b = d3["close_time"].to_numpy()
    y3 = d3["target"].to_numpy().astype(np.int8)
    pc3, _, _ = cnn_scores(d3, ct3b, y3, 40, cnn_seeds, epochs, stride3)
    m3 = ~np.isnan(pc3)
    pc3f = np.where(m3, pc3, np.nanmedian(pc3[m3]))
    store["cnn3m"] = evaluate("cnn3m", align_to(ct3b, pc3f, ct), ct, y, rows)

    def mix(*keys):
        return np.mean([store[k] for k in keys], axis=0)

    evaluate("lgb1+cnn1", mix("lgb1m", "cnn1m"), ct, y, rows)
    evaluate("lgb1+lgb3", mix("lgb1m", "lgb3m"), ct, y, rows)
    evaluate("lgb1+cnn3", mix("lgb1m", "cnn3m"), ct, y, rows)
    evaluate("cnn1+cnn3", mix("cnn1m", "cnn3m"), ct, y, rows)
    evaluate("all4", mix("lgb1m", "lgb3m", "cnn1m", "cnn3m"), ct, y, rows)
    evaluate("lgb1+lgb3+cnn1", mix("lgb1m", "lgb3m", "cnn1m"), ct, y, rows)

    ks = list(store)
    log("[corr]")
    for i, a in enumerate(ks):
        for b in ks[i + 1:]:
            c = float(np.corrcoef(store[a], store[b])[0, 1])
            log(f"  {a} vs {b} = {c:.4f}")

    log(f"\n{'变体':<16}{'static':>9}{'AUC':>8}{'按天':>8}{'窗口4':>8}{'4段sd':>8}")
    for k, r in rows.items():
        log(f"{k:<16}{r['static']:>9.4f}{r['auc']:>8.4f}{r['daily']:>8.4f}"
            f"{r['w4']:>8.4f}{r['w4sd']:>8.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
