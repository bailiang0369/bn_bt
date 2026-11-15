#!/usr/bin/env python3
"""记忆回放：近期窗 + 历史「预测正确」样本（**新文件，不改现役代码**）。

背景（用户 2026-11-16）
----------------------
用户此前在另一个项目里用过：训练集 = 近期固定 N 行 + 过往训练中**预测正确**的样本
（另设上限，如近期 10 万 + 历史 2 万），最终稳定阶段拿到约 +2pp 准确率。
本文件在本仓库的因果 walk-forward 口径下复现这个做法。

三条必须先立住的纪律
--------------------
1. **因果**：第 m 月的记忆池只能来自 m' < m 的月，且这些样本的「正确」判定必须由
   **当时那个只训练到 m' 之前的模型**给出。绝不能用最终模型回头判对错（那是泄漏）。
2. **记忆样本是按「易例」抽的**：正确且高置信 = 模型已经会的部分。所以要同时看
   月 min / 月 std / 破 50% 月，不能只看总体胜率。
3. **权重**：记忆样本天然是少数（50k 对 1.05M 只有 4.5%），直接混进去统计上无效果。
   用 MEM_W 放大其样本权重，让等效占比接近用户原项目的 15~20%。

因此本实验的正确对照是「**短近期窗**（如 730d）无记忆 vs 有记忆」，
而不是「现役长窗（1460d）再加记忆」——后者只会稀释。

env
---
  HORIZON=10 WIN_DAYS=730 MEM_N=50000 MEM_W=4.0 MEM_PER_MONTH=2000
  SEEDS=42,43,44,45,46 STRIDE=HORIZON ES_FRAC=0.05 THR_FRAC=0.10 ROUNDS=2000
  START_MONTH=2026-01 END_MONTH=2026-10 TPD_LIST=2,5,8,15,30 PRIMARY_TPD=8
  PAYOUTS=0.80,0.828 BARS_PER_DAY=1440

校验（必须先过）：MEM_N=0 且 HORIZON=10 / WIN_DAYS=1460 / STRIDE=1 时应复现
现役基线 tpd=8 → 63.32% @19.9 单/天（±噪声）。复现不出说明实现有偏差，整组作废。
"""
from __future__ import annotations

import datetime as dt
import os
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from morph import kline_long as kl  # noqa: E402
from morph.h10_signal import add_flow, roll_worst  # noqa: E402
from morph.two_legs import add_target, matrix  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

import lightgbm as lgb  # noqa: E402

HORIZON = int(os.environ.get("HORIZON", "10"))
WIN_DAYS = int(os.environ.get("WIN_DAYS", "730"))
MEM_N = int(os.environ.get("MEM_N", "50000"))       # 0 = 关闭（对照组）
MEM_W = float(os.environ.get("MEM_W", "4.0"))       # 记忆样本的样本权重
MEM_PER_MONTH = int(os.environ.get("MEM_PER_MONTH", "2000"))  # 每月最多入库多少条
SEEDS = [int(s) for s in os.environ.get("SEEDS", "42,43,44,45,46").split(",") if s.strip()]
ROUNDS = int(os.environ.get("ROUNDS", "2000"))
ES_FRAC = float(os.environ.get("ES_FRAC", "0.05"))
THR_FRAC = float(os.environ.get("THR_FRAC", "0.10"))
STRIDE = int(os.environ.get("STRIDE", str(max(1, HORIZON))))
START_MONTH = os.environ.get("START_MONTH", "2026-01")
END_MONTH = os.environ.get("END_MONTH", "2026-10")
TPD_LIST = [float(x) for x in os.environ.get("TPD_LIST", "2,5,8,15,30").split(",") if x.strip()]
PRIMARY_TPD = float(os.environ.get("PRIMARY_TPD", "8"))
BARS_PER_DAY = int(os.environ.get("BARS_PER_DAY", "1440"))
PAYOUTS = [float(x) for x in os.environ.get("PAYOUTS", "0.80,0.828").split(",") if x.strip()]
T = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T:7.1f}s] {msg}", flush=True)


def fit_w(Xtr, ytr, cols, Xes, yes, wtr, seed):
    """带样本权重的单模型训练；接口对齐 two_legs.fit 的语义（早停看 es AUC）。"""
    params = dict(objective="binary", metric="auc", boosting_type="gbdt",
                  learning_rate=0.05, num_leaves=63, feature_fraction=0.9,
                  bagging_fraction=0.9, bagging_freq=1, min_child_samples=100,
                  verbose=-1, seed=seed, num_threads=os.cpu_count() or 4)
    ds = lgb.Dataset(Xtr, ytr, weight=wtr, feature_name=cols)
    va = lgb.Dataset(Xes, yes, reference=ds)
    return lgb.train(params, ds, num_boost_round=ROUNDS, valid_sets=[ds, va],
                     callbacks=[lgb.early_stopping(50, verbose=False)])


def pick_memory(pool: list[tuple[int, int, float]], lo_idx: int,
                cap: int) -> tuple[np.ndarray, np.ndarray]:
    """从历史池里挑「窗外」的正确样本；按月轮转抽样，避免某一月垄断。

    pool 元素 = (df 行号, 月序号, 当时模型的分数)。行号 < lo_idx 才算窗外。
    """
    if cap <= 0 or not pool:
        return np.empty(0, np.int64), np.empty(0, np.float64)
    cand = [p for p in pool if p[0] < lo_idx]
    if not cand:
        return np.empty(0, np.int64), np.empty(0, np.float64)
    by_mon: dict[int, list] = {}
    for idx, mj, sc in cand:
        by_mon.setdefault(mj, []).append((idx, sc))
    months_sorted = sorted(by_mon)
    out_idx: list[int] = []
    out_sc: list[float] = []
    j = 0
    while len(out_idx) < cap:
        added = False
        for mj in months_sorted:
            arr = by_mon[mj]
            if j < len(arr):
                out_idx.append(arr[j][0])
                out_sc.append(arr[j][1])
                added = True
                if len(out_idx) >= cap:
                    break
        if not added:
            break
        j += 1
    return np.asarray(out_idx, np.int64), np.asarray(out_sc, np.float64)


def main() -> None:
    raw = kl.load_raw()
    df, feats = kl.build(raw)
    df, feats = add_flow(df, feats)
    df = add_target(df, HORIZON).filter(
        pl.col("target").is_not_null() & pl.col("close").is_not_null()).sort("close_time")
    df = df.with_columns(
        pl.col("close_time").cast(pl.Datetime("ms")).dt.strftime("%Y-%m").alias("ym"))
    X_all = matrix(df, feats)
    y_all = df["target"].to_numpy().astype(int)
    ct = df["close_time"].cast(pl.Int64).to_numpy()
    ym = df["ym"].to_numpy()
    log(f"特征 {len(feats)} 列 | 可用 {df.height:,} 行 | H={HORIZON} | 近期窗 {WIN_DAYS}d | "
        f"MEM_N={MEM_N} MEM_W={MEM_W:g} PER_MONTH={MEM_PER_MONTH} | stride={STRIDE}")

    months = [m for m in sorted(set(ym.tolist())) if START_MONTH <= m <= END_MONTH]
    per: dict[float, dict[str, list]] = {t: dict(s=[], y=[], mon=[]) for t in TPD_LIST}
    rows = []
    pool: list[tuple[int, int, float]] = []
    mem_sizes: list[int] = []

    for mj, m in enumerate(months):
        m_start = int(dt.datetime.strptime(m + "-01", "%Y-%m-%d")
                      .replace(tzinfo=dt.timezone.utc).timestamp() * 1000)
        lo = m_start - WIN_DAYS * 86_400_000
        i_tr = np.where((ct < m_start) & (ct >= lo))[0]
        i_te = np.where(ym == m)[0]
        if len(i_tr) < 100_000 or len(i_te) < 1_000:
            log(f"[{m}] 样本不足（{len(i_tr):,}/{len(i_te):,}），跳过")
            continue

        h_all = len(i_tr)
        c1 = int(h_all * (1 - ES_FRAC - THR_FRAC))
        c2 = int(h_all * (1 - THR_FRAC))
        seg = [i_tr[:c1], i_tr[c1:c2], i_tr[c2:]]
        if STRIDE > 1:
            seg[0], seg[1] = seg[0][::STRIDE], seg[1][::STRIDE]
        i_tr_s, i_es, i_th = seg

        i_mem, _ = pick_memory(pool, int(i_tr[0]), MEM_N)      # 只取窗外
        n_tr = len(i_tr_s)
        if len(i_mem):
            idx_tr = np.concatenate([i_tr_s, i_mem])
            w_tr = np.concatenate([np.ones(n_tr, np.float32),
                                   np.full(len(i_mem), MEM_W, np.float32)])
        else:
            idx_tr, w_tr = i_tr_s, np.ones(n_tr, np.float32)
        y_tr = y_all[idx_tr]
        mem_sizes.append(len(i_mem))

        Xtr, Xes, Xth = X_all[idx_tr], X_all[i_es], X_all[i_th]
        Xte = X_all[i_te]
        yes = y_all[i_es]
        p_th = np.zeros(len(i_th))
        p_te = np.zeros(len(i_te))
        its = []
        for seed in SEEDS:
            mdl = fit_w(Xtr, y_tr, feats, Xes, yes, w_tr, seed)
            p_th += mdl.predict(Xth, num_iteration=mdl.best_iteration) / len(SEEDS)
            p_te += mdl.predict(Xte, num_iteration=mdl.best_iteration) / len(SEEDS)
            its.append(mdl.best_iteration or ROUNDS)

        auc = float(roc_auc_score(y_all[i_te], p_te))
        days = (ct[i_te] // 86_400_000).astype(np.int64)
        nd = len(np.unique(days))
        s = p_te - .5
        line = (f"[{m}] recent={n_tr:,}+mem={len(i_mem):,}(w{MEM_W:g}) es={len(i_es):,} "
                f"thr={len(i_th):,} test={len(i_te):,}({nd}天) iter={int(np.mean(its))} "
                f"AUC={auc:.4f}")

        for tpd in TPD_LIST:
            thr = float(np.quantile(np.abs(p_th - .5), 1 - tpd / BARS_PER_DAY))
            hit = np.abs(s) >= thr
            nb = int(hit.sum())
            if nb < 30:
                line += f" | tpd{tpd:g}:单量太少"
                continue
            w = ((s[hit] > 0).astype(int) == y_all[i_te][hit]).astype(np.int8)
            per[tpd]["s"].append(s[hit])
            per[tpd]["y"].append(y_all[i_te][hit])
            per[tpd]["mon"].append(float(w.mean()))
            if tpd == PRIMARY_TPD:
                line += f" | 阈值={thr:.4f} {nb / nd:.1f}单/天 胜率={w.mean() * 100:.2f}%"
                # 入库：本月「被选中且预测正确」的样本，按月限量
                take = np.where(hit & (w == 1))[0][:MEM_PER_MONTH]
                pool.extend((int(i_te[j]), mj, float(p_te[j])) for j in take)
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
    p = per[PRIMARY_TPD]
    print(f"\n[主操作点 tpd={PRIMARY_TPD:g}] 逐月胜率（用于配对比较）：")
    print("  " + "  ".join(f"{r['ym'][2:]}:{v * 100:.1f}%"
                          for r, v in zip(rows, p["mon"])))
    print(f"逐月入场记忆条数：{mem_sizes}")
    log("DONE")


if __name__ == "__main__":
    main()
