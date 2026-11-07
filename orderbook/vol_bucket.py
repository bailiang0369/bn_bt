"""按"近期波动率尺度"分桶：验证低波动期是否更好预测、是否该只在低波动档下单。

用户假设（2026-11-07）
----------------------
固定的 bps 阈值不对：周末低波动时 0~2bp 就足以判断 3 分钟方向，而消息/美股开盘前后
20bp 也可能在 1 分钟走完。→ 用**过去波动率的中位数**作为自适应尺度，分桶看可预测性。

实现
----
  s_t  = 过去 7 天 |r_3m| 的中位数（bps），严格因果（只用 t 之前的 bar）
  桶   = 用 s_t 的滚动分位切成 5 档（每档样本均衡）
  评估 = ① 每档内：AUC、平均 |r_3m|、样本占比
         ② 每档内按占比分配单量（总 K=15×天数）后的 top-K 准确率 → 与全局 top-K 对比
         ③ "只在 s_t <= 阈值 时交易"的覆盖率-准确率曲线

判据：若低波动档的档内 top-K 明显高于全局，且覆盖率不太低，则"波动率分层选单"可用；
      若各档准确率接近，则分桶只是换个说法，没有增益。
"""
import os
import sys
import time

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.metrics import roc_auc_score  # noqa: E402

from orderbook.caliber_check import fit  # noqa: E402
from orderbook.znorm_eval import (DAILY_K, H, day_index, load, log, rolling_z)  # noqa: E402

DAY = 86_400_000
BARS_7D = 7 * 1440


def bucket_eval(z, y, s_t, dd, k_per_day, absr3, nb=5):
    """按 s_t 等数量切桶（rank 分桶，避免 s_t 有大量并列值时桶退化）。"""
    n_days = len(np.unique(dd))
    K = int(k_per_day * n_days)
    order = np.argsort(s_t, kind="mergesort")
    bounds = np.linspace(0, len(order), nb + 1).astype(int)
    rows, correct, used = [], 0, 0
    for b in range(nb):
        idx = order[bounds[b]:bounds[b + 1]]
        if len(idx) < 200:
            rows.append((b, len(idx), np.nan, np.nan, np.nan, np.nan, np.nan,
                         np.nan, np.nan))
            continue
        lo, hi = s_t[idx].min(), s_t[idx].max()
        k = max(int(round(K * len(idx) / len(y))), 20)
        pick = idx[np.argsort(-np.abs(z[idx]))[:k]]
        acc = float(((z[pick] > 0).astype(int) == y[pick]).mean())
        auc = float(roc_auc_score(y[idx], z[idx]))
        correct += int(((z[pick] > 0).astype(int) == y[pick]).sum())
        used += k
        rows.append((b, len(idx), lo, hi, s_t[idx].mean(),
                     float(np.nanmedian(absr3[idx])), auc, acc, k))
    layered = correct / max(used, 1)
    base_idx = np.argsort(-np.abs(z))[:K]
    base = float(((z[base_idx] > 0).astype(int) == y[base_idx]).mean())
    return rows, layered, base


def main():
    offs = [int(x) for x in os.environ.get("OFFS", "0").split(",")]
    seeds = [int(x) for x in os.environ.get("SEEDS", "42,43,44,45,46").split(",")]
    rounds = int(os.environ.get("ROUNDS", "60"))
    nb = int(os.environ.get("NB", "5"))
    t0 = time.time()

    df = load()
    ct = df["close_time"].to_numpy()
    y = df["target"].to_numpy().astype(np.int8)
    close = df["close"].to_numpy().astype(np.float64)
    cols = [c for c in df.columns if c not in ("target", "close_time")]
    X = df.select(cols).to_numpy().astype(np.float32)
    del df

    # s_t：过去 7 天 |r_3m| 中位数（bps），因果（shift(1) 后滚动）
    r3 = np.full(len(close), np.nan)
    r3[:-H] = (close[H:] / close[:-H] - 1.0) * 1e4
    s_t = (pl.Series(np.abs(r3)).shift(1).rolling_median(BARS_7D, min_samples=2880)
           .to_numpy())
    log(f"[vol] s_t 中位数={np.nanmedian(s_t):.2f}bp  p10={np.nanpercentile(s_t,10):.2f} "
        f"p90={np.nanpercentile(s_t,90):.2f}")

    L = X.shape[0]
    k1_0, k2_0 = int(L * 0.64), int(L * 0.80)
    for off in offs:
        k1, k2 = k1_0 - off * 1440, k2_0 - off * 1440
        va_m = (ct >= int(ct[k1])) & (ct < int(ct[k2]))
        te_m = ct >= int(ct[k2])
        te_idx = np.where(te_m)[0]
        s = np.zeros(L, dtype=np.float32)
        s[te_idx] = fit(X, y, np.arange(0, k1 - H), cols, te_idx, seeds, rounds)
        z = rolling_z(s, 1440)[te_m]
        yy = y[te_m]
        sd = s_t[te_m]
        dd = day_index(ct[te_m])
        ok = np.isfinite(sd)
        absr3 = np.abs(r3[te_m])
        z, yy, sd, dd, absr3 = z[ok], yy[ok], sd[ok], dd[ok], absr3[ok]
        log(f"\n===== off={off} n={len(yy)} 天={len(np.unique(dd))} 基准={yy.mean():.4f} =====")

        rows, layered, base = bucket_eval(z, yy, sd, dd, DAILY_K, absr3, nb)
        log(f"{'档':>3}{'样本数':>8}{'占比':>7}  {'s_t区间bp':>13}"
            f"{'s_t均值':>9}{'|r3|中位':>9}{'AUC':>8}{'档内topK':>9}{'单量':>7}")
        for b, n, lo, hi, sv, med, auc, acc, k in rows:
            log(f"{b:>3}{n:>8}{n/len(yy):>7.2%}  [{lo:>5.1f},{hi:>5.1f}]"
                f"{sv:>9.2f}{med:>9.1f}{auc:>8.4f}{acc:>9.4f}{k:>7}")

        log(f"[对比] 全局 topK={base:.4f}  分层取单={layered:.4f}  Δ={(layered-base)*100:+.2f}pp")
        med_s = np.nanmedian(sd)
        log(f"\n[只在低波动档交易] s_t 中位数={med_s:.2f}bp  阈值=中位数×倍数")
        log(f"{'倍数':>6}{'阈值bp':>9}{'覆盖率':>8}{'样本数':>8}{'子集AUC':>9}{'topK准确率':>11}")
        for mult in (0.6, 0.8, 1.0, 1.5, 2.0):
            thr = med_s * mult
            m = sd <= thr
            if m.sum() < 500:
                continue
            k = int(DAILY_K * len(np.unique(dd)) * m.mean())
            pick = np.argsort(-np.abs(z[m]))[:k]
            acc = float(((z[m][pick] > 0).astype(int) == yy[m][pick]).mean())
            auc = float(roc_auc_score(yy[m], z[m]))
            log(f"{mult:>6.1f}{thr:>9.2f}{m.mean():>8.2%}{int(m.sum()):>8}"
                f"{auc:>9.4f}{acc:>11.4f}")
    log(f"\n[done] {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
