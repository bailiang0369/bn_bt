"""指标稳定性实验：同一模型、只换随机种子，看 static / roll_1d / roll_3d 各自抖多少。

背景
----
2026-11-03 那次运行报"盘口腿 AUC 0.5412 / 全局 topK 0.6410"；
今天同参数重训得到 **完全相同的 AUC 0.5412**，但 static topK 只有 **0.5958**（差 4.5pp）。
AUC 一样而 topK 差这么多 → 差异在分数分布的尾部，而不是判别力。

如果 static top-K 本身就是高方差估计量（K=1440 ≈ 1% 尾部，靠 |z| 排序），
那 0.6410 就是某次种子的幸运尾部，不该作为目标；roll_1d 若稳定得多，才该用它做决策。

设计
----
3 组种子（42-46 / 142-146 / 242-246），各 5 seeds、60 轮、同一 train/val/test，
输出每组的 static / roll_1d / roll_3d / AUC / 覆盖天数 / 单天峰值，最后打印**跨组标准差**。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orderbook.caliber_check import detail, fit  # noqa: E402
from orderbook.znorm_eval import (DAILY_K, H, complete_days, daily_topk_eval,  # noqa: E402
                                 day_index, load, log, rolling_z)

from sklearn.metrics import roc_auc_score  # noqa: E402


def main():
    groups = [g for g in os.environ.get(
        "SEED_GROUPS", "42,43,44,45,46;142,143,144,145,146;242,243,244,245,246").split(";") if g]
    rounds = int(os.environ.get("ROUNDS", "60"))
    off = int(os.environ.get("OFF", "0"))

    df = load()
    ct = df["close_time"].to_numpy()
    y = df["target"].to_numpy().astype(np.int8)
    cols = [c for c in df.columns if c not in ("target", "close_time")]
    X = df.select(cols).to_numpy().astype(np.float32)
    del df
    L = X.shape[0]
    k1, k2 = int(L * 0.64) - off * 1440, int(L * 0.80) - off * 1440
    va_m = (ct >= int(ct[k1])) & (ct < int(ct[k2]))
    te_m = ct >= int(ct[k2])
    te_idx = np.where(te_m)[0]
    dd = day_index(ct[te_m])
    mt = complete_days(dd, 1200)
    tr = np.arange(0, k1 - H)
    log(f"[data] rows={L} feats={len(cols)} off={off} test={len(te_idx)} 天={len(np.unique(dd))}")

    log(f"\n{'种子组':<16}{'static':>9}{'roll_1d':>9}{'roll_3d':>9}{'AUC':>8}{'覆盖天':>8}{'峰值':>6}")
    res = {}
    for gi, g in enumerate(groups):
        seeds = [int(x) for x in g.split(",") if x.strip()]
        s = np.zeros(L)
        s[te_idx] = fit(X, y, tr, cols, te_idx, seeds, rounds)
        zs = (s - s[va_m].mean()) / (s[va_m].std() + 1e-9)
        row = {}
        for tag, z in (("static", zs),
                       ("roll_1d", rolling_z(s, 1440)),
                       ("roll_3d", rolling_z(s, 4320))):
            acc, cover, nd, peak = detail(z[te_m], y[te_m], dd, DAILY_K)
            row[tag] = acc
            row[tag + "_cover"] = cover
            row[tag + "_peak"] = peak
        row["auc"] = float(roc_auc_score(y[te_m], zs[te_m]))
        row["daily"] = float(daily_topk_eval(zs[te_m], y[te_m], dd, mt, DAILY_K).mean())
        res[gi] = row
        log(f"{g:<16}{row['static']:>9.4f}{row['roll_1d']:>9.4f}{row['roll_3d']:>9.4f}"
            f"{row['auc']:>8.4f}{row['static_cover']:>8}{row['static_peak']:>6}")

    log(f"\n{'指标':<10}{'均值':>8}{'标准差':>8}{'极差':>8}")
    for k in ("static", "roll_1d", "roll_3d", "auc"):
        v = np.array([res[g][k] for g in res])
        log(f"{k:<10}{v.mean():>8.4f}{v.std(ddof=1):>8.4f}{v.max()-v.min():>8.4f}")
    log(f"\n[结论] 若 static 的极差明显大于 roll_1d，则 static topK 是噪声指标，"
        f"历史 0.6410 属于尾部幸运，不可作为优化目标。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
