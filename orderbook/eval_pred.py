"""按「测试集总日均」口径评估逐行预测：全局 top-K，K = 每日单量 × 天数。

为什么是这个口径（用户 2026-11-03 明确）：
  - 目标是 **测试集总量 / 天数 ≈ 15 单/天**，不是"每天必须凑满 15 单"；
  - 因此不按天取样，而是全 test 按 |score| 排序取前 K = per_day × n_days 单；
  - 信号多的天多下单、没信号的天不下单，属于正常（长期平均达标即可）。
对照口径：`daily_topk_eval`（每天取 top-15）会强行给弱信号天凑单，系统性低估。

用法：
    python -m orderbook.eval_pred --pred tmp/fusionkl/fusion_kl_pred.parquet
    python -m orderbook.eval_pred --pred xxx.parquet --sweep 10,15,20,30 \
        --cols "lr=0.130*z_ob+0.029*z_kl"
"""
import sys
import argparse
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SKIP = {"close_time", "target", "day", "split"}


def day_index(ct_ms):
    return (np.asarray(ct_ms) // 86_400_000).astype(np.int64)


def topk_acc(score, y, k):
    """全局 top-k：按 |score| 取前 k，方向 = sign(score)。"""
    k = min(k, len(y))
    idx = np.argsort(-np.abs(score))[:k]
    return float(((score[idx] > 0).astype(int) == y[idx]).mean())


def day_stats(score, y, d, k):
    """选中的 k 单落在多少天、每天几单、这些天的日均准确率。"""
    idx = np.argsort(-np.abs(score))[:k]
    dd = d[idx]
    u, c = np.unique(dd, return_counts=True)
    accs = []
    for day in u:
        m = dd == day
        accs.append(float(((score[idx][m] > 0).astype(int) == y[idx][m]).mean()))
    accs = np.array(accs)
    return len(u), c.min(), c.max(), float(accs.mean()), float(np.median(accs))


def main():
    ap = argparse.ArgumentParser(description="全局 top-K（测试集总日均）口径评估")
    ap.add_argument("--pred", required=True, help="逐行预测 parquet（需含 close_time/target + 分数列）")
    ap.add_argument("--per-day", type=int, default=15, help="目标日均单量")
    ap.add_argument("--sweep", default="10,15,20,25,30", help="额外扫描的日均单量")
    ap.add_argument("--cols", default="", help="附加分数列，逗号分隔 name=expr，可用已有列名")
    ap.add_argument("--hour", default="", help="UTC 时段过滤，如 18-24（支持跨零点 22-6）")
    ap.add_argument("--znorm", default="", help="分数归一化：roll<W>=过去 W 根滚动 z（因果）；day=按天 z（含同日未来，仅诊断）")
    a = ap.parse_args()

    df = pl.read_parquet(a.pred).sort("close_time")
    ct = df["close_time"].to_numpy()
    y = df["target"].to_numpy().astype(np.int8)
    d = day_index(ct)
    n_days = len(np.unique(d))
    n = len(y)

    scores = {}
    for c in df.columns:
        if c in SKIP:
            continue
        v = df[c].to_numpy()
        if np.issubdtype(v.dtype, np.number):
            scores[c] = v.astype(np.float64)

    if a.cols:
        env = dict(scores)
        env["np"] = np
        for part in a.cols.split(","):
            part = part.strip()
            if not part:
                continue
            name, expr = part.split("=", 1)
            scores[name.strip()] = np.asarray(eval(expr, {"__builtins__": {}}, env), dtype=np.float64).ravel()

    # 分数归一化（在全部 test 行上做，之后才按时段过滤，保证滚动历史连续）
    if a.znorm:
        if a.znorm.startswith("roll"):
            w = int(a.znorm[4:])
            for k in list(scores):
                s = pl.Series(scores[k])
                m = s.shift(1).rolling_mean(w)
                sd = s.shift(1).rolling_std(w)
                scores[k] = ((s - m) / (sd + 1e-9)).fill_null(0.0).to_numpy()
            tag = f"roll{w}（因果，回看 {w} 根）"
        elif a.znorm == "day":
            for k in list(scores):
                t = pl.DataFrame({"v": scores[k], "d": d})
                z = t.select(((pl.col("v") - pl.col("v").mean().over("d")) /
                              (pl.col("v").std().over("d") + 1e-9)).alias("z"))["z"].to_numpy()
                scores[k] = np.nan_to_num(z, nan=0.0)
            tag = "day（按天，含同日未来——仅诊断用）"
        else:
            raise SystemExit(f"未知 --znorm: {a.znorm}")
        print(f"[znorm] {tag}")

    if a.hour:
        lo, hi = (int(x) for x in a.hour.split("-"))
        hh = ((ct % 86_400_000) // 3_600_000).astype(int)
        keep = (hh >= lo) & (hh < hi) if lo < hi else ((hh >= lo) | (hh < hi))
        ct, y, d = ct[keep], y[keep], d[keep]
        scores = {k: v[keep] for k, v in scores.items()}
        n_days = len(np.unique(d))
        n = len(y)
        print(f"[hour] 只保留 UTC {lo}-{hi} 点：n={n} 天数={n_days}")

    print(f"[data] {a.pred}")
    print(f"[data] n={n} 天数={n_days} 基准={y.mean():.4f}  "
          f"日均{a.per_day}单 → 全局 K={a.per_day * n_days}（覆盖 {a.per_day * n_days / n:.4f}）")

    print(f"\n{'打分':<12}{'AUC':>9}{'全局topK':>10}{'覆盖天数':>9}{'单/天min':>10}{'单/天max':>10}{'该天日均':>10}")
    rows = []
    K = a.per_day * n_days
    for name, s in scores.items():
        acc = topk_acc(s, y, K)
        auc = float(roc_auc_score(y, s))
        cov_days, cmin, cmax, dm, dmed = day_stats(s, y, d, K)
        rows.append((name, auc, acc))
        print(f"{name:<12}{auc:>9.4f}{acc:>10.4f}{cov_days:>9}"
              f"{cmin:>10}{cmax:>10}{dm:>10.4f}")

    best = max(rows, key=lambda r: r[2])
    print(f"\n[结论] 全局 top{K}（日均{a.per_day}单）最优 {best[0]} = {best[2]:.4f}  "
          f"（目标 0.65 → {'达成' if best[2] >= 0.65 else '未达成'}）")

    perdays = sorted({int(x) for x in a.sweep.split(",") if x.strip()} | {a.per_day})
    print(f"\n[日均单量扫描] 全局 top{perdays[0] * n_days}~top{perdays[-1] * n_days}")
    print(f"{'打分':<12}" + "".join(f"{('K/d=' + str(p)):>12}" for p in perdays))
    for name, s in scores.items():
        line = f"{name:<12}"
        for p in perdays:
            line += f"{topk_acc(s, y, p * n_days):>12.4f}"
        print(line)


if __name__ == "__main__":
    main()
