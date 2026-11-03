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
    # 滚动监控（模拟实盘：用过去 W 天已实现结果决定当天是否下单）
    python -m orderbook.eval_pred --pred xxx.parquet --rm \
        --rm-window-days 45 --rm-thresholds "0.64,0.62,0.60"
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


def rolling_monitor(ct, y, d, s, per_day, window_days, thresholds):
    """因果滚动监控：只用 t 之前 window_days 天的已实现结果，决定第 t 天是否交易。

    返回 (baseline_acc, trailing, days, results)。trailing[i] = 第 i 天结束时的滚动准确率
    （用 days[i-W:i] 这段，不含第 i 天，严格因果），用于"当天要不要下单"的决策。
    results[thr] = (仅在 trailing>=thr 的天交易时的全局 top-K 准确率, 交易天数, 总天数)。
    """
    days = np.unique(d)
    n_days = len(days)
    day_idx = {int(u): np.where(d == u)[0] for u in days}

    base = topk_acc(s, y, per_day * n_days)

    trailing = np.full(n_days, np.nan)
    for i in range(n_days):
        lo = max(0, i - window_days)
        if lo >= i:
            continue
        idx = np.concatenate([day_idx[int(u)] for u in days[lo:i]])
        nw = i - lo
        if len(idx) >= 10:
            trailing[i] = topk_acc(s[idx], y[idx], per_day * nw)

    results = {}
    rng = np.random.default_rng(0)
    for thr in thresholds:
        # 冷启动（前 W 天没有历史）默认交易，避免开局长期空仓
        trade = np.array([np.isnan(trailing[i]) or trailing[i] >= thr
                          for i in range(n_days)])
        if not trade.any():
            results[thr] = (float("nan"), 0, n_days, float("nan"))
            continue
        nd = int(trade.sum())
        idx = np.concatenate([day_idx[int(u)] for u in days[trade]])
        acc = topk_acc(s[idx], y[idx], per_day * nd)
        # 随机对照：随机保留同样天数（20 次平均），剥离"K 变小→top-K 更极端"的机械效应
        ra = []
        for _ in range(20):
            pick = rng.choice(n_days, nd, replace=False)
            ridx = np.concatenate([day_idx[int(u)] for u in days[pick]])
            ra.append(topk_acc(s[ridx], y[ridx], per_day * nd))
        results[thr] = (acc, nd, n_days, float(np.mean(ra)))
    return base, trailing, days, results


def main():
    ap = argparse.ArgumentParser(description="全局 top-K（测试集总日均）口径评估")
    ap.add_argument("--pred", required=True, help="逐行预测 parquet（需含 close_time/target + 分数列）")
    ap.add_argument("--per-day", type=int, default=15, help="目标日均单量")
    ap.add_argument("--sweep", default="10,15,20,25,30", help="额外扫描的日均单量")
    ap.add_argument("--cols", default="", help="附加分数列，逗号分隔 name=expr，可用已有列名")
    ap.add_argument("--hour", default="", help="UTC 时段过滤，如 18-24（支持跨零点 22-6）")
    ap.add_argument("--znorm", default="", help="分数归一化：roll<W>=过去 W 根滚动 z（因果）；day=按天 z（含同日未来，仅诊断）")
    ap.add_argument("--rm", action="store_true", help="滚动监控模式：用过去 W 天已实现结果决定当天是否交易")
    ap.add_argument("--rm-score", default="z_ob+0.2*z_kl", help="监控/交易的打分表达式")
    ap.add_argument("--rm-window-days", type=int, default=45, help="滚动回看窗口（天）")
    ap.add_argument("--rm-thresholds", default="0.62,0.60,0.58", help="止损阈值（滚动准确率）")
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

    env = dict(scores)
    env["np"] = np
    if a.cols:
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

    # ---------- 滚动监控（因果止损）----------
    if a.rm:
        s = np.asarray(eval(a.rm_score, {"__builtins__": {}}, env), dtype=np.float64).ravel()
        thr = sorted({float(x) for x in a.rm_thresholds.split(",") if x.strip()})
        base, trailing, days, res = rolling_monitor(ct, y, d, s, a.per_day, a.rm_window_days, thr)
        print(f"\n[滚动监控] 打分={a.rm_score}  窗口={a.rm_window_days}天  基线全局topK={base:.4f}")
        print(f"{'止损阈值':<10}{'交易天数':>9}{'覆盖率':>9}{'停单天数':>9}{'交易段准确率':>12}{'随机对照':>10}{'净增益':>9}")
        for t in thr:
            acc, nd, tot, ra = res[t]
            g = acc - ra if not (np.isnan(acc) or np.isnan(ra)) else float("nan")
            print(f"{t:<10.3f}{nd:>9}{nd / tot:>9.2f}{tot - nd:>9}"
                  f"{acc:>12.4f}{ra:>10.4f}{g:>9.4f}")
        td = trailing[~np.isnan(trailing)]
        if len(td):
            print(f"[滚动曲线] 有效点={len(td)}  均={td.mean():.4f}  中位={np.median(td):.4f}  "
                  f"p10={np.percentile(td, 10):.4f}  最差={td.min():.4f}")
            # 采样打印曲线（每 5 个点）
            step = max(1, len(td) // 12)
            print("[滚动曲线采样] " + "  ".join(f"{v:.3f}" for v in td[::step]))
        return

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
