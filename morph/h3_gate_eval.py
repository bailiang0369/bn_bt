#!/usr/bin/env python3
"""h=3 二阶段下单过滤的评估口径（离线复核 + Kaggle 入口共用）。

为什么要单独一个模块
--------------------
`morph/h3_fuse_cross.py:paired_boot()` 比较的是「同一批样本上的两个模型」；闸门比较的是
「同一个模型、放行子集不同」。而且 `融合实验记录.md` §17 的 K = 15×放行天数 会随覆盖率
自动缩小，把「少做」机械地变成「做得准」。这里把口径固定成两条：

* **固定单量（主判据）**：停单日的分数置 0，仍取 K = 15×总天数。与基线在**同一 K、
  同一批天**上做按天块配对 bootstrap（`续接说明.md` §4.1），差值不含缩量效应。
* **缩量（业务参考）**：K = 15×放行天数，只在放行样本里取单；同时报同 K 的全局
  top-K 对照与「随机放行同样天数」的 placebo。

两条都同时报覆盖率、每日单量分布、单量-胜率曲线（`续接说明.md` §8.1 必须防的坑）。

离线用法（对已落盘的 test 预测做基线复现 + 功效核算，不训练）：
    python -m morph.h3_gate_eval --pred <h3_ab_pred.parquet> --boot 400
阈值只能在 val 定，本模块不做任何阈值选择；test 预测仅用于估计方差与复现口径。
"""
from __future__ import annotations

import argparse

import numpy as np
import polars as pl

DAY_MS = 86_400_000
TIERS = (("日均15单", None), ("1%", 0.01), ("5%", 0.05), ("25%", 0.25))


def log(msg: str) -> None:
    print(msg, flush=True)


def day_id(ct: np.ndarray) -> np.ndarray:
    return (np.asarray(ct) // DAY_MS).astype(np.int64)


def topk_win(score: np.ndarray, y: np.ndarray, k: int) -> float:
    """按 |score| 取前 k 单的胜率；score 必须已中心化（p-0.5），>0 判涨。"""
    k = min(int(k), len(y))
    if k <= 0:
        return float("nan")
    idx = np.argpartition(-np.abs(score), k - 1)[:k]
    return float(((score[idx] > 0).astype(np.int8) == y[idx]).mean())


def k_of(frac: float | None, n: int, n_days: int, tpd: int) -> int:
    return tpd * n_days if frac is None else max(1, int(frac * n))


def four_tiers(score: np.ndarray, y: np.ndarray, days: np.ndarray,
               tpd: int = 15) -> dict[str, float]:
    nd = len(np.unique(days))
    return {name: topk_win(score, y, k_of(frac, len(y), nd, tpd))
            for name, frac in TIERS}


def report(tag: str, score: np.ndarray, y: np.ndarray, days: np.ndarray,
           tpd: int = 15) -> dict[str, float]:
    m = four_tiers(score, y, days, tpd)
    log(f"[{tag}] " + " | ".join(f"@{k}={v * 100:.2f}%" for k, v in m.items())
        + f" | n={len(y):,} 天={len(np.unique(days))}")
    return m


def make_plans(days: np.ndarray, n_boot: int = 1000, seed: int = 7) -> list[tuple]:
    """按天块有放回重采样；与 h3_fuse_cross.paired_boot 完全同口径。"""
    uniq = np.unique(days)
    pos = {d: np.flatnonzero(days == d) for d in uniq}
    rng = np.random.default_rng(seed)
    plans = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, len(uniq), replace=True)
        plans.append((np.concatenate([pos[d] for d in pick]), len(pick)))
    return plans


def boot_diffs(a: np.ndarray, b: np.ndarray, y: np.ndarray,
               plans: list[tuple], tpd: int = 15) -> dict[str, dict]:
    """差值 = b - a；返回每档的均值 / CI95 / 标准差。"""
    out: dict[str, dict] = {}
    for name, frac in TIERS:
        diffs = np.empty(len(plans))
        for m, (idx, nd) in enumerate(plans):
            k = k_of(frac, len(idx), nd, tpd)
            diffs[m] = (topk_win(b[idx], y[idx], k)
                        - topk_win(a[idx], y[idx], k))
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        out[name] = dict(diff=float(diffs.mean()), lo=float(lo), hi=float(hi),
                         sd=float(diffs.std(ddof=1)))
    return out


def paired_boot(a: np.ndarray, b: np.ndarray, y: np.ndarray, plans: list[tuple],
                tag: str, tpd: int = 15) -> dict[str, dict]:
    """主判据：配对 bootstrap 差值 CI 不跨 0。"""
    res = boot_diffs(a, b, y, plans, tpd)
    for name, r in res.items():
        verdict = "显著" if r["lo"] * r["hi"] > 0 else "不显著"
        log(f"[{tag}] @{name}: 差={r['diff'] * 100:+.2f}pp "
            f"CI95[{r['lo'] * 100:+.2f},{r['hi'] * 100:+.2f}] {verdict}")
    return res


# ---------------------------------------------------------------- 闸门
def apply_gate(score: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    """固定单量口径：停单日分数置 0（|score|=0 排在最后，不会被选中）。"""
    return np.where(allowed, score, 0.0)


def gate_by_days(days: np.ndarray, trade_days: np.ndarray) -> np.ndarray:
    return np.isin(days, trade_days)


def coverage(allowed: np.ndarray, days: np.ndarray) -> dict[str, float]:
    ud = np.unique(days)
    traded = np.unique(days[allowed])
    return dict(day_cov=len(traded) / len(ud), row_cov=float(allowed.mean()),
                trade_days=len(traded), total_days=len(ud))


def daily_orders(score: np.ndarray, days: np.ndarray, allowed: np.ndarray,
                 tpd: int = 15) -> dict[str, float]:
    """固定单量口径下每天实际几单；「日均 15 单」是平均假象，必须看分布。"""
    k = tpd * len(np.unique(days))
    idx = np.argpartition(-np.abs(apply_gate(score, allowed)), k - 1)[:k]
    _, c = np.unique(days[idx], return_counts=True)
    return dict(mean=float(c.mean()), std=float(c.std()), p50=float(np.median(c)),
                min=float(c.min()), max=float(c.max()), zero_days=int(len(c) < len(np.unique(days))))


def size_curve(score: np.ndarray, y: np.ndarray, allowed: np.ndarray,
               tpd: int = 15, fracs=(0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.25)) -> None:
    """单量-胜率曲线：同一覆盖率下闸门 vs 全局，避免只报一个胜率。"""
    b = apply_gate(score, allowed)
    log(f"[曲线] {'覆盖':>7}{'单量':>8}{'全局':>9}{'闸门':>9}{'差值pp':>9}")
    for q in fracs:
        k = max(1, int(q * len(y)))
        ga, gb = topk_win(score, y, k), topk_win(b, y, k)
        log(f"[曲线] {q:>7.1%}{k:>8,}{ga * 100:>8.2f}%{gb * 100:>8.2f}%"
            f"{(gb - ga) * 100:>+9.2f}")


def evaluate_gate(score: np.ndarray, y: np.ndarray, days: np.ndarray,
                  allowed: np.ndarray, plans: list[tuple], tag: str,
                  tpd: int = 15, n_placebo: int = 200, seed: int = 11) -> dict:
    """闸门完整评估：固定单量主判据 + 缩量业务口径 + 随机放行 placebo。"""
    cov = coverage(allowed, days)
    log(f"[{tag}] 覆盖率 天={cov['day_cov']:.1%} ({cov['trade_days']}/{cov['total_days']}) "
        f"样本={cov['row_cov']:.1%}")
    b = apply_gate(score, allowed)
    k_full = tpd * cov["total_days"]
    if int(allowed.sum()) < k_full:
        log(f"[{tag}] [!] 放行样本 {int(allowed.sum()):,} < 固定单量 K={k_full:,}，"
            f"固定口径会把 0 分样本选进来，结果不可用")
    log(f"[{tag}] 基线/闸门四档（固定单量 K={k_full:,}）")
    report(f"{tag} 基线", score, y, days, tpd)
    report(f"{tag} 闸门", b, y, days, tpd)
    res = paired_boot(score, b, y, plans, f"{tag} 配对", tpd)

    # 缩量口径（§17 的 K=15×放行天数）+ 同 K 全局对照
    k_shrink = tpd * cov["trade_days"]
    idx = np.flatnonzero(allowed)
    acc_shrink = topk_win(score[idx], y[idx], k_shrink)
    acc_samek = topk_win(score, y, k_shrink)
    log(f"[{tag}] 缩量 K={k_shrink:,}: 交易段={acc_shrink * 100:.2f}% "
        f"同K全局={acc_samek * 100:.2f}% 差={(acc_shrink - acc_samek) * 100:+.2f}pp")

    # placebo：随机放行同样天数，检验「停掉这些天」本身是否有技能
    rng = np.random.default_rng(seed)
    ud = np.unique(days)
    nd = max(1, int(round(cov["trade_days"])))
    placebo = {name: np.empty(n_placebo) for name, _ in TIERS}
    for m in range(n_placebo):
        pick = rng.choice(ud, nd, replace=False)
        pb = apply_gate(score, gate_by_days(days, pick))
        for name, frac in TIERS:
            placebo[name][m] = topk_win(pb, y, k_of(frac, len(y), len(ud), tpd))
    for name, _ in TIERS:
        arr = placebo[name]
        gate_acc = topk_win(b, y, k_of(dict(TIERS)[name], len(y), len(ud), tpd))
        p = float((arr >= gate_acc).mean())
        log(f"[{tag}] placebo @{name}: 闸门={gate_acc * 100:.2f}% "
            f"随机均值={arr.mean() * 100:.2f}% p95={np.percentile(arr, 95) * 100:.2f}% "
            f"经验p={p:.3f}")

    dist = daily_orders(score, days, allowed, tpd)
    log(f"[{tag}] 每日单量 mean={dist['mean']:.1f} std={dist['std']:.1f} "
        f"p50={dist['p50']:.0f} min={dist['min']:.0f} max={dist['max']:.0f} "
        f"零单天={dist['zero_days']}")
    size_curve(score, y, allowed, tpd)
    return dict(cov=cov, paired=res, shrink=acc_shrink, same_k=acc_samek, daily=dist)


# ---------------------------------------------------------------- 状态量
def rolling_trailing(score: np.ndarray, y: np.ndarray, days: np.ndarray,
                     window_days: int = 45, tpd: int = 15) -> tuple[np.ndarray, np.ndarray]:
    """§17 滚动监控：第 i 天用 days[i-W:i]（不含当天）的 top-K 准确率。严格因果。

    返回 (uniq_days, trail)，冷启动期 trail=nan。调用方只应喂 out-of-sample 段
    （val+test），不要把 train 的样本内预测混进来抬高滚动准确率。
    """
    uniq = np.unique(days)
    pos = {d: np.flatnonzero(days == d) for d in uniq}
    trail = np.full(len(uniq), np.nan)
    for i in range(len(uniq)):
        lo = max(0, i - window_days)
        if lo >= i:
            continue
        idx = np.concatenate([pos[uniq[j]] for j in range(lo, i)])
        trail[i] = topk_win(score[idx], y[idx], tpd * (i - lo))
    return uniq, trail


def trailing_gate(days: np.ndarray, uniq: np.ndarray, trail: np.ndarray,
                  threshold: float) -> np.ndarray:
    """冷启动（trail=nan）默认放行，与 eval_pred.rolling_monitor 一致。"""
    keep = np.isnan(trail) | (trail >= threshold)
    return gate_by_days(days, uniq[keep])


# ---------------------------------------------------------------- 功效
def mde_scan(score: np.ndarray, y: np.ndarray, days: np.ndarray,
             plans: list[tuple], coverages=(0.9, 0.7, 0.5, 0.34),
             ref_score: np.ndarray | None = None, tpd: int = 15,
             n_rep: int = 3, seed: int = 13) -> None:
    """启动实验前先算最小可检测效应（§4.5）。

    * coverages：用「随机放行该比例天数」的无技能闸门估计差值噪声尺度；
    * 覆盖率 1.0 的参照用两个独立变体的配对差值（若给了 ref_score）；
    * MDE(CI) = 1.96×SD ≈ 让 CI 不跨 0 所需的最小真效应；
      MDE(80% power) = 2.8×SD，规划实验时更稳妥。
    """
    ud = np.unique(days)
    rng = np.random.default_rng(seed)
    if ref_score is not None:
        r = boot_diffs(score, ref_score, y, plans, tpd)
        log("[MDE] 参照（全量、两变体配对差值，复现 A 组 CI）")
        for name, v in r.items():
            log(f"[MDE]   @{name}: CI半宽={(v['hi'] - v['lo']) / 2 * 100:.2f}pp "
                f"MDE(CI)={1.96 * v['sd'] * 100:.2f}pp "
                f"MDE(80%)={2.8 * v['sd'] * 100:.2f}pp")
    for cov in coverages:
        sds = {name: [] for name, _ in TIERS}
        for _ in range(n_rep):
            nd = max(1, int(round(len(ud) * cov)))
            pick = rng.choice(ud, nd, replace=False)
            b = apply_gate(score, gate_by_days(days, pick))
            r = boot_diffs(score, b, y, plans, tpd)
            for name, _ in TIERS:
                sds[name].append(r[name]["sd"])
        log(f"[MDE] 覆盖率 {cov:.0%}（随机放行，{n_rep} 次中位 SD）")
        for name, _ in TIERS:
            se = float(np.median(sds[name]))
            log(f"[MDE]   @{name}: MDE(CI)={1.96 * se * 100:.2f}pp "
                f"MDE(80%)={2.8 * se * 100:.2f}pp")


# ---------------------------------------------------------------- 离线 CLI
def main() -> None:
    ap = argparse.ArgumentParser(description="h3 闸门实验：离线基线复现 + MDE 核算")
    ap.add_argument("--pred", required=True, help="h3_ab_pred.parquet（含 split/variant/close_time/target/score）")
    ap.add_argument("--variant", default="ob")
    ap.add_argument("--splits", default="time,row")
    ap.add_argument("--ref-variant", default="ob+cross", help="全量配对噪声参照变体")
    ap.add_argument("--boot", type=int, default=400)
    ap.add_argument("--coverages", default="0.9,0.7,0.5,0.34")
    ap.add_argument("--tpd", type=int, default=15)
    a = ap.parse_args()

    df = pl.read_parquet(a.pred)
    covs = [float(x) for x in a.coverages.split(",") if x.strip()]
    for how in [s.strip() for s in a.splits.split(",") if s.strip()]:
        d = df.filter((pl.col("split") == how) & (pl.col("variant") == a.variant)) \
              .sort("close_time")
        if not d.height:
            log(f"[{how}] 没有 variant={a.variant} 的预测，跳过")
            continue
        score = (d["score"].to_numpy() - 0.5).astype(np.float64)
        y = d["target"].to_numpy().astype(np.int8)
        days = day_id(d["close_time"].to_numpy())
        log(f"\n===== split={how} variant={a.variant} rows={d.height:,} =====")
        report(f"{how} 基线", score, y, days, a.tpd)
        plans = make_plans(days, a.boot)
        ref = df.filter((pl.col("split") == how) & (pl.col("variant") == a.ref_variant)) \
                .sort("close_time")
        ref_score = ((ref["score"].to_numpy() - 0.5).astype(np.float64)
                     if ref.height == d.height else None)
        if ref_score is not None:
            paired_boot(score, ref_score, y, plans, f"{how} {a.variant}->{a.ref_variant} 复现", a.tpd)
        mde_scan(score, y, days, plans, coverages=covs, ref_score=ref_score, tpd=a.tpd)
    log("DONE")


if __name__ == "__main__":
    main()
