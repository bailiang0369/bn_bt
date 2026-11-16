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
from morph.h10_signal import LGB_PRESETS, add_flow, roll_worst  # noqa: E402
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

# 记忆的用途（2026-11-16 新增）：
#   train  = 历史预测进训练集（原有行为，即 mem90 实验）
#   export = 历史预测**不进训练集**，只按月收集，输出「信号强度分档 → 实际胜率」
#            与滚动因果否决评估（回答：历史预测有没有可用的排序信息）
#   off    = 完全不收集
MEM_MODE = os.environ.get("MEM_MODE", "train").strip().lower()
USE_MEM_TRAIN = MEM_MODE == "train"
COLLECT_HIST = MEM_MODE in ("train", "export")
HIST_BINS = int(os.environ.get("HIST_BINS", "5"))          # 命中样本按信号强度分几档
HIST_MIN = int(os.environ.get("HIST_MIN", "250"))          # 历史样本够多少条才启用否决
HIST_MIN_BUCKET = int(os.environ.get("HIST_MIN_BUCKET", "40"))
_rp = os.environ.get("REPORT_PAY", "").strip()
REPORT_PAY = float(_rp) if _rp else PAYOUTS[0]      # 诊断用的主赔率，默认取 PAYOUTS[0]

# 阈值稳健化（2026-11-16）：同一次训练里并行对比多种「阈值估计」方式，完全配对。
# 病根：thr 由 test 前 THR_FRAC 段（90d×10%≈9 天）估 99.4% 分位，样本太少 → 阈值月度
# 抖动 8 倍、单量 1.5~51 单/天。这里测四种估计方式：
#   raw  = 现状（前 9 天段的分位）
#   ewma = 与上月阈值指数平滑（THR_ALPHA 为当月权重）
#   hist = 过去 N 个月 test 全 bar 的分位（样本扩大到 K 个月）
#   mix  = raw 与 hist 的算术平均
THR_MODES = [x.strip() for x in os.environ.get("THR_MODES", "").split(",") if x.strip()]
THR_ALPHA = float(os.environ.get("THR_ALPHA", "0.5"))
THR_HIST_MIN = int(os.environ.get("THR_HIST_MIN", "2"))     # hist 模式至少积累几个月

# LightGBM 超参档位（2026-11-16 auc_probe 实测）：本仓库一直硬编码 num_leaves=63 /
# min_child_samples=100，在 1.05M 行 × 49 列、AUC≈0.53 的极低信噪比下明显过拟合。
# 强正则化实测 H10 AUC +0.0041（9/10 月改善）、H15 +0.0026（8/10）、H5 +0.0014（8/10）。
# 档位表已迁到 `h10_signal.py`（生产入口）作单一真源，这里只导入，避免两处漂移。
LGB_REG = os.environ.get("LGB_REG", "base").strip().lower()

# 量能派生特征开关（2026-11-16 vp_probe 意外发现）
# kl.build 里 `pick(cols, ("volume","um_volume","spot_volume"))` 在原始 K 线上找不到
# volume 列 → 跳过 lg_volz/volratio/volprice/taker*共 7 列，基线一直是 49 列。
# 给 raw 补一列 volume 后变 56 列，在 reg2 基线下 AUC 0.5324→0.5333（+0.0009）。
# 这里做严格对照：配置完全相同，只切这一列。
WITH_VOL = os.environ.get("VOL", "0") == "1"


def log(msg: str) -> None:
    print(f"[{time.time() - T:7.1f}s] {msg}", flush=True)


def fit_w(Xtr, ytr, cols, Xes, yes, wtr, seed):
    """带样本权重的单模型训练；接口对齐 two_legs.fit 的语义（早停看 es AUC）。"""
    params = dict(objective="binary", metric="auc", boosting_type="gbdt",
                  learning_rate=0.05, num_leaves=63, feature_fraction=0.9,
                  bagging_fraction=0.9, bagging_freq=1, min_child_samples=100,
                  verbose=-1, seed=seed, num_threads=os.cpu_count() or 4)
    params.update(LGB_PRESETS.get(LGB_REG, {}))
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


def breakeven(pay: float) -> float:
    """该赔率下的盈亏平衡胜率。"""
    return 1.0 / (1.0 + pay)


def grade(s_abs: np.ndarray, nb: int) -> np.ndarray:
    """把当月命中样本按 |p-0.5| 内部排序切成 nb 档（0=最弱），返回档号。

    档号是「本月信号里相对第几强」，跨月语义可比，不受分数尺度漂移影响。
    注意：用当月命中集合的整体分布定档，属于事后分位；实盘需用滚动分位近似。
    """
    n = len(s_abs)
    if n == 0:
        return np.empty(0, np.int8)
    order = np.argsort(s_abs, kind="stable")
    g = np.empty(n, np.int8)
    g[order] = np.minimum((np.arange(n) * nb) // n, nb - 1)
    return g


def report_diag(monthly, hist, pay: float) -> None:
    """A) 事后分档胜率（诊断排序能力）B) 滚动因果否决 C) 逐月明细 D) 阈值漂移。"""
    if not monthly:
        print("\n[!] export 模式无可用月份")
        return
    be = breakeven(pay)
    nb = HIST_BINS

    # ---------- A. 全样本分档（含未来信息，仅用于判断有没有排序信息）----------
    print(f"\n=== A. 分档胜率（命中样本按信号强度切 {nb} 档，0=最弱）| BE={be * 100:.1f}% ===")
    print(f"{'tpd':>6}{'n':>8}" + "".join(f"{'档' + str(i):>9}" for i in range(nb))
          + f"{'单调':>7}{'首尾差':>8}{'最强档EV':>10}")
    for tpd in TPD_LIST:
        if tpd not in hist or not hist[tpd]:
            continue
        g = np.concatenate([h[1] for h in hist[tpd]])
        ok = np.concatenate([h[2] for h in hist[tpd]])       # 0/1 = 该笔是否猜对
        wrs = []
        for i in range(nb):
            m = g == i
            wrs.append(float(ok[m].mean()) if m.sum() else float("nan"))
        mono = all(wrs[i] <= wrs[i + 1] + 1e-9 for i in range(nb - 1))
        spread = wrs[-1] - wrs[0]
        top_ev = pay * wrs[-1] - (1 - wrs[-1])
        print(f"{tpd:>6.1f}{len(ok):>8d}"
              + "".join(f"{w * 100:>8.2f}%" if w == w else f"{'-':>9}" for w in wrs)
              + f"{'是' if mono else '否':>7}{spread * 100:>+7.2f}pp{top_ev:>+10.3f}")

    # ---------- B. 滚动因果否决（历史各档胜率 → 否决本月同档）----------
    print(f"\n=== B. 滚动因果否决（用 m'<m 的各档胜率定坏档，否决本月同档）===")
    print(f"{'tpd':>6}{'月数':>6}{'未决单/天':>11}{'未决胜率':>11}{'未决EV':>9}"
          f"{'否决后单/天':>13}{'否决后胜率':>13}{'否决后EV':>10}{'删单比例':>10}{'坏档':>6}")
    for tpd in TPD_LIST:
        if tpd not in hist or not hist[tpd]:
            continue
        b_y: list[np.ndarray] = []
        b_n = 0
        k_y: list[np.ndarray] = []
        k_n = 0
        nd_sum = 0
        nmon = 0
        bad_rec: list[str] = []
        for h in hist[tpd]:
            mi = h[0]
            past = [p for p in hist[tpd] if p[0] < mi]
            if not past:
                continue
            pg = np.concatenate([p[1] for p in past])
            py = np.concatenate([p[2] for p in past])
            if len(py) < HIST_MIN:
                continue
            bad = np.zeros(nb, bool)
            for i in range(nb):
                m = pg == i
                if m.sum() >= HIST_MIN_BUCKET:
                    bad[i] = py[m].mean() < be
            g = h[1]
            y = h[2]
            keep = ~bad[g]
            row = monthly[mi]
            nd = row["nd"]
            nmon += 1
            nd_sum += nd
            b_y.append(y)
            b_n += len(y)
            if keep.any():
                k_y.append(y[keep])
                k_n += int(keep.sum())
            bad_rec.append("".join(str(i) for i in np.where(bad)[0]) or "-")
        if nmon == 0 or not b_y:
            print(f"{tpd:>6.1f}{nmon:>6d}{'（历史不足，无法评估）':>30}")
            continue
        by = np.concatenate(b_y)
        bw = float(by.mean())
        kw = float(np.concatenate(k_y).mean()) if k_y else float("nan")
        print(f"{tpd:>6.1f}{nmon:>6d}{b_n / nd_sum:>11.1f}{bw * 100:>10.2f}%"
              f"{pay * bw - (1 - bw):>+9.3f}{k_n / nd_sum:>13.1f}{kw * 100:>12.2f}%"
              f"{pay * kw - (1 - kw):>+10.3f}{(1 - k_n / b_n) * 100:>9.1f}%"
              f"{','.join(sorted(set(bad_rec))):>6}")

    # ---------- C. 主操作点逐月明细 ----------
    if PRIMARY_TPD in hist and hist[PRIMARY_TPD]:
        print(f"\n=== C. tpd={PRIMARY_TPD:g} 逐月分档胜率（各档样本数相等）===")
        print(f"{'月':>7}" + "".join(f"{'档' + str(i):>9}" for i in range(nb))
              + f"{'本月历史坏档':>13}{'删单%':>7}")
        for h in hist[PRIMARY_TPD]:
            mi, g, y = h[0], h[1], h[2]
            past = [p for p in hist[PRIMARY_TPD] if p[0] < mi]
            if past and len(np.concatenate([p[2] for p in past])) >= HIST_MIN:
                pg = np.concatenate([p[1] for p in past])
                py = np.concatenate([p[2] for p in past])
                bad = np.zeros(nb, bool)
                for i in range(nb):
                    m = pg == i
                    if m.sum() >= HIST_MIN_BUCKET:
                        bad[i] = py[m].mean() < be
                tag = "".join(str(i) for i in np.where(bad)[0]) or "-"
                cut = float(bad[g].mean()) * 100
            else:
                tag, cut = "（历史不足）", float("nan")
            cells = ""
            for i in range(nb):
                m = g == i
                cells += f"{y[m].mean() * 100:>8.2f}%" if m.sum() else f"{'-':>9}"
            print(f"{monthly[mi]['ym']:>7}{cells}{tag:>13}{cut:>7.1f}")

    # ---------- D. 阈值漂移 ----------
    print(f"\n=== D. 阈值/单量漂移（阈值由 test 前 {THR_FRAC:.0%} 段估分位）===")
    ths = sorted(monthly[0]["thrs"]) if monthly else []
    print(f"{'月':>7}{'test天':>7}" + "".join(f"{'thr@' + format(t, 'g'):>10}" for t in ths))
    for row in monthly:
        print(f"{row['ym']:>7}{row['nd']:>7d}"
              + "".join(f"{row['thrs'].get(t, float('nan')):>10.4f}" for t in ths))


def thr_report(alt: dict, pay: float, nmon: int) -> None:
    """E) 阈值估计方案对比。同一模型、同一分数，只换阈值的估计来源。

    「月数」列 = 有成交的月 / 总月。覆盖率不足的方案其胜率/EV 不可信（选择偏差）。
    """
    if not alt:
        return
    be = breakeven(pay)
    print(f"\n=== E. 阈值方案对比（同一模型，仅换阈值估计来源）| BE={be * 100:.1f}% ===")
    print(f"{'tpd':>6}{'方案':>7}{'月数':>7}{'单/天':>8}{'n':>7}{'胜率':>9}{'EV':>9}"
          f"{'月std':>9}{'单量CV':>8}{'单量max/min':>12}")
    for tpd in TPD_LIST:
        for mode in alt:
            a = alt[mode].get(tpd)
            if not a:
                continue
            w = np.concatenate(a["w"])
            n = np.array(a["n"], float)
            nds = np.array(a["nd"], float)
            wr = float(w.mean())
            mw = np.array([float(x.mean()) for x in a["w"]])
            mstd = float(mw.std()) * 100
            cv = float(n.std(ddof=1) / n.mean()) if len(n) > 1 else float("nan")
            partial = len(a["mi"]) < nmon
            cov = f"{len(a['mi'])}/{nmon}"
            ratio = f"{n.max():.0f}/{n.min():.0f}"
            print(f"{tpd:>6.1f}{mode:>7}{cov:>7}"
                  f"{n.sum() / nds.sum():>8.1f}{int(n.sum()):>7d}"
                  f"{wr * 100:>8.2f}{'*' if partial else '%':>1}"
                  f"{pay * wr - (1 - wr):>+9.3f}"
                  f"{mstd:>7.2f}pp{cv:>8.2f}{ratio:>12}")
    modes = [m for m in alt if PRIMARY_TPD in alt[m]]
    if modes:
        mi_all = [i for m in modes for i in alt[m][PRIMARY_TPD]["mi"]]
        nmon = max(mi_all) + 1
        print(f"\n[tpd={PRIMARY_TPD:g}] 逐月单量（看抖动是否被压平）：")
        print(f"{'方案':>7}" + "".join(f"{'m' + str(i + 1):>8}" for i in range(nmon)))
        for m in modes:
            d = dict(zip(alt[m][PRIMARY_TPD]["mi"], alt[m][PRIMARY_TPD]["n"]))
            print(f"{m:>7}" + "".join(f"{d.get(i, 0):>8d}" for i in range(nmon)))


def main() -> None:
    raw = kl.load_raw()
    if WITH_VOL and "volume" not in raw.columns:
        raw = raw.with_columns(
            (pl.col("um_buy_vol").cast(pl.Float64)
             + pl.col("um_sell_vol").cast(pl.Float64)).alias("volume"))
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
        f"MEM_MODE={MEM_MODE} MEM_N={MEM_N} MEM_W={MEM_W:g} PER_MONTH={MEM_PER_MONTH} | "
        f"stride={STRIDE}")

    months = [m for m in sorted(set(ym.tolist())) if START_MONTH <= m <= END_MONTH]
    per: dict[float, dict[str, list]] = {t: dict(s=[], y=[], mon=[]) for t in TPD_LIST}
    rows = []
    pool: list[tuple[int, int, float]] = []
    mem_sizes: list[int] = []
    # export 诊断用：monthly[key] = 该月 test 的分数/标签/各 tpd 阈值
    monthly: list[dict] = []
    # hist[tpd] = [(月索引, 命中样本档号, 命中样本是否猜对 0/1), ...]
    hist: dict[float, list[tuple[int, np.ndarray, np.ndarray]]] = {t: [] for t in TPD_LIST}
    # 阈值方案对比（THR_MODES 非空时启用）
    alt: dict[str, dict[float, dict]] = {}
    thr_prev: dict[float, float] = {}
    thr_hist_bars: list[np.ndarray] = []

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

        i_mem, _ = (pick_memory(pool, int(i_tr[0]), MEM_N) if USE_MEM_TRAIN
                    else (np.empty(0, np.int64), np.empty(0, np.float64)))
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
        p_es = np.zeros(len(i_es))
        its = []
        for seed in SEEDS:
            mdl = fit_w(Xtr, y_tr, feats, Xes, yes, w_tr, seed)
            p_th += mdl.predict(Xth, num_iteration=mdl.best_iteration) / len(SEEDS)
            p_te += mdl.predict(Xte, num_iteration=mdl.best_iteration) / len(SEEDS)
            if "wide" in THR_MODES:
                p_es += mdl.predict(Xes, num_iteration=mdl.best_iteration) / len(SEEDS)
            its.append(mdl.best_iteration or ROUNDS)

        auc = float(roc_auc_score(y_all[i_te], p_te))
        days = (ct[i_te] // 86_400_000).astype(np.int64)
        nd = len(np.unique(days))
        s = p_te - .5
        mi = len(monthly)
        monthly.append(dict(ym=m, nd=nd, thrs={}))
        line = (f"[{m}] recent={n_tr:,}+mem={len(i_mem):,}(w{MEM_W:g}) es={len(i_es):,} "
                f"thr={len(i_th):,} test={len(i_te):,}({nd}天) iter={int(np.mean(its))} "
                f"AUC={auc:.4f}")

        # 历史分数池（只用 m'<m，避免前视），供 hist/mix 阈值方案使用
        hist_pool = (np.concatenate(thr_hist_bars)
                     if THR_MODES and len(thr_hist_bars) >= THR_HIST_MIN else None)
        # wide：i_es ∪ i_th 同尺度扩窗（i_es 是早停段，模型对其略乐观，但仅用于估分位）
        wide_pool = (np.concatenate([np.abs(p_th - .5), np.abs(p_es - .5)])
                     if "wide" in THR_MODES else None)

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
            monthly[mi]["thrs"][tpd] = thr
            if COLLECT_HIST:
                # 存「是否猜对」而非标签本身，report 里直接取均值即胜率
                hist[tpd].append((mi, grade(np.abs(s[hit]), HIST_BINS),
                                  ((s[hit] > 0).astype(int) == y_all[i_te][hit]).astype(np.int8)))
            for mode in THR_MODES:
                if mode == "raw":
                    t2 = thr
                elif mode == "ewma":
                    pv = thr_prev.get(tpd)
                    t2 = thr if pv is None else THR_ALPHA * thr + (1 - THR_ALPHA) * pv
                    thr_prev[tpd] = t2
                elif mode in ("hist", "mix"):
                    th = (float(np.quantile(hist_pool, 1 - tpd / BARS_PER_DAY))
                          if hist_pool is not None else thr)
                    t2 = th if mode == "hist" else 0.5 * (thr + th)
                elif mode == "wide":
                    t2 = float(np.quantile(wide_pool, 1 - tpd / BARS_PER_DAY))
                elif mode == "widemi":      # wide 与 raw 的折中，降低对早停段的依赖
                    t2 = 0.5 * (thr + float(np.quantile(wide_pool, 1 - tpd / BARS_PER_DAY)))
                else:
                    continue
                h2 = np.abs(s) >= t2
                if h2.sum() < 30:
                    continue
                a = alt.setdefault(mode, {}).setdefault(
                    tpd, dict(w=[], n=[], nd=[], mi=[], thr=[]))
                a["w"].append((s[h2] > 0).astype(int) == y_all[i_te][h2])
                a["n"].append(int(h2.sum()))
                a["nd"].append(nd)
                a["mi"].append(mi)
                a["thr"].append(t2)
        if THR_MODES:
            thr_hist_bars.append(np.abs(s).astype(np.float32))
            if tpd == PRIMARY_TPD:
                line += f" | 阈值={thr:.4f} {nb / nd:.1f}单/天 胜率={w.mean() * 100:.2f}%"
                # 入库：本月「被选中且预测正确」的样本，按月限量（w 是 hit 子集，需回映射）
                sel = np.where(hit)[0]
                take = sel[w == 1][:MEM_PER_MONTH]
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
    if COLLECT_HIST:
        report_diag(monthly, hist, REPORT_PAY)
    if THR_MODES:
        thr_report(alt, REPORT_PAY, len(monthly))
    log("DONE")


if __name__ == "__main__":
    main()
