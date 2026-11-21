#!/usr/bin/env python3
"""新阶段：h10 逐 bar 实时模拟（独立目录，不改动任何旧阶段文件）。

本文件在 Kaggle 上一次完成两件事：
  1. 按现役 h10 walk-forward 口径（train/es/thr/test，5 seeds，因果阈值）训练并落
     逐行分数 / 模型 / 阈值，作为「批量回测参考」；
  2. 用同一份模型与阈值按时间顺序逐 bar 重放，模拟延迟、去重、并发槽位与结算，
     输出实际可下单口径的胜率 / EV / 日单量分布；并做未来数据扰动因果审计。

环境（与现役 h10 默认值一致，可用 env 覆盖）：
  HORIZON=10 ROUNDS=2000 SEEDS=42,43,44,45,46 WIN_DAYS=1460
  ES_FRAC=0.05 THR_FRAC=0.10 TPD=8 BARS_PER_DAY=1440 PAYOUT=0.80
  START_MONTH=2026-01 END_MONTH=2026-10
  OFFSET_MIN=0 SLOTS=3 DEDUP=none COOLDOWN_MIN=10 PERTURB_N=12
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from morph import kline_long as kl  # noqa: E402
from morph.two_legs import add_target, fit  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

HORIZON = int(os.environ.get("HORIZON", "10"))
ROUNDS = int(os.environ.get("ROUNDS", "2000"))
SEEDS = [int(s) for s in os.environ.get("SEEDS", "42,43,44,45,46").split(",") if s.strip()]
WIN_DAYS = int(os.environ.get("WIN_DAYS", "1460"))
ES_FRAC = float(os.environ.get("ES_FRAC", "0.05"))
THR_FRAC = float(os.environ.get("THR_FRAC", "0.10"))
TPD = float(os.environ.get("TPD", "8"))
BARS_PER_DAY = int(os.environ.get("BARS_PER_DAY", "1440"))
PAYOUT = float(os.environ.get("PAYOUT", "0.80"))
START_MONTH = os.environ.get("START_MONTH", "2026-01")
END_MONTH = os.environ.get("END_MONTH", "2026-10")
OFFSET_MIN = int(os.environ.get("OFFSET_MIN", "0"))
SLOTS = int(os.environ.get("SLOTS", "3"))
DEDUP = os.environ.get("DEDUP", "none")
COOLDOWN_MIN = int(os.environ.get("COOLDOWN_MIN", "10"))
PERTURB_N = int(os.environ.get("PERTURB_N", "12"))
SYMBOL = os.environ.get("SYMBOL", "BTCUSDT")
KLINE_FROM = int(os.environ.get("KLINE_FROM", "2020"))
WORK = Path(os.environ.get("LIVE_SIM_OUT", "/kaggle/working/live_sim"))
MS_MIN, MS_DAY = 60_000, 86_400_000
WARMUP_DAYS, FUTURE_BARS = 3, 30
T0 = time.time()

kl.SYMBOL, kl.KLINE_FROM = SYMBOL, KLINE_FROM
import morph.two_legs as _tl  # noqa: E402
_tl.ROUNDS = ROUNDS


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def matrix(df: pl.DataFrame, cols: list[str]) -> np.ndarray:
    return df.select([pl.col(c).cast(pl.Float32) for c in cols]).to_numpy()


def add_flow(df: pl.DataFrame, feats: list[str]) -> tuple[pl.DataFrame, list[str]]:
    c = df.columns
    need = ("um_buy_vol", "um_sell_vol", "spot_buy_vol", "spot_sell_vol")
    if not all(x in c for x in need):
        return df, feats
    um = ((pl.col("um_buy_vol") - pl.col("um_sell_vol"))
          / (pl.col("um_buy_vol") + pl.col("um_sell_vol") + 1e-9))
    sp = ((pl.col("spot_buy_vol") - pl.col("spot_sell_vol"))
          / (pl.col("spot_buy_vol") + pl.col("spot_sell_vol") + 1e-9))
    e = [um.alias("lg_flow_um"), sp.alias("lg_flow_spot"), (um - sp).alias("lg_flow_div")]
    names = ["lg_flow_um", "lg_flow_spot", "lg_flow_div"]
    for n in (15, 60):
        e += [um.rolling_mean(n, min_samples=max(2, n // 2)).alias(f"lg_flow_um{n}"),
              sp.rolling_mean(n, min_samples=max(2, n // 2)).alias(f"lg_flow_spot{n}"),
              (um - sp).rolling_mean(n, min_samples=max(2, n // 2)).alias(f"lg_flow_div{n}")]
        names += [f"lg_flow_um{n}", f"lg_flow_spot{n}", f"lg_flow_div{n}"]
    return df.with_columns(e), feats + names


def features(raw: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    df, feats = kl.build(raw)
    return add_flow(df, feats)


def roll_worst(w: np.ndarray, n: int = 100) -> tuple[float, float]:
    if len(w) < n:
        return float("nan"), float("nan")
    cs = np.concatenate([[0], np.cumsum(w)])
    r = (cs[n:] - cs[:-n]) / n
    return float(r.min()), float((r < .5).mean())


def build_reference(raw_feat: pl.DataFrame) -> list[dict]:
    out = []
    months = [m for m in sorted(raw_feat["ym"].unique().to_list())
              if START_MONTH <= m <= END_MONTH]
    log(f"walk-forward {len(months)} 月（{months[0]}~{months[-1]}）| H={HORIZON} | "
        f"窗={WIN_DAYS}天 seeds={len(SEEDS)} rounds={ROUNDS}")
    for m in months:
        ms = int(dt.datetime.strptime(m + "-01", "%Y-%m-%d")
                 .replace(tzinfo=dt.timezone.utc).timestamp() * 1000)
        lo = ms - WIN_DAYS * MS_DAY
        tr_all = raw_feat.filter((pl.col("close_time") < ms) & (pl.col("close_time") >= lo))
        te = raw_feat.filter(pl.col("ym") == m)
        if tr_all.height < 100_000 or te.height < 1_000:
            log(f"[{m}] 样本不足，跳过")
            continue
        c1 = int(tr_all.height * (1 - ES_FRAC - THR_FRAC))
        c2 = int(tr_all.height * (1 - THR_FRAC))
        tr, es, th = tr_all[:c1], tr_all[c1:c2], tr_all[c2:]
        feats = [x for x in tr_all.columns if x.startswith("lg_")]
        xtr, ytr = matrix(tr, feats), tr["target"].to_numpy().astype(int)
        xes, yes = matrix(es, feats), es["target"].to_numpy().astype(int)
        xth, xte = matrix(th, feats), matrix(te, feats)
        pth = np.zeros(th.height)
        pte = np.zeros(te.height)
        boosters = []
        for seed in SEEDS:
            mdl = fit(xtr, ytr, feats, xes, yes, seed=seed)
            boosters.append(mdl)
            pth += mdl.predict(xth, num_iteration=mdl.best_iteration) / len(SEEDS)
            pte += mdl.predict(xte, num_iteration=mdl.best_iteration) / len(SEEDS)
        thr = float(np.quantile(np.abs(pth - .5), 1 - TPD / BARS_PER_DAY))
        y = te["target"].to_numpy().astype(np.int8)
        hit = np.abs(pte - .5) >= thr
        win = float(((pte[hit] - .5 > 0).astype(np.int8) == y[hit]).mean())
        meta = dict(ym=m, horizon=HORIZON, tpd=TPD, payout=PAYOUT, thr=thr,
                    offset_min=OFFSET_MIN, slots=SLOTS, dedup=DEDUP,
                    cooldown_min=COOLDOWN_MIN, seeds=SEEDS, feats=feats,
                    train_max_ms=int(tr_all["close_time"].max()), month_start_ms=ms,
                    test_first_ms=int(te["close_time"][0]),
                    test_last_ms=int(te["close_time"][-1]),
                    n_test=te.height, n_hit=int(hit.sum()), batch_win=win)
        out.append(dict(meta=meta, ct=te["close_time"].to_numpy(), p=pte, y=y,
                        thr=thr, boosters=boosters, feats=feats))
        log(f"[{m}] 参考 batch {int(hit.sum())}单/{win * 100:.2f}% thr={thr:.5f}")
    return out


def simulate(item: dict) -> dict:
    ct, p, y, thr = item["ct"], item["p"], item["y"], item["thr"]
    s = p - .5
    hit = np.abs(s) >= thr
    raw_win = ((s[hit] > 0).astype(np.int8) == y[hit]).astype(np.int8)
    cooldown = COOLDOWN_MIN * MS_MIN
    opens: list[int] = []
    last: dict[int, int] = {}
    last_any = -np.inf
    events, wins = [], []
    reasons = dict(no_slot=0, cooldown=0)
    for i in np.flatnonzero(hit):
        dec = int(ct[i]) + OFFSET_MIN * MS_MIN
        opens = [x for x in opens if x > dec]
        d = 1 if s[i] > 0 else 0
        why = ""
        if DEDUP == "any_dir" and dec - last_any < cooldown:
            why = "cooldown"
        elif DEDUP == "same_dir" and dec - last.get(d, -np.inf) < cooldown:
            why = "cooldown"
        if not why and len(opens) >= SLOTS:
            why = "no_slot"
        settle = dec + HORIZON * MS_MIN
        if why:
            reasons[why] += 1
            accepted = False
        else:
            accepted = True
            opens.append(settle)
            last[d], last_any = dec, dec
            wins.append(int(d == y[i]))
        events.append(dict(close_time=int(ct[i]), decision_time=dec, settle_time=settle,
                           direction=d, target=int(y[i]), p=float(p[i]),
                           accepted=accepted, reason=why))
    w = np.array(wins, dtype=np.int8)
    win = float(w.mean()) if len(w) else float("nan")
    rmin, rbad = roll_worst(w)
    days = (pl.Series([e["decision_time"] for e in events if e["accepted"]], dtype=pl.Int64)
            .cast(pl.Datetime("ms")).dt.strftime("%Y-%m-%d").to_numpy()) if len(w) else np.array([])
    cnt = np.unique(days, return_counts=True)[1] if len(days) else np.array([])
    return dict(events=events, wins=wins, raw_n=int(hit.sum()),
                raw_win=float(raw_win.mean()), n=len(w), win=win,
                ev=PAYOUT * win - (1 - win) if len(w) else float("nan"),
                roll_min=rmin, roll_bad=rbad, reasons=reasons,
                day_mean=float(cnt.mean()) if len(cnt) else 0.,
                day_std=float(cnt.std()) if len(cnt) else 0.,
                day_min=int(cnt.min()) if len(cnt) else 0,
                day_max=int(cnt.max()) if len(cnt) else 0)


def check_causal(raw: pl.DataFrame, item: dict) -> list[dict]:
    rng = np.random.default_rng(20261121)
    ct_all = raw["close_time"].to_numpy()
    m = item["meta"]
    picks = np.linspace(m["test_first_ms"], m["test_last_ms"], PERTURB_N + 2)[1:-1]
    cols = ("open", "high", "low", "close", "volume", "um_volume", "um_buy_vol",
            "um_sell_vol", "spot_volume", "spot_buy_vol", "spot_sell_vol")
    res = []
    for t in picks:
        t = int(ct_all[np.searchsorted(ct_all, t)])
        a, feats = features(raw.filter((pl.col("close_time") >= t - WARMUP_DAYS * MS_DAY)
                                       & (pl.col("close_time") <= t + FUTURE_BARS * MS_MIN)))
        idx = int(np.searchsorted(a["close_time"].to_numpy(), t))
        mods = []
        for c in cols:
            if c not in a.columns:
                continue
            arr = a[c].to_numpy().copy()
            arr[idx + 1:] *= rng.uniform(.5, 1.8, len(arr) - idx - 1)
            mods.append(pl.Series(c, arr))
        b = add_flow(a.with_columns(mods), feats)[0]
        xa = matrix(a.slice(idx, 1), feats)[0]
        xb = matrix(b.slice(idx, 1), feats)[0]
        p0 = sum(x.predict(matrix(a.slice(idx, 1), feats))[0]
                 for x in item["boosters"]) / len(item["boosters"])
        p1 = sum(x.predict(matrix(b.slice(idx, 1), feats))[0]
                 for x in item["boosters"]) / len(item["boosters"])
        res.append(dict(close_time=t, feat_diff=float(np.nanmax(np.abs(xa - xb))),
                        pred_diff=float(abs(p0 - p1))))
    return res


def main() -> None:
    os.makedirs(WORK / "models", exist_ok=True)
    raw = kl.load_raw()
    log("原始 K 线加载完成，开始构造特征")
    raw_feat = features(raw)
    raw_feat = add_target(raw_feat, HORIZON).filter(
        pl.col("target").is_not_null() & pl.col("close").is_not_null()).sort("close_time")
    raw_feat = raw_feat.with_columns(pl.col("close_time").cast(pl.Datetime("ms"))
                                     .dt.strftime("%Y-%m").alias("ym"))
    refs = build_reference(raw_feat)
    if not refs:
        raise SystemExit("无可用月份")
    events, wins, months, causal = [], [], [], []
    for item in refs:
        m = item["meta"]["ym"]
        sim = simulate(item)
        months.append(dict(ym=m, batch_n=sim["raw_n"], batch_win=sim["raw_win"],
                           replay_n=sim["n"], replay_win=sim["win"], ev=sim["ev"],
                           day_mean=sim["day_mean"], day_std=sim["day_std"],
                           day_min=sim["day_min"], day_max=sim["day_max"],
                           reject=sim["reasons"], roll_min=sim["roll_min"]))
        events.extend(dict(ym=m, **e) for e in sim["events"])
        wins.extend(sim["wins"])
        log(f"[{m}] batch {sim['raw_n']}单/{sim['raw_win'] * 100:.2f}% → "
            f"replay {sim['n']}单/{sim['win'] * 100:.2f}% EV={sim['ev']:+.3f} "
            f"日单量={sim['day_mean']:.1f}±{sim['day_std']:.1f} 拒单={sim['reasons']}")
        if PERTURB_N:
            cs = check_causal(raw, item)
            causal.extend(dict(ym=m, **c) for c in cs)
            bad = [c for c in cs if c["feat_diff"] > 1e-9 or c["pred_diff"] > 1e-9]
            log(f"[{m}] 未来扰动 {len(cs) - len(bad)}/{len(cs)} 通过"
                f"（max_feat_diff={max((c['feat_diff'] for c in cs), default=0):.2e}）")
        for seed, mdl in zip(item["meta"]["seeds"], item["boosters"]):
            (WORK / "models" / f"{m}_s{seed}.txt").write_text(
                mdl.model_to_string(num_iteration=mdl.best_iteration), encoding="utf-8")
        pl.DataFrame({"close_time": item["ct"], "target": item["y"], "p": item["p"]}) \
            .write_parquet(WORK / f"pred_{m}.parquet")
        (WORK / f"meta_{m}.json").write_text(
            json.dumps(item["meta"], ensure_ascii=False, indent=2), encoding="utf-8")
    w = np.array(wins, dtype=np.int8)
    win = float(w.mean()) if len(w) else float("nan")
    report = dict(
        config=dict(horizon=HORIZON, tpd=TPD, payout=PAYOUT, offset_min=OFFSET_MIN,
                    slots=SLOTS, dedup=DEDUP, cooldown_min=COOLDOWN_MIN,
                    perturb_n=PERTURB_N, seeds=SEEDS, rounds=ROUNDS, win_days=WIN_DAYS),
        months=months,
        total=dict(batch_n=sum(x["batch_n"] for x in months), replay_n=len(w), win=win,
                   ev=PAYOUT * win - (1 - win) if len(w) else None,
                   month_min=min(x["replay_win"] for x in months),
                   roll_min=float(roll_worst(w)[0]) if len(w) else None),
        causal=causal,
    )
    (WORK / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    pl.DataFrame(events).write_parquet(WORK / "events.parquet")
    print("\n" + "=" * 72)
    print(f"[live_sim] config={report['config']}")
    print(f"[live_sim] 批量逐行命中 {report['total']['batch_n']} 单 → "
          f"逐 bar 可下 {report['total']['replay_n']} 单，胜率 {win * 100:.2f}%，"
          f"EV={report['total']['ev']:+.3f}，最差月 {report['total']['month_min'] * 100:.2f}%")
    print(f"[live_sim] 产物目录 {WORK}")
    print("=" * 72)


if __name__ == "__main__":
    main()
