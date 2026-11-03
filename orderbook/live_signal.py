"""实盘信号过滤器：用最近 W 天已实现结果决定「今天是否下单」。

与 `orderbook/eval_pred.py --rm` 的区别：
  - eval_pred 是**回测**：跑完整个 test 段，出一条滚动准确率曲线；
  - 本模块是**当下这一刻的决策**：读一个不断增长的实盘历史文件，输出今天 TRADE/NO_TRADE。

历史文件契约（每行 = 一根已结算 bar）：
  - close_time : bar 收盘时间（毫秒）
  - target     : 已实现标签（1=涨 / 0=跌），必须已结算（未来 h 根已知）
  - 分数列     : 由 --score 表达式给出，如 z_ob / z_kl / "z_ob+0.2*z_kl"

规则（参数来自 `融合实验记录.md` Run 17 的三段验证）：
  用「过去 45 个完整交易日」的已实现结果算全局 top-K 准确率（K = 日均 15 单 × 天数），
  `< 0.62` → 今天停单；否则照常按日均 15 单下单。严格因果（不含今天，避免偷看）。

用法：
  python -m orderbook.live_signal --history live.parquet --score "z_ob+0.2*z_kl"
  python -m orderbook.live_signal --history live.parquet --append new.parquet --score "..."
  python -m orderbook.live_signal --history live.parquet --score "..." --simulate
  # 作为库
  from orderbook.live_signal import decide
  r = decide(df, score="z_ob+0.2*z_kl")
  if not r["trade"]:
      ... 停单 ...
"""
import sys
import json
import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from orderbook.eval_pred import day_index, topk_acc

# Run 17 多段验证：窗口 45 天、阈值 0.62 是三段唯一都为正的设定
DEFAULTS = dict(per_day=15, window_days=45, threshold=0.62, min_days=15)


def score_of(df, expr):
    """按表达式算打分，可用任意数值列名，如 "z_ob+0.2*z_kl"。"""
    env = {c: df[c].to_numpy() for c in df.columns}
    env["np"] = np
    return np.asarray(eval(expr, {"__builtins__": {}}, env), dtype=np.float64).ravel()


def decide(df, score="z_ob+0.2*z_kl", per_day=15, window_days=45,
           threshold=0.62, min_days=15, drop_last_days=1):
    """返回决策 dict（trade / roll_acc / window / reason ...）。

    drop_last_days=1（默认）：最后一天是「今天」（结果尚未完整），
    窗口只用到它之前的历史，保证因果。
    """
    df = df.sort("close_time")
    ct = df["close_time"].to_numpy()
    y = df["target"].to_numpy().astype(np.int8)
    s = score_of(df, score)
    d = day_index(ct)
    days = np.unique(d)

    n_hist = len(days) - drop_last_days
    hist = days[:n_hist] if n_hist > 0 else days[:0]
    decision_days = [int(x) for x in days[n_hist:]]

    if len(hist) < min_days:
        return dict(trade=True, reason=f"cold_start(历史天={len(hist)}<{min_days})",
                    roll_acc=None, n_days=int(len(hist)), window_days=window_days,
                    threshold=threshold, per_day=per_day, score=score,
                    window=None, decision_days=decision_days)

    win = hist[-window_days:]
    idx = np.where(np.isin(d, win))[0]
    acc = topk_acc(s[idx], y[idx], per_day * len(win))
    return dict(
        trade=bool(acc >= threshold), reason="ok", roll_acc=float(acc),
        n_days=int(len(win)), window_days=window_days, threshold=threshold,
        per_day=per_day, score=score,
        window=[int(win[0]), int(win[-1])], decision_days=decision_days,
    )


def _fmt_day(x):
    return dt.datetime.fromtimestamp(int(x) * 86400, dt.timezone.utc).strftime("%Y-%m-%d")


def _print_decision(r):
    tag = "TRADE" if r["trade"] else "NO_TRADE"
    print(f"\n[decision] {tag}   score={r['score']}  窗口={r['window_days']}天  "
          f"阈值={r['threshold']}")
    if r["roll_acc"] is not None:
        rel = "≥" if r["trade"] else "<"
        print(f"  滚动 top-K 准确率 = {r['roll_acc']:.4f}  （{rel} {r['threshold']}，"
              f"覆盖 {r['n_days']} 天，日均 {r['per_day']} 单）")
        if r["window"]:
            print(f"  窗口：{_fmt_day(r['window'][0])} ~ {_fmt_day(r['window'][1])}")
        dd = r["decision_days"]
        if len(dd) <= 3:
            ds = ", ".join(_fmt_day(x) for x in dd)
        else:
            ds = f"{_fmt_day(dd[0])} ~ {_fmt_day(dd[-1])}（{len(dd)} 天）"
        print(f"  决策日：{ds}")
    else:
        print(f"  原因：{r['reason']}（历史不足，默认放行）")


def _simulate(df, score, per_day, window_days, threshold, min_days):
    """逐日因果模拟：第 t 天只用 t 之前的历史决策，验证规则。"""
    df = df.sort("close_time")
    ct = df["close_time"].to_numpy()
    y = df["target"].to_numpy().astype(np.int8)
    s = score_of(df, score)
    d = day_index(ct)
    days = np.unique(d)
    day_idx = {int(u): np.where(d == u)[0] for u in days}

    dec = {}
    for i in range(len(days)):
        hist = days[max(0, i - window_days):i]
        if len(hist) < min_days:
            dec[int(days[i])] = True
            continue
        idx = np.concatenate([day_idx[int(u)] for u in hist])
        dec[int(days[i])] = bool(topk_acc(s[idx], y[idx], per_day * len(hist)) >= threshold)

    traded = [u for u in days if dec[int(u)]]

    def acc_on(sub):
        if not len(sub):
            return float("nan")
        idx = np.concatenate([day_idx[int(u)] for u in sub])
        return topk_acc(s[idx], y[idx], per_day * len(sub))

    print(f"\n[模拟] 分数={score}  窗口={window_days}天  阈值={threshold}")
    print(f"  基线（全部 {len(days)} 天）全局 topK = {acc_on(days):.4f}")
    print(f"  交易 {len(traded)}/{len(days)} 天（覆盖 {len(traded) / len(days):.2f}）"
          f"  交易段全局 topK = {acc_on(traded):.4f}")
    tail = days[-12:]
    print("  最近决策：" + "  ".join(
        f"{_fmt_day(u)}={'T' if dec[int(u)] else 'N'}" for u in tail))


def main():
    ap = argparse.ArgumentParser(description="实盘信号过滤器（最近 W 天滚动 top-K 决定是否下单）")
    ap.add_argument("--history", required=True, help="实盘历史 parquet（close_time/target/分数列）")
    ap.add_argument("--score", default="z_ob+0.2*z_kl")
    ap.add_argument("--per-day", type=int, default=DEFAULTS["per_day"])
    ap.add_argument("--window-days", type=int, default=DEFAULTS["window_days"])
    ap.add_argument("--threshold", type=float, default=DEFAULTS["threshold"])
    ap.add_argument("--min-days", type=int, default=DEFAULTS["min_days"])
    ap.add_argument("--drop-last-days", type=int, default=1)
    ap.add_argument("--append", default=None, help="追加已结算 parquet 写回 --history 后再决策")
    ap.add_argument("--out", default=None, help="决策写成 JSON")
    ap.add_argument("--simulate", action="store_true", help="逐日模拟验证规则")
    a = ap.parse_args()

    hist_path = Path(a.history)
    df = pl.read_parquet(hist_path) if hist_path.exists() else pl.DataFrame()

    if a.append:
        add = pl.read_parquet(a.append)
        df = pl.concat([df, add], how="diagonal") if df.height else add
        df = df.unique(subset="close_time", keep="last").sort("close_time")
        df.write_parquet(hist_path)
        print(f"[append] 历史更新为 {df.height} 行 → {hist_path}")

    if df.height == 0:
        print("[decision] TRADE（无历史，冷启动）")
        return

    if a.simulate:
        _simulate(df, a.score, a.per_day, a.window_days, a.threshold, a.min_days)
        return

    r = decide(df, a.score, a.per_day, a.window_days, a.threshold, a.min_days, a.drop_last_days)
    _print_decision(r)
    if a.out:
        Path(a.out).write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[out] {a.out}")


if __name__ == "__main__":
    main()
