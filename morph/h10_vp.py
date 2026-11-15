#!/usr/bin/env python3
"""h=10 因果口径 + VP 结构特征：单变量 A/B（**不修改 `h10_signal.py` 一个字**）。

做法
----
运行时把 `kline_long.build` 包一层：拿到基线特征矩阵后立刻追加 VP 特征，
然后原封不动调用 `h10_signal.main()`。训练、早停、因果阈值、月度汇总全部沿用
已验证的那套，因此结果与基线（AUC 0.5366 / 63.32% @19.9 单/天 / 月min 55.74% /
月std 4.54pp）**严格同口径可比**。

对照组与判据（跑之前先写死，禁止事后挑点）
------------------------------------------
- 对照：`h10sig`（无 VP），主操作点 tpd=8。
- 有信号：tpd=8 胜率 ≥ 64.3%（+1pp）**或** test AUC 提升 ≥ 0.005；
- 无效：≤ 63.3%（即打不过基线），或月度 std / 破 50% 月变差。
- VP 参数只在 val（thr 段）上体现，不在 test 上调。

env
---
  VP_RES=20      桶宽（bp）
  VP_ANCHOR=15   剖面重算间隔（bar），只用已完成块 → 严格因果
  VP_W_SHORT=12 / VP_W_MID=32 / VP_W_LONG=96   3h / 8h / 24h 三档视窗（单位：块）
  VP_R=25        局部剖面半宽（桶数）
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from morph import h10_signal as base  # noqa: E402
from morph.vp_profile import add_vp  # noqa: E402

_orig_build = base.kl.build


def _build_with_vp(raw):
    df, feats = _orig_build(raw)
    n0 = len(feats)
    df, feats = add_vp(df, feats)
    base.log(f"VP 特征追加 {len(feats) - n0} 列 → 特征总数 {len(feats)}")
    return df, feats


def main() -> None:
    """bootstrap 通过 `module.main()` 调入口，所以必须显式定义（v1 就因为这个报 ERROR）。"""
    if base.BAR > 1:
        raise SystemExit("本脚本只支持 BAR=1（BAR>1 走 build_bar，特征口径不同，未接）")
    base.kl.build = _build_with_vp      # 只影响本进程，不落盘、不改文件
    base.main()


if __name__ == "__main__":
    main()
