"""按 feature_spec.md 底座收拾最终宽表，直接覆盖 BTCUSDT.parquet。

只做两件事：
1) 直接删除冗余的订单簿逐档分箱（b0..b4 的 q_/n_/vwap_ 拆箱列），保留聚合类 ob_*/atk_*；
2) 额外补两个指标（按用户要求）：
   - stoch(60,2,2)（慢速随机：fastK=60, slowK=2, slowD=2），只用 OHLC；
   - supertrend(10, 3.0)：输出 supertrend 线与 supertrend_dir(±1)。

注意：feature_spec.md 明确 OHLC 衍生特征不在本仓库做，这里仅按用户要求额外加 stoch 与 supertrend。
pl_exp 是公共工具，放在 tool/ 下，本脚本通过 sys.path 引入。
脚本对"已处理过"的 parquet 幂等（列已存在则跳过），可重复运行不出错。
"""

import re
import sys
from pathlib import Path

import polars as pl

BASE = Path(__file__).resolve().parent.parent   # 仓库根（bn_bt/，因本文件已移入 orderbook_kaggle/）
sys.path.insert(0, str(BASE / "tool"))          # 引入 tool/pl_exp
from pl_exp import PolarsExp                     # noqa: E402

SRC = BASE / "data" / "BTCUSDT.parquet"


def main():
    df = pl.read_parquet(SRC)
    n0, c0 = df.height, df.width

    # 1) 直接删除冗余订单簿逐档分箱（b0..b4 的 q_/n_/vwap_ 拆箱）
    bin_re = re.compile(r"_(ask|bid)_b[0-4]")
    drop_cols = [c for c in df.columns if bin_re.search(c)]
    if drop_cols:
        df = df.drop(drop_cols)
        print(f"[drop] 删除冗余订单簿分箱列 {len(drop_cols)} 个（保留聚合类 ob_*/atk_*）")
    else:
        print("[drop] 未发现冗余分箱列（已处理过），跳过")

    pe = PolarsExp()
    # 2a) stoch(60,2,2)
    if "stoch_k_60_2" not in df.columns:
        stoch_k, stoch_d = pe.stoch("high", "low", "close", 60, 2, 2, alias_suffix=True)
        df = df.with_columns([stoch_k, stoch_d])
        print("[add] stoch(60,2,2) 已添加")
    else:
        print("[add] stoch 已存在，跳过")
    # 2b) supertrend(10, 3.0)
    if "supertrend" not in df.columns:
        df = pe.supertrend(df, "high", "low", "close", 10, 3.0)
        print("[add] supertrend(10,3.0) 已添加")
    else:
        print("[add] supertrend 已存在，跳过")

    df.write_parquet(SRC)  # 直接覆盖原始文件
    print(f"[done] 输入 {n0} 行 × {c0} 列；输出 {df.height} 行 × {df.width} 列 -> {SRC}")


if __name__ == "__main__":
    main()
