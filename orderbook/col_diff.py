"""对比 ETH / BTC 两份 merged 特征的列结构（只读 parquet schema，秒级）。"""
import glob

import polars as pl


def find(sym: str) -> str:
    hits = glob.glob(f"/kaggle/input/**/{sym}.parquet", recursive=True)
    if not hits:
        raise FileNotFoundError(sym)
    return hits[0]


def bucket(c: str) -> str:
    for pre in ("ob_f_", "ob_s_", "ob_b_", "atk_f_", "atk_s_", "amp_"):
        if c.startswith(pre):
            return pre
    return "other"


eth = list(pl.read_parquet_schema(find("ETHUSDT")).keys())
btc = list(pl.read_parquet_schema(find("BTCUSDT")).keys())

print(f"[count] ETH={len(eth)}  BTC={len(btc)}  diff={len(eth) - len(btc)}")
for name, cols in (("ETH", eth), ("BTC", btc)):
    agg: dict[str, int] = {}
    for c in cols:
        agg[bucket(c)] = agg.get(bucket(c), 0) + 1
    print(f"[{name}] {agg}")

only_eth = [c for c in eth if c not in btc]
only_btc = [c for c in btc if c not in eth]
print(f"\n[仅 ETH 有] {len(only_eth)} 列:")
for c in only_eth:
    print("  +", c)
print(f"\n[仅 BTC 有] {len(only_btc)} 列:")
for c in only_btc:
    print("  -", c)

print("\n[ETH other 明细]")
print([c for c in eth if bucket(c) == "other"])
print("\n[BTC other 明细]")
print([c for c in btc if bucket(c) == "other"])
