"""BTCUSDT 未来 3 分钟涨跌方向预测 —— LightGBM 训练与评估。

核心约束（时序数据）：
- 绝不随机划分：按时间顺序 训练 64% / 验证 16% / 测试 20%。
- 防数据泄露：特征全部来自 t 时刻及之前；target 用 t+3 收盘价，仅作标签，
  通过 close_time+3min 时间对齐得到，绝不把未来信息喂进特征。
- 本地可直接跑；在 Kaggle Notebook 中自动识别 /kaggle/input 挂载路径。

特征来源：已处理好的 BTCUSDT.parquet（feature_spec.md 底座 + stoch(60,2,2) + supertrend(10,3)），
本脚本不额外造特征，避免引入无关指标/泄漏。
"""

import os
import numpy as np
import polars as pl
import lightgbm as lgb
from sklearn.metrics import (
    roc_auc_score, accuracy_score, confusion_matrix, classification_report,
)

KAGGLE_DATASET_SLUG = "btcusdt-orderbook-features"
KAGGLE_PATH = f"/kaggle/input/{KAGGLE_DATASET_SLUG}/BTCUSDT.parquet"


def load_data():
    import glob
    local_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "BTCUSDT.parquet")
    hits = []
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        # Kaggle 现挂载在 /kaggle/input/datasets/<owner>/<slug>/...，用 glob 兜住不同布局
        hits = sorted(glob.glob("/kaggle/input/**/BTCUSDT.parquet", recursive=True))
    if hits:
        path = hits[0]
    elif os.path.exists(local_path):
        path = local_path
    else:
        raise FileNotFoundError("未找到 BTCUSDT.parquet（/kaggle/input 与本地 data/ 均无）")
    print(f"[load] 读取 {path}")
    return pl.read_parquet(path)


def build_xy(df: pl.DataFrame, horizon_min: int = 3):
    df = df.sort("close_time")
    # target：未来 horizon_min 分钟（close_time + horizon_min*60_000ms 那一行的收盘价）> 当前收盘价 记为 1，否则 0。
    # 关键：连接键必须用 (close_time - horizon_min*60_000)，这样当前行 T 才能取到 T+horizon 的 close；
    # 写成 + 会变成“过去方向”，特征里全是价格/收益率，会被完美拟合 → 虚假 AUC。
    fut = df.select(
        (pl.col("close_time") - horizon_min * 60_000).alias("close_time"),
        pl.col("close").alias("_cf"),
    )
    df = df.join(fut, on="close_time", how="left")
    df = df.with_columns((pl.col("_cf") > pl.col("close")).cast(pl.Int8).alias("target"))
    # 最终统一去 null：必须在【全部特征工程 + target 构造之后、训练之前】一次性做。
    # 不能提前做——滚动/shift/未来 target 连接都依赖连续行索引，提前删行会导致后续
    # 补充特征时引用过去/未来行错位、算错。这里一次性清掉前导滚动特征 NaN 与尾部 target NaN。
    df = df.drop_nulls().drop("_cf")

    drop_cols = {"target", "close_time"}
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df.select(feature_cols).to_numpy().astype(np.float32)
    y = df["target"].to_numpy().astype(int)
    return X, y, feature_cols


def time_split(n, train=0.64, val=0.16):
    i1 = int(n * train)
    i2 = int(n * (train + val))
    return (0, i1), (i1, i2), (i2, n)


def main(horizon_min: int = 3):
    df = load_data()
    X, y, feature_cols = build_xy(df, horizon_min)
    n = len(y)
    (tr0, tr1), (va0, va1), (te0, te1) = time_split(n)
    print(f"[config] horizon={horizon_min}min  (target = t+{horizon_min}min 收盘方向)")
    print(f"[split] 时序切分  训练[{tr0}:{tr1}]  验证[{va0}:{va1}]  测试[{te0}:{te1}]")
    print(f"        正类占比  训练={y[tr0:tr1].mean():.3f}  验证={y[va0:va1].mean():.3f}  测试={y[te0:te1].mean():.3f}")

    train_set = lgb.Dataset(X[tr0:tr1], y[tr0:tr1], feature_name=feature_cols)
    val_set = lgb.Dataset(X[va0:va1], y[va0:va1], reference=train_set)

    params = dict(
        objective="binary", metric="auc", boosting_type="gbdt",
        learning_rate=0.05, num_leaves=63, feature_fraction=0.9,
        bagging_fraction=0.9, bagging_freq=1, min_child_samples=100,
        verbose=-1, seed=42,
        # 不指定 device -> 默认 CPU 训练；即便在 Kaggle GPU 笔记本里也不会强制用 GPU
    )
    model = lgb.train(
        params, train_set, num_boost_round=1000,
        valid_sets=[train_set, val_set],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(50)],
    )

    best = model.best_iteration
    print(f"\n[best_iteration] {best}")
    for name, sl, el in [("训练", tr0, tr1), ("验证", va0, va1), ("测试", te0, te1)]:
        p = model.predict(X[sl:el], num_iteration=best)
        pred = (p >= 0.5).astype(int)
        print(f"[{name}] AUC={roc_auc_score(y[sl:el], p):.4f}  ACC={accuracy_score(y[sl:el], pred):.4f}")

    pte = model.predict(X[te0:te1], num_iteration=best)
    pred_te = (pte >= 0.5).astype(int)
    print("\n[测试集 混淆矩阵]")
    print(confusion_matrix(y[te0:te1], pred_te))
    print("\n[测试集 分类报告]")
    print(classification_report(y[te0:te1], pred_te, digits=4))

    # 诚实选择性准确率：阈值仅在验证集定（瞄准最自信 1%），原样套测试集，不在测试上挑运营点。
    days_all = (df["close_time"].to_numpy() // 86_400_000)
    te_days = len(np.unique(days_all[te0:te1]))
    pva = model.predict(X[va0:va1], num_iteration=best)
    pte_full = model.predict(X[te0:te1], num_iteration=best)
    conf_va, conf_te = np.abs(pva - 0.5), np.abs(pte_full - 0.5)
    T = np.quantile(conf_va, 0.99)
    sel = conf_te >= T
    if sel.sum():
        acc_s = accuracy_score(y[te0:te1][sel], (pte_full[sel] >= 0.5).astype(int))
        print(f"\n[诚实选择性 @val1%] 覆盖={sel.mean()*100:.2f}%  单/天={sel.sum()/te_days:.1f}  选择性ACC={acc_s:.4f}")

    imp = np.array(model.feature_importance(importance_type="gain"))
    order = np.argsort(imp)[::-1]
    print("\n[Top15 特征重要性(gain)]")
    for i in order[:15]:
        print(f"  {feature_cols[i]:30s} {imp[i]:.1f}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="BTC 未来 N 分钟涨跌方向 LightGBM 训练")
    ap.add_argument("--horizon", type=int, default=3, help="预测 horizon（分钟），默认 3")
    args = ap.parse_args()
    main(horizon_min=args.horizon)
