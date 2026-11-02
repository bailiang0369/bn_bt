import numpy as np
import polars as pl
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, accuracy_score

df = pl.read_parquet("data/BTCUSDT.parquet").sort("close_time")
# 正确(未来)目标: close[T+3] > close[T]
fut = df.select((pl.col("close_time") - 180000).alias("close_time"), pl.col("close").alias("_cf"))
df = df.join(fut, on="close_time", how="left")
df = df.with_columns((pl.col("_cf") > pl.col("close")).cast(pl.Int8).alias("target")).drop_nulls("target").drop("_cf")

drop = {"target", "close_time"}
feats = [c for c in df.columns if c not in drop]
X = df.select(feats).to_numpy().astype(np.float32)
y = df["target"].to_numpy().astype(int)
n = len(y)
i1, i2 = int(n * 0.64), int(n * 0.80)
Xtr, Xva, Xte = X[:i1], X[i1:i2], X[i2:]
ytr, yva, yte = y[:i1], y[i1:i2], y[i2:]

dtr = lgb.Dataset(Xtr, ytr, feature_name=feats)
dva = lgb.Dataset(Xva, yva, reference=dtr)
params = dict(objective="binary", metric="auc", boosting_type="gbdt", learning_rate=0.05,
              num_leaves=63, feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1,
              min_child_samples=100, verbose=-1, seed=42)
model = lgb.train(params, dtr, num_boost_round=1000, valid_sets=[dtr, dva],
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
p = model.predict(Xte, num_iteration=model.best_iteration)
conf = np.abs(p - 0.5)
pred = (p >= 0.5).astype(int)
print(f"test AUC={roc_auc_score(yte, p):.4f}  full ACC={accuracy_score(yte, pred):.4f}")

ct = df["close_time"].to_numpy()
test_days = (ct[i2:].max() - ct[i2:].min()) / 86400000.0
print(f"测试集跨度约 {test_days:.0f} 天, 目标 15单/天 -> 覆盖率 ≈ {15*test_days/len(yte)*100:.2f}%\n")
print(f"{'coverage':>9s} {'交易数':>8s} {'单/天':>7s} {'ACC':>7s}")
for cov in [0.003, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0]:
    k = max(1, int(len(p) * cov))
    idx = np.argsort(-conf)[:k]
    acc = accuracy_score(yte[idx], pred[idx])
    print(f"{cov*100:8.1f}% {k:8d} {k/test_days:7.2f} {acc:7.4f}")
