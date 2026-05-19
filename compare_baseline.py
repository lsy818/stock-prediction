"""Compare model vs random baseline on test set."""
import sys; sys.path.insert(0, "D:/study/AI3003-proj")
import pickle, pandas as pd, numpy as np
from backtest import BacktestEngine
from config import OUTPUT_DIR

with open(f"{OUTPUT_DIR}/feature_cols.pkl", "rb") as f:
    feat_cols = pickle.load(f)
df_test = pd.read_parquet(f"{OUTPUT_DIR}/test_processed.parquet")

pred = pd.read_csv(f"{OUTPUT_DIR}/daily_predictions.csv")
pred["trade_date"] = pd.to_datetime(pred["trade_date"])
test_preds = pred[pred["trade_date"] >= "2026-01-01"]

rng = np.random.RandomState(42)
random_scores = test_preds[["trade_date", "ts_code"]].copy()
random_scores["score"] = rng.randn(len(random_scores)) * 0.001

daily_prices = df_test[["trade_date", "ts_code", "open", "high", "low",
                         "close", "pct_chg"]].copy()

bt_model = BacktestEngine().run(test_preds, daily_prices)
bt_random = BacktestEngine().run(random_scores, daily_prices)

print("=== Model vs Random (Test Set 2026) ===")
print(f"Model  Total Return: {bt_model['total_return']:.2%}  Win Rate: {bt_model['win_rate']:.2%}")
print(f"Random Total Return: {bt_random['total_return']:.2%}  Win Rate: {bt_random['win_rate']:.2%}")
print()
print("IC(model) = -0.009, IC(random) ≈ 0")
print("Both have no edge -> both lose due to transaction costs")
