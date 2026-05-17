"""Small test: load 5 days of data and compute features for 10 stocks."""
import sys
sys.path.insert(0, r"D:\study\AI3003-proj")
import pandas as pd
import numpy as np

# Override config for test
import config
config.TRAIN_START = "2026-05-08"
config.TEST_END = "2026-05-15"

from data_loader import load_basic_info, load_trade_calendar, load_daily_data, load_metric_data, load_moneyflow_data, load_st_stocks, build_unified_dataframe
from features import compute_stock_features, add_cross_sectional_features, get_feature_columns

print("=== Quick Pipeline Test ===\n")

# Load basic info
basic_df = load_basic_info()
print(f"1. Basic stocks: {len(basic_df)}")

# Get dates
all_dates = load_trade_calendar()
ts = pd.Timestamp("2026-05-08")
te = pd.Timestamp("2026-05-15")
date_list = [d for d in all_dates if ts <= pd.Timestamp(d) <= te]
print(f"2. Test dates: {len(date_list)} -> {date_list}")

# Load daily data
valid_codes = set(basic_df["ts_code"].tolist())
all_frames = []
for d in date_list:
    import os
    fpath = os.path.join(config.DAILY_DIR, f"{d}.csv")
    if os.path.exists(fpath):
        day_df = pd.read_csv(fpath, dtype={"ts_code": str})
        day_df = day_df[day_df["ts_code"].isin(valid_codes)]
        all_frames.append(day_df)

daily_df = pd.concat(all_frames, ignore_index=True)
daily_df["trade_date"] = pd.to_datetime(daily_df["trade_date"], format="%Y%m%d")
print(f"3. Daily rows: {len(daily_df)}, stocks: {daily_df['ts_code'].nunique()}")

# Test feature computation on first 10 stocks
test_codes = daily_df["ts_code"].unique()[:10]
df_small = daily_df[daily_df["ts_code"].isin(test_codes)].copy()
print(f"4. Test on {len(test_codes)} stocks, {len(df_small)} rows")

# Compute features per stock
results = []
for code in test_codes:
    group = df_small[df_small["ts_code"] == code]
    feat = compute_stock_features(group)
    results.append(feat)

df_feat = pd.concat(results, ignore_index=True)
df_feat = add_cross_sectional_features(df_feat)
df_feat = df_feat.fillna(0.0)
feat_cols = get_feature_columns(df_feat)
print(f"5. Feature shape: {df_feat.shape}, features: {len(feat_cols)}")
print(f"   Feature columns: {feat_cols}")

# Test dataset
from dataset import StockDataset
ds = StockDataset(df_feat, feat_cols)
print(f"6. Dataset samples: {len(ds)}")

if len(ds) > 0:
    x, y = ds[0]
    print(f"7. Sample X shape: {x.shape}, y: {y.item():.6f}")
    print("   X stats: mean={:.4f}, std={:.4f}".format(x.mean().item(), x.std().item()))

print("\n=== All tests passed! ===")
