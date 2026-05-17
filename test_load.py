"""Quick test of data loading pipeline."""
import sys
sys.path.insert(0, r"D:\study\AI3003-proj")
from data_loader import load_basic_info, load_trade_calendar
import pandas as pd

b = load_basic_info()
print(f"Stocks after market filter: {len(b)}")

dates = load_trade_calendar()
ts = pd.Timestamp("2019-01-01")
te = pd.Timestamp("2026-05-17")
filtered = [d for d in dates if ts <= pd.Timestamp(d) <= te]
print(f"Filtered trading days: {len(filtered)}")
print(f"Range: {filtered[0]} ~ {filtered[-1]}")
