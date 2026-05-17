"""
Data loading and preprocessing.
Reads daily CSVs, builds unified multi-stock time series.
"""
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from config import (
    DAILY_DIR, METRIC_DIR, MONEYFLOW_DIR, STOCK_ST_DIR,
    BASIC_PATH, TRADE_CAL_PATH,
    TRAIN_START, TEST_END,
    EXCLUDE_MARKETS, EXCLUDE_ST, MIN_LIST_DAYS,
    OUTPUT_DIR,
)


def load_basic_info():
    """Load stock basic info and filter eligible stocks."""
    df = pd.read_csv(BASIC_PATH, dtype={"ts_code": str})
    # Filter out excluded markets
    df = df[~df["market"].isin(EXCLUDE_MARKETS)]
    df["list_date"] = pd.to_datetime(df["list_date"], format="%Y%m%d", errors="coerce")
    return df


def load_trade_calendar():
    """Load trading calendar and return sorted list of trading dates."""
    df = pd.read_csv(TRADE_CAL_PATH)
    df = df[df["is_open"] == 1]
    dates = sorted(df["cal_date"].astype(str).unique())
    return dates


def load_daily_data(date_list, basic_df):
    """Load daily OHLCV data, keeping only essential columns with float32."""
    valid_codes = set(basic_df["ts_code"].tolist())
    usecols = ["ts_code", "trade_date", "open", "high", "low",
               "close", "pre_close", "pct_chg", "vol", "amount", "vwap"]

    all_frames = []
    for d in tqdm(date_list, desc="Loading daily data"):
        fpath = os.path.join(DAILY_DIR, f"{d}.csv")
        if not os.path.exists(fpath):
            continue
        try:
            day_df = pd.read_csv(fpath, dtype={"ts_code": str, "trade_date": str},
                                 usecols=[c for c in usecols if c in usecols])
            day_df = day_df[day_df["ts_code"].isin(valid_codes)]
            day_df = day_df[(day_df[["open", "high", "low", "close"]] > 0).all(axis=1)]
            if len(day_df) > 0:
                all_frames.append(day_df)
        except Exception:
            continue

    if not all_frames:
        return pd.DataFrame()

    df = pd.concat(all_frames, ignore_index=True)
    df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
    # Downcast to float32 to save memory
    for col in ["open", "high", "low", "close", "pre_close", "pct_chg", "vol", "amount", "vwap"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], downcast="float")
    return df


def load_metric_data(date_list, basic_df):
    """Load metric (fundamental) data — only essential columns."""
    valid_codes = set(basic_df["ts_code"].tolist())
    usecols = ["ts_code", "trade_date", "turnover_rate", "volume_ratio",
               "pe_ttm", "pb", "ps_ttm", "dv_ttm", "total_mv", "circ_mv"]

    all_frames = []
    for d in tqdm(date_list, desc="Loading metric data"):
        fpath = os.path.join(METRIC_DIR, f"{d}.csv")
        if not os.path.exists(fpath):
            continue
        try:
            day_df = pd.read_csv(fpath, dtype={"ts_code": str, "trade_date": str},
                                 usecols=[c for c in usecols if c in usecols])
            day_df = day_df[day_df["ts_code"].isin(valid_codes)]
            if len(day_df) > 0:
                all_frames.append(day_df)
        except Exception:
            continue

    if not all_frames:
        return pd.DataFrame()

    df = pd.concat(all_frames, ignore_index=True)
    df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
    for col in usecols:
        if col in df.columns and col not in ("ts_code", "trade_date"):
            df[col] = pd.to_numeric(df[col], downcast="float")
    return df


def load_moneyflow_data(date_list, basic_df):
    """Load moneyflow data — only essential columns (net flow, big orders)."""
    valid_codes = set(basic_df["ts_code"].tolist())
    usecols = ["ts_code", "trade_date", "buy_lg_amount", "sell_lg_amount",
               "buy_elg_amount", "sell_elg_amount", "net_mf_amount"]

    all_frames = []
    for d in tqdm(date_list, desc="Loading moneyflow"):
        fpath = os.path.join(MONEYFLOW_DIR, f"{d}.csv")
        if not os.path.exists(fpath):
            continue
        try:
            day_df = pd.read_csv(fpath, dtype={"ts_code": str, "trade_date": str},
                                 usecols=[c for c in usecols if c in usecols])
            day_df = day_df[day_df["ts_code"].isin(valid_codes)]
            if len(day_df) > 0:
                all_frames.append(day_df)
        except Exception:
            continue

    if not all_frames:
        return pd.DataFrame()

    df = pd.concat(all_frames, ignore_index=True)
    df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
    for col in usecols:
        if col in df.columns and col not in ("ts_code", "trade_date"):
            df[col] = pd.to_numeric(df[col], downcast="float")
    return df


def load_st_stocks(date_list, basic_df):
    """Load ST stock list for each trading day. Returns set of ST codes per date."""
    st_dict = {}  # date_str -> set of ts_codes
    for d in tqdm(date_list, desc="Loading ST data"):
        fpath = os.path.join(STOCK_ST_DIR, f"{d}.csv")
        if not os.path.exists(fpath):
            continue
        try:
            day_df = pd.read_csv(fpath, dtype={"ts_code": str, "trade_date": str})
            st_dict[d] = set(day_df["ts_code"].tolist())
        except Exception:
            continue
    return st_dict


def build_unified_dataframe(
    daily_df, metric_df=None, moneyflow_df=None, st_dict=None
):
    """Merge daily, metric, moneyflow data into a memory-efficient DataFrame."""
    df = daily_df.copy()

    # Optimize: categorical for stock codes
    df["ts_code"] = df["ts_code"].astype("category")

    # Merge metric data
    if metric_df is not None and len(metric_df) > 0:
        metric_df["ts_code"] = metric_df["ts_code"].astype("category")
        available = [c for c in metric_df.columns if c in df.columns or c != "ts_code"]
        keep_cols = ["ts_code", "trade_date"] + [c for c in metric_df.columns
                     if c not in ("ts_code", "trade_date")]
        df = df.merge(
            metric_df[keep_cols],
            on=["ts_code", "trade_date"], how="left", suffixes=("", "_m")
        )

    # Merge moneyflow data
    if moneyflow_df is not None and len(moneyflow_df) > 0:
        moneyflow_df["ts_code"] = moneyflow_df["ts_code"].astype("category")
        mf_cols = [c for c in moneyflow_df.columns if c not in ("ts_code", "trade_date")]
        df = df.merge(
            moneyflow_df[["ts_code", "trade_date"] + mf_cols],
            on=["ts_code", "trade_date"], how="left", suffixes=("", "_mf")
        )

    # Sort
    df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

    # Mark ST stocks (vectorized)
    if st_dict is not None:
        df["is_st"] = False
        for date_str, st_codes in st_dict.items():
            date_ts = pd.Timestamp(date_str)
            mask = (df["trade_date"] == date_ts) & (df["ts_code"].isin(st_codes))
            df.loc[mask, "is_st"] = True
    else:
        df["is_st"] = False

    return df


def filter_and_prepare(df, basic_df, date_range):
    """
    Apply stock pool filters and prepare final dataset.
    """
    # Filter by date range
    mask = (df["trade_date"] >= pd.Timestamp(date_range[0])) & \
           (df["trade_date"] <= pd.Timestamp(date_range[1]))
    df = df[mask].copy()

    # Filter ST
    if EXCLUDE_ST:
        df = df[~df["is_st"]]

    # Minimum listing days filter: ensures enough history
    # We do this per stock
    df = df.groupby("ts_code", group_keys=False).filter(
        lambda g: len(g) >= MIN_LIST_DAYS
    )

    return df


def load_all_data(train_start=TRAIN_START, test_end=TEST_END):
    """
    Main entry: load all data, build unified DataFrame.
    Caches result to parquet for faster reload.
    """
    cache_path = os.path.join(OUTPUT_DIR, "unified_data.parquet")
    if os.path.exists(cache_path):
        print(f"[INFO] Loading cached data from {cache_path}")
        return pd.read_parquet(cache_path)

    print("[INFO] Loading basic info...")
    basic_df = load_basic_info()
    print(f"  -> {len(basic_df)} stocks after market filter")

    print("[INFO] Loading trade calendar...")
    all_dates = load_trade_calendar()
    # all_dates are YYYYMMDD strings, train_start/test_end are YYYY-MM-DD
    ts = pd.Timestamp(train_start)
    te = pd.Timestamp(test_end)
    date_list = [d for d in all_dates
                 if ts <= pd.Timestamp(d) <= te]
    print(f"  -> {len(date_list)} trading days")

    print("[INFO] Loading daily OHLCV...")
    daily_df = load_daily_data(date_list, basic_df)

    print("[INFO] Loading metric data...")
    metric_df = load_metric_data(date_list, basic_df)

    print("[INFO] Loading moneyflow data...")
    moneyflow_df = load_moneyflow_data(date_list, basic_df)

    print("[INFO] Loading ST stock list...")
    st_dict = load_st_stocks(date_list, basic_df)

    print("[INFO] Building unified DataFrame...")
    df = build_unified_dataframe(daily_df, metric_df, moneyflow_df, st_dict)

    print(f"[INFO] Unified data: {len(df)} rows, shape={df.shape}")
    print(f"[INFO] Saving cache to {cache_path}")
    df.to_parquet(cache_path, index=False)

    return df


if __name__ == "__main__":
    df = load_all_data()
    print(df.head())
    print(df.columns.tolist())
