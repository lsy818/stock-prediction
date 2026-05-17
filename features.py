"""
Feature engineering for stock prediction.
Computes technical indicators and cross-sectional features.
Uses pandas built-in rolling operations for speed.
"""
import numpy as np
import pandas as pd
from tqdm import tqdm


def compute_stock_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all temporal features for a single stock's time series.
    Uses pandas rolling/ewm for vectorized, fast computation.

    Input: DataFrame with [trade_date, open, high, low, close, vol, amount, vwap, pre_close, ...]
    Output: DataFrame with additional feature columns.
    """
    df = df.sort_values("trade_date").reset_index(drop=True)
    n = len(df)

    c = df["close"]
    h = df["high"]
    l = df["low"]
    o = df["open"]
    v = df["vol"]
    vwap = df["vwap"]
    pc = df["pre_close"]

    # --- Returns ---
    df["return_1d"] = c / c.shift(1) - 1.0
    mask0 = df["return_1d"].isna()
    df.loc[mask0, "return_1d"] = c[mask0].values / pc[mask0].values - 1.0
    df["log_return"] = np.log(df["return_1d"] + 1.0)

    # --- Intraday features ---
    df["intraday_range"] = (h - l) / (c + 1e-10)
    df["close_position"] = (c - l) / (h - l + 1e-10)
    df["vwap_premium"] = c / (vwap + 1e-10) - 1.0

    # --- MA ratios ---
    for w in [5, 10, 20, 60]:
        ma = c.rolling(w, min_periods=1).mean()
        df[f"close_ma{w}_ratio"] = c / (ma + 1e-10) - 1.0

    # --- Volume ---
    df["volume_ratio"] = v / (v.rolling(20, min_periods=1).mean() + 1e-10)

    # --- RSI ---
    def _rsi(series, window):
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1/window, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/window, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        return 100.0 - 100.0 / (1.0 + rs)

    if n >= 6:
        df["rsi_6"] = _rsi(c, 6)
    else:
        df["rsi_6"] = 50.0

    if n >= 14:
        df["rsi_14"] = _rsi(c, 14)
    else:
        df["rsi_14"] = 50.0

    # --- MACD ---
    if n >= 26:
        ema12 = c.ewm(span=12, adjust=False).mean()
        ema26 = c.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        macd_signal = macd_line.ewm(span=9, adjust=False).mean()
        df["macd"] = macd_line
        df["macd_signal"] = macd_signal
        df["macd_hist"] = macd_line - macd_signal
    else:
        df["macd"] = 0.0
        df["macd_signal"] = 0.0
        df["macd_hist"] = 0.0

    # --- KDJ ---
    if n >= 9:
        low_n = l.rolling(9, min_periods=1).min()
        high_n = h.rolling(9, min_periods=1).max()
        rsv = (c - low_n) / (high_n - low_n + 1e-10) * 100.0
        # Smoothed K, D, J
        k = rsv.ewm(alpha=1/3, adjust=False).mean()
        d = k.ewm(alpha=1/3, adjust=False).mean()
        j = 3 * k - 2 * d
        df["kdj_k"] = k.fillna(50)
        df["kdj_d"] = d.fillna(50)
        df["kdj_j"] = j.fillna(50)
    else:
        df["kdj_k"] = 50.0
        df["kdj_d"] = 50.0
        df["kdj_j"] = 50.0

    # --- Bollinger Bands ---
    if n >= 20:
        ma20 = c.rolling(20, min_periods=1).mean()
        std20 = c.rolling(20, min_periods=1).std()
        upper = ma20 + 2 * std20
        lower = ma20 - 2 * std20
        df["boll_upper"] = upper
        df["boll_lower"] = lower
        df["boll_bandwidth"] = (upper - lower) / (ma20 + 1e-10)
        df["boll_pct_b"] = (c - lower) / (upper - lower + 1e-10)
    else:
        df["boll_upper"] = c
        df["boll_lower"] = c
        df["boll_bandwidth"] = 0.0
        df["boll_pct_b"] = 0.5

    # --- ATR ---
    if n >= 14:
        prev_c = c.shift(1).fillna(o)
        tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
        df["atr_14"] = tr.ewm(alpha=1/14, adjust=False).mean()
        df["atr_pct"] = df["atr_14"] / (c + 1e-10)
    else:
        df["atr_14"] = 0.0
        df["atr_pct"] = 0.0

    # --- Williams %R ---
    if n >= 14:
        h14 = h.rolling(14, min_periods=1).max()
        l14 = l.rolling(14, min_periods=1).min()
        df["wr_14"] = (h14 - c) / (h14 - l14 + 1e-10) * -100.0
    else:
        df["wr_14"] = -50.0

    # --- CCI ---
    if n >= 20:
        tp = (h + l + c) / 3.0
        ma_tp = tp.rolling(20, min_periods=1).mean()
        md_tp = tp.rolling(20, min_periods=1).apply(lambda x: np.abs(x - x.mean()).mean())
        df["cci_20"] = (tp - ma_tp) / (0.015 * md_tp + 1e-10)
    else:
        df["cci_20"] = 0.0

    # --- OBV ---
    df["obv"] = (np.sign(c.diff().fillna(0)) * v).cumsum()

    # --- Moneyflow Features ---
    if "net_mf_amount" in df.columns:
        amt = df["amount"]
        df["net_mf_ratio"] = df["net_mf_amount"] / (amt + 1e-10)
        big_buy = df.get("buy_lg_amount", 0).fillna(0) + df.get("buy_elg_amount", 0).fillna(0)
        big_sell = df.get("sell_lg_amount", 0).fillna(0) + df.get("sell_elg_amount", 0).fillna(0)
        df["big_order_ratio"] = (big_buy - big_sell) / (amt + 1e-10)
        df["mf_intensity"] = df["net_mf_amount"].abs() / (amt + 1e-10)
    else:
        df["net_mf_ratio"] = 0.0
        df["big_order_ratio"] = 0.0
        df["mf_intensity"] = 0.0

    # --- Fundamental placeholders ---
    for col in ["pe_ttm", "pb", "ps_ttm", "dv_ttm", "total_mv", "circ_mv",
                "turnover_rate", "turnover_rate_f"]:
        if col not in df.columns:
            df[col] = np.nan

    return df


def add_cross_sectional_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cross-sectional (per-day) rank and z-score features."""
    rank_cols = ["return_1d", "volume_ratio"]
    if "turnover_rate" in df.columns:
        rank_cols.append("turnover_rate")
    if "total_mv" in df.columns:
        rank_cols.append("total_mv")

    for col in rank_cols:
        if col in df.columns:
            df[f"{col}_rank"] = df.groupby("trade_date")[col].rank(pct=True)

    zscore_cols = ["return_1d", "log_return", "rsi_14", "macd_hist"]
    for col in zscore_cols:
        if col in df.columns:
            grp = df.groupby("trade_date")[col]
            df[f"{col}_z"] = grp.transform(lambda x: (x - x.mean()) / (x.std() + 1e-10))

    for col in ["pe_ttm", "pb", "ps_ttm"]:
        if col in df.columns:
            daily_median = df.groupby("trade_date")[col].transform("median")
            df[col] = df[col].fillna(daily_median)
            df[f"{col}_rank"] = df.groupby("trade_date")[col].rank(pct=True)

    return df


def compute_all_features(df: pd.DataFrame, basic_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Compute all features. Processes each stock independently (no leakage),
    then adds cross-sectional features per day.
    """
    print("[INFO] Computing per-stock temporal features...")
    result_dfs = []
    grouped = df.groupby("ts_code")

    for code, group in tqdm(grouped, total=df["ts_code"].nunique()):
        try:
            result_dfs.append(compute_stock_features(group))
        except Exception:
            continue

    df_feat = pd.concat(result_dfs, ignore_index=True)

    print("[INFO] Computing cross-sectional features...")
    df_feat = add_cross_sectional_features(df_feat)
    # Only fill numeric columns (categorical ts_code can't take 0.0)
    numeric_cols = df_feat.select_dtypes(include=[np.number]).columns
    df_feat[numeric_cols] = df_feat[numeric_cols].fillna(0.0)

    print(f"[INFO] Final feature shape: {df_feat.shape}")
    return df_feat


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return feature column names (exclude metadata and raw prices)."""
    exclude = {
        "ts_code", "trade_date", "symbol", "name", "area", "industry",
        "cnspell", "market", "list_date", "act_name", "act_ent_type",
        "is_st", "change", "pct_chg", "pre_close",
    }
    raw_price = {"open", "high", "low", "close", "vol", "amount", "vwap"}
    exclude.update(raw_price)
    # Keep ratio versions but exclude some derived intermediates
    exclude.update({"boll_upper", "boll_lower", "_target_raw", "_target_cs",
                     "_close"})  # _close is internal

    feat_cols = [c for c in df.columns if c not in exclude]
    return feat_cols
