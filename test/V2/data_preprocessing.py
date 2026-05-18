import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm

DATA_DIR = '../../data'
OUTPUT_DIR = '../../data/processed'

def get_csi300_symbols():
    print("Extracting CSI 300 universe...")
    weight_files = glob.glob(os.path.join(DATA_DIR, 'index_weight', '*_000300.SH.csv'))
    symbols = set()
    for f in weight_files:
        try:
            df = pd.read_csv(f)
            if 'con_code' in df.columns:
                symbols.update(df['con_code'].dropna().unique())
        except Exception as e:
            print(f"Error reading {f}: {e}")
    print(f"Total unique CSI 300 symbols found: {len(symbols)}")
    
    # NEW LOGIC: Find ST stocks and remove them entirely
    print("Filtering out ST stocks...")
    st_files = glob.glob(os.path.join(DATA_DIR, 'stock_st', '*.csv'))
    st_symbols = set()
    for f in tqdm(st_files, desc="Parsing ST files"):
        try:
            df = pd.read_csv(f)
            if 'ts_code' in df.columns:
                st_symbols.update(df['ts_code'].dropna().unique())
        except Exception as e:
            pass
            
    print(f"Total unique ST symbols found over history: {len(st_symbols)}")
    symbols = symbols - st_symbols
    print(f"Filtered CSI 300 universe size (no ST ever): {len(symbols)}")
    
    return list(symbols)

def merge_daily_data(symbols):
    print("Merging daily, metric, and moneyflow data...")
    daily_files = sorted(glob.glob(os.path.join(DATA_DIR, 'daily', '*.csv')))
    
    dfs = []
    for f in tqdm(daily_files, desc="Reading and joining files"):
        try:
            date_str = os.path.basename(f)
            
            # Read daily
            df_daily = pd.read_csv(f)
            df_daily = df_daily[df_daily['ts_code'].isin(symbols)]
            if df_daily.empty:
                continue
                
            # Read metric
            metric_f = os.path.join(DATA_DIR, 'metric', date_str)
            if os.path.exists(metric_f):
                df_metric = pd.read_csv(metric_f)
                if 'trade_date' in df_metric.columns:
                    df_metric = df_metric.drop(columns=['trade_date'])
                if 'close' in df_metric.columns:
                    df_metric = df_metric.drop(columns=['close'])
                df_daily = df_daily.merge(df_metric, on='ts_code', how='left')
                
            # Read moneyflow
            mf_f = os.path.join(DATA_DIR, 'moneyflow', date_str)
            if os.path.exists(mf_f):
                df_mf = pd.read_csv(mf_f)
                if 'trade_date' in df_mf.columns:
                    df_mf = df_mf.drop(columns=['trade_date'])
                df_daily = df_daily.merge(df_mf, on='ts_code', how='left')
                
            dfs.append(df_daily)
        except Exception as e:
            pass
            
    all_data = pd.concat(dfs, ignore_index=True)
    
    # Sort by code and date
    all_data['trade_date'] = pd.to_datetime(all_data['trade_date'].astype(str))
    all_data = all_data.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)
    return all_data

def calculate_technical_indicators(df):
    print("Calculating technical indicators...")
    # Group by stock
    grouped = df.groupby('ts_code')
    
    # Moving Averages
    df['ma5'] = grouped['close'].transform(lambda x: x.rolling(5).mean())
    df['ma10'] = grouped['close'].transform(lambda x: x.rolling(10).mean())
    df['ma20'] = grouped['close'].transform(lambda x: x.rolling(20).mean())
    
    # Price Momentum
    df['mom5'] = grouped['close'].transform(lambda x: x.pct_change(5))
    df['mom10'] = grouped['close'].transform(lambda x: x.pct_change(10))
    
    # Volatility (Historical 20d volatility)
    df['volatility20'] = grouped['pct_chg'].transform(lambda x: x.rolling(20).std())
    
    # VWAP (if not perfectly provided, or just use the provided one)
    # We can also compute Volume moving average
    df['vol_ma5'] = grouped['vol'].transform(lambda x: x.rolling(5).mean())
    
    # MACD simplified
    ema12 = grouped['close'].transform(lambda x: x.ewm(span=12, adjust=False).mean())
    ema26 = grouped['close'].transform(lambda x: x.ewm(span=26, adjust=False).mean())
    df['macd'] = ema12 - ema26
    
    # RSI simplified (14 days)
    delta = grouped['close'].transform(lambda x: x.diff())
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.groupby(df['ts_code']).transform(lambda x: x.ewm(com=13, adjust=False).mean())
    ema_down = down.groupby(df['ts_code']).transform(lambda x: x.ewm(com=13, adjust=False).mean())
    rs = ema_up / ema_down
    df['rsi14'] = 100 - (100 / (1 + rs))

    # --- NEW ADVANCED FEATURES (V2 Part 2) ---
    # 1. Moneyflow Momentum (5-day rolling mean of net money flow amount)
    if 'net_mf_amount' in df.columns:
        df['net_mf_amount_ma5'] = grouped['net_mf_amount'].transform(lambda x: x.rolling(5).mean())
        # Ratio of net money flow to total amount (to normalize)
        df['mf_to_amount_ratio'] = df['net_mf_amount'] / (df['amount'] + 1e-8)
        
    # 2. Valuation Deviation (PE deviation from 60-day MA)
    if 'pe_ttm' in df.columns:
        df['pe_ttm_ma60'] = grouped['pe_ttm'].transform(lambda x: x.rolling(60).mean())
        df['pe_ttm_deviation'] = (df['pe_ttm'] - df['pe_ttm_ma60']) / (df['pe_ttm_ma60'] + 1e-8)
    elif 'pe' in df.columns:
        df['pe_ma60'] = grouped['pe'].transform(lambda x: x.rolling(60).mean())
        df['pe_deviation'] = (df['pe'] - df['pe_ma60']) / (df['pe_ma60'] + 1e-8)

    return df

def generate_labels(df):
    print("Generating labels...")
    # T+1 Return (Next day's pct_chg)
    # Since pct_chg is typically today's return, the label is the pct_chg of the NEXT day.
    df['label_return_1d'] = df.groupby('ts_code')['pct_chg'].shift(-1)
    
    # Drop rows with NaN in labels (the last day for each stock will have no label)
    df = df.dropna(subset=['label_return_1d'])
    
    return df

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Get Universe
    symbols = get_csi300_symbols()
    
    # 2. Merge Data
    raw_df = merge_daily_data(symbols)
    print(f"Raw data shape: {raw_df.shape}")
    
    # 3. Clean Missing Values 
    # For MVP, fill missing with forward fill within group, then fillna(0) to avoid lookahead bias
    cols_to_fill = raw_df.columns.drop(['ts_code', 'trade_date'])
    raw_df[cols_to_fill] = raw_df.groupby('ts_code')[cols_to_fill].ffill()
    raw_df[cols_to_fill] = raw_df[cols_to_fill].fillna(0)
    
    # 4. Feature Engineering
    features_df = calculate_technical_indicators(raw_df)
    
    # 5. Labels
    final_df = generate_labels(features_df)
    
    # 6. Drop NaNs resulted from rolling windows
    final_df = final_df.dropna().reset_index(drop=True)
    
    print(f"Final data shape after dropping NaNs: {final_df.shape}")
    
    # Save to parquet
    out_path = os.path.join(OUTPUT_DIR, 'csi300_features.parquet')
    final_df.to_parquet(out_path, index=False)
    print(f"Data saved to {out_path}")

if __name__ == '__main__':
    main()
