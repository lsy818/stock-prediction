import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm

DATA_DIR = '../data'
OUTPUT_DIR = '../data/processed'

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
    return list(symbols)

def merge_daily_data(symbols):
    print("Merging daily data...")
    daily_files = sorted(glob.glob(os.path.join(DATA_DIR, 'daily', '*.csv')))
    
    dfs = []
    # To speed up MVP, we could just process the last 5 years or all. 
    # Processing all 2500 files takes a bit of time but should be manageable.
    for f in tqdm(daily_files, desc="Reading daily files"):
        try:
            df = pd.read_csv(f)
            # Filter stocks
            df = df[df['ts_code'].isin(symbols)]
            dfs.append(df)
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
    # For MVP, fill missing with forward fill within group
    cols_to_fill = raw_df.columns.drop(['ts_code', 'trade_date'])
    raw_df[cols_to_fill] = raw_df.groupby('ts_code')[cols_to_fill].ffill().bfill()
    
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
