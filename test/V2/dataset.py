import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class StockSequenceDataset(Dataset):
    def __init__(self, df, feature_cols, target_col, seq_len=30, target_date_start=None, target_date_end=None):
        """
        df: pre-sorted dataframe by 'ts_code' and 'trade_date'
        """
        self.seq_len = seq_len
        
        # Convert df to numpy first, then to tensor (avoids copying inside __getitem__)
        self.features = torch.tensor(df[feature_cols].values, dtype=torch.float32)
        self.targets = torch.tensor(df[target_col].values, dtype=torch.float32)
        
        self.codes = df['ts_code'].values
        self.dates = df['trade_date'].dt.strftime('%Y-%m-%d').values
        
        # Ensure target dates are strings for comparison with group_dates
        if target_date_start is not None:
            target_date_start = pd.to_datetime(target_date_start).strftime('%Y-%m-%d')
        if target_date_end is not None:
            target_date_end = pd.to_datetime(target_date_end).strftime('%Y-%m-%d')
            
        self.valid_indices = []
        
        # Find group boundaries efficiently
        stock_changes = (df['ts_code'] != df['ts_code'].shift(1)).values
        
        current_stock_start = 0
        for i in range(len(df)):
            if stock_changes[i]:
                current_stock_start = i
                
            if i - current_stock_start + 1 >= seq_len:
                target_date = self.dates[i]
                if target_date_start is not None and target_date < target_date_start:
                    continue
                if target_date_end is not None and target_date > target_date_end:
                    continue
                self.valid_indices.append(i)
                
        self.valid_indices = np.array(self.valid_indices, dtype=np.int32)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        end_idx = self.valid_indices[idx]
        start_idx = end_idx - self.seq_len + 1
        
        X = self.features[start_idx : end_idx + 1]
        y = self.targets[end_idx]
        date = self.dates[end_idx]
        code = self.codes[end_idx]
        
        return X, y, date, code

def get_dataloaders(parquet_path, seq_len=30, batch_size=512):
    print("Loading parquet data...")
    df = pd.read_parquet(parquet_path)
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    
    # Define features and target
    target_col = 'label_return_1d'
    # Drop non-feature columns
    feature_cols = [c for c in df.columns if c not in ['ts_code', 'trade_date', target_col]]
    print(f"Using {len(feature_cols)} features: {feature_cols}")

    # Train/Val/Test Split (Time based filtering on targets)
    train_start, train_end = pd.to_datetime('2016-01-04'), pd.to_datetime('2024-12-31')
    val_start, val_end = pd.to_datetime('2025-01-01'), pd.to_datetime('2025-12-31')
    test_start, test_end = pd.to_datetime('2026-01-01'), pd.to_datetime('2026-12-31')

    # Standardization (Fit on train ONLY to prevent data leakage)
    train_mask = (df['trade_date'] >= train_start) & (df['trade_date'] <= train_end)
    mean = df[train_mask][feature_cols].mean()
    std = df[train_mask][feature_cols].std() + 1e-8 # prevent division by zero

    # Apply standardization globally
    df.loc[:, feature_cols] = (df[feature_cols] - mean) / std

    print("Building datasets (sliding windows)... this may take a moment.")
    train_dataset = StockSequenceDataset(df, feature_cols, target_col, seq_len=seq_len, 
                                         target_date_start=train_start, target_date_end=train_end)
    val_dataset = StockSequenceDataset(df, feature_cols, target_col, seq_len=seq_len,
                                       target_date_start=val_start, target_date_end=val_end)
    test_dataset = StockSequenceDataset(df, feature_cols, target_col, seq_len=seq_len,
                                        target_date_start=test_start, target_date_end=test_end)

    print(f"Train targets: {len(train_dataset)}, Val targets: {len(val_dataset)}, Test targets: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    return train_loader, val_loader, test_loader, len(feature_cols)

if __name__ == "__main__":
    # Test
    train_loader, val_loader, num_features = get_dataloaders('../data/processed/csi300_features.parquet', seq_len=30)
    for X, y, dates, codes in train_loader:
        print(f"X batch shape: {X.shape}") # expected: [batch_size, seq_len, num_features]
        print(f"y batch shape: {y.shape}") # expected: [batch_size]
        break
